import os
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
from config import config as cfg
from test_tools.common import detect_all, grab_all_frames
from test_tools.ct.operations import find_longest, multiple_tracking
from test_tools.faster_crop_align_xray import FasterCropAlignXRay
from test_tools.utils import get_crop_box
from utils.plugin_loader import PluginLoader
from pathlib import Path

# --------------------- 参数 ---------------------
mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1, 1)
std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1, 1)

max_frame = 400
dataset_dir = "/data2/lp/AltFreezing/no_cache_videos"  # 数据集根目录
cfg_path = "i3d_ori.yaml"
ckpt_path = "checkpoints/model.pth"
clip_size = 32  # 根据 cfg.clip_size 可改
optimal_threshold = 0.04  # 用于二分类阈值

# -------------------------------------------------

def process_video(video_path, classifier, crop_align_func):
    """处理单个视频，返回每帧预测概率"""
    cache_file = f"{video_path}_{max_frame}.pth"

    # ------------------- 检测 -------------------
    if os.path.exists(cache_file):
        detect_res, all_lm68 = torch.load(cache_file)
        frames = grab_all_frames(video_path, max_size=max_frame, cvt=True)
    else:
        detect_res, all_lm68, frames = detect_all(video_path, return_frames=True, max_size=max_frame)
        torch.save((detect_res, all_lm68), cache_file)

    if len(frames) == 0:
        return []  # 视频没有帧

    shape = frames[0].shape[:2]

    # ----------------- 处理检测结果 -----------------
    all_detect_res = []
    for faces, faces_lm68 in zip(detect_res, all_lm68):
        new_faces = []
        for (box, lm5, score), face_lm68 in zip(faces, faces_lm68):
            new_face = (box, lm5, face_lm68, score)
            new_faces.append(new_face)
        all_detect_res.append(new_faces)
    detect_res = all_detect_res

    tracks = multiple_tracking(detect_res)
    tuples = [(0, len(detect_res))] * len(tracks)
    if len(tracks) == 0:
        tuples, tracks = find_longest(detect_res)

    data_storage = {}
    frame_boxes = {}
    super_clips = []

    for track_i, ((start, end), track) in enumerate(zip(tuples, tracks)):
        super_clips.append(len(track))
        for face, frame_idx, j in zip(track, range(start, end), range(len(track))):
            box, lm5, lm68 = face[:3]
            big_box = get_crop_box(shape, box, scale=0.5)

            top_left = big_box[:2][None, :]
            new_lm5 = lm5 - top_left
            new_lm68 = lm68 - top_left
            new_box = (box.reshape(2, 2) - top_left).reshape(-1)
            info = (new_box, new_lm5, new_lm68, big_box)

            x1, y1, x2, y2 = big_box
            cropped = frames[frame_idx][y1:y2, x1:x2]
            base_key = f"{track_i}_{j}_"
            data_storage[f"{base_key}img"] = cropped
            data_storage[f"{base_key}ldm"] = info
            data_storage[f"{base_key}idx"] = frame_idx
            frame_boxes[frame_idx] = np.rint(box).astype(int)

    # ----------------- 生成 clips -----------------
    clips_for_video = []
    pad_length = clip_size - 1
    for super_clip_idx, super_clip_size in enumerate(super_clips):
        inner_index = list(range(super_clip_size))
        if super_clip_size < clip_size:
            post_module = inner_index[1:-1][::-1] + inner_index
            l_post = len(post_module)
            post_module = post_module * (pad_length // l_post + 1)
            post_module = post_module[:pad_length]

            pre_module = inner_index + inner_index[1:-1][::-1]
            l_pre = len(pre_module)
            pre_module = pre_module * (pad_length // l_pre + 1)
            pre_module = pre_module[-pad_length:]

            inner_index = pre_module + inner_index + post_module

        frame_range = [
            inner_index[i:i+clip_size] for i in range(len(inner_index)) if i + clip_size <= len(inner_index)
        ]
        for indices in frame_range:
            clip = [(super_clip_idx, t) for t in indices]
            clips_for_video.append(clip)

    # ----------------- 推理 -----------------
    frame_res = {}
    for clip in tqdm(clips_for_video, desc=f"Processing {os.path.basename(video_path)}", leave=False, disable=True):
        images = [data_storage[f"{i}_{j}_img"] for i, j in clip]
        landmarks = [data_storage[f"{i}_{j}_ldm"] for i, j in clip]
        frame_ids = [data_storage[f"{i}_{j}_idx"] for i, j in clip]

        _, images_align = crop_align_func(landmarks, images)
        images_tensor = torch.as_tensor(images_align, dtype=torch.float32).cuda().permute(3,0,1,2).unsqueeze(0)
        # import pdb; pdb.set_trace()
        images_tensor = images_tensor.sub(mean).div(std)
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            output = classifier(images_tensor)
        pred = float(torch.sigmoid(output["final_output"]))


        for f_id in frame_ids:
            frame_res.setdefault(f_id, []).append(pred)

    # 平均每帧预测
    scores = []
    for frame_idx in range(len(frames)):
        if frame_idx in frame_res:
            scores.append(np.mean(frame_res[frame_idx]))
        else:
            scores.append(None)
    return scores


# ----------------- 主流程 -----------------
if __name__ == "__main__":
    # 加载配置和模型
    cfg.init_with_yaml()
    cfg.update_with_yaml(cfg_path)
    cfg.freeze()

    classifier = PluginLoader.get_classifier(cfg.classifier_type)()
    classifier.cuda()
    classifier.eval()
    classifier.load(ckpt_path)
    crop_align_func = FasterCropAlignXRay(cfg.imsize)

    # 遍历数据集目录
    all_labels = []
    all_scores = []
    video_ids = []

    for label_name, label_value in [("part_7", 0)]:
        video_dir = os.path.join(dataset_dir, label_name)
        # video_paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))

        # for video_path in video_paths:

        # for video_path in Path(video_dir).rglob("*.mp4"):
        #     scores = process_video(video_path, classifier, crop_align_func)
        #     # 取整个视频的平均概率作为该视频预测
        #     scores_clean = [s if s is not None else np.nan for s in scores]
        #     video_score = np.nanmean(scores_clean) if len(scores_clean) > 0 else 0.0
        #     all_scores.append(video_score)
        #     all_labels.append(label_value)
        #     video_ids.append(video_path)
        #     print(f"{video_path}: label={label_value}, score={video_score:.4f}")

        max_videos = 10  # 只处理前 100 个视频
        video_count = 0

        for video_path in Path(video_dir).rglob("*.mp4"):
            # if video_count >= max_videos:
            #     break  # 达到上限就停止

            scores = process_video(video_path, classifier, crop_align_func)

            # --------- 清理 None 值 ---------
            scores_clean = [s for s in scores if s is not None]

            if len(scores_clean) == 0:
                video_score = np.nan  # 或者 0.0，根据你需求
                print(f"⚠ Warning: video {video_path} has no valid scores")
            else:
                video_score = np.nanmean(scores_clean)

            all_scores.append(video_score)
            all_labels.append(label_value)
            video_ids.append(video_path)
            print(f"{video_path}: label={label_value}, score={video_score:.4f}, count={video_count}")

            video_count += 1

    # ----------------- 计算指标 -----------------
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    # auc = roc_auc_score(all_labels, all_scores)
    # ap = average_precision_score(all_labels, all_scores)
    # print(f"\nDataset AUC: {auc:.4f}")
    # print(f"Dataset AP: {ap:.4f}")
    
    # ----------------- 保存CSV -----------------
    results_df = pd.DataFrame({
        "video_id": video_ids,
        "label": all_labels,
        "score": all_scores
    })

    csv_path = "vript_part_7_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
