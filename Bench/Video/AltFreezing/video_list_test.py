import os
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.plugin_loader import PluginLoader
from config import config as cfg
from test_tools.common import detect_all, grab_all_frames
from test_tools.ct.operations import find_longest, multiple_tracking
from test_tools.utils import get_crop_box
from test_tools.faster_crop_align_xray import FasterCropAlignXRay
from test_tools.supply_writer import SupplyWriter

# ------------------- 配置 -------------------
mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1, 1)
std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1, 1)

video_dir = "/data2/lp/AltFreezing/examples"  # ← 改成你的视频目录
label = 0
out_dir = "./"
os.makedirs(out_dir, exist_ok=True)
max_frame = 400
clip_threshold = 0.002584857167676091  # 供 SupplyWriter 使用

# ------------------- 初始化模型 -------------------
cfg.init_with_yaml()
cfg.update_with_yaml("ftcn_tt.yaml")
cfg.freeze()

classifier = PluginLoader.get_classifier(cfg.classifier_type)()
classifier.cuda()
classifier.eval()
classifier.load("checkpoints/ftcn_tt.pth")

crop_align_func = FasterCropAlignXRay(cfg.imsize)

# ------------------- 存储结果 -------------------
video_ids = []
all_labels = []
all_scores = []

# ------------------- 遍历视频 -------------------
video_paths = list(Path(video_dir).rglob("*.mp4"))
for video_path in tqdm(video_paths, desc="Processing videos"):

    video_path = str(video_path)
    cache_file = f"{video_path}_{max_frame}.pth"

    # 1️⃣ 人脸检测
    if os.path.exists(cache_file):
        detect_res, all_lm68 = torch.load(cache_file)
        frames = grab_all_frames(video_path, max_size=max_frame, cvt=True)
        # import pdb; pdb.set_trace()
    else:
        detect_res, all_lm68, frames = detect_all(video_path, return_frames=True, max_size=max_frame)
        torch.save((detect_res, all_lm68), cache_file)

    if len(frames) == 0:
        print(f"  ⚠ {video_path} has no frames, skipped.")
        continue

    shape = frames[0].shape[:2]

    # 2️⃣ 重组检测结果
    all_detect_res = []
    for faces, faces_lm68 in zip(detect_res, all_lm68):
        new_faces = []
        for (box, lm5, score), face_lm68 in zip(faces, faces_lm68):
            new_faces.append((box, lm5, face_lm68, score))
        all_detect_res.append(new_faces)
    detect_res = all_detect_res

    # 3️⃣ 超级片段追踪
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
            data_storage[base_key + "img"] = cropped
            data_storage[base_key + "ldm"] = info
            data_storage[base_key + "idx"] = frame_idx
            frame_boxes[frame_idx] = np.rint(box).astype(int)

    # 4️⃣ 生成 clip
    clips_for_video = []
    clip_size = cfg.clip_size
    pad_length = clip_size - 1
    for super_clip_idx, super_clip_size in enumerate(super_clips):
        inner_index = list(range(super_clip_size))
        if super_clip_size < clip_size:  # padding
            post_module = inner_index[1:-1][::-1] + inner_index
            post_module = post_module * (pad_length // len(post_module) + 1)
            post_module = post_module[:pad_length]
            pre_module = inner_index + inner_index[1:-1][::-1]
            pre_module = pre_module * (pad_length // len(pre_module) + 1)
            pre_module = pre_module[-pad_length:]
            inner_index = pre_module + inner_index + post_module
        frame_range = [
            inner_index[i: i + clip_size] for i in range(len(inner_index)) if i + clip_size <= len(inner_index)
        ]
        for indices in frame_range:
            clip = [(super_clip_idx, t) for t in indices]
            clips_for_video.append(clip)

    # 5️⃣ 分类预测
    frame_res = {}
    for clip in clips_for_video:
        images = [data_storage[f"{i}_{j}_img"] for i, j in clip]
        landmarks = [data_storage[f"{i}_{j}_ldm"] for i, j in clip]
        frame_ids = [data_storage[f"{i}_{j}_idx"] for i, j in clip]
        _, images_align = crop_align_func(landmarks, images)
        images = torch.as_tensor(images_align, dtype=torch.float32).cuda().permute(3, 0, 1, 2)
        images = images.unsqueeze(0).sub(mean).div(std)
        with torch.no_grad():
            output = classifier(images)
        pred = float(output["final_output"])
        for f_id in frame_ids:
            if f_id not in frame_res:
                frame_res[f_id] = []
            frame_res[f_id].append(pred)

    # 6️⃣ 视频平均概率
    scores_per_frame = [np.mean(frame_res[f]) if f in frame_res else np.nan for f in range(len(frames))]
    video_score = np.nanmean(scores_per_frame)
    print(video_score)
    video_ids.append(os.path.basename(video_path))
    all_labels.append(label)  # 默认 label，可修改
    all_scores.append(video_score)

    # # 7️⃣ 可选：保存带标注视频
    # basename = os.path.splitext(os.path.basename(video_path))[0] + ".avi"
    # out_file = os.path.join(out_dir, basename)
    # SupplyWriter(video_path, out_file, clip_threshold).run(frames, scores_per_frame, [frame_boxes.get(f) for f in range(len(frames))])

# ------------------- 保存 CSV -------------------
# results_df = pd.DataFrame({
#     "video_id": video_ids,
#     "label": all_labels,
#     "score": all_scores
# })
# results_csv = os.path.join(out_dir, "part_7_results.csv")
# results_df.to_csv(results_csv, index=False)
# print(f"\nSaved results to {results_csv}")
