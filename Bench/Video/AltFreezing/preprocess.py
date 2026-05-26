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
import shutil
# --------------------- 参数 ---------------------
mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1, 1)
std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1, 1)

max_frame = 400
dataset_dir = "/data/lp/IVBridgeDataset_Videos_Split/GenVidBench"  # 数据集根目录
cfg_path = "i3d_ori.yaml"
ckpt_path = "checkpoints/model.pth"
clip_size = 32  # 根据 cfg.clip_size 可改
optimal_threshold = 0.04  # 用于二分类阈值

# -------------------------------------------------

def process_video(video_path, classifier, crop_align_func):
    """处理单个视频，返回每帧预测概率"""
    cache_file = f"{video_path}_{max_frame}.pth"

    # ------------------- 检测 -------------------
    if not os.path.exists(cache_file):
        print(cache_file)
        detect_res, all_lm68, frames = detect_all(video_path, return_frames=True, max_size=max_frame)
        torch.save((detect_res, all_lm68), cache_file)

# import os
# import shutil

# def process_video(video_path, classifier, crop_align_func):
#     """处理单个视频，返回每帧预测概率"""
#     cache_file = f"{video_path}_{max_frame}.pth"

#     # ------------------- 检测 -------------------
#     if not os.path.exists(cache_file):
#         print(f"⚠ Cache not found for {video_path}")

#         # 目标子目录（当前工作目录下）
#         target_dir = os.path.join(os.getcwd(), "no_cache_videos")
#         os.makedirs(target_dir, exist_ok=True)

#         # 目标路径
#         target_path = os.path.join(target_dir, os.path.basename(video_path))

#         # 如果文件还没被移动过
#         if not os.path.exists(target_path):
#             shutil.move(video_path, target_path)
#             print(f"📦 Moved video to {target_path}")
#         else:
#             print(f"ℹ Video already moved: {target_path}")


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

    for label_name, label_value in [("Vript", 0)]:
        video_dir = os.path.join(dataset_dir, label_name)

        for video_path in Path(video_dir).rglob("*.mp4"):

            process_video(video_path, classifier, crop_align_func)