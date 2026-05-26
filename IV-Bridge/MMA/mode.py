import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

np.random.seed(42)

root_dir = "/data/lp/KDD26-benchmark/VFTData"
output_file = "all_detectors_video_level_metrics.json"

results = {}

# ========= 遍历每个检测器子目录 =========
for detector in sorted(os.listdir(root_dir)):
    detector_path = os.path.join(root_dir, detector)

    if not os.path.isdir(detector_path):
        continue

    print(f"\nProcessing detector: {detector}")
    results[detector] = {}

    # ===== 遍历该检测器下的CSV =====
    for file in sorted(os.listdir(detector_path)):
        if not file.endswith(".csv"):
            continue

        print(f"  -> Dataset file: {file}")
        path = os.path.join(detector_path, file)
        df = pd.read_csv(path)

        video_groups = df.groupby("video_id")

        video_labels = []
        stats_preds = {
            "mean": [],
            "var": [],
            "max": [],
            "min": [],
            "median": [],
            "random": []
        }

        # ===== 视频级聚合 =====
        for vid, group in video_groups:
            scores = group["y_prob"].values.astype(float)
            label = int(group["label"].iloc[0])

            video_labels.append(label)

            stats_preds["mean"].append(np.mean(scores))
            stats_preds["var"].append(np.var(scores))
            stats_preds["max"].append(np.max(scores))
            stats_preds["min"].append(np.min(scores))
            stats_preds["median"].append(np.median(scores))
            stats_preds["random"].append(np.random.choice(scores))

        video_labels = np.array(video_labels)

        # ===== 计算指标 =====
        file_result = {}

        for stat_name, preds in stats_preds.items():
            preds = np.array(preds)

            # 防止极端情况全是同一类导致AUC报错
            if len(np.unique(video_labels)) < 2:
                auc, ap = None, None
            else:
                auc = roc_auc_score(video_labels, preds)
                ap = average_precision_score(video_labels, preds)

            file_result[stat_name] = {
                "AUC": None if auc is None else round(float(auc), 6),
                "AP":  None if ap is None else round(float(ap), 6)
            }

        results[detector][file] = file_result

# ===== 保存结果 =====
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print("\nAll done! Results saved to", output_file)
