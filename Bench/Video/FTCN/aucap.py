import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# ======================
# CSV 目录
# ======================
csv_dir = "./"   # ← 改成你的目录

all_dfs = []

# ======================
# 读取并合并 CSV
# ======================
for fname in os.listdir(csv_dir):
    if fname.endswith(".csv"):
        print(fname)
        path = os.path.join(csv_dir, fname)
        df = pd.read_csv(path)

        # 保证列存在
        assert {"video_id", "label", "score"}.issubset(df.columns), f"{fname} 列名不对"

        # 去除 score 为空的行
        df = df.dropna(subset=["score"])

        # 有些 score 可能是空字符串
        df = df[df["score"] != ""]

        # 转成数值（防止字符串）
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        df = df.dropna(subset=["score"])

        all_dfs.append(df)

# 合并
merged_df = pd.concat(all_dfs, ignore_index=True)

print(f"Total samples after cleaning: {len(merged_df)}")

# ======================
# 计算 AUC / AP
# ======================
y_true = merged_df["label"].values
y_score = merged_df["score"].values

auc = roc_auc_score(y_true, y_score)
ap = average_precision_score(y_true, y_score)

print(f"AUC: {auc:.6f}")
print(f"AP : {ap:.6f}")
