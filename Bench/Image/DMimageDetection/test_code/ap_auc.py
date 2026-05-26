# %%
import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

csv_dir = "results_tst"   # ← 改成你的目录

results = []

# 遍历目录下所有 CSV 文件，包括子目录
csv_files = glob.glob(os.path.join(csv_dir, "**", "*.csv"), recursive=True)
print(f"Found {len(csv_files)} CSV files\n")

for csv_path in sorted(csv_files):
    print(f"Processing: {os.path.basename(csv_path)}")

    try:
        df = pd.read_csv(csv_path)

        # ========= 检查列 =========
        required_cols = ["label", "Grag2021_progan", "Grag2021_latent", "src"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        # ========= 修正 label =========
        # 如果 src 包含 "real"，label=False；否则 True
        df["label"] = df["src"].apply(lambda x: False if "real" in str(x).lower() else True)

        # ========= 计算平均分 =========
        df["prob"] = (
            df["Grag2021_progan"].astype(float) +
            df["Grag2021_latent"].astype(float)
        ) / 2.0

        # ========= 覆盖保存 =========
        df.to_csv(csv_path, index=False)

        # ========= 计算指标 =========
        labels = df["label"].astype(int).values  # 注意 AUC/AP 需要整数或0/1
        scores = df["prob"].values

        if len(np.unique(labels)) < 2:
            auc, ap = None, None
            print("  ⚠ Only one class present, skip metrics.")
        else:
            auc = roc_auc_score(labels, scores)
            ap = average_precision_score(labels, scores)

        print(f"  AUC: {auc:.4f} | AP: {ap:.4f}\n")

        results.append({
            "file": os.path.basename(csv_path),
            "AUC": None if auc is None else float(auc),
            "AP":  None if ap is None else float(ap)
        })

    except Exception as e:
        print(f"  ❌ Error: {e}\n")


# ========= 汇总结果 =========
print("\n====== Summary ======")
for r in results:
    print(f"{r['file']:30s} AUC={r['AUC']:.4f}  AP={r['AP']:.4f}")


# %%
