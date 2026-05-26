# %%
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

csv_files = [
    # "vript_cogvideo_results.csv",  # 这个保留 real
    # "vript_mora_results.csv",
    # "vript_musev_results.csv",
    "vript_svd_results.csv",
]

# 读第一个（保留 real + fake）
df_base = pd.read_csv(csv_files[0])

# # 其余只取 fakes
# dfs_fake_only = []
# for f in csv_files[:]:
#     df_tmp = pd.read_csv(f)
#     df_fake = df_tmp[df_tmp["label"] == 1]  # 只保留 fake
#     dfs_fake_only.append(df_fake)

# 合并
df_final = pd.concat([df_base], ignore_index=True)

print("Real videos:", sum(df_final["label"] == 0))
print("Fake videos:", sum(df_final["label"] == 1))
print("Total videos:", len(df_final))

# ---------------- 指标 ----------------
y_true = df_final["label"].values
y_score = df_final["prob_1"].values  # fake 概率

auc = roc_auc_score(y_true, y_score)
ap = average_precision_score(y_true, y_score)

print(f"\nFinal Mixed AUC: {auc:.4f}")
print(f"Final Mixed AP:  {ap:.4f}")

# 保存融合结果
df_final.to_csv("merged_eval.csv", index=False)

# %%
