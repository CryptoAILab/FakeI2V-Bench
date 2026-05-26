import os
import csv

# 要扫描的根目录
ROOT_DIR = "/data/lp/IVBridge-OriCode/datasets"

# 输出 CSV 文件
OUTPUT_CSV = "images_list.csv"

# 支持的图片后缀
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# 收集所有图片相对路径
image_paths = []

for root, dirs, files in os.walk(ROOT_DIR):
    for file in files:
        if file.lower().endswith(IMG_EXTS):
            # 相对路径（相对于 ROOT_DIR）
            rel_path = os.path.relpath(os.path.join(root, file), ROOT_DIR)
            image_paths.append(rel_path)

# 保存为 CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    # 写入表头
    writer.writerow(["src"])
    # 写入每一行图片相对路径
    for path in image_paths:
        writer.writerow([path])

print(f"共找到 {len(image_paths)} 张图片，已保存到 {OUTPUT_CSV}")
