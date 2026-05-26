import os
import shutil
from math import ceil

target_dir = "no_cache_videos"   # 你的目录
num_parts = 8

# 读取所有文件（只要文件，不要子目录）
files = [f for f in os.listdir(target_dir)
         if os.path.isfile(os.path.join(target_dir, f))]

files.sort()  # 排序保证稳定

total_files = len(files)
print(f"Total files: {total_files}")

# 每份大约多少个
chunk_size = ceil(total_files / num_parts)

for i in range(num_parts):
    part_dir = os.path.join(target_dir, f"part_{i}")
    os.makedirs(part_dir, exist_ok=True)

    start = i * chunk_size
    end = min((i + 1) * chunk_size, total_files)

    for f in files[start:end]:
        src_path = os.path.join(target_dir, f)
        dst_path = os.path.join(part_dir, f)

        shutil.move(src_path, dst_path)

    print(f"part_{i}: {end - start} files")

print("✅ Done splitting files.")
