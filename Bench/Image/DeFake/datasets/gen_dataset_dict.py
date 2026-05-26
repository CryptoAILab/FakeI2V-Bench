import os

DATA_ROOT = '/data/lp/IVBridge-OriCode/datasets'
CAPTION_ROOT = '/data/lp/IVBridge-OriCode/DeFake/captions'

entries = []

for root, dirs, files in os.walk(DATA_ROOT):
    # 判断这个目录是否是“数据集叶子目录”
    real_path = os.path.join(root, '0_real')
    fake_path = os.path.join(root, '1_fake')

    if not os.path.isdir(fake_path):
        continue  # 没 fake 不是生成数据集

    # 数据集名字 = 当前目录名（最后一级）
    key = os.path.basename(root)

    entry = f"""    dict(
        real_path='{real_path}',
        fake_path='{fake_path}',
        prompts_real='{CAPTION_ROOT}/{key}/{key}_real.txt',
        prompts_fake='{CAPTION_ROOT}/{key}/{key}_fake.txt',
        data_mode='wang2020',
        key='{key}'
    ),"""

    entries.append(entry)

print("DATASET_PATHS_IVB = [")
for e in sorted(entries):
    print(e)
print("]")

