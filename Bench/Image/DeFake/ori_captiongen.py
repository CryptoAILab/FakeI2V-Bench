import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from blipmodels import blip_decoder
from tqdm import tqdm

# ================== 基本配置 ==================
ROOT_DATA = '/data/lp/IVBridge-OriCode/datasets'
RESULT_ROOT = '/data/lp/IVBridge-OriCode/DeFake/captions'
IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp')


BATCH_SIZE = 64      # ⭐ 可以改 16 / 32 / 64 看显存
NUM_WORKERS = 8
device = "cuda:5" if torch.cuda.is_available() else "cpu"

# ================== BLIP 模型 ==================
blip_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth"
blip = blip_decoder(pretrained=blip_url, image_size=224, vit='base')
blip.eval().to(device)

tform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# ================== 类别识别 ==================
def get_label_from_folder(folder_name: str):
    name = folder_name.lower()
    if any(k in name for k in ['0real', '0_real', 'real']):
        return 'real'
    if any(k in name for k in ['1fake', '1_fake', 'fake']):
        return 'fake'
    return None


# ================== 收集所有图片路径 ==================
samples = []  # [(img_path, dataset_name, label)]

for root, dirs, files in os.walk(ROOT_DATA):
    folder_name = os.path.basename(root)
    label = get_label_from_folder(folder_name)
    if label is None:
        continue

    dataset_name = os.path.basename(os.path.dirname(root))

    for f in files:
        if f.lower().endswith(IMG_EXT):
            img_path = os.path.join(root, f)
            samples.append((img_path, dataset_name, label))

print(f"🔥 Total images found: {len(samples)}")


# ================== 自定义 Dataset ==================
class ImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, dataset_name, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, img_path, dataset_name, label


dataset = ImageDataset(samples, tform)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# ================== 打开所有输出文件（避免频繁IO） ==================
file_handles = {}  # (dataset, label) -> file object

def get_file(dataset, label):
    key = (dataset, label)
    if key not in file_handles:
        out_dir = os.path.join(RESULT_ROOT, dataset)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{dataset}_{label}.txt")
        file_handles[key] = open(out_path, 'a')
    return file_handles[key]


# ================== 批量推理 ==================
with torch.no_grad():
    for imgs, paths, datasets, labels in tqdm(loader):
        imgs = imgs.to(device, non_blocking=True)

        captions = blip.generate(
            imgs,
            sample=False,
            num_beams=3,
            max_length=60,
            min_length=5
        )

        for p, cap, ds, lb in zip(paths, captions, datasets, labels):
            # rel_path = os.path.relpath(p, ROOT_DATA)
            # f = get_file(ds, lb)
            # f.write(f"{rel_path}: {cap}\n")
            name = os.path.basename(p)
            f = get_file(ds, lb)
            f.write(f"{name}: {cap}\n")

# 关闭文件
for f in file_handles.values():
    f.close()

print("🎉 All captions generated!")
