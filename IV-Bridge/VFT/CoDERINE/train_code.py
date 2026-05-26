import argparse
import os
import sys
import torch
import torch.nn as nn  # 添加这一行
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import Dataset
from PIL import Image
import pickle
from tqdm import tqdm
from code_model import VITContrastiveHF
import pandas as pd
import random
from adv_dataset import DATASET_PATHS_ADV

# DATASET_PATHS_D3 = [
#     dict(
#         real_path='/data2/lp/IVBridge_datasets/crop/val/CDFV2/0real',
#         fake_path='/data2/lp/IVBridge_datasets/crop/val/CDFV2/1fake',
#         data_mode='wang2020',
#         key='CDFV2'
#     ),
#     dict(
#         real_path='/data2/lp/IVBridge_datasets/crop/val/DFD/0real',
#         fake_path='/data2/lp/IVBridge_datasets/crop/val/DFD/1fake',
#         data_mode='wang2020',
#         key='DFD'
#     ),
#     dict(
#         real_path='/data2/lp/IVBridge_datasets/crop/val/GenVideo/0real',
#         fake_path='/data2/lp/IVBridge_datasets/crop/val/GenVideo/1fake',
#         data_mode='wang2020',
#         key='GenVideo'
#     ),
#     dict(
#         real_path='/data2/lp/IVBridge_datasets/crop/val/GVB/0real',
#         fake_path='/data2/lp/IVBridge_datasets/crop/val/GVB/1fake',
#         data_mode='wang2020',
#         key='GVB'
#     ),
#         dict(
#         real_path='/data2/lp/IVBridge_datasets/crop/val/Facial/0real',
#         fake_path='/data2/lp/IVBridge_datasets/crop/val/Facial/1fake',
#         data_mode='wang2020',
#         key='Facial'
#     ),
#     dict(
#         real_path='/data2/lp/IVBridge_datasets/crop/val/General/0real',
#         fake_path='/data2/lp/IVBridge_datasets/crop/val/General/1fake',
#         data_mode='wang2020',
#         key='General'
#     )
# ]

# DATASET_PATHS_D3 = [
#     dict(
#         real_path='/data2/lp/IVBridge_datasets/crop/val/emb4/0real',
#         fake_path='/data2/lp/IVBridge_datasets/crop/val/emb4/1fake',
#         data_mode='wang2020',
#         key='emb4'
#     ),
#     dict(
#         real_path='/data2/lp/IVBridge_datasets/crop/val/emb8/0real',
#         fake_path='/data2/lp/IVBridge_datasets/crop/val/emb8/1fake',
#         data_mode='wang2020',
#         key='emb8'
#     ),
#     dict(
#         real_path='/data2/lp/IVBridge_datasets/crop/val/emb16/0real',
#         fake_path='/data2/lp/IVBridge_datasets/crop/val/emb16/1fake',
#         data_mode='wang2020',
#         key='emb16'
#     ),
#     dict(
#         real_path='/data2/lp/IVBridge_datasets/crop/val/emb32/0real',
#         fake_path='/data2/lp/IVBridge_datasets/crop/val/emb32/1fake',
#         data_mode='wang2020',
#         key='emb32'
#     ),
#         dict(
#         real_path='/data2/lp/IVBridge_datasets/crop/val/imprint0/0real',
#         fake_path='/data2/lp/IVBridge_datasets/crop/val/imprint0/1fake',
#         data_mode='wang2020',
#         key='imprint0'
#     ),
#             dict(
#         real_path='/data2/lp/IVBridge_datasets/crop/val/imprint30/0real',
#         fake_path='/data2/lp/IVBridge_datasets/crop/val/imprint30/1fake',
#         data_mode='wang2020',
#         key='imprint30'
#     ),
#             dict(
#         real_path='/data2/lp/IVBridge_datasets/crop/val/imprint70/0real',
#         fake_path='/data2/lp/IVBridge_datasets/crop/val/imprint70/1fake',
#         data_mode='wang2020',
#         key='imprint70'
#     ),
#             dict(
#         real_path='/data2/lp/IVBridge_datasets/crop/val/imprint90/0real',
#         fake_path='/data2/lp/IVBridge_datasets/crop/val/imprint90/1fake',
#         data_mode='wang2020',
#         key='imprint90'
#     ),
#             dict(
#         real_path='/data2/lp/IVBridge_datasets/crop/val/imprint120/0real',
#         fake_path='/data2/lp/IVBridge_datasets/crop/val/imprint120/1fake',
#         data_mode='wang2020',
#         key='imprint120'
#     ),
#             dict(
#         real_path='/data2/lp/IVBridge_datasets/crop/val/imprint150/0real',
#         fake_path='/data2/lp/IVBridge_datasets/crop/val/imprint150/1fake',
#         data_mode='wang2020',
#         key='imprint150'
#     ),
# ]

DATASET_PATHS_D3 = [
    dict(
        real_path='/data2/lp/IVBridge_datasets/crop/val/adv_ori/0real',
        fake_path='/data2/lp/IVBridge_datasets/crop/val/adv_ori/1fake',
        data_mode='wang2020',
        key='adv_ori'
    ),
]

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)
def set_seed(SEED=0):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > thres)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc


def calculate_acc_svm(y_true, y_pred):
    y_true[y_true == 1] = -1
    y_true[y_true == 0] = 1
    r_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1])  # > thres)
    f_acc = accuracy_score(y_true[y_true == -1], y_pred[y_true == -1])  # > thres)
    acc = accuracy_score(y_true, y_pred)  # > thres)
    return r_acc, f_acc, acc


@torch.inference_mode()
def validate(model, loader, dataset_name, opt=None):
    with torch.no_grad():
        y_true, y_pred, video_list = [], [], []
        print("Length of dataset: %d" % (len(loader)))
        for img, label, video_name in tqdm(loader, disable=True):
            in_tens = img.cuda()
            y_pred.extend(model(in_tens).flatten().tolist())
            y_true.extend(label.flatten().tolist())
            video_list.extend(video_name)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    video_list = np.array(video_list)
    threshold = 0

    if model.classificator_type == "linear" or model.classificator_type == "knn":
        r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, threshold)
    else:
        r_acc0, f_acc0, acc0 = calculate_acc_svm(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    return auc, r_acc0, f_acc0, acc0, y_true, y_pred, video_list


def train(model, train_loader, criterion, optimizer, epoch, device='cuda'):
    model.train()  # 将模型设置为训练模式
    model.to(device)

    print(f"Epoch [{epoch + 1}] - Training")
    running_loss = 0.0
    pbar = tqdm(train_loader, disable=False)  # 创建进度条对象
    for img, label, _ in pbar:  # 遍历训练数据加载器
        img, label = img.to(device), label.to(device)
        label = label.float()
        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img.size(0)

        # 实时更新进度条的后缀信息（当前 batch 的 loss）
        pbar.set_postfix({"loss": loss.item()})

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch + 1}] - Loss: {epoch_loss:.4f}")
    # torch.save(model.state_dict(), f"/data/lp/CoDE/CoDE_model/weight/VFT/CoDE/checkpoint_epoch_{epoch+1}.pth")

def recursively_read(
    rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg", "bmp", "JPG"]
):
    out = []
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split(".")[1] in exts) and (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain=""):
    if ".pickle" in path:
        with open(path, "rb") as f:
            image_list = pickle.load(f)
        image_list = [item for item in image_list if must_contain in item]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list


class RealFakeDataset(Dataset):
    def __init__(
        self,
        real_path,
        fake_path,
        data_mode,
        transform,
    ):

        assert data_mode in ["wang2020", "ours"]
        # = = = = = = data path = = = = = = = = = #
        if type(real_path) == str and type(fake_path) == str:
            real_list, fake_list = self.read_path(real_path, fake_path, data_mode)
        else:
            real_list = []
            fake_list = []
            for real_p, fake_p in zip(real_path, fake_path):
                real_l, fake_l = self.read_path(real_p, fake_p, data_mode)
                real_list += real_l
                fake_list += fake_l

        self.total_list = real_list + fake_list


        # = = = = = =  label = = = = = = = = = #

        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1
        self.transform = transform

    def read_path(self, real_path, fake_path, data_mode):

        if data_mode == "wang2020":
            real_list = get_list(real_path, must_contain="real")
            fake_list = get_list(fake_path, must_contain="fake")
        else:
            real_list = get_list(real_path)
            fake_list = get_list(fake_path)

        print("real: %d, fake: %d" % (len(real_list), len(fake_list)))
        return real_list, fake_list

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):

        img_path = self.total_list[idx]

        label = self.labels_dict[img_path]

        img = Image.open(img_path).convert("RGB")

        img = self.transform(img)
        img_name = img_path.split('/')[-1]
        video_name = ('_').join(img_name.split('_')[:-1])
        return img, label, video_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--real_path", type=str, default=None, help="dir name or a pickle"
    )
    parser.add_argument(
        "--fake_path", type=str, default=None, help="dir name or a pickle"
    )
    parser.add_argument("--data_mode", type=str, default=None, help="wang2020 or ours")
    parser.add_argument("--result_folder", type=str, default="./results", help="")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--classificator_type", type=str, default="linear")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for optimizer")
    opt = parser.parse_args()

    os.makedirs(opt.result_folder, exist_ok=True)

    model = VITContrastiveHF(classificator_type=opt.classificator_type)
    model.to("cuda")  # 将模型移动到GPU

    transform = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    print("Model loaded..")

    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()  # 假设是二分类任务
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)

    dataset_paths = DATASET_PATHS_ADV  # 假设你已经定义了DATASET_PATHS_D3

    train_dataset = RealFakeDataset(
        real_path = '/data2/lp/IVBridge_datasets/crop/train0/0real',
        fake_path = '/data2/lp/IVBridge_datasets/crop/train0/1fake',
        data_mode = 'wang2020',
        transform=transform,
    )
    print(f"train_dataset: {len(train_dataset)}")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
    )
    result_root = "/data/lp/CoDE/CoDE_model/results/train-ADV/code"
    # 训练模型
    for epoch in range(opt.num_epochs):

        train(model, train_loader, criterion, optimizer, epoch)

        for dataset_path in dataset_paths:
            set_seed()
            print(f"dataset_path: {dataset_path}")
            # 验证模型
            dataset_name = dataset_path["key"]
            val_dataset = RealFakeDataset(
                dataset_path["real_path"],
                dataset_path["fake_path"],
                dataset_path["data_mode"],
                transform=transform,
            )

            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.num_workers,
            )
            auc, r_acc0, f_acc0, acc0, y_true, y_pred, IDlist = validate(model, val_loader, dataset_name, opt=opt)

            df_result = pd.DataFrame({
                'video_id': IDlist,
                'label': y_true, 
                'y_prob': y_pred,
                })

            result_dir = os.path.join(result_root, 'epoch_{}'.format(epoch))
            mkdir(result_dir)
            df_result.to_csv(os.path.join(result_dir, '{}.csv'.format(dataset_name)), index=False)

            with open(result_dir + '/{}.txt'.format(dataset_name), 'w') as f:
                f.write("(Val @ epoch {}) acc: {};\n".format(epoch, acc0))
                f.write("(Val @ epoch {}) r_acc: {}; f_acc: {}\n".format(epoch, r_acc0, f_acc0))
                f.write("(Val @ epoch {}) auc: {}\n".format(epoch, auc))

