from time import process_time_ns
import torch
import clip
from PIL import Image
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix
import itertools
import torch.nn.functional as F
import pickle
import torch.nn as nn
from torch.utils.data import random_split
from torch import nn
from torchvision import transforms
import sys
import argparse
import time
import random
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import accuracy_score, average_precision_score
from blipmodels import blip_decoder
import pandas as pd
from datasets.ivb_dataset import DATASET_PATHS_IVB
from datasets.image_datasets import DATASET_PATHS_IMAGE


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_list, num_classes):
        super(NeuralNet, self).__init__()
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, hidden_size_list[0])
        self.fc2 = nn.Linear(hidden_size_list[0], hidden_size_list[1])
        self.fc3 = nn.Linear(hidden_size_list[1], num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out

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
def validate(model,linear, loader, dataset_name, opt=None):
    with torch.no_grad():
        y_true, y_pred, video_list = [], [], []
        print("Length of dataset: %d" % (len(loader)))
        for image, text, label, video_name in tqdm(loader):
            # import pdb; pdb.set_trace()
            image.cuda()
            text = [t[:77] for t in text]
            text = clip.tokenize(list(text)).cuda()
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            emb = torch.cat((image_features, text_features),1)
            output = linear(emb.float())
            predict = torch.softmax(output, dim=1)[:, 1]
            # import pdb; pdb.set_trace()
            # predict = output.argmax(1)
            # predict = predict.cpu().numpy()

            y_pred.extend(predict.flatten().tolist())
            y_true.extend(label.flatten().tolist())
            video_list.extend(video_name)

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    video_list = np.array(video_list)
    threshold = 0.5
    df_result = pd.DataFrame({
        "video_id": video_list,
        "label": y_true,
        "y_prob": y_pred
    })
    df_result.to_csv('{}/{}.csv'.format(opt.result_folder, dataset_name), index=False)
    auc = metrics.roc_auc_score(y_true, y_pred)
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, threshold)
    ap = average_precision_score(y_true, y_pred)
    return acc0, auc, ap

def recursively_read(path, 
    must_contain
):
    exts=["png", "jpg", "JPEG", "jpeg", "bmp", "JPG"]
    frames = []
    
    for file in os.listdir(path):            
        if (file.split('.')[-1] in exts) and (must_contain in path):
            frames.append(os.path.join(path, file))
    return frames

def get_list(path, must_contain=""):
    if ".pickle" in path:
        with open(path, "rb") as f:
            image_list = pickle.load(f)
        image_list = [item for item in image_list if must_contain in item]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list

model2, preprocess = clip.load("ViT-B/32")
def preprocess_image(img_path, image_size=224):
    img = Image.open(img_path)
    img = img.resize((image_size, image_size))
    return preprocess(img)

class RealFakeDataset(Dataset):
    def __init__(
        self,
        real_path,
        fake_path,
        data_mode,
        prompts_real,
        prompts_fake,
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

        # self.total_list = real_list + fake_list
        self.real_prompts = self.load_prompts(prompts_real)
        self.fake_prompts = self.load_prompts(prompts_fake)

        # = = = = = =  label = = = = = = = = = #

        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1
        # = = = = = =  prompt = = = = = = = = = #
        self.prompts = {}
        for i in real_list:
            # frame = ('/').join(i.split("/")[-2:])
            frame = i.split("/")[-1]
            if frame in self.real_prompts:  # 检查键是否存在
                self.prompts[i] = self.real_prompts[frame]
            # 如果 frame 不存在，跳过当前的 i

        for i in fake_list:
            # frame = ('/').join(i.split("/")[-2:])
            frame = i.split("/")[-1]
            if frame in self.fake_prompts:  # 检查键是否存在
                self.prompts[i] = self.fake_prompts[frame]
            # 如果 frame 不存在，跳过当前的 i
        
        self.total_list = list(self.prompts.keys())

        # = = = = = =  transform = = = = = = = = = #
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        # = = = = = =  blip = = = = = = = = = #
        # blip_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
        # self.blip = blip_decoder(pretrained=blip_url, image_size=224, vit='base')
    def read_path(self, real_path, fake_path, data_mode):

        if data_mode == "wang2020":
            real_list = get_list(real_path, must_contain="real")
            fake_list = get_list(fake_path, must_contain="fake")
        else:
            real_list = get_list(real_path)
            fake_list = get_list(fake_path)

        # assert len(real_list) == len(fake_list)
        print("real: %d, fake: %d" % (len(real_list), len(fake_list)))
        return real_list, fake_list

    def __len__(self):
        return len(self.total_list)
    
    def load_prompts(self, prompts_file):
        prompts_dict = {}
        with open(prompts_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # key, value = line.split(':')
                print(line)
                key, value = line.rsplit(':', 1)
                print(value)
                # import pdb; pdb.set_trace()
                # key = line
                # value = "this is an image"
                key = key.strip()
                value = value.strip()
                prompts_dict[key] = value
        return prompts_dict

    def __getitem__(self, idx):

        img_path = self.total_list[idx]
        label = self.labels_dict[img_path]
        video_name = (img_path.split('/')[-1]).split('.')[0]
        #image
        image = preprocess_image(img_path,image_size=224).to("cuda")

        #text
        # self.blip.eval()
        # self.blip = self.blip.to("cuda")

        img = Image.open(img_path).convert('RGB')

        img = self.transform(img)
        img = img.unsqueeze(0).to("cuda")

        # caption = self.blip.generate(img, sample=False, num_beams=3, max_length=60, min_length=5) 
        text = self.prompts[img_path]

        # text = clip.tokenize(list(caption)).to("cuda")

        return image, text, label, video_name
    

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
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--classificator_type", type=str, default="linear")
    opt = parser.parse_args()

    os.makedirs(opt.result_folder, exist_ok=True)


    # model 
    model2, preprocess = clip.load("ViT-B/32", device="cuda")
    model = torch.load("weights/finetune_clip.pt").to("cuda")
    linear = NeuralNet(1024,[512,256],2).to("cuda")
    linear = torch.load('weights/clip_linear.pt')
    
    # model, preprocess = clip.load("ViT-B/32", device="cuda")
    # model.load_state_dict(torch.load('weights/train_clip_model.pth'))
    # linear = NeuralNet(1024,[512,256],2).to("cuda")
    # linear.load_state_dict(torch.load('weights/train_linear_model.pth'))


    print("Model loaded..")
    model.eval()
    model.cuda()
    dataset_paths = DATASET_PATHS_IMAGE

    sum_acc = 0
    sum_auc = 0
    sum_ap  = 0
    num_sets = len(dataset_paths)


    for dataset_path in dataset_paths:
        print(f"dataset_path: {dataset_path}")
        dataset_name = dataset_path["key"]
        dataset = RealFakeDataset(
            dataset_path["real_path"],
            dataset_path["fake_path"],
            dataset_path["data_mode"],
            dataset_path["prompts_real"],
            dataset_path["prompts_fake"],
        )
        print(len(dataset))

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
        )
        acc0, auc, ap = validate(model,linear, loader, dataset_name, opt=opt)

        sum_acc += acc0
        sum_auc += auc
        sum_ap  += ap

        with open(os.path.join(opt.result_folder, "re.txt"), "a") as f:
            f.write(
                dataset_path["key"]
                + ": "
                + str(round(acc0 * 100, 2))
                + "  "
                + str(round(auc * 100, 2))
                + "  "
                + str(round(ap * 100, 2))
                + "\n"
            )
    mean_acc = sum_acc / num_sets
    mean_auc = sum_auc / num_sets
    mean_ap  = sum_ap  / num_sets

    print("\n====== Overall Average (Unweighted) ======")
    print(f"Acc: {mean_acc*100:.2f}  AUC: {mean_auc*100:.2f}  AP: {mean_ap*100:.2f}")

    with open(os.path.join(opt.result_folder, "re.txt"), "a") as f:
        f.write("=====================================\n")
        f.write(f"Overall: {mean_acc*100:.2f}  {mean_auc*100:.2f}  {mean_ap*100:.2f}\n")
