import argparse
from ast import arg
import os
import csv
import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, roc_auc_score
from torch.utils.data import Dataset
import sys
from models import get_model
from PIL import Image 
import pickle
from tqdm import tqdm
from io import BytesIO
from copy import deepcopy
import random
import pandas as pd
import shutil
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm
from collections import defaultdict
# python validate.py  --arch=CLIP:ViT-L/14   --ckpt=pretrained_weights/fc_weights.pth   --result_folder=clip_vitl14 
DATASET_PATHS = [
    dict(
        real_path='/data2/lp/IVBridge_datasets/crop/test/CDFV2/0real',
        fake_path='/data2/lp/IVBridge_datasets/crop/test/CDFV2/1fake',
        data_mode='wang2020',
        key='CDFV2'
    ),
    dict(
        real_path='/data2/lp/IVBridge_datasets/crop/test/DFD/0real',
        fake_path='/data2/lp/IVBridge_datasets/crop/test/DFD/1fake',
        data_mode='wang2020',
        key='DFD'
    ),
    dict(
        real_path='/data2/lp/IVBridge_datasets/crop/test/GenVideo/0real',
        fake_path='/data2/lp/IVBridge_datasets/crop/test/GenVideo/1fake',
        data_mode='wang2020',
        key='GenVideo'
    ),
    dict(
        real_path='/data2/lp/IVBridge_datasets/crop/test/GVB/0real',
        fake_path='/data2/lp/IVBridge_datasets/crop/test/GVB/1fake',
        data_mode='wang2020',
        key='GVB'
    )
]

SEED = 0
def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}





def find_best_threshold(y_true, y_pred):
    "We assume first half is real 0, and the second half is fake 1"

    N = y_true.shape[0]

    if y_pred[0:N//2].max() <= y_pred[N//2:N].min(): # perfectly separable case
        return (y_pred[0:N//2].max() + y_pred[N//2:N].min()) / 2 

    best_acc = 0 
    best_thres = 0 
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp>=thres] = 1 
        temp[temp<thres] = 0 

        acc = (temp == y_true).sum() / N  
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc 
    
    return best_thres
        

 
def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality) # ranging from 0-95, 75 is default
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def gaussian_blur(img, sigma):
    img = np.array(img)

    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

    return Image.fromarray(img)



def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > thres)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc    


def validate(model, loader, data_name):

    with torch.no_grad():
        y_true, y_pred, video_list = [], [], []
        print ("Length of dataset: %d" %(len(loader)))
        for imgs, label, video_name in tqdm(loader):
            b, t, _, h, w = imgs.shape
            imgs = imgs.reshape(-1, 3, h, w)
            in_tens = imgs.cuda()
            # import pdb; pdb.set_trace()
            output = model(in_tens).sigmoid().reshape(b, t, -1)
            output = torch.mean(output, dim=1)
            # import pdb; pdb.set_trace()
            y_pred.extend(output.flatten().tolist())
            y_true.extend(label.flatten().tolist())
            video_list.extend(video_name)

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    video_list = np.array(video_list)
    threshold = 0.5
    acc = accuracy_score(y_true, y_pred > threshold)
    ap = average_precision_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > threshold)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > threshold)

    df_result = pd.DataFrame({
        "video_id": video_list,
        "label": y_true,
        "y_prob": y_pred
    })
    df_result.to_csv('/data/lp/UniversalFakeDetect/result/NOWM/{}.csv'.format(data_name))
    return acc, ap, auc, r_acc, f_acc, y_true, y_pred, video_list

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 

IMG_EXTS = ["png", "jpg", "jpeg", "bmp", "JPG", "JPEG"]


def recursively_read(rootdir):
    out = []
    for r, _, f in os.walk(rootdir):
        for file in f:
            if file.split(".")[-1] in IMG_EXTS:
                out.append(os.path.join(r, file))
    return out

class RealFakeVideoDataset(Dataset):
    def __init__(self, real_path, fake_path, transform, max_videos = None):
        self.transform = transform

        self.real_list = recursively_read(real_path)
        self.fake_list = recursively_read(fake_path)

        self.video_dict = self.build_video_index(self.real_list, self.fake_list)
        # self.video_keys = list(self.video_dict.keys())

        real_keys = [k for k in self.video_dict if self.video_dict[k]["label"] == 0]
        fake_keys = [k for k in self.video_dict if self.video_dict[k]["label"] == 1]

        random.shuffle(real_keys)
        random.shuffle(fake_keys)

        if max_videos is not None:
            half = max_videos // 2
            self.video_keys = real_keys[:half] + fake_keys[:half]
        else:
            self.video_keys = real_keys + fake_keys

        random.shuffle(self.video_keys)

        print(f"Total videos: {len(self.video_keys)}")

    def build_video_index(self, real_list, fake_list):
        video_dict = {}

        def extract_frame_id(path):
            name = os.path.basename(path)
            return int(name.split("_")[-1].split(".")[0])

        def process(paths, label):
            local_dict = defaultdict(list)
            for p in paths:
                name = os.path.basename(p)
                parts = name.split("_")
                video_key = "_".join(parts[:-1])
                local_dict[video_key].append(p)

            for k in local_dict:
                local_dict[k] = sorted(local_dict[k], key=extract_frame_id)

                if len(local_dict[k]) < 8:
                    continue
                    raise ValueError(f"Video {k} has only {len(local_dict[k])} frames")
                    

                video_dict[k] = {
                    "frames": local_dict[k],
                    "label": label
                }

        process(real_list, 0)
        process(fake_list, 1)
        return video_dict


    def __len__(self):
        return len(self.video_keys)

    def __getitem__(self, idx):
        video_name = self.video_keys[idx]
        record = self.video_dict[video_name]

        frames = record["frames"]
        label = record["label"]

        start = random.randint(0, len(frames) - 8)
        clip_paths = frames[start:start + 8]
        # print(clip_paths)
        imgs = []
        for p in clip_paths:
            img = Image.open(p).convert("RGB")
            img = self.transform(img)
            imgs.append(img)

        imgs = torch.stack(imgs)  # [8, C, H, W]

        return imgs, label, video_name


if __name__ == '__main__':


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--real_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--fake_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--data_mode', type=str, default=None, help='wang2020 or ours')
    parser.add_argument('--max_sample', type=int, default=10000, help='only check this number of images for both fake/real')

    parser.add_argument('--arch', type=str, default='res50')
    parser.add_argument('--ckpt', type=str, default='./pretrained_weights/fc_weights.pth')

    parser.add_argument('--result_folder', type=str, default='result/707F2V', help='')
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--jpeg_quality', type=int, default=None, help="100, 90, 80, ... 30. Used to test robustness of our model. Not apply if None")
    parser.add_argument('--gaussian_sigma', type=int, default=None, help="0,1,2,3,4.     Used to test robustness of our model. Not apply if None")


    opt = parser.parse_args()

    
    if os.path.exists(opt.result_folder):
        shutil.rmtree(opt.result_folder)
    os.makedirs(opt.result_folder)

    # model = get_model(opt.arch)
    # state_dict = torch.load(opt.ckpt, map_location='cpu')
    # model.fc.load_state_dict(state_dict)

    # # checkpoint = torch.load(opt.ckpt, map_location='cpu')
    # # state_dict = checkpoint['model']
    # # new_state_dict = {'weight':state_dict['fc.weight'], 'bias':state_dict['fc.bias']}
    # # model.fc.load_state_dict(new_state_dict)
    # print ("Model loaded..")
    # model.eval()
    # model.cuda()

    
    from models import get_model
    model = get_model(opt.arch)
    checkpoint = torch.load(opt.ckpt, map_location='cpu')
    state_dict = checkpoint['model']
    new_state_dict = {'weight':state_dict['fc.weight'], 'bias':state_dict['fc.bias']}
    model.fc.load_state_dict(new_state_dict)
    model.cuda().eval()

    if (opt.real_path == None) or (opt.fake_path == None) or (opt.data_mode == None):
        dataset_paths = DATASET_PATHS
    else:
        dataset_paths = [ dict(real_path=opt.real_path, fake_path=opt.fake_path, data_mode=opt.data_mode) ]

    stat_from = "imagenet" if opt.arch.lower().startswith("imagenet") else "clip"
    transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
        ])

    for dataset_path in (dataset_paths):
        set_seed()
        data_name = dataset_path['key']
        dataset = RealFakeVideoDataset(  dataset_path['real_path'], 
                                    dataset_path['fake_path'],
                                    transform=transform
                                    )

        loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)
        acc, ap, auc, r_acc, f_acc, y_true, y_pred, video_list = validate(model, loader, data_name)
        print("acc", acc)
        print("ap", ap)
        print("auc", auc)
        


