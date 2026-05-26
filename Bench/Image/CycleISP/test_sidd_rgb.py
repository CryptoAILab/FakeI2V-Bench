"""
## CycleISP: Real Image Restoration Via Improved Data Synthesis
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## CVPR 2020
## https://arxiv.org/abs/2003.07761
"""

import numpy as np
import os
import argparse
from tqdm import tqdm
from torchvision import transforms
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
import scipy.io as sio
from networks.denoising_rgb import DenoiseNet
from dataloaders.data_rgb import get_validation_data
import utils
import cv2
from skimage import img_as_ubyte

parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--input_dir', default='/data/lp/IVBridge-OriCode/LNP/CNNDetection/dataset/',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='/data/lp/IVBridge-OriCode/LNP/CNNDetection/train_dataset/',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/denoising/sidd_rgb.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='1', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir)
count = 0

IMG_EXT = ('.jpg', '.png', '.jpeg', '.bmp')

def find_leaf_dirs(root):
    leaf_dirs = []
    for r, d, files in os.walk(root):
        # 这个目录下有图片文件
        has_img = any(f.lower().endswith(IMG_EXT) for f in files)
        # 且没有子目录（最底层）
        if has_img and len(d) == 0:
            leaf_dirs.append(r)
    return leaf_dirs

vals = find_leaf_dirs(args.input_dir)
print(f"🔥 Found {len(vals)} leaf image folders")


for input_dir in vals:
    # 取数据集名字（比如 no_wm/0_real）
    subname = os.path.relpath(input_dir, args.input_dir)
    result_dir = os.path.join(args.result_dir, subname)

    utils.mkdir(result_dir)
    print(f"\n📂 Processing: {input_dir}")


    test_dataset = get_validation_data(input_dir)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=0, drop_last=False)
    print(len(test_dataset))
    model_restoration = DenoiseNet()
    utils.load_checkpoint(model_restoration,args.weights)
    print("===>Testing using weights: ", args.weights)
    model_restoration.cuda()
    model_restoration=nn.DataParallel(model_restoration)
    model_restoration.eval()
    with torch.no_grad():
        psnr_val_rgb = []
        for ii, data_test in enumerate(test_loader, 0):

            rgb_noisy = data_test[0].cuda()
            filenames = data_test[1]
            rgb_restored = model_restoration(rgb_noisy)
            rgb_restored = torch.clamp(rgb_restored,0,1)

            rgb_noisy = rgb_noisy.permute(0, 2, 3, 1).cpu().detach().numpy()
            rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()

            if args.save_images:
                for batch in range(len(rgb_noisy)):
                    denoised_img = img_as_ubyte(rgb_restored[batch])
                    cv2.imwrite(result_dir + '/' + filenames[batch][:-4] + '.jpg', denoised_img * 255)
