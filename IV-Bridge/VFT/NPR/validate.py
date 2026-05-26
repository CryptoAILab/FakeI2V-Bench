import torch
import numpy as np
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, roc_auc_score
from options.test_options import TestOptions
from data import create_dataloader
from PIL import Image
import pickle
from torch.utils.data import Dataset
import os
import random
import torchvision.transforms as transforms
import pandas as pd
from tqdm import tqdm
DATASET_PATHS_D3 = [
        dict(
        real_path='/data2/lp/IVBridge_datasets/crop/test/split/0real',
        fake_path='/data2/lp/IVBridge_datasets/crop/test/split/1fake',
        data_mode='wang2020',
        key='split'
    ),
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
#     )
]

# def validate(model, opt, dataset_name):
#     loader = create_dataloader(opt)
def validate(model, loader, dataset_name, opt=None):
    with torch.no_grad():
        y_true, y_pred, video_list  = [], [], []
        for img, label, video_name in tqdm(loader):
            in_tens = img.cuda()
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())
            video_list.extend(video_name)

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    video_list = np.array(video_list)
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
  
    df_result = pd.DataFrame({
        "video_id": video_list,
        "label": y_true,
        "y_prob": y_pred
    })

    df_result.to_csv('results/{}.csv'.format(dataset_name), index=False)

    return acc, ap,auc, r_acc, f_acc, y_true, y_pred, video_list


def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg", "bmp"]):
    out = [] 
    for r, d, f in os.walk(rootdir):
        count = 0
        for file in f:
            if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
                count += 1
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

        # assert len(real_list) == len(fake_list)
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

if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    # # get model
    # model = resnet50(num_classes=1)
    # # model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=True)
    # try:
    #     model.load_state_dict(torch.load('/data/lp/NPR-DeepfakeDetection/NPR.pth', map_location='cpu'), strict=True)
    # except:
    #     from collections import OrderedDict
    #     from copy import deepcopy
    #     state_dict = torch.load('/data/lp/NPR-DeepfakeDetection/NPR.pth', map_location='cpu')['model']
    #     pretrained_dict = OrderedDict()
    #     for ki in state_dict.keys():
    #         pretrained_dict[ki[7:]] = deepcopy(state_dict[ki])
    #     model.load_state_dict(pretrained_dict, strict=True)

    model = resnet50(num_classes=1)
    model.load_state_dict(torch.load('checkpoints/experiment_name2025_08_15_17_24_01/model_epoch_6.pth', map_location='cpu'), strict=True)

    model.cuda()
    model.eval()

    opt.no_resize = False # Due to the different shapes of images in the dataset, resizing is required during batch detection.
    opt.no_crop = True

    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize)
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.cropSize)

    if opt.isTrain and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)
    if not opt.isTrain and opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        # rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))
        rz_func = transforms.Resize((opt.loadSize, opt.loadSize))


    transform = transforms.Compose([
        rz_func,
        # transforms.Lambda(lambda img: data_augment(img, opt)),
        crop_func,
        flip_func,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])



    dataset_paths = DATASET_PATHS_D3
    for dataset_path in dataset_paths:
        dataset_name = dataset_path["key"]
        print(f"dataset_path: {dataset_path}")
        
        dataset = RealFakeDataset(
            dataset_path["real_path"],
            dataset_path["fake_path"],
            dataset_path["data_mode"],
            transform= transform,
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.num_threads),
        )

        acc, ap,auc, r_acc, f_acc, y_true, y_pred, video_list = validate(model,loader, dataset_name, opt)


