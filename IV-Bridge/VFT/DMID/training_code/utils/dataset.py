'''                                        
Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.
                        
Licensed under the Apache License, Version 2.0 (the "License");       
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at                    
                                           
    http://www.apache.org/licenses/LICENSE-2.0
                                                      
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,    
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                         
See the License for the specific language governing permissions and
limitations under the License.
''' 

import os
import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler
from torchvision.datasets import ImageFolder
from PIL import ImageFile
from .processing import make_processing, add_processing_arguments
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as transforms

class PathNameDataset(ImageFolder):
    def __init__(self, **keys):
        super().__init__(**keys)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        img_name = path.split('/')[-1]
        video_name = ('_').join(img_name.split('_')[:-1])
        return {"img": sample, "target": target, "path": video_name}


def get_dataset(opt, dataroot):
    dset_lst = []
    # NOTE: get the classes for the current directory
    if os.path.isdir(os.path.join(dataroot, "0real")):
        classes = ['.',]
    else:
        classes = os.listdir(dataroot)

    # transform = make_processing(opt)
    transform = transforms.ToTensor()
    
    print(classes)
    for cls in classes:
        root = dataroot + "/" + cls
        if os.path.isdir(root + "/0real"):
            dset = PathNameDataset(root=root, transform=transform)
            print("#images %6d in %s" % (len(dset), root))
            dset_lst.append(dset)

    return torch.utils.data.ConcatDataset(dset_lst)


def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1.0 / torch.tensor(ratio, dtype=torch.float)
    if torch.all(w==w[0]):
        print(f"RandomSampler: # {ratio}")
        sampler = RandomSampler(dataset, replacement = False)
    else:
        w = w / torch.sum(w)
        print(f"WeightedRandomSampler: # {ratio}, Weightes {w}")
        sample_weights = w[targets]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights)
        )
    return sampler


def add_dataloader_arguments(parser):
    # parser is an argparse.ArgumentParser
    #
    # This adds the arguments necessary for dataloader
    parser.add_argument(
        "--dataroot", type=str, help="Path to the dataset to use during training"
    )
    # The path containing the train and the validation data to train on

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_threads", default=8, type=int, help='# threads for loading data')
    parser = add_processing_arguments(parser)
    return parser


def create_dataloader(opt, subdir='.', is_train=True):
    dataroot = os.path.join(opt.dataroot, subdir)
    dataset = get_dataset(opt, dataroot)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        sampler=get_bal_sampler(dataset) if is_train else None,
        num_workers=int(opt.num_threads),
    )
    return data_loader
