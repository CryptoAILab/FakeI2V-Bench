import time
import os
import sys
import subprocess
import shlex
import glob
import torch
import argparse

# wrapper to run test experiments

# argparse model checkpoint
parser = argparse.ArgumentParser('Model Test Pipeline')
parser.add_argument('checkpoint_dir', help='directory of experiment checkpoints')
parser.add_argument('dataset_type', help='gen_models, faceforensics, etc')
parser.add_argument('partition', help='which partition to run [val|test]')
args = parser.parse_args()
checkpoint_dir = args.checkpoint_dir
checkpoints = glob.glob(os.path.join(checkpoint_dir, '*_net_D.pth'))

def get_dataset_paths(dataroot, datasets, partition):
    fake_datasets = [os.path.join(dataroot, dataset[0], partition)
                     for dataset in datasets]
    real_datasets = [os.path.join(dataroot, dataset[1], partition)
                     for dataset in datasets]
    dataset_names = [dataset[2] for dataset in datasets]
    return fake_datasets, real_datasets, dataset_names

# datasets
if args.dataset_type == 'gen_models':
    dataroot = 'dataset/faces/'
    partition = args.partition
    datasets = [
        ('celebahq/pgan-pretrained-128-png',
         'celebahq/real-tfr-1024-resized128', 'celebahq-pgan-pretrained'),
        ('celebahq/sgan-pretrained-128-png',
         'celebahq/real-tfr-1024-resized128', 'celebahq-sgan-pretrained'),
        ('celebahq/glow-pretrained-128-png',
         'celebahq/real-tfr-1024-resized128', 'celebahq-glow-pretrained'),
        ('celeba/mfa-defaults', 'celeba/mfa-real', 'celeba-gmm'),
        ('ffhq/pgan-9k-128-png', 'ffhq/real-tfr-1024-resized128', 'ffhq-pgan'),
        ('ffhq/sgan-pretrained-128-png', 'ffhq/real-tfr-1024-resized128',
         'ffhq-sgan'),
        ('ffhq/sgan2-pretrained-128-png', 'ffhq/real-tfr-1024-resized128',
         'ffhq-sgan2'),
    ]
    fake_datasets, real_datasets, dataset_names = get_dataset_paths(
        dataroot, datasets, partition)
elif args.dataset_type == 'ivbridge_image':
    dataroot = '/data/lp/IVBridge-OriCode/datasets/'
    partition = args.partition
    datasets = []

    for dataset_name in sorted(os.listdir(dataroot)):
        dataset_root = os.path.join(dataroot, dataset_name)
        if not os.path.isdir(dataset_root):
            continue

        # 找所有 fake 目录（最深层）
        fake_dirs = glob.glob(os.path.join(dataset_root, '**', '1_fake'), recursive=True)

        for fdir in fake_dirs:
            parent_dir = os.path.dirname(fdir)  # e.g. progan/bedroom

            # 对应的 real 目录（可能没有）
            rdir = os.path.join(parent_dir, '0_real')
            real_rel = os.path.relpath(rdir, dataroot) if os.path.exists(rdir) else None
            fake_rel = os.path.relpath(fdir, dataroot)

            # key 需要唯一：dataset + 子目录名
            sub_name = os.path.basename(parent_dir)
            key = f"{dataset_name}_{sub_name}"

            datasets.append((fake_rel, real_rel, key))
            
    print(datasets)
    fake_datasets, real_datasets, dataset_names = get_dataset_paths(
        dataroot, datasets, '')
    
elif args.dataset_type == 'faceforensics':
    dataroot = 'dataset/faces/'
    partition = args.partition
    datasets = [
        ('faceforensics_aligned/NeuralTextures/manipulated',
         'faceforensics_aligned/NeuralTextures/original', 'NT'),
        ('faceforensics_aligned/Deepfakes/manipulated',
         'faceforensics_aligned/Deepfakes/original', 'DF'),
        ('faceforensics_aligned/Face2Face/manipulated',
         'faceforensics_aligned/Face2Face/original', 'F2F'),
        ('faceforensics_aligned/FaceSwap/manipulated',
         'faceforensics_aligned/FaceSwap/original', 'FS'),
    ]
    fake_datasets, real_datasets, dataset_names = get_dataset_paths(
        dataroot, datasets, partition)
else:
    raise NotImplementedError

# print the datasets to test on
print(real_datasets)
print(fake_datasets)
print(dataset_names)

for checkpoint in checkpoints:
    for fake, real, name in zip(fake_datasets, real_datasets, dataset_names):
        which_epoch = os.path.basename(checkpoint).split('_')[0]
        if which_epoch != 'bestval':
            # only runs using the bestval checkpoint
            continue
        test_command = ('python test.py --train_config %s' %
                        (os.path.join(checkpoint_dir, 'opt.yml')))
        test_command += ' --which_epoch %s' % which_epoch
        test_command += ' --gpu_ids 0'
        test_command += ' --real_im_path %s' % real
        test_command += ' --fake_im_path %s' % fake
        test_command += ' --partition %s' % args.partition
        test_command += ' --dataset_name %s' % name

        print(test_command)
        os.system(test_command)

