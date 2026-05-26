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
if args.dataset_type == 'IV-Bridge':
    dataroot = '/data2/lp/IVBridge_datasets/crop/test/'
    partition = ''
    datasets = [
        # ('CDFV2/1fake', 'CDFV2/0real', 'CDFV2'),
        # ('DFD/1fake', 'DFD/0real', 'DFD'),
        # ('GenVideo/1fake', 'GenVideo/0real', 'GenVideo'),
        # ('GVB/1fake', 'GVB/0real', 'GVB'),
        ('split/1fake', 'split/0real', 'split'),
        # ('emb4/1fake', 'emb4/0real', 'emb4'),
        # ('emb8/1fake', 'emb8/0real', 'emb8'),
        # ('emb16/1fake', 'emb16/0real', 'emb16'),
        # ('emb32/1fake', 'emb32/0real', 'emb32'),
        # ('imprint0/1fake', 'imprint0/0real', 'imprint0'),
        # ('imprint30/1fake', 'imprint30/0real', 'imprint30'),
        # ('imprint70/1fake', 'imprint70/0real', 'imprint70'),
        # ('imprint90/1fake', 'imprint90/0real', 'imprint90'),
        # ('imprint120/1fake', 'imprint120/0real', 'imprint120'),
        # ('imprint150/1fake', 'imprint150/0real', 'imprint150')
        # ('cnn_target/1fake', 'cnn_target/0real', 'cnn_target'),
        # ('cnn_untarget/1fake', 'cnn_untarget/0real', 'cnn_untarget'),

        # ('code_target/1fake', 'code_target/0real', 'code_target'),
        # ('code_untarget/1fake', 'code_untarget/0real', 'code_untarget'),

        # ('defake_target/1fake', 'defake_target/0real', 'defake_target'),
        # ('defake_untarget/1fake', 'defake_untarget/0real', 'defake_untarget'),

        # ('dmid_target/1fake', 'dmid_target/0real', 'dmid_target'),
        # ('dmid_untarget/1fake', 'dmid_untarget/0real', 'dmid_untarget'),

        # ('drct_target/1fake', 'drct_target/0real', 'drct_target'),
        # ('drct_untarget/1fake', 'drct_untarget/0real', 'drct_untarget'),

        # ('gram_target/1fake', 'gram_target/0real', 'gram_target'),
        # ('gram_untarget/1fake', 'gram_untarget/0real', 'gram_untarget'),

        # ('npr_target/1fake', 'npr_target/0real', 'npr_target'),
        # ('npr_untarget/1fake', 'npr_untarget/0real', 'npr_untarget'),

        # ('patch_target/1fake', 'patch_target/0real', 'patch_target'),
        # ('patch_untarget/1fake', 'patch_untarget/0real', 'patch_untarget'),

        # ('rine_target/1fake', 'rine_target/0real', 'rine_target'),
        # ('rine_untarget/1fake', 'rine_untarget/0real', 'rine_untarget'),

        # ('unifd_target/1fake', 'unifd_target/0real', 'unifd_target'),
        # ('unifd_untarget/1fake', 'unifd_untarget/0real', 'unifd_untarget'),

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
        test_command += ' --gpu_ids 2'
        test_command += ' --real_im_path %s' % real
        test_command += ' --fake_im_path %s' % fake
        test_command += ' --partition %s' % args.partition
        test_command += ' --dataset_name %s' % name

        print("检测命令:",test_command)
        os.system(test_command)

