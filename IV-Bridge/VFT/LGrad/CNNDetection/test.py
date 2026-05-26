import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
from tensorboardX import SummaryWriter
import numpy as np
from validate import validate
from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger
from tqdm import tqdm
import pandas as pd
from networks.resnet import resnet50
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

# test config
# vals = ['emb4', 'emb8', 'emb16', 'emb32','imprint0', 'imprint30', 'imprint70', 'imprint90', 'imprint120', 'imprint150']
# multiclass = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

vals = ['noflip']
multiclass = [0]
# vals = ['imprint0','cnn_target','code_target','defake_target','dmid_target','drct_target','gram_target','npr_target','patch_target','rine_target','unifd_target',
#         'cnn_untarget','code_untarget','defake_untarget','dmid_untarget','drct_untarget','gram_untarget','npr_untarget','patch_untarget','rine_untarget','unifd_untarget']
# multiclass = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#               0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True

    return val_opt


if __name__ == '__main__':
    opt = TrainOptions().parse()
    Testdataroot = os.path.join(opt.dataroot, 'val')
    # Testdataroot = opt.dataroot
    print(Testdataroot)
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))
    print('  '.join(list(sys.argv)) )
    val_opt = get_val_opt()
    Testopt = TestOptions().parse(print_options=False)
    data_loader = create_dataloader(opt)
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    result_root ='results'

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    
    model = resnet50(num_classes=1)
    state_dict = torch.load('checkpoints/experiment_name2025_08_15_17_38_03/model_epoch_16.pth', map_location='cpu')
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()
    
    def testmodel(epoch):
        print('*'*25);accs = [];aps = []
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        for v_id, val in enumerate(tqdm(vals)):
            Testopt.dataroot = '{}/{}'.format(Testdataroot, val)
            Testopt.classes = os.listdir(Testopt.dataroot) if multiclass[v_id] else ['']
            Testopt.no_resize = True    # testing without resizing by default
            Testopt.no_crop = True    # testing without cropping by default
            auc, acc, ap, r_acc, f_acc, y_true, y_pred, IDlist = validate(model, Testopt)
            accs.append(acc);aps.append(ap)
            print("({} {:10}) auc: {:.1f}; acc: {:.1f}; ap: {:.1f}; r_acc: {:.1f}; f_acc: {:.1f}".format(v_id, val, auc*100, acc*100, ap*100, r_acc*100, f_acc*100))
            df_result = pd.DataFrame({
                "video_id": IDlist,
                "label": y_true,
                "y_prob": y_pred
            })
            result_dir = os.path.join(result_root, 'epoch_{}'.format(epoch))
            mkdir(result_dir)
            df_result.to_csv(os.path.join(result_dir, '{}.csv'.format(val)), index=False)

            with open(result_dir + '/{}.txt'.format(val), 'w') as f:
                f.write("(Val @ epoch {}) acc: {}; ap: {}\n".format(epoch, acc, ap))
                f.write("(Val @ epoch {}) r_acc: {}; f_acc: {}\n".format(epoch, r_acc, f_acc))
                f.write("(Val @ epoch {}) auc: {}\n".format(epoch, auc))
        print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) 
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))


    model.eval();testmodel(epoch = 'split')
    