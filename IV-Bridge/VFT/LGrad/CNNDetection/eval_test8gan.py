import sys
import time
import os
import csv
import torch
from util import Logger
from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
import networks.resnet as resnet
import numpy as np
import pandas as pd

results_dir = './results/'
# CUDA_VISIBLE_DEVICES=0 python eval_test8gan.py --dataroot  {Test-dir} --model_path {Model-Path}

vals = ['emb4', 'emb8', 'emb16', 'emb32', 'imprint0','imprint30','imprint60','imprint90','imprint120','imprint150']
#['CDFV2', 'DFD', 'GenVideo', 'GVB']
multiclass = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

opt = TestOptions().parse(print_options=False)
model_name = os.path.basename(opt.model_path).replace('.pth', '')

dataroot = opt.dataroot
print(f'Dataroot {opt.dataroot}')
print(f'Model_path {opt.model_path}')

# get model
model = resnet50(num_classes=1)
model.load_state_dict(torch.load(opt.model_path, map_location='cpu'))
model.cuda()
model.eval()

rows = [["{} model testing on...".format(model_name)],
        ['testset', 'accuracy', 'avg precision', 'real acc', 'fake acc']]
accs = [];aps = []
print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
for v_id, val in enumerate(vals):
    opt.dataroot = '{}/{}'.format(dataroot, val)
    opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']
    opt.no_resize = False    # testing without resizing by default
    opt.no_crop = True    # testing without cropping by default
    auc, acc, ap, r_acc, f_acc, y_true, y_pred, IDlist = validate(model, opt)
    accs.append(acc);aps.append(ap)
    print("({} {:10}) acc: {:.2f}; ap: {:.2f}".format(v_id, val, acc*100, ap*100))
    print("({} {:10}) r_acc: {:.2f}; f_acc: {:.2f}".format(v_id, val, r_acc*100, f_acc*100))
    rows.append([val, acc, ap, r_acc, f_acc])

    df_result = pd.DataFrame({
        'video_id': IDlist,
        'y_prob': y_pred,
        'label': y_true, 
        })
    result_csv = results_dir + '/{}.csv'.format(val)
    with open(result_csv, 'w') as f:
        df_result.to_csv(f, index=False)



print("({} {:10}) acc: {:.2f}; ap: {:.2f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) 
csv_name = results_dir + '/{}.csv'.format(model_name)
with open(csv_name, 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(rows)
