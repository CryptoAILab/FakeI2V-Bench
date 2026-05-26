import os
import csv
import torch

from validate import validate
from networks.resnet_LNP import resnet50
from options.test_options import TestOptions
from eval_config import *
import pandas as pd


# Running tests
opt = TestOptions().parse(print_options=False)
model_name = os.path.basename(model_path).replace('.pth', '')
rows = [["{} model testing on...".format(model_name)],
        ['testset', 'accuracy', 'avg precision', 'real acc', 'fake acc']]

print("{} model testing on...".format(model_name))
for v_id, val in enumerate(vals):
    opt.dataroot = '{}/{}'.format(dataroot, val)
    opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']
    opt.no_resize = True    # testing without resizing by default

    model = resnet50(num_classes=1)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, ap, auc, r_acc, f_acc, y_true, y_pred, IDlist = validate(model, opt)
    rows.append([val, acc, ap, r_acc, f_acc])
    print("({}) acc: {}; ap: {}; r_acc: {}; f_acc: {}".format(val, acc, ap, r_acc, f_acc))

    df_result = pd.DataFrame({
        'video_id': IDlist,
        'label': y_true, 
        'y_prob': y_pred,
        })
    result_csv = results_dir + '/{}.csv'.format(val)
    with open(result_csv, 'w') as f:
        df_result.to_csv(f, index=False)

csv_name = results_dir + '/{}.csv'.format(model_name)
with open(csv_name, 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(rows)
