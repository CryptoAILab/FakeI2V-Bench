import os
import csv
import torch

from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
from eval_config import *


# Running tests
opt = TestOptions().parse(print_options=False)
model_name = os.path.basename(model_path).replace('.pth', '')
rows = [["{} model testing on...".format(model_name)],
        ['testset', 'accuracy','auc', 'avg precision']]

accs = []
aps = []
aucs = []

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

    acc, ap, auc, _, _, _, _ = validate(model, opt)
    rows.append([val, acc, auc, ap])
    aucs.append(auc)
    aps.append(ap)
    accs.append(acc)
    print("({}) acc: {}; ap: {}; auc: {}".format(val, acc, ap, auc))

mean_acc = sum(accs) / len(accs)
mean_ap  = sum(aps) / len(aps)
mean_auc = sum(aucs) / len(aucs)

print("\n====== Overall Average (Unweighted) ======")
print(f"Acc: {mean_acc*100:.2f}  AP: {mean_ap*100:.2f}  AUC: {mean_auc*100:.2f}")


csv_name = results_dir + '/{}.csv'.format(model_name)
with open(csv_name, 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(rows)

with open(csv_name, 'a') as f:  # 用 'a' 追加
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerow(['Average', mean_acc, mean_ap, mean_auc])
