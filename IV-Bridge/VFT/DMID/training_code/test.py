import os
import tqdm
from utils import TrainingModel, create_dataloader
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
try:
    from tensorboardX import SummaryWriter
except:
    from torch.utils.tensorboard import SummaryWriter
import argparse
from utils.training import add_training_arguments
from utils.dataset import add_dataloader_arguments
import numpy as np
import torch
import pandas as pd

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser = add_training_arguments(parser)
    parser = add_dataloader_arguments(parser)

    opt = parser.parse_args()
    vals = ['split']
    for val in vals:
        valid_data_loader = create_dataloader(opt, subdir="test/{}".format(val), is_train=False)
        
        print()
        print("# validation batches = %d" % len(valid_data_loader))
        model = TrainingModel(opt, subdir=opt.name)
        dat = torch.load('checkpoints/Name_of_the_folder_to_save/model_epoch_best.pth', map_location='cpu')
        state_dict = dat['model']

        # 给所有 key 加上 "model." 前缀
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict["model." + k] = v

        model.load_state_dict(new_state_dict, strict=False)

        writer = SummaryWriter(os.path.join(model.save_dir, "logs"))
        print()


        valid_data_loader = create_dataloader(opt, subdir="test/{}".format(val), is_train=False)
        y_true, y_pred, IDlist = model.predict(valid_data_loader)
        acc = balanced_accuracy_score(y_true, y_pred > 0.0)
        auc = roc_auc_score(y_true, y_pred)
        lr = model.get_learning_rate()
        writer.add_scalar("lr", lr, model.total_steps)
        writer.add_scalar("valid/accuracy", acc, model.total_steps)
        writer.add_scalar("valid/auc", auc, model.total_steps)
        print("val acc = {}; val auc = {}".format(acc, auc), flush=True)

        df_result = pd.DataFrame({
            'video_id': IDlist,
            'label': y_true, 
            'y_prob': y_pred,
            })
        df_result.to_csv(f'{val}.csv', index=False)