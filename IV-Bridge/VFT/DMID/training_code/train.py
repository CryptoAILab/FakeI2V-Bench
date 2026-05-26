'''                                                     
Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.            
                                                        
Licensed under the Apache License, Version 2.0 (the "License");       
you may not use this file except in compliance with the License.
You may obtain a copy of the License at    
                                           
    http://www.apache.org/licenses/LICENSE-2.0
                        
Unless required by applicable law or agreed to in writing, software   
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implie
d.                                         
See the License for the specific language governing permissions and
limitations under the License.                        
'''

import os
import tqdm
from utils import TrainingModel, create_dataloader, EarlyStopping
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
    parser.add_argument(
        "--num_epoches", type=int, default=100, help="# of epoches at starting learning rate"
    )
    parser.add_argument(
        "--save_freq_epoch", type=int, default=5, help="# of epoches at validation"
    )    
    parser.add_argument(
        "--earlystop_epoch",
        type=int,
        default=5,
        help="Number of epochs without loss reduction before lowering the learning rate",
    )

    opt = parser.parse_args()
    train_data_loader = create_dataloader(opt, subdir="train0", is_train=True)
    val = 'DFD'
    valid_data_loader = create_dataloader(opt, subdir="val/{}".format(val), is_train=False)
    
    print()
    print("# validation batches = %d" % len(valid_data_loader))
    print("#   training batches = %d" % len(train_data_loader))
    model = TrainingModel(opt, subdir=opt.name)
    dat = torch.load('/data/lp/DMimageDetection/test_code/weights/Grag2021_progan/model_epoch_best.pth', map_location='cpu')
    if 'model' in dat:
        if ('module._conv_stem.weight' in dat['model']) or ('module.fc.fc1.weight' in dat['model']) or ('module.fc.weight' in dat['model']):
            model.load_state_dict(
                {key[7:]: dat['model'][key] for key in dat['model']})
        else:
            model.model.load_state_dict(dat['model'])
    elif 'state_dict' in dat:
        model.load_state_dict(dat['state_dict'])
    elif 'net' in dat:
        model.load_state_dict(dat['net'])
    elif 'main.0.weight' in dat:
        model.load_state_dict(dat)
    elif '_fc.weight' in dat:
        model.load_state_dict(dat)
    elif 'conv1.weight' in dat:
        model.load_state_dict(dat)
    else:
        print(list(dat.keys()))
        assert False
    writer = SummaryWriter(os.path.join(model.save_dir, "logs"))
    writer_loss_steps = len(train_data_loader) // 32
    early_stopping = None
    start_epoch = model.total_steps // len(train_data_loader)
    print()

    result_root = '/data/lp/DMimageDetection/training_code/VFTresults/819'

    for epoch in range(start_epoch, opt.num_epoches+1):
        if epoch > start_epoch:
            # Training
            pbar = tqdm.tqdm(train_data_loader)
            for data in pbar:
                loss = model.train_on_batch(data).item()
                total_steps = model.total_steps
                pbar.set_description(f"Train loss: {loss:.4f}")
                if total_steps % writer_loss_steps == 0:
                    writer.add_scalar("train/loss", loss, total_steps)

            # Save model
            model.save_networks(epoch)

        # Validation
        if epoch % opt.save_freq_epoch == 0:
            print("Validation ...", flush=True)
            vals = ['DFD','CDFV2', 'GenVideo', 'GVB']
            for val in vals:
                valid_data_loader = create_dataloader(opt, subdir="val/{}".format(val), is_train=False)
                y_true, y_pred, IDlist = model.predict(valid_data_loader)
                acc = balanced_accuracy_score(y_true, y_pred > 0.0)
                auc = roc_auc_score(y_true, y_pred)
                lr = model.get_learning_rate()
                writer.add_scalar("lr", lr, model.total_steps)
                writer.add_scalar("valid/accuracy", acc, model.total_steps)
                writer.add_scalar("valid/auc", auc, model.total_steps)
                print("After {} epoches: val acc = {}; val auc = {}".format(epoch, acc, auc), flush=True)
                df_result = pd.DataFrame({
                    'video_id': IDlist,
                    'label': y_true, 
                    'y_prob': y_pred,
                    })
                result_dir = os.path.join(result_root, 'epoch_{}'.format(epoch))
                mkdir(result_dir)
                df_result.to_csv(os.path.join(result_dir, '{}.csv'.format(val)), index=False)

                with open(os.path.join(result_dir, '{}.txt'.format(val)), 'w') as f:
                    f.write('Acc: {}\n'.format(acc))
                    f.write('AUC: {}\n'.format(auc))
        # Early Stopping
        if early_stopping is None:
            early_stopping = EarlyStopping(
                init_score=acc, patience=opt.earlystop_epoch,
                delta=0.001, verbose=True,
            )
        else:
            if early_stopping(acc):
                print('Save best model', flush=True)
                model.save_networks('best')
            if early_stopping.early_stop:
                cont_train = model.adjust_learning_rate()
                if cont_train:
                    print("Learning rate dropped by 10, continue training ...", flush=True)
                    early_stopping.reset_counter()
                else:
                    print("Early stopping.", flush=True)
                    break
