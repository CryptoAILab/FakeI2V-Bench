import os
import time
from tensorboardX import SummaryWriter

from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions
from tqdm import tqdm
import pandas as pd
import torch
from models import get_model

# python train.py --name=clip_vitl14 --wang2020_data_path=datasets/ --data_mode=wang2020  --arch=CLIP:ViT-L/14  --fix_backbone
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

"""Currently assumes jpg_prob, blur_prob 0 or 1"""
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.data_label = 'Crafter'
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt



if __name__ == '__main__':
    
    opt = TrainOptions().parse()
    opt.wang2020_data_path = "/data2/lp/IVBridge_datasets/crop"
    opt.data_mode = 'wang2020'
    val_opt = get_val_opt()
 
    model = Trainer(opt)

    # model = get_model(opt.arch)
    state_dict = torch.load(opt.ckpt, map_location='cpu')
    model.model.fc.load_state_dict(state_dict)

    data_loader = create_dataloader(opt)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

        
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    start_time = time.time()
    print ("Length of data loader: %d" %(len(data_loader)))
    for epoch in range(opt.niter):
        
        for i, data in enumerate(tqdm(data_loader, disable=False)):
            model.total_steps += 1

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', model.loss, model.total_steps)
                print("Iter time: ", ((time.time()-start_time)/model.total_steps)  )

            if model.total_steps in [10,30,50,100,1000,5000,10000] and False: # save models at these iters 
                model.save_networks('model_iters_%s.pth' % model.total_steps)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d' % (epoch))
            model.save_networks( 'model_epoch_best.pth' )
            model.save_networks( 'model_epoch_%s.pth' % epoch )

        # # Validation
        # model.eval()
        # vals = ['emb4', 'emb8', 'emb16', 'emb32','imprint0', 'imprint30', 'imprint70', 'imprint90', 'imprint120', 'imprint150']
        # for val in vals:
        #     val_opt.data_label = val
        #     val_loader = create_dataloader(val_opt)
        #     acc, ap, auc, r_acc, f_acc, y_true, y_pred, IDlist= validate(model.model, val_loader, val)
        #     print("({} @ epoch {}) acc: {}; ap: {}; r_acc: {}; f_acc: {}".format(val, epoch, acc, ap, r_acc, f_acc))
        #     print("({} @ epoch {}) auc: {}".format(val, epoch, auc))
        #     val_writer.add_scalar('accuracy', acc, model.total_steps)
        #     val_writer.add_scalar('ap', ap, model.total_steps)
        #     print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))
        #     df_result = pd.DataFrame({
        #         'video_id': IDlist,
        #         'label': y_true, 
        #         'y_prob': y_pred,
        #         })

        #     results_dir = os.path.join('/data/lp/UniversalFakeDetect/result/707','epoch{}'.format(epoch))
        #     mkdir(results_dir)
        #     df_result.to_csv(os.path.join(results_dir,'{}.csv'.format(val)), index=False)

        #     with open(results_dir + '/{}.txt'.format(val), 'w') as f:
        #         f.write("(Val @ epoch {}) acc: {}; ap: {}\n".format(epoch, acc, ap))
        #         f.write("(Val @ epoch {}) r_acc: {}; f_acc: {}\n".format(epoch, r_acc, f_acc))
        #         f.write("(Val @ epoch {}) auc: {}\n".format(epoch, auc))
        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
            else:
                print("Early stopping.")
                break
        model.train()


