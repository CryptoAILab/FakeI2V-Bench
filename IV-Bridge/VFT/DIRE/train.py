from utils.config import cfg  # isort: split

import os
import time

from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils.datasets import create_dataloader
from utils.earlystop import EarlyStopping
from utils.eval import get_val_cfg, validate
from utils.trainer import Trainer
from utils.utils import Logger
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

    val_cfg = get_val_cfg(cfg, split="val", copy=True)
    dataset_root = cfg.dataset_root
    cfg.dataset_root = os.path.join(cfg.dataset_root, "train0")
    print(cfg.dataset_root)
    data_loader = create_dataloader(cfg)
    dataset_size = len(data_loader)

    result_root = 'results'
    log = Logger()
    log.open(cfg.logs_path, mode="a")
    log.write("Num of training images = %d\n" % (dataset_size * cfg.batch_size))
    log.write("Config:\n" + str(cfg.to_dict()) + "\n")

    train_writer = SummaryWriter(os.path.join(cfg.exp_dir, "train0"))
    val_writer = SummaryWriter(os.path.join(cfg.exp_dir, "val"))

    trainer = Trainer(cfg)
    early_stopping = EarlyStopping(patience=cfg.earlystop_epoch, delta=-0.001, verbose=True)
    for epoch in range(cfg.nepoch):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for data in tqdm(data_loader, dynamic_ncols=True):
            trainer.total_steps += 1
            epoch_iter += cfg.batch_size

            trainer.set_input(data)
            trainer.optimize_parameters()

            if trainer.total_steps % cfg.loss_freq == 0:
                log.write(f"Train loss: {trainer.loss} at step: {trainer.total_steps}\n")
            train_writer.add_scalar("loss", trainer.loss, trainer.total_steps)

            if trainer.total_steps % cfg.save_latest_freq == 0:
                log.write(
                    "saving the latest model %s (epoch %d, model.total_steps %d)\n"
                    % (cfg.exp_name, epoch, trainer.total_steps)
                )
                trainer.save_networks("latest")

        if epoch % cfg.save_epoch_freq == 0:
            log.write("saving the model at the end of epoch %d, iters %d\n" % (epoch, trainer.total_steps))
            trainer.save_networks("latest")
            trainer.save_networks(epoch)

        # Validation
        trainer.eval()
        vals = ['split']
        for val in vals:
            test_cfg = get_val_cfg(val_cfg, split=val, copy=True)
            val_results, y_true, y_pred, IDlist = validate(trainer.model, test_cfg)
            val_writer.add_scalar("AP", val_results["AP"], trainer.total_steps)
            val_writer.add_scalar("ACC", val_results["ACC"], trainer.total_steps)
            log.write(f"AUC: {val_results['AUC']}; AP: {val_results['AP']}; ACC: {val_results['ACC']}; R_ACC: {val_results['R_ACC']}; F_ACC: {val_results['F_ACC']}\n")

            df_result = pd.DataFrame({
            "video_id": IDlist,
            "label": y_true,
            "y_prob": y_pred
            })

            result_dir = os.path.join(result_root, 'epoch_{}'.format(epoch))
            mkdir(result_dir)
            df_result.to_csv(os.path.join(result_dir, '{}.csv'.format(val)), index=False)

            with open(result_dir + '/{}.txt'.format(val), 'w') as f:
                f.write("(Val @ epoch {}) acc: {}; ap: {}\n".format(epoch, val_results['ACC'], val_results['AP']))
                f.write("(Val @ epoch {}) r_acc: {}; f_acc: {}\n".format(epoch, val_results['R_ACC'], val_results['F_ACC']))
                f.write("(Val @ epoch {}) auc: {}\n".format(epoch, val_results['AUC']))

        if cfg.earlystop:
            early_stopping(val_results["ACC"], trainer)
            if early_stopping.early_stop:
                if trainer.adjust_learning_rate():
                    log.write("Learning rate dropped by 10, continue training...\n")
                    early_stopping = EarlyStopping(patience=cfg.earlystop_epoch, delta=-0.002, verbose=True)
                else:
                    log.write("Early stopping.\n")
                    break
        if cfg.warmup:
            # print(trainer.scheduler.get_lr()[0])
            trainer.scheduler.step()
        trainer.train()


