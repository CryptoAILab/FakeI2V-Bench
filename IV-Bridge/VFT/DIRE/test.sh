#!/bin/bash
### make sure that you have modified the EXP_NAME, CKPT, DATASETS_TEST
# eval "$(conda shell.bash hook)"
# conda activate dire

EXP_NAME="lsun_adm_release"
CKPT="/data/lp/IVBridge-version-usenix/IVBridge-main/DIRE/data/exp/lsun_adm/ckpt/model_epoch_best.pth"

python test.py --gpus 0 --ckpt $CKPT --exp_name $EXP_NAME 