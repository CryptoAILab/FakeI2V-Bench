#!/bin/bash
### make sure that you have modified the EXP_NAME, DATASETS, DATASETS_TEST
# eval "$(conda shell.bash hook)"
#conda activate dire
EXP_NAME="lsun_adm"
DATASETS="/data2/lp/IVBridge_datasets/DIRE/dire"
DATASETS_TEST="/data2/lp/IVBridge_datasets/DIRE/dire/"
python train.py --gpus 1 --exp_name $EXP_NAME datasets $DATASETS datasets_test $DATASETS_TEST
