#!/bin/bash
### make sure that you have modified the EXP_NAME, CKPT, DATASETS_TEST
# eval "$(conda shell.bash hook)"
# conda activate dire

EXP_NAME="lsun_adm_release"
CKPT="/data/lp/IVBridge-OriCode/DIRE/DIRE/checkpoints/lsun_adm.pth"
DATASETS_TEST="progan,stylegan,biggan,cyclegan,stargan,gaugan,crn,imle,seeingdark,san,deepfake,stylegan2,diffusion_datasets/dalle,diffusion_datasets/glide_50_27,diffusion_datasets/glide_100_10,diffusion_datasets/glide_100_27,diffusion_datasets/guided,diffusion_datasets/ldm_100,diffusion_datasets/ldm_200,diffusion_datasets/ldm_200_cfg"
python test.py --gpus 0 --ckpt $CKPT --exp_name $EXP_NAME datasets_test $DATASETS_TEST