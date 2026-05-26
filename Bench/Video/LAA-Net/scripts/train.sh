#! /bin/bash

CUDA_VISIBLE_DEVICES=2 python scripts/train.py --cfg configs/resnet_fpn_hm.yaml
