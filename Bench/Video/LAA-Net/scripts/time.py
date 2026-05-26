import os
import sys
import time
import argparse
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())
import math
import random
from models import *
import torch
import torch.nn as nn
from thop import profile
from configs.get_config import load_config
from logs.logger import Logger, LOG_DIR

def parse_args(args=None):
    arg_parser = argparse.ArgumentParser('Processing testing...')
    arg_parser.add_argument('--cfg', '-c', help='Config file', required=True)
    arg_parser.add_argument('--image', '-i', type=str, help='Image for the single testing mode!')
    args = arg_parser.parse_args(args)
    
    return args

if sys.argv[1:] is not None:
    args = sys.argv[1:]
else:
    args = sys.argv[:-1]
args = parse_args(args)

# Loading config file
cfg = load_config(args.cfg)
logger = Logger(task='testing')

# build and load/initiate pretrained model
model = build_model(cfg.MODEL, MODELS)
model = load_pretrained(model, cfg.TEST.pretrained)
core_model = model.cuda()

dummy_input = torch.randn(1, 3, 224, 224).cuda()  # 修改为适合你模型的输入尺寸



# -------- Inference Time --------
with torch.no_grad():
    # Warm-up (避免显卡首次运行的延迟影响)
    for _ in range(10):
        _ = core_model(dummy_input)

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(100):
        _ = core_model(dummy_input)
    torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / 10 * 1000  # 单次推理时间，毫秒
    print(f"Avg Inference Time: {avg_time:.2f} ms")

flops, params = profile(core_model, inputs=(dummy_input,), verbose=False)
print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
print(f"Params: {params / 1e6:.2f} M")