import argparse
import warnings
import sys
import os
import pandas as pd
from thop import profile
def get_parser():
    parser = argparse.ArgumentParser(description="AIGCDetection @cby Training")
    parser.add_argument("--model_name", default='convnext_base_in22k', help="Setting the model name", type=str)
    parser.add_argument("--embedding_size", default=1024, help="Setting the embedding_size", type=int)
    parser.add_argument("--num_classes", default=2, help="Setting the num classes", type=int)
    parser.add_argument('--freeze_extractor', action='store_true', help='Whether to freeze extractor?')
    parser.add_argument("--model_path", default='/data/lp/DRCT/pretrained/DRCT-2M/sdv2/convnext_base_in22k_224_drct_amp_crop/16_acc0.9993.pth', help="Setting the model path", type=str)
    parser.add_argument('--no_strict', action='store_true', help='Whether to load model without strict?')
    parser.add_argument("--root_path", default='/disk4/chenby/dataset/MSCOCO',
                        help="Setting the root path for dataset loader", type=str)
    parser.add_argument("--fake_root_path", default='/disk4/chenby/dataset/DRCT-2M',
                        help="Setting the fake root path for dataset loader", type=str)
    parser.add_argument('--is_dire', action='store_true', help='Whether to using DIRE?')
    parser.add_argument("--regex", default='*.*', help="Setting the regex for dataset loader", type=str)
    parser.add_argument('--test_all', action='store_true', help='Whether to test_all?')
    parser.add_argument('--post_aug_mode', default=None, help='Stetting the post aug mode during test phase.')
    parser.add_argument('--save_txt', default=None, help='Stetting the save_txt path.')
    parser.add_argument("--fake_indexes", default='1',
                        help="Setting the fake indexes, multi class using '1,2,3,...' ", type=str)
    parser.add_argument("--dataset_name", default='MSCOCO', help="Setting the dataset name", type=str)
    parser.add_argument("--device_id", default='1',
                        help="Setting the GPU id, multi gpu split by ',', such as '0,1,2,3'", type=str)
    parser.add_argument("--input_size", default=224, help="Image input size", type=int)
    parser.add_argument('--is_crop', action='store_true', help='Whether to crop image?')
    parser.add_argument("--batch_size", default=64, help="Setting the batch size", type=int)
    parser.add_argument("--epoch_start", default=0, help="Setting the epoch start", type=int)
    parser.add_argument("--num_epochs", default=50, help="Setting the num epochs", type=int)
    parser.add_argument("--num_workers", default=4, help="Setting the num workers", type=int)
    parser.add_argument('--is_warmup', action='store_true', help='Whether to using lr warmup')
    parser.add_argument("--lr", default=1e-3, help="Setting the learning rate", type=float)
    parser.add_argument("--save_flag", default='', help="Setting the save flag", type=str)
    parser.add_argument("--sampler_mode", default='', help="Setting the sampler mode", type=str)
    parser.add_argument('--is_test', action='store_true', help='Whether to predict the test set?')
    parser.add_argument('--is_amp', action='store_true', help='Whether to using amp autocast(使用混合精度加速)?')
    parser.add_argument("--inpainting_dir", default='full_inpainting', help="rec_image dir", type=str)
    parser.add_argument("--threshold", default=0.5, help="Setting the valid or testing threshold.", type=float)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    return args


warnings.filterwarnings("ignore")
sys.path.append('..')
args = get_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)

import torch
import torch.nn as nn
import pytorch_warmup as warmup
from network.models import get_models
import time

if __name__ == '__main__':
    batch_size = args.batch_size * torch.cuda.device_count()
    writeFile = f"../output/{args.dataset_name}/{args.fake_indexes.replace(',', '_')}/" \
                f"{args.model_name.split('/')[-1]}_{args.input_size}{args.save_flag}/logs"
    store_name = writeFile.replace('/logs', '/weights')
    print(f'Using gpus:{args.device_id},batch size:{batch_size},gpu_count:{torch.cuda.device_count()},num_classes:{args.num_classes}')
    # Load model
    model = get_models(model_name=args.model_name, num_classes=args.num_classes,
                       freeze_extractor=args.freeze_extractor, embedding_size=args.embedding_size)
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'), strict=not args.no_strict)
        print('Model found in {}'.format(args.model_path))
    else:
        print('No model found, initializing random model.')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    params = count_parameters(model.module if hasattr(model, "module") else model)
    print(f"Total Trainable Parameters: {params / 1e6:.2f} M")

    # 构造一个 dummy 输入
    dummy_input = torch.randn(1, 3, args.input_size, args.input_size).to(next(model.parameters()).device)

    # 如果使用了 DataParallel，需要取出原始模型
    core_model = model.module if hasattr(model, "module") else model

    # 计算 FLOPs 和参数量
    flops, params = profile(core_model, inputs=(dummy_input,), verbose=False)
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    core_model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(10, 3, args.input_size, args.input_size).to(next(model.parameters()).device)

        # CUDA 预热（仅 GPU 时建议）
        for _ in range(10):
            _ = core_model(dummy_input)

        # 正式计时
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(1000):
            _ = core_model(dummy_input)
        torch.cuda.synchronize()
        end = time.time()

        avg_time = (end - start) / 1000
        print(f"Average Inference Time: {avg_time * 1000:.2f} ms per image")