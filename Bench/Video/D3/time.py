import os
import argparse
import torch
import numpy as np
import random
from tqdm import tqdm
import datetime
import time
from sklearn.metrics import average_precision_score, roc_auc_score

from data import D3_dataset_AP
from models import D3_model


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AP / AUC Evaluation with Deployment Cost')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu-id', type=str, default="0")
    parser.add_argument('--loss', type=str, default='l2', choices=['l2', 'cos'])
    parser.add_argument(
        '--encoder',
        type=str,
        default='XCLIP-16',
        choices=[
            'CLIP-16', 'CLIP-32', 'XCLIP-16', 'XCLIP-32',
            'DINO-base', 'DINO-large',
            'ResNet-18', 'VGG-16',
            'EfficientNet-b4', 'MobileNet-v3'
        ]
    )
    parser.add_argument('--real-csv', type=str, required=True)
    parser.add_argument('--fake-csv', type=str, required=True)

    args = parser.parse_args()

    seed_everything(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    print(f"Encoder: {args.encoder}, Loss: {args.loss}")
    print(f"Real CSV: {args.real_csv}")
    print(f"Fake CSV: {args.fake_csv}")

    # =========================
    # Load Model
    # =========================
    model = D3_model(
        encoder_type=args.encoder,
        loss_type=args.loss
    ).cuda()
    model.eval()

    num_params = count_parameters(model)
    print(f"#Parameters: {num_params / 1e6:.2f} M")

    # =========================
    # Load Dataset
    # =========================
    eval_dataset = D3_dataset_AP(
        real_csv=args.real_csv,
        fake_csv=args.fake_csv
    )

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False
    )

    print(f"Total samples: {len(eval_dataset)}")

    # =========================
    # Evaluation + Timing
    # =========================
    y_true, y_pred = [], []
    total_time = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch_frames, batch_label in tqdm(eval_loader, desc="Evaluating"):
            batch_frames = batch_frames.cuda(non_blocking=True)

            torch.cuda.synchronize()
            start_time = time.time()

            _, _, batch_dis_std = model(batch_frames)

            torch.cuda.synchronize()
            elapsed = time.time() - start_time

            total_time += elapsed
            num_samples += batch_frames.size(0)

            y_pred.extend(batch_dis_std.cpu().numpy().flatten())
            y_true.extend(batch_label.cpu().numpy().flatten())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # =========================
    # Metrics
    # =========================
    ap_score = average_precision_score(1 - y_true, y_pred)
    auc_score = roc_auc_score(1 - y_true, y_pred)
    print(num_samples)
    avg_time_ms = (total_time / num_samples) * 1000

    # =========================
    # Print & Save
    # =========================
    result_str = (
        f"Deployment Cost Evaluation\n"
        f"===========================\n"
        f"Encoder: {args.encoder}\n"
        f"Loss Type: {args.loss}\n"
        f"Real CSV: {args.real_csv}\n"
        f"Fake CSV: {args.fake_csv}\n"
        f"---------------------------\n"
        f"#Param (M): {num_params / 1e6:.2f}\n"
        f"Inference Time (ms/sample): {avg_time_ms:.2f}\n"
        f"AUC: {auc_score:.4f}\n"
        f"AP: {ap_score:.4f}\n"
    )

    print("\n" + result_str)

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/{args.encoder}_{args.loss}_{timestamp}.txt"

    with open(output_file, 'w') as f:
        f.write(result_str)

    print(f"Results saved to {output_file}")
