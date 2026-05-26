import os
import argparse
import torch
import numpy as np
import random
from tqdm import tqdm
import datetime
from sklearn.metrics import average_precision_score, roc_auc_score
from data import D3_dataset_AP
from models import D3_model

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training script with configurable parameters.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--gpu-id', type=str, default="0",
                        help='CUDA GPU device ID(s), e.g., "0" or "1,2,3" (default: "5")')
    parser.add_argument('--loss', type=str, default='l2', choices=['l2', 'cos'],
                        help='Loss function type (default: l2)')
    parser.add_argument('--encoder', type=str, default='XCLIP-16', 
                        help='Encoder model name (default: XCLIP-16)',
                        choices=['CLIP-16', 'CLIP-32', 'XCLIP-16', 'XCLIP-32', 'DINO-base', 'DINO-large', 'ResNet-18', 'VGG-16', 'EfficientNet-b4', 'MobileNet-v3'])
    parser.add_argument('--real-csv', type=str, default=None,
                        help='Path to the real data CSV file ')
    parser.add_argument('--fake-csv', type=str, default=None,
                        help='Path to the fake/synthetic data CSV file')
    args = parser.parse_args()

    seed = args.seed
    gpu_id = args.gpu_id
    loss_type = args.loss
    encoder_type = args.encoder
    real_csv = args.real_csv
    fake_csv = args.fake_csv

    # real_csv = 'datasets/csv/t1.csv'
    # fake_csv = 'datasets/csv/t2.csv' 
    
    print(f"Starting AP evaluation for {encoder_type} with {loss_type} loss")
    print(f"Real CSV: {real_csv}")
    print(f"Fake CSV: {fake_csv}")
    
    # Load Model
    model = D3_model(encoder_type=encoder_type, loss_type=loss_type).cuda()
    model.eval()
    
    # Load Dataset
    eval_dataset = D3_dataset_AP(real_csv=real_csv, fake_csv=fake_csv)
    print(f"Total samples: {len(eval_dataset)}")
    
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=1, 
        pin_memory=True,
        drop_last=False
    )
    
    # Eval
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_frames, batch_label in tqdm(eval_loader, desc="Evaluating"):
            batch_inputs = batch_frames.cuda()
            _, _, batch_dis_std = model(batch_inputs)
            # y_pred.extend(batch_dis_std.cpu().flatten().numpy())
            # y_pred.extend(torch.sigmoid(batch_dis_std).cpu().flatten().numpy())
            y_pred.extend((1 - torch.sigmoid(batch_dis_std)).cpu().flatten().numpy())

            y_true.extend(batch_label.cpu().flatten().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print(y_pred)
    ap_score = average_precision_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    result_str = (
        f"AP Evaluation Results\n"
        f"Encoder: {encoder_type}\n"
        f"Loss Type: {loss_type}\n"
        f"Real CSV: {real_csv}\n"
        f"Fake CSV: {fake_csv}\n"
        f"Total Samples: {len(y_true)}\n"
        f"AP Score: {ap_score:.4f}\n"
        f"AUC Score: {auc_score:.4f}\n"
    )
    
    print("\n" + "="*50)
    print(result_str.strip())
    print("="*50)
    
    os.makedirs("results", exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/result_{timestamp}.txt"

    with open(output_file, 'w') as f:
        f.write(result_str)

    print(f"\nResults saved to {output_file}")