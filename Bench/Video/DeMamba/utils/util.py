import models
from models.VideoMAE import Videomae_Net
from models.FTCN import ViT_B_FTCN
import time
import torch
import math
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, average_precision_score, roc_auc_score
from dataloader import generate_dataset_loader
import os

def build_model(model_name):
    if model_name == 'F3Net':
        model = models.Det_F3_Net()
    if model_name == 'NPR':
        model = models.resnet50_npr()
    if model_name == 'STIL':
        model = models.Det_STIL()
    if model_name == 'XCLIP_DeMamba':
        model = models.XCLIP_DeMamba()
    if model_name == 'CLIP_DeMamba':
        model = models.CLIP_DeMamba()
    if model_name == 'XCLIP':
        model = models.XCLIP()
    if model_name == 'CLIP':
        model = models.CLIP_Base()
    if model_name == 'ViT_B_MINTIME':
        model = models.ViT_B_MINTIME()
    if model_name == 'VideoMAE':
        model = Videomae_Net()
    if model_name == 'FTCN':
        model = models.ViT_B_FTCN()
    return model

def eval_model(cfg, model, device, val_loader, criterion, constrastive_loss, val_batch_size, alpha = 0.3):
    model.eval()
    outpred_list = []
    gt_label_list = []
    video_list = []
    valLoss = 0
    lossTrainNorm = 0
    print("******** Start Testing. ********")
    features = []
    label = []

    with torch.no_grad():  # No need to track gradients during validation
        for i, (_, input, target, binary_label, video_id) in enumerate(tqdm(val_loader, desc="Validation", total=len(val_loader))):
            if i == 0:
                ss_time = time.time()
            
            input = input[:,0]
            varInput = torch.autograd.Variable(input.float().to(device))
            varTarget = torch.autograd.Variable(target.contiguous().to(device))
            var_Binary_Target = torch.autograd.Variable(binary_label.contiguous().to(device))

            # logit = model(varInput)
            logit= model(varInput)
            lossvalue = criterion(logit, var_Binary_Target)


            # lossvalue = (1 - alpha) * criterion(logit, var_Binary_Target) + alpha * constrastive_loss(embeddings, var_Binary_Target.squeeze(-1))
            
            # features.append(embeddings.cpu().numpy())
            label.append(var_Binary_Target.cpu().detach().numpy())

            valLoss += lossvalue.item()
            lossTrainNorm += 1
            outpred_list.append(logit[:,0].sigmoid().cpu().detach().numpy())
            gt_label_list.append(varTarget.cpu().detach().numpy())
            video_list.append(video_id)

    # features = np.concatenate(features, 0)
    label = np.concatenate(label, 0)

    valLoss = valLoss / lossTrainNorm

    outpred = np.concatenate(outpred_list, 0)
    gt_label = np.concatenate(gt_label_list, 0)
    video_list = np.concatenate(video_list, 0)
    pred_labels = [1 if item > 0.5 else 0 for item in outpred]
    true_labels = np.argmax(gt_label, axis=1)

    pred_accuracy = accuracy_score(true_labels, pred_labels)

    return pred_accuracy, video_list, pred_labels, true_labels, outpred, features, label

def train_one_epoch(cfg, model, device, criterion, constrastive_loss, scheduler, optimizer, epochID, max_epoch, max_acc, train_loader, val_loader, snapshot_path, alpha = 0.3):
    model.train()
    trainLoss = 0
    lossTrainNorm = 0

    scheduler.step()
    pbar = tqdm(total=cfg['bath_per_epoch'])

    for batchID, (index, input, target, binary_label) in enumerate(train_loader):
        if batchID > cfg['bath_per_epoch']:
            break
        if batchID == 0:
            ss_time = time.time()
        input = input[:,0].float()
        varInput = torch.autograd.Variable(input).to(device)
        varTarget = torch.autograd.Variable(target.contiguous().to(device))
        var_Binary_Target = torch.autograd.Variable(binary_label.contiguous().to(device))
        optimizer.zero_grad()
        # print("test")
        # print(varInput.shape)
        # print(var_Binary_Target.shape)

        # logit = model(varInput)
        # # print(logit.shape)


        logit= model(varInput)
        lossvalue = criterion(logit, var_Binary_Target)        

        # lossvalue = (1 - alpha) * criterion(logit, var_Binary_Target) + alpha * constrastive_loss(embeddings, var_Binary_Target.squeeze(-1))
        # print(lossvalue)

        lossvalue.backward()
        optimizer.step()

        trainLoss += lossvalue.item()
        lossTrainNorm += 1
        pbar.set_postfix(loss=trainLoss / lossTrainNorm)
        pbar.update(1)
        del lossvalue


    trainLoss = trainLoss / lossTrainNorm
    
    if (epochID+1) % 10 == 0:

        val_list = ['/data2/lp/FacialDatasets/SourceFrames/CelebDFV2.csv',
                     '/data2/lp/FacialDatasets/SourceFrames/DFD.csv',
                     '/data2/lp/FacialDatasets/SourceFrames/FaceShifter.csv']
        # val_list = ['/data2/lp/GenVidBench/SourceFrames/GenVidBench.csv']

        for val_video in val_list:
            video_name = (val_video.split('/')[-1]).split('.')[0]
            _, val_loader = generate_dataset_loader(cfg, val_data_path = val_video)
            pred_accuracy, video_id, pred_labels, true_labels, outpred, features, label = eval_model(cfg, model, device,  val_loader, criterion, constrastive_loss, cfg['val_batch_size'])    

            snapshot_path = snapshot_path + video_name
            if not os.path.exists(snapshot_path):
                os.makedirs(snapshot_path)
            torch.save(
                {"epoch": epochID + 1, "model_state_dict": model.state_dict()},
                snapshot_path + "/last"+ ".pth",
                )

            if pred_accuracy > max_acc:
                max_epoch, max_acc = epochID, pred_accuracy
                torch.save(
                {"epoch": epochID + 1, "model_state_dict": model.state_dict()},
                snapshot_path + "/best_acc"+ ".pth",
                )

            df_result = pd.DataFrame({
                'data_path': video_id,
                'predicted_label': pred_labels,
                'actual_label': true_labels,
                'predicted_prob':outpred
            })
            df_result.to_csv(snapshot_path+'/Epoch_'+str(epochID)+'_result.csv', index=False)

            temp_result_txt = snapshot_path+'/Epoch_'+str(epochID)+'_accuracy.txt'
            with open(temp_result_txt, 'w') as file:
                true_labels = df_result['actual_label']
                pred_probs = df_result['predicted_prob'] 

                auc = roc_auc_score(true_labels, pred_probs)
                ap = average_precision_score(true_labels, pred_probs)
                file.write(f"总正确率: {pred_accuracy:.2%}\n")
                file.write(f"AUC是: {auc:.2%}\n")
                file.write(f"AP是: {ap:.2%}\n")

            # # prefixes = ["fake/ModelScope","fake/MorphStudio","fake/MoonValley",
            # #             "fake/HotShot","fake/Show_1","fake/Gen2", "fake/Crafter", 
            # #             "fake/Lavie", "fake/Sora", "fake/WildScrape"     
            # #             ]
            # # video_nums = [7000, 7000, 6260, 7000, 7000, 13800, 14000, 14000, 560, 9260]

            # prefixes = ['fake/cogvideo', 'fake/mora', 'fake/musev', 'fake/svd']
            # # video_nums = [13800, 13800, 13800, 13800]
            



            # # real 
            # real_condition = df_result['data_path'].apply(lambda x: x.startswith("real"))
            # temp_df_val_real = df_result[real_condition]

            # for index, temp_prefixes in enumerate(prefixes):
            #     condition = df_result['data_path'].apply(lambda x: x.startswith(temp_prefixes))
            #     temp_df_val = df_result[condition]
            #     print(temp_df_val.shape[0])



            #     # balance_temp_df_val_real = temp_df_val_real.sample(n=int(video_nums[index]), random_state=42)
            #     balance_temp_df_val_real = temp_df_val_real
            #     temp_df_val = pd.concat([temp_df_val, balance_temp_df_val_real])
            #     print(temp_df_val.shape[0])
            #     temp_df_val['correct'] = temp_df_val['predicted_label'] == temp_df_val['actual_label']
            #     accuracy = temp_df_val['correct'].mean()

            #     true_labels = temp_df_val['actual_label']
            #     pred_probs = temp_df_val['predicted_prob']  # 假设这是模型预测的概率
            #     auc = roc_auc_score(true_labels, pred_probs)
            #     ap = average_precision_score(true_labels, pred_probs)


            #     with open(temp_result_txt, 'a') as file:
            #         name = temp_prefixes.split('/')[-1]
            #         file.write(f"文件名: {name}, ACC是: {accuracy}\n")
            #         file.write(f"文件名: {name}, AP是: {ap}\n")
            #         file.write(f"文件名: {name}, AUC是: {auc}\n")


            print("*****Average Training loss",str(trainLoss),"*****\n")
            print("*****Epoch", str(epochID), "*****Acc ", str(pred_accuracy), '*****',
                '\n', "*****Max acc epoch", str(max_epoch), "*****Acc ", str(max_acc), '*****\n')
    end_time = time.time()

    return max_epoch, max_acc, end_time - ss_time
