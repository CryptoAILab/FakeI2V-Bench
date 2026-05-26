#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

# ========== Config ==========
MODEL_PATH="invid_mae_i2v_i2v_best_model.pth"
FAKE_PATH="/data/lp/IVBridgeDataset_Videos_Split/GenVideo/SourceVideos/Show_1"
REAL_PATH="/data/lp/IVBridgeDataset_Videos_Split/GenVideo/MSRVTT"
LABEL_NUM=2

# ========== Run ==========
python mae.py \
  --train False \
  --task "detection" \
  --load_pre_trained_model_state "$MODEL_PATH" \
  --fake_videos_path "$FAKE_PATH" \
  --real_videos_path "$REAL_PATH" \
  --label_number $LABEL_NUM
