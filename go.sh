#!/bin/bash

# ========================================
# NRFormer+ TKDE Experiments
# ========================================

MODEL="NRFormer_Plus"
EPOCHS=300
DES="optimized_v1"

# --- 1D-data (Daily resolution) ---
echo "========== Training on 1D-data =========="
python train.py --model_name $MODEL --dataset 1D-data --epochs $EPOCHS --model_des $DES \
    --IsDayOfYearEmbedding True --num_temporal_att_layer 3 --num_spatial_att_layer 2

# echo "========== Testing on 1D-data =========="
# python test.py --model_name $MODEL --dataset 1D-data --model_des $DES

# --- 4H-data (4-Hour resolution) ---
echo "========== Training on 4H-data =========="
python train.py --model_name $MODEL --dataset 4H-data --epochs $EPOCHS --model_des $DES \
    --IsDayOfYearEmbedding True --num_temporal_att_layer 3 --num_spatial_att_layer 2

# echo "========== Testing on 4H-data =========="
# python test.py --model_name $MODEL --dataset 4H-data --model_des $DES

echo "========== All experiments done =========="
