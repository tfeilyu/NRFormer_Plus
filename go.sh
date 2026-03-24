#!/bin/bash
# ============================================================
# NRFormer+ TKDE Iterative Experiment Runner
#
# Usage:
#   bash go.sh                    # Run current experiment
#   bash go.sh --phase 1          # Run Phase 1 experiments
#   bash go.sh --phase 2          # Run Phase 2 experiments
#   python compare_results.py     # Compare all experiment results
# ============================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3
MODEL="NRFormer_Plus"
DATASET="1D-data"       # Start with 1D-data (faster), then 4H-data
EPOCHS=200

# Auto compare + git push after experiments
finish() {
    echo ""
    echo "===== Comparing all experiment results ====="
    python compare_results.py
    echo ""
    echo "===== Pushing results to GitHub ====="
    git add logs/ EXPERIMENT_LOG.md 2>/dev/null
    git commit -m "Auto: experiment results $(date '+%Y%m%d-%H%M%S')" 2>/dev/null
    git push 2>/dev/null && echo "Push OK" || echo "Push failed (check git auth)"
    echo "===== All done ====="
}

# ============================================================
# Phase 1: Baseline + Architecture Fixes (已修复FFN/PE/dropout)
# ============================================================
if [ "$1" == "--phase" ] && [ "$2" == "1" ]; then

echo "===== Phase 1: Baseline Calibration ====="

# Exp 1.0: Baseline (current defaults after all fixes)
python train.py --model_name $MODEL --dataset $DATASET --epochs $EPOCHS \
    --model_des p1_baseline \
    --hidden_channels 32 --num_temporal_att_layer 3 --num_spatial_att_layer 2 \
    --IsDayOfYearEmbedding True --temporal_dropout 0.1 --ffn_ratio 4 --spatial_heads 4

# Exp 1.1: hidden=64
python train.py --model_name $MODEL --dataset $DATASET --epochs $EPOCHS \
    --model_des p1_h64 \
    --hidden_channels 64 --num_temporal_att_layer 3 --num_spatial_att_layer 2 \
    --IsDayOfYearEmbedding True --temporal_dropout 0.1 --ffn_ratio 4 --spatial_heads 4

# Exp 1.2: hidden=96
python train.py --model_name $MODEL --dataset $DATASET --epochs $EPOCHS \
    --model_des p1_h96 \
    --hidden_channels 96 --num_temporal_att_layer 3 --num_spatial_att_layer 2 \
    --IsDayOfYearEmbedding True --temporal_dropout 0.1 --ffn_ratio 4 --spatial_heads 4

finish
exit 0
fi

# ============================================================
# Phase 2: Depth & Regularization (用Phase 1最佳hidden_channels)
# ============================================================
if [ "$1" == "--phase" ] && [ "$2" == "2" ]; then

BEST_H=64  # <-- Update after Phase 1 analysis

echo "===== Phase 2: Depth & Regularization (hidden=$BEST_H) ====="

# Exp 2.1: temporal_layers=4
python train.py --model_name $MODEL --dataset $DATASET --epochs $EPOCHS \
    --model_des p2_t4 \
    --hidden_channels $BEST_H --num_temporal_att_layer 4 --num_spatial_att_layer 2 \
    --IsDayOfYearEmbedding True --temporal_dropout 0.1 --ffn_ratio 4 --spatial_heads 4

# Exp 2.2: spatial_layers=3
python train.py --model_name $MODEL --dataset $DATASET --epochs $EPOCHS \
    --model_des p2_s3 \
    --hidden_channels $BEST_H --num_temporal_att_layer 3 --num_spatial_att_layer 3 \
    --IsDayOfYearEmbedding True --temporal_dropout 0.1 --ffn_ratio 4 --spatial_heads 4

# Exp 2.3: both deeper
python train.py --model_name $MODEL --dataset $DATASET --epochs $EPOCHS \
    --model_des p2_t4s3 \
    --hidden_channels $BEST_H --num_temporal_att_layer 4 --num_spatial_att_layer 3 \
    --IsDayOfYearEmbedding True --temporal_dropout 0.1 --ffn_ratio 4 --spatial_heads 4

# Exp 2.4: batch_size=16
python train.py --model_name $MODEL --dataset $DATASET --epochs $EPOCHS \
    --model_des p2_bs16 --batch_size 16 \
    --hidden_channels $BEST_H --num_temporal_att_layer 3 --num_spatial_att_layer 2 \
    --IsDayOfYearEmbedding True --temporal_dropout 0.1 --ffn_ratio 4 --spatial_heads 4

# Exp 2.5: batch_size=32
python train.py --model_name $MODEL --dataset $DATASET --epochs $EPOCHS \
    --model_des p2_bs32 --batch_size 32 \
    --hidden_channels $BEST_H --num_temporal_att_layer 3 --num_spatial_att_layer 2 \
    --IsDayOfYearEmbedding True --temporal_dropout 0.1 --ffn_ratio 4 --spatial_heads 4

finish
exit 0
fi

# ============================================================
# Default: Run single experiment (quick iteration)
# ============================================================
DES=${1:-"default"}
echo "===== Running single experiment on $DATASET (des=$DES) ====="
python train.py --model_name $MODEL --dataset $DATASET --epochs $EPOCHS \
    --model_des $DES \
    --hidden_channels 32 --num_temporal_att_layer 3 --num_spatial_att_layer 2 \
    --IsDayOfYearEmbedding True --temporal_dropout 0.1 --ffn_ratio 4 --spatial_heads 4

finish
