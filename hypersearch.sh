#!/bin/bash
# ============================================================
# NRFormer+ Hyperparameter Search Script
# Based on i8_residual (physics residual correction mode)
#
# Usage:
#   bash hypersearch.sh                    # Run all phases sequentially on GPU 0
#   bash hypersearch.sh --phase 1          # Phase 1 only: training params
#   bash hypersearch.sh --phase 2          # Phase 2 only: architecture
#   bash hypersearch.sh --phase 3          # Phase 3 only: physics & regularization
#   bash hypersearch.sh --phase 1 2        # Phase 1 on GPU 2
#
# After completion: python compare_results.py to see all results
# ============================================================

MODEL="NRFormer_Plus"
DATASET="1D-data"
EPOCHS=200

# ── Fixed base config (from i6_r20 best) ──
BASE="--model_name $MODEL --dataset $DATASET --epochs $EPOCHS"
BASE="$BASE --IsDayOfYearEmbedding True --hidden_channels 32"
BASE="$BASE --num_temporal_att_layer 3 --num_spatial_att_layer 2"
BASE="$BASE --use_log_space True --scheduler cosine --use_rain_gate True"
BASE="$BASE --fusion_type 2way --spatial_swap True"
BASE="$BASE --num_region_clusters 20"
BASE="$BASE --physics_mode residual"
BASE="$BASE --early_stop_steps 30"

# ── Auto compare after experiments ──
finish() {
    echo ""
    echo "===== Comparing all experiment results ====="
    python compare_results.py
    echo ""
    echo "===== Pushing results to GitHub ====="
    git add logs/ EXPERIMENT_LOG.md 2>/dev/null
    git commit -m "Auto: hypersearch results $(date '+%Y%m%d-%H%M%S')" 2>/dev/null
    git push 2>/dev/null && echo "Push OK" || echo "Push failed (check git auth)"
    echo "===== All done ====="
}

GPU=${2:-"0"}

# ============================================================
# Phase 1: Training Parameters (LR, warmup, weight_decay, batch_size)
#   Most impactful for fixing training stability
#   i8_residual best_ep=14, need to improve training depth
# ============================================================
if [ "$1" == "--phase" ] && [ "$2" == "1" ]; then
GPU=${3:-"0"}
echo "===== Phase 1: Training Parameters (GPU $GPU) ====="

# Exp 1: Lower LR (0.0005) — slower but more stable
echo "[1/6] hp_lr5e4"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE \
    --model_des hp_lr5e4 --weight_lr 0.0005 --warmup_epochs 5 \
    --temporal_dropout 0.3 --ffn_ratio 1 --spatial_heads 8

# Exp 2: Lower LR (0.0003) — even slower
echo "[2/6] hp_lr3e4"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE \
    --model_des hp_lr3e4 --weight_lr 0.0003 --warmup_epochs 5 \
    --temporal_dropout 0.3 --ffn_ratio 1 --spatial_heads 8

# Exp 3: Longer warmup (10 epochs)
echo "[3/6] hp_warm10"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE \
    --model_des hp_warm10 --weight_lr 0.001 --warmup_epochs 10 \
    --temporal_dropout 0.3 --ffn_ratio 1 --spatial_heads 8

# Exp 4: Larger batch size (16)
echo "[4/6] hp_bs16"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE \
    --model_des hp_bs16 --weight_lr 0.001 --warmup_epochs 5 --batch_size 16 \
    --temporal_dropout 0.3 --ffn_ratio 1 --spatial_heads 8

# Exp 5: LR 0.0005 + batch 16 (combined)
echo "[5/6] hp_lr5e4_bs16"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE \
    --model_des hp_lr5e4_bs16 --weight_lr 0.0005 --warmup_epochs 5 --batch_size 16 \
    --temporal_dropout 0.3 --ffn_ratio 1 --spatial_heads 8

# Exp 6: LR 0.0005 + warmup 10 + batch 16 (full stable)
echo "[6/6] hp_stable"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE \
    --model_des hp_stable --weight_lr 0.0005 --warmup_epochs 10 --batch_size 16 \
    --temporal_dropout 0.3 --ffn_ratio 1 --spatial_heads 8

finish
exit 0
fi

# ============================================================
# Phase 2: Architecture (hidden_dim, layers, end_channels, dropout)
#   Use best training params from Phase 1 (or default)
# ============================================================
if [ "$1" == "--phase" ] && [ "$2" == "2" ]; then
GPU=${3:-"0"}

# ── Plug in best training params from Phase 1 here ──
TRAIN="--weight_lr 0.001 --warmup_epochs 5 --batch_size 8"
TRAIN="$TRAIN --temporal_dropout 0.3 --ffn_ratio 1 --spatial_heads 8"

echo "===== Phase 2: Architecture (GPU $GPU) ====="

# Exp 1: hidden=48 (50% larger)
echo "[1/6] hp_h48"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE $TRAIN \
    --model_des hp_h48 --hidden_channels 48

# Exp 2: hidden=64
echo "[2/6] hp_h64"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE $TRAIN \
    --model_des hp_h64 --hidden_channels 64

# Exp 3: 4 temporal layers (deeper)
echo "[3/6] hp_tl4"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE $TRAIN \
    --model_des hp_tl4 --num_temporal_att_layer 4

# Exp 4: 3 spatial layers (deeper)
echo "[4/6] hp_sl3"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE $TRAIN \
    --model_des hp_sl3 --num_spatial_att_layer 3

# Exp 5: end_channels=256 (smaller projection)
echo "[5/6] hp_ec256"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE $TRAIN \
    --model_des hp_ec256 --end_channels 256

# Exp 6: dropout=0.2 (less regularization)
echo "[6/6] hp_drop02"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE $TRAIN \
    --model_des hp_drop02 --temporal_dropout 0.2

finish
exit 0
fi

# ============================================================
# Phase 3: Physics & Regularization tuning
#   Fine-tune the residual correction and region clusters
# ============================================================
if [ "$1" == "--phase" ] && [ "$2" == "3" ]; then
GPU=${3:-"0"}

# ── Use best training + architecture from Phase 1+2 ──
TRAIN="--weight_lr 0.001 --warmup_epochs 5 --batch_size 8"
TRAIN="$TRAIN --temporal_dropout 0.3 --ffn_ratio 1 --spatial_heads 8"

echo "===== Phase 3: Physics & Regularization (GPU $GPU) ====="

# Exp 1: More region clusters (25)
echo "[1/5] hp_r25"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE $TRAIN \
    --model_des hp_r25 --num_region_clusters 25

# Exp 2: More region clusters (30)
echo "[2/5] hp_r30"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE $TRAIN \
    --model_des hp_r30 --num_region_clusters 30

# Exp 3: Residual + horizon weighting (inverse_acf)
echo "[3/5] hp_hw_acf"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE $TRAIN \
    --model_des hp_hw_acf --horizon_weight inverse_acf

# Exp 4: Residual + no wind features (F8: wind is useless)
echo "[4/5] hp_nowind"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE $TRAIN \
    --model_des hp_nowind --Is_wind_angle False --Is_wind_speed False

# Exp 5: Residual + feature mode (compare: is residual actually better with best params?)
echo "[5/5] hp_feature"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE $TRAIN \
    --model_des hp_feature --physics_mode feature

finish
exit 0
fi

# ============================================================
# Default: Run all 3 phases sequentially
# ============================================================
if [ -z "$1" ]; then
GPU=${1:-"0"}
echo "===== Running all 3 phases on GPU $GPU ====="
echo ""
bash $0 --phase 1 $GPU
bash $0 --phase 2 $GPU
bash $0 --phase 3 $GPU
echo "===== All phases complete ====="
exit 0
fi

echo "Usage:"
echo "  bash hypersearch.sh                  # Run all phases on GPU 0"
echo "  bash hypersearch.sh --phase 1        # Phase 1: training params"
echo "  bash hypersearch.sh --phase 2        # Phase 2: architecture"
echo "  bash hypersearch.sh --phase 3        # Phase 3: physics & regularization"
echo "  bash hypersearch.sh --phase 1 2      # Phase 1 on GPU 2"
