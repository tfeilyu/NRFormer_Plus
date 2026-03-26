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
# Phase 2: Architecture Search (on residual mode + LR=0.0003)
#   Phase 1 best: LR=0.0003, best_ep=32, avg6=1.808, avg12=2.011
#   Now search architecture params with this optimal LR
# ============================================================
if [ "$1" == "--phase" ] && [ "$2" == "2" ]; then
GPU=${3:-"0"}

# ── Best training params from Phase 1: LR=0.0003 ──
TRAIN="--weight_lr 0.0003 --warmup_epochs 5 --batch_size 8"
TRAIN="$TRAIN --temporal_dropout 0.3 --ffn_ratio 1 --spatial_heads 8"

echo "===== Phase 2: Architecture on residual+lr3e4 (GPU $GPU) ====="

# Exp 1: hidden=48
echo "[1/6] hp2_h48"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE $TRAIN \
    --model_des hp2_h48 --hidden_channels 48

# Exp 2: 4 temporal layers
echo "[2/6] hp2_tl4"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE $TRAIN \
    --model_des hp2_tl4 --num_temporal_att_layer 4

# Exp 3: 3 spatial layers
echo "[3/6] hp2_sl3"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE $TRAIN \
    --model_des hp2_sl3 --num_spatial_att_layer 3

# Exp 4: end_channels=256 (smaller projection)
echo "[4/6] hp2_ec256"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE $TRAIN \
    --model_des hp2_ec256 --end_channels 256

# Exp 5: dropout=0.2
echo "[5/6] hp2_drop02"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE $TRAIN \
    --model_des hp2_drop02 --temporal_dropout 0.2

# Exp 6: ffn_ratio=2 (moderate FFN expansion)
echo "[6/6] hp2_ffn2"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE $TRAIN \
    --model_des hp2_ffn2 --ffn_ratio 2

finish
exit 0
fi

# ============================================================
# Phase 3: Feature mode (i6_r20) LR search + cross-mode comparison
#   Key question: is hp_lr3e4's improvement from LR or from residual+LR?
#   Also fine-tune region clusters and other components
# ============================================================
if [ "$1" == "--phase" ] && [ "$2" == "3" ]; then
GPU=${3:-"0"}

TRAIN="--temporal_dropout 0.3 --ffn_ratio 1 --spatial_heads 8"

echo "===== Phase 3: Feature mode tuning + cross-mode comparison (GPU $GPU) ====="

# ── A. i6_r20 (feature mode) with different LRs ──
# Control: is LR=0.0003 universally better, or only for residual?

echo "[1/7] hp3_feat_lr3e4 (i6_r20 + LR=0.0003)"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE $TRAIN \
    --model_des hp3_feat_lr3e4 --physics_mode feature --weight_lr 0.0003 \
    --warmup_epochs 5 --batch_size 8

echo "[2/7] hp3_feat_lr5e4 (i6_r20 + LR=0.0005)"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE $TRAIN \
    --model_des hp3_feat_lr5e4 --physics_mode feature --weight_lr 0.0005 \
    --warmup_epochs 5 --batch_size 8

echo "[3/7] hp3_feat_lr1e3 (i6_r20 + LR=0.001, original)"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE $TRAIN \
    --model_des hp3_feat_lr1e3 --physics_mode feature --weight_lr 0.001 \
    --warmup_epochs 5 --batch_size 8

# ── B. Fine-tuning on best residual config (LR=0.0003) ──

echo "[4/7] hp3_r25 (residual + 25 clusters)"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE $TRAIN \
    --model_des hp3_r25 --weight_lr 0.0003 --warmup_epochs 5 --batch_size 8 \
    --num_region_clusters 25

echo "[5/7] hp3_r30 (residual + 30 clusters)"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE $TRAIN \
    --model_des hp3_r30 --weight_lr 0.0003 --warmup_epochs 5 --batch_size 8 \
    --num_region_clusters 30

echo "[6/7] hp3_nowind (residual + no wind)"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE $TRAIN \
    --model_des hp3_nowind --weight_lr 0.0003 --warmup_epochs 5 --batch_size 8 \
    --Is_wind_angle False --Is_wind_speed False

echo "[7/7] hp3_nophys (no physics + LR=0.0003, cleanest baseline)"
CUDA_VISIBLE_DEVICES=$GPU python train.py $BASE $TRAIN \
    --model_des hp3_nophys --physics_mode feature --use_physics False \
    --weight_lr 0.0003 --warmup_epochs 5 --batch_size 8

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
