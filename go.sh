#!/bin/bash
# ============================================================
# NRFormer+ TKDE Iterative Experiment Runner
#
# Usage:
#   bash go.sh --iter 1         # Run Iteration 1: log-space + residual (3 GPUs)
#   bash go.sh --phase 1        # Run Phase 1: capacity search (3 GPUs)
#   bash go.sh --phase 2        # Run Phase 2: depth & batch size
#   bash go.sh test_name        # Run single experiment on GPU 0
#   bash go.sh test_name 2      # Run single experiment on GPU 2
#   python compare_results.py   # Compare all results
# ============================================================

MODEL="NRFormer_Plus"
DATASET="1D-data"
EPOCHS=200
COMMON="--model_name $MODEL --dataset $DATASET --epochs $EPOCHS --IsDayOfYearEmbedding True --temporal_dropout 0.1 --ffn_ratio 4 --spatial_heads 4 --hidden_channels 32 --num_temporal_att_layer 3 --num_spatial_att_layer 2"

# Auto compare + git push after experiments
finish() {
    echo ""
    echo "===== Comparing all experiment results ====="
    python compare_results.py
    echo ""
    echo "===== Pushing results to GitHub ====="
    git add logs/ EXPERIMENT_LOG.md compare_results.py .gitignore 2>/dev/null
    git commit -m "Auto: experiment results $(date '+%Y%m%d-%H%M%S')" 2>/dev/null
    git push 2>/dev/null && echo "Push OK" || echo "Push failed (check git auth)"
    echo "===== All done ====="
}

# ============================================================
# Iteration 5: Architectural Alignment with NRFormer
#   Root cause: NRFormer+ deviates from NRFormer's proven design
#   in ways that HURT performance. Fix the damage first.
#   Base: i2_cosine_rain (log + cosine + rain)
# ============================================================
if [ "$1" == "--iter" ] && [ "$2" == "5" ]; then

BEST="--use_log_space True --scheduler cosine --warmup_epochs 5 --use_rain_gate True"
# Match NRFormer's proven temporal attention params
NRFIX="--temporal_dropout 0.3 --ffn_ratio 1 --spatial_heads 8"

echo "===== Iteration 5: NRFormer Alignment (3 GPUs parallel) ====="

# GPU 0: align temporal+spatial params only (keep 3way fusion)
CUDA_VISIBLE_DEVICES=0 python train.py $COMMON $BEST $NRFIX \
    --model_des i5_align &

# GPU 1: align + 2way fusion + spatial swap (full NRFormer match)
CUDA_VISIBLE_DEVICES=1 python train.py $COMMON $BEST $NRFIX \
    --model_des i5_full --fusion_type 2way --spatial_swap True &

# GPU 2: align + 2way + spatial swap + horizon weighting + drop wind
CUDA_VISIBLE_DEVICES=2 python train.py $COMMON $BEST $NRFIX \
    --model_des i5_full_hw --fusion_type 2way --spatial_swap True \
    --horizon_weight inverse_acf \
    --Is_wind_angle False --Is_wind_speed False &

echo "3 experiments running on GPU 0,1,2. Waiting..."
wait
finish
exit 0
fi

# ============================================================
# Iteration 4: LR & Regularization Tuning
#   i1_log (MAE=2.289, best_ep=3) is still best overall.
#   Try lower LR to get more stable training + reduce dropout.
# ============================================================
if [ "$1" == "--iter" ] && [ "$2" == "4" ]; then

LOG="--use_log_space True"

echo "===== Iteration 4: LR & Regularization (3 GPUs parallel) ====="

# GPU 0: log + lower LR 0.0005 (half of default, slower but stabler)
CUDA_VISIBLE_DEVICES=0 python train.py $COMMON $LOG \
    --model_des i4_lr5e4 --weight_lr 0.0005 &

# GPU 1: log + lower LR + rain gate (combine best features)
CUDA_VISIBLE_DEVICES=1 python train.py $COMMON $LOG \
    --model_des i4_lr5e4_rain --weight_lr 0.0005 --use_rain_gate True &

# GPU 2: log + LR 0.0003 + rain gate (even lower LR)
CUDA_VISIBLE_DEVICES=2 python train.py $COMMON $LOG \
    --model_des i4_lr3e4_rain --weight_lr 0.0003 --use_rain_gate True &

echo "3 experiments running on GPU 0,1,2. Waiting..."
wait
finish
exit 0
fi

# ============================================================
# Iteration 1: Log-space + Residual Learning (Data-Driven)
#   Basis: F1 (ACF=0.946) + F2 (skewness=18, 2 orders of magnitude)
#   3 experiments on GPU 0,1,2 in parallel
# ============================================================
if [ "$1" == "--iter" ] && [ "$2" == "1" ]; then

echo "===== Iteration 1: Log-space + Residual Learning (3 GPUs parallel) ====="

# GPU 0: log-space only
CUDA_VISIBLE_DEVICES=0 python train.py $COMMON \
    --model_des i1_log --use_log_space True &

# GPU 1: residual only
CUDA_VISIBLE_DEVICES=1 python train.py $COMMON \
    --model_des i1_res --use_residual True &

# GPU 2: log-space + residual combined
CUDA_VISIBLE_DEVICES=2 python train.py $COMMON \
    --model_des i1_log_res --use_log_space True --use_residual True &

echo "3 experiments running on GPU 0,1,2. Waiting..."
wait
finish
exit 0
fi

# ============================================================
# Iteration 3: Multi-scale Spatial + Best Config Combos
#   Basis: F4 (235 isolated nodes), F5 (636km anomaly range)
#   Build on best: log-space + cosine+rain (best 24-step)
# ============================================================
if [ "$1" == "--iter" ] && [ "$2" == "3" ]; then

BEST="--use_log_space True --scheduler cosine --warmup_epochs 5 --use_rain_gate True"

echo "===== Iteration 3: Virtual Global Nodes + Combos (3 GPUs parallel) ====="

# GPU 0: best config + 5 global nodes
CUDA_VISIBLE_DEVICES=0 python train.py $COMMON $BEST \
    --model_des i3_g5 --num_global_nodes 5 &

# GPU 1: best config + 10 global nodes
CUDA_VISIBLE_DEVICES=1 python train.py $COMMON $BEST \
    --model_des i3_g10 --num_global_nodes 10 &

# GPU 2: best config + 20 global nodes
CUDA_VISIBLE_DEVICES=2 python train.py $COMMON $BEST \
    --model_des i3_g20 --num_global_nodes 20 &

echo "3 experiments running on GPU 0,1,2. Waiting..."
wait
finish
exit 0
fi

# ============================================================
# Iteration 2: Training Strategy + Rain-aware Gate
#   Basis: i1_log stopped at epoch 3 (LR too high, oscillation)
#          + F3 (radon washout: humid +1.25 nSv/h)
#   Build on best config: log-space
# ============================================================
if [ "$1" == "--iter" ] && [ "$2" == "2" ]; then

LOG="--use_log_space True"

echo "===== Iteration 2: Training Strategy + Rain Gate (3 GPUs parallel) ====="

# GPU 0: log + cosine schedule with warmup (fix oscillation)
CUDA_VISIBLE_DEVICES=0 python train.py $COMMON $LOG \
    --model_des i2_cosine --scheduler cosine --warmup_epochs 5 --weight_lr 0.001 &

# GPU 1: log + rain-aware gate
CUDA_VISIBLE_DEVICES=1 python train.py $COMMON $LOG \
    --model_des i2_rain --use_rain_gate True &

# GPU 2: log + cosine + rain gate (combined)
CUDA_VISIBLE_DEVICES=2 python train.py $COMMON $LOG \
    --model_des i2_cosine_rain --scheduler cosine --warmup_epochs 5 --weight_lr 0.001 --use_rain_gate True &

echo "3 experiments running on GPU 0,1,2. Waiting..."
wait
finish
exit 0
fi

# ============================================================
# Phase 1: Capacity Search — 3 exps on GPU 0,1,2 in parallel
# ============================================================
if [ "$1" == "--phase" ] && [ "$2" == "1" ]; then

echo "===== Phase 1: Baseline Calibration (3 GPUs parallel) ====="

# GPU 0: hidden=32
CUDA_VISIBLE_DEVICES=0 python train.py $COMMON \
    --model_des p1_baseline --hidden_channels 32 \
    --num_temporal_att_layer 3 --num_spatial_att_layer 2 &

# GPU 1: hidden=64
CUDA_VISIBLE_DEVICES=1 python train.py $COMMON \
    --model_des p1_h64 --hidden_channels 64 \
    --num_temporal_att_layer 3 --num_spatial_att_layer 2 &

# GPU 2: hidden=96
CUDA_VISIBLE_DEVICES=2 python train.py $COMMON \
    --model_des p1_h96 --hidden_channels 96 \
    --num_temporal_att_layer 3 --num_spatial_att_layer 2 &

echo "3 experiments running on GPU 0,1,2. Waiting..."
wait
finish
exit 0
fi

# ============================================================
# Phase 2: Depth & Batch Size — 4 GPUs parallel, 2 rounds
# ============================================================
if [ "$1" == "--phase" ] && [ "$2" == "2" ]; then

BEST_H=32  # Updated after Phase 1: hidden=32 is best

echo "===== Phase 2 Round 1: Depth (hidden=$BEST_H, GPU 0-2) ====="

CUDA_VISIBLE_DEVICES=0 python train.py $COMMON \
    --model_des p2_t4 --hidden_channels $BEST_H \
    --num_temporal_att_layer 4 --num_spatial_att_layer 2 &

CUDA_VISIBLE_DEVICES=1 python train.py $COMMON \
    --model_des p2_s3 --hidden_channels $BEST_H \
    --num_temporal_att_layer 3 --num_spatial_att_layer 3 &

CUDA_VISIBLE_DEVICES=2 python train.py $COMMON \
    --model_des p2_t4s3 --hidden_channels $BEST_H \
    --num_temporal_att_layer 4 --num_spatial_att_layer 3 &

wait
echo "===== Phase 2 Round 2: Batch size (GPU 0-1) ====="

CUDA_VISIBLE_DEVICES=0 python train.py $COMMON \
    --model_des p2_bs16 --batch_size 16 --hidden_channels $BEST_H \
    --num_temporal_att_layer 3 --num_spatial_att_layer 2 &

CUDA_VISIBLE_DEVICES=1 python train.py $COMMON \
    --model_des p2_bs32 --batch_size 32 --hidden_channels $BEST_H \
    --num_temporal_att_layer 3 --num_spatial_att_layer 2 &

wait
finish
exit 0
fi

# ============================================================
# Default: Single experiment
# ============================================================
DES=${1:-"default"}
GPU=${2:-"0"}
echo "===== Running experiment '$DES' on GPU $GPU ====="
CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON \
    --model_des $DES

finish
