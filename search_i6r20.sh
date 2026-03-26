#!/bin/bash
# ============================================================
# NRFormer+ Comprehensive Hyperparameter Search on i6_r20
#
# Base model: i6_r20 (physics_mode=feature, current best avg24)
#   avg6=1.835, avg12=2.028, avg24=2.267, RMSE=10.702, best_ep=19
#
# Key insight from hypersearch Phase 1:
#   LR=0.0003 on residual mode → avg6=1.808, avg12=2.011, RMSE=10.462
#   Need to test if LR changes also help feature mode
#
# Usage:
#   bash search_i6r20.sh --phase 1        # Learning rate & training
#   bash search_i6r20.sh --phase 2        # Architecture
#   bash search_i6r20.sh --phase 3        # Components & ablation
#   bash search_i6r20.sh --phase 4        # Best combo refinement
#   bash search_i6r20.sh --phase 1 2      # Phase 1 on GPU 2
# ============================================================

MODEL="NRFormer_Plus"
DATASET="1D-data"
EPOCHS=200
COMMON="--model_name $MODEL --dataset $DATASET --epochs $EPOCHS"

# ── i6_r20 fixed config ──
FEAT="--IsDayOfYearEmbedding True --hidden_channels 32"
FEAT="$FEAT --num_temporal_att_layer 3 --num_spatial_att_layer 2"
FEAT="$FEAT --use_log_space True --use_rain_gate True"
FEAT="$FEAT --fusion_type 2way --spatial_swap True"
FEAT="$FEAT --num_region_clusters 20"
FEAT="$FEAT --physics_mode feature"

# ── NRFormer-aligned params (proven best) ──
NRFIX="--temporal_dropout 0.3 --ffn_ratio 1 --spatial_heads 8"

# Auto compare + push
finish() {
    echo ""
    echo "===== Comparing all experiment results ====="
    python compare_results.py
    echo ""
    echo "===== Pushing results to GitHub ====="
    git add logs/ 2>/dev/null
    git commit -m "Auto: search_i6r20 results $(date '+%Y%m%d-%H%M%S')" 2>/dev/null
    git push 2>/dev/null && echo "Push OK" || echo "Push failed"
    echo "===== Phase done ====="
}

# ============================================================
# Phase 1: Learning Rate & Training Strategy (most impactful)
#   hp_lr3e4 showed LR=0.0003 trains to ep32 and is much better
#   Test if this transfers to feature mode
# ============================================================
if [ "$1" == "--phase" ] && [ "$2" == "1" ]; then
GPU=${3:-"0"}
echo "===== Phase 1: LR & Training on i6_r20 (GPU $GPU) ====="

# Exp 1: LR=0.0003 (the big one — does it help feature mode too?)
echo "[1/7] s1_lr3e4"
CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON $FEAT $NRFIX \
    --model_des s1_lr3e4 --weight_lr 0.0003 --warmup_epochs 5 \
    --scheduler cosine --early_stop_steps 30

# Exp 2: LR=0.0005
echo "[2/7] s1_lr5e4"
CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON $FEAT $NRFIX \
    --model_des s1_lr5e4 --weight_lr 0.0005 --warmup_epochs 5 \
    --scheduler cosine --early_stop_steps 30

# Exp 3: LR=0.0002 (even lower)
echo "[3/7] s1_lr2e4"
CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON $FEAT $NRFIX \
    --model_des s1_lr2e4 --weight_lr 0.0002 --warmup_epochs 5 \
    --scheduler cosine --early_stop_steps 50

# Exp 4: LR=0.0003 + warmup=10
echo "[4/7] s1_lr3e4_w10"
CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON $FEAT $NRFIX \
    --model_des s1_lr3e4_w10 --weight_lr 0.0003 --warmup_epochs 10 \
    --scheduler cosine --early_stop_steps 30

# Exp 5: LR=0.0003 + warmup=3 (shorter warmup)
echo "[5/7] s1_lr3e4_w3"
CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON $FEAT $NRFIX \
    --model_des s1_lr3e4_w3 --weight_lr 0.0003 --warmup_epochs 3 \
    --scheduler cosine --early_stop_steps 30

# Exp 6: LR=0.001 + patience=30 (original LR, more patience — control)
echo "[6/7] s1_lr1e3_p30"
CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON $FEAT $NRFIX \
    --model_des s1_lr1e3_p30 --weight_lr 0.001 --warmup_epochs 5 \
    --scheduler cosine --early_stop_steps 30

# Exp 7: LR=0.0003 + multistep scheduler (no cosine, test scheduler effect)
echo "[7/7] s1_lr3e4_multi"
CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON $FEAT $NRFIX \
    --model_des s1_lr3e4_multi --weight_lr 0.0003 --warmup_epochs 0 \
    --scheduler multistep --early_stop_steps 50

finish
exit 0
fi

# ============================================================
# Phase 2: Architecture Search
#   Use best LR from Phase 1 (update BEST_LR after Phase 1)
# ============================================================
if [ "$1" == "--phase" ] && [ "$2" == "2" ]; then
GPU=${3:-"0"}

# ── Plug in best LR from Phase 1 ──
BEST_LR="--weight_lr 0.0003 --warmup_epochs 5 --scheduler cosine --early_stop_steps 30"

echo "===== Phase 2: Architecture on i6_r20 + best LR (GPU $GPU) ====="

# Exp 1: hidden=48
echo "[1/6] s2_h48"
CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON $FEAT $NRFIX $BEST_LR \
    --model_des s2_h48 --hidden_channels 48

# Exp 2: hidden=24 (smaller — less overfitting?)
echo "[2/6] s2_h24"
CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON $FEAT $NRFIX $BEST_LR \
    --model_des s2_h24 --hidden_channels 24

# Exp 3: TL=4 (deeper temporal)
echo "[3/6] s2_tl4"
CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON $FEAT $NRFIX $BEST_LR \
    --model_des s2_tl4 --num_temporal_att_layer 4

# Exp 4: TL=2 (shallower temporal)
echo "[4/6] s2_tl2"
CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON $FEAT $NRFIX $BEST_LR \
    --model_des s2_tl2 --num_temporal_att_layer 2

# Exp 5: SL=3 (deeper spatial)
echo "[5/6] s2_sl3"
CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON $FEAT $NRFIX $BEST_LR \
    --model_des s2_sl3 --num_spatial_att_layer 3

# Exp 6: end_channels=256
echo "[6/6] s2_ec256"
CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON $FEAT $NRFIX $BEST_LR \
    --model_des s2_ec256 --end_channels 256

finish
exit 0
fi

# ============================================================
# Phase 3: Component Ablation & Tuning
#   Test each NRFormer+ component's contribution with best LR
# ============================================================
if [ "$1" == "--phase" ] && [ "$2" == "3" ]; then
GPU=${3:-"0"}

BEST_LR="--weight_lr 0.0003 --warmup_epochs 5 --scheduler cosine --early_stop_steps 30"

echo "===== Phase 3: Component ablation on i6_r20 + best LR (GPU $GPU) ====="

# Exp 1: No physics module (is physics still neutral with lower LR?)
echo "[1/7] s3_nophys"
CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON $FEAT $NRFIX $BEST_LR \
    --model_des s3_nophys --use_physics False

# Exp 2: No rain gate
echo "[2/7] s3_norain"
CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON $FEAT $NRFIX $BEST_LR \
    --model_des s3_norain --use_rain_gate False

# Exp 3: No wind features
echo "[3/7] s3_nowind"
CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON $FEAT $NRFIX $BEST_LR \
    --model_des s3_nowind --Is_wind_angle False --Is_wind_speed False

# Exp 4: No DayOfYear embedding
echo "[4/7] s3_nodoy"
CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON $FEAT $NRFIX $BEST_LR \
    --model_des s3_nodoy --IsDayOfYearEmbedding False

# Exp 5: Region clusters=15
echo "[5/7] s3_r15"
CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON $FEAT $NRFIX $BEST_LR \
    --model_des s3_r15 --num_region_clusters 15

# Exp 6: Region clusters=25
echo "[6/7] s3_r25"
CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON $FEAT $NRFIX $BEST_LR \
    --model_des s3_r25 --num_region_clusters 25

# Exp 7: dropout=0.2 (less regularization)
echo "[7/7] s3_drop02"
CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON $FEAT $NRFIX $BEST_LR \
    --model_des s3_drop02 --temporal_dropout 0.2

finish
exit 0
fi

# ============================================================
# Phase 4: Best Combo Refinement
#   Combine best findings from Phase 1-3
#   Update these experiments after Phase 1-3 complete
# ============================================================
if [ "$1" == "--phase" ] && [ "$2" == "4" ]; then
GPU=${3:-"0"}

echo "===== Phase 4: Best combo refinement (GPU $GPU) ====="

# ── Placeholder: update with Phase 1-3 winners ──
# Example combos (modify after Phase 1-3 results):

# Exp 1: Best LR + best architecture + best components
echo "[1/4] s4_best"
CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON $FEAT $NRFIX \
    --model_des s4_best --weight_lr 0.0003 --warmup_epochs 5 \
    --scheduler cosine --early_stop_steps 50 \
    --num_region_clusters 25

# Exp 2: Same but no physics (if Phase 3 shows physics still neutral)
echo "[2/4] s4_best_nophys"
CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON $FEAT $NRFIX \
    --model_des s4_best_nophys --weight_lr 0.0003 --warmup_epochs 5 \
    --scheduler cosine --early_stop_steps 50 \
    --num_region_clusters 25 --use_physics False

# Exp 3: Best LR + no wind + no doy (minimal feature set)
echo "[3/4] s4_minimal"
CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON $FEAT $NRFIX \
    --model_des s4_minimal --weight_lr 0.0003 --warmup_epochs 5 \
    --scheduler cosine --early_stop_steps 50 \
    --Is_wind_angle False --Is_wind_speed False --IsDayOfYearEmbedding False

# Exp 4: Reproduce i6_r20 with LR=0.0003 3 times (for mean±std)
echo "[4/4] s4_seed42"
CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON $FEAT $NRFIX \
    --model_des s4_seed42 --weight_lr 0.0003 --warmup_epochs 5 \
    --scheduler cosine --early_stop_steps 30

finish
exit 0
fi

# ============================================================
# Help
# ============================================================
echo "NRFormer+ i6_r20 Hyperparameter Search"
echo ""
echo "Usage:"
echo "  bash search_i6r20.sh --phase 1 [GPU]   # LR & training (7 exps)"
echo "  bash search_i6r20.sh --phase 2 [GPU]   # Architecture (6 exps)"
echo "  bash search_i6r20.sh --phase 3 [GPU]   # Components & ablation (7 exps)"
echo "  bash search_i6r20.sh --phase 4 [GPU]   # Best combo refinement (4 exps)"
echo ""
echo "Run phases sequentially. Update BEST_LR in Phase 2+ after Phase 1."
echo "Total: 24 experiments"
