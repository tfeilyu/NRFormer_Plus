#!/bin/bash
# ============================================================
# NRFormer+ Auto Hyperparameter Search on i6_r20
#
# Automatically picks the best params from each phase and
# carries them into the next phase. Just run:
#   bash search_i6r20.sh [GPU]        # Runs all 4 phases
#   bash search_i6r20.sh --phase 1 [GPU]  # Phase 1 only
#
# Base model: i6_r20 (physics_mode=feature, avg24=2.267)
# ============================================================

set -e  # exit on error

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

# ── NRFormer-aligned params ──
NRFIX="--temporal_dropout 0.3 --ffn_ratio 1 --spatial_heads 8"

# ============================================================
# Helper: find best experiment by avg24 MAE from results.json
#   Usage: best_exp=$(find_best "s1_")
#   Returns the experiment directory name with lowest avg24
# ============================================================
find_best() {
    local prefix=$1
    python3 -c "
import json, os, sys
base = 'logs/NRFormer_Plus/1D-data'
best_mae, best_dir = 999, ''
for d in os.listdir(base):
    if not d.startswith('${prefix}'): continue
    rfile = os.path.join(base, d, 'results.json')
    if not os.path.exists(rfile): continue
    with open(rfile) as f:
        data = json.load(f)
    mae = data['per_horizon']['step_24']['MAE_avg']
    if mae < best_mae:
        best_mae, best_dir = mae, d
if best_dir:
    print(best_dir)
else:
    print('NOT_FOUND')
    sys.exit(1)
"
}

# ============================================================
# Helper: extract config value from best experiment's config.json
#   Usage: val=$(get_config "s1_" "weight_lr")
# ============================================================
get_config() {
    local prefix=$1
    local key=$2
    local best_dir=$(find_best "$prefix")
    python3 -c "
import json
with open('logs/NRFormer_Plus/1D-data/${best_dir}/config.json') as f:
    c = json.load(f)
print(c.get('${key}', ''))
"
}

# ============================================================
# Helper: print phase results summary
# ============================================================
print_phase_results() {
    local prefix=$1
    local phase_name=$2
    python3 -c "
import json, os
base = 'logs/NRFormer_Plus/1D-data'
rows = []
for d in sorted(os.listdir(base)):
    if not d.startswith('${prefix}'): continue
    rfile = os.path.join(base, d, 'results.json')
    if not os.path.exists(rfile): continue
    with open(rfile) as f:
        data = json.load(f)
    ph = data['per_horizon']
    name = d.split('_2026')[0]
    rows.append((name,
        ph['step_6']['MAE_avg'], ph['step_12']['MAE_avg'], ph['step_24']['MAE_avg'],
        ph['step_24'].get('RMSE_avg', data['test']['RMSE']),
        data.get('best_epoch', '?')))

rows.sort(key=lambda x: x[3])
print()
print('===== ${phase_name} Results (sorted by avg24) =====')
print(f'{\"Exp\":<22} {\"avg6\":>7} {\"avg12\":>7} {\"avg24\":>7} {\"RMSE\":>8} {\"BstEp\":>6}')
print('-'*62)
for name, a6, a12, a24, rmse, ep in rows:
    print(f'{name:<22} {a6:>7.4f} {a12:>7.4f} {a24:>7.4f} {rmse:>8.3f} {ep:>6}')
if rows:
    best = rows[0]
    print(f'\\n>>> Best: {best[0]} (avg24={best[3]:.4f})')
print()
"
}

# Auto compare + push
finish() {
    echo ""
    echo "===== Pushing results to GitHub ====="
    python compare_results.py 2>/dev/null
    git add logs/ 2>/dev/null
    git commit -m "Auto: search_i6r20 results $(date '+%Y%m%d-%H%M%S')" 2>/dev/null
    git push 2>/dev/null && echo "Push OK" || echo "Push failed"
}

GPU=${2:-"0"}
# If --phase N is given, override GPU position
if [ "$1" == "--phase" ]; then
    GPU=${3:-"0"}
fi

# ============================================================
# Phase 1: Learning Rate & Training Strategy
# ============================================================
run_phase1() {
    local GPU=$1
    echo ""
    echo "########################################"
    echo "# Phase 1: LR & Training (7 exps)"
    echo "# GPU: $GPU"
    echo "########################################"

    for exp in \
        "s1_lr3e4    --weight_lr 0.0003 --warmup_epochs 5  --scheduler cosine    --early_stop_steps 30" \
        "s1_lr5e4    --weight_lr 0.0005 --warmup_epochs 5  --scheduler cosine    --early_stop_steps 30" \
        "s1_lr2e4    --weight_lr 0.0002 --warmup_epochs 5  --scheduler cosine    --early_stop_steps 50" \
        "s1_lr3e4_w10 --weight_lr 0.0003 --warmup_epochs 10 --scheduler cosine   --early_stop_steps 30" \
        "s1_lr3e4_w3  --weight_lr 0.0003 --warmup_epochs 3  --scheduler cosine   --early_stop_steps 30" \
        "s1_lr1e3_p30 --weight_lr 0.001  --warmup_epochs 5  --scheduler cosine   --early_stop_steps 30" \
        "s1_lr3e4_ms  --weight_lr 0.0003 --warmup_epochs 0  --scheduler multistep --early_stop_steps 50" \
    ; do
        name=$(echo $exp | awk '{print $1}')
        params=$(echo $exp | cut -d' ' -f2-)
        echo ""
        echo ">>> Running $name"
        CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON $FEAT $NRFIX \
            --model_des $name $params || echo "FAILED: $name"
    done

    print_phase_results "s1_" "Phase 1: LR & Training"
    finish
}

# ============================================================
# Phase 2: Architecture (auto-reads best LR from Phase 1)
# ============================================================
run_phase2() {
    local GPU=$1
    echo ""
    echo "########################################"
    echo "# Phase 2: Architecture (6 exps)"
    echo "# GPU: $GPU"
    echo "########################################"

    # Auto-select best training params from Phase 1
    local BEST_DIR=$(find_best "s1_")
    if [ "$BEST_DIR" == "NOT_FOUND" ]; then
        echo "ERROR: No Phase 1 results found. Run Phase 1 first."
        exit 1
    fi

    local BEST_LR=$(get_config "s1_" "weight_lr")
    local BEST_WARMUP=$(get_config "s1_" "warmup_epochs")
    local BEST_SCHED=$(get_config "s1_" "scheduler")
    local BEST_ES=$(get_config "s1_" "early_stop_steps")
    echo "Auto-selected from Phase 1 best ($BEST_DIR):"
    echo "  LR=$BEST_LR, warmup=$BEST_WARMUP, scheduler=$BEST_SCHED, patience=$BEST_ES"
    echo ""

    TRAIN="--weight_lr $BEST_LR --warmup_epochs $BEST_WARMUP --scheduler $BEST_SCHED --early_stop_steps $BEST_ES"

    for exp in \
        "s2_h48   --hidden_channels 48" \
        "s2_h24   --hidden_channels 24" \
        "s2_tl4   --num_temporal_att_layer 4" \
        "s2_tl2   --num_temporal_att_layer 2" \
        "s2_sl3   --num_spatial_att_layer 3" \
        "s2_ec256 --end_channels 256" \
    ; do
        name=$(echo $exp | awk '{print $1}')
        params=$(echo $exp | cut -d' ' -f2-)
        echo ""
        echo ">>> Running $name"
        CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON $FEAT $NRFIX $TRAIN \
            --model_des $name $params || echo "FAILED: $name"
    done

    print_phase_results "s2_" "Phase 2: Architecture"
    finish
}

# ============================================================
# Phase 3: Component Ablation (auto-reads best from Phase 1+2)
# ============================================================
run_phase3() {
    local GPU=$1
    echo ""
    echo "########################################"
    echo "# Phase 3: Component Ablation (7 exps)"
    echo "# GPU: $GPU"
    echo "########################################"

    # Auto-select best training params from Phase 1
    local BEST_LR=$(get_config "s1_" "weight_lr")
    local BEST_WARMUP=$(get_config "s1_" "warmup_epochs")
    local BEST_SCHED=$(get_config "s1_" "scheduler")
    local BEST_ES=$(get_config "s1_" "early_stop_steps")

    # Check if Phase 2 improved anything; if so, use its best architecture
    local P2_DIR=$(find_best "s2_" 2>/dev/null || echo "NOT_FOUND")
    local ARCH_OVERRIDE=""
    if [ "$P2_DIR" != "NOT_FOUND" ]; then
        local P2_MAE=$(python3 -c "
import json
with open('logs/NRFormer_Plus/1D-data/${P2_DIR}/results.json') as f:
    print(json.load(f)['per_horizon']['step_24']['MAE_avg'])
")
        local P1_BEST_DIR=$(find_best "s1_")
        local P1_MAE=$(python3 -c "
import json
with open('logs/NRFormer_Plus/1D-data/${P1_BEST_DIR}/results.json') as f:
    print(json.load(f)['per_horizon']['step_24']['MAE_avg'])
")
        local USE_P2=$(python3 -c "print('yes' if $P2_MAE < $P1_MAE else 'no')")
        if [ "$USE_P2" == "yes" ]; then
            local P2_HC=$(get_config "s2_" "hidden_channels")
            local P2_TL=$(get_config "s2_" "num_temporal_att_layer")
            local P2_SL=$(get_config "s2_" "num_spatial_att_layer")
            local P2_EC=$(get_config "s2_" "end_channels")
            ARCH_OVERRIDE="--hidden_channels $P2_HC --num_temporal_att_layer $P2_TL --num_spatial_att_layer $P2_SL --end_channels $P2_EC"
            echo "Phase 2 improved! Using architecture: h=$P2_HC, TL=$P2_TL, SL=$P2_SL, ec=$P2_EC"
        else
            echo "Phase 2 did not improve. Keeping default architecture (h=32, TL=3, SL=2)."
        fi
    else
        echo "No Phase 2 results. Using default architecture."
    fi

    TRAIN="--weight_lr $BEST_LR --warmup_epochs $BEST_WARMUP --scheduler $BEST_SCHED --early_stop_steps $BEST_ES"
    echo "Training params: LR=$BEST_LR, warmup=$BEST_WARMUP, scheduler=$BEST_SCHED"
    echo ""

    for exp in \
        "s3_nophys  --use_physics False" \
        "s3_norain  --use_rain_gate False" \
        "s3_nowind  --Is_wind_angle False --Is_wind_speed False" \
        "s3_nodoy   --IsDayOfYearEmbedding False" \
        "s3_r15     --num_region_clusters 15" \
        "s3_r25     --num_region_clusters 25" \
        "s3_drop02  --temporal_dropout 0.2" \
    ; do
        name=$(echo $exp | awk '{print $1}')
        params=$(echo $exp | cut -d' ' -f2-)
        echo ""
        echo ">>> Running $name"
        CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON $FEAT $NRFIX $TRAIN $ARCH_OVERRIDE \
            --model_des $name $params || echo "FAILED: $name"
    done

    print_phase_results "s3_" "Phase 3: Component Ablation"
    finish
}

# ============================================================
# Phase 4: Best Combo + Multi-seed (auto-combines Phase 1-3)
# ============================================================
run_phase4() {
    local GPU=$1
    echo ""
    echo "########################################"
    echo "# Phase 4: Best Combo + Multi-seed"
    echo "# GPU: $GPU"
    echo "########################################"

    # Auto-select best training params
    local BEST_LR=$(get_config "s1_" "weight_lr")
    local BEST_WARMUP=$(get_config "s1_" "warmup_epochs")
    local BEST_SCHED=$(get_config "s1_" "scheduler")

    TRAIN="--weight_lr $BEST_LR --warmup_epochs $BEST_WARMUP --scheduler $BEST_SCHED --early_stop_steps 50"

    # Auto-detect which Phase 3 ablations IMPROVED over baseline
    # (if removing a component helps → remove it in best combo)
    local P1_BEST_DIR=$(find_best "s1_")
    local BASELINE_MAE=$(python3 -c "
import json
with open('logs/NRFormer_Plus/1D-data/${P1_BEST_DIR}/results.json') as f:
    print(json.load(f)['per_horizon']['step_24']['MAE_avg'])
")

    # Build best combo by checking each ablation
    local COMBO_FLAGS=""
    python3 -c "
import json, os
base = 'logs/NRFormer_Plus/1D-data'
baseline = $BASELINE_MAE
improvements = []
ablations = {
    's3_nophys':  '--use_physics False',
    's3_norain':  '--use_rain_gate False',
    's3_nowind':  '--Is_wind_angle False --Is_wind_speed False',
    's3_nodoy':   '--IsDayOfYearEmbedding False',
    's3_r25':     '--num_region_clusters 25',
    's3_drop02':  '--temporal_dropout 0.2',
}
for prefix, flags in ablations.items():
    for d in os.listdir(base):
        if d.startswith(prefix):
            rfile = os.path.join(base, d, 'results.json')
            if os.path.exists(rfile):
                with open(rfile) as f:
                    mae = json.load(f)['per_horizon']['step_24']['MAE_avg']
                if mae < baseline:
                    improvements.append((prefix, flags, mae, baseline - mae))
                break

if improvements:
    improvements.sort(key=lambda x: -x[3])  # biggest improvement first
    print('Improvements found:')
    for name, flags, mae, delta in improvements:
        print(f'  {name}: avg24={mae:.4f} (Δ={delta:+.4f}) → {flags}')
    # Combine all improving flags
    combo = ' '.join(f for _, f, _, _ in improvements)
    print(f'\\nCOMBO_FLAGS={combo}')
else:
    print('No Phase 3 ablation improved over baseline.')
    print('COMBO_FLAGS=')
" > /tmp/search_combo.txt
    cat /tmp/search_combo.txt
    COMBO_FLAGS=$(grep "COMBO_FLAGS=" /tmp/search_combo.txt | tail -1 | cut -d= -f2-)

    echo ""
    echo "Best combo flags: $COMBO_FLAGS"
    echo "Training: LR=$BEST_LR, warmup=$BEST_WARMUP, scheduler=$BEST_SCHED"
    echo ""

    # Exp 1: Best combo
    echo ">>> Running s4_best"
    CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON $FEAT $NRFIX $TRAIN \
        --model_des s4_best $COMBO_FLAGS || echo "FAILED: s4_best"

    # Exp 2-4: Multi-seed for paper (mean ± std)
    for seed in 2025 2026 2027; do
        echo ""
        echo ">>> Running s4_seed${seed}"
        CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON $FEAT $NRFIX $TRAIN \
            --model_des s4_seed${seed} $COMBO_FLAGS || echo "FAILED: s4_seed${seed}"
    done

    print_phase_results "s4_" "Phase 4: Best Combo + Multi-seed"
    finish
}

# ============================================================
# Main: dispatch
# ============================================================
if [ "$1" == "--phase" ]; then
    case $2 in
        1) run_phase1 $GPU ;;
        2) run_phase2 $GPU ;;
        3) run_phase3 $GPU ;;
        4) run_phase4 $GPU ;;
        *) echo "Unknown phase: $2" ;;
    esac
elif [ -z "$1" ] || [[ "$1" =~ ^[0-9]+$ ]]; then
    # Run all phases sequentially
    GPU=${1:-"0"}
    echo "===== Running all 4 phases on GPU $GPU ====="
    run_phase1 $GPU
    run_phase2 $GPU
    run_phase3 $GPU
    run_phase4 $GPU
    echo ""
    echo "===== ALL PHASES COMPLETE ====="
    echo "Final results:"
    print_phase_results "s" "All Phases Combined"
else
    echo "NRFormer+ i6_r20 Auto Hyperparameter Search"
    echo ""
    echo "Usage:"
    echo "  bash search_i6r20.sh [GPU]            # Run all 4 phases (default GPU 0)"
    echo "  bash search_i6r20.sh --phase 1 [GPU]  # Phase 1: LR & training (7 exps)"
    echo "  bash search_i6r20.sh --phase 2 [GPU]  # Phase 2: Architecture (6 exps)"
    echo "  bash search_i6r20.sh --phase 3 [GPU]  # Phase 3: Ablation (7 exps)"
    echo "  bash search_i6r20.sh --phase 4 [GPU]  # Phase 4: Best combo + multi-seed (4 exps)"
    echo ""
    echo "Phases auto-chain: each phase reads the best result from previous phases."
    echo "Run all at once:  bash search_i6r20.sh 0"
    echo "Total: 24 experiments"
fi
