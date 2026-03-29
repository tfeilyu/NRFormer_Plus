#!/bin/bash
# ============================================================
# Clean rerun script for post-fix validation experiments.
#
# Why this exists:
# 1. Old ablation runs were polluted by argparse bool parsing.
# 2. Old "multi-seed" runs did not actually pass different seeds.
# 3. This script uses fresh model_des prefixes, so new results do not
#    get mixed with old logs when you compare them.
#
# Usage:
#   bash clean_rerun_validations.sh 0
#   bash clean_rerun_validations.sh --phase 1 0
#   bash clean_rerun_validations.sh --phase 2 0
#   bash clean_rerun_validations.sh --phase 3 0
#   RUN_TAG=clean2 bash clean_rerun_validations.sh --phase 3 1
#
# Phases:
#   1. Clean feature-mode ablations
#   2. Cross-family candidate validation
#   3. True multi-seed validation on auto-selected top candidates
#   4. Summary only
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="NRFormer_Plus"
DATASET="1D-data"
EPOCHS=200
RUN_TAG="${RUN_TAG:-clean}"
SEEDS="${SEEDS:-2025 2026 2027}"
TOPK="${TOPK:-4}"
SELECTED_FILE="logs/NRFormer_Plus/1D-data/${RUN_TAG}_phase3_selected.txt"

COMMON="--model_name $MODEL --dataset $DATASET --epochs $EPOCHS"
ARCH_CORE="--hidden_channels 32 --num_temporal_att_layer 3 --num_spatial_att_layer 2"
NRFIX="--temporal_dropout 0.3 --ffn_ratio 1 --spatial_heads 8"
BACKBONE="--use_log_space True --scheduler cosine --warmup_epochs 5"

# Trusted feature-mode baseline after the log-space + r20 line won overall.
FEAT_BASE="$ARCH_CORE $NRFIX $BACKBONE"
FEAT_BASE="$FEAT_BASE --IsDayOfYearEmbedding True --use_rain_gate True"
FEAT_BASE="$FEAT_BASE --fusion_type 2way --spatial_swap True"
FEAT_BASE="$FEAT_BASE --num_region_clusters 20 --physics_mode feature"
FEAT_BASE="$FEAT_BASE --weight_lr 0.001 --early_stop_steps 30"

# Best residual family from previous search; rerun cleanly after parser fix.
RES_BASE="$ARCH_CORE $BACKBONE"
RES_BASE="$RES_BASE --IsDayOfYearEmbedding True --use_rain_gate True"
RES_BASE="$RES_BASE --fusion_type 2way --spatial_swap True"
RES_BASE="$RES_BASE --num_region_clusters 20 --physics_mode residual"
RES_BASE="$RES_BASE --weight_lr 0.0003 --early_stop_steps 30"
RES_BASE="$RES_BASE --temporal_dropout 0.3 --ffn_ratio 1 --spatial_heads 8"

print_usage() {
    cat <<EOF
Clean rerun script for NRFormer+ validation experiments.

Usage:
  bash clean_rerun_validations.sh [GPU]
  bash clean_rerun_validations.sh --phase 1 [GPU]
  bash clean_rerun_validations.sh --phase 2 [GPU]
  bash clean_rerun_validations.sh --phase 3 [GPU]
  bash clean_rerun_validations.sh --phase 4

Environment:
  RUN_TAG=clean2      Change the fresh model_des prefix.
  SEEDS="2025 2026"   Override seed list for phase 3.
  TOPK=3              Number of auto-selected candidates for phase 3.
EOF
}

run_exp() {
    local gpu=$1
    local name=$2
    shift 2

    echo ""
    echo ">>> Running ${name}"
    CUDA_VISIBLE_DEVICES="$gpu" python train.py $COMMON --model_des "${name}" "$@"
}

find_best_model() {
    local prefix=$1
    python3 - "$prefix" <<'PY'
import json
import math
import sys
from pathlib import Path

prefix = sys.argv[1]
base = Path("logs/NRFormer_Plus/1D-data")
best = None

if base.exists():
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "config.json"
        res_path = run_dir / "results.json"
        if not cfg_path.exists() or not res_path.exists():
            continue

        with open(cfg_path) as f:
            cfg = json.load(f)
        model_des = cfg.get("model_des", "")
        if not model_des.startswith(prefix):
            continue

        with open(res_path) as f:
            res = json.load(f)
        avg24 = res["per_horizon"]["step_24"]["MAE_avg"]

        if best is None or avg24 < best[0]:
            best = (avg24, model_des)

if best is None:
    print("NOT_FOUND")
    sys.exit(1)

print(best[1])
PY
}

build_args_from_model() {
    local model_des=$1
    python3 - "$model_des" <<'PY'
import json
import sys
from pathlib import Path

target = sys.argv[1]
base = Path("logs/NRFormer_Plus/1D-data")
best_cfg = None
best_avg24 = None

keys = [
    "batch_size",
    "use_RevIN",
    "IsLocationEncoder",
    "IsLocationInfo",
    "Is_wind_angle",
    "Is_wind_speed",
    "Is_air_temperature",
    "Is_dew_point",
    "IsDayOfYearEmbedding",
    "hidden_channels",
    "end_channels",
    "num_temporal_att_layer",
    "num_spatial_att_layer",
    "temporal_dropout",
    "ffn_ratio",
    "spatial_heads",
    "use_log_space",
    "use_residual",
    "use_rain_gate",
    "scheduler",
    "warmup_epochs",
    "weight_lr",
    "num_global_nodes",
    "fusion_type",
    "spatial_swap",
    "horizon_weight",
    "num_region_clusters",
    "physics_type",
    "use_physics",
    "simple_meteo",
    "spatial_v_source",
    "early_stop_steps",
    "physics_mode",
    "physics_lambda",
]

if base.exists():
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "config.json"
        res_path = run_dir / "results.json"
        if not cfg_path.exists() or not res_path.exists():
            continue

        with open(cfg_path) as f:
            cfg = json.load(f)
        if cfg.get("model_des") != target:
            continue

        with open(res_path) as f:
            res = json.load(f)
        avg24 = res["per_horizon"]["step_24"]["MAE_avg"]
        if best_avg24 is None or avg24 < best_avg24:
            best_avg24 = avg24
            best_cfg = cfg

if best_cfg is None:
    print("")
    sys.exit(1)

parts = []
for key in keys:
    if key not in best_cfg:
        continue
    value = best_cfg[key]
    if isinstance(value, bool):
        value = "True" if value else "False"
    parts.extend([f"--{key}", str(value)])

print(" ".join(parts))
PY
}

select_phase3_candidates() {
    local run_tag=$1
    local topk=$2
    local selected_file=$3
    python3 - "$run_tag" "$topk" "$selected_file" <<'PY'
import json
import sys
from pathlib import Path

run_tag = sys.argv[1]
topk = int(sys.argv[2])
selected_file = Path(sys.argv[3])
base = Path("logs/NRFormer_Plus/1D-data")
best_by_model = {}

if base.exists():
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "config.json"
        res_path = run_dir / "results.json"
        if not cfg_path.exists() or not res_path.exists():
            continue

        with open(cfg_path) as f:
            cfg = json.load(f)
        model_des = cfg.get("model_des", "")
        if not model_des.startswith(f"{run_tag}_"):
            continue
        if model_des.startswith(f"{run_tag}_ms_"):
            continue

        with open(res_path) as f:
            res = json.load(f)
        avg24 = res["per_horizon"]["step_24"]["MAE_avg"]

        prev = best_by_model.get(model_des)
        if prev is None or avg24 < prev:
            best_by_model[model_des] = avg24

rows = sorted(best_by_model.items(), key=lambda x: x[1])[:topk]
selected_file.parent.mkdir(parents=True, exist_ok=True)
selected_file.write_text("\n".join(model for model, _ in rows) + ("\n" if rows else ""))

for model, _ in rows:
    print(model)
PY
}

print_selected_results() {
    local prefix=$1
    python3 - "$prefix" <<'PY'
import json
import os
import sys
from pathlib import Path

prefix = sys.argv[1]
base = Path("logs/NRFormer_Plus/1D-data")
rows = []

if not base.exists():
    print(f"No logs found under {base}")
    sys.exit(0)

for run_dir in sorted(base.iterdir()):
    if not run_dir.is_dir():
        continue
    cfg_path = run_dir / "config.json"
    res_path = run_dir / "results.json"
    if not cfg_path.exists() or not res_path.exists():
        continue

    with open(cfg_path) as f:
        cfg = json.load(f)
    model_des = cfg.get("model_des", "")
    if not model_des.startswith(prefix):
        continue

    with open(res_path) as f:
        res = json.load(f)

    ph = res["per_horizon"]
    rows.append((
        model_des,
        cfg.get("seed", "?"),
        ph["step_6"]["MAE_avg"],
        ph["step_12"]["MAE_avg"],
        ph["step_24"]["MAE_avg"],
        ph["step_24"].get("RMSE_avg", res["test"]["RMSE"]),
        ph["step_24"].get("MAPE_avg", res["test"]["MAPE"]),
        res.get("best_epoch", "?"),
    ))

rows.sort(key=lambda x: x[4])

print()
print(f"===== Results for prefix: {prefix} =====")
if not rows:
    print("No matching runs found.")
    sys.exit(0)

print(f"{'Model':<28} {'Seed':>6} {'avg6':>7} {'avg12':>7} {'avg24':>7} {'RMSE':>8} {'MAPE':>8} {'Ep':>4}")
print("-" * 84)
for row in rows:
    model_des, seed, avg6, avg12, avg24, rmse, mape, epoch = row
    print(f"{model_des:<28} {str(seed):>6} {avg6:>7.4f} {avg12:>7.4f} {avg24:>7.4f} {rmse:>8.4f} {mape*100:>7.2f}% {str(epoch):>4}")
PY
}

print_group_stats() {
    local prefix=$1
    python3 - "$prefix" <<'PY'
import json
import math
import os
import statistics
import sys
from pathlib import Path

prefix = sys.argv[1]
base = Path("logs/NRFormer_Plus/1D-data")
metrics = {"avg6": [], "avg12": [], "avg24": [], "rmse": [], "mape": []}

if not base.exists():
    print(f"No logs found under {base}")
    sys.exit(0)

for run_dir in sorted(base.iterdir()):
    if not run_dir.is_dir():
        continue
    cfg_path = run_dir / "config.json"
    res_path = run_dir / "results.json"
    if not cfg_path.exists() or not res_path.exists():
        continue

    with open(cfg_path) as f:
        cfg = json.load(f)
    model_des = cfg.get("model_des", "")
    if not model_des.startswith(prefix):
        continue

    with open(res_path) as f:
        res = json.load(f)
    ph = res["per_horizon"]
    metrics["avg6"].append(ph["step_6"]["MAE_avg"])
    metrics["avg12"].append(ph["step_12"]["MAE_avg"])
    metrics["avg24"].append(ph["step_24"]["MAE_avg"])
    metrics["rmse"].append(ph["step_24"].get("RMSE_avg", res["test"]["RMSE"]))
    metrics["mape"].append(ph["step_24"].get("MAPE_avg", res["test"]["MAPE"]))

count = len(metrics["avg24"])
print()
print(f"===== Seed stats for: {prefix} =====")
if count == 0:
    print("No matching runs found.")
    sys.exit(0)

def mean_std(values):
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std

for key in ("avg6", "avg12", "avg24", "rmse", "mape"):
    mean, std = mean_std(metrics[key])
    suffix = "%" if key == "mape" else ""
    scale = 100 if key == "mape" else 1
    print(f"{key:>5}: {mean*scale:.4f}{suffix} ± {std*scale:.4f}{suffix}  (n={count})")
PY
}

phase1() {
    local gpu=$1
    echo "===== Phase 1: Clean feature-mode ablations (GPU ${gpu}) ====="

    run_exp "$gpu" "${RUN_TAG}_feat_base" $FEAT_BASE
    run_exp "$gpu" "${RUN_TAG}_feat_nowind" $FEAT_BASE --Is_wind_angle False --Is_wind_speed False
    run_exp "$gpu" "${RUN_TAG}_feat_nophys" $FEAT_BASE --use_physics False
    run_exp "$gpu" "${RUN_TAG}_feat_norain" $FEAT_BASE --use_rain_gate False
    run_exp "$gpu" "${RUN_TAG}_feat_nodoy" $FEAT_BASE --IsDayOfYearEmbedding False
    run_exp "$gpu" "${RUN_TAG}_feat_r15" $FEAT_BASE --num_region_clusters 15
    run_exp "$gpu" "${RUN_TAG}_feat_r25" $FEAT_BASE --num_region_clusters 25
    run_exp "$gpu" "${RUN_TAG}_feat_drop02" $FEAT_BASE --temporal_dropout 0.2

    print_selected_results "${RUN_TAG}_feat_"

    local phase1_best
    phase1_best=$(find_best_model "${RUN_TAG}_feat_")
    echo ""
    echo "Auto-selected Phase 1 best: ${phase1_best}"
}

phase2() {
    local gpu=$1
    echo "===== Phase 2: Cross-family candidate validation (GPU ${gpu}) ====="

    local phase1_best
    phase1_best=$(find_best_model "${RUN_TAG}_feat_")
    if [ "$phase1_best" == "NOT_FOUND" ]; then
        echo "ERROR: No Phase 1 results found. Run Phase 1 first."
        exit 1
    fi

    local feat_auto_args
    feat_auto_args=$(build_args_from_model "$phase1_best")

    echo "Auto-selected Phase 1 best for feature carry-over: ${phase1_best}"

    run_exp "$gpu" "${RUN_TAG}_feat_tl4" $feat_auto_args --num_temporal_att_layer 4
    run_exp "$gpu" "${RUN_TAG}_feat_lr3e4_ms" $feat_auto_args \
        --scheduler multistep --warmup_epochs 0 \
        --weight_lr 0.0003 --early_stop_steps 50

    run_exp "$gpu" "${RUN_TAG}_res_base" $RES_BASE
    run_exp "$gpu" "${RUN_TAG}_res_drop02" $RES_BASE --temporal_dropout 0.2
    run_exp "$gpu" "${RUN_TAG}_res_ffn2" $RES_BASE --ffn_ratio 2
    run_exp "$gpu" "${RUN_TAG}_res_nowind" $RES_BASE --Is_wind_angle False --Is_wind_speed False
    run_exp "$gpu" "${RUN_TAG}_lowlr_nophys" $ARCH_CORE $BACKBONE $NRFIX \
        --IsDayOfYearEmbedding True --use_rain_gate True \
        --fusion_type 2way --spatial_swap True \
        --num_region_clusters 20 --physics_mode feature --use_physics False \
        --weight_lr 0.0003 --early_stop_steps 30

    print_selected_results "${RUN_TAG}_"

    local phase2_top
    phase2_top=$(select_phase3_candidates "$RUN_TAG" "$TOPK" "$SELECTED_FILE" | tr '\n' ' ' | sed 's/[[:space:]]*$//')
    echo ""
    echo "Auto-selected Phase 2 top-${TOPK} candidates for seed validation: ${phase2_top}"
}

phase3() {
    local gpu=$1
    echo "===== Phase 3: True multi-seed validation on auto-selected candidates (GPU ${gpu}) ====="
    echo "Seeds: ${SEEDS}"

    mapfile -t selected_models < <(select_phase3_candidates "$RUN_TAG" "$TOPK" "$SELECTED_FILE")
    if [ "${#selected_models[@]}" -eq 0 ]; then
        echo "ERROR: No candidates found from phases 1-2. Run earlier phases first."
        exit 1
    fi

    echo "Selected candidates:"
    for model_des in "${selected_models[@]}"; do
        echo "  - ${model_des}"
    done

    for model_des in "${selected_models[@]}"; do
        local base_args
        local suffix
        base_args=$(build_args_from_model "$model_des")
        suffix="${model_des#${RUN_TAG}_}"

        for seed in $SEEDS; do
            run_exp "$gpu" "${RUN_TAG}_ms_${suffix}_s${seed}" $base_args --seed "$seed"
        done
    done

    print_selected_results "${RUN_TAG}_ms_"
    for model_des in "${selected_models[@]}"; do
        local suffix
        suffix="${model_des#${RUN_TAG}_}"
        print_group_stats "${RUN_TAG}_ms_${suffix}_"
    done
}

phase4() {
    print_selected_results "${RUN_TAG}_"

    if [ ! -f "$SELECTED_FILE" ]; then
        mapfile -t _phase4_selected < <(select_phase3_candidates "$RUN_TAG" "$TOPK" "$SELECTED_FILE")
    fi

    if [ -f "$SELECTED_FILE" ]; then
        while IFS= read -r model_des; do
            [ -n "$model_des" ] || continue
            local suffix
            suffix="${model_des#${RUN_TAG}_}"
            print_group_stats "${RUN_TAG}_ms_${suffix}_"
        done < "$SELECTED_FILE"
    fi
}

GPU="${2:-0}"
if [ "${1:-}" == "--phase" ]; then
    phase="${2:-}"
    GPU="${3:-0}"
    case "$phase" in
        1) phase1 "$GPU" ;;
        2) phase2 "$GPU" ;;
        3) phase3 "$GPU" ;;
        4) phase4 ;;
        *) print_usage; exit 1 ;;
    esac
    exit 0
fi

if [ "${1:-}" == "--help" ] || [ "${1:-}" == "-h" ]; then
    print_usage
    exit 0
fi

GPU="${1:-0}"
echo "===== Running clean rerun phases 1-3 on GPU ${GPU} ====="
phase1 "$GPU"
phase2 "$GPU"
phase3 "$GPU"
phase4
