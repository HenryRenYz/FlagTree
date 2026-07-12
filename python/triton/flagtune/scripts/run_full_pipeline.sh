#!/usr/bin/env bash
# ============================================================================
# run_full_pipeline.sh  (adapted for FlagTree)
# Master orchestrator: train → predict → measure
# Uses triton.flagtune package for training/prediction.
# Model exported to TRITON_FLAGTUNE_MODEL_DIR for use by FlagTree autotuner.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLAGTUNE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ADAPTER="mm"
TRAIN_DB=""
PREDICT_MODELS=()
OUTPUT_DIR=""
TRAIN_MODE="shape50_config100"
TOP_K=10
WARMUP=1000
REP=100
DTYPE="bfloat16"
OP="mm"
GA_GENERATIONS=0
GA_POPULATION_SIZE=50
GA_ELITE_SIZE=10
GA_OFFSPRING_PER_GENERATION=20
N_ESTIMATORS=1200
MAX_DEPTH=8
LEARNING_RATE=0.03
SEED=2026
N_JOBS=4

show_help() {
    cat <<'EOF'
Usage:
  ./scripts/run_full_pipeline.sh [options]

Required:
  --train-db <path>           Training BenchmarkCache SQLite database
  --predict-model <name>      Target model name (repeatable)
  --output-dir <dir>          Output root directory

Optional:
  --adapter <name>            Operator adapter (default: mm)
  --train-mode <mode>         Training sampling mode (default: shape50_config100)
  --top-k <int>               Top-K (default: 10)
  --warmup <int>              Benchmark warmup (default: 1000)
  --rep <int>                 Benchmark repetitions (default: 100)
  --dtype <dtype>             Data type (default: bfloat16)
  --op <name>                 Operator name (default: mm)
  --ga-generations <int>      GA generations (0=disabled)

Example:
  ./scripts/run_full_pipeline.sh \
      --train-db /home/secure/.flaggems/done/model_TunedConfig.db \
      --predict-model Qwen3.5-35B-A3B-p32768d1024 \
      --output-dir mm_xgb_outputs/run1/
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --adapter)              ADAPTER="$2"; shift 2 ;;
        --train-db)             TRAIN_DB="$2"; shift 2 ;;
        --predict-model)        PREDICT_MODELS+=("$2"); shift 2 ;;
        --output-dir)           OUTPUT_DIR="$2"; shift 2 ;;
        --train-mode)           TRAIN_MODE="$2"; shift 2 ;;
        --top-k)                TOP_K="$2"; shift 2 ;;
        --warmup)               WARMUP="$2"; shift 2 ;;
        --rep)                  REP="$2"; shift 2 ;;
        --dtype)                DTYPE="$2"; shift 2 ;;
        --op)                   OP="$2"; shift 2 ;;
        --ga-generations)       GA_GENERATIONS="$2"; shift 2 ;;
        --ga-population-size)   GA_POPULATION_SIZE="$2"; shift 2 ;;
        --ga-elite-size)        GA_ELITE_SIZE="$2"; shift 2 ;;
        --ga-offspring-per-generation) GA_OFFSPRING_PER_GENERATION="$2"; shift 2 ;;
        --n-estimators)         N_ESTIMATORS="$2"; shift 2 ;;
        --max-depth)            MAX_DEPTH="$2"; shift 2 ;;
        --learning-rate)        LEARNING_RATE="$2"; shift 2 ;;
        --seed)                 SEED="$2"; shift 2 ;;
        --n-jobs)               N_JOBS="$2"; shift 2 ;;
        -h|--help)              show_help; exit 0 ;;
        *)
            echo "[ERROR] Unknown parameter: $1"
            exit 1 ;;
    esac
done

if [[ -z "$TRAIN_DB" ]]; then echo "[ERROR] --train-db required"; exit 1; fi
if [[ ${#PREDICT_MODELS[@]} -eq 0 ]]; then echo "[ERROR] --predict-model required"; exit 1; fi
if [[ -z "$OUTPUT_DIR" ]]; then echo "[ERROR] --output-dir required"; exit 1; fi

MODEL_DIR="$OUTPUT_DIR/model"
TOPK_DIR="$OUTPUT_DIR/predicted_topk"
LATENCY_DIR="$OUTPUT_DIR/latency"
mkdir -p "$MODEL_DIR" "$TOPK_DIR" "$LATENCY_DIR"

timed_step() {
    local name="$1"; shift
    echo ""; echo "========================================"
    echo "[PIPELINE] $name"
    echo "========================================"
    local start_ts; start_ts="$(date +%s)"
    "$@"
    local status=$?; local end_ts; end_ts="$(date +%s)"
    local elapsed=$((end_ts - start_ts))
    if [[ $status -eq 0 ]]; then
        echo "[PIPELINE] $name done (${elapsed}s)"
    else
        echo "[PIPELINE] $name failed (exit=$status)"
        exit $status
    fi
}

# Stage 1: Train
timed_step "1/3 Train XGBoost ranking model" \
    bash "$SCRIPT_DIR/train_xgb_ranker.sh" \
        --adapter "$ADAPTER" \
        --train-db "$TRAIN_DB" \
        --export-model-dir "$MODEL_DIR" \
        --train-mode "$TRAIN_MODE" \
        --n-estimators "$N_ESTIMATORS" \
        --max-depth "$MAX_DEPTH" \
        --learning-rate "$LEARNING_RATE" \
        --seed "$SEED" \
        --n-jobs "$N_JOBS"

echo ""
echo "========================================"
echo "[PIPELINE] Model exported to: $MODEL_DIR"
echo "[PIPELINE] Use in FlagTree:"
echo "[PIPELINE]   export TRITON_USE_FLAGTUNE=1"
echo "[PIPELINE]   export TRITON_FLAGTUNE_MODEL_DIR=$MODEL_DIR"
echo "========================================"
