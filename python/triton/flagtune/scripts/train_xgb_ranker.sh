#!/usr/bin/env bash
# train_xgb_ranker.sh  (adapted for FlagTree)
# Trains XGBoost ranking model using triton.flagtune
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLAGTUNE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

ADAPTER="mm"
TRAIN_DB=""
EXPORT_MODEL_DIR=""
TRAIN_MODE="shape100_config100"
N_ESTIMATORS=1200
MAX_DEPTH=8
LEARNING_RATE=0.03
SEED=2026
N_JOBS=4

while [[ $# -gt 0 ]]; do
    case "$1" in
        --adapter)          ADAPTER="$2"; shift 2 ;;
        --train-db)         TRAIN_DB="$2"; shift 2 ;;
        --export-model-dir) EXPORT_MODEL_DIR="$2"; shift 2 ;;
        --train-mode)       TRAIN_MODE="$2"; shift 2 ;;
        --n-estimators)     N_ESTIMATORS="$2"; shift 2 ;;
        --max-depth)        MAX_DEPTH="$2"; shift 2 ;;
        --learning-rate)    LEARNING_RATE="$2"; shift 2 ;;
        --seed)             SEED="$2"; shift 2 ;;
        --n-jobs)           N_JOBS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --train-db <db> --export-model-dir <dir>"
            exit 0 ;;
        *) echo "[ERROR] Unknown: $1"; exit 1 ;;
    esac
done

[[ -z "$TRAIN_DB" ]] && { echo "[ERROR] --train-db required"; exit 1; }
[[ -z "$EXPORT_MODEL_DIR" ]] && { echo "[ERROR] --export-model-dir required"; exit 1; }

echo "[INFO] Training XGBoost ranking model"
echo "[INFO] train-db: $TRAIN_DB"
echo "[INFO] export-dir: $EXPORT_MODEL_DIR"
echo "[INFO] train-mode: $TRAIN_MODE"

cd "$FLAGTUNE_DIR"

$PYTHON_BIN -c "
from triton.flagtune.core.ranking import XGBoostRankingTrainer
from triton.flagtune.adapters.mm.parameter_space import mm_parameter_space
from triton.flagtune.adapters.mm.input_space import mm_input_space
from triton.flagtune.adapters.mm.feature_pipeline import MMFeaturePipeline
from triton.flagtune.adapters.mm.data_source import BenchmarkCacheDataSource
from pathlib import Path

ps = mm_parameter_space()
isp = mm_input_space()
pl = MMFeaturePipeline()
ds = BenchmarkCacheDataSource()

trainer = XGBoostRankingTrainer(
    param_space=ps,
    input_space=isp,
    feature_pipeline=pl,
    data_source=ds,
    xgb_params={
        'n_estimators': $N_ESTIMATORS,
        'max_depth': $MAX_DEPTH,
        'learning_rate': $LEARNING_RATE,
        'n_jobs': $N_JOBS,
    },
    seed=$SEED,
)
model, info = trainer.fit('$TRAIN_DB', train_mode='$TRAIN_MODE')
trainer.export(model, info, Path('$EXPORT_MODEL_DIR'))
print(f'Model exported to $EXPORT_MODEL_DIR')
print(f'Features: {info[\"feature_count\"]}')
print(f'Training shapes: {info[\"train_shape_count\"]}')
print(f'Total configs: {info[\"global_train_config_count\"]}')
"

echo "[DONE] Model exported to $EXPORT_MODEL_DIR"
