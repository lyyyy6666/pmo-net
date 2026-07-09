#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

python -m pmonet.experiments.train_wind_knowair \
  --mode train \
  --root_path "${REPO_ROOT}/data/KnowAir" \
  --use_wind_advection true \
  --use_observable_dyn_loss false \
  --lambda_observable_dyn 0.0 \
  --lambda_nonnegative 0.1 \
  --lambda_temporal_smooth 0.01 \
  --save_dir "${REPO_ROOT}/outputs/knowair" \
  --checkpoints "${REPO_ROOT}/checkpoints/knowair" \
  "$@"
