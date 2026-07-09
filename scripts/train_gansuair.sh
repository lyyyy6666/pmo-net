#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

python -m pmonet.experiments.train_wind_gansu \
  --mode train \
  --root_path "${REPO_ROOT}/data" \
  --data_path Gansu_Air.csv \
  --coords_path "${REPO_ROOT}/data/station_coords_physics_first.npy" \
  --use_wind_advection true \
  --use_observable_dyn_loss false \
  --lambda_observable_dyn 0.0 \
  --lambda_nonnegative 0.1 \
  --lambda_temporal_smooth 0.01 \
  --save_dir "${REPO_ROOT}/outputs/gansuair" \
  --checkpoints "${REPO_ROOT}/checkpoints/gansuair" \
  "$@"
