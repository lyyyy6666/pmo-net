#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-gansuair}"
shift || true

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

case "${DATASET}" in
  gansuair)
    python -m pmonet.experiments.train_wind_gansu --mode test --root_path "${REPO_ROOT}/data" --coords_path "${REPO_ROOT}/data/station_coords_physics_first.npy" --use_wind_advection true --use_observable_dyn_loss false --lambda_observable_dyn 0.0 --lambda_nonnegative 0.1 --lambda_temporal_smooth 0.01 --save_dir "${REPO_ROOT}/outputs/gansuair" --checkpoints "${REPO_ROOT}/checkpoints/gansuair" "$@"
    ;;
  knowair)
    python -m pmonet.experiments.train_wind_knowair --mode test --root_path "${REPO_ROOT}/data/KnowAir" --use_wind_advection true --use_observable_dyn_loss false --lambda_observable_dyn 0.0 --lambda_nonnegative 0.1 --lambda_temporal_smooth 0.01 --save_dir "${REPO_ROOT}/outputs/knowair" --checkpoints "${REPO_ROOT}/checkpoints/knowair" "$@"
    ;;
  beijing)
    python -m pmonet.experiments.train_wind_beijing --mode test --root_path "${REPO_ROOT}/data/Beijing1718/processed" --save_dir "${REPO_ROOT}/outputs/beijing" --checkpoints "${REPO_ROOT}/checkpoints/beijing" "$@"
    ;;
  *)
    echo "Unknown dataset: ${DATASET}. Use one of: gansuair, knowair, beijing." >&2
    exit 2
    ;;
esac
