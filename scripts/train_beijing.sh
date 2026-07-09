#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

python -m pmonet.experiments.train_wind_beijing \
  --mode train \
  --root_path "${REPO_ROOT}/data/Beijing1718/processed" \
  --save_dir "${REPO_ROOT}/outputs/beijing" \
  --checkpoints "${REPO_ROOT}/checkpoints/beijing" \
  "$@"
