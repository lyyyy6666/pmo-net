# Audit Report

## Scope

This release was created from the original project directory without deleting or modifying original files. The cleaned copy is located in `PMO-Net-release/`.

## Final Model Files

Included as final PMO-Net release code:

- `src/pmonet/models/pmonet.py`: wind-aware dual-ODE PMO-Net final model, copied from `xlstm_mixer/models/xlstm_dualode_wind_consistency.py`.
- `src/pmonet/modules/multigraph_conv.py`: multi-graph convolution module.
- `src/pmonet/modules/spatial_modules.py`: static context and spatial modules.
- `src/pmonet/modules/revin.py`: RevIN normalization.
- `src/pmonet/models/base_model.py`: minimal release base class.

Included as legacy/reference code:

- `src/pmonet/models/pmonet_legacy.py`: old non-wind dual-ODE PMO-Net entry, copied for traceability and ablation reference.

## Training, Evaluation, Data, and Graph Code

Included:

- `src/pmonet/experiments/train_wind_gansu.py`
- `src/pmonet/experiments/train_wind_knowair.py`
- `src/pmonet/experiments/train_wind_beijing.py`
- `src/pmonet/experiments/train_wind_beijing_airdualode_style.py`
- `src/pmonet/data/gansu.py`
- `src/pmonet/data/knowair.py`
- `src/pmonet/data/beijing.py`
- `src/pmonet/data/beijing_airdualode_style.py`
- `src/pmonet/data/_legacy_rimst_loader.py`
- `src/pmonet/graphs/build_knn_graph.py`
- `scripts/train_gansuair.sh`
- `scripts/train_knowair.sh`
- `scripts/train_beijing.sh`
- `scripts/evaluate.sh`

The experiment entry points were import-path adjusted from `xlstm_mixer.*` to `pmonet.*`. Output paths in the shell scripts are relative to the release repository.

## Historical or Discarded Experiment Files

Not included:

- `hyberparamter_exp/`, `transfer_exp/`, `tools/debug/`, most `visualization/`, notebooks, plotting-only scripts, and old benchmark shell scripts.
- Earlier entry points such as `train_final.py`, `train_final_knowair.py`, `train_physics_first.py`, `train_baseline.py`, and baseline-only scripts.
- `results/pmo_tuning_*`, `results/pmo_simple/*`, transfer results, debug results, and old optimization outputs, except where small final CSV values were explicitly mapped.

Reason: these are historical experiments, failed/debug paths, plotting utilities, or unrelated benchmark/transfer work outside the release target.

## Files Not Suitable for Public Release

Excluded:

- `checkpoints/`, `*.pth`, `*.pt`, `*.ckpt`
- raw and processed data under original `data/`
- prediction arrays such as `pred_*.npy` and `true_*.npy`
- `logs/`, TensorBoard-like run directories, wandb outputs if present
- `__pycache__/`, generated figures, notebooks with local environment output
- private or server-specific paths found in old docs/notebooks

The release `.gitignore` blocks these file types from accidental addition.

## Result Files

Final experimental result files are not included in the public release copy. The following are intentionally excluded:

- manuscript table CSVs
- `best_metrics.csv` and `test_metrics.csv`
- prediction arrays
- checkpoint-derived outputs
- logs and TensorBoard/wandb artifacts

## Missing or Needing Confirmation

- Numerical result files must be archived separately if required by the journal or research data policy.
- Confirm whether the public GitHub repository should remain code-only, or whether a separate data archive DOI will host final metrics and model outputs.

## Safety Check Notes

The release directory was checked for common leak patterns and large artifacts. No checkpoints, raw data tensors, `__pycache__`, `.pt`, `.pth`, `.ckpt`, TensorBoard logs, or wandb directories are intended to be present.
