# PMO-Net

Clean release implementation for the PMO-Net air-quality forecasting model used for research data and code availability.

This repository contains the final wind-aware dual-ODE PMO-Net model, dataset loaders, graph-aware modules, training/evaluation entry points, configuration files, graph construction utilities, and reproducibility notes. Raw datasets, checkpoints, final result CSVs, prediction arrays, logs, and private experiment artifacts are intentionally excluded.

## Structure

```text
configs/                 Dataset-specific configuration notes
src/pmonet/              Release Python package
src/pmonet/models/       PMO-Net model implementations
src/pmonet/modules/      RevIN, static context, and graph convolution modules
src/pmonet/data/         Dataset loaders
src/pmonet/experiments/  Training and evaluation entry points
scripts/                 Shell wrappers with relative paths
results/                 Placeholder; final experimental results are not included
data/                    Placeholder for user-provided data
docs/                    Audit, result mapping, and reproducibility notes
tests/                   Smoke tests
```

## Installation

```bash
pip install -r requirements.txt
```

For local development without installing the package:

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

## Data Preparation

Raw and processed data are not included. Prepare the datasets under `data/` following `configs/` and `data/README.md`.

Expected inputs include processed dynamic series, station coordinates, graph adjacency files, feature metadata, timestamps, and meteorology tensors for the wind-aware physical branch.

Final PMO-Net uses wind-aware advection for GansuAir and KnowAir when confirmed u/v meteorology is available. The official GansuAir and KnowAir shell scripts explicitly pass `--use_wind_advection true`. Beijing remains conservative by default because this release does not assume a confirmed u/v wind input for that workflow.

To rebuild a geographic KNN graph from station coordinates:

```bash
PYTHONPATH=src python -m pmonet.graphs.build_knn_graph --station_csv data/station.csv --output data/knn_adj.csv --k 5
```

## Training

```bash
bash scripts/train_gansuair.sh
bash scripts/train_knowair.sh
bash scripts/train_beijing.sh
```

The GansuAir and KnowAir scripts enable wind-aware advection and use the manuscript three-term objective by default:

```text
L_total = L_pred + 0.1 L_nonneg + 0.01 L_smooth
```

The optional observable dynamics consistency loss is disabled by default and is not part of the manuscript main experiments.

Extra arguments are forwarded to the Python entry points. Example:

```bash
bash scripts/train_gansuair.sh --epochs 50 --batch_size 16
```

## Evaluation

```bash
bash scripts/evaluate.sh gansuair --checkpoint checkpoints/gansuair/best_model.pth
bash scripts/evaluate.sh knowair --checkpoint checkpoints/knowair/best_model.pth
bash scripts/evaluate.sh beijing --checkpoint checkpoints/beijing/best_model.pth
```

Checkpoints are not included in this release.

## Results

Final experimental result files and manuscript table CSVs are not included in this public source-code release. Use the manuscript and any separately approved research data archive for reported numerical results.

## Smoke Test

```bash
PYTHONPATH=src python tests/smoke_forward.py
```

The smoke test uses dummy tensors and validates the PMO-Net forward path only.

## Data Availability

The public datasets used in this study are available from their original sources, as cited in the manuscript. Raw datasets, checkpoints, and intermediate experimental outputs are not included in this repository due to storage and licensing considerations. The repository provides data-format instructions and configuration files for preparing the required inputs.

## Code Availability

The source code of PMO-Net is available in this repository, including the model implementation, training and evaluation scripts, configuration files, and graph construction utilities.

## Citation

Citation information will be added after manuscript acceptance.

```bibtex
@article{pmonet2026,
  title   = {PMO-Net for Air Quality Forecasting},
  author  = {TBD},
  journal = {TBD},
  year    = {2026}
}
```
