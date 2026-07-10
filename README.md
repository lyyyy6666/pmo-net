# PMO-Net

Physics-Informed Multi-Graph ODE Network for Spatiotemporal AQI Prediction.

This repository provides the source code for PMO-Net, including the model implementation, graph-aware modules, dataset loaders, and training/evaluation scripts.

## Requirements

```bash
pip install -r requirements.txt
```

For local execution, set the Python path:

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

## Data Preparation

Datasets are not stored in this GitHub repository. They will be provided separately through Google Drive:

[Google Drive Dataset Link](PUT_GOOGLE_DRIVE_LINK_HERE)

After downloading the data, place the processed files under `data/` according to the format described in `data/README.md` and the dataset-specific settings in `configs/`.

Graph construction utilities are provided under `src/pmonet/graphs/`. For example:

```bash
PYTHONPATH=src python -m pmonet.graphs.build_knn_graph --station_csv data/station.csv --output data/knn_adj.csv --k 5
```

## Training

```bash
bash scripts/train_gansuair.sh
bash scripts/train_knowair.sh
bash scripts/train_beijing.sh
```

Additional command-line arguments can be appended to the scripts. Example:

```bash
bash scripts/train_gansuair.sh --epochs 50 --batch_size 16
```

## Testing

Evaluate a trained checkpoint with:

```bash
bash scripts/evaluate.sh gansuair --checkpoint checkpoints/gansuair/best_model.pth
bash scripts/evaluate.sh knowair --checkpoint checkpoints/knowair/best_model.pth
bash scripts/evaluate.sh beijing --checkpoint checkpoints/beijing/best_model.pth
```

A lightweight forward-pass smoke test is also included:

```bash
PYTHONPATH=src python tests/smoke_forward.py
```

## Repository Structure

```text
configs/                 Dataset-specific configuration files
data/                    Data format instructions
docs/                    Reproducibility and audit notes
results/                 Notes for result organization
scripts/                 Training and evaluation shell scripts
src/pmonet/              PMO-Net source package
tests/                   Smoke test
requirements.txt         Python dependencies
```

## Notes

- The repository is intended for academic code sharing and reproducibility.
- Dataset files should be downloaded separately from the Google Drive link above.
- Checkpoints and local experiment outputs should be kept outside version control.
