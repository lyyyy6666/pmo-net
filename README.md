# PMO-Net

Physics-Informed Multi-Graph ODE Network for Spatiotemporal AQI Prediction.

This repository provides the clean release implementation of PMO-Net, including the core model, graph-aware layers, data loading utilities, and training scripts.

## Requirements

```bash
pip install -r requirements.txt
```

## Dataset

The processed datasets will be provided through Google Drive:

[Google Drive Dataset Link](PUT_GOOGLE_DRIVE_LINK_HERE)

Please place the downloaded files under `data/`.

## Training

```bash
python train.py --config config.yaml --dataset gansuair
python train.py --config config.yaml --dataset knowair
python train.py --config config.yaml --dataset beijing
```

## Structure

```text
models/        PMO-Net main model
layers/        Graph, ODE, and gating layers
losses/        Training losses
dataset.py     Dataset loading
graph_utils.py Graph utilities
metrics.py     Metric functions
train.py       Training entry
config.yaml    Default configuration
data/          Dataset directory
```

## Note

Large datasets, model weights, logs, and intermediate results are not stored in this repository.
