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
