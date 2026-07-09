# Reproducibility Notes

## Environment

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

The final PMO-Net model uses PyTorch and the `xlstm` package. The model implementation has a GRU fallback when `require_xlstm_backend=false`, which is used only for smoke testing if the xLSTM backend is unavailable.

## Data Layout

Data are not included. Prepare the processed datasets under `data/`:

- GansuAir: `data/Gansu_Air.csv`, graph files, station coordinates, and `meteo_physics_first.*` files.
- KnowAir: `data/KnowAir/KnowAir.npy`, `station.csv`, `final_adj.npy`, `graph_data.npz`, and feature metadata.
- Beijing: `data/Beijing1718/processed/Beijing.npy`, station metadata, graph files, timestamps, and feature metadata.

Final PMO-Net uses wind-aware advection for GansuAir and KnowAir when confirmed u/v meteorology is available. The official GansuAir and KnowAir shell scripts pass `--use_wind_advection true`; Beijing remains disabled by default unless a confirmed u/v wind input is prepared and explicitly enabled.

## Graph Construction

If a geographic adjacency matrix must be rebuilt from station coordinates:

```bash
PYTHONPATH=src python -m pmonet.graphs.build_knn_graph \
  --station_csv data/station.csv \
  --output data/knn_adj.csv \
  --lon_col longitude \
  --lat_col latitude \
  --k 5
```

## Training

```bash
bash scripts/train_gansuair.sh
bash scripts/train_knowair.sh
bash scripts/train_beijing.sh
```

Additional command-line arguments are passed through to the Python entry points.

The default training objective used by these release scripts is the manuscript three-term loss:

```text
L_total = L_pred + 0.1 L_nonneg + 0.01 L_smooth
```

`L_pred` is a time-weighted MSE with `beta=0.05` and normalized exponential weights in code. The optional observable dynamics consistency loss remains in the implementation for diagnostics/experimentation but is disabled by default with `--use_observable_dyn_loss false --lambda_observable_dyn 0.0`.

## Evaluation

```bash
bash scripts/evaluate.sh gansuair --checkpoint checkpoints/gansuair/best_model.pth
bash scripts/evaluate.sh knowair --checkpoint checkpoints/knowair/best_model.pth
bash scripts/evaluate.sh beijing --checkpoint checkpoints/beijing/best_model.pth
```

Checkpoints are not included in this release copy.

## Results

Final experimental result files, manuscript table CSVs, prediction arrays, and logs are not included in this public source-code release. Numerical results should be verified from the manuscript and any separately approved research data archive.

## Smoke Test

```bash
PYTHONPATH=src python tests/smoke_forward.py
```

This uses random dummy tensors and validates only the model forward interface.
