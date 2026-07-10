from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import build_dataset
from losses.losses import PMONetLoss
from metrics import calculate_metrics
from models.pmonet import xLSTM_WindDualODE_Mixer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PMO-Net.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dataset", type=str, required=True, choices=["gansuair", "knowair", "beijing"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_config(path: str, dataset: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        full_cfg = yaml.safe_load(handle)
    if dataset not in full_cfg:
        raise KeyError(f"Dataset section {dataset!r} not found in {path}")
    return full_cfg[dataset]


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    training = cfg.setdefault("training", {})
    for key in ("epochs", "batch_size", "lr", "device"):
        value = getattr(args, key)
        if value is not None:
            training[key] = value
    return cfg


def build_loader(dataset_name: str, split: str, cfg: dict[str, Any], shuffle: bool) -> tuple[Any, DataLoader]:
    dataset, collate_fn = build_dataset(dataset_name, split, cfg)
    training = cfg.get("training", {})
    loader = DataLoader(
        dataset,
        batch_size=int(training.get("batch_size", 16)),
        shuffle=shuffle,
        num_workers=int(training.get("num_workers", 0)),
        collate_fn=collate_fn,
        drop_last=False,
    )
    return dataset, loader


def to_device(value: Any, device: torch.device) -> Any:
    return value.to(device) if torch.is_tensor(value) else value


def normalize_batch(dataset_name: str, batch: Any, pred_len: int, device: torch.device) -> dict[str, Any]:
    if dataset_name == "gansuair":
        if len(batch) == 10:
            seq_x, seq_y, meteo_future, meteo_future_raw, _time_feat, adj_geo, adj_poi, adj_land, poi_feat, landuse_feat = batch
        else:
            seq_x, seq_y, meteo_future_raw, _time_feat, adj_geo, adj_poi, adj_land, poi_feat, landuse_feat = batch
            meteo_future = meteo_future_raw
        static_feat = None
        if poi_feat is not None and landuse_feat is not None:
            static_feat = torch.cat([poi_feat, landuse_feat], dim=-1)
            if static_feat.dim() == 3:
                static_feat = static_feat[0]
        return {
            "x_enc": seq_x.to(device),
            "target": seq_y[:, -pred_len:, :, :].to(device),
            "meteo_future": meteo_future.to(device),
            "meteo_future_raw": meteo_future_raw.to(device),
            "static_feat": to_device(static_feat, device),
            "adj_geo": adj_geo,
            "adj_poi": adj_poi,
            "adj_land": adj_land,
        }

    return {
        "x_enc": batch["x_enc"].to(device),
        "target": batch["future_y"].to(device),
        "meteo_future": batch["meteo_future"].to(device),
        "meteo_future_raw": batch["meteo_future_raw"].to(device),
        "static_feat": to_device(batch.get("static_feat"), device),
        "adj_geo": batch["adj_geo"],
        "adj_poi": None,
        "adj_land": None,
    }


def get_target_stats(dataset: Any) -> tuple[float, float]:
    mean = float(getattr(dataset, "target_mean", 0.0))
    std = float(getattr(dataset, "target_std", 1.0))
    if hasattr(dataset, "scaler") and getattr(dataset, "scaler") is not None and hasattr(dataset, "target_idx"):
        scaler = dataset.scaler
        target_idx = int(dataset.target_idx)
        mean = float(scaler.mean_[target_idx])
        std = float(scaler.scale_[target_idx])
    return mean, std


def build_model(dataset_name: str, dataset: Any, cfg: dict[str, Any], device: torch.device) -> xLSTM_WindDualODE_Mixer:
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    sample = dataset[0]
    if dataset_name == "gansuair":
        if len(sample) == 10:
            x_sample, y_sample, _meteo, _meteo_raw, _time, adj_geo, adj_poi, adj_land, poi_feat, landuse_feat = sample
        else:
            x_sample, y_sample, _meteo, _time, adj_geo, adj_poi, adj_land, poi_feat, landuse_feat = sample
        static_feat_dim = 0
        if poi_feat is not None and landuse_feat is not None:
            static_feat_dim = int(poi_feat.shape[-1] + landuse_feat.shape[-1])
        coords_path = Path(data_cfg.get("coords_path", "station_coords_physics_first.npy"))
        if not coords_path.is_absolute():
            coords_path = Path(data_cfg.get("root_path", "./data")) / coords_path
        if not coords_path.exists() and bool(model_cfg.get("use_wind_advection", False)):
            raise FileNotFoundError(f"GansuAir wind-aware training requires station coordinates: {coords_path}")
        coords = torch.as_tensor(np.load(coords_path), dtype=torch.float32) if coords_path.exists() else torch.zeros(x_sample.shape[1], 2)
        enc_in = int(x_sample.shape[-1])
        dec_out = int(y_sample.shape[-1])
        num_nodes = int(x_sample.shape[1])
    else:
        x_sample = sample["x_enc"]
        y_sample = sample["future_y"]
        adj_geo = sample["adj_geo"]
        adj_poi = None
        adj_land = None
        static_feat_dim = 0
        coords = dataset.coords
        enc_in = int(x_sample.shape[-1])
        dec_out = int(y_sample.shape[-1])
        num_nodes = int(x_sample.shape[1])

    if bool(model_cfg.get("use_wind_advection", False)) and (
        getattr(dataset, "wind_u_meteo_pos", None) is None or getattr(dataset, "wind_v_meteo_pos", None) is None
    ):
        raise ValueError("use_wind_advection=True requires resolved wind_u_meteo_pos and wind_v_meteo_pos.")

    return xLSTM_WindDualODE_Mixer(
        pred_len=int(model_cfg.get("pred_len", 24)),
        seq_len=int(model_cfg.get("seq_len", model_cfg.get("hist_len", 24))),
        enc_in=enc_in,
        dec_out=dec_out,
        num_nodes=num_nodes,
        static_feat_dim=static_feat_dim,
        d_model=int(model_cfg.get("d_model", 256)),
        xlstm_num_blocks=int(model_cfg.get("xlstm_num_blocks", 3)),
        xlstm_dropout=float(model_cfg.get("dropout", 0.1)),
        channel_mixer_hidden=int(model_cfg.get("channel_mixer_hidden", 512)),
        use_static_context=static_feat_dim > 0,
        use_spatial_mixing=True,
        use_sparse=False,
        use_phy_ode=bool(model_cfg.get("use_phy_ode", True)),
        use_unk_ode=bool(model_cfg.get("use_unk_ode", True)),
        use_adaptive_gating=bool(model_cfg.get("use_adaptive_gating", True)),
        latent_dim=int(model_cfg.get("latent_dim", 64)),
        adj_mx=adj_geo.clone().float() if torch.is_tensor(adj_geo) else torch.as_tensor(adj_geo).float(),
        adj_geo=adj_geo.clone().float() if torch.is_tensor(adj_geo) else torch.as_tensor(adj_geo).float(),
        adj_poi=adj_poi.clone().float() if torch.is_tensor(adj_poi) else None,
        adj_land=adj_land.clone().float() if torch.is_tensor(adj_land) else None,
        coords=coords.clone().float() if torch.is_tensor(coords) else torch.as_tensor(coords).float(),
        wind_u_idx=getattr(dataset, "wind_u_meteo_pos", None),
        wind_v_idx=getattr(dataset, "wind_v_meteo_pos", None),
        use_wind_advection=bool(model_cfg.get("use_wind_advection", False)),
        distance_scale_km=float(model_cfg.get("distance_scale_km", 50.0)),
        coords_are_latlon=True,
        use_geo_mask_for_wind=True,
        wind_graph_norm=str(model_cfg.get("wind_graph_norm", "row")),
        require_xlstm_backend=bool(model_cfg.get("require_xlstm_backend", True)),
        dt_hours=float(model_cfg.get("time_interval_hours", 3.0)),
        latent_dt_mode=str(model_cfg.get("latent_dt_mode", "normalized")),
    ).to(device)


def run_epoch(
    dataset_name: str,
    model: torch.nn.Module,
    criterion: PMONetLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    pred_len = int(cfg.get("model", {}).get("pred_len", 24))
    target_idx = cfg.get("model", {}).get("target_idx", 0)
    use_wind = bool(cfg.get("model", {}).get("use_wind_advection", False))
    losses: list[float] = []
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []

    for batch in loader:
        batch = normalize_batch(dataset_name, batch, pred_len, device)
        if is_train:
            optimizer.zero_grad()
        output = model.forecast(
            x_enc=batch["x_enc"],
            static_feat=batch["static_feat"],
            target_idx=int(target_idx) if isinstance(target_idx, int) else None,
            meteo_future=batch["meteo_future_raw"] if use_wind else None,
        )
        loss_dict = criterion(output, batch["target"])
        loss = loss_dict["total"]
        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        preds.append(loss_dict["output_real"].cpu().numpy())
        trues.append(loss_dict["target_real"].cpu().numpy())

    metrics = calculate_metrics(np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)) if preds else {}
    return float(np.mean(losses)) if losses else float("nan"), metrics


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config, args.dataset), args)
    training = cfg.get("training", {})
    device = torch.device(training.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    train_dataset, train_loader = build_loader(args.dataset, "train", cfg, shuffle=True)
    _, val_loader = build_loader(args.dataset, "val", cfg, shuffle=False)
    model = build_model(args.dataset, train_dataset, cfg, device)
    target_mean, target_std = get_target_stats(train_dataset)
    loss_cfg = cfg.get("loss", {})
    criterion = PMONetLoss(
        pred_len=int(cfg.get("model", {}).get("pred_len", 24)),
        target_mean=target_mean,
        target_std=target_std,
        beta=float(loss_cfg.get("beta", 0.05)),
        lambda_nonnegative=float(loss_cfg.get("lambda_nonnegative", 0.1)),
        lambda_temporal_smooth=float(loss_cfg.get("lambda_temporal_smooth", 0.01)),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training.get("lr", 3e-4)))

    best_val = float("inf")
    save_dir = Path(training.get("save_dir", "outputs")) / args.dataset
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(training.get("epochs", 100))
    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = run_epoch(args.dataset, model, criterion, train_loader, optimizer, cfg, device)
        val_loss, val_metrics = run_epoch(args.dataset, model, criterion, val_loader, None, cfg, device)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"val_MAE={val_metrics.get('MAE', float('nan')):.4f} val_RMSE={val_metrics.get('RMSE', float('nan')):.4f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state_dict": model.state_dict(), "config": cfg}, save_dir / "best_model.pth")


if __name__ == "__main__":
    main()
