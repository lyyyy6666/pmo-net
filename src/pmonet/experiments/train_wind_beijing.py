from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pmonet.experiments.train_wind_knowair import (  # noqa: E402
    KnowAirWindLoss,
    ObservableDynamicsConsistency,
    str2bool,
)
from pmonet.data.beijing import (  # noqa: E402
    BEIJING_SPLITS,
    Dataset_Beijing_ForWind,
    custom_collate_fn_beijing_physics_first,
)
from pmonet.models.pmonet import (  # noqa: E402
    XLSTM_BACKEND_AVAILABLE,
    xLSTM_WindDualODE_Mixer,
)


BEIJING_SEGMENTS = {
    "overall_original_scale": (0, 24),
    "h1_24_original_scale": (0, 8),
    "h25_48_original_scale": (8, 16),
    "h49_72_original_scale": (16, 24),
}

EXPECTED_FEATURE_NAMES = ["PM2.5", "temperature", "pressure", "humidity", "wind_u", "wind_v"]
EXPECTED_TARGET_IDX = 0
EXPECTED_WIND_U_IDX = 4
EXPECTED_WIND_V_IDX = 5
EXPECTED_WIND_U_METEO_POS = 3
EXPECTED_WIND_V_METEO_POS = 4


def calculate_metrics(pred, true, threshold=0.1):
    pred_flat = np.asarray(pred, dtype=np.float64).reshape(-1)
    true_flat = np.asarray(true, dtype=np.float64).reshape(-1)
    mask = np.isfinite(pred_flat) & np.isfinite(true_flat)
    pred_flat = pred_flat[mask]
    true_flat = true_flat[mask]

    if pred_flat.size == 0:
        return {
            "MAE": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "WMAPE": np.nan,
            "R2": np.nan,
            "SMAPE": np.nan,
        }

    mae = float(np.mean(np.abs(pred_flat - true_flat)))
    mse = float(np.mean((pred_flat - true_flat) ** 2))
    rmse = float(np.sqrt(mse))
    true_mean = float(np.mean(true_flat))
    ss_res = float(np.sum((true_flat - pred_flat) ** 2))
    ss_tot = float(np.sum((true_flat - true_mean) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan

    non_zero_mask = np.abs(true_flat) > threshold
    if np.sum(non_zero_mask) > 0:
        mape = float(
            np.mean(np.abs((pred_flat[non_zero_mask] - true_flat[non_zero_mask]) / true_flat[non_zero_mask])) * 100
        )
    else:
        mape = 0.0

    total_abs_error = float(np.sum(np.abs(pred_flat - true_flat)))
    total_abs_true = float(np.sum(np.abs(true_flat)))
    wmape = float((total_abs_error / total_abs_true) * 100) if total_abs_true > 1e-5 else 0.0
    smape_den = np.abs(pred_flat) + np.abs(true_flat)
    smape_mask = smape_den > 1e-6
    if np.any(smape_mask):
        smape = float(np.mean(2.0 * np.abs(pred_flat[smape_mask] - true_flat[smape_mask]) / smape_den[smape_mask]) * 100)
    else:
        smape = 0.0
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "WMAPE": wmape, "R2": r2, "SMAPE": smape}


def flatten_metric_fields(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {
        f"{prefix}_MAE": float(metrics["MAE"]),
        f"{prefix}_RMSE": float(metrics["RMSE"]),
        f"{prefix}_MAPE": float(metrics["MAPE"]),
        f"{prefix}_WMAPE": float(metrics["WMAPE"]),
        f"{prefix}_R2": float(metrics["R2"]),
        f"{prefix}_SMAPE": float(metrics["SMAPE"]),
    }


def write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def parse_args():
    parser = argparse.ArgumentParser(description="Beijing1718 wind-consistency PMO-Net training")
    parser.add_argument("--mode", type=str, default="smoke", choices=["train", "test", "smoke"])

    parser.add_argument("--root_path", type=str, default="./data/Beijing1718/processed")
    parser.add_argument("--data_path", type=str, default="Beijing.npy")
    parser.add_argument("--station_path", type=str, default="station.csv")
    parser.add_argument("--adj_geo_path", type=str, default="final_adj.npy")
    parser.add_argument("--graph_npz_path", type=str, default="graph_data.npz")
    parser.add_argument("--feature_metadata_path", type=str, default="beijing_feature_columns.json")
    parser.add_argument("--timestamps_path", type=str, default="timestamps.csv")

    parser.add_argument("--features", type=str, default="MS", choices=["MS"])
    parser.add_argument("--target_idx", type=int, default=0)
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--hist_len", type=int, default=24)
    parser.add_argument("--label_len", type=int, default=12)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--time_interval_hours", type=float, default=3.0)
    parser.add_argument("--latent_dt_mode", type=str, default="normalized", choices=["normalized", "physical"])
    parser.add_argument("--time_feature_mode", type=str, default="cyclic6", choices=["cyclic6", "simple3"])
    parser.add_argument("--coord_order", type=str, default="lonlat", choices=["lonlat", "latlon"])

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--xlstm_num_blocks", type=int, default=3)
    parser.add_argument("--channel_mixer_hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_phy_ode", type=str2bool, default=True)
    parser.add_argument("--use_unk_ode", type=str2bool, default=True)
    parser.add_argument("--use_adaptive_gating", type=str2bool, default=True)
    parser.add_argument("--gating_hidden_dim", type=int, default=64)
    parser.add_argument("--require_xlstm_backend", type=str2bool, default=True)

    parser.add_argument("--use_wind_advection", type=str2bool, default=False)
    parser.add_argument("--use_observable_dyn_loss", type=str2bool, default=False)
    parser.add_argument("--lambda_observable_dyn", type=float, default=0.0)
    parser.add_argument("--wind_u_idx", type=int, default=4)
    parser.add_argument("--wind_v_idx", type=int, default=5)
    parser.add_argument(
        "--observable_dyn_mode",
        type=str,
        default="diffusion_decay",
        choices=["diffusion_decay", "advection_diffusion"],
    )
    parser.add_argument("--distance_scale_km", type=float, default=50.0)
    parser.add_argument("--wind_graph_norm", type=str, default="row", choices=["none", "row"])
    parser.add_argument("--lambda_nonnegative", type=float, default=0.1)
    parser.add_argument("--lambda_temporal_smooth", type=float, default=0.01)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--max_test_batches", type=int, default=None)

    parser.add_argument("--save_dir", type=str, default="./results/wind_beijing")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/wind_beijing")
    parser.add_argument("--model_path", type=str, default="best_model.pth")
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def setup_logging():
    log_dir = Path.cwd() / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"train_final_wind_beijing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.info("log file: %s", log_file)
    return logger


def load_beijing_feature_metadata(root_path: str, metadata_name: str) -> dict[str, Any]:
    path = Path(root_path) / metadata_name
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict metadata in {path}")
    return payload


def validate_args(args) -> None:
    if args.hist_len != args.seq_len:
        raise ValueError(f"Beijing requires hist_len == seq_len. Got hist_len={args.hist_len}, seq_len={args.seq_len}")
    if args.features != "MS":
        raise ValueError("Beijing script requires features='MS'.")
    if args.target_idx != EXPECTED_TARGET_IDX:
        raise ValueError(f"Beijing script expects target_idx={EXPECTED_TARGET_IDX}, got {args.target_idx}")
    if args.wind_u_idx != EXPECTED_WIND_U_IDX or args.wind_v_idx != EXPECTED_WIND_V_IDX:
        raise ValueError(
            f"Beijing script expects raw wind indices ({EXPECTED_WIND_U_IDX}, {EXPECTED_WIND_V_IDX}), "
            f"got ({args.wind_u_idx}, {args.wind_v_idx})"
        )
    if args.pred_len != 24:
        raise ValueError("Beijing script expects pred_len=24 for 72h evaluation.")
    if args.time_interval_hours != 3.0:
        raise ValueError(f"Beijing requires time_interval_hours=3.0, got {args.time_interval_hours}")
    if args.max_train_batches == 0 or args.max_val_batches == 0 or args.max_test_batches == 0:
        raise ValueError("0 means no batches; use None for full runs or positive integer for smoke test.")

    metadata = load_beijing_feature_metadata(args.root_path, args.feature_metadata_path)
    feature_names = metadata.get("feature_names")
    if feature_names != EXPECTED_FEATURE_NAMES:
        raise ValueError(f"Unexpected Beijing feature_names: {feature_names}")
    if int(metadata.get("target_idx", -1)) != args.target_idx:
        raise ValueError(f"Metadata target_idx {metadata.get('target_idx')} does not match args.target_idx {args.target_idx}")
    if int(metadata.get("wind_u_idx", -1)) != args.wind_u_idx or int(metadata.get("wind_v_idx", -1)) != args.wind_v_idx:
        raise ValueError(
            f"Metadata wind indices ({metadata.get('wind_u_idx')}, {metadata.get('wind_v_idx')}) "
            f"do not match args ({args.wind_u_idx}, {args.wind_v_idx})"
        )


def prepare_dataset(flag: str, args) -> Dataset_Beijing_ForWind:
    dataset = Dataset_Beijing_ForWind(
        root_path=args.root_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target_idx=args.target_idx,
        scale=True,
        data_path=args.data_path,
        station_path=args.station_path,
        adj_geo_path=args.adj_geo_path,
        graph_npz_path=args.graph_npz_path,
        metadata_path=args.feature_metadata_path,
        timestamps_path=args.timestamps_path,
        time_feature_mode=args.time_feature_mode,
        wind_u_idx=args.wind_u_idx,
        wind_v_idx=args.wind_v_idx,
        coord_order=args.coord_order,
    )
    dataset.feature_metadata_path = str(Path(args.root_path) / args.feature_metadata_path)
    dataset.wind_resolution = "from_beijing_processed_metadata"
    dataset.observable_dyn_mode = args.observable_dyn_mode
    return dataset


def prepare_loader(dataset, args, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=custom_collate_fn_beijing_physics_first,
    )


def log_split_summary(logger, dataset: Dataset_Beijing_ForWind, split_name: str) -> None:
    split_def = BEIJING_SPLITS[dataset.flag]
    logger.info(
        "[%s] split date range: %s -> %s",
        split_name,
        split_def.start_datetime,
        split_def.end_datetime,
    )
    logger.info(
        "[%s] split index range: %d -> %d (%s -> %s)",
        split_name,
        dataset.split_start_idx,
        dataset.split_end_idx,
        dataset.datetime_strings[dataset.split_start_idx],
        dataset.datetime_strings[dataset.split_end_idx],
    )
    logger.info("[%s] sample count: %d", split_name, len(dataset))
    if len(dataset) > 0:
        first_window = dataset.get_sample_window(0)
        last_window = dataset.get_sample_window(len(dataset) - 1)
        logger.info(
            "[%s] first sample: hist=%s -> %s, future=%s -> %s",
            split_name,
            first_window["hist_start"],
            first_window["hist_end"],
            first_window["future_start"],
            first_window["future_end"],
        )
        logger.info(
            "[%s] last sample: hist=%s -> %s, future=%s -> %s",
            split_name,
            last_window["hist_start"],
            last_window["hist_end"],
            last_window["future_start"],
            last_window["future_end"],
        )


def log_dataset_debug(logger, dataset: Dataset_Beijing_ForWind, split_name: str) -> None:
    logger.info("[%s] Beijing.npy shape=%s", split_name, (dataset.total_time_steps, dataset.num_nodes, dataset.num_features))
    logger.info("[%s] target_idx=%d", split_name, dataset.target_idx)
    logger.info("[%s] feature_names=%s", split_name, dataset.feature_names)
    logger.info("[%s] meteo_cols=%s", split_name, dataset.meteo_cols)
    logger.info("[%s] target_idx in meteo_cols? %s", split_name, dataset.target_idx in dataset.meteo_cols)
    logger.info("[%s] coords.shape=%s first3=%s", split_name, tuple(dataset.coords.shape), dataset.coords[:3].tolist())
    logger.info("[%s] adj_geo.shape=%s", split_name, tuple(dataset.adj_geo.shape))
    logger.info("[%s] scaler target mean/std=%.6f / %.6f", split_name, dataset.target_mean, dataset.target_std)
    logger.info("[%s] feature_metadata_path=%s", split_name, getattr(dataset, "feature_metadata_path", None))
    logger.info("[%s] wind_source=%s resolved_from=%s", split_name, getattr(dataset, "wind_source", None), getattr(dataset, "wind_resolution", None))
    logger.info("[%s] wind_u_idx_raw=%s wind_v_idx_raw=%s", split_name, getattr(dataset, "wind_u_idx_raw", None), getattr(dataset, "wind_v_idx_raw", None))
    logger.info("[%s] wind_u_meteo_pos=%s wind_v_meteo_pos=%s", split_name, dataset.wind_u_meteo_pos, dataset.wind_v_meteo_pos)
    logger.info("[%s] coord_order=%s", split_name, getattr(dataset, "coord_order", None))


def build_model(args, dataset: Dataset_Beijing_ForWind, device: torch.device):
    sample = dataset[0]
    coords_lonlat = dataset.coords.clone().float()
    if args.coord_order == "latlon":
        coords_lonlat = torch.stack([coords_lonlat[:, 1], coords_lonlat[:, 0]], dim=-1)
    model = xLSTM_WindDualODE_Mixer(
        pred_len=args.pred_len,
        seq_len=args.seq_len,
        enc_in=sample["x_enc"].shape[-1],
        dec_out=sample["future_y"].shape[-1],
        num_nodes=sample["x_enc"].shape[1],
        static_feat_dim=0,
        d_model=args.d_model,
        xlstm_num_blocks=args.xlstm_num_blocks,
        xlstm_num_heads=8,
        xlstm_dropout=args.dropout,
        channel_mixer_hidden=args.channel_mixer_hidden,
        use_static_context=False,
        use_spatial_mixing=True,
        fusion_type="attention",
        use_sparse=False,
        use_phy_ode=args.use_phy_ode,
        use_unk_ode=args.use_unk_ode,
        use_adaptive_gating=args.use_adaptive_gating,
        gating_hidden_dim=args.gating_hidden_dim,
        latent_dim=args.latent_dim,
        ode_hidden_dim=64,
        adj_mx=sample["adj_geo"].clone().float(),
        adj_geo=sample["adj_geo"].clone().float(),
        adj_poi=None,
        adj_land=None,
        coords=coords_lonlat,
        wind_u_idx=dataset.wind_u_meteo_pos,
        wind_v_idx=dataset.wind_v_meteo_pos,
        use_wind_advection=args.use_wind_advection,
        distance_scale_km=args.distance_scale_km,
        coords_are_latlon=True,
        use_geo_mask_for_wind=True,
        wind_graph_norm=args.wind_graph_norm,
        require_xlstm_backend=args.require_xlstm_backend,
        dt_hours=args.time_interval_hours,
        latent_dt_mode=args.latent_dt_mode,
    ).to(device)
    return model


def build_loss(args, dataset: Dataset_Beijing_ForWind, device: torch.device):
    observable_dynamics = None
    coords_lonlat = dataset.coords.clone().float()
    if args.coord_order == "latlon":
        coords_lonlat = torch.stack([coords_lonlat[:, 1], coords_lonlat[:, 0]], dim=-1)
    if args.use_observable_dyn_loss and args.lambda_observable_dyn > 0.0:
        if args.observable_dyn_mode == "advection_diffusion":
            wind_u_idx = int(dataset.wind_u_meteo_pos)
            wind_v_idx = int(dataset.wind_v_meteo_pos)
        else:
            wind_u_idx = 0
            wind_v_idx = 1
        observable_dynamics = ObservableDynamicsConsistency(
            adj_geo=dataset.adj_geo.clone().float(),
            coords=coords_lonlat,
            target_mean=dataset.target_mean,
            target_std=dataset.target_std,
            wind_u_idx=wind_u_idx,
            wind_v_idx=wind_v_idx,
            distance_scale_km=args.distance_scale_km,
            coord_order=args.coord_order,
            use_geo_mask_for_wind=True,
            dt_hours=args.time_interval_hours,
            observable_dyn_mode=args.observable_dyn_mode,
            wind_graph_norm=args.wind_graph_norm,
        ).to(device)
    return KnowAirWindLoss(
        pred_len=args.pred_len,
        lambda_nonnegative=args.lambda_nonnegative,
        lambda_temporal_smooth=args.lambda_temporal_smooth,
        lambda_observable_dyn=args.lambda_observable_dyn,
        target_mean=dataset.target_mean,
        target_std=dataset.target_std,
        observable_dynamics=observable_dynamics,
    ).to(device)


def build_horizon_report(pred_real: np.ndarray, true_real: np.ndarray) -> dict[str, float]:
    report: dict[str, float] = {}
    for prefix, (start, end) in BEIJING_SEGMENTS.items():
        end = min(end, pred_real.shape[1])
        if start < end:
            report.update(flatten_metric_fields(prefix, calculate_metrics(pred_real[:, start:end], true_real[:, start:end])))
    return report


def log_horizon_report(logger, split_name: str, metrics: dict[str, float]) -> None:
    for prefix in BEIJING_SEGMENTS:
        logger.info(
            "[%s] %s MAE=%.4f RMSE=%.4f MAPE=%.2f%% WMAPE=%.2f%% R2=%.4f SMAPE=%.2f%%",
            split_name,
            prefix,
            metrics[f"{prefix}_MAE"],
            metrics[f"{prefix}_RMSE"],
            metrics[f"{prefix}_MAPE"],
            metrics[f"{prefix}_WMAPE"],
            metrics[f"{prefix}_R2"],
            metrics[f"{prefix}_SMAPE"],
        )


def find_checkpoint_path(args) -> Path:
    candidates = []
    if args.checkpoint:
        candidates.append(Path(args.checkpoint))
    candidates.append(Path(args.save_dir) / "best_model.pth")
    candidates.append(Path(args.checkpoints) / Path(args.model_path).name)
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find checkpoint. Tried: {[str(p) for p in candidates]}")


def save_best_checkpoint(model: torch.nn.Module, args) -> Path:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoints)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / "best_model.pth"
    torch.save(model.state_dict(), best_model_path)
    mirror_path = checkpoint_dir / Path(args.model_path).name
    if mirror_path != best_model_path:
        shutil.copyfile(best_model_path, mirror_path)
    return best_model_path


def convert_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    x_enc = batch["x_enc"].to(device)
    future_y = batch["future_y"].to(device)
    meteo_future = batch["meteo_future"].to(device)
    meteo_future_raw = batch["meteo_future_raw"].to(device)
    return {
        "x_enc": x_enc,
        "future_y": future_y,
        "seq_y": batch["seq_y"].to(device) if batch["seq_y"] is not None else None,
        "meteo_future": meteo_future,
        "meteo_future_raw": meteo_future_raw,
        "coords": batch["coords"],
        "adj_geo": batch["adj_geo"],
    }


def log_single_batch_debug(logger, batch_info: dict[str, Any], output: torch.Tensor, dataset: Dataset_Beijing_ForWind) -> None:
    logger.info("batch shape x_enc=%s", tuple(batch_info["x_enc"].shape))
    logger.info("batch shape future_y=%s", tuple(batch_info["future_y"].shape))
    logger.info("batch shape seq_y=%s", tuple(batch_info["seq_y"].shape) if batch_info["seq_y"] is not None else None)
    logger.info("batch shape meteo_future=%s", tuple(batch_info["meteo_future"].shape))
    logger.info("batch shape coords=%s", tuple(batch_info["coords"].shape))
    logger.info("batch shape adj_geo=%s", tuple(batch_info["adj_geo"].shape))
    logger.info("output shape=%s", tuple(output.shape))
    logger.info("target_idx=%d", dataset.target_idx)
    logger.info("meteo_future excludes PM2.5=%s", dataset.target_idx not in dataset.meteo_cols)
    logger.info("meteo_cols=%s", dataset.meteo_cols)
    logger.info("scaler target mean/std=%.6f / %.6f", dataset.target_mean, dataset.target_std)


def collect_eval(model, loader, dataset, args, criterion, device, max_batches: int | None = None):
    model.eval()
    losses = []
    preds_real = []
    trues_real = []
    preds_norm = []
    trues_norm = []
    breakdown_sum = {"pred": 0.0, "L_nonneg": 0.0, "L_smooth": 0.0, "L_dyn": 0.0}
    last_debug = {"latent": {}, "observable": {}}

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch_info = convert_batch(batch, device)
            meteo_for_model = batch_info["meteo_future_raw"] if args.use_wind_advection else None
            meteo_for_dyn = (
                batch_info["meteo_future_raw"]
                if (
                    args.use_observable_dyn_loss
                    and args.lambda_observable_dyn > 0.0
                    and args.observable_dyn_mode == "advection_diffusion"
                )
                else None
            )
            output, latent_debug = model.forecast(
                x_enc=batch_info["x_enc"],
                static_feat=None,
                target_idx=args.target_idx,
                meteo_future=meteo_for_model,
                return_debug=True,
            )
            loss_dict = criterion(
                output=output,
                target=batch_info["future_y"],
                x_enc_norm=batch_info["x_enc"],
                meteo_future=meteo_for_dyn,
                target_idx=args.target_idx,
            )
            losses.append(float(loss_dict["total"].item()))
            for key in breakdown_sum:
                breakdown_sum[key] += float(loss_dict[key].item())
            preds_real.append(loss_dict["output_real"].cpu().numpy())
            trues_real.append(loss_dict["target_real"].cpu().numpy())
            preds_norm.append(loss_dict["output_norm"].cpu().numpy())
            trues_norm.append(loss_dict["target_norm"].cpu().numpy())
            last_debug = {"latent": latent_debug, "observable": loss_dict["dyn_debug"]}

    if not preds_real:
        raise ValueError("No evaluation batches were processed.")

    preds_real_np = np.concatenate(preds_real, axis=0)
    trues_real_np = np.concatenate(trues_real, axis=0)
    preds_norm_np = np.concatenate(preds_norm, axis=0)
    trues_norm_np = np.concatenate(trues_norm, axis=0)
    breakdown_avg = {key: value / len(losses) for key, value in breakdown_sum.items()}
    metrics = build_horizon_report(preds_real_np, trues_real_np)
    return float(np.mean(losses)), breakdown_avg, preds_real_np, trues_real_np, preds_norm_np, trues_norm_np, metrics, last_debug


def log_wind_debug(logger, dataset: Dataset_Beijing_ForWind, meteo_future: torch.Tensor, debug: dict[str, Any], raw_or_normalized: str) -> None:
    if dataset.wind_u_meteo_pos is None or dataset.wind_v_meteo_pos is None:
        return
    wind_u = meteo_future[..., dataset.wind_u_meteo_pos]
    wind_v = meteo_future[..., dataset.wind_v_meteo_pos]
    logger.info("wind_u_idx=%s wind_v_idx=%s", dataset.wind_u_idx, dataset.wind_v_idx)
    logger.info("wind_u_meteo_pos=%s wind_v_meteo_pos=%s", dataset.wind_u_meteo_pos, dataset.wind_v_meteo_pos)
    logger.info("wind_u_name=%s wind_v_name=%s wind_source=%s", dataset.wind_u_name, dataset.wind_v_name, dataset.wind_source)
    logger.info(
        "wind_u min/max/mean/std=%.6f/%.6f/%.6f/%.6f",
        float(wind_u.min().item()),
        float(wind_u.max().item()),
        float(wind_u.mean().item()),
        float(wind_u.std().item()),
    )
    logger.info(
        "wind_v min/max/mean/std=%.6f/%.6f/%.6f/%.6f",
        float(wind_v.min().item()),
        float(wind_v.max().item()),
        float(wind_v.mean().item()),
        float(wind_v.std().item()),
    )
    wind_speed = torch.sqrt(wind_u ** 2 + wind_v ** 2)
    logger.info(
        "wind_speed min/max/mean/std=%.6f/%.6f/%.6f/%.6f",
        float(wind_speed.min().item()),
        float(wind_speed.max().item()),
        float(wind_speed.mean().item()),
        float(wind_speed.std().item()),
    )
    logger.info("raw_or_normalized=%s", raw_or_normalized)
    logger.info("A_w symmetry error=%s", debug.get("symmetry_error", debug.get("A_w_symmetry_error", 0.0)))
    logger.info(
        "A_w mean/max/row_sum_mean/row_sum_max=%s/%s/%s/%s",
        debug.get("A_w_mean", 0.0),
        debug.get("A_w_max", 0.0),
        debug.get("A_w_row_sum_mean", 0.0),
        debug.get("A_w_row_sum_max", 0.0),
    )


def log_debug_diagnostics(logger, debug: dict[str, Any], loss_dict: dict[str, Any] | None = None) -> None:
    logger.info(
        "ODE debug alpha_mean/min/max=%.6f/%.6f/%.6f diffusion_abs_mean=%.6f advection_abs_mean=%.6f "
        "reaction_abs_mean=%.6f decay_abs_mean=%.6f dz_phy_norm=%.6f dz_data_norm=%.6f dz_fused_norm=%.6f",
        float(debug.get("alpha_mean", 0.0)),
        float(debug.get("alpha_min", 0.0)),
        float(debug.get("alpha_max", 0.0)),
        float(debug.get("diffusion_abs_mean", 0.0)),
        float(debug.get("advection_abs_mean", debug.get("adv_abs_mean", 0.0))),
        float(debug.get("reaction_abs_mean", 0.0)),
        float(debug.get("decay_abs_mean", 0.0)),
        float(debug.get("dz_phy_norm", 0.0)),
        float(debug.get("dz_data_norm", 0.0)),
        float(debug.get("dz_fused_norm", 0.0)),
    )
    logger.info(
        "ODE debug A_w_mean/max/row_sum_mean/row_sum_max=%.6f/%.6f/%.6f/%.6f k_diff=%.6f k_adv=%.6f gamma=%.6f",
        float(debug.get("A_w_mean", 0.0)),
        float(debug.get("A_w_max", 0.0)),
        float(debug.get("A_w_row_sum_mean", 0.0)),
        float(debug.get("A_w_row_sum_max", 0.0)),
        float(debug.get("k_diff", 0.0)),
        float(debug.get("k_adv", 0.0)),
        float(debug.get("gamma", 0.0)),
    )
    if loss_dict is not None:
        l_pred = float(loss_dict["pred"].item())
        l_dyn = float(loss_dict["L_dyn"].item())
        l_dyn_weighted = float(loss_dict["L_dyn_weighted"].item())
        logger.info("loss debug L_pred=%.6f L_dyn_raw=%.6f lambda*L_dyn=%.6f", l_pred, l_dyn, l_dyn_weighted)


def train_model(args, logger):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)
    logger.info("latent_dt_mode=%s", args.latent_dt_mode)
    logger.info("time_interval_hours=%s", args.time_interval_hours)
    logger.info("use_wind_advection=%s", args.use_wind_advection)
    logger.info("use_observable_dyn_loss=%s", args.use_observable_dyn_loss)
    logger.info("observable_dyn_mode=%s", args.observable_dyn_mode)

    train_dataset = prepare_dataset("train", args)
    val_dataset = prepare_dataset("val", args)
    test_dataset = prepare_dataset("test", args)

    log_split_summary(logger, train_dataset, "train")
    log_split_summary(logger, val_dataset, "val")
    log_split_summary(logger, test_dataset, "test")
    log_dataset_debug(logger, train_dataset, "train")

    train_loader = prepare_loader(train_dataset, args, shuffle=True)
    val_loader = prepare_loader(val_dataset, args, shuffle=False)
    logger.info("train batches per epoch=%d", len(train_loader))

    model = build_model(args, train_dataset, device)
    criterion = build_loss(args, train_dataset, device)
    optimizer = optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    logger.info("XLSTM_BACKEND_AVAILABLE=%s", XLSTM_BACKEND_AVAILABLE)
    logger.info("require_xlstm_backend=%s", args.require_xlstm_backend)
    logger.info("context_extractor_type=%s", getattr(model, "context_extractor_type", None))

    sanity_batch = convert_batch(next(iter(train_loader)), device)
    sanity_output, sanity_debug = model.forecast(
        x_enc=sanity_batch["x_enc"],
        static_feat=None,
        target_idx=args.target_idx,
        meteo_future=sanity_batch["meteo_future_raw"] if args.use_wind_advection else None,
        return_debug=True,
    )
    log_single_batch_debug(logger, sanity_batch, sanity_output, train_dataset)
    logger.info("sanity latent_dt=%s", sanity_debug.get("latent_dt"))
    if args.use_wind_advection or (args.use_observable_dyn_loss and args.observable_dyn_mode == "advection_diffusion"):
        log_wind_debug(logger, train_dataset, sanity_batch["meteo_future_raw"], sanity_debug, "raw_beijing_meteo_excluding_pm25")
        log_wind_debug(logger, train_dataset, sanity_batch["meteo_future"], sanity_debug, "normalized_beijing_meteo_excluding_pm25")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss_total = 0.0
        train_breakdown = {"pred": 0.0, "L_nonneg": 0.0, "L_smooth": 0.0, "L_dyn": 0.0}
        train_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break
            batch_info = convert_batch(batch, device)
            optimizer.zero_grad()
            output = model.forecast(
                x_enc=batch_info["x_enc"],
                static_feat=None,
                target_idx=args.target_idx,
                meteo_future=batch_info["meteo_future_raw"] if args.use_wind_advection else None,
            )
            loss_dict = criterion(
                output=output,
                target=batch_info["future_y"],
                x_enc_norm=batch_info["x_enc"],
                meteo_future=(
                    batch_info["meteo_future_raw"]
                    if (
                        args.use_observable_dyn_loss
                        and args.lambda_observable_dyn > 0.0
                        and args.observable_dyn_mode == "advection_diffusion"
                    )
                    else None
                ),
                target_idx=args.target_idx,
            )
            loss = loss_dict["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(criterion.parameters()), max_norm=1.0)
            optimizer.step()

            train_loss_total += float(loss.item())
            for key in train_breakdown:
                train_breakdown[key] += float(loss_dict[key].item())
            train_steps += 1

        scheduler.step()
        train_steps = max(train_steps, 1)
        train_loss_avg = train_loss_total / train_steps
        train_breakdown = {key: value / train_steps for key, value in train_breakdown.items()}

        val_loss, val_breakdown, val_preds_real, val_trues_real, _val_preds_norm, _val_trues_norm, val_metrics, val_debug = collect_eval(
            model=model,
            loader=val_loader,
            dataset=val_dataset,
            args=args,
            criterion=criterion,
            device=device,
            max_batches=args.max_val_batches,
        )

        logger.info("=" * 80)
        logger.info("epoch %d/%d", epoch + 1, args.epochs)
        logger.info(
            "train loss=%.6f pred=%.6f L_nonneg=%.6f L_smooth=%.6f L_dyn=%.6f",
            train_loss_avg,
            train_breakdown["pred"],
            train_breakdown["L_nonneg"],
            train_breakdown["L_smooth"],
            train_breakdown["L_dyn"],
        )
        logger.info(
            "val loss=%.6f pred=%.6f L_nonneg=%.6f L_smooth=%.6f L_dyn=%.6f",
            val_loss,
            val_breakdown["pred"],
            val_breakdown["L_nonneg"],
            val_breakdown["L_smooth"],
            val_breakdown["L_dyn"],
        )
        log_debug_diagnostics(logger, val_debug["latent"])
        if args.use_observable_dyn_loss and args.lambda_observable_dyn > 0.0:
            log_debug_diagnostics(logger, val_debug["observable"])
        log_horizon_report(logger, "val", val_metrics)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = save_best_checkpoint(model, args)
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            np.save(save_dir / "pred_val_original.npy", val_preds_real)
            np.save(save_dir / "true_val_original.npy", val_trues_real)
            write_metrics_csv(save_dir / "best_metrics.csv", val_metrics)
            logger.info("best model updated -> %s", best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("early stopping triggered")
                break


def test_model(args, logger):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)
    logger.info("latent_dt_mode=%s", args.latent_dt_mode)
    logger.info("time_interval_hours=%s", args.time_interval_hours)
    logger.info("use_wind_advection=%s", args.use_wind_advection)
    logger.info("use_observable_dyn_loss=%s", args.use_observable_dyn_loss)
    logger.info("observable_dyn_mode=%s", args.observable_dyn_mode)

    train_dataset = prepare_dataset("train", args)
    test_dataset = prepare_dataset("test", args)
    log_split_summary(logger, train_dataset, "train")
    log_split_summary(logger, test_dataset, "test")

    test_loader = prepare_loader(test_dataset, args, shuffle=False)
    model = build_model(args, train_dataset, device)
    criterion = build_loss(args, train_dataset, device)
    checkpoint_path = find_checkpoint_path(args)
    logger.info("loading checkpoint: %s", checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)

    test_loss, test_breakdown, preds_real, trues_real, preds_norm, trues_norm, metrics, debug = collect_eval(
        model=model,
        loader=test_loader,
        dataset=test_dataset,
        args=args,
        criterion=criterion,
        device=device,
        max_batches=args.max_test_batches,
    )

    logger.info("=" * 80)
    logger.info(
        "test loss=%.6f pred=%.6f L_nonneg=%.6f L_smooth=%.6f L_dyn=%.6f",
        test_loss,
        test_breakdown["pred"],
        test_breakdown["L_nonneg"],
        test_breakdown["L_smooth"],
        test_breakdown["L_dyn"],
    )
    log_debug_diagnostics(logger, debug["latent"])
    if args.use_observable_dyn_loss and args.lambda_observable_dyn > 0.0:
        log_debug_diagnostics(logger, debug["observable"])
    log_horizon_report(logger, "test", metrics)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "pred_test_original.npy", preds_real)
    np.save(save_dir / "true_test_original.npy", trues_real)
    np.save(save_dir / "pred_test_normalized.npy", preds_norm)
    np.save(save_dir / "true_test_normalized.npy", trues_norm)
    write_metrics_csv(save_dir / "test_metrics.csv", metrics)
    logger.info("pred_test_original.npy shape=%s", tuple(preds_real.shape))


def smoke_case_args(base_args, **updates):
    cloned = deepcopy(base_args)
    for key, value in updates.items():
        setattr(cloned, key, value)
    return cloned


def run_smoke_case(case_name: str, case_args, logger, train_dataset, train_loader, device: torch.device) -> None:
    logger.info("-" * 80)
    logger.info("smoke case: %s", case_name)
    logger.info(
        "settings: use_wind_advection=%s use_observable_dyn_loss=%s observable_dyn_mode=%s lambda_observable_dyn=%.6f require_xlstm_backend=%s",
        case_args.use_wind_advection,
        case_args.use_observable_dyn_loss,
        case_args.observable_dyn_mode,
        case_args.lambda_observable_dyn,
        case_args.require_xlstm_backend,
    )
    model = build_model(case_args, train_dataset, device)
    criterion = build_loss(case_args, train_dataset, device)
    batch = convert_batch(next(iter(train_loader)), device)

    if train_dataset.meteo_cols != [1, 2, 3, 4, 5]:
        raise AssertionError(f"Unexpected meteo_cols: {train_dataset.meteo_cols}")
    if train_dataset.wind_u_meteo_pos != EXPECTED_WIND_U_METEO_POS or train_dataset.wind_v_meteo_pos != EXPECTED_WIND_V_METEO_POS:
        raise AssertionError(
            f"Unexpected wind meteo positions: {train_dataset.wind_u_meteo_pos}, {train_dataset.wind_v_meteo_pos}"
        )
    if batch["meteo_future_raw"].shape[-1] != 5:
        raise AssertionError(f"meteo_future_raw last dim should be 5, got {batch['meteo_future_raw'].shape[-1]}")

    output, debug = model.forecast(
        x_enc=batch["x_enc"],
        static_feat=None,
        target_idx=case_args.target_idx,
        meteo_future=batch["meteo_future_raw"] if case_args.use_wind_advection else None,
        return_debug=True,
    )
    if tuple(output.shape[1:]) != (24, 35, 1):
        raise AssertionError(f"Unexpected output shape: {tuple(output.shape)}")

    dyn_meteo = (
        batch["meteo_future_raw"]
        if (
            case_args.use_observable_dyn_loss
            and case_args.lambda_observable_dyn > 0.0
            and case_args.observable_dyn_mode == "advection_diffusion"
        )
        else None
    )
    loss_dict = criterion(
        output=output,
        target=batch["future_y"],
        x_enc_norm=batch["x_enc"],
        meteo_future=dyn_meteo,
        target_idx=case_args.target_idx,
    )
    logger.info("batch x_enc=%s future_y=%s meteo_future=%s", tuple(batch["x_enc"].shape), tuple(batch["future_y"].shape), tuple(batch["meteo_future"].shape))
    logger.info("output shape=%s loss total=%.6f pred=%.6f L_dyn=%.6f", tuple(output.shape), float(loss_dict["total"]), float(loss_dict["pred"]), float(loss_dict["L_dyn"]))
    logger.info("meteo_future_raw excludes PM2.5=%s", train_dataset.target_idx not in train_dataset.meteo_cols)
    wind_u = batch["meteo_future_raw"][..., train_dataset.wind_u_meteo_pos]
    wind_v = batch["meteo_future_raw"][..., train_dataset.wind_v_meteo_pos]
    logger.info(
        "wind_u raw min/max/mean/std=%.6f/%.6f/%.6f/%.6f",
        float(wind_u.min()), float(wind_u.max()), float(wind_u.mean()), float(wind_u.std())
    )
    logger.info(
        "wind_v raw min/max/mean/std=%.6f/%.6f/%.6f/%.6f",
        float(wind_v.min()), float(wind_v.max()), float(wind_v.mean()), float(wind_v.std())
    )
    log_single_batch_debug(logger, batch, output, train_dataset)
    if case_args.use_wind_advection or (case_args.use_observable_dyn_loss and case_args.observable_dyn_mode == "advection_diffusion"):
        log_wind_debug(logger, train_dataset, batch["meteo_future_raw"], debug if case_args.use_wind_advection else loss_dict["dyn_debug"], "raw_beijing_meteo_excluding_pm25")


def smoke_test(args, logger):
    smoke_args = deepcopy(args)
    if not XLSTM_BACKEND_AVAILABLE and smoke_args.require_xlstm_backend:
        smoke_args.require_xlstm_backend = False
        logger.warning("xLSTM backend is unavailable locally; smoke test is using require_xlstm_backend=false (GRU fallback).")

    device = torch.device(smoke_args.device if torch.cuda.is_available() else "cpu")
    logger.info("smoke device: %s", device)

    train_dataset = prepare_dataset("train", smoke_args)
    val_dataset = prepare_dataset("val", smoke_args)
    test_dataset = prepare_dataset("test", smoke_args)
    log_split_summary(logger, train_dataset, "train")
    log_split_summary(logger, val_dataset, "val")
    log_split_summary(logger, test_dataset, "test")
    log_dataset_debug(logger, train_dataset, "train")

    train_loader = prepare_loader(train_dataset, smoke_args, shuffle=False)
    val_loader = prepare_loader(val_dataset, smoke_args, shuffle=False)
    test_loader = prepare_loader(test_dataset, smoke_args, shuffle=False)
    logger.info("loader batches train/val/test=%d/%d/%d", len(train_loader), len(val_loader), len(test_loader))

    cases = [
        ("no_wind_no_dyn", smoke_case_args(smoke_args, use_wind_advection=False, use_observable_dyn_loss=False, lambda_observable_dyn=0.0, observable_dyn_mode="diffusion_decay")),
        ("no_wind_diffusion_decay", smoke_case_args(smoke_args, use_wind_advection=False, use_observable_dyn_loss=True, lambda_observable_dyn=0.1, observable_dyn_mode="diffusion_decay")),
        ("no_wind_advection_diffusion", smoke_case_args(smoke_args, use_wind_advection=False, use_observable_dyn_loss=True, lambda_observable_dyn=0.1, observable_dyn_mode="advection_diffusion")),
        ("wind_only", smoke_case_args(smoke_args, use_wind_advection=True, use_observable_dyn_loss=False, lambda_observable_dyn=0.0, observable_dyn_mode="diffusion_decay")),
        ("wind_plus_observable_advection_diffusion", smoke_case_args(smoke_args, use_wind_advection=True, use_observable_dyn_loss=True, lambda_observable_dyn=0.1, observable_dyn_mode="advection_diffusion")),
    ]
    for case_name, case_args in cases:
        run_smoke_case(case_name, case_args, logger, train_dataset, train_loader, device)
    logger.info("smoke test passed for all cases.")


def main():
    args = parse_args()
    validate_args(args)
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("PMO-Net wind Beijing branch")
    logger.info("=" * 80)
    logger.info("config: %s", vars(args))

    if args.mode == "train":
        train_model(args, logger)
    elif args.mode == "test":
        test_model(args, logger)
    else:
        smoke_test(args, logger)


if __name__ == "__main__":
    main()
