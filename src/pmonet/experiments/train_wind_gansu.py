from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pmonet.data.gansu import (  # noqa: E402
    Dataset_RIMST_PhysicsFirst,
    custom_collate_fn_physics_first,
)
from pmonet.models.pmonet import (  # noqa: E402
    XLSTM_BACKEND_AVAILABLE,
    xLSTM_WindDualODE_Mixer,
)


HOURLY_HORIZON_STEPS = {
    "3h": 2,
    "6h": 5,
    "12h": 11,
    "24h": 23,
}

HOURLY_HORIZON_FIELDS = {
    "3h": "h3",
    "6h": "h6",
    "12h": "h12",
    "24h": "h24",
}

REQUIRED_METEO_COLUMNS = {"u10", "v10"}


def str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value from {value!r}")


def calculate_metrics(pred, true, threshold=0.1):
    pred_flat = np.asarray(pred, dtype=np.float64).reshape(-1)
    true_flat = np.asarray(true, dtype=np.float64).reshape(-1)
    mask = np.isfinite(pred_flat) & np.isfinite(true_flat)
    pred_flat = pred_flat[mask]
    true_flat = true_flat[mask]

    if pred_flat.size == 0:
        return {"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "WMAPE": np.nan, "R2": np.nan}

    mae = float(np.mean(np.abs(pred_flat - true_flat)))
    mse = float(np.mean((pred_flat - true_flat) ** 2))
    rmse = float(np.sqrt(mse))
    try:
        true_mean = float(np.mean(true_flat))
        ss_res = float(np.sum((true_flat - pred_flat) ** 2))
        ss_tot = float(np.sum((true_flat - true_mean) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan
    except Exception:
        r2 = np.nan

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
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "WMAPE": wmape, "R2": r2}


def flatten_metric_fields(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {
        f"{prefix}_MAE": float(metrics["MAE"]),
        f"{prefix}_RMSE": float(metrics["RMSE"]),
        f"{prefix}_MAPE": float(metrics["MAPE"]),
        f"{prefix}_WMAPE": float(metrics["WMAPE"]),
        f"{prefix}_R2": float(metrics["R2"]),
    }


def write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


class SimpleArgs:
    def __init__(self):
        self.target_idx = 0
        self.features = "M"


class ObservableDynamicsConsistency(nn.Module):
    def __init__(
        self,
        adj_geo: torch.Tensor,
        coords: torch.Tensor,
        target_mean: float,
        target_std: float,
        wind_u_idx: int,
        wind_v_idx: int,
        distance_scale_km: float = 50.0,
        coord_order: str = "lonlat",
        use_geo_mask_for_wind: bool = True,
        wind_graph_norm: str = "row",
        dt_hours: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.wind_u_idx = wind_u_idx
        self.wind_v_idx = wind_v_idx
        self.distance_scale_km = float(distance_scale_km)
        self.coord_order = coord_order
        self.use_geo_mask_for_wind = use_geo_mask_for_wind
        self.wind_graph_norm = wind_graph_norm
        self.dt_hours = float(dt_hours)
        self.eps = float(eps)

        adj_geo = adj_geo.float()
        if adj_geo.dim() == 3:
            adj_geo = adj_geo[0]
        coords = coords.float()
        coords_xy = self._convert_coords_to_xy_km(coords)
        pairwise_dist_km, pairwise_direction = self._build_pairwise_geometry(coords_xy)
        num_nodes = coords.shape[0]
        non_self_mask = 1.0 - torch.eye(num_nodes, dtype=coords.dtype, device=coords.device)
        geo_mask = self._build_geo_mask(adj_geo, non_self_mask)

        adj_sym = 0.5 * (adj_geo + adj_geo.transpose(0, 1))
        adj_sym = adj_sym * non_self_mask
        degree = adj_sym.sum(dim=-1)
        degree_inv_sqrt = torch.pow(degree.clamp_min(self.eps), -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        d_inv_sqrt = torch.diag(degree_inv_sqrt)
        identity = torch.eye(adj_sym.shape[0], device=adj_sym.device, dtype=adj_sym.dtype)
        laplacian = identity - d_inv_sqrt @ adj_sym @ d_inv_sqrt

        self.register_buffer("pairwise_dist_km", pairwise_dist_km)
        self.register_buffer("pairwise_direction", pairwise_direction)
        self.register_buffer("geo_mask", geo_mask)
        self.register_buffer("laplacian_geo", laplacian)
        self.register_buffer("target_mean", torch.tensor(float(target_mean), dtype=torch.float32))
        self.register_buffer("target_std", torch.tensor(float(target_std), dtype=torch.float32))

        self.raw_k_diff = nn.Parameter(torch.tensor(-4.59511985013459))
        self.raw_k_adv = nn.Parameter(torch.tensor(-6.907255172729492))
        self.raw_gamma = nn.Parameter(torch.tensor(-6.907255172729492))

    def _convert_coords_to_xy_km(self, coords: torch.Tensor) -> torch.Tensor:
        if self.coord_order == "lonlat":
            lon = coords[:, 0]
            lat = coords[:, 1]
        elif self.coord_order == "latlon":
            lat = coords[:, 0]
            lon = coords[:, 1]
        else:
            raise ValueError(f"coord_order must be 'lonlat' or 'latlon', got {self.coord_order!r}")
        lat_rad = lat * np.pi / 180.0
        lon_rad = lon * np.pi / 180.0
        lat0 = lat_rad.mean()
        earth_radius_km = 6371.0
        x = earth_radius_km * torch.cos(lat0) * lon_rad
        y = earth_radius_km * lat_rad
        return torch.stack([x, y], dim=-1)

    def _build_pairwise_geometry(self, coords_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        diff = coords_xy.unsqueeze(0) - coords_xy.unsqueeze(1)
        dist = torch.norm(diff, dim=-1).clamp_min(self.eps)
        direction = diff / dist.unsqueeze(-1)
        return dist, direction

    def _build_geo_mask(self, adj_geo: torch.Tensor, non_self_mask: torch.Tensor) -> torch.Tensor:
        if not self.use_geo_mask_for_wind:
            return non_self_mask
        adj_binary = (adj_geo > 0).to(dtype=adj_geo.dtype)
        adj_binary = torch.maximum(adj_binary, adj_binary.transpose(0, 1))
        return adj_binary * non_self_mask

    def get_positive_parameters(self) -> dict[str, torch.Tensor]:
        return {
            "k_y_diff": F.softplus(self.raw_k_diff),
            "k_y_adv": F.softplus(self.raw_k_adv),
            "gamma_y": F.softplus(self.raw_gamma),
        }

    def build_wind_graph(self, meteo_t: torch.Tensor) -> torch.Tensor:
        wind = torch.stack([meteo_t[..., self.wind_u_idx], meteo_t[..., self.wind_v_idx]], dim=-1)
        projection = torch.einsum("bik,ijk->bij", wind, self.pairwise_direction)
        directional_speed = F.relu(projection)
        distance_kernel = torch.exp(-self.pairwise_dist_km / max(self.distance_scale_km, self.eps))
        a_w = self.geo_mask.unsqueeze(0) * distance_kernel.unsqueeze(0) * directional_speed
        if self.wind_graph_norm == "row":
            a_w = a_w / a_w.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return a_w

    def forward(
        self,
        x_enc_norm: torch.Tensor,
        y_pred_real: torch.Tensor,
        meteo_future: torch.Tensor,
        target_idx: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        y0_norm = x_enc_norm[:, -1:, :, target_idx : target_idx + 1]
        y0_real = y0_norm * self.target_std.view(1, 1, 1, 1) + self.target_mean.view(1, 1, 1, 1)
        y_state_real = torch.cat([y0_real, y_pred_real[:, :-1]], dim=1)
        y_next_real = y_pred_real
        d_y_dt = (y_next_real - y_state_real) / self.dt_hours

        params = self.get_positive_parameters()
        k_y_diff = params["k_y_diff"]
        k_y_adv = params["k_y_adv"]
        gamma_y = params["gamma_y"]

        diff_term = -k_y_diff * torch.einsum("ij,bhjd->bhid", self.laplacian_geo, y_state_real)
        advection_steps = []
        debug_a = []
        for step in range(meteo_future.shape[1]):
            a_w = self.build_wind_graph(meteo_future[:, step])
            y_step = y_state_real[:, step]
            inflow = torch.einsum("bji,bjd->bid", a_w, y_step)
            outflow = a_w.sum(dim=-1).unsqueeze(-1) * y_step
            advection_steps.append(k_y_adv * (inflow - outflow))
            debug_a.append(a_w)
        adv_term = torch.stack(advection_steps, dim=1)
        rhs = diff_term + adv_term - gamma_y * y_state_real

        dyn_residual = (d_y_dt - rhs) / (self.target_std.view(1, 1, 1, 1) + self.eps)
        l_dyn = (dyn_residual ** 2).mean()

        a_w_all = torch.stack(debug_a, dim=1)
        debug = {
            "A_w_shape": tuple(a_w_all.shape),
            "A_w_symmetry_error": float((a_w_all - a_w_all.transpose(-1, -2)).abs().mean().detach().cpu()),
            "A_w_mean": float(a_w_all.mean().detach().cpu()),
            "A_w_max": float(a_w_all.max().detach().cpu()),
            "A_w_row_sum_mean": float(a_w_all.sum(dim=-1).mean().detach().cpu()),
            "A_w_row_sum_max": float(a_w_all.sum(dim=-1).max().detach().cpu()),
            "diffusion_abs_mean": float(diff_term.abs().mean().detach().cpu()),
            "adv_abs_mean": float(adv_term.abs().mean().detach().cpu()),
            "reaction_abs_mean": 0.0,
            "decay_abs_mean": float((gamma_y * y_state_real).abs().mean().detach().cpu()),
            "L_dyn": float(l_dyn.detach().cpu()),
            "k_y_diff": float(k_y_diff.detach().cpu()),
            "k_y_adv": float(k_y_adv.detach().cpu()),
            "gamma_y": float(gamma_y.detach().cpu()),
        }
        return l_dyn, debug


class WindGansuLoss(nn.Module):
    def __init__(
        self,
        pred_len: int,
        lambda_nonnegative: float,
        lambda_temporal_smooth: float,
        lambda_observable_dyn: float,
        output_scaler_mean: np.ndarray,
        output_scaler_std: np.ndarray,
        target_mean: float,
        target_std: float,
        observable_dynamics: Optional[ObservableDynamicsConsistency],
    ):
        super().__init__()
        self.lambda_nonnegative = lambda_nonnegative
        self.lambda_temporal_smooth = lambda_temporal_smooth
        self.lambda_observable_dyn = lambda_observable_dyn
        self.observable_dynamics = observable_dynamics
        self.register_buffer("output_scaler_mean", torch.as_tensor(output_scaler_mean, dtype=torch.float32).view(1, 1, 1, -1))
        self.register_buffer("output_scaler_std", torch.as_tensor(output_scaler_std, dtype=torch.float32).view(1, 1, 1, -1))
        self.register_buffer("target_mean", torch.tensor(float(target_mean), dtype=torch.float32).view(1, 1, 1, 1))
        self.register_buffer("target_std", torch.tensor(float(target_std), dtype=torch.float32).view(1, 1, 1, 1))
        time_weights = torch.exp(-torch.arange(pred_len) * 0.05)
        self.register_buffer("time_weights", time_weights / time_weights.sum())

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        x_enc_norm: Optional[torch.Tensor] = None,
        meteo_future: Optional[torch.Tensor] = None,
        target_idx: Optional[int] = None,
    ) -> dict[str, Any]:
        output_norm = output
        target_norm = target
        output_real = output * self.output_scaler_std + self.output_scaler_mean
        target_real = target * self.output_scaler_std + self.output_scaler_mean

        l_pred = (((output_norm - target_norm) ** 2) * self.time_weights.view(1, -1, 1, 1)).mean()
        l_nonneg = F.relu(-output_real).mean()
        l_smooth = ((output_norm[:, 1:] - output_norm[:, :-1]) ** 2).mean() if output.shape[1] > 1 else output.new_tensor(0.0)

        if self.lambda_observable_dyn > 0.0 and self.observable_dynamics is not None:
            l_dyn, dyn_debug = self.observable_dynamics(
                x_enc_norm=x_enc_norm,
                y_pred_real=output_real,
                meteo_future=meteo_future,
                target_idx=target_idx,
            )
        else:
            l_dyn = output.new_tensor(0.0)
            dyn_debug = {"A_w_shape": None, "A_w_symmetry_error": 0.0, "A_w_mean": 0.0, "A_w_max": 0.0, "adv_abs_mean": 0.0, "L_dyn": 0.0}

        total = l_pred + self.lambda_nonnegative * l_nonneg + self.lambda_temporal_smooth * l_smooth + self.lambda_observable_dyn * l_dyn
        return {
            "total": total,
            "pred": l_pred,
            "L_nonneg": l_nonneg,
            "L_smooth": l_smooth,
            "L_dyn": l_dyn,
            "L_dyn_weighted": self.lambda_observable_dyn * l_dyn,
            "dyn_debug": dyn_debug,
            "output_real": output_real.detach(),
            "output_norm": output_norm.detach(),
            "target_real": target_real.detach(),
            "target_norm": target_norm.detach(),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Slim Gansu wind-aware PMO-Net training")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--root_path", type=str, default="./data")
    parser.add_argument("--data_path", type=str, default="Gansu_Air.csv")
    parser.add_argument("--adj_geo_path", type=str, default="knn_adj.csv")
    parser.add_argument("--poi_path", type=str, default="poi_attribute_adj.npy")
    parser.add_argument("--landuse_path", type=str, default="landuse_attribute_adj.npy")
    parser.add_argument("--coords_path", type=str, default="./data/station_coords_physics_first.npy")
    parser.add_argument("--meteo_future_path", type=str, default="./data/meteo_physics_first.npy")

    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--label_len", type=int, default=12)
    parser.add_argument("--time_interval_hours", type=float, default=1.0)
    parser.add_argument("--latent_dt_mode", type=str, default="normalized", choices=["normalized", "physical"])

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

    parser.add_argument("--features", type=str, default="MS", choices=["M", "MS"])
    parser.add_argument("--target_idx", type=int, default=0)
    parser.add_argument("--use_wind_advection", type=str2bool, default=False)
    parser.add_argument("--wind_u_idx", type=int, default=None)
    parser.add_argument("--wind_v_idx", type=int, default=None)
    parser.add_argument("--distance_scale_km", type=float, default=50.0)
    parser.add_argument("--coord_order", type=str, default="lonlat", choices=["lonlat", "latlon"])
    parser.add_argument("--wind_graph_norm", type=str, default="row", choices=["none", "row"])
    parser.add_argument("--use_observable_dyn_loss", type=str2bool, default=False)
    parser.add_argument("--lambda_observable_dyn", type=float, default=0.0)
    parser.add_argument("--lambda_nonnegative", type=float, default=0.1)
    parser.add_argument("--lambda_temporal_smooth", type=float, default=0.01)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--max_test_batches", type=int, default=None)

    parser.add_argument("--save_dir", type=str, default="./results/wind_gansu")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/wind_gansu")
    parser.add_argument("--model_path", type=str, default="best_model.pth")
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def setup_logging():
    log_dir = Path.cwd() / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"train_final_wind_gansu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"log file: {log_file}")
    return logger


def resolve_existing_sidecar_path(root_path: str, requested_path: Optional[str], default_filename: str, sibling_base: Optional[Path] = None) -> tuple[Path, list[Path]]:
    root = Path(root_path)
    candidates: list[Path] = []
    if requested_path is not None:
        requested = Path(requested_path)
        candidates.append(requested)
        if requested.exists():
            return requested.resolve(), candidates
        if not requested.is_absolute():
            rooted = root / requested
            candidates.append(rooted)
            if rooted.exists():
                return rooted.resolve(), candidates
            rooted_basename = root / requested.name
            candidates.append(rooted_basename)
            if rooted_basename.exists():
                return rooted_basename.resolve(), candidates
    if sibling_base is not None:
        sibling_candidate = sibling_base.with_name(default_filename)
        candidates.append(sibling_candidate)
        if sibling_candidate.exists():
            return sibling_candidate.resolve(), candidates
    default_candidate = root / default_filename
    candidates.append(default_candidate)
    if default_candidate.exists():
        return default_candidate.resolve(), candidates
    raise FileNotFoundError(f"Could not resolve sidecar file '{default_filename}'. Tried: {[str(path) for path in candidates]}")


def resolve_sidecar_paths(args) -> dict[str, Any]:
    meteo_path, meteo_candidates = resolve_existing_sidecar_path(args.root_path, args.meteo_future_path, "meteo_physics_first.npy")
    columns_path, _ = resolve_existing_sidecar_path(args.root_path, None, "meteo_physics_first_columns.json", sibling_base=meteo_path)
    station_order_path, _ = resolve_existing_sidecar_path(args.root_path, None, "meteo_physics_first_station_order.json", sibling_base=meteo_path)
    datetime_path, _ = resolve_existing_sidecar_path(args.root_path, None, "meteo_physics_first_datetime.csv", sibling_base=meteo_path)
    return {
        "meteo_path": meteo_path,
        "meteo_candidates": meteo_candidates,
        "columns_path": columns_path,
        "station_order_path": station_order_path,
        "datetime_path": datetime_path,
    }


def resolve_wind_indices(args) -> tuple[int, int, list[str], dict[str, Any]]:
    sidecar_paths = resolve_sidecar_paths(args)
    columns = json.loads(sidecar_paths["columns_path"].read_text(encoding="utf-8"))
    lower_to_idx = {str(name).lower(): idx for idx, name in enumerate(columns)}
    if not REQUIRED_METEO_COLUMNS.issubset(lower_to_idx.keys()):
        raise ValueError(f"Required meteo columns {sorted(REQUIRED_METEO_COLUMNS)} not found in {columns}")
    u_idx = lower_to_idx["u10"]
    v_idx = lower_to_idx["v10"]
    if args.wind_u_idx is not None and args.wind_u_idx != u_idx:
        raise ValueError(f"Requested wind_u_idx={args.wind_u_idx}, but resolved u10 index is {u_idx}")
    if args.wind_v_idx is not None and args.wind_v_idx != v_idx:
        raise ValueError(f"Requested wind_v_idx={args.wind_v_idx}, but resolved v10 index is {v_idx}")
    return u_idx, v_idx, columns, sidecar_paths


def get_target_stats(dataset, args) -> tuple[float, float]:
    return float(dataset.scaler.mean_[args.target_idx]), float(dataset.scaler.scale_[args.target_idx])


def get_output_scaler_stats(dataset, args) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(dataset.scaler.mean_, dtype=np.float32)
    std = np.asarray(dataset.scaler.scale_, dtype=np.float32)
    if args.features == "MS":
        mean = mean[args.target_idx : args.target_idx + 1]
        std = std[args.target_idx : args.target_idx + 1]
    return mean, std


def build_static_batch(poi_feat_batch, landuse_feat_batch, device):
    if poi_feat_batch is not None and landuse_feat_batch is not None:
        return torch.cat([poi_feat_batch, landuse_feat_batch], dim=-1).to(device)
    if poi_feat_batch is not None:
        return poi_feat_batch.to(device)
    if landuse_feat_batch is not None:
        return landuse_feat_batch.to(device)
    return None


def prepare_dataset(flag: str, args):
    simple_args = SimpleArgs()
    simple_args.target_idx = args.target_idx
    simple_args.features = args.features

    u_idx = None
    v_idx = None
    columns = None
    sidecar_paths = None
    require_future_meteo = args.use_wind_advection or args.use_observable_dyn_loss or args.lambda_observable_dyn > 0.0
    if require_future_meteo:
        u_idx, v_idx, columns, sidecar_paths = resolve_wind_indices(args)

    dataset = Dataset_RIMST_PhysicsFirst(
        args=simple_args,
        root_path=args.root_path,
        data_path=args.data_path,
        poi_path=args.poi_path,
        landuse_path=args.landuse_path,
        adj_geo_path=args.adj_geo_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        scale=True,
        meteo_path=str(sidecar_paths["meteo_path"]) if sidecar_paths is not None else "meteo_physics_first.npy",
        meteo_columns_path=str(sidecar_paths["columns_path"]) if sidecar_paths is not None else "meteo_physics_first_columns.json",
        meteo_station_order_path=str(sidecar_paths["station_order_path"]) if sidecar_paths is not None else "meteo_physics_first_station_order.json",
        meteo_datetime_path=str(sidecar_paths["datetime_path"]) if sidecar_paths is not None else "meteo_physics_first_datetime.csv",
        return_meteo_pair=require_future_meteo,
    )
    dataset.wind_u_meteo_pos = u_idx
    dataset.wind_v_meteo_pos = v_idx
    dataset.meteo_column_names = columns
    dataset.meteo_source_file = str(sidecar_paths["meteo_path"]) if sidecar_paths is not None else None
    dataset.meteo_is_raw_physical = bool(sidecar_paths is not None)
    return dataset


def prepare_loader(dataset, args, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=custom_collate_fn_physics_first,
    )


def load_coords(coords_path: str) -> torch.Tensor:
    coords_np = np.load(coords_path).astype(np.float32)
    return torch.from_numpy(coords_np).float()


def coords_to_lonlat(coords: torch.Tensor, coord_order: str) -> torch.Tensor:
    if coord_order == "lonlat":
        return coords
    if coord_order == "latlon":
        return torch.stack([coords[:, 1], coords[:, 0]], dim=-1)
    raise ValueError(f"coord_order must be 'lonlat' or 'latlon', got {coord_order!r}")


def log_wind_stats(logger, meteo: torch.Tensor, u_idx: int, v_idx: int, prefix: str) -> None:
    wind_u = meteo[..., u_idx]
    wind_v = meteo[..., v_idx]
    wind_speed = torch.sqrt(wind_u ** 2 + wind_v ** 2)
    logger.info(
        "%s wind_u mean/std/min/max=%.6f/%.6f/%.6f/%.6f",
        prefix,
        float(wind_u.mean().item()),
        float(wind_u.std().item()),
        float(wind_u.min().item()),
        float(wind_u.max().item()),
    )
    logger.info(
        "%s wind_v mean/std/min/max=%.6f/%.6f/%.6f/%.6f",
        prefix,
        float(wind_v.mean().item()),
        float(wind_v.std().item()),
        float(wind_v.min().item()),
        float(wind_v.max().item()),
    )
    logger.info(
        "%s wind_speed mean/std/min/max=%.6f/%.6f/%.6f/%.6f",
        prefix,
        float(wind_speed.mean().item()),
        float(wind_speed.std().item()),
        float(wind_speed.min().item()),
        float(wind_speed.max().item()),
    )


def log_debug_diagnostics(logger, debug: dict[str, Any], loss_dict: Optional[dict[str, Any]] = None) -> None:
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
        if l_pred > 0.0 and l_dyn > 10.0 * l_pred:
            logger.warning("L_dyn raw is >10x prediction loss. Recommend lowering lambda_observable_dyn.")


def log_coords_debug(logger, coords: torch.Tensor, debug: dict[str, Any]) -> None:
    logger.info("coords first_5_rows=%s", coords[:5].tolist())
    logger.info(
        "pairwise_dist_km min/mean/max=%.6f/%.6f/%.6f",
        float(debug.get("pairwise_dist_min", 0.0)),
        float(debug.get("pairwise_dist_mean", 0.0)),
        float(debug.get("pairwise_dist_max", 0.0)),
    )


def validate_args(args, logger):
    if args.max_val_batches == 0 or args.max_test_batches == 0:
        raise ValueError("0 means no batches; use None for full evaluation or positive integer for smoke test.")
    if args.use_wind_advection or args.use_observable_dyn_loss or args.lambda_observable_dyn > 0.0:
        if args.coords_path is None:
            raise ValueError("use_wind_advection/use_observable_dyn_loss requires coords_path.")
        u_idx, v_idx, columns, sidecar_paths = resolve_wind_indices(args)
        logger.info("resolved_meteo_future_path=%s", str(sidecar_paths["meteo_path"]))
        logger.info("resolved_meteo_future_path_exists=%s", sidecar_paths["meteo_path"].exists())
        logger.info("resolved wind columns=%s, u10_idx=%s, v10_idx=%s", columns, u_idx, v_idx)
    if args.latent_dt_mode == "physical":
        logger.warning(
            "latent_dt_mode=physical uses dt=time_interval_hours=%.6f instead of normalized 1/pred_len. "
            "This can strongly amplify latent ODE updates and destabilize training.",
            args.time_interval_hours,
        )


def build_model(args, dataset, coords: torch.Tensor, device: torch.device):
    sample = dataset[0]
    if len(sample) == 10:
        sample_x, sample_y, _meteo_future_norm, _meteo_future_raw, _time_feat, adj_geo, adj_poi, adj_land, poi_feat, landuse_feat = sample
    else:
        sample_x, sample_y, _meteo_future, _time_feat, adj_geo, adj_poi, adj_land, poi_feat, landuse_feat = sample
    static_feat_dim = 0
    if poi_feat is not None and landuse_feat is not None:
        static_feat_dim = poi_feat.shape[1] + landuse_feat.shape[1]
    adj_geo_np = adj_geo.numpy() if isinstance(adj_geo, torch.Tensor) else adj_geo
    adj_poi_np = adj_poi.numpy() if isinstance(adj_poi, torch.Tensor) else adj_poi
    adj_land_np = adj_land.numpy() if isinstance(adj_land, torch.Tensor) else adj_land
    coords_lonlat = coords_to_lonlat(coords, args.coord_order)
    return xLSTM_WindDualODE_Mixer(
        pred_len=args.pred_len,
        seq_len=args.seq_len,
        enc_in=sample_x.shape[2],
        dec_out=sample_y.shape[2],
        num_nodes=sample_x.shape[1],
        static_feat_dim=static_feat_dim,
        d_model=args.d_model,
        xlstm_num_blocks=args.xlstm_num_blocks,
        xlstm_num_heads=8,
        xlstm_dropout=args.dropout,
        channel_mixer_hidden=args.channel_mixer_hidden,
        use_static_context=static_feat_dim > 0,
        use_spatial_mixing=True,
        fusion_type="attention",
        use_sparse=False,
        use_phy_ode=args.use_phy_ode,
        use_unk_ode=args.use_unk_ode,
        use_adaptive_gating=args.use_adaptive_gating,
        gating_hidden_dim=args.gating_hidden_dim,
        latent_dim=args.latent_dim,
        ode_hidden_dim=64,
        adj_mx=torch.from_numpy(adj_geo_np).float(),
        adj_geo=torch.from_numpy(adj_geo_np).float(),
        adj_poi=torch.from_numpy(adj_poi_np).float() if adj_poi_np is not None else None,
        adj_land=torch.from_numpy(adj_land_np).float() if adj_land_np is not None else None,
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


def build_loss(args, dataset, adj_geo: torch.Tensor, coords: torch.Tensor, device: torch.device):
    target_mean, target_std = get_target_stats(dataset, args)
    output_scaler_mean, output_scaler_std = get_output_scaler_stats(dataset, args)
    coords_lonlat = coords_to_lonlat(coords, args.coord_order)
    observable_dynamics = None
    if args.use_observable_dyn_loss and args.lambda_observable_dyn > 0.0:
        observable_dynamics = ObservableDynamicsConsistency(
            adj_geo=adj_geo.float(),
            coords=coords_lonlat.float(),
            target_mean=target_mean,
            target_std=target_std,
            wind_u_idx=dataset.wind_u_meteo_pos,
            wind_v_idx=dataset.wind_v_meteo_pos,
            distance_scale_km=args.distance_scale_km,
            coord_order="lonlat",
            use_geo_mask_for_wind=True,
            wind_graph_norm=args.wind_graph_norm,
            dt_hours=args.time_interval_hours,
        ).to(device)
    return WindGansuLoss(
        pred_len=args.pred_len,
        lambda_nonnegative=args.lambda_nonnegative,
        lambda_temporal_smooth=args.lambda_temporal_smooth,
        lambda_observable_dyn=args.lambda_observable_dyn,
        output_scaler_mean=output_scaler_mean,
        output_scaler_std=output_scaler_std,
        target_mean=target_mean,
        target_std=target_std,
        observable_dynamics=observable_dynamics,
    ).to(device)


def build_horizon_report(pred_real: np.ndarray, true_real: np.ndarray, pred_len: int) -> dict[str, float]:
    report: dict[str, float] = {}
    report.update(flatten_metric_fields("overall_original_scale", calculate_metrics(pred_real, true_real)))
    for horizon_name, step_idx in HOURLY_HORIZON_STEPS.items():
        if step_idx >= pred_len:
            continue
        report.update(
            flatten_metric_fields(
                f"{HOURLY_HORIZON_FIELDS[horizon_name]}_original_scale",
                calculate_metrics(pred_real[:, step_idx], true_real[:, step_idx]),
            )
        )
    return report


def log_hourly_metrics(logger, split_name: str, metrics: dict[str, float], pred_len: int) -> None:
    logger.info(
        f"[{split_name}] overall original-scale MAE={metrics['overall_original_scale_MAE']:.4f} "
        f"RMSE={metrics['overall_original_scale_RMSE']:.4f} "
        f"MAPE={metrics['overall_original_scale_MAPE']:.2f}% "
        f"WMAPE={metrics['overall_original_scale_WMAPE']:.2f}% "
        f"R2={metrics['overall_original_scale_R2']:.4f}"
    )
    for horizon_name, step_idx in HOURLY_HORIZON_STEPS.items():
        prefix = HOURLY_HORIZON_FIELDS[horizon_name]
        key = f"{prefix}_original_scale_MAE"
        if key not in metrics:
            logger.warning(f"[{split_name}] skip {horizon_name} because pred_len={pred_len} < required step index {step_idx}")
            continue
        logger.info(
            f"[{split_name}] {horizon_name} original-scale MAE={metrics[f'{prefix}_original_scale_MAE']:.4f} "
            f"RMSE={metrics[f'{prefix}_original_scale_RMSE']:.4f} "
            f"MAPE={metrics[f'{prefix}_original_scale_MAPE']:.2f}% "
            f"WMAPE={metrics[f'{prefix}_original_scale_WMAPE']:.2f}% "
            f"R2={metrics[f'{prefix}_original_scale_R2']:.4f}"
        )


def find_checkpoint_path(args) -> Path:
    candidates = [Path(args.checkpoint)] if args.checkpoint else []
    candidates += [Path(args.save_dir) / "best_model.pth", Path(args.checkpoints) / Path(args.model_path).name]
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


def collect_eval(model, loader, dataset, args, criterion, device):
    if args.max_val_batches == 0 or args.max_test_batches == 0:
        raise ValueError("0 means no batches; use None for full evaluation or positive integer for smoke test.")
    model.eval()
    losses = []
    preds_real = []
    trues_real = []
    preds_norm = []
    trues_norm = []
    breakdown_sum = {"pred": 0.0, "L_nonneg": 0.0, "L_smooth": 0.0, "L_dyn": 0.0}
    limit = args.max_test_batches if args.mode == "test" else args.max_val_batches
    last_debug: dict[str, Any] = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if limit is not None and batch_idx >= limit:
                break
            if len(batch) == 10:
                seq_x, seq_y, meteo_future_norm, meteo_future_raw, _time_feat, _adj_geo, _adj_poi, _adj_land, poi_feat_batch, landuse_feat_batch = batch
            else:
                seq_x, seq_y, meteo_future_raw, _time_feat, _adj_geo, _adj_poi, _adj_land, poi_feat_batch, landuse_feat_batch = batch
                meteo_future_norm = meteo_future_raw
            seq_x = seq_x.to(device)
            seq_y = seq_y.to(device)
            meteo_future_norm = meteo_future_norm.to(device)
            meteo_future_raw = meteo_future_raw.to(device)
            static_batch = build_static_batch(poi_feat_batch, landuse_feat_batch, device)
            seq_y_target = seq_y[:, -args.pred_len :, :, :]

            output, debug = model.forecast(
                x_enc=seq_x,
                static_feat=static_batch,
                target_idx=args.target_idx if args.features == "MS" else None,
                meteo_future=meteo_future_raw if args.use_wind_advection else None,
                return_debug=True,
            )
            loss_dict = criterion(
                output=output,
                target=seq_y_target,
                x_enc_norm=seq_x,
                meteo_future=meteo_future_raw if (args.use_observable_dyn_loss and args.lambda_observable_dyn > 0.0) else None,
                target_idx=args.target_idx,
            )

            losses.append(float(loss_dict["total"].item()))
            for key in breakdown_sum:
                breakdown_sum[key] += float(loss_dict[key].item())
            preds_real.append(loss_dict["output_real"].cpu().numpy())
            trues_real.append(loss_dict["target_real"].cpu().numpy())
            preds_norm.append(loss_dict["output_norm"].cpu().numpy())
            trues_norm.append(loss_dict["target_norm"].cpu().numpy())
            last_debug = debug

    if not preds_real:
        raise ValueError("0 means no batches; use None for full evaluation or positive integer for smoke test.")

    preds_real_np = np.concatenate(preds_real, axis=0)
    trues_real_np = np.concatenate(trues_real, axis=0)
    preds_norm_np = np.concatenate(preds_norm, axis=0)
    trues_norm_np = np.concatenate(trues_norm, axis=0)

    metrics_real = build_horizon_report(preds_real_np, trues_real_np, pred_len=preds_real_np.shape[1])
    breakdown_avg = {key: value / len(losses) for key, value in breakdown_sum.items()}
    return float(np.mean(losses)), breakdown_avg, preds_real_np, trues_real_np, preds_norm_np, trues_norm_np, metrics_real, last_debug


def train_model(args, logger):
    validate_args(args, logger)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    coords = load_coords(args.coords_path)
    train_dataset = prepare_dataset("train", args)
    val_dataset = prepare_dataset("val", args)
    train_loader = prepare_loader(train_dataset, args, shuffle=True)
    val_loader = prepare_loader(val_dataset, args, shuffle=False)

    logger.info("XLSTM_BACKEND_AVAILABLE=%s", XLSTM_BACKEND_AVAILABLE)
    logger.info("require_xlstm_backend=%s", args.require_xlstm_backend)
    model = build_model(args, train_dataset, coords, device)
    sample_adj = train_dataset[0][5] if len(train_dataset[0]) == 10 else train_dataset[0][4]
    criterion = build_loss(args, train_dataset, sample_adj, coords, device)
    optimizer = optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    logger.info("context_extractor_type=%s", getattr(model, "context_extractor_type", None))
    if not args.require_xlstm_backend:
        logger.warning("require_xlstm_backend=false allows GRU fallback and is not recommended for formal experiments.")
    logger.info("coord_order=%s", args.coord_order)
    logger.info("coords first_5_rows=%s", coords[:5].tolist())
    coords_lonlat = coords_to_lonlat(coords, args.coord_order)
    pairwise = torch.cdist(coords_lonlat.float(), coords_lonlat.float())
    mask = ~torch.eye(pairwise.shape[0], dtype=torch.bool)
    logger.info(
        "pairwise distance(deg-space quick check) min/mean/max=%.6f/%.6f/%.6f",
        float(pairwise[mask].min().item()),
        float(pairwise[mask].mean().item()),
        float(pairwise[mask].max().item()),
    )

    sanity_batch = next(iter(train_loader))
    if len(sanity_batch) == 10:
        sanity_seq_x, sanity_seq_y, sanity_meteo_norm, sanity_meteo_raw, _sanity_time, _sanity_adj_geo, _sanity_adj_poi, _sanity_adj_land, sanity_poi, sanity_landuse = sanity_batch
    else:
        sanity_seq_x, sanity_seq_y, sanity_meteo_raw, _sanity_time, _sanity_adj_geo, _sanity_adj_poi, _sanity_adj_land, sanity_poi, sanity_landuse = sanity_batch
        sanity_meteo_norm = sanity_meteo_raw
    log_wind_stats(logger, sanity_meteo_raw, train_dataset.wind_u_meteo_pos, train_dataset.wind_v_meteo_pos, "raw")
    log_wind_stats(logger, sanity_meteo_norm, train_dataset.wind_u_meteo_pos, train_dataset.wind_v_meteo_pos, "normalized")
    sanity_output, sanity_debug = model.forecast(
        x_enc=sanity_seq_x.to(device),
        static_feat=build_static_batch(sanity_poi, sanity_landuse, device),
        target_idx=args.target_idx if args.features == "MS" else None,
        meteo_future=sanity_meteo_raw.to(device) if args.use_wind_advection else None,
        return_debug=True,
    )
    _ = sanity_output
    log_coords_debug(logger, coords_to_lonlat(coords, args.coord_order), sanity_debug)

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
            if len(batch) == 10:
                seq_x, seq_y, meteo_future_norm, meteo_future_raw, _time_feat, _adj_geo, _adj_poi, _adj_land, poi_feat_batch, landuse_feat_batch = batch
            else:
                seq_x, seq_y, meteo_future_raw, _time_feat, _adj_geo, _adj_poi, _adj_land, poi_feat_batch, landuse_feat_batch = batch
                meteo_future_norm = meteo_future_raw
            seq_x = seq_x.to(device)
            seq_y = seq_y.to(device)
            meteo_future_norm = meteo_future_norm.to(device)
            meteo_future_raw = meteo_future_raw.to(device)
            static_batch = build_static_batch(poi_feat_batch, landuse_feat_batch, device)
            seq_y_target = seq_y[:, -args.pred_len :, :, :]

            optimizer.zero_grad()
            output, debug = model.forecast(
                x_enc=seq_x,
                static_feat=static_batch,
                target_idx=args.target_idx if args.features == "MS" else None,
                meteo_future=meteo_future_raw if args.use_wind_advection else None,
                return_debug=True,
            )
            loss_dict = criterion(
                output=output,
                target=seq_y_target,
                x_enc_norm=seq_x,
                meteo_future=meteo_future_raw if args.use_observable_dyn_loss else None,
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

            if batch_idx == 0:
                log_debug_diagnostics(logger, debug, loss_dict)

        scheduler.step()
        train_steps = max(train_steps, 1)
        train_loss_avg = train_loss_total / train_steps
        train_breakdown = {key: value / train_steps for key, value in train_breakdown.items()}

        val_loss, _val_breakdown, val_preds_real, val_trues_real, _val_preds_norm, _val_trues_norm, val_metrics_real, val_debug = collect_eval(
            model=model,
            loader=val_loader,
            dataset=val_dataset,
            args=args,
            criterion=criterion,
            device=device,
        )

        logger.info("=" * 80)
        logger.info(f"epoch {epoch + 1}/{args.epochs}")
        logger.info(
            f"train loss={train_loss_avg:.6f} pred={train_breakdown['pred']:.6f} "
            f"L_nonneg={train_breakdown['L_nonneg']:.6f} L_smooth={train_breakdown['L_smooth']:.6f} L_dyn={train_breakdown['L_dyn']:.6f}"
        )
        if args.use_phy_ode and hasattr(model, "phy_ode_func"):
            params = model.phy_ode_func.get_positive_parameters()
            logger.info(
                "latent physical coeffs k_diff=%.6f k_adv=%.6f gamma=%.6f",
                float(params["k_diff"].detach().cpu()),
                float(params["k_adv"].detach().cpu()),
                float(params["gamma"].detach().cpu()),
            )
        logger.info(
            f"val loss={val_loss:.6f} pred={_val_breakdown['pred']:.6f} "
            f"L_nonneg={_val_breakdown['L_nonneg']:.6f} L_smooth={_val_breakdown['L_smooth']:.6f} L_dyn={_val_breakdown['L_dyn']:.6f}"
        )
        log_debug_diagnostics(logger, val_debug)
        log_hourly_metrics(logger, "val", val_metrics_real, pred_len=val_preds_real.shape[1])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = save_best_checkpoint(model, args)
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            np.save(save_dir / "pred_val_original.npy", val_preds_real)
            np.save(save_dir / "true_val_original.npy", val_trues_real)
            write_metrics_csv(save_dir / "best_metrics.csv", val_metrics_real)
            logger.info(f"best model updated -> {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("early stopping triggered")
                break


def test_model(args, logger):
    validate_args(args, logger)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    coords = load_coords(args.coords_path)
    train_dataset = prepare_dataset("train", args)
    test_dataset = prepare_dataset("test", args)
    test_loader = prepare_loader(test_dataset, args, shuffle=False)

    logger.info("XLSTM_BACKEND_AVAILABLE=%s", XLSTM_BACKEND_AVAILABLE)
    logger.info("require_xlstm_backend=%s", args.require_xlstm_backend)
    model = build_model(args, train_dataset, coords, device)
    sample_adj = train_dataset[0][5] if len(train_dataset[0]) == 10 else train_dataset[0][4]
    criterion = build_loss(args, train_dataset, sample_adj, coords, device)
    checkpoint_path = find_checkpoint_path(args)
    logger.info(f"loading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)

    test_loss, test_breakdown, preds_real, trues_real, preds_norm, trues_norm, metrics_real, test_debug = collect_eval(
        model=model,
        loader=test_loader,
        dataset=test_dataset,
        args=args,
        criterion=criterion,
        device=device,
    )

    logger.info("=" * 80)
    logger.info(
        f"test loss={test_loss:.6f} pred={test_breakdown['pred']:.6f} "
        f"L_nonneg={test_breakdown['L_nonneg']:.6f} L_smooth={test_breakdown['L_smooth']:.6f} L_dyn={test_breakdown['L_dyn']:.6f}"
    )
    log_debug_diagnostics(logger, test_debug)
    log_hourly_metrics(logger, "test", metrics_real, pred_len=preds_real.shape[1])

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "pred_test_original.npy", preds_real)
    np.save(save_dir / "true_test_original.npy", trues_real)
    np.save(save_dir / "pred_test_normalized.npy", preds_norm)
    np.save(save_dir / "true_test_normalized.npy", trues_norm)
    write_metrics_csv(save_dir / "test_metrics.csv", metrics_real)


def main():
    args = parse_args()
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("PMO-Net wind Gansu slim branch")
    logger.info("=" * 80)
    logger.info(f"config: {vars(args)}")

    if args.mode == "train":
        train_model(args, logger)
    else:
        test_model(args, logger)


if __name__ == "__main__":
    main()
