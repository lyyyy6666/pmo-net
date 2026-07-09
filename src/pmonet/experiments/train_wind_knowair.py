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

from pmonet.data.knowair import (  # noqa: E402
    OFFICIAL_SPLITS,
    Dataset_KnowAir_ForWind,
    custom_collate_fn_knowair_physics_first,
)
from pmonet.models.pmonet import (  # noqa: E402
    XLSTM_BACKEND_AVAILABLE,
    xLSTM_WindDualODE_Mixer,
)


KNOWAIR_SEGMENTS = {
    "overall_original_scale": (0, 24),
    "h1_24_original_scale": (0, 8),
    "h25_48_original_scale": (8, 16),
    "h49_72_original_scale": (16, 24),
}

KNOWAIR_WIND_METADATA_CANDIDATES = [
    "knowair_feature_columns.json",
    "KnowAir_feature_columns.json",
    "feature_columns.json",
    "columns.json",
]

KNOWAIR_DEFAULT_WIND_SOURCE = "u950_v950"
KNOWAIR_WIND_SOURCE_TO_FEATURES = {
    "u950_v950": ("u_component_of_wind+950", "v_component_of_wind+950"),
    "100m": ("100m_u_component_of_wind", "100m_v_component_of_wind"),
}


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


def load_knowair_feature_metadata(root_path: str) -> tuple[Optional[Path], Optional[list[str]], Optional[dict[str, Any]]]:
    metadata_path = resolve_knowair_wind_metadata(root_path)
    if metadata_path is None:
        return None, None, None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        names = payload.get("feature_names")
    elif isinstance(payload, list):
        names = payload
        payload = {"feature_names": names}
    else:
        raise ValueError(f"Unsupported KnowAir feature metadata format in {metadata_path}")
    if not isinstance(names, list) or not names:
        raise ValueError(f"KnowAir feature metadata in {metadata_path} must provide a non-empty feature_names list.")
    return metadata_path, [str(name) for name in names], payload


def build_knowair_feature_index_map(feature_names: list[str]) -> dict[str, int]:
    return {str(name): idx for idx, name in enumerate(feature_names)}


def resolve_knowair_target_index_from_metadata(feature_names: list[str]) -> int:
    index_map = build_knowair_feature_index_map(feature_names)
    for candidate in ("PM2.5", "pm25", "PM25", "pm2.5"):
        if candidate in index_map:
            return int(index_map[candidate])
    raise ValueError(f"Could not find PM2.5 target in KnowAir feature metadata: {feature_names}")


def resolve_knowair_wind_indices_from_metadata(
    feature_names: list[str],
    wind_source: str,
) -> tuple[int, int, str, str]:
    if wind_source not in KNOWAIR_WIND_SOURCE_TO_FEATURES:
        raise ValueError(
            f"Unsupported KnowAir wind_source={wind_source!r}. "
            f"Available choices: {sorted(KNOWAIR_WIND_SOURCE_TO_FEATURES.keys())}"
        )
    u_name, v_name = KNOWAIR_WIND_SOURCE_TO_FEATURES[wind_source]
    index_map = build_knowair_feature_index_map(feature_names)
    if u_name not in index_map or v_name not in index_map:
        raise ValueError(
            f"KnowAir metadata does not contain required wind features {u_name!r}, {v_name!r}. "
            f"Available features: {feature_names}"
        )
    return int(index_map[u_name]), int(index_map[v_name]), u_name, v_name


def resolve_knowair_wind_setup(args) -> dict[str, Any]:
    metadata_path, feature_names, metadata_payload = load_knowair_feature_metadata(args.root_path)
    result = {
        "metadata_path": metadata_path,
        "feature_names": feature_names,
        "metadata_payload": metadata_payload,
        "wind_source": args.wind_source,
        "target_idx_raw": None,
        "wind_u_idx_raw": None,
        "wind_v_idx_raw": None,
        "wind_u_name": None,
        "wind_v_name": None,
        "wind_u_meteo_pos": None,
        "wind_v_meteo_pos": None,
        "resolved_from": None,
    }
    if feature_names is None:
        return result

    target_idx_raw = resolve_knowair_target_index_from_metadata(feature_names)
    result["target_idx_raw"] = target_idx_raw
    u_idx_raw_auto, v_idx_raw_auto, u_name, v_name = resolve_knowair_wind_indices_from_metadata(feature_names, args.wind_source)
    result["wind_u_name"] = u_name
    result["wind_v_name"] = v_name

    if args.wind_u_idx is not None or args.wind_v_idx is not None:
        if args.wind_u_idx is None or args.wind_v_idx is None:
            raise ValueError("wind_u_idx and wind_v_idx must be provided together.")
        if int(args.wind_u_idx) != int(u_idx_raw_auto) or int(args.wind_v_idx) != int(v_idx_raw_auto):
            raise ValueError(
                "Provided KnowAir wind indices do not match local metadata. "
                f"Expected raw indices ({u_idx_raw_auto}, {v_idx_raw_auto}) for wind_source={args.wind_source!r}, "
                f"got ({args.wind_u_idx}, {args.wind_v_idx})."
            )
        result["resolved_from"] = "manual_verified_against_metadata"
        result["wind_u_idx_raw"] = int(args.wind_u_idx)
        result["wind_v_idx_raw"] = int(args.wind_v_idx)
    else:
        result["resolved_from"] = "auto_from_local_metadata"
        result["wind_u_idx_raw"] = int(u_idx_raw_auto)
        result["wind_v_idx_raw"] = int(v_idx_raw_auto)

    if result["wind_u_idx_raw"] == target_idx_raw or result["wind_v_idx_raw"] == target_idx_raw:
        raise ValueError("Resolved KnowAir wind indices overlap with PM2.5 target index, which is invalid.")

    meteo_cols = [idx for idx in range(len(feature_names)) if idx != target_idx_raw]
    result["wind_u_meteo_pos"] = int(meteo_cols.index(result["wind_u_idx_raw"]))
    result["wind_v_meteo_pos"] = int(meteo_cols.index(result["wind_v_idx_raw"]))
    return result


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
        dt_hours: float = 3.0,
        eps: float = 1e-6,
        observable_dyn_mode: str = "advection_diffusion",
        wind_graph_norm: str = "row",
    ):
        super().__init__()
        self.wind_u_idx = int(wind_u_idx)
        self.wind_v_idx = int(wind_v_idx)
        self.distance_scale_km = float(distance_scale_km)
        self.coord_order = coord_order
        self.use_geo_mask_for_wind = use_geo_mask_for_wind
        self.dt_hours = float(dt_hours)
        self.eps = float(eps)
        self.observable_dyn_mode = observable_dyn_mode
        self.wind_graph_norm = wind_graph_norm

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
        diff_term = -params["k_y_diff"] * torch.einsum("ij,bhjd->bhid", self.laplacian_geo, y_state_real)

        if self.observable_dyn_mode == "advection_diffusion":
            advection_steps = []
            a_w_steps = []
            for step in range(meteo_future.shape[1]):
                a_w = self.build_wind_graph(meteo_future[:, step])
                y_step = y_state_real[:, step]
                inflow = torch.einsum("bji,bjd->bid", a_w, y_step)
                outflow = a_w.sum(dim=-1).unsqueeze(-1) * y_step
                advection_steps.append(params["k_y_adv"] * (inflow - outflow))
                a_w_steps.append(a_w)
            adv_term = torch.stack(advection_steps, dim=1)
            a_w_all = torch.stack(a_w_steps, dim=1)
        else:
            adv_term = torch.zeros_like(diff_term)
            a_w_all = torch.zeros(
                y_state_real.shape[0],
                y_state_real.shape[1],
                y_state_real.shape[2],
                y_state_real.shape[2],
                dtype=y_state_real.dtype,
                device=y_state_real.device,
            )
        rhs = diff_term + adv_term - params["gamma_y"] * y_state_real

        dyn_residual = (d_y_dt - rhs) / (self.target_std.view(1, 1, 1, 1) + self.eps)
        l_dyn = (dyn_residual ** 2).mean()

        debug = {
            "observable_dyn_mode": self.observable_dyn_mode,
            "A_w_shape": tuple(a_w_all.shape),
            "A_w_symmetry_error": float((a_w_all - a_w_all.transpose(-1, -2)).abs().mean().detach().cpu()),
            "A_w_mean": float(a_w_all.mean().detach().cpu()),
            "A_w_max": float(a_w_all.max().detach().cpu()),
            "A_w_row_sum_mean": float(a_w_all.sum(dim=-1).mean().detach().cpu()),
            "A_w_row_sum_max": float(a_w_all.sum(dim=-1).max().detach().cpu()),
            "diffusion_abs_mean": float(diff_term.abs().mean().detach().cpu()),
            "adv_abs_mean": float(adv_term.abs().mean().detach().cpu()),
            "reaction_abs_mean": 0.0,
            "decay_abs_mean": float((params["gamma_y"] * y_state_real).abs().mean().detach().cpu()),
            "L_dyn": float(l_dyn.detach().cpu()),
            "k_y_diff": float(params["k_y_diff"].detach().cpu()),
            "k_y_adv": float(params["k_y_adv"].detach().cpu()),
            "gamma_y": float(params["gamma_y"].detach().cpu()),
        }
        return l_dyn, debug


class KnowAirWindLoss(nn.Module):
    def __init__(
        self,
        pred_len: int,
        lambda_nonnegative: float,
        lambda_temporal_smooth: float,
        lambda_observable_dyn: float,
        target_mean: float,
        target_std: float,
        observable_dynamics: Optional[ObservableDynamicsConsistency],
    ):
        super().__init__()
        self.lambda_nonnegative = float(lambda_nonnegative)
        self.lambda_temporal_smooth = float(lambda_temporal_smooth)
        self.lambda_observable_dyn = float(lambda_observable_dyn)
        self.observable_dynamics = observable_dynamics
        self.register_buffer("target_mean", torch.tensor(float(target_mean), dtype=torch.float32).view(1, 1, 1, 1))
        self.register_buffer("target_std", torch.tensor(float(target_std), dtype=torch.float32).view(1, 1, 1, 1))
        time_weights = torch.exp(-torch.arange(pred_len, dtype=torch.float32) * 0.05)
        self.register_buffer("time_weights", time_weights / time_weights.sum())

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        x_enc_norm: torch.Tensor,
        meteo_future: Optional[torch.Tensor],
        target_idx: int,
    ) -> dict[str, Any]:
        output_norm = output
        target_norm = target
        output_real = output_norm * self.target_std + self.target_mean
        target_real = target_norm * self.target_std + self.target_mean

        l_pred = (((output_norm - target_norm) ** 2) * self.time_weights.view(1, -1, 1, 1)).mean()
        l_nonneg = F.relu(-output_real).mean()
        if output.shape[1] > 1:
            l_smooth = ((output_norm[:, 1:] - output_norm[:, :-1]) ** 2).mean()
        else:
            l_smooth = output.new_tensor(0.0)

        if self.lambda_observable_dyn > 0.0 and self.observable_dynamics is not None:
            needs_wind = getattr(self.observable_dynamics, "observable_dyn_mode", "advection_diffusion") == "advection_diffusion"
            if needs_wind and meteo_future is None:
                raise ValueError("meteo_future is required when observable dynamics loss is enabled.")
            l_dyn, dyn_debug = self.observable_dynamics(
                x_enc_norm=x_enc_norm,
                y_pred_real=output_real,
                meteo_future=meteo_future,
                target_idx=target_idx,
            )
        else:
            l_dyn = output.new_tensor(0.0)
            dyn_debug = {
                "A_w_shape": None,
                "A_w_symmetry_error": 0.0,
                "A_w_mean": 0.0,
                "A_w_max": 0.0,
                "adv_abs_mean": 0.0,
                "L_dyn": 0.0,
            }

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
    parser = argparse.ArgumentParser(description="Slim KnowAir wind-consistency PMO-Net training")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--dataset_num", type=int, default=1, choices=[1, 2, 3])

    parser.add_argument("--root_path", type=str, default="./data/KnowAir")
    parser.add_argument("--data_path", type=str, default="KnowAir.npy")
    parser.add_argument("--station_path", type=str, default="station.csv")
    parser.add_argument("--adj_geo_path", type=str, default="final_adj.npy")
    parser.add_argument("--graph_npz_path", type=str, default="graph_data.npz")

    parser.add_argument("--features", type=str, default="MS", choices=["MS"])
    parser.add_argument("--target_idx", type=int, default=17)
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
    parser.add_argument("--wind_u_idx", type=int, default=None)
    parser.add_argument("--wind_v_idx", type=int, default=None)
    parser.add_argument("--wind_source", type=str, default=KNOWAIR_DEFAULT_WIND_SOURCE, choices=sorted(KNOWAIR_WIND_SOURCE_TO_FEATURES.keys()))
    parser.add_argument("--knowair_wind_confirmed", type=str2bool, default=False)
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

    parser.add_argument("--save_dir", type=str, default="./results/wind_knowair")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/wind_knowair")
    parser.add_argument("--model_path", type=str, default="best_model.pth")
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def setup_logging():
    log_dir = Path.cwd() / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"train_final_wind_knowair_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"log file: {log_file}")
    return logger


def resolve_knowair_wind_metadata(root_path: str) -> Optional[Path]:
    root = Path(root_path)
    for name in KNOWAIR_WIND_METADATA_CANDIDATES:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def validate_args(args) -> None:
    if args.hist_len != args.seq_len:
        raise ValueError(f"KnowAir requires hist_len == seq_len. Got hist_len={args.hist_len}, seq_len={args.seq_len}")
    if args.features != "MS":
        raise ValueError("KnowAir slim script requires features='MS'.")
    if args.target_idx != 17:
        raise ValueError(f"KnowAir slim script expects target_idx=17 for PM2.5, got {args.target_idx}")
    if args.pred_len != 24:
        raise ValueError("KnowAir slim script expects pred_len=24 for 72h evaluation.")
    if args.time_interval_hours != 3.0:
        raise ValueError(f"KnowAir requires time_interval_hours=3.0, got {args.time_interval_hours}")
    if args.max_train_batches == 0 or args.max_val_batches == 0 or args.max_test_batches == 0:
        raise ValueError("0 means no batches; use None for full training/evaluation or positive integer for smoke test.")

    metadata_path, feature_names, _metadata_payload = load_knowair_feature_metadata(args.root_path)
    if feature_names is not None:
        metadata_target_idx = resolve_knowair_target_index_from_metadata(feature_names)
        if args.target_idx != metadata_target_idx:
            raise ValueError(
                f"target_idx={args.target_idx} does not match local KnowAir metadata target index {metadata_target_idx}."
            )

    observable_dyn_requested = args.use_observable_dyn_loss and args.lambda_observable_dyn > 0.0
    wind_required_for_observable = observable_dyn_requested and args.observable_dyn_mode == "advection_diffusion"
    wind_requested = args.use_wind_advection or wind_required_for_observable
    if not wind_requested:
        return

    if metadata_path is None:
        raise ValueError(
            "KnowAir wind columns are not locally confirmed because no local columns metadata file was found. "
            "Current repo cannot prove which meteo columns are raw wind_u/wind_v, so wind must stay disabled. "
            "Please provide local metadata or keep wind off."
        )
    if not args.knowair_wind_confirmed and (args.wind_u_idx is not None or args.wind_v_idx is not None):
        raise ValueError(
            "Manual KnowAir wind indices require --knowair_wind_confirmed true after you verify them against local metadata."
        )
    try:
        resolve_knowair_wind_setup(args)
    except Exception as exc:
        raise ValueError(
            f"Failed to confirm KnowAir wind columns safely. You may explicitly pass "
            f"--wind_u_idx/--wind_v_idx together with --knowair_wind_confirmed true only if they match local metadata. "
            f"Details: {exc}"
        ) from exc


def prepare_dataset(flag: str, args) -> Dataset_KnowAir_ForWind:
    wind_setup = resolve_knowair_wind_setup(args)
    dataset = Dataset_KnowAir_ForWind(
        root_path=args.root_path,
        flag=flag,
        dataset_num=args.dataset_num,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target_idx=args.target_idx,
        scale=True,
        data_path=args.data_path,
        station_path=args.station_path,
        adj_geo_path=args.adj_geo_path,
        graph_npz_path=args.graph_npz_path,
        time_feature_mode=args.time_feature_mode,
        wind_u_idx=wind_setup["wind_u_idx_raw"],
        wind_v_idx=wind_setup["wind_v_idx_raw"],
        coord_order=args.coord_order,
    )
    dataset.feature_names = wind_setup["feature_names"]
    dataset.feature_metadata_path = str(wind_setup["metadata_path"]) if wind_setup["metadata_path"] is not None else None
    dataset.wind_source = wind_setup["wind_source"]
    dataset.wind_u_name = wind_setup["wind_u_name"]
    dataset.wind_v_name = wind_setup["wind_v_name"]
    dataset.wind_u_idx_raw = wind_setup["wind_u_idx_raw"]
    dataset.wind_v_idx_raw = wind_setup["wind_v_idx_raw"]
    dataset.wind_resolution = wind_setup["resolved_from"]
    dataset.observable_dyn_mode = args.observable_dyn_mode
    return dataset

def prepare_loader(dataset, args, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=custom_collate_fn_knowair_physics_first,
    )


def log_split_summary(logger, dataset: Dataset_KnowAir_ForWind, split_name: str) -> None:
    split_def = OFFICIAL_SPLITS[dataset.dataset_num][dataset.flag]
    logger.info(
        "[%s] official date range: %s -> %s",
        split_name,
        split_def.start_date,
        split_def.end_date,
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


def log_dataset_debug(logger, dataset: Dataset_KnowAir_ForWind, split_name: str) -> None:
    logger.info("[%s] KnowAir.npy shape=%s", split_name, (dataset.total_time_steps, dataset.num_nodes, dataset.num_features))
    logger.info("[%s] target_idx=%d", split_name, dataset.target_idx)
    logger.info("[%s] meteo_cols len=%d values=%s", split_name, len(dataset.meteo_cols), dataset.meteo_cols)
    logger.info("[%s] target_idx in meteo_cols? %s", split_name, dataset.target_idx in dataset.meteo_cols)
    logger.info("[%s] coords.shape=%s first3=%s", split_name, tuple(dataset.coords.shape), dataset.coords[:3].tolist())
    logger.info(
        "[%s] adj_geo.shape=%s graph_adj_match_final_adj=%s",
        split_name,
        tuple(dataset.adj_geo.shape),
        True,
    )
    logger.info("[%s] scaler target mean/std=%.6f / %.6f", split_name, dataset.target_mean, dataset.target_std)
    logger.info("[%s] feature_metadata_path=%s", split_name, getattr(dataset, "feature_metadata_path", None))
    logger.info("[%s] wind_source=%s resolved_from=%s", split_name, getattr(dataset, "wind_source", None), getattr(dataset, "wind_resolution", None))
    logger.info("[%s] wind_u_idx_raw=%s wind_v_idx_raw=%s", split_name, getattr(dataset, "wind_u_idx_raw", None), getattr(dataset, "wind_v_idx_raw", None))
    logger.info("[%s] wind_u_meteo_pos=%s wind_v_meteo_pos=%s", split_name, dataset.wind_u_meteo_pos, dataset.wind_v_meteo_pos)
    logger.info("[%s] coord_order=%s", split_name, getattr(dataset, "coord_order", None))


def build_model(args, dataset: Dataset_KnowAir_ForWind, device: torch.device):
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


def build_loss(args, dataset: Dataset_KnowAir_ForWind, device: torch.device):
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
    for prefix, (start, end) in KNOWAIR_SEGMENTS.items():
        end = min(end, pred_real.shape[1])
        if start < end:
            report.update(flatten_metric_fields(prefix, calculate_metrics(pred_real[:, start:end], true_real[:, start:end])))
    return report


def log_horizon_report(logger, split_name: str, metrics: dict[str, float]) -> None:
    for prefix in KNOWAIR_SEGMENTS:
        logger.info(
            "[%s] %s MAE=%.4f RMSE=%.4f MAPE=%.2f%% WMAPE=%.2f%% R2=%.4f",
            split_name,
            prefix,
            metrics[f"{prefix}_MAE"],
            metrics[f"{prefix}_RMSE"],
            metrics[f"{prefix}_MAPE"],
            metrics[f"{prefix}_WMAPE"],
            metrics[f"{prefix}_R2"],
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


def log_single_batch_debug(logger, batch_info: dict[str, Any], output: torch.Tensor, dataset: Dataset_KnowAir_ForWind) -> None:
    logger.info("batch shape x_enc=%s", tuple(batch_info["x_enc"].shape))
    logger.info("batch shape future_y=%s", tuple(batch_info["future_y"].shape))
    logger.info("batch shape seq_y=%s", tuple(batch_info["seq_y"].shape) if batch_info["seq_y"] is not None else None)
    logger.info("batch shape meteo_future=%s", tuple(batch_info["meteo_future"].shape))
    logger.info("batch shape coords=%s", tuple(batch_info["coords"].shape))
    logger.info("batch shape adj_geo=%s", tuple(batch_info["adj_geo"].shape))
    logger.info("output shape=%s", tuple(output.shape))
    logger.info("target_idx=%d", dataset.target_idx)
    logger.info("meteo_future excludes PM2.5=%s", dataset.target_idx not in dataset.meteo_cols)
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


def log_wind_debug(logger, dataset: Dataset_KnowAir_ForWind, meteo_future: torch.Tensor, debug: dict[str, Any], raw_or_normalized: str) -> None:
    if dataset.wind_u_meteo_pos is None or dataset.wind_v_meteo_pos is None:
        return
    wind_u = meteo_future[..., dataset.wind_u_meteo_pos]
    wind_v = meteo_future[..., dataset.wind_v_meteo_pos]
    logger.info("wind_u_idx=%s wind_v_idx=%s", dataset.wind_u_idx, dataset.wind_v_idx)
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
    if not args.require_xlstm_backend:
        logger.warning("require_xlstm_backend=false allows GRU fallback and is not recommended for formal experiments.")
    if args.latent_dt_mode == "physical":
        logger.warning(
            "latent_dt_mode=physical uses dt=time_interval_hours=%.6f instead of normalized 1/pred_len. "
            "This can strongly amplify latent ODE updates and destabilize training.",
            args.time_interval_hours,
        )

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
        log_wind_debug(
            logger,
            train_dataset,
            sanity_batch["meteo_future_raw"],
            sanity_debug,
            raw_or_normalized="raw_knownair_meteo_excluding_pm25",
        )
        log_wind_debug(
            logger,
            train_dataset,
            sanity_batch["meteo_future"],
            sanity_debug,
            raw_or_normalized="normalized_knownair_meteo_excluding_pm25",
        )
    logger.info("future wind covariates are taken from dataset['meteo_future'] aligned as (B, pred_len, N, meteo_dim).")

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

        val_loss, val_breakdown, val_preds_real, val_trues_real, _val_preds_norm, _val_trues_norm, val_metrics, _val_debug = collect_eval(
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
        if args.use_phy_ode and hasattr(model, "phy_ode_func"):
            params = model.phy_ode_func.get_positive_parameters()
            logger.info(
                "latent physical coeffs k_diff=%.6f k_adv=%.6f gamma=%.6f",
                float(params["k_diff"].detach().cpu()),
                float(params["k_adv"].detach().cpu()),
                float(params["gamma"].detach().cpu()),
            )
        logger.info(
            "val loss=%.6f pred=%.6f L_nonneg=%.6f L_smooth=%.6f L_dyn=%.6f",
            val_loss,
            val_breakdown["pred"],
            val_breakdown["L_nonneg"],
            val_breakdown["L_smooth"],
            val_breakdown["L_dyn"],
        )
        log_debug_diagnostics(logger, _val_debug["latent"])
        if args.use_observable_dyn_loss and args.lambda_observable_dyn > 0.0:
            log_debug_diagnostics(logger, _val_debug["observable"])
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

    if args.use_wind_advection or (args.use_observable_dyn_loss and args.observable_dyn_mode == "advection_diffusion"):
        first_batch = convert_batch(next(iter(test_loader)), device)
        log_wind_debug(
            logger,
            test_dataset,
            first_batch["meteo_future_raw"],
            debug["latent"] if args.use_wind_advection else debug["observable"],
            raw_or_normalized="raw_knownair_meteo_excluding_pm25",
        )
        log_wind_debug(
            logger,
            test_dataset,
            first_batch["meteo_future"],
            debug["latent"] if args.use_wind_advection else debug["observable"],
            raw_or_normalized="normalized_knownair_meteo_excluding_pm25",
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "pred_test_original.npy", preds_real)
    np.save(save_dir / "true_test_original.npy", trues_real)
    np.save(save_dir / "pred_test_normalized.npy", preds_norm)
    np.save(save_dir / "true_test_normalized.npy", trues_norm)
    write_metrics_csv(save_dir / "test_metrics.csv", metrics)
    logger.info("pred_test_original.npy shape=%s", tuple(preds_real.shape))


def main():
    args = parse_args()
    validate_args(args)
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("PMO-Net wind KnowAir slim branch")
    logger.info("=" * 80)
    logger.info("config: %s", vars(args))

    if args.mode == "train":
        train_model(args, logger)
    else:
        test_model(args, logger)


if __name__ == "__main__":
    main()
