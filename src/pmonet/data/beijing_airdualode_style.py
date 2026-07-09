from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
AIRDUALODE_FEATURE_NAMES = ["PM2.5", "temperature", "pressure", "humidity", "wind_speed", "wind_direction"]
AIRDUALODE_COLUMNS = ["time"] + AIRDUALODE_FEATURE_NAMES


class NumpyStandardScaler:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> "NumpyStandardScaler":
        data = np.asarray(data, dtype=np.float32)
        mean = data.mean(axis=0)
        scale = data.std(axis=0)
        scale[scale == 0] = 1.0
        self.mean_ = mean.astype(np.float32)
        self.scale_ = scale.astype(np.float32)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fitted before transform.")
        return ((data - self.mean_) / self.scale_).astype(np.float32)


@dataclass(frozen=True)
class SplitRange:
    start_idx: int
    end_idx: int


def parse_time(text: str) -> datetime:
    return datetime.strptime(text, TIME_FORMAT)


def build_time_features(datetimes: list[datetime], mode: str = "cyclic6") -> np.ndarray:
    rows: list[list[float]] = []
    for dt in datetimes:
        hour = dt.hour / 24.0
        weekday = dt.weekday() / 7.0
        month = (dt.month - 1) / 12.0
        if mode == "simple3":
            rows.append([hour, weekday, month])
        elif mode == "cyclic6":
            hour_angle = 2.0 * math.pi * hour
            weekday_angle = 2.0 * math.pi * weekday
            month_angle = 2.0 * math.pi * month
            rows.append(
                [
                    math.sin(hour_angle),
                    math.cos(hour_angle),
                    math.sin(weekday_angle),
                    math.cos(weekday_angle),
                    math.sin(month_angle),
                    math.cos(month_angle),
                ]
            )
        else:
            raise ValueError(f"Unsupported time feature mode: {mode}")
    return np.asarray(rows, dtype=np.float32)


def load_station_coords(path: Path, coord_order: str = "lonlat") -> tuple[np.ndarray, list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if coord_order not in {"lonlat", "latlon"}:
        raise ValueError(f"coord_order must be 'lonlat' or 'latlon', got {coord_order!r}")
    if coord_order == "lonlat":
        coords = np.asarray([[float(row["longitude"]), float(row["latitude"])] for row in rows], dtype=np.float32)
    else:
        coords = np.asarray([[float(row["latitude"]), float(row["longitude"])] for row in rows], dtype=np.float32)
    return coords, rows


def compute_7_1_2_splits(total_steps: int) -> dict[str, SplitRange]:
    train_end = int(total_steps * 0.7)
    val_end = int(total_steps * 0.8)
    return {
        "train": SplitRange(0, train_end),
        "val": SplitRange(train_end, val_end),
        "test": SplitRange(val_end, total_steps),
    }


def load_station_hourly_csv(path: Path) -> tuple[list[datetime], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != AIRDUALODE_COLUMNS:
            raise ValueError(f"{path} columns must be {AIRDUALODE_COLUMNS}, got {reader.fieldnames}")
        times: list[datetime] = []
        rows: list[list[float]] = []
        for row in reader:
            times.append(parse_time(row["time"]))
            feature_row: list[float] = []
            for column in AIRDUALODE_FEATURE_NAMES:
                text = row[column].strip()
                if text == "" or text.lower() in {"nan", "null", "none", "na", "n/a"}:
                    feature_row.append(np.nan)
                else:
                    feature_row.append(float(text))
            rows.append(feature_row)
    return times, np.asarray(rows, dtype=np.float32)


class Dataset_Beijing_AirDualODE_Style(Dataset):
    def __init__(
        self,
        root_path: str | Path,
        flag: str = "train",
        size: list[int] | tuple[int, int, int] | None = None,
        features: str = "MS",
        seq_len: int = 24,
        label_len: int = 12,
        pred_len: int = 24,
        target_idx: int = 0,
        scale: bool = True,
        station_path: str = "station.csv",
        stations_dir: str = "stations",
        adj_geo_path: str = "final_adj.npy",
        graph_npz_path: str = "graph_data.npz",
        metadata_path: str = "metadata.json",
        freq: str = "3h",
        time_feature_mode: str = "cyclic6",
        coord_order: str = "lonlat",
    ) -> None:
        if flag not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported flag: {flag}")
        if size is not None:
            if len(size) != 3:
                raise ValueError(f"size must be [seq_len, label_len, pred_len], got {size}")
            seq_len, label_len, pred_len = int(size[0]), int(size[1]), int(size[2])
        if features != "MS":
            raise ValueError("Dataset_Beijing_AirDualODE_Style currently supports features='MS' only.")
        if label_len <= 0 or label_len > seq_len:
            raise ValueError(f"label_len must be in [1, seq_len], got label_len={label_len}, seq_len={seq_len}")
        if freq != "3h":
            raise ValueError("This AirDualODE-style Beijing loader currently supports freq='3h' only.")

        self.root_path = Path(root_path)
        self.flag = flag
        self.features = features
        self.seq_len = int(seq_len)
        self.label_len = int(label_len)
        self.pred_len = int(pred_len)
        self.target_idx = int(target_idx)
        self.scale = bool(scale)
        self.station_path = station_path
        self.stations_dir = stations_dir
        self.adj_geo_path = adj_geo_path
        self.graph_npz_path = graph_npz_path
        self.metadata_path = metadata_path
        self.freq = freq
        self.time_feature_mode = time_feature_mode
        self.coord_order = coord_order

        self._load_all()

    def _load_all(self) -> None:
        metadata = json.loads((self.root_path / self.metadata_path).read_text(encoding="utf-8"))
        if metadata.get("feature_names") != AIRDUALODE_FEATURE_NAMES:
            raise ValueError(f"Unexpected metadata feature_names: {metadata.get('feature_names')}")
        self.metadata = metadata
        self.feature_names = list(AIRDUALODE_FEATURE_NAMES)

        self.coords_np, self.station_rows = load_station_coords(self.root_path / self.station_path, coord_order=self.coord_order)
        self.station_names = [row["station"] for row in self.station_rows]
        self.num_nodes = len(self.station_names)
        self.coords = torch.from_numpy(self.coords_np).float()

        first_station_path = self.root_path / self.stations_dir / f"{self.station_names[0]}.csv"
        base_times_hourly, _ = load_station_hourly_csv(first_station_path)
        per_station_raw: list[np.ndarray] = []
        all_same_times = True
        for station_name in self.station_names:
            times_hourly, station_values_hourly = load_station_hourly_csv(self.root_path / self.stations_dir / f"{station_name}.csv")
            if len(times_hourly) != len(base_times_hourly) or any(t1 != t2 for t1, t2 in zip(times_hourly, base_times_hourly)):
                all_same_times = False
                break
            per_station_raw.append(station_values_hourly)
        if not all_same_times:
            raise ValueError("All station csv files must share the same hourly time axis for stable loading.")

        if self.freq == "3h":
            downsample_idx = np.arange(0, len(base_times_hourly), 3, dtype=np.int64)
        else:
            downsample_idx = np.arange(0, len(base_times_hourly), dtype=np.int64)
        self.datetimes = [base_times_hourly[idx] for idx in downsample_idx]
        self.datetime_strings = [dt.strftime(TIME_FORMAT) for dt in self.datetimes]

        station_major = [values[downsample_idx] for values in per_station_raw]
        raw_data = np.stack(station_major, axis=1).astype(np.float32)
        self.total_time_steps, _, self.num_features = raw_data.shape
        if self.num_features != len(AIRDUALODE_FEATURE_NAMES):
            raise ValueError(f"Expected {len(AIRDUALODE_FEATURE_NAMES)} features, got {self.num_features}")

        self.meteo_cols = [idx for idx in range(self.num_features) if idx != self.target_idx]
        self.meteo_dim = len(self.meteo_cols)
        self.wind_enabled = False
        self.wind_u_idx = None
        self.wind_v_idx = None
        self.wind_u_meteo_pos = None
        self.wind_v_meteo_pos = None
        self.wind_source = metadata.get("wind_source")
        self.wind_u_name = None
        self.wind_v_name = None
        self.wind_u_idx_raw = None
        self.wind_v_idx_raw = None

        self.adj_geo_np = np.load(self.root_path / self.adj_geo_path).astype(np.float32)
        self.adj_geo = torch.from_numpy(self.adj_geo_np).float()
        graph_npz = np.load(self.root_path / self.graph_npz_path, allow_pickle=True)
        if "adj_mx" not in graph_npz.files:
            raise ValueError("graph_data.npz must contain adj_mx")
        if not np.allclose(graph_npz["adj_mx"].astype(np.float32), self.adj_geo_np, atol=1e-6):
            raise ValueError("graph_data.npz['adj_mx'] does not match final_adj.npy")

        self.time_feat_series = build_time_features(self.datetimes, mode=self.time_feature_mode)
        self.time_dim = int(self.time_feat_series.shape[-1])

        split_ranges = compute_7_1_2_splits(self.total_time_steps)
        self.split_ranges = split_ranges
        train_range = split_ranges["train"]
        train_block = raw_data[train_range.start_idx : train_range.end_idx]
        if train_block.size == 0:
            raise ValueError("Empty train block after 7:1:2 split.")

        self.scaler = None
        if self.scale:
            train_flat = train_block.reshape(-1, self.num_features)
            mean = np.zeros((self.num_features,), dtype=np.float32)
            std = np.ones((self.num_features,), dtype=np.float32)
            for feature_idx in range(self.num_features):
                values = train_flat[:, feature_idx]
                values = values[np.isfinite(values)]
                if values.size == 0:
                    raise ValueError(f"Train split has no finite values for feature index {feature_idx}.")
                mean[feature_idx] = np.asarray(values.mean(), dtype=np.float32)
                feature_std = float(values.std())
                std[feature_idx] = np.float32(feature_std if feature_std > 0 else 1.0)
            self.scaler = NumpyStandardScaler()
            self.scaler.mean_ = mean
            self.scaler.scale_ = std

            full_flat = raw_data.reshape(-1, self.num_features)
            scaled_flat = np.full_like(full_flat, np.nan, dtype=np.float32)
            finite_mask = np.isfinite(full_flat)
            for feature_idx in range(self.num_features):
                mask_col = finite_mask[:, feature_idx]
                if np.any(mask_col):
                    scaled_flat[mask_col, feature_idx] = (
                        (full_flat[mask_col, feature_idx] - self.scaler.mean_[feature_idx]) / self.scaler.scale_[feature_idx]
                    )
            self.x_all = scaled_flat.reshape(self.total_time_steps, self.num_nodes, self.num_features).astype(np.float32)
        else:
            self.x_all = raw_data.astype(np.float32)

        self.raw_data = raw_data.astype(np.float32)
        self.meteo_all_raw = self.raw_data[:, :, self.meteo_cols].astype(np.float32)
        meteo_train = self.meteo_all_raw[train_range.start_idx : train_range.end_idx].reshape(-1, self.meteo_dim)
        self.meteo_train_mean = np.zeros((self.meteo_dim,), dtype=np.float32)
        self.meteo_train_std = np.ones((self.meteo_dim,), dtype=np.float32)
        for feature_idx in range(self.meteo_dim):
            values = meteo_train[:, feature_idx]
            values = values[np.isfinite(values)]
            if values.size == 0:
                raise ValueError(f"Train split has no finite meteo values for meteo feature index {feature_idx}.")
            self.meteo_train_mean[feature_idx] = np.asarray(values.mean(), dtype=np.float32)
            feature_std = float(values.std())
            self.meteo_train_std[feature_idx] = np.float32(feature_std if feature_std > 0 else 1.0)
        self.meteo_all_norm = np.full_like(self.meteo_all_raw, np.nan, dtype=np.float32)
        finite_meteo = np.isfinite(self.meteo_all_raw)
        for feature_idx in range(self.meteo_dim):
            mask_col = finite_meteo[:, :, feature_idx]
            if np.any(mask_col):
                self.meteo_all_norm[:, :, feature_idx][mask_col] = (
                    (self.meteo_all_raw[:, :, feature_idx][mask_col] - self.meteo_train_mean[feature_idx]) / self.meteo_train_std[feature_idx]
                )

        self.target_mean = float(self.scaler.mean_[self.target_idx]) if self.scaler is not None else 0.0
        self.target_std = float(self.scaler.scale_[self.target_idx]) if self.scaler is not None else 1.0

        split_range = split_ranges[self.flag]
        self.split_start_idx = int(split_range.start_idx)
        self.split_end_idx = int(split_range.end_idx) - 1

        self.sample_starts = self._build_valid_indices(split_range)

    def _build_valid_indices(self, split_range: SplitRange) -> np.ndarray:
        max_start = split_range.end_idx - self.seq_len - self.pred_len
        if max_start < split_range.start_idx:
            return np.asarray([], dtype=np.int64)
        valid: list[int] = []
        for start in range(split_range.start_idx, max_start + 1):
            end = start + self.seq_len + self.pred_len
            x_window = self.x_all[start : start + self.seq_len]
            y_window = self.x_all[start + self.seq_len : end, :, self.target_idx : self.target_idx + 1]
            meteo_window = self.meteo_all_norm[start + self.seq_len : end]
            meteo_window_raw = self.meteo_all_raw[start + self.seq_len : end]
            if (
                np.isnan(x_window).any()
                or np.isnan(y_window).any()
                or np.isnan(meteo_window).any()
                or np.isnan(meteo_window_raw).any()
            ):
                continue
            valid.append(start)
        return np.asarray(valid, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.sample_starts.shape[0])

    def _expand_time_features(self, start: int, end: int) -> np.ndarray:
        time_slice = self.time_feat_series[start:end]
        expanded = np.broadcast_to(time_slice[:, None, :], (time_slice.shape[0], self.num_nodes, self.time_dim))
        return expanded.copy().astype(np.float32)

    def get_sample_window(self, index: int) -> dict[str, Any]:
        start = int(self.sample_starts[index])
        hist_start = start
        hist_end = start + self.seq_len - 1
        future_start = start + self.seq_len
        future_end = future_start + self.pred_len - 1
        return {
            "sample_index": index,
            "start_index": start,
            "hist_start_index": hist_start,
            "hist_end_index": hist_end,
            "future_start_index": future_start,
            "future_end_index": future_end,
            "hist_start": self.datetime_strings[hist_start],
            "hist_end": self.datetime_strings[hist_end],
            "future_start": self.datetime_strings[future_start],
            "future_end": self.datetime_strings[future_end],
        }

    def inverse_transform_target(self, data: np.ndarray) -> np.ndarray:
        return data * self.target_std + self.target_mean

    def __getitem__(self, index: int) -> dict[str, Any]:
        start = int(self.sample_starts[index])
        hist_slice = slice(start, start + self.seq_len)
        future_slice = slice(start + self.seq_len, start + self.seq_len + self.pred_len)

        x_enc = torch.from_numpy(self.x_all[hist_slice]).float()
        hist_y = torch.from_numpy(self.x_all[hist_slice, :, self.target_idx : self.target_idx + 1]).float()
        future_y = torch.from_numpy(self.x_all[future_slice, :, self.target_idx : self.target_idx + 1]).float()
        meteo_hist = torch.from_numpy(self.meteo_all_norm[hist_slice]).float()
        meteo_future = torch.from_numpy(self.meteo_all_norm[future_slice]).float()
        meteo_hist_raw = torch.from_numpy(self.meteo_all_raw[hist_slice]).float()
        meteo_future_raw = torch.from_numpy(self.meteo_all_raw[future_slice]).float()
        time_feat_hist = torch.from_numpy(self._expand_time_features(hist_slice.start, hist_slice.stop)).float()
        time_feat_future = torch.from_numpy(self._expand_time_features(future_slice.start, future_slice.stop)).float()

        label_hist_slice = slice(start + self.seq_len - self.label_len, start + self.seq_len)
        label_hist_y = torch.from_numpy(self.x_all[label_hist_slice, :, self.target_idx : self.target_idx + 1]).float()
        seq_y = torch.cat([label_hist_y, future_y], dim=0)

        sample_window = self.get_sample_window(index)
        return {
            "x_enc": x_enc,
            "hist_y": hist_y,
            "future_y": future_y,
            "meteo_hist": meteo_hist,
            "meteo_future": meteo_future,
            "meteo_hist_raw": meteo_hist_raw,
            "meteo_future_raw": meteo_future_raw,
            "time_feat_hist": time_feat_hist,
            "time_feat_future": time_feat_future,
            "coords": self.coords,
            "adj_geo": self.adj_geo,
            "wind_u_hist": None,
            "wind_v_hist": None,
            "wind_u_future": None,
            "wind_v_future": None,
            "seq_y": seq_y,
            "static_feat": None,
            "sample_index": sample_window["sample_index"],
            "hist_start": sample_window["hist_start"],
            "hist_end": sample_window["hist_end"],
            "future_start": sample_window["future_start"],
            "future_end": sample_window["future_end"],
        }


def custom_collate_fn_beijing_airdualode_style(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Empty batch")

    stacked_keys = [
        "x_enc",
        "hist_y",
        "future_y",
        "meteo_hist",
        "meteo_future",
        "meteo_hist_raw",
        "meteo_future_raw",
        "time_feat_hist",
        "time_feat_future",
    ]
    result: dict[str, Any] = {key: default_collate([item[key] for item in batch]) for key in stacked_keys}
    result["seq_y"] = default_collate([item["seq_y"] for item in batch])
    result["coords"] = batch[0]["coords"]
    result["adj_geo"] = batch[0]["adj_geo"]
    result["static_feat"] = None
    result["wind_u_hist"] = None
    result["wind_v_hist"] = None
    result["wind_u_future"] = None
    result["wind_v_future"] = None
    result["sample_index"] = [item["sample_index"] for item in batch]
    result["hist_start"] = [item["hist_start"] for item in batch]
    result["hist_end"] = [item["hist_end"] for item in batch]
    result["future_start"] = [item["future_start"] for item in batch]
    result["future_end"] = [item["future_end"] for item in batch]
    return result
