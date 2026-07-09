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
BEIJING_FREQ_HOURS = 3


@dataclass(frozen=True)
class SplitDateRange:
    start_datetime: str
    end_datetime: str


BEIJING_SPLITS: dict[str, SplitDateRange] = {
    "train": SplitDateRange("2017-01-01 15:00:00", "2017-11-30 21:00:00"),
    "val": SplitDateRange("2017-12-01 00:00:00", "2017-12-31 21:00:00"),
    "test": SplitDateRange("2018-01-01 00:00:00", "2018-01-31 12:00:00"),
}


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

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fitted before inverse_transform.")
        return (data * self.scale_ + self.mean_).astype(np.float32)


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


def load_timestamps(path: Path) -> list[datetime]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    timestamps = [parse_time(row["timestamp"]) for row in rows]
    if not timestamps:
        raise ValueError(f"No timestamps found in {path}")
    for idx in range(len(timestamps) - 1):
        diff_hours = (timestamps[idx + 1] - timestamps[idx]).total_seconds() / 3600.0
        if abs(diff_hours - BEIJING_FREQ_HOURS) > 1e-6:
            raise ValueError(f"Timestamps must be 3-hourly. Found diff={diff_hours} at index {idx}")
    return timestamps


class Dataset_Beijing_PhysicsFirst(Dataset):
    def __init__(
        self,
        root_path: str | Path,
        flag: str = "train",
        hist_len: int = 24,
        pred_len: int = 24,
        target_idx: int = 0,
        scale: bool = True,
        data_path: str = "Beijing.npy",
        station_path: str = "station.csv",
        adj_geo_path: str = "final_adj.npy",
        graph_npz_path: str = "graph_data.npz",
        metadata_path: str = "beijing_feature_columns.json",
        timestamps_path: str = "timestamps.csv",
        time_feature_mode: str = "cyclic6",
        wind_u_idx: int | None = 4,
        wind_v_idx: int | None = 5,
        coord_order: str = "lonlat",
    ) -> None:
        if flag not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported flag: {flag}")
        if hist_len <= 0 or pred_len <= 0:
            raise ValueError("hist_len and pred_len must be positive.")

        self.root_path = Path(root_path)
        self.flag = flag
        self.hist_len = int(hist_len)
        self.pred_len = int(pred_len)
        self.target_idx = int(target_idx)
        self.scale = bool(scale)
        self.data_path = data_path
        self.station_path = station_path
        self.adj_geo_path = adj_geo_path
        self.graph_npz_path = graph_npz_path
        self.metadata_path = metadata_path
        self.timestamps_path = timestamps_path
        self.time_feature_mode = time_feature_mode
        self.wind_u_idx = wind_u_idx
        self.wind_v_idx = wind_v_idx
        self.coord_order = coord_order

        self._load_all()

    def _load_all(self) -> None:
        dynamic_path = self.root_path / self.data_path
        raw_data = np.load(dynamic_path).astype(np.float32)
        if raw_data.ndim != 3:
            raise ValueError(f"Expected 3D Beijing tensor, got shape {raw_data.shape}")

        self.total_time_steps, self.num_nodes, self.num_features = raw_data.shape
        if not (0 <= self.target_idx < self.num_features):
            raise ValueError(f"target_idx {self.target_idx} out of range for {self.num_features} features")

        metadata_payload = json.loads((self.root_path / self.metadata_path).read_text(encoding="utf-8"))
        self.feature_names = [str(name) for name in metadata_payload.get("feature_names", [])]
        if len(self.feature_names) != self.num_features:
            raise ValueError(
                f"Metadata feature_names length {len(self.feature_names)} does not match tensor feature dim {self.num_features}"
            )
        self.feature_metadata = metadata_payload
        self.wind_source = metadata_payload.get("wind_source")
        self.wind_direction_convention = metadata_payload.get("wind_direction_convention")
        self.wind_u_name = self.feature_names[self.wind_u_idx] if self.wind_u_idx is not None else None
        self.wind_v_name = self.feature_names[self.wind_v_idx] if self.wind_v_idx is not None else None
        self.wind_u_idx_raw = self.wind_u_idx
        self.wind_v_idx_raw = self.wind_v_idx

        self.datetimes = load_timestamps(self.root_path / self.timestamps_path)
        if len(self.datetimes) != self.total_time_steps:
            raise ValueError(
                f"timestamps.csv length {len(self.datetimes)} does not match tensor time dim {self.total_time_steps}"
            )
        self.datetime_strings = [dt.strftime(TIME_FORMAT) for dt in self.datetimes]
        self.datetime_to_index = {dt: idx for idx, dt in enumerate(self.datetimes)}

        self.coords_np, self.station_rows = load_station_coords(self.root_path / self.station_path, coord_order=self.coord_order)
        if self.coords_np.shape != (self.num_nodes, 2):
            raise ValueError(
                f"station.csv rows {self.coords_np.shape[0]} do not match Beijing node count {self.num_nodes}"
            )
        self.coords = torch.from_numpy(self.coords_np).float()

        adj_path = self.root_path / self.adj_geo_path
        self.adj_geo_np = np.load(adj_path).astype(np.float32)
        if self.adj_geo_np.shape != (self.num_nodes, self.num_nodes):
            raise ValueError(
                f"adj_geo shape {self.adj_geo_np.shape} does not match expected {(self.num_nodes, self.num_nodes)}"
            )
        self.adj_geo = torch.from_numpy(self.adj_geo_np).float()

        graph_npz = np.load(self.root_path / self.graph_npz_path, allow_pickle=True)
        if "adj_mx" not in graph_npz.files:
            raise ValueError("graph_data.npz must contain adj_mx")
        graph_adj = graph_npz["adj_mx"].astype(np.float32)
        if not np.allclose(graph_adj, self.adj_geo_np, atol=1e-6):
            raise ValueError("graph_data.npz['adj_mx'] does not match final_adj.npy")

        self.meteo_cols = [idx for idx in range(self.num_features) if idx != self.target_idx]
        self.meteo_dim = len(self.meteo_cols)
        self.wind_enabled = self.wind_u_idx is not None and self.wind_v_idx is not None
        if (self.wind_u_idx is None) != (self.wind_v_idx is None):
            raise ValueError("wind_u_idx and wind_v_idx must be both set or both None")
        if self.wind_enabled:
            if self.wind_u_idx == self.target_idx or self.wind_v_idx == self.target_idx:
                raise ValueError("wind indices must not point to target_idx")
            if self.wind_u_idx not in self.meteo_cols or self.wind_v_idx not in self.meteo_cols:
                raise ValueError("wind indices must refer to non-target feature columns")
            self.wind_u_meteo_pos = self.meteo_cols.index(self.wind_u_idx)
            self.wind_v_meteo_pos = self.meteo_cols.index(self.wind_v_idx)
        else:
            self.wind_u_meteo_pos = None
            self.wind_v_meteo_pos = None

        train_range = BEIJING_SPLITS["train"]
        self.train_start_idx = self.datetime_to_index[parse_time(train_range.start_datetime)]
        self.train_end_idx = self.datetime_to_index[parse_time(train_range.end_datetime)]
        train_block = raw_data[self.train_start_idx : self.train_end_idx + 1]

        self.scaler = None
        if self.scale:
            train_flat = train_block.reshape(-1, self.num_features)
            self.scaler = NumpyStandardScaler().fit(train_flat)
            full_flat = raw_data.reshape(-1, self.num_features)
            scaled_flat = self.scaler.transform(full_flat)
            self.x_all = scaled_flat.reshape(self.total_time_steps, self.num_nodes, self.num_features).astype(np.float32)
        else:
            self.x_all = raw_data.astype(np.float32)

        self.raw_data = raw_data.astype(np.float32)
        raw_meteo_all = raw_data[:, :, self.meteo_cols].astype(np.float32)
        self.meteo_all_raw = raw_meteo_all
        meteo_train = raw_meteo_all[self.train_start_idx : self.train_end_idx + 1].reshape(-1, self.meteo_dim)
        self.meteo_train_mean = meteo_train.mean(axis=0).astype(np.float32)
        self.meteo_train_std = meteo_train.std(axis=0).astype(np.float32)
        self.meteo_train_std[self.meteo_train_std == 0] = 1.0
        self.meteo_all_norm = ((self.meteo_all_raw - self.meteo_train_mean) / self.meteo_train_std).astype(np.float32)

        self.time_feat_series = build_time_features(self.datetimes, mode=self.time_feature_mode)
        self.time_dim = int(self.time_feat_series.shape[-1])

        self.target_mean = float(self.scaler.mean_[self.target_idx]) if self.scaler is not None else 0.0
        self.target_std = float(self.scaler.scale_[self.target_idx]) if self.scaler is not None else 1.0

        split_range = BEIJING_SPLITS[self.flag]
        self.split_start_idx = self.datetime_to_index[parse_time(split_range.start_datetime)]
        self.split_end_idx = self.datetime_to_index[parse_time(split_range.end_datetime)]

        max_start = self.split_end_idx - self.hist_len - self.pred_len + 1
        if max_start < self.split_start_idx:
            self.sample_starts = np.asarray([], dtype=np.int64)
        else:
            self.sample_starts = np.arange(self.split_start_idx, max_start + 1, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.sample_starts.shape[0])

    def _expand_time_features(self, start: int, end: int) -> np.ndarray:
        time_slice = self.time_feat_series[start:end]
        expanded = np.broadcast_to(time_slice[:, None, :], (time_slice.shape[0], self.num_nodes, self.time_dim))
        return expanded.copy().astype(np.float32)

    def get_sample_window(self, index: int) -> dict[str, Any]:
        start = int(self.sample_starts[index])
        hist_start = start
        hist_end = start + self.hist_len - 1
        future_start = start + self.hist_len
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
        hist_slice = slice(start, start + self.hist_len)
        future_slice = slice(start + self.hist_len, start + self.hist_len + self.pred_len)

        x_enc = torch.from_numpy(self.x_all[hist_slice]).float()
        hist_y = torch.from_numpy(self.x_all[hist_slice, :, self.target_idx : self.target_idx + 1]).float()
        future_y = torch.from_numpy(self.x_all[future_slice, :, self.target_idx : self.target_idx + 1]).float()
        meteo_hist = torch.from_numpy(self.meteo_all_norm[hist_slice]).float()
        meteo_future = torch.from_numpy(self.meteo_all_norm[future_slice]).float()
        meteo_hist_raw = torch.from_numpy(self.meteo_all_raw[hist_slice]).float()
        meteo_future_raw = torch.from_numpy(self.meteo_all_raw[future_slice]).float()
        time_feat_hist = torch.from_numpy(self._expand_time_features(hist_slice.start, hist_slice.stop)).float()
        time_feat_future = torch.from_numpy(self._expand_time_features(future_slice.start, future_slice.stop)).float()

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
            "sample_index": sample_window["sample_index"],
            "hist_start": sample_window["hist_start"],
            "hist_end": sample_window["hist_end"],
            "future_start": sample_window["future_start"],
            "future_end": sample_window["future_end"],
        }


class Dataset_Beijing_ForWind(Dataset_Beijing_PhysicsFirst):
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
        data_path: str = "Beijing.npy",
        station_path: str = "station.csv",
        adj_geo_path: str = "final_adj.npy",
        graph_npz_path: str = "graph_data.npz",
        metadata_path: str = "beijing_feature_columns.json",
        timestamps_path: str = "timestamps.csv",
        time_feature_mode: str = "cyclic6",
        wind_u_idx: int | None = 4,
        wind_v_idx: int | None = 5,
        coord_order: str = "lonlat",
    ) -> None:
        if size is not None:
            if len(size) != 3:
                raise ValueError(f"size must be [seq_len, label_len, pred_len], got {size}")
            seq_len, label_len, pred_len = int(size[0]), int(size[1]), int(size[2])
        if features != "MS":
            raise ValueError("Dataset_Beijing_ForWind currently supports features='MS' only.")
        if label_len <= 0 or label_len > seq_len:
            raise ValueError(f"label_len must be in [1, seq_len], got label_len={label_len}, seq_len={seq_len}")

        self.features = features
        self.seq_len = int(seq_len)
        self.label_len = int(label_len)

        super().__init__(
            root_path=root_path,
            flag=flag,
            hist_len=self.seq_len,
            pred_len=int(pred_len),
            target_idx=target_idx,
            scale=scale,
            data_path=data_path,
            station_path=station_path,
            adj_geo_path=adj_geo_path,
            graph_npz_path=graph_npz_path,
            metadata_path=metadata_path,
            timestamps_path=timestamps_path,
            time_feature_mode=time_feature_mode,
            wind_u_idx=wind_u_idx,
            wind_v_idx=wind_v_idx,
            coord_order=coord_order,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        start = int(self.sample_starts[index])
        label_hist_slice = slice(start + self.seq_len - self.label_len, start + self.seq_len)
        label_hist_y = torch.from_numpy(
            self.x_all[label_hist_slice, :, self.target_idx : self.target_idx + 1]
        ).float()
        seq_y = torch.cat([label_hist_y, sample["future_y"]], dim=0)

        sample["seq_y"] = seq_y
        sample["static_feat"] = None
        return sample


def custom_collate_fn_beijing_physics_first(batch: list[dict[str, Any]]) -> dict[str, Any]:
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

    if "seq_y" in batch[0]:
        result["seq_y"] = default_collate([item["seq_y"] for item in batch])
    else:
        result["seq_y"] = None

    result["coords"] = batch[0]["coords"]
    result["adj_geo"] = batch[0]["adj_geo"]
    result["static_feat"] = None if batch[0].get("static_feat") is None else default_collate(
        [item["static_feat"] for item in batch]
    )

    optional_keys = ["wind_u_hist", "wind_v_hist", "wind_u_future", "wind_v_future"]
    for key in optional_keys:
        first_value = batch[0][key]
        result[key] = default_collate([item[key] for item in batch]) if first_value is not None else None

    result["sample_index"] = [item["sample_index"] for item in batch]
    result["hist_start"] = [item["hist_start"] for item in batch]
    result["hist_end"] = [item["hist_end"] for item in batch]
    result["future_start"] = [item["future_start"] for item in batch]
    result["future_end"] = [item["future_end"] for item in batch]
    return result
