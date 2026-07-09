from __future__ import annotations

import csv
import math
import ast
import csv
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import Dataset


def _load_dataset_rimst_class():
    try:
        from ._legacy_rimst_loader import Dataset_RIMST as ImportedDatasetRIMST
        return ImportedDatasetRIMST
    except Exception:
        data_loader_path = Path(__file__).resolve().parent / "_legacy_rimst_loader.py"
        source = data_loader_path.read_text(encoding="utf-8")
        module_ast = ast.parse(source)
        class_node = None
        for node in module_ast.body:
            if isinstance(node, ast.ClassDef) and node.name == "Dataset_RIMST":
                class_node = node
                break
        if class_node is None:
            raise RuntimeError("Could not locate Dataset_RIMST in data_loader.py")

        class_source = ast.get_source_segment(source, class_node)

        class MiniSelection:
            def __init__(self, values):
                self.values = np.asarray(values)

        class MiniILoc:
            def __init__(self, frame):
                self.frame = frame

            def __getitem__(self, key):
                if isinstance(key, tuple):
                    row_sel, col_sel = key
                else:
                    row_sel, col_sel = key, slice(None)
                values = self.frame._values[row_sel, col_sel]
                return MiniSelection(values)

        class MiniDataFrame:
            def __init__(self, values, columns):
                self._values = np.asarray(values)
                self.columns = list(columns)
                self.iloc = MiniILoc(self)

            @property
            def values(self):
                return self._values

        class MiniPandas:
            @staticmethod
            def read_csv(path, header="infer"):
                path = Path(path)
                with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
                    reader = list(csv.reader(handle))
                if not reader:
                    raise ValueError(f"Empty CSV file: {path}")
                if header is None:
                    rows = reader
                    columns = list(range(len(rows[0])))
                else:
                    columns = reader[0]
                    rows = reader[1:]
                values = []
                for row in rows:
                    parsed = []
                    for cell in row:
                        cell = cell.strip()
                        parsed.append(float(cell) if cell != "" else np.nan)
                    values.append(parsed)
                return MiniDataFrame(np.asarray(values, dtype=float), columns)

        class MiniStandardScaler:
            def fit(self, data):
                data = np.asarray(data, dtype=float)
                self.mean_ = data.mean(axis=0)
                self.scale_ = data.std(axis=0)
                self.scale_[self.scale_ == 0] = 1.0
                return self

            def transform(self, data):
                data = np.asarray(data, dtype=float)
                return (data - self.mean_) / self.scale_

            def inverse_transform(self, data):
                data = np.asarray(data, dtype=float)
                return data * self.scale_ + self.mean_

        namespace = {
            "os": os,
            "np": np,
            "pd": MiniPandas,
            "torch": torch,
            "Dataset": Dataset,
            "StandardScaler": MiniStandardScaler,
        }
        exec(class_source, namespace)
        return namespace["Dataset_RIMST"]


Dataset_RIMST = _load_dataset_rimst_class()


class Dataset_RIMST_PhysicsFirst(Dataset_RIMST):
    """
    Physics-first extension of Dataset_RIMST.

    Returns:
        seq_x, seq_y, meteo_future, time_feat, adj_geo, adj_poi, adj_land, poi_feat, landuse_feat

    Extra attributes used by wind-aware scripts:
        - meteo_future_raw: raw meteorology for physical advection
        - meteo_future_norm: normalized meteorology for model-side covariates
    """

    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="M",
        data_path="Gansu_Air.csv",
        poi_path="poi_attribute_adj.npy",
        landuse_path="landuse_attribute_adj.npy",
        adj_geo_path="knn_adj.csv",
        scale=True,
        timeenc=0,
        freq="h",
        seasonal_patterns=None,
        meteo_path="meteo_physics_first.npy",
        meteo_columns_path="meteo_physics_first_columns.json",
        meteo_station_order_path="meteo_physics_first_station_order.json",
        meteo_datetime_path="meteo_physics_first_datetime.csv",
        return_meteo_pair: bool = False,
    ):
        self.meteo_path = meteo_path
        self.meteo_columns_path = meteo_columns_path
        self.meteo_station_order_path = meteo_station_order_path
        self.meteo_datetime_path = meteo_datetime_path
        self.return_meteo_pair = bool(return_meteo_pair)
        super().__init__(
            args=args,
            root_path=root_path,
            flag=flag,
            size=size,
            features=features,
            data_path=data_path,
            poi_path=poi_path,
            landuse_path=landuse_path,
            adj_geo_path=adj_geo_path,
            scale=scale,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=seasonal_patterns,
        )

    def __read_data__(self):
        super().__read_data__()
        root = Path(self.root_path)
        meteo_file = Path(self.meteo_path)
        if not meteo_file.is_absolute():
            meteo_file = root / meteo_file
        station_order_file = Path(self.meteo_station_order_path)
        if not station_order_file.is_absolute():
            station_order_file = root / station_order_file
        datetime_file = Path(self.meteo_datetime_path)
        if not datetime_file.is_absolute():
            datetime_file = root / datetime_file

        if not meteo_file.exists():
            raise FileNotFoundError(
                "meteo_physics_first.npy not found. Please run build_u10v10_meteo_tensor.py first."
            )
        if not station_order_file.exists():
            raise FileNotFoundError(
                "meteo_physics_first_station_order.json not found. Please run build_u10v10_meteo_tensor.py first."
            )
        if not datetime_file.exists():
            raise FileNotFoundError(
                "meteo_physics_first_datetime.csv not found. Please run build_u10v10_meteo_tensor.py first."
            )

        self.meteo_data_raw = np.load(meteo_file).astype(np.float32)
        self.meteo_station_order = __import__("json").loads(station_order_file.read_text(encoding="utf-8"))
        self.meteo_datetimes = self._load_datetimes(datetime_file)
        self.time_feat_all = self._build_time_features(self.meteo_datetimes, self.meteo_data_raw.shape[1]).astype(np.float32)

        if self.meteo_data_raw.shape[0] != self._data_all.shape[0]:
            raise ValueError(
                f"Meteo time length {self.meteo_data_raw.shape[0]} does not match Gansu_Air time length {self._data_all.shape[0]}."
            )
        if self.meteo_data_raw.shape[1] != self._data_all.shape[1]:
            raise ValueError(
                f"Meteo station count {self.meteo_data_raw.shape[1]} does not match Gansu_Air station count {self._data_all.shape[1]}."
            )
        if len(self.meteo_datetimes) != self._data_all.shape[0]:
            raise ValueError(
                f"Datetime length {len(self.meteo_datetimes)} does not match Gansu_Air time length {self._data_all.shape[0]}."
            )

        num_train = int(len(self._data_all) * 0.7)
        num_test = int(len(self._data_all) * 0.2)
        num_vali = len(self._data_all) - num_train - num_test

        meteo_train = self.meteo_data_raw[:num_train].reshape(-1, self.meteo_data_raw.shape[-1])
        self.meteo_train_mean = meteo_train.mean(axis=0).astype(np.float32)
        self.meteo_train_std = meteo_train.std(axis=0).astype(np.float32)
        self.meteo_train_std[self.meteo_train_std == 0] = 1.0
        self.meteo_data_norm = ((self.meteo_data_raw - self.meteo_train_mean) / self.meteo_train_std).astype(np.float32)

        border1s = [0, num_train - self.seq_len, len(self._data_all) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(self._data_all)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.meteo_x = self.meteo_data_norm[border1:border2]
        self.meteo_x_raw = self.meteo_data_raw[border1:border2]
        self.time_feat_x = self.time_feat_all[border1:border2]

    @staticmethod
    def _load_datetimes(path: Path) -> list[str]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [row["datetime"] for row in reader]

    @staticmethod
    def _build_time_features(datetimes: list[str], num_nodes: int) -> np.ndarray:
        feats = []
        for dt in datetimes:
            date_part, time_part = dt.split(" ")
            year, month, day = [int(x) for x in date_part.split("-")]
            hour = int(time_part.split(":")[0])
            weekday = __import__("datetime").datetime(year, month, day).weekday()
            hour_angle = 2.0 * math.pi * hour / 24.0
            weekday_angle = 2.0 * math.pi * weekday / 7.0
            feats.append(
                [
                    math.sin(hour_angle),
                    math.cos(hour_angle),
                    math.sin(weekday_angle),
                    math.cos(weekday_angle),
                ]
            )
        feat_arr = np.asarray(feats, dtype=np.float32)  # (T, 4)
        feat_arr = np.repeat(feat_arr[:, None, :], num_nodes, axis=1)  # (T, N, 4)
        return feat_arr

    def __getitem__(self, index):
        seq_x, seq_y, adj_geo, adj_poi, adj_land, poi_feat, landuse_feat = super().__getitem__(index)

        s_begin = index
        s_end = s_begin + self.seq_len
        future_begin = s_end
        future_end = s_end + self.pred_len

        meteo_future_norm = self.meteo_x[future_begin:future_end]
        meteo_future_raw = self.meteo_x_raw[future_begin:future_end]
        time_feat = self.time_feat_x[future_begin:future_end]

        meteo_future_norm = torch.from_numpy(meteo_future_norm).float()
        meteo_future_raw = torch.from_numpy(meteo_future_raw).float()
        time_feat = torch.from_numpy(time_feat).float()

        if self.return_meteo_pair:
            return (
                seq_x,
                seq_y,
                meteo_future_norm,
                meteo_future_raw,
                time_feat,
                adj_geo,
                adj_poi,
                adj_land,
                poi_feat,
                landuse_feat,
            )
        return (
            seq_x,
            seq_y,
            meteo_future_raw,
            time_feat,
            adj_geo,
            adj_poi,
            adj_land,
            poi_feat,
            landuse_feat,
        )


def custom_collate_fn_physics_first(batch):
    seq_x_list, seq_y_list = [], []
    meteo_list, meteo_raw_list, time_feat_list = [], [], []
    adj_geo_list, adj_poi_list, adj_land_list = [], [], []
    poi_feat_list, landuse_feat_list = [], []
    has_meteo_pair = len(batch[0]) == 10

    for item in batch:
        if has_meteo_pair:
            seq_x, seq_y, meteo_future_norm, meteo_future_raw, time_feat, adj_geo, adj_poi, adj_land, poi_feat, landuse_feat = item
            meteo_list.append(meteo_future_norm.clone())
            meteo_raw_list.append(meteo_future_raw.clone())
        else:
            seq_x, seq_y, meteo_future_raw, time_feat, adj_geo, adj_poi, adj_land, poi_feat, landuse_feat = item
            meteo_list.append(meteo_future_raw.clone())
        seq_x_list.append(seq_x.clone())
        seq_y_list.append(seq_y.clone())
        time_feat_list.append(time_feat.clone())
        adj_geo_list.append(adj_geo)
        adj_poi_list.append(adj_poi)
        adj_land_list.append(adj_land)
        poi_feat_list.append(poi_feat)
        landuse_feat_list.append(landuse_feat)

    seq_x_batch = default_collate(seq_x_list)
    seq_y_batch = default_collate(seq_y_list)
    meteo_future_batch = default_collate(meteo_list)
    time_feat_batch = default_collate(time_feat_list)

    adj_geo = adj_geo_list[0]
    adj_poi = adj_poi_list[0]
    adj_land = adj_land_list[0]

    poi_feat_batch = default_collate(poi_feat_list) if poi_feat_list[0] is not None else None
    landuse_feat_batch = default_collate(landuse_feat_list) if landuse_feat_list[0] is not None else None

    if has_meteo_pair:
        meteo_future_raw_batch = default_collate(meteo_raw_list)
        return (
            seq_x_batch,
            seq_y_batch,
            meteo_future_batch,
            meteo_future_raw_batch,
            time_feat_batch,
            adj_geo,
            adj_poi,
            adj_land,
            poi_feat_batch,
            landuse_feat_batch,
        )
    return (
        seq_x_batch,
        seq_y_batch,
        meteo_future_batch,
        time_feat_batch,
        adj_geo,
        adj_poi,
        adj_land,
        poi_feat_batch,
        landuse_feat_batch,
    )
