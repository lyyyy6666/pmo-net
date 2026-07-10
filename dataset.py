from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


class Dataset_RIMST(Dataset):
    """
    RIMST Dataset Loader for Spatially-Informed xLSTM-Mixer
    
    Dynamic Data: (B, L, N, C) where:
        - B: Batch Size
        - L: Lookback Window (e.g., 720)
        - N: Number of stations/nodes (23)
        - C: Number of dynamic variables (8)
    
    Static Data:
        - POI distribution: from poi.npy
        - LANDUSE: from landuse.npy
    
    Adjacency Matrices:
        - adj_geo: Geographic distance adjacency matrix (sparse)
        - adj_poi: Semantic similarity matrix from poi.npy (dense)
        - adj_land: Functional similarity matrix from landuse.npy (dense)
    """
    def __init__(self, args, root_path, flag='train', size=None,
                 features='M', data_path='rimst_dynamic.csv',
                 poi_path='poi_attribute_adj.npy',
                 landuse_path='landuse_attribute_adj.npy',
                 adj_geo_path='knn_adj.csv',
                 scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        if size is None:
            self.seq_len = 720  # Default lookback window
            self.label_len = 96
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.poi_path = poi_path
        self.landuse_path = landuse_path
        self.adj_geo_path = adj_geo_path
        
        self.__read_data__()
    
    def __read_data__(self):
        """
        Load dynamic time series, static features, and adjacency matrix
        """
        # Load dynamic time series data
        # Format: CSV with header, first column is stationID (0-22), remaining columns are variables
        dynamic_file = os.path.join(self.root_path, self.data_path)
        
        if dynamic_file.endswith('.csv'):
            df_raw = pd.read_csv(dynamic_file)
            # First column is stationID, remaining columns are variables
            station_ids = df_raw.iloc[:, 0].values.astype(int)  # (T,)
            variable_data = df_raw.iloc[:, 1:].values  # (T, C) where C is number of variables
            
            # Get unique station IDs and number of variables
            unique_stations = np.unique(station_ids)
            unique_stations = np.sort(unique_stations)  # Ensure sorted order
            N = len(unique_stations)  # Number of stations (should be 23)
            C = variable_data.shape[1]  # Number of variables per station
            
            # Group data by station ID and reshape to (T, N, C)
            # Each station should have the same number of time steps
            # We need to pivot the data: group by stationID and stack variables
            data_list = []
            for station_id in unique_stations:
                station_mask = station_ids == station_id
                station_data = variable_data[station_mask]  # (T_station, C)
                data_list.append(station_data)
            
            # Check if all stations have the same number of time steps
            time_steps = [d.shape[0] for d in data_list]
            if len(set(time_steps)) > 1:
                # If different, pad or truncate to minimum
                min_time = min(time_steps)
                data_list = [d[:min_time] for d in data_list]
                print(f"Warning: Stations have different time steps, using minimum: {min_time}")
            
            # Stack to form (T, N, C)
            # Each element in data_list is (T, C), stack along axis=1 to get (T, N, C)
            data = np.stack(data_list, axis=1)  # (T, N, C)
            
            print(f"Dynamic data loaded: shape={data.shape}, N={N}, C={C}")
            
        elif dynamic_file.endswith('.npy'):
            data = np.load(dynamic_file)  # Expected shape: (T, N, C)
        else:
            raise ValueError(f"Unsupported file format for dynamic data: {dynamic_file}")
        
        # Split train/val/test
        num_train = int(len(data) * 0.7)
        num_test = int(len(data) * 0.2)
        num_vali = len(data) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # Normalize dynamic data
        if self.scale:
            train_data = data[border1s[0]:border2s[0]]  # (T_train, N, C)
            # Fit scaler on training data, flatten for StandardScaler
            train_flat = train_data.reshape(-1, train_data.shape[-1])  # (T_train*N, C)
            self.scaler = StandardScaler()
            self.scaler.fit(train_flat)
            # Transform all data
            data_flat = data.reshape(-1, data.shape[-1])  # (T*N, C)
            data_scaled = self.scaler.transform(data_flat)
            data = data_scaled.reshape(data.shape)  # (T, N, C)
        else:
            self.scaler = None
        
        self.data_x = data[border1:border2]  # (T_split, N, C)
        self.data_y = data[border1:border2]  # Same for now, will be sliced in __getitem__
        self._data_all = data
        
        N = data.shape[1]  # Number of nodes
        
        # Load POI adjacency matrix: (N, N) - already a similarity matrix
        # 支持泛化实验：当 poi_path 为 None 时，跳过加载（仅使用地理邻接矩阵）
        if self.poi_path is not None and self.poi_path.strip() != '':
            poi_file = os.path.join(self.root_path, self.poi_path)
            if poi_file.endswith('.npy'):
                self.adj_poi = np.load(poi_file)  # (N, N) - similarity matrix
                self.poi_features = None  # No feature matrix, only adjacency
            else:
                raise ValueError(f"POI file must be .npy format: {poi_file}")
        else:
            # 泛化实验模式：不使用 POI 邻接矩阵
            self.adj_poi = None
            self.poi_features = None
        
        # Load LANDUSE adjacency matrix: (N, N) - already a similarity matrix
        # 支持泛化实验：当 landuse_path 为 None 时，跳过加载（仅使用地理邻接矩阵）
        if self.landuse_path is not None and self.landuse_path.strip() != '':
            landuse_file = os.path.join(self.root_path, self.landuse_path)
            if landuse_file.endswith('.npy'):
                self.adj_land = np.load(landuse_file)  # (N, N) - similarity matrix
                self.landuse_features = None  # No feature matrix, only adjacency
            else:
                raise ValueError(f"LANDUSE file must be .npy format: {landuse_file}")
        else:
            # 泛化实验模式：不使用 LANDUSE 邻接矩阵
            self.adj_land = None
            self.landuse_features = None
        
        # Load geographic adjacency matrix (KNN): (N, N) - can be asymmetric and sparse
        adj_geo_file = os.path.join(self.root_path, self.adj_geo_path)
        if adj_geo_file.endswith('.csv'):
            # Load from CSV (no header)
            self.adj_geo = pd.read_csv(adj_geo_file, header=None).values  # (N, N)
        elif adj_geo_file.endswith('.npy'):
            self.adj_geo = np.load(adj_geo_file)  # (N, N)
        elif adj_geo_file.endswith('.npz'):
            # Load sparse matrix from .npz
            sparse_adj = np.load(adj_geo_file)
            if 'adj' in sparse_adj:
                self.adj_geo = sparse_adj['adj']
            else:
                # Try to reconstruct from sparse format
                try:
                    import scipy.sparse as sp
                    if 'data' in sparse_adj and 'indices' in sparse_adj and 'indptr' in sparse_adj:
                        self.adj_geo = sp.csr_matrix(
                            (sparse_adj['data'], sparse_adj['indices'], sparse_adj['indptr']),
                            shape=sparse_adj.get('shape', (N, N))
                        ).toarray()
                    else:
                        raise ValueError(f"Cannot load sparse adjacency from {adj_geo_file}")
                except ImportError:
                    raise ValueError(f"scipy required for .npz format, but not available")
        else:
            raise ValueError(f"Geographic adjacency file must be .csv, .npy or .npz format: {adj_geo_file}")
        
        # Note: KNN adjacency may be asymmetric, we can make it symmetric if needed
        # For now, we keep it as is (asymmetric)
        
        # Ensure all adjacency matrices match number of nodes
        assert self.adj_geo.shape == (N, N), \
            f"adj_geo shape {self.adj_geo.shape} != expected ({N}, {N})"
        
        # 泛化实验模式：adj_poi 和 adj_land 可能为 None
        if self.adj_poi is not None:
            assert self.adj_poi.shape == (N, N), \
                f"adj_poi shape {self.adj_poi.shape} != expected ({N}, {N})"
        if self.adj_land is not None:
            assert self.adj_land.shape == (N, N), \
                f"adj_land shape {self.adj_land.shape} != expected ({N}, {N})"
        
        # Convert to torch tensors
        # For sparse adj_geo, keep as dense for now (will convert to sparse in forward)
        self.adj_geo = torch.from_numpy(self.adj_geo).float()
        if self.adj_poi is not None:
            self.adj_poi = torch.from_numpy(self.adj_poi).float()
        if self.adj_land is not None:
            self.adj_land = torch.from_numpy(self.adj_land).float()
        
        # Store POI and LANDUSE features if available
        if self.poi_features is not None:
            self.poi_features = torch.from_numpy(self.poi_features).float()
        if self.landuse_features is not None:
            self.landuse_features = torch.from_numpy(self.landuse_features).float()
        
        print(f"RIMST Dataset [{self.set_type}]:")
        print(f"  Dynamic data shape: {self.data_x.shape}")
        print(f"  adj_geo shape: {self.adj_geo.shape} (sparse)")
        if self.adj_poi is not None:
            print(f"  adj_poi shape: {self.adj_poi.shape} (dense)")
        else:
            print(f"  adj_poi: None (泛化实验模式，仅使用地理邻接矩阵)")
        if self.adj_land is not None:
            print(f"  adj_land shape: {self.adj_land.shape} (dense)")
        else:
            print(f"  adj_land: None (泛化实验模式，仅使用地理邻接矩阵)")
        if self.poi_features is not None:
            print(f"  POI features shape: {self.poi_features.shape}")
        if self.landuse_features is not None:
            print(f"  LANDUSE features shape: {self.landuse_features.shape}")
    
    def __getitem__(self, index):
        """
        Returns:
            seq_x: (L, N, C) - input sequence (all variables for covariates)
            seq_y: (label_len + pred_len, N, C_out) - target sequence
                  - If features='M' (M-M): C_out = C (all variables)
                  - If features='MS' (M-S): C_out = 1 (only target_idx column)
            adj_geo: (N, N) - geographic adjacency matrix (sparse)
            adj_poi: (N, N) - POI similarity matrix (dense)
            adj_land: (N, N) - LANDUSE similarity matrix (dense)
            poi_features: (N, D_poi) or None - POI features
            landuse_features: (N, D_land) or None - LANDUSE features
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        # Extract sequences: (L, N, C) and (label_len + pred_len, N, C)
        seq_x = self.data_x[s_begin:s_end]  # (L, N, C) - keep all variables as covariates
        seq_y = self.data_y[r_begin:r_end]  # (label_len + pred_len, N, C)
        
        # ========== CRITICAL: Handle M-S (Multivariate-to-Single) mode ==========
        # When features='MS', slice seq_y to only include target_idx column
        if self.features == 'MS':
            # Get target_idx from args (default to 0 if not specified)
            target_idx = getattr(self.args, 'target_idx', 0) if hasattr(self.args, 'target_idx') else 0
            
            # Slice seq_y to only include target_idx column: (L, N, C) -> (L, N, 1)
            seq_y = seq_y[:, :, target_idx:target_idx+1]  # (label_len + pred_len, N, 1)
        
        # Convert to torch tensors
        seq_x = torch.from_numpy(seq_x).float()
        seq_y = torch.from_numpy(seq_y).float()
        
        # Adjacency matrices are the same for all samples
        adj_geo = self.adj_geo.clone()  # (N, N) - sparse
        # 泛化实验模式：adj_poi 和 adj_land 可能为 None
        adj_poi = self.adj_poi.clone() if self.adj_poi is not None else None  # (N, N) - dense or None
        adj_land = self.adj_land.clone() if self.adj_land is not None else None  # (N, N) - dense or None
        
        # POI and LANDUSE features (if available)
        poi_feat = self.poi_features.clone() if self.poi_features is not None else None
        landuse_feat = self.landuse_features.clone() if self.landuse_features is not None else None
        
        return seq_x, seq_y, adj_geo, adj_poi, adj_land, poi_feat, landuse_feat
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        """
        Inverse transform for dynamic data
        data: (..., N, C) or (..., N*C)
        """
        if self.scaler is None:
            return data
        
        original_shape = data.shape
        if len(original_shape) > 2:
            # Reshape to (..., C) for scaler
            data_flat = data.reshape(-1, original_shape[-1])
            data_inv = self.scaler.inverse_transform(data_flat)
            return data_inv.reshape(original_shape)
        else:
            return self.scaler.inverse_transform(data)

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


KNOWAIR_START_DATETIME = datetime(2015, 1, 1, 0, 0, 0)
KNOWAIR_FREQ_HOURS = 3


@dataclass(frozen=True)
class KnowAirSplitDateRange:
    start_date: str
    end_date: str


OFFICIAL_SPLITS: dict[int, dict[str, SplitDateRange]] = {
    1: {
        "train": KnowAirSplitDateRange("2015-01-01", "2016-12-31"),
        "val": KnowAirSplitDateRange("2017-01-01", "2017-12-31"),
        "test": KnowAirSplitDateRange("2018-01-01", "2018-12-31"),
    },
    2: {
        "train": KnowAirSplitDateRange("2015-11-01", "2016-02-28"),
        "val": KnowAirSplitDateRange("2016-11-01", "2017-02-28"),
        "test": KnowAirSplitDateRange("2017-11-01", "2018-02-28"),
    },
    3: {
        "train": KnowAirSplitDateRange("2016-09-01", "2016-11-30"),
        "val": KnowAirSplitDateRange("2016-12-01", "2016-12-31"),
        "test": KnowAirSplitDateRange("2017-01-01", "2017-01-31"),
    },
}


class KnowAirStandardScaler:
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


def build_knowair_datetime_axis(length: int) -> list[datetime]:
    return [KNOWAIR_START_DATETIME + timedelta(hours=KNOWAIR_FREQ_HOURS * i) for i in range(length)]


def datetime_to_index(date_text: str, end_of_day: bool = False) -> int:
    dt = datetime.strptime(date_text, "%Y-%m-%d")
    if end_of_day:
        dt = dt.replace(hour=21, minute=0, second=0)
    delta = dt - KNOWAIR_START_DATETIME
    total_hours = delta.days * 24 + delta.seconds // 3600
    if total_hours % KNOWAIR_FREQ_HOURS != 0:
        raise ValueError(f"Date {date_text} is not aligned to 3-hour grid.")
    return total_hours // KNOWAIR_FREQ_HOURS


def build_knowair_time_features(datetimes: list[datetime], mode: str = "cyclic6") -> np.ndarray:
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


def load_knowair_station_coords(path: Path, coord_order: str = "lonlat") -> tuple[np.ndarray, list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if coord_order not in {"lonlat", "latlon"}:
        raise ValueError(f"coord_order must be 'lonlat' or 'latlon', got {coord_order!r}")
    if coord_order == "lonlat":
        coords = np.asarray([[float(row["longitude"]), float(row["latitude"])] for row in rows], dtype=np.float32)
    else:
        coords = np.asarray([[float(row["latitude"]), float(row["longitude"])] for row in rows], dtype=np.float32)
    return coords, rows


class Dataset_KnowAir_PhysicsFirst(Dataset):
    def __init__(
        self,
        root_path: str | Path,
        flag: str = "train",
        dataset_num: int = 1,
        hist_len: int = 24,
        pred_len: int = 24,
        target_idx: int = 17,
        scale: bool = True,
        data_path: str = "KnowAir.npy",
        station_path: str = "station.csv",
        adj_geo_path: str = "final_adj.npy",
        graph_npz_path: str = "graph_data.npz",
        time_feature_mode: str = "cyclic6",
        wind_u_idx: int | None = None,
        wind_v_idx: int | None = None,
        coord_order: str = "lonlat",
    ) -> None:
        if flag not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported flag: {flag}")
        if dataset_num not in OFFICIAL_SPLITS:
            raise ValueError(f"Unsupported dataset_num: {dataset_num}")
        if hist_len <= 0 or pred_len <= 0:
            raise ValueError("hist_len and pred_len must be positive.")

        self.root_path = Path(root_path)
        self.flag = flag
        self.dataset_num = dataset_num
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.target_idx = target_idx
        self.scale = scale
        self.data_path = data_path
        self.station_path = station_path
        self.adj_geo_path = adj_geo_path
        self.graph_npz_path = graph_npz_path
        self.time_feature_mode = time_feature_mode
        self.wind_u_idx = wind_u_idx
        self.wind_v_idx = wind_v_idx
        self.coord_order = coord_order

        self._load_all()

    def _load_all(self) -> None:
        dynamic_path = self.root_path / self.data_path
        raw_data = np.load(dynamic_path).astype(np.float32)
        if raw_data.ndim != 3:
            raise ValueError(f"Expected 3D KnowAir tensor, got shape {raw_data.shape}")

        self.total_time_steps, self.num_nodes, self.num_features = raw_data.shape
        if not (0 <= self.target_idx < self.num_features):
            raise ValueError(f"target_idx {self.target_idx} out of range for {self.num_features} features")

        self.datetimes = build_knowair_datetime_axis(self.total_time_steps)
        self.datetime_strings = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in self.datetimes]

        self.coords_np, self.station_rows = load_knowair_station_coords(self.root_path / self.station_path, coord_order=self.coord_order)
        if self.coords_np.shape != (self.num_nodes, 2):
            raise ValueError(
                f"station.csv rows {self.coords_np.shape[0]} do not match KnowAir node count {self.num_nodes}"
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
        if not np.array_equal(graph_adj, self.adj_geo_np):
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

        train_range = OFFICIAL_SPLITS[self.dataset_num]["train"]
        self.train_start_idx = datetime_to_index(train_range.start_date, end_of_day=False)
        self.train_end_idx = datetime_to_index(train_range.end_date, end_of_day=True)
        train_block = raw_data[self.train_start_idx : self.train_end_idx + 1]

        self.scaler = None
        if self.scale:
            train_flat = train_block.reshape(-1, self.num_features)
            self.scaler = KnowAirStandardScaler().fit(train_flat)
            full_flat = raw_data.reshape(-1, self.num_features)
            scaled_flat = self.scaler.transform(full_flat)
            self.x_all = scaled_flat.reshape(self.total_time_steps, self.num_nodes, self.num_features).astype(np.float32)
        else:
            self.x_all = raw_data.astype(np.float32)

        raw_meteo_all = raw_data[:, :, self.meteo_cols].astype(np.float32)
        self.meteo_all_raw = raw_meteo_all
        meteo_train = raw_meteo_all[self.train_start_idx : self.train_end_idx + 1].reshape(-1, self.meteo_dim)
        self.meteo_train_mean = meteo_train.mean(axis=0).astype(np.float32)
        self.meteo_train_std = meteo_train.std(axis=0).astype(np.float32)
        self.meteo_train_std[self.meteo_train_std == 0] = 1.0
        self.meteo_all_norm = ((self.meteo_all_raw - self.meteo_train_mean) / self.meteo_train_std).astype(np.float32)

        self.time_feat_series = build_knowair_time_features(self.datetimes, mode=self.time_feature_mode)
        self.time_dim = int(self.time_feat_series.shape[-1])

        self.target_mean = (
            float(self.scaler.mean_[self.target_idx]) if self.scaler is not None else 0.0
        )
        self.target_std = (
            float(self.scaler.scale_[self.target_idx]) if self.scaler is not None else 1.0
        )

        split_range = OFFICIAL_SPLITS[self.dataset_num][self.flag]
        self.split_start_idx = datetime_to_index(split_range.start_date, end_of_day=False)
        self.split_end_idx = datetime_to_index(split_range.end_date, end_of_day=True)

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


class Dataset_KnowAir_ForWind(Dataset_KnowAir_PhysicsFirst):
    """
    KnowAir dataset wrapper for the wind-consistency final branch.

    Returned sample keys keep the physics-first tensors that are still useful
    (`x_enc`, `future_y`, `meteo_future`, `coords`, `adj_geo`) and additionally
    expose `seq_y` in the old PMO-Net shape convention:
        seq_y = concat(last label_len target steps, future target steps)
    """

    def __init__(
        self,
        root_path: str | Path,
        flag: str = "train",
        dataset_num: int = 1,
        size: list[int] | tuple[int, int, int] | None = None,
        features: str = "MS",
        seq_len: int = 24,
        label_len: int = 12,
        pred_len: int = 24,
        target_idx: int = 17,
        scale: bool = True,
        data_path: str = "KnowAir.npy",
        station_path: str = "station.csv",
        adj_geo_path: str = "final_adj.npy",
        graph_npz_path: str = "graph_data.npz",
        time_feature_mode: str = "cyclic6",
        wind_u_idx: int | None = None,
        wind_v_idx: int | None = None,
        coord_order: str = "lonlat",
    ) -> None:
        if size is not None:
            if len(size) != 3:
                raise ValueError(f"size must be [seq_len, label_len, pred_len], got {size}")
            seq_len, label_len, pred_len = int(size[0]), int(size[1]), int(size[2])
        if features != "MS":
            raise ValueError("Dataset_KnowAir_ForWind currently supports features='MS' only.")
        if label_len <= 0 or label_len > seq_len:
            raise ValueError(f"label_len must be in [1, seq_len], got label_len={label_len}, seq_len={seq_len}")

        self.features = features
        self.seq_len = int(seq_len)
        self.label_len = int(label_len)

        super().__init__(
            root_path=root_path,
            flag=flag,
            dataset_num=dataset_num,
            hist_len=self.seq_len,
            pred_len=int(pred_len),
            target_idx=target_idx,
            scale=scale,
            data_path=data_path,
            station_path=station_path,
            adj_geo_path=adj_geo_path,
            graph_npz_path=graph_npz_path,
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


def custom_collate_fn_knowair_physics_first(batch: list[dict[str, Any]]) -> dict[str, Any]:
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


TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
BEIJING_FREQ_HOURS = 3


@dataclass(frozen=True)
class BeijingSplitDateRange:
    start_datetime: str
    end_datetime: str


BEIJING_SPLITS: dict[str, SplitDateRange] = {
    "train": BeijingSplitDateRange("2017-01-01 15:00:00", "2017-11-30 21:00:00"),
    "val": BeijingSplitDateRange("2017-12-01 00:00:00", "2017-12-31 21:00:00"),
    "test": BeijingSplitDateRange("2018-01-01 00:00:00", "2018-01-31 12:00:00"),
}


class BeijingStandardScaler:
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


def build_beijing_time_features(datetimes: list[datetime], mode: str = "cyclic6") -> np.ndarray:
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


def load_beijing_station_coords(path: Path, coord_order: str = "lonlat") -> tuple[np.ndarray, list[dict[str, str]]]:
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

        self.coords_np, self.station_rows = load_beijing_station_coords(self.root_path / self.station_path, coord_order=self.coord_order)
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
            self.scaler = BeijingStandardScaler().fit(train_flat)
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

        self.time_feat_series = build_beijing_time_features(self.datetimes, mode=self.time_feature_mode)
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


def build_dataset(dataset: str, split: str, cfg: dict[str, Any]):
    name = dataset.lower()
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    if name == "gansuair":
        args = SimpleNamespace(
            seq_len=int(model_cfg.get("seq_len", 24)),
            label_len=int(model_cfg.get("label_len", 12)),
            pred_len=int(model_cfg.get("pred_len", 24)),
        )
        ds = Dataset_RIMST_PhysicsFirst(
            args=args,
            root_path=data_cfg.get("root_path", "./data"),
            flag=split,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=model_cfg.get("features", "MS"),
            data_path=data_cfg.get("data_path", "Gansu_Air.csv"),
            poi_path=data_cfg.get("poi_path", "poi_attribute_adj.npy"),
            landuse_path=data_cfg.get("landuse_path", "landuse_attribute_adj.npy"),
            adj_geo_path=data_cfg.get("adj_geo_path", "knn_adj.csv"),
            meteo_path=data_cfg.get("meteo_path", "meteo_physics_first.npy"),
            meteo_columns_path=data_cfg.get("meteo_columns_path", "meteo_physics_first_columns.json"),
            meteo_station_order_path=data_cfg.get("meteo_station_order_path", "meteo_physics_first_station_order.json"),
            meteo_datetime_path=data_cfg.get("meteo_datetime_path", "meteo_physics_first_datetime.csv"),
            return_meteo_pair=True,
        )
        columns_path = Path(data_cfg.get("meteo_columns_path", "meteo_physics_first_columns.json"))
        if not columns_path.is_absolute():
            columns_path = Path(data_cfg.get("root_path", "./data")) / columns_path
        if columns_path.exists():
            columns = json.loads(columns_path.read_text(encoding="utf-8"))
            lower_to_idx = {str(name).lower(): idx for idx, name in enumerate(columns)}
            if "u10" in lower_to_idx and "v10" in lower_to_idx:
                ds.wind_u_meteo_pos = lower_to_idx["u10"]
                ds.wind_v_meteo_pos = lower_to_idx["v10"]
                ds.meteo_columns = columns
        return ds, custom_collate_fn_physics_first
    if name == "knowair":
        target_idx = model_cfg.get("target_idx", 17)
        wind_u_idx = model_cfg.get("wind_u_idx")
        wind_v_idx = model_cfg.get("wind_v_idx")
        metadata_path = Path(data_cfg.get("root_path", "./data/knowair")) / data_cfg.get(
            "feature_metadata_path", "knowair_feature_columns.json"
        )
        if metadata_path.exists():
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            feature_names = payload.get("feature_names", payload) if isinstance(payload, dict) else payload
            index_map = {str(name): idx for idx, name in enumerate(feature_names)}
            if target_idx == "auto":
                for candidate in ("PM2.5", "pm25", "PM25", "pm2.5"):
                    if candidate in index_map:
                        target_idx = index_map[candidate]
                        break
            if wind_u_idx is None or wind_v_idx is None:
                source_to_features = {
                    "u950_v950": ("u_component_of_wind+950", "v_component_of_wind+950"),
                    "100m": ("100m_u_component_of_wind", "100m_v_component_of_wind"),
                }
                wind_source = model_cfg.get("wind_source", "u950_v950")
                if wind_source in source_to_features:
                    u_name, v_name = source_to_features[wind_source]
                    wind_u_idx = index_map.get(u_name)
                    wind_v_idx = index_map.get(v_name)
        if target_idx == "auto":
            target_idx = 17
        ds = Dataset_KnowAir_ForWind(
            root_path=data_cfg.get("root_path", "./data/KnowAir"),
            flag=split,
            dataset_num=int(model_cfg.get("dataset_num", 1)),
            seq_len=int(model_cfg.get("seq_len", 24)),
            label_len=int(model_cfg.get("label_len", 12)),
            pred_len=int(model_cfg.get("pred_len", 24)),
            target_idx=int(target_idx),
            data_path=data_cfg.get("data_path", "KnowAir.npy"),
            station_path=data_cfg.get("station_path", "station.csv"),
            adj_geo_path=data_cfg.get("adj_geo_path", "final_adj.npy"),
            graph_npz_path=data_cfg.get("graph_npz_path", "graph_data.npz"),
            wind_u_idx=wind_u_idx,
            wind_v_idx=wind_v_idx,
        )
        return ds, custom_collate_fn_knowair_physics_first
    if name == "beijing":
        ds = Dataset_Beijing_ForWind(
            root_path=data_cfg.get("root_path", "./data/Beijing1718/processed"),
            flag=split,
            seq_len=int(model_cfg.get("seq_len", model_cfg.get("hist_len", 24))),
            label_len=int(model_cfg.get("label_len", 12)),
            pred_len=int(model_cfg.get("pred_len", 24)),
            target_idx=int(model_cfg.get("target_idx", 0)),
            data_path=data_cfg.get("data_path", "Beijing.npy"),
            station_path=data_cfg.get("station_path", "station.csv"),
            adj_geo_path=data_cfg.get("adj_geo_path", "final_adj.npy"),
            graph_npz_path=data_cfg.get("graph_npz_path", "graph_data.npz"),
            metadata_path=data_cfg.get("feature_metadata_path", data_cfg.get("metadata_path", "beijing_feature_columns.json")),
            timestamps_path=data_cfg.get("timestamps_path", "timestamps.csv"),
            wind_u_idx=model_cfg.get("wind_u_idx"),
            wind_v_idx=model_cfg.get("wind_v_idx"),
        )
        return ds, custom_collate_fn_beijing_physics_first
    raise ValueError(f"Unsupported dataset: {dataset}")
