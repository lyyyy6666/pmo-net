from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def load_coordinates(path: Path, lon_col: str, lat_col: str) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty station file: {path}")
    coords = np.asarray([[float(row[lon_col]), float(row[lat_col])] for row in rows], dtype=np.float64)
    return coords


def haversine_distance_km(coords: np.ndarray) -> np.ndarray:
    lon = np.deg2rad(coords[:, 0])
    lat = np.deg2rad(coords[:, 1])
    dlon = lon[:, None] - lon[None, :]
    dlat = lat[:, None] - lat[None, :]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def build_knn_adjacency(coords: np.ndarray, k: int, sigma_km: float | None = None, symmetric: bool = True) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive.")
    n_nodes = coords.shape[0]
    if k >= n_nodes:
        raise ValueError(f"k={k} must be smaller than number of nodes={n_nodes}.")

    dist = haversine_distance_km(coords)
    if sigma_km is None:
        nonzero = dist[dist > 0]
        sigma_km = float(np.median(nonzero)) if nonzero.size else 1.0
    sigma_km = max(float(sigma_km), 1e-6)

    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for i in range(n_nodes):
        neighbors = np.argsort(dist[i])[1 : k + 1]
        weights = np.exp(-dist[i, neighbors] / sigma_km).astype(np.float32)
        adj[i, neighbors] = weights
    if symmetric:
        adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0.0)
    return adj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a KNN geographic adjacency matrix from station coordinates.")
    parser.add_argument("--station_csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lon_col", type=str, default="longitude")
    parser.add_argument("--lat_col", type=str, default="latitude")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--sigma_km", type=float, default=None)
    parser.add_argument("--directed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coords = load_coordinates(args.station_csv, args.lon_col, args.lat_col)
    adj = build_knn_adjacency(coords, k=args.k, sigma_km=args.sigma_km, symmetric=not args.directed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".npy":
        np.save(args.output, adj)
    else:
        np.savetxt(args.output, adj, delimiter=",", fmt="%.8f")
    print(f"wrote {args.output} shape={adj.shape}")


if __name__ == "__main__":
    main()
