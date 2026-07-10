from __future__ import annotations

import numpy as np


def calculate_metrics(pred, true, threshold: float = 0.1) -> dict[str, float]:
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
