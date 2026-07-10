from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class PMONetLoss(nn.Module):
    """Default three-term objective used for PMO-Net training."""

    def __init__(
        self,
        pred_len: int,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        beta: float = 0.05,
        lambda_nonnegative: float = 0.1,
        lambda_temporal_smooth: float = 0.01,
    ) -> None:
        super().__init__()
        self.lambda_nonnegative = float(lambda_nonnegative)
        self.lambda_temporal_smooth = float(lambda_temporal_smooth)
        self.register_buffer("target_mean", torch.tensor(float(target_mean)).view(1, 1, 1, 1))
        self.register_buffer("target_std", torch.tensor(float(target_std)).view(1, 1, 1, 1))
        weights = torch.exp(-torch.arange(pred_len, dtype=torch.float32) * float(beta))
        self.register_buffer("time_weights", weights / weights.sum())

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
        output_real = output * self.target_std + self.target_mean
        target_real = target * self.target_std + self.target_mean

        l_pred = (((output - target) ** 2) * self.time_weights.view(1, -1, 1, 1)).mean()
        l_nonneg = F.relu(-output_real).mean()
        if output.shape[1] > 1:
            l_smooth = ((output[:, 1:] - output[:, :-1]) ** 2).mean()
        else:
            l_smooth = output.new_tensor(0.0)

        total = l_pred + self.lambda_nonnegative * l_nonneg + self.lambda_temporal_smooth * l_smooth
        return {
            "total": total,
            "pred": l_pred.detach(),
            "nonnegative": l_nonneg.detach(),
            "smooth": l_smooth.detach(),
            "output_real": output_real.detach(),
            "target_real": target_real.detach(),
        }
