from __future__ import annotations

import torch
from torch import nn


class BaseModel(nn.Module):
    """Minimal base class used by the PMO-Net release models."""

    def __init__(self, seq_len: int = 96, pred_len: int = 96, enc_in: int = 1) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in

    def forecast(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor | None = None):
        raise NotImplementedError
