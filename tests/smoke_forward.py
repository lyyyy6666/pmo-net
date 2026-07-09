from __future__ import annotations

import torch

from pmonet.models import xLSTM_WindDualODE_Mixer


def main() -> None:
    torch.manual_seed(7)
    batch_size = 2
    seq_len = 4
    pred_len = 3
    num_nodes = 5
    enc_in = 4

    adj = torch.eye(num_nodes)
    coords = torch.stack([torch.linspace(100.0, 101.0, num_nodes), torch.linspace(35.0, 36.0, num_nodes)], dim=-1)
    model = xLSTM_WindDualODE_Mixer(
        pred_len=pred_len,
        seq_len=seq_len,
        enc_in=enc_in,
        dec_out=enc_in,
        num_nodes=num_nodes,
        d_model=16,
        latent_dim=8,
        xlstm_num_blocks=1,
        xlstm_num_heads=2,
        channel_mixer_hidden=32,
        use_static_context=False,
        use_spatial_mixing=False,
        adj_mx=adj,
        coords=coords,
        use_wind_advection=False,
        require_xlstm_backend=False,
    )
    x = torch.randn(batch_size, seq_len, num_nodes, enc_in)
    y = model(x)
    assert y.shape == (batch_size, pred_len, num_nodes, enc_in), y.shape
    assert torch.isfinite(y).all()
    print(f"smoke_forward passed: output_shape={tuple(y.shape)}")


if __name__ == "__main__":
    main()
