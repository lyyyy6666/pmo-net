from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
    from xlstm import sLSTMBlockConfig, sLSTMLayerConfig
    XLSTM_BACKEND_AVAILABLE = True
except Exception:
    xLSTMBlockStack = None
    xLSTMBlockStackConfig = None
    sLSTMBlockConfig = None
    sLSTMLayerConfig = None
    XLSTM_BACKEND_AVAILABLE = False

from layers.graph_layers import MultiGraphConv, Normalize as RevIN, StaticContextEmbedding
from layers.gating import AdaptiveGating
from layers.ode_layers import DataDrivenODEFunc, WindAwarePhysicalODEFunc


class BaseModel(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, enc_in: int):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in


class xLSTM_WindDualODE_Mixer(BaseModel):
    def __init__(
        self,
        pred_len: int,
        seq_len: int,
        enc_in: int,
        dec_out: int = None,
        num_nodes: int = 23,
        static_feat_dim: int = 50,
        d_model: int = 256,
        xlstm_num_blocks: int = 3,
        xlstm_num_heads: int = 8,
        xlstm_dropout: float = 0.1,
        channel_mixer_hidden: int = 512,
        use_static_context: bool = True,
        use_spatial_mixing: bool = True,
        fusion_type: str = "attention",
        use_sparse: bool = True,
        use_phy_ode: bool = True,
        use_unk_ode: bool = True,
        use_adaptive_gating: bool = True,
        gating_hidden_dim: int = 64,
        latent_dim: int = 128,
        ode_hidden_dim: int = 64,
        adj_mx: Optional[torch.Tensor] = None,
        adj_geo: Optional[torch.Tensor] = None,
        adj_poi: Optional[torch.Tensor] = None,
        adj_land: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        wind_u_idx: Optional[int] = None,
        wind_v_idx: Optional[int] = None,
        use_wind_advection: bool = False,
        distance_scale_km: float = 50.0,
        coords_are_latlon: bool = True,
        use_geo_mask_for_wind: bool = True,
        wind_graph_norm: str = "row",
        eps: float = 1e-6,
        require_xlstm_backend: bool = True,
        dt_hours: float = 1.0,
        latent_dt_mode: str = "normalized",
    ):
        super().__init__(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in)

        self.num_nodes = num_nodes
        self.d_model = d_model
        self.enc_in = enc_in
        self.dec_out = dec_out if dec_out is not None else enc_in
        self.latent_dim = latent_dim
        self.use_phy_ode = use_phy_ode
        self.use_unk_ode = use_unk_ode
        self.use_adaptive_gating = use_adaptive_gating
        self.use_wind_advection = use_wind_advection
        self.require_xlstm_backend = require_xlstm_backend
        self.dt_hours = float(dt_hours)
        self.latent_dt_mode = latent_dt_mode
        self.wind_graph_norm = wind_graph_norm
        self.context_extractor_type = None

        if self.latent_dt_mode not in {"normalized", "physical"}:
            raise ValueError(f"latent_dt_mode must be 'normalized' or 'physical', got {latent_dt_mode!r}")

        if self.require_xlstm_backend and not XLSTM_BACKEND_AVAILABLE:
            raise ImportError("xLSTM backend is required but not available.")

        self.reversible_instance_norm = RevIN(enc_in, affine=True)

        if use_static_context:
            self.static_context_embedding = StaticContextEmbedding(
                num_nodes=num_nodes,
                static_feat_dim=static_feat_dim,
                d_model=d_model,
                hidden_dim=128,
                dropout=xlstm_dropout,
            )
        else:
            self.static_context_embedding = None

        self.dynamic_embedding = nn.Linear(enc_in, d_model)

        if XLSTM_BACKEND_AVAILABLE:
            slstm_config = sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    num_heads=xlstm_num_heads,
                    conv1d_kernel_size=0,
                )
            )
            self.context_extractor = xLSTMBlockStack(
                xLSTMBlockStackConfig(
                    slstm_block=slstm_config,
                    num_blocks=xlstm_num_blocks,
                    embedding_dim=d_model,
                    add_post_blocks_norm=True,
                    dropout=xlstm_dropout,
                    bias=True,
                    slstm_at="all",
                    context_length=seq_len,
                )
            )
            self.context_extractor_type = type(self.context_extractor).__name__
        else:
            self.context_extractor = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=max(1, xlstm_num_blocks),
                batch_first=True,
                dropout=xlstm_dropout if xlstm_num_blocks > 1 else 0.0,
            )
            self.context_extractor_type = type(self.context_extractor).__name__

        self.z0_projection = nn.Sequential(
            nn.Linear(d_model, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        self.context_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        if use_phy_ode:
            if adj_mx is None:
                adj_mx = torch.ones(num_nodes, num_nodes) / num_nodes
            self.phy_ode_func = WindAwarePhysicalODEFunc(
                latent_dim=latent_dim,
                num_nodes=num_nodes,
                adj_mx=adj_mx,
                hidden_dim=ode_hidden_dim,
                coords=coords,
                wind_u_idx=wind_u_idx,
                wind_v_idx=wind_v_idx,
                use_wind_advection=use_wind_advection,
                distance_scale_km=distance_scale_km,
                coords_are_latlon=coords_are_latlon,
                use_geo_mask_for_wind=use_geo_mask_for_wind,
                wind_graph_norm=wind_graph_norm,
                eps=eps,
            )

        if use_unk_ode:
            self.unk_ode_func = DataDrivenODEFunc(
                latent_dim=latent_dim,
                num_nodes=num_nodes,
                context_dim=d_model,
                hidden_dim=ode_hidden_dim,
            )

        if use_phy_ode and use_unk_ode and use_adaptive_gating:
            self.adaptive_gating = AdaptiveGating(
                latent_dim=latent_dim,
                num_nodes=num_nodes,
                hidden_dim=gating_hidden_dim,
            )
        else:
            self.adaptive_gating = None

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        if use_spatial_mixing and adj_geo is not None:
            if len(adj_geo.shape) == 3:
                adj_geo = adj_geo[0]
            if adj_poi is not None and len(adj_poi.shape) == 3:
                adj_poi = adj_poi[0]
            if adj_land is not None and len(adj_land.shape) == 3:
                adj_land = adj_land[0]

            if adj_poi is None:
                adj_poi = adj_geo
            if adj_land is None:
                adj_land = adj_geo

            self.spatial_mixer = MultiGraphConv(
                in_dim=d_model,
                out_dim=d_model,
                num_nodes=num_nodes,
                adj_geo=adj_geo,
                adj_poi=adj_poi,
                adj_land=adj_land,
                dropout=xlstm_dropout,
                activation="gelu",
                fusion_type=fusion_type,
                use_sparse=use_sparse,
                normalize_adj=True,
            )
        else:
            self.spatial_mixer = None

        self.channel_mixer = nn.Sequential(
            nn.Linear(d_model, channel_mixer_hidden),
            nn.GELU(),
            nn.Dropout(xlstm_dropout),
            nn.Linear(channel_mixer_hidden, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(xlstm_dropout),
        )

        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(xlstm_dropout),
            nn.Linear(d_model // 2, self.dec_out),
        )

        self.time_embedding = nn.Parameter(torch.randn(pred_len, d_model) * 0.02)

    def _get_latent_dt(self, time_steps: torch.Tensor) -> torch.Tensor:
        if self.latent_dt_mode == "normalized":
            if time_steps.numel() > 1:
                return time_steps[1] - time_steps[0]
            return time_steps.new_tensor(1.0)
        return time_steps.new_tensor(self.dt_hours)

    def solve_fused_ode(
        self,
        z0: torch.Tensor,
        time_steps: torch.Tensor,
        context: torch.Tensor,
        meteo_future: Optional[torch.Tensor] = None,
        collect_debug: bool = False,
    ) -> torch.Tensor:
        batch_size = z0.shape[0]
        total_steps = len(time_steps)
        dt = self._get_latent_dt(time_steps)

        if self.use_phy_ode and self.use_wind_advection:
            if meteo_future is None:
                raise ValueError("meteo_future must be provided when use_wind_advection=True.")
            expected_shape = (batch_size, total_steps - 1, self.num_nodes)
            if meteo_future.dim() != 4:
                raise ValueError(
                    "meteo_future must have shape (B, pred_len, N, meteo_dim) when use_wind_advection=True."
                )
            if meteo_future.shape[0] != expected_shape[0] or meteo_future.shape[1] != expected_shape[1]:
                raise ValueError(
                    f"Expected meteo_future leading shape {(expected_shape[0], expected_shape[1])}, "
                    f"got {(meteo_future.shape[0], meteo_future.shape[1])}"
                )
            if meteo_future.shape[2] != self.num_nodes:
                raise ValueError(f"Expected meteo_future.shape[2] == {self.num_nodes}, got {meteo_future.shape[2]}")

        trajectory = [z0]
        z = z0
        debug_records: list[dict[str, float]] | None = [] if collect_debug else None
        for step_idx in range(1, total_steps):
            if self.use_phy_ode and self.use_unk_ode:
                if self.use_adaptive_gating and self.adaptive_gating is not None:
                    alpha_t = self.adaptive_gating(z)
                else:
                    alpha_t = torch.ones_like(z) * 0.5
            elif self.use_phy_ode:
                alpha_t = torch.ones_like(z)
            else:
                alpha_t = torch.zeros_like(z)

            if self.use_phy_ode:
                meteo_t = meteo_future[:, step_idx - 1] if meteo_future is not None else None
                dz_phy = self.phy_ode_func(z, meteo_t=meteo_t, collect_debug=collect_debug)
            else:
                dz_phy = torch.zeros_like(z)

            if self.use_unk_ode:
                z_nodes = z.reshape(batch_size * self.num_nodes, self.latent_dim)
                dz_data_nodes = self.unk_ode_func(z_nodes, context)
                dz_data = dz_data_nodes.reshape(batch_size, self.num_nodes * self.latent_dim)
            else:
                dz_data = torch.zeros_like(z)

            dz_fused = alpha_t * dz_phy + (1.0 - alpha_t) * dz_data
            if collect_debug and debug_records is not None:
                if self.use_phy_ode:
                    phy_debug = self.phy_ode_func.last_debug
                else:
                    phy_debug = {}
                debug_records.append(
                    {
                        "alpha_mean": float(alpha_t.mean().detach().cpu()),
                        "alpha_min": float(alpha_t.min().detach().cpu()),
                        "alpha_max": float(alpha_t.max().detach().cpu()),
                        "diffusion_abs_mean": float(phy_debug.get("diffusion_abs_mean", 0.0)),
                        "advection_abs_mean": float(phy_debug.get("adv_abs_mean", 0.0)),
                        "reaction_abs_mean": float(phy_debug.get("reaction_abs_mean", 0.0)),
                        "decay_abs_mean": float(phy_debug.get("decay_abs_mean", 0.0)),
                        "dz_phy_norm": float(phy_debug.get("dz_phy_norm", 0.0)),
                        "dz_data_norm": float(dz_data.norm(dim=-1).mean().detach().cpu()),
                        "dz_fused_norm": float(dz_fused.norm(dim=-1).mean().detach().cpu()),
                        "A_w_mean": float(phy_debug.get("A_w_mean", 0.0)),
                        "A_w_max": float(phy_debug.get("A_w_max", 0.0)),
                        "A_w_row_sum_mean": float(phy_debug.get("A_w_row_sum_mean", 0.0)),
                        "A_w_row_sum_max": float(phy_debug.get("A_w_row_sum_max", 0.0)),
                    }
                )
            z = z + dt * dz_fused
            trajectory.append(z)

        if collect_debug and debug_records:
            self.last_fused_debug = {
                key: float(sum(record[key] for record in debug_records) / len(debug_records))
                for key in debug_records[0]
            }
        return torch.stack(trajectory, dim=0)

    def forecast(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor] = None,
        static_feat: Optional[torch.Tensor] = None,
        target_idx: Optional[int] = None,
        meteo_future: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ):
        batch_size, seq_len, num_nodes, num_channels = x_enc.shape

        x_reshaped = x_enc.permute(0, 2, 1, 3).contiguous()
        x_reshaped = x_reshaped.reshape(batch_size * num_nodes, seq_len, num_channels)
        x_norm = self.reversible_instance_norm(x_reshaped, "norm")
        x_norm = x_norm.reshape(batch_size, num_nodes, seq_len, num_channels).permute(0, 2, 1, 3).contiguous()

        x_embed = self.dynamic_embedding(x_norm)

        if self.static_context_embedding is not None and static_feat is not None:
            static_embed = self.static_context_embedding(static_feat)
            static_broadcast = self.static_context_embedding.broadcast(static_embed, batch_size, seq_len)
            x_embed = x_embed + static_broadcast

        x_seq_flat = x_embed.permute(0, 2, 1, 3).contiguous().reshape(batch_size * num_nodes, seq_len, self.d_model)
        if XLSTM_BACKEND_AVAILABLE:
            xlstm_out = self.context_extractor(x_seq_flat)
        else:
            xlstm_out, _ = self.context_extractor(x_seq_flat)
        h_t = xlstm_out[:, -1, :]

        z0_nodes = self.z0_projection(h_t)
        z0 = z0_nodes.reshape(batch_size, num_nodes * self.latent_dim)

        c_ctx = self.context_projection(h_t)

        time_steps = torch.linspace(0, 1, self.pred_len + 1, device=x_enc.device)
        fused_z = self.solve_fused_ode(
            z0,
            time_steps,
            c_ctx,
            meteo_future=meteo_future,
            collect_debug=return_debug,
        )[1:]

        fused_z_reshaped = fused_z.reshape(self.pred_len, batch_size, num_nodes, self.latent_dim)
        fused_z_reshaped = fused_z_reshaped.permute(1, 0, 2, 3).contiguous()
        fused_z_flat = fused_z_reshaped.reshape(batch_size * num_nodes, self.pred_len, self.latent_dim)

        decoded = self.decoder(fused_z_flat)
        decoded = decoded.reshape(batch_size, num_nodes, self.pred_len, self.d_model)
        decoded = decoded.permute(0, 2, 1, 3).contiguous()

        x_channel = self.channel_mixer(decoded)
        x_channel = x_channel + decoded

        if self.spatial_mixer is not None:
            x_spatial = x_channel.permute(0, 2, 1, 3).contiguous()
            x_spatial = x_spatial.reshape(batch_size * self.pred_len, num_nodes, self.d_model)
            x_spatial = self.spatial_mixer(x_spatial)
            x_spatial = x_spatial.reshape(batch_size, self.pred_len, num_nodes, self.d_model)
            x_spatial = x_spatial + x_channel
        else:
            x_spatial = x_channel

        time_emb = self.time_embedding.unsqueeze(0).unsqueeze(2).expand(batch_size, self.pred_len, num_nodes, self.d_model)
        x_with_time = x_spatial + time_emb

        output = self.output_projection(x_with_time)

        output_reshaped = output.permute(0, 2, 1, 3).contiguous()
        output_reshaped = output_reshaped.reshape(batch_size * num_nodes, self.pred_len, self.dec_out)
        output_denorm = self.reversible_instance_norm(output_reshaped, "denorm")

        if self.dec_out == 1 and output_denorm.shape[-1] > 1:
            idx = target_idx if target_idx is not None else 0
            output_denorm = output_denorm[..., idx : idx + 1]

        output_final = output_denorm.reshape(batch_size, num_nodes, self.pred_len, self.dec_out)
        output_final = output_final.permute(0, 2, 1, 3).contiguous()

        if return_debug:
            latent_dt = self._get_latent_dt(time_steps)
            debug = {}
            if self.use_phy_ode:
                debug.update(self.phy_ode_func.last_debug)
            debug.update(getattr(self, "last_fused_debug", {}))
            debug["dt_hours"] = self.dt_hours
            debug["latent_dt_mode"] = self.latent_dt_mode
            debug["latent_dt"] = float(latent_dt.detach().cpu())
            debug["coords_are_latlon"] = self.phy_ode_func.coords_are_latlon if self.use_phy_ode else None
            debug["context_extractor_type"] = self.context_extractor_type
            debug["XLSTM_BACKEND_AVAILABLE"] = XLSTM_BACKEND_AVAILABLE
            debug["require_xlstm_backend"] = self.require_xlstm_backend
            debug["wind_graph_norm"] = self.wind_graph_norm
            debug["pairwise_dist_min"] = (
                float(self.phy_ode_func.pairwise_dist_km[self.phy_ode_func.non_self_mask.bool()].min().detach().cpu())
                if self.use_phy_ode
                else 0.0
            )
            debug["pairwise_dist_max"] = (
                float(self.phy_ode_func.pairwise_dist_km[self.phy_ode_func.non_self_mask.bool()].max().detach().cpu())
                if self.use_phy_ode
                else 0.0
            )
            debug["pairwise_dist_mean"] = (
                float(self.phy_ode_func.pairwise_dist_km[self.phy_ode_func.non_self_mask.bool()].mean().detach().cpu())
                if self.use_phy_ode
                else 0.0
            )
            return output_final, debug
        return output_final

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor] = None,
        x_dec: Optional[torch.Tensor] = None,
        x_mark_dec: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        static_feat: Optional[torch.Tensor] = None,
        adj_geo: Optional[torch.Tensor] = None,
        adj_poi: Optional[torch.Tensor] = None,
        adj_land: Optional[torch.Tensor] = None,
        meteo_future: Optional[torch.Tensor] = None,
        target_idx: Optional[int] = None,
        return_debug: bool = False,
    ):
        return self.forecast(
            x_enc=x_enc,
            x_mark_enc=x_mark_enc,
            static_feat=static_feat,
            target_idx=target_idx,
            meteo_future=meteo_future,
            return_debug=return_debug,
        )
