from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

from ..modules.multigraph_conv import MultiGraphConv
from ..modules.spatial_modules import StaticContextEmbedding
from ..modules.revin import Normalize as RevIN
from .base_model import BaseModel


class WindAwarePhysicalODEFunc(nn.Module):
    """
    Wind-aware topology-constrained latent dynamics.

    Base latent dynamics:
        dZ/dt = -k_diff * L_geo * Z + R_theta(Z) - gamma * Z

    Wind-aware extension:
        dZ/dt = -k_diff * L_geo * Z
              + k_adv * (A_w(t)^T Z - D_out(t) Z)
              + R_theta(Z)
              - gamma * Z

    Note:
    - This acts on latent state Z, not directly on observable pollutant concentration.
    - The advection term is a wind-aware directed transport prior over latent topology.
    """

    def __init__(
        self,
        latent_dim: int,
        num_nodes: int,
        adj_mx: torch.Tensor,
        hidden_dim: int = 64,
        coords: Optional[torch.Tensor] = None,
        wind_u_idx: Optional[int] = None,
        wind_v_idx: Optional[int] = None,
        use_wind_advection: bool = False,
        distance_scale_km: float = 50.0,
        coords_are_latlon: bool = True,
        use_geo_mask_for_wind: bool = True,
        wind_graph_norm: str = "row",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        self.wind_u_idx = wind_u_idx
        self.wind_v_idx = wind_v_idx
        self.use_wind_advection = use_wind_advection
        self.distance_scale_km = float(distance_scale_km)
        self.coords_are_latlon = coords_are_latlon
        self.use_geo_mask_for_wind = use_geo_mask_for_wind
        self.wind_graph_norm = wind_graph_norm
        self.eps = float(eps)

        if self.wind_graph_norm not in {"none", "row"}:
            raise ValueError(f"wind_graph_norm must be 'none' or 'row', got {wind_graph_norm!r}")

        adj_mx = adj_mx.float()
        if adj_mx.dim() == 3:
            adj_mx = adj_mx[0]

        laplacian = self._compute_normalized_laplacian(adj_mx)
        self.register_buffer("laplacian", laplacian)

        if coords is None:
            if use_wind_advection:
                raise ValueError("coords must be provided when use_wind_advection=True.")
            coords = torch.zeros(num_nodes, 2, dtype=torch.float32)
        coords = coords.float()
        if coords.dim() != 2 or coords.shape != (num_nodes, 2):
            raise ValueError(f"coords must have shape {(num_nodes, 2)}, got {tuple(coords.shape)}")
        coords_xy = self._convert_coords_to_xy_km(coords)
        pairwise_dist_km, pairwise_direction = self._build_pairwise_geometry(coords_xy)
        non_self_mask = 1.0 - torch.eye(num_nodes, dtype=coords_xy.dtype, device=coords_xy.device)
        geo_mask = self._build_geo_mask(adj_mx, non_self_mask)

        self.register_buffer("coords_xy_km", coords_xy)
        self.register_buffer("pairwise_dist_km", pairwise_dist_km)
        self.register_buffer("pairwise_direction", pairwise_direction)
        self.register_buffer("non_self_mask", non_self_mask)
        self.register_buffer("geo_mask", geo_mask)

        self.raw_diffusion_coeff = nn.Parameter(self._inverse_softplus(1e-2))
        self.raw_advection_coeff = nn.Parameter(self._inverse_softplus(1e-3))
        self.raw_decay_coeff = nn.Parameter(self._inverse_softplus(1e-3))

        self.reaction_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.last_debug: dict[str, torch.Tensor | float | bool] = {
            "use_wind_advection": bool(use_wind_advection),
            "symmetry_error": 0.0,
            "A_w_mean": 0.0,
            "A_w_max": 0.0,
            "A_w_row_sum_mean": 0.0,
            "A_w_row_sum_max": 0.0,
            "adv_abs_mean": 0.0,
            "diffusion_abs_mean": 0.0,
            "reaction_abs_mean": 0.0,
            "decay_abs_mean": 0.0,
            "dz_phy_norm": 0.0,
            "A_w_shape": None,
            "A_w_has_nan": False,
            "k_diff": 0.0,
            "k_adv": 0.0,
            "gamma": 0.0,
        }

    def _inverse_softplus(self, value: float) -> torch.Tensor:
        value_tensor = torch.tensor(float(value), dtype=torch.float32)
        return torch.log(torch.expm1(value_tensor))

    def _compute_normalized_laplacian(self, adj: torch.Tensor) -> torch.Tensor:
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree.clamp_min(self.eps), -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        d_inv_sqrt = torch.diag(degree_inv_sqrt)
        identity = torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
        return identity - d_inv_sqrt @ adj @ d_inv_sqrt

    def _convert_coords_to_xy_km(self, coords: torch.Tensor) -> torch.Tensor:
        if not self.coords_are_latlon:
            return coords
        lon = coords[:, 0]
        lat = coords[:, 1]
        lat_rad = lat * math.pi / 180.0
        lon_rad = lon * math.pi / 180.0
        lat0 = lat_rad.mean()
        earth_radius_km = 6371.0
        x = earth_radius_km * torch.cos(lat0) * lon_rad
        y = earth_radius_km * lat_rad
        return torch.stack([x, y], dim=-1)

    def _build_pairwise_geometry(self, coords_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        diff = coords_xy.unsqueeze(0) - coords_xy.unsqueeze(1)
        dist = torch.norm(diff, dim=-1).clamp_min(self.eps)
        direction = diff / dist.unsqueeze(-1)
        return dist, direction

    def get_positive_parameters(self) -> dict[str, torch.Tensor]:
        return {
            "k_diff": F.softplus(self.raw_diffusion_coeff),
            "k_adv": F.softplus(self.raw_advection_coeff),
            "gamma": F.softplus(self.raw_decay_coeff),
        }

    def _build_geo_mask(self, adj_mx: torch.Tensor, non_self_mask: torch.Tensor) -> torch.Tensor:
        if not self.use_geo_mask_for_wind:
            return non_self_mask
        adj_binary = (adj_mx > 0).to(dtype=adj_mx.dtype)
        adj_binary = torch.maximum(adj_binary, adj_binary.transpose(0, 1))
        return adj_binary * non_self_mask

    def build_wind_graph(self, meteo_t: torch.Tensor) -> torch.Tensor:
        if meteo_t is None:
            raise ValueError("meteo_t must be provided when use_wind_advection=True.")
        if self.wind_u_idx is None or self.wind_v_idx is None:
            raise ValueError("wind_u_idx and wind_v_idx must be provided when use_wind_advection=True.")
        if meteo_t.dim() != 3:
            raise ValueError(f"meteo_t must have shape (B, N, meteo_dim), got {tuple(meteo_t.shape)}")
        if meteo_t.shape[1] != self.num_nodes:
            raise ValueError(f"Expected meteo_t.shape[1] == {self.num_nodes}, got {meteo_t.shape[1]}")
        if (
            self.wind_u_idx < 0
            or self.wind_v_idx < 0
            or self.wind_u_idx >= meteo_t.shape[-1]
            or self.wind_v_idx >= meteo_t.shape[-1]
        ):
            raise ValueError(
                f"Wind indices ({self.wind_u_idx}, {self.wind_v_idx}) out of range for meteo_dim={meteo_t.shape[-1]}"
            )

        wind = torch.stack([meteo_t[..., self.wind_u_idx], meteo_t[..., self.wind_v_idx]], dim=-1)
        projection = torch.einsum("bik,ijk->bij", wind, self.pairwise_direction)
        directional_speed = F.relu(projection)
        distance_kernel = torch.exp(-self.pairwise_dist_km / max(self.distance_scale_km, self.eps))
        mask = self.geo_mask if self.use_geo_mask_for_wind else self.non_self_mask
        a_w = mask.unsqueeze(0) * distance_kernel.unsqueeze(0) * directional_speed
        if self.wind_graph_norm == "row":
            a_w = a_w / a_w.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return a_w

    def _compute_advection(
        self,
        x_reshaped: torch.Tensor,
        meteo_t: Optional[torch.Tensor],
        k_adv: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.use_wind_advection:
            zero_adv = torch.zeros_like(x_reshaped)
            return zero_adv, None

        a_w = self.build_wind_graph(meteo_t)
        inflow = torch.einsum("bji,bjd->bid", a_w, x_reshaped)
        outflow = a_w.sum(dim=-1).unsqueeze(-1) * x_reshaped
        advection_core = inflow - outflow
        advection = k_adv * advection_core
        return advection, a_w

    def _update_debug(
        self,
        a_w: Optional[torch.Tensor],
        diffusion: torch.Tensor,
        advection: torch.Tensor,
        reaction: torch.Tensor,
        decay: torch.Tensor,
        dz_phy: torch.Tensor,
        k_diff: torch.Tensor,
        k_adv: torch.Tensor,
        gamma: torch.Tensor,
    ) -> None:
        if a_w is None:
            self.last_debug = {
                "use_wind_advection": False,
                "symmetry_error": 0.0,
                "A_w_mean": 0.0,
                "A_w_max": 0.0,
                "A_w_row_sum_mean": 0.0,
                "A_w_row_sum_max": 0.0,
                "A_w_shape": None,
                "A_w_has_nan": False,
                "adv_abs_mean": float(advection.abs().mean().detach().cpu()),
                "diffusion_abs_mean": float(diffusion.abs().mean().detach().cpu()),
                "reaction_abs_mean": float(reaction.abs().mean().detach().cpu()),
                "decay_abs_mean": float(decay.abs().mean().detach().cpu()),
                "dz_phy_norm": float(dz_phy.norm(dim=-1).mean().detach().cpu()),
                "k_diff": float(k_diff.detach().cpu()),
                "k_adv": float(k_adv.detach().cpu()),
                "gamma": float(gamma.detach().cpu()),
            }
            return

        symmetry_error = (a_w - a_w.transpose(-1, -2)).abs().mean()
        row_sums = a_w.sum(dim=-1)
        self.last_debug = {
            "use_wind_advection": True,
            "A_w": a_w.detach(),
            "A_w_shape": tuple(a_w.shape),
            "A_w_has_nan": bool(torch.isnan(a_w).any().detach().cpu()),
            "symmetry_error": float(symmetry_error.detach().cpu()),
            "A_w_mean": float(a_w.mean().detach().cpu()),
            "A_w_max": float(a_w.max().detach().cpu()),
            "A_w_row_sum_mean": float(row_sums.mean().detach().cpu()),
            "A_w_row_sum_max": float(row_sums.max().detach().cpu()),
            "adv_abs_mean": float(advection.abs().mean().detach().cpu()),
            "diffusion_abs_mean": float(diffusion.abs().mean().detach().cpu()),
            "reaction_abs_mean": float(reaction.abs().mean().detach().cpu()),
            "decay_abs_mean": float(decay.abs().mean().detach().cpu()),
            "dz_phy_norm": float(dz_phy.norm(dim=-1).mean().detach().cpu()),
            "k_diff": float(k_diff.detach().cpu()),
            "k_adv": float(k_adv.detach().cpu()),
            "gamma": float(gamma.detach().cpu()),
        }

    def forward(
        self,
        x: torch.Tensor,
        meteo_t: Optional[torch.Tensor] = None,
        collect_debug: bool = False,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        x_reshaped = x.reshape(batch_size, self.num_nodes, self.latent_dim)
        params = self.get_positive_parameters()
        k_diff = params["k_diff"]
        k_adv = params["k_adv"]
        gamma = params["gamma"]

        lap_x = torch.einsum("ij,bjd->bid", self.laplacian, x_reshaped)
        diffusion = -k_diff * lap_x

        advection, a_w = self._compute_advection(x_reshaped, meteo_t, k_adv=k_adv)
        reaction = self.reaction_net(x_reshaped)
        decay = -gamma * x_reshaped

        dx = diffusion + advection + reaction + decay
        if collect_debug:
            self._update_debug(
                a_w,
                diffusion=diffusion,
                advection=advection,
                reaction=reaction,
                decay=decay,
                dz_phy=dx,
                k_diff=k_diff,
                k_adv=k_adv,
                gamma=gamma,
            )
        return dx.reshape(batch_size, self.num_nodes * self.latent_dim)


class DataDrivenODEFunc(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_nodes: int,
        context_dim: int = 256,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        self.residual_net = nn.Sequential(
            nn.Linear(latent_dim + context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([z, context], dim=-1)
        return self.residual_net(combined)


class AdaptiveGating(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_nodes: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        self.gating_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        z_nodes = z.reshape(batch_size * self.num_nodes, self.latent_dim)
        alpha_nodes = self.gating_net(z_nodes)
        alpha_nodes = alpha_nodes.expand(-1, self.latent_dim)
        return alpha_nodes.reshape(batch_size, self.num_nodes * self.latent_dim)


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
