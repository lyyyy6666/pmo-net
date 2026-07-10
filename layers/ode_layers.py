from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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


