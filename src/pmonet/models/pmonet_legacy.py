from __future__ import annotations

"""
xLSTM-DualODE-Mixer: 100%严格对齐论文版本
核心改进（完全按照论文公式）：
1. ✓ ODE求解前融合（论文公式13）
2. ✓ 从xLSTM的h_T提取z_0（论文公式6）
3. ✓ 从xLSTM的h_T提取c_ctx（论文公式7，降级方案）
4. ✓ 自适应门控α_t（论文公式11-12）- 每个时间步根据Z(t)动态计算
5. ✓ 单次ODE求解（性能优化）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np

from xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from xlstm import sLSTMBlockConfig, sLSTMLayerConfig

from ..modules.revin import Normalize as RevIN
from ..modules.spatial_modules import StaticContextEmbedding
from ..modules.multigraph_conv import MultiGraphConv
from .base_model import BaseModel


class PhysicalODEFunc(nn.Module):
    """
    物理知识驱动的ODE函数（论文公式9）
    f_phy(Z(t)) = -k_diff * L * Z(t) + MLP_phy(Z(t)) - γ * Z(t)
    
    基于扩散-反应-衰减方程（论文公式8）
    """
    def __init__(
        self,
        latent_dim: int,
        num_nodes: int,
        adj_mx: torch.Tensor,
        hidden_dim: int = 64
    ):
        super(PhysicalODEFunc, self).__init__()
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        
        # 预计算归一化拉普拉斯矩阵 L = I - D^(-1/2) A D^(-1/2)
        L = self._compute_normalized_laplacian(adj_mx)
        self.register_buffer('laplacian', L)
        
        # 可学习的物理参数
        self.diffusion_coeff = nn.Parameter(torch.tensor(0.1))  # k_diff
        self.decay_coeff = nn.Parameter(torch.tensor(0.01))     # γ
        
        # MLP_phy：建模非线性化学反应项 R(·)
        self.reaction_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def _compute_normalized_laplacian(self, adj: torch.Tensor) -> torch.Tensor:
        """
        计算归一化拉普拉斯矩阵: L = I - D^(-1/2) A D^(-1/2)
        对应论文Step 2的图离散化
        """
        device = adj.device
        
        # 计算度矩阵
        degree = adj.sum(dim=1)
        
        # D^(-1/2)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.
        D_inv_sqrt = torch.diag(degree_inv_sqrt)
        
        # L = I - D^(-1/2) A D^(-1/2)
        I = torch.eye(adj.shape[0], device=device)
        L = I - D_inv_sqrt @ adj @ D_inv_sqrt
        
        return L
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        物理ODE（论文公式9）:
        f_phy(Z(t)) = -k_diff * L * Z(t) + MLP_phy(Z(t)) - γ * Z(t)
        
        Args:
            x: (B, N*latent_dim) 当前状态
        Returns:
            dx: (B, N*latent_dim) 状态变化率
        """
        B = x.shape[0]
        
        # Reshape: (B, N*latent_dim) -> (B, N, latent_dim)
        x_reshaped = x.reshape(B, self.num_nodes, self.latent_dim)
        
        # 扩散项: -k_diff * L * Z(t)
        # 注意：负号表示从高浓度向低浓度扩散
        Lx = torch.einsum('ij,bjd->bid', self.laplacian, x_reshaped)
        diffusion = -self.diffusion_coeff * Lx
        
        # 化学反应项: MLP_phy(Z(t))
        reaction = self.reaction_net(x_reshaped)
        
        # 衰减项: -γ * Z(t)
        decay = -self.decay_coeff * x_reshaped
        
        # 总变化率（论文公式9）
        dx = diffusion + reaction + decay
        
        # Reshape back
        return dx.reshape(B, self.num_nodes * self.latent_dim)


class DataDrivenODEFunc(nn.Module):
    """
    数据驱动的ODE函数（论文公式10）
    f_data(Z(t), c_ctx) = MLP_res(Concat(Z(t), c_ctx))
    
    用于建模物理PDE无法捕捉的高阶残差动力学（如交通排放等突发事件）
    """
    def __init__(
        self,
        latent_dim: int,
        num_nodes: int,
        context_dim: int = 256,
        hidden_dim: int = 64
    ):
        super(DataDrivenODEFunc, self).__init__()
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        
        # MLP_res：残差网络（论文公式10）
        self.residual_net = nn.Sequential(
            nn.Linear(latent_dim + context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        数据驱动ODE（论文公式10）:
        f_data(Z(t), c_ctx) = MLP_res(Concat(Z(t), c_ctx))
        
        Args:
            z: (B*N, latent_dim) 当前状态
            context: (B*N, context_dim) xLSTM提取的全局上下文
        Returns:
            dz: (B*N, latent_dim) 状态变化率
        """
        # Concat(Z(t), c_ctx)
        combined = torch.cat([z, context], dim=-1)
        
        # MLP_res
        dz = self.residual_net(combined)
        
        return dz


class AdaptiveGating(nn.Module):
    """
    自适应门控机制（论文公式11）
    α_t = σ(W_g · Z(t) + b_g)
    
    关键特性：
    - α_t 在每个时间步根据当前状态 Z(t) 动态计算
    - 在稳定区域退化为纯物理模型 (α_t → 1)
    - 在湍流区域切换为数据驱动学习 (α_t → 0)
    """
    def __init__(
        self,
        latent_dim: int,
        num_nodes: int,
        hidden_dim: int = 64
    ):
        super(AdaptiveGating, self).__init__()
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        
        # W_g 和 b_g（论文公式11）
        self.gating_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # σ激活函数
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        计算自适应门控系数（论文公式11）
        α_t = σ(W_g · Z(t) + b_g)
        
        Args:
            z: (B, N*latent_dim) 当前状态
        Returns:
            alpha: (B, N*latent_dim) 门控系数
        """
        B = z.shape[0]
        
        # Reshape: (B, N*latent_dim) -> (B*N, latent_dim)
        z_nodes = z.reshape(B * self.num_nodes, self.latent_dim)
        
        # 计算门控系数（论文公式11）
        alpha_nodes = self.gating_net(z_nodes)  # (B*N, 1)
        
        # 扩展到整个latent_dim并reshape回去
        alpha_nodes = alpha_nodes.expand(-1, self.latent_dim)  # (B*N, latent_dim)
        alpha = alpha_nodes.reshape(B, self.num_nodes * self.latent_dim)
        
        return alpha


class xLSTM_DualODE_Mixer(BaseModel):
    """
    xLSTM-DualODE-Mixer（100%严格对齐论文版本）
    
    论文方法完整实现：
    - 公式5: h_T, c_T = xLSTM(x_t, h_{t-1}, c_{t-1})
    - 公式6: z_0 = Linear(h_T)
    - 公式7: c_ctx = Linear_ctx(c_T)  # 降级为h_T
    - 公式9: f_phy(Z) = -k_diff * L * Z + MLP_phy(Z) - γ * Z
    - 公式10: f_data(Z, c_ctx) = MLP_res(Concat(Z, c_ctx))
    - 公式11: α_t = σ(W_g · Z(t) + b_g)
    - 公式12: dZ/dt = α_t ⊙ f_phy(Z) + (1-α_t) ⊙ f_data(Z)
    - 公式13: Z(t+τ) = Z(t) + τ·(α_t ⊙ f_phy + (1-α_t) ⊙ f_data)
    """
    
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
        fusion_type: str = 'attention',
        use_sparse: bool = True,
        use_phy_ode: bool = True,
        use_unk_ode: bool = True,
        use_adaptive_gating: bool = True,    # ← 新增
        gating_hidden_dim: int = 64,         # ← 新增
        latent_dim: int = 128,
        ode_hidden_dim: int = 64,
        adj_mx: Optional[torch.Tensor] = None,
        adj_geo: Optional[torch.Tensor] = None,
        adj_poi: Optional[torch.Tensor] = None,
        adj_land: Optional[torch.Tensor] = None,
    ):
        super().__init__(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in)
        
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.enc_in = enc_in
        self.dec_out = dec_out if dec_out is not None else enc_in
        self.latent_dim = latent_dim
        self.use_phy_ode = use_phy_ode
        self.use_unk_ode = use_unk_ode
        
        # RevIN归一化
        self.reversible_instance_norm = RevIN(enc_in, affine=True)
        
        # 静态上下文嵌入
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
        
        # 动态特征嵌入
        self.dynamic_embedding = nn.Linear(enc_in, d_model)
        
        # 历史上下文提取器（xLSTM）- 论文公式5
        slstm_config = sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                num_heads=xlstm_num_heads,
                conv1d_kernel_size=0
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
        
        # 论文公式6：z_0 = Linear(h_T)
        self.z0_projection = nn.Sequential(
            nn.Linear(d_model, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # 论文公式7：c_ctx = Linear_ctx(c_T)
        # 注意：由于xLSTM API限制，使用h_T作为降级方案
        self.context_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        
        # 双ODE分支
        if use_phy_ode:
            if adj_mx is None:
                adj_mx = torch.ones(num_nodes, num_nodes) / num_nodes
            # 物理ODE（论文公式9）
            self.phy_ode_func = PhysicalODEFunc(
                latent_dim=latent_dim,
                num_nodes=num_nodes,
                adj_mx=adj_mx,
                hidden_dim=ode_hidden_dim
            )
        
        if use_unk_ode:
            # 数据驱动ODE（论文公式10）
            self.unk_ode_func = DataDrivenODEFunc(
                latent_dim=latent_dim,
                num_nodes=num_nodes,
                context_dim=d_model,
                hidden_dim=ode_hidden_dim
            )
        
        # 自适应门控机制（论文公式11）
        self.use_adaptive_gating = use_adaptive_gating  # 保存标志
        if use_phy_ode and use_unk_ode and use_adaptive_gating:
            self.adaptive_gating = AdaptiveGating(
                latent_dim=latent_dim,
                num_nodes=num_nodes,
                hidden_dim=gating_hidden_dim  # 使用参数
            )
        else:
            self.adaptive_gating = None


        # if use_phy_ode and use_unk_ode:
        #     self.adaptive_gating = AdaptiveGating(
        #         latent_dim=latent_dim,
        #         num_nodes=num_nodes,
        #         hidden_dim=ode_hidden_dim
        #     )
        
        # 解码器：从潜在空间解码回特征空间
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Spatial-Mixing（预处理邻接矩阵）
        if use_spatial_mixing and adj_geo is not None:
            # 确保邻接矩阵是2D的
            if len(adj_geo.shape) == 3:
                adj_geo = adj_geo[0]
            if adj_poi is not None and len(adj_poi.shape) == 3:
                adj_poi = adj_poi[0]
            if adj_land is not None and len(adj_land.shape) == 3:
                adj_land = adj_land[0]
            
            # 处理缺失的邻接矩阵
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
                activation='gelu',
                fusion_type=fusion_type,
                use_sparse=use_sparse,
                normalize_adj=True,
            )
        else:
            self.spatial_mixer = None
        
        # Channel-Mixing
        self.channel_mixer = nn.Sequential(
            nn.Linear(d_model, channel_mixer_hidden),
            nn.GELU(),
            nn.Dropout(xlstm_dropout),
            nn.Linear(channel_mixer_hidden, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(xlstm_dropout)
        )
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(xlstm_dropout),
            nn.Linear(d_model // 2, self.dec_out)
        )
        
        # 时间嵌入
        self.time_embedding = nn.Parameter(torch.randn(pred_len, d_model) * 0.02)
    
    def solve_fused_ode(
        self,
        z0: torch.Tensor,
        time_steps: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        融合求解ODE（论文公式11-13的严格实现）
        
        关键改进：
        - 公式11: α_t = σ(W_g · Z(t) + b_g)  # 每个时间步动态计算
        - 公式12: dZ/dt = α_t ⊙ f_phy(Z(t)) + (1-α_t) ⊙ f_data(Z(t))
        - 公式13: Z(t+τ) = Z(t) + τ·(α_t ⊙ f_phy + (1-α_t) ⊙ f_data)
        
        Args:
            z0: (B, N*latent_dim) 初始状态
            time_steps: (T,) 时间步
            context: (B*N, context_dim) xLSTM提取的上下文特征
        Returns:
            z_trajectory: (T, B, N*latent_dim) 状态轨迹
        """
        B = z0.shape[0]
        T = len(time_steps)
        dt = time_steps[1] - time_steps[0] if T > 1 else 1.0
        
        z_trajectory = [z0]
        z = z0
        
        for t in range(1, T):
            # ===== 关键修正：每个时间步根据Z(t)计算α_t（论文公式11）=====
            if self.use_phy_ode and self.use_unk_ode:
                if self.use_adaptive_gating and self.adaptive_gating is not None:
                    alpha_t = self.adaptive_gating(z)  # 动态计算
                else:
                    alpha_t = torch.ones_like(z) * 0.5  # 固定50%混合（消融对照）
            elif self.use_phy_ode:
                # 只启用物理ODE：alpha_t = 1（全部权重给物理项）
                alpha_t = torch.ones_like(z)
            else:
                # 只启用数据ODE：alpha_t = 0（全部权重给数据项）
                alpha_t = torch.zeros_like(z)

            # if self.use_phy_ode and self.use_unk_ode:
            #     alpha_t = self.adaptive_gating(z)  # (B, N*latent_dim)
            # else:
            #     alpha_t = torch.ones_like(z)
            
            # 计算物理ODE的导数（论文公式9）
            if self.use_phy_ode:
                dz_phy = self.phy_ode_func(z)  # (B, N*latent_dim)
            else:
                dz_phy = torch.zeros_like(z)
            
            # 计算数据驱动ODE的导数（论文公式10）
            if self.use_unk_ode:
                z_nodes = z.reshape(B * self.num_nodes, self.latent_dim)
                dz_data_nodes = self.unk_ode_func(z_nodes, context)
                dz_data = dz_data_nodes.reshape(B, self.num_nodes * self.latent_dim)
            else:
                dz_data = torch.zeros_like(z)
            
            # ===== 自适应融合（论文公式12）=====
            # dZ/dt = α_t ⊙ f_phy(Z(t)) + (1-α_t) ⊙ f_data(Z(t))
            dz_fused = alpha_t * dz_phy + (1 - alpha_t) * dz_data
        
            
            # ===== 欧拉法更新（论文公式13）=====
            # Z(t+τ) = Z(t) + τ·(α_t ⊙ f_phy + (1-α_t) ⊙ f_data)
            z = z + dt * dz_fused
            
            z_trajectory.append(z)
        
        return torch.stack(z_trajectory, dim=0)
    
    def forecast(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor] = None,
        static_feat: Optional[torch.Tensor] = None,
        target_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        前向传播（100%对齐论文版本）
        
        Args:
            x_enc: (B, L, N, C) 输入序列
        Returns:
            output: (B, H, N, C) 预测结果
        """
        B, L, N, C = x_enc.shape
        
        # Step 1: RevIN归一化
        x_reshaped = x_enc.permute(0, 2, 1, 3).contiguous()
        x_reshaped = x_reshaped.reshape(B * N, L, C)
        x_norm = self.reversible_instance_norm(x_reshaped, "norm")
        x_norm = x_norm.reshape(B, N, L, C).permute(0, 2, 1, 3).contiguous()
        
        # Step 2: 嵌入
        x_embed = self.dynamic_embedding(x_norm)  # (B, L, N, d_model)
        
        # 静态上下文融合
        if self.static_context_embedding is not None and static_feat is not None:
            static_embed = self.static_context_embedding(static_feat)
            static_broadcast = self.static_context_embedding.broadcast(static_embed, B, L)
            x_embed = x_embed + static_broadcast
        
        # Step 3: 提取历史上下文（使用xLSTM）- 论文公式5
        x_seq_flat = x_embed.permute(0, 2, 1, 3).contiguous().reshape(B * N, L, self.d_model)
        xlstm_out = self.context_extractor(x_seq_flat)  # (B*N, L, d_model)
        
        # 提取最终隐藏状态 h_T（论文公式5）
        h_T = xlstm_out[:, -1, :]  # (B*N, d_model)
        
        # Step 4: 从h_T生成初始状态z_0（论文公式6）
        z0_nodes = self.z0_projection(h_T)  # (B*N, latent_dim)
        z0 = z0_nodes.reshape(B, N * self.latent_dim)
        
        # Step 5: 从h_T提取上下文c_ctx（论文公式7的降级方案）
        # 理想情况应该从c_T提取，但xLSTM API限制，使用h_T
        c_ctx = self.context_projection(h_T)  # (B*N, d_model)
        
        # Step 6: 融合求解ODE（论文公式11-13）
        time_steps = torch.linspace(0, 1, self.pred_len + 1).to(x_enc.device)
        fused_z = self.solve_fused_ode(z0, time_steps, c_ctx)[1:]  # (pred_len, B, N*latent_dim)
        
        # Step 7: 解码回特征空间
        fused_z_reshaped = fused_z.reshape(self.pred_len, B, N, self.latent_dim)
        fused_z_reshaped = fused_z_reshaped.permute(1, 0, 2, 3).contiguous()
        fused_z_flat = fused_z_reshaped.reshape(B * N, self.pred_len, self.latent_dim)
        
        decoded = self.decoder(fused_z_flat)  # (B*N, pred_len, d_model)
        decoded = decoded.reshape(B, N, self.pred_len, self.d_model)
        decoded = decoded.permute(0, 2, 1, 3).contiguous()  # (B, pred_len, N, d_model)
        
        # Step 8: Channel-Mixing
        x_channel = self.channel_mixer(decoded)
        x_channel = x_channel + decoded
        
        # Step 9: Spatial-Mixing
        if self.spatial_mixer is not None:
            x_spatial = x_channel.permute(0, 2, 1, 3).contiguous()
            x_spatial = x_spatial.reshape(B * self.pred_len, N, self.d_model)
            x_spatial = self.spatial_mixer(x_spatial)  # 使用预处理的邻接矩阵
            x_spatial = x_spatial.reshape(B, self.pred_len, N, self.d_model)
            x_spatial = x_spatial + x_channel
        else:
            x_spatial = x_channel
        
        # Step 10: 添加时间嵌入
        time_emb = self.time_embedding.unsqueeze(0).unsqueeze(2)
        time_emb = time_emb.expand(B, self.pred_len, N, self.d_model)
        x_with_time = x_spatial + time_emb
        
        # Step 11: 输出投影
        output = self.output_projection(x_with_time)
        
        # Step 12: RevIN反归一化
        output_reshaped = output.permute(0, 2, 1, 3).contiguous()
        output_reshaped = output_reshaped.reshape(B * N, self.pred_len, self.dec_out)
        
        # if self.dec_out == 1 and target_idx is not None:
        #     output_denorm = self.reversible_instance_norm(output_reshaped, "denorm", target_idx=target_idx)
        # else:
        output_denorm = self.reversible_instance_norm(output_reshaped, "denorm")

        if self.dec_out == 1 and output_denorm.shape[-1] > 1:
            idx = target_idx if target_idx is not None else 0
            # 只取对应的那一列，恢复成 [..., 1]
            output_denorm = output_denorm[..., idx:idx+1]
        
        output_final = output_denorm.reshape(B, N, self.pred_len, self.dec_out)
        output_final = output_final.permute(0, 2, 1, 3).contiguous()
        
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
    ) -> torch.Tensor:
        """主前向函数"""
        return self.forecast(x_enc, x_mark_enc, static_feat)
