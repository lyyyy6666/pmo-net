"""
Multi-Graph Convolution Layer (完全改进版)
核心改进：
1. ✓ 初始化时预处理邻接矩阵
2. ✓ 统一三个图的稀疏优化
3. ✓ 使用register_buffer管理矩阵
4. ✓ forward不需要传入邻接矩阵（更简洁）
5. ✓ 支持对称归一化（标准GCN方式）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GraphConv(nn.Module):
    """
    基础图卷积层
    支持稠密和稀疏邻接矩阵
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_bias: bool = True,
    ):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # 线性变换
        self.weight = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
    
    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        adj_sparse: Optional[torch.sparse.FloatTensor] = None
    ) -> torch.Tensor:
        """
        图卷积操作
        
        Args:
            x: (B*L, N, in_dim) 节点特征
            adj: (N, N) 稠密邻接矩阵（adj_sparse为None时使用）
            adj_sparse: (N, N) 稀疏邻接矩阵（可选，更高效）
        
        Returns:
            out: (B*L, N, out_dim) 输出特征
        """
        if adj_sparse is not None:
            # 使用稀疏矩阵（更高效）
            B_L, N, in_dim = x.shape
            
            # 转置以便矩阵乘法
            x_transposed = x.transpose(0, 1)  # (N, B*L, in_dim)
            x_flat = x_transposed.reshape(N, B_L * in_dim)  # (N, B*L*in_dim)
            
            # 稀疏矩阵乘法（在FP32下进行）
            device_type = 'cuda' if x_flat.is_cuda else 'cpu'
            with torch.autocast(device_type=device_type, enabled=False):
                x_flat_fp32 = x_flat.float()
                h_flat = torch.sparse.mm(adj_sparse, x_flat_fp32)  # (N, B*L*in_dim)
            
            # 恢复原始dtype
            h_flat = h_flat.to(x.dtype)
            
            # Reshape回去
            h_reshaped = h_flat.reshape(N, B_L, in_dim)
            h = h_reshaped.transpose(0, 1)  # (B*L, N, in_dim)
        else:
            # 使用稠密矩阵
            h = torch.einsum('ij,bjk->bik', adj, x)  # (B*L, N, in_dim)
        
        # 线性变换
        h = self.weight(h)  # (B*L, N, out_dim)
        h = self.layer_norm(h)
        h = self.dropout(h)
        h = self.activation(h)
        
        return h


class AttentionFusion(nn.Module):
    """
    基于注意力的融合模块
    动态学习多个图视图的组合权重
    """
    def __init__(
        self,
        d_model: int,
        num_views: int = 3,
        dropout: float = 0.1,
    ):
        super(AttentionFusion, self).__init__()
        self.num_views = num_views
        
        # 查询和键的投影
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        
        # 可学习的视图嵌入
        self.view_embeddings = nn.Parameter(torch.randn(num_views, d_model) * 0.01)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        view_outputs: list,
        node_features: torch.Tensor
    ) -> torch.Tensor:
        """
        融合多个图视图的输出
        
        Args:
            view_outputs: List[(B*L, N, d_model)] 每个视图的输出
            node_features: (B*L, N, d_model) 原始节点特征
        
        Returns:
            fused: (B*L, N, d_model) 融合后的输出
        """
        B_L, N, d_model = view_outputs[0].shape
        
        # 计算注意力权重
        query = self.query_proj(node_features)  # (B*L, N, d_model)
        keys = self.key_proj(self.view_embeddings.unsqueeze(0).unsqueeze(0))  # (1, 1, num_views, d_model)
        keys = keys.expand(B_L, N, -1, -1)  # (B*L, N, num_views, d_model)
        
        # 计算注意力分数
        query_expanded = query.unsqueeze(2)  # (B*L, N, 1, d_model)
        attention_scores = torch.sum(query_expanded * keys, dim=-1) / (d_model ** 0.5)  # (B*L, N, num_views)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (B*L, N, num_views)
        
        # 加权组合
        view_outputs_stack = torch.stack(view_outputs, dim=2)  # (B*L, N, num_views, d_model)
        attention_weights_expanded = attention_weights.unsqueeze(-1)  # (B*L, N, num_views, 1)
        
        fused = torch.sum(view_outputs_stack * attention_weights_expanded, dim=2)  # (B*L, N, d_model)
        
        # 残差连接和归一化
        fused = fused + node_features
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        
        return fused


class ParameterFusion(nn.Module):
    """
    基于可学习参数的融合模块（简单版）
    """
    def __init__(
        self,
        num_views: int = 3,
        learnable: bool = True,
    ):
        super(ParameterFusion, self).__init__()
        self.num_views = num_views
        
        if learnable:
            self.view_weights = nn.Parameter(torch.ones(num_views) / num_views)
        else:
            self.register_buffer('view_weights', torch.ones(num_views) / num_views)
    
    def forward(self, view_outputs: list, node_features: torch.Tensor = None) -> torch.Tensor:
        """
        使用可学习权重融合多个视图
        
        Args:
            view_outputs: List[(B*L, N, d_model)] 每个视图的输出
            node_features: (B*L, N, d_model) 原始节点特征（可选，用于残差）
        
        Returns:
            fused: (B*L, N, d_model) 融合后的输出
        """
        # 归一化权重
        weights = F.softmax(self.view_weights, dim=0)
        
        # 加权组合
        fused = sum(w * out for w, out in zip(weights, view_outputs))
        
        return fused


class MultiGraphConv(nn.Module):
    """
    多图卷积层（完全改进版）
    
    关键改进：
    1. 初始化时预处理邻接矩阵
    2. 统一三个图的稀疏优化
    3. forward不需要传入邻接矩阵
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_nodes: int,
        adj_geo: torch.Tensor,
        adj_poi: torch.Tensor,
        adj_land: torch.Tensor,
        dropout: float = 0.1,
        activation: str = 'gelu',
        fusion_type: str = 'attention',
        use_sparse: bool = True,
        normalize_adj: bool = True,
        normalize_type: str = 'row',  # 'row' or 'symmetric'
    ):
        super(MultiGraphConv, self).__init__()
        self.num_nodes = num_nodes
        self.use_sparse = use_sparse
        self.normalize_type = normalize_type
        
        # 三个独立的图卷积模块
        self.graph_conv_geo = GraphConv(in_dim, out_dim, dropout, activation)
        self.graph_conv_poi = GraphConv(in_dim, out_dim, dropout, activation)
        self.graph_conv_land = GraphConv(in_dim, out_dim, dropout, activation)
        
        # 融合模块
        if fusion_type == 'attention':
            self.fusion = AttentionFusion(out_dim, num_views=3, dropout=dropout)
        elif fusion_type == 'parameter':
            self.fusion = ParameterFusion(num_views=3, learnable=True)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        self.fusion_type = fusion_type
        
        # 预处理邻接矩阵
        if normalize_adj:
            if normalize_type == 'symmetric':
                adj_geo_norm = self.normalize_adjacency_symmetric(adj_geo)
                adj_poi_norm = self.normalize_adjacency_symmetric(adj_poi)
                adj_land_norm = self.normalize_adjacency_symmetric(adj_land)
            else:
                adj_geo_norm = self.normalize_adjacency_row(adj_geo)
                adj_poi_norm = self.normalize_adjacency_row(adj_poi)
                adj_land_norm = self.normalize_adjacency_row(adj_land)
        else:
            adj_geo_norm = adj_geo.float()
            adj_poi_norm = adj_poi.float()
            adj_land_norm = adj_land.float()
        
        # 注册为buffer（会自动跟随模型移动到GPU/CPU）
        if use_sparse:
            # 转为稀疏矩阵并注册
            self.register_buffer('adj_geo_sparse', self.dense_to_sparse(adj_geo_norm))
            self.register_buffer('adj_poi_sparse', self.dense_to_sparse(adj_poi_norm))
            self.register_buffer('adj_land_sparse', self.dense_to_sparse(adj_land_norm))
            
            # 不需要保存稠密版本
            self.adj_geo_dense = None
            self.adj_poi_dense = None
            self.adj_land_dense = None
        else:
            # 只保存稠密版本
            self.register_buffer('adj_geo_dense', adj_geo_norm)
            self.register_buffer('adj_poi_dense', adj_poi_norm)
            self.register_buffer('adj_land_dense', adj_land_norm)
            
            self.adj_geo_sparse = None
            self.adj_poi_sparse = None
            self.adj_land_sparse = None
    
    def normalize_adjacency_row(
        self,
        adj: torch.Tensor,
        add_self_loops: bool = True
    ) -> torch.Tensor:
        """
        行归一化邻接矩阵：D^(-1) A
        适合有向图
        """
        if add_self_loops:
            adj = adj + torch.eye(adj.shape[0], device=adj.device)
        
        # 行归一化
        rowsum = adj.sum(dim=1, keepdim=True)
        rowsum = torch.clamp(rowsum, min=1e-12)
        adj_norm = adj / rowsum
        
        return adj_norm.float()
    
    def normalize_adjacency_symmetric(
        self,
        adj: torch.Tensor,
        add_self_loops: bool = True
    ) -> torch.Tensor:
        """
        对称归一化邻接矩阵：D^(-1/2) A D^(-1/2)
        适合无向图，标准GCN方法
        """
        if add_self_loops:
            adj = adj + torch.eye(adj.shape[0], device=adj.device)
        
        # 计算度矩阵
        degree = adj.sum(dim=1)
        
        # D^(-1/2)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.
        D_inv_sqrt = torch.diag(degree_inv_sqrt)
        
        # D^(-1/2) A D^(-1/2)
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
        
        return adj_norm.float()
    
    def dense_to_sparse(self, adj: torch.Tensor) -> torch.sparse.FloatTensor:
        """
        将稠密邻接矩阵转换为稀疏张量
        
        Args:
            adj: (N, N) 稠密邻接矩阵
        
        Returns:
            adj_sparse: (N, N) 稀疏张量（FP32）
        """
        # 转为FP32（稀疏操作要求）
        adj_fp32 = adj.float()
        device = adj_fp32.device
        
        # 找到非零元素
        indices = torch.nonzero(adj_fp32, as_tuple=False).t()
        values = adj_fp32[indices[0], indices[1]]
        
        # 创建稀疏张量
        adj_sparse = torch.sparse_coo_tensor(
            indices,
            values,
            size=torch.Size([adj_fp32.shape[0], adj_fp32.shape[1]]),
            dtype=torch.float32,
            device=device
        )
        
        return adj_sparse
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        多图卷积前向传播（改进版：不需要传入邻接矩阵）
        
        Args:
            x: (B*L, N, in_dim) 输入节点特征
        
        Returns:
            fused: (B*L, N, out_dim) 融合后的输出
        """
        # 对每个视图应用图卷积（使用预处理的邻接矩阵）
        if self.use_sparse:
            out_geo = self.graph_conv_geo(x, None, self.adj_geo_sparse)
            out_poi = self.graph_conv_poi(x, None, self.adj_poi_sparse)
            out_land = self.graph_conv_land(x, None, self.adj_land_sparse)
        else:
            out_geo = self.graph_conv_geo(x, self.adj_geo_dense, None)
            out_poi = self.graph_conv_poi(x, self.adj_poi_dense, None)
            out_land = self.graph_conv_land(x, self.adj_land_dense, None)
        
        # 融合输出
        view_outputs = [out_geo, out_poi, out_land]
        
        if self.fusion_type == 'attention':
            fused = self.fusion(view_outputs, x)
        else:
            fused = self.fusion(view_outputs, x)
        
        return fused
