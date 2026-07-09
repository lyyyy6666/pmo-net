"""
Spatial Modules for SI-xLSTM-Mixer
Includes:
1. StaticContextEmbedding: Embeds static heterogeneous features
2. SpatialMixer: Graph-biased spatial mixing with adaptive adjacency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticContextEmbedding(nn.Module):
    """
    Static Context Embedding Module (SCE)
    
    Embeds heterogeneous static features (POI, LANDUSE, etc.) into high-dimensional
    latent space and broadcasts them for fusion with dynamic features.
    
    Input: static_feat (N, D_static)
    Output: static_embed (B, L, N, D_model) - broadcasted for fusion
    """
    def __init__(
        self,
        num_nodes: int,
        static_feat_dim: int,
        d_model: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_poi_embedding: bool = True,
        use_landuse_embedding: bool = True,
        poi_vocab_size: int = 100,  # Vocabulary size for POI categories
        landuse_vocab_size: int = 20,  # Vocabulary size for LANDUSE categories
    ):
        super(StaticContextEmbedding, self).__init__()
        self.num_nodes = num_nodes
        self.static_feat_dim = static_feat_dim
        self.d_model = d_model
        self.use_poi_embedding = use_poi_embedding
        self.use_landuse_embedding = use_landuse_embedding
        
        # Separate embeddings for different feature types
        # Assume static_feat structure: [POI_features, LANDUSE_features, other_features]
        # This is a flexible design - adjust based on actual data structure
        
        # POI embedding (if POI is categorical/sparse)
        if use_poi_embedding:
            # If POI is sparse vector, use embedding lookup
            # Otherwise, use MLP for dense POI features
            self.poi_embedding = nn.Sequential(
                nn.Linear(static_feat_dim // 3 if static_feat_dim >= 3 else static_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model // 2)
            )
        else:
            self.poi_embedding = None
        
        # LANDUSE embedding
        if use_landuse_embedding:
            self.landuse_embedding = nn.Sequential(
                nn.Linear(static_feat_dim // 3 if static_feat_dim >= 3 else static_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model // 2)
            )
        else:
            self.landuse_embedding = None
        
        # Other static features MLP
        remaining_dim = static_feat_dim
        if use_poi_embedding:
            remaining_dim -= static_feat_dim // 3 if static_feat_dim >= 3 else 0
        if use_landuse_embedding:
            remaining_dim -= static_feat_dim // 3 if static_feat_dim >= 3 else 0
        
        self.other_features_mlp = nn.Sequential(
            nn.Linear(max(remaining_dim, 1), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model - (d_model // 2 if use_poi_embedding else 0) - (d_model // 2 if use_landuse_embedding else 0))
        )
        
        # Final projection to d_model
        self.final_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, static_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            static_feat: (N, D_static) - static features for each node
        
        Returns:
            static_embed: (N, D_model) - embedded static features
        """
        N, D_static = static_feat.shape
        
        # Split features (this is a simplified assumption - adjust based on actual data)
        # In practice, you may need to adjust the splitting logic
        if self.use_poi_embedding and self.use_landuse_embedding:
            split_size = D_static // 3
            poi_feat = static_feat[:, :split_size]
            landuse_feat = static_feat[:, split_size:2*split_size]
            other_feat = static_feat[:, 2*split_size:]
        elif self.use_poi_embedding:
            split_size = D_static // 2
            poi_feat = static_feat[:, :split_size]
            other_feat = static_feat[:, split_size:]
            landuse_feat = None
        elif self.use_landuse_embedding:
            split_size = D_static // 2
            landuse_feat = static_feat[:, :split_size]
            other_feat = static_feat[:, split_size:]
            poi_feat = None
        else:
            poi_feat = None
            landuse_feat = None
            other_feat = static_feat
        
        # Embed each component
        embeddings = []
        
        if self.poi_embedding is not None and poi_feat is not None:
            poi_embed = self.poi_embedding(poi_feat)  # (N, d_model//2)
            embeddings.append(poi_embed)
        
        if self.landuse_embedding is not None and landuse_feat is not None:
            landuse_embed = self.landuse_embedding(landuse_feat)  # (N, d_model//2)
            embeddings.append(landuse_embed)
        
        if other_feat is not None:
            other_embed = self.other_features_mlp(other_feat)  # (N, d_model - ...)
            embeddings.append(other_embed)
        
        # Concatenate all embeddings
        if len(embeddings) > 1:
            static_embed = torch.cat(embeddings, dim=-1)  # (N, D_combined)
            # Project to d_model if needed
            if static_embed.shape[-1] != self.d_model:
                static_embed = F.linear(static_embed, 
                                      torch.zeros(self.d_model, static_embed.shape[-1]).to(static_embed.device))
        else:
            static_embed = embeddings[0]
        
        # Final projection and normalization
        static_embed = self.final_projection(static_embed)  # (N, d_model)
        
        return static_embed
    
    def broadcast(self, static_embed: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """
        Broadcast static embedding to match dynamic feature shape
        
        Args:
            static_embed: (N, D_model)
            batch_size: B
            seq_len: L
        
        Returns:
            broadcasted: (B, L, N, D_model)
        """
        N, D_model = static_embed.shape
        # Expand: (N, D_model) -> (1, 1, N, D_model) -> (B, L, N, D_model)
        broadcasted = static_embed.unsqueeze(0).unsqueeze(0)  # (1, 1, N, D_model)
        broadcasted = broadcasted.expand(batch_size, seq_len, N, D_model)  # (B, L, N, D_model)
        return broadcasted


class SpatialMixer(nn.Module):
    """
    Adaptive Graph-Biased Spatial Mixer
    
    Implements graph-biased mechanism that fuses prior knowledge (physical adjacency)
    with adaptive learned graph structure.
    
    Key components:
    1. Learnable node embeddings E -> A_learned = ReLU(E @ E^T)
    2. Fusion: A_final = Softmax(A_learned + beta * A_physical)
    3. GCN/GAT mechanism: H_out = Activation(A_final @ H_in @ W)
    
    Input: (B, L, N, D) - can handle L dimension independently or fold it
    Output: (B, L, N, D) - same shape
    """
    def __init__(
        self,
        num_nodes: int,
        d_model: int,
        dropout: float = 0.1,
        beta: float = 0.5,  # Weight for physical adjacency (can be learnable)
        learnable_beta: bool = True,
        activation: str = 'relu',
        use_gat: bool = False,  # Use GAT instead of GCN
        num_heads: int = 4,  # For GAT
    ):
        super(SpatialMixer, self).__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.use_gat = use_gat
        self.num_heads = num_heads
        
        # Learnable node embeddings for adaptive adjacency
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, d_model) * 0.01)
        
        # Beta parameter for balancing learned vs physical adjacency
        if learnable_beta:
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer('beta', torch.tensor(beta))
        
        if use_gat:
            # Graph Attention Network
            # Note: This requires torch_geometric. If not available, fall back to GCN
            try:
                from torch_geometric.nn import GATConv
                self.gat_layers = nn.ModuleList([
                    GATConv(d_model, d_model, heads=num_heads, dropout=dropout, concat=False)
                    for _ in range(1)
                ])
            except ImportError:
                self.use_gat = False
                print("Warning: torch_geometric not available, falling back to GCN")
        
        if not use_gat:
            # Graph Convolution Network (simplified)
            self.gcn_weight = nn.Linear(d_model, d_model, bias=False)
            self.layer_norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
    
    def compute_adaptive_adjacency(self, physical_adj: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive adjacency matrix by fusing learned and physical adjacency
        
        Args:
            physical_adj: (N, N) - physical adjacency matrix
        
        Returns:
            A_final: (N, N) - final fused adjacency matrix
        """
        # Compute learned adjacency: A_learned = ReLU(E @ E^T)
        E = self.node_embeddings  # (N, D_model)
        A_learned = torch.matmul(E, E.transpose(0, 1))  # (N, N)
        A_learned = F.relu(A_learned)  # (N, N)
        
        # Normalize learned adjacency (optional)
        A_learned = F.softmax(A_learned, dim=-1)
        
        # Fuse with physical adjacency: A_final = Softmax(A_learned + beta * A_physical)
        # Normalize physical adjacency first
        physical_adj_norm = F.softmax(physical_adj, dim=-1)
        
        # Combine
        A_combined = A_learned + self.beta * physical_adj_norm  # (N, N)
        A_final = F.softmax(A_combined, dim=-1)  # (N, N)
        
        return A_final
    
    def forward(
        self, 
        x: torch.Tensor, 
        physical_adj: torch.Tensor,
        fold_time: bool = True
    ) -> torch.Tensor:
        """
        Spatial mixing operation
        
        Args:
            x: (B, L, N, D) - input features
            physical_adj: (N, N) - physical adjacency matrix
            fold_time: if True, process each time step independently; if False, fold L into batch
        
        Returns:
            out: (B, L, N, D) - spatially mixed features
        """
        B, L, N, D = x.shape
        
        # Compute adaptive adjacency
        A_final = self.compute_adaptive_adjacency(physical_adj)  # (N, N)
        
        if fold_time:
            # Process each time step independently: (B, L, N, D) -> process L times
            # Reshape to (B*L, N, D) for batch processing
            x_reshaped = x.reshape(B * L, N, D)  # (B*L, N, D)
            
            if self.use_gat:
                # GAT requires edge_index format
                # For now, use GCN-style approach with adjacency matrix
                # Convert adjacency to edge_index if needed (simplified here)
                out = self._apply_gcn(x_reshaped, A_final)  # (B*L, N, D)
            else:
                # GCN: H_out = Activation(A_final @ H_in @ W)
                # x_reshaped: (B*L, N, D), A_final: (N, N)
                h = torch.matmul(A_final, x_reshaped)  # (B*L, N, D)
                h = self.gcn_weight(h)  # (B*L, N, D)
                h = self.layer_norm(h)
                h = self.dropout(h)
                out = self.activation(h)  # (B*L, N, D)
            
            # Reshape back: (B*L, N, D) -> (B, L, N, D)
            out = out.reshape(B, L, N, D)
        else:
            # Process all time steps together (more memory intensive)
            # This would require 4D adjacency or different approach
            # For simplicity, we'll use fold_time=True as default
            out = self.forward(x, physical_adj, fold_time=True)
        
        return out
    
    def _apply_gcn(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Apply GCN-style operation
        
        Args:
            x: (B*L, N, D)
            adj: (N, N)
        
        Returns:
            out: (B*L, N, D)
        """
        h = torch.matmul(adj, x)  # (B*L, N, D)
        h = self.gcn_weight(h)  # (B*L, N, D)
        h = self.layer_norm(h)
        h = self.dropout(h)
        out = self.activation(h)
        return out

