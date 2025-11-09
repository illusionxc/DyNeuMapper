# FILE: models/dyneumapper_model.py
# -*- coding: utf-8 -*-
"""
Defines the DyneuMapper model architecture, streamlined for inference.
"""
import math
from typing import Optional, Dict, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear

from .hetero_subgraph_model import HeteroGNNEncoder_Subgraph


class PositionalEncoding(nn.Module):
    """Adds positional encoding to the input tensor."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, steps: Union[int, torch.Tensor]) -> torch.Tensor:
        """Applies positional encoding."""
        pos_encoding = self.pe[steps]
        while pos_encoding.dim() < x.dim():
            pos_encoding = pos_encoding.unsqueeze(1)
        x = x + pos_encoding
        return self.dropout(x)


class BatchDistanceAwareCrossAttention(nn.Module):
    """
    A cross-attention mechanism that is aware of Manhattan distances between
    the query and context elements.
    """
    def __init__(self, embed_dim: int, heads: int, distance_lambda: float = -0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = heads
        self.head_dim = embed_dim // heads
        assert self.head_dim * heads == self.embed_dim, "embed_dim must be divisible by heads"

        self.distance_lambda = distance_lambda
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self,
                query: torch.Tensor,
                context: Optional[torch.Tensor],
                dist_matrix: Optional[torch.Tensor] = None,
                context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs the forward pass."""
        if context is None or context.numel() == 0:
            return query

        B, N_ctx, D = context.shape
        q = self.q_proj(query)
        k = self.k_proj(context)
        v = self.v_proj(context)

        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N_ctx, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N_ctx, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.distance_lambda != 0 and dist_matrix is not None:
            dist_bias = dist_matrix.unsqueeze(1) * self.distance_lambda
            scores = scores + dist_bias

        if context_mask is not None:
            scores = scores.masked_fill(context_mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, 1, D)
        output = self.out_proj(attn_output.squeeze(1))

        return self.norm(query + output)


class DyneuMapperNet_DQN(nn.Module):
    """
    DyneuMapperNet model for DQN-based agents. It integrates a GNN, positional encoding,
    and distance-aware cross-attention to produce Q-values for actions.
    This version is optimized for inference.
    """
    def __init__(self,
                 snn_node_dim_for_gnn: int,
                 noc_node_dim_for_gnn: int,
                 snn_edge_dim_for_gnn: int,
                 gnn_hidden_dim: int,
                 gnn_out_dim: int,
                 gnn_heads: int,
                 snn_static_feature_dim: int,
                 noc_embed_dim: int,
                 context_embed_dim: int,
                 head_hidden_dim: int,
                 config: Dict):
        super().__init__()

        self.use_temporal_encoding = config.get('dyneumapper_use_temporal_encoding', True)
        self.use_distance_awareness = config.get('dyneumapper_use_distance_awareness', True)

        # Component Modules
        self.gnn = HeteroGNNEncoder_Subgraph(
            snn_node_dim_for_gnn, noc_node_dim_for_gnn, snn_edge_dim_for_gnn,
            gnn_hidden_dim, gnn_out_dim, gnn_heads
        )
        self.static_feature_projector = Linear(snn_static_feature_dim, gnn_out_dim)
        self.context_fusion_mlp = nn.Sequential(
            Linear(gnn_out_dim + noc_embed_dim, head_hidden_dim), nn.ReLU(),
            Linear(head_hidden_dim, context_embed_dim)
        )
        self.cross_attention = BatchDistanceAwareCrossAttention(
            embed_dim=gnn_out_dim,
            heads=gnn_heads,
            distance_lambda=config.get('dyneumapper_distance_lambda', -0.1)
        )

        if self.use_temporal_encoding:
            self.temporal_encoder_query = PositionalEncoding(d_model=gnn_out_dim)
            self.temporal_encoder_context = PositionalEncoding(d_model=context_embed_dim)

        self.q_head = nn.Sequential(
            Linear(gnn_out_dim + noc_embed_dim, head_hidden_dim), nn.ReLU(),
            Linear(head_hidden_dim, 1)
        )

    def forward(
        self,
        query_static_feature: torch.Tensor,
        candidate_noc_embeds: torch.Tensor,
        attention_context: Optional[torch.Tensor],
        placement_step: int,
        distance_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Performs a forward pass for a single state to compute Q-values, used during inference.
        """
        query = self.static_feature_projector(query_static_feature)

        if self.use_temporal_encoding:
            query = self.temporal_encoder_query(query, placement_step)

        query = query.unsqueeze(0)
        if attention_context is not None:
            attention_context = attention_context.unsqueeze(0)

        dist_mat_to_use = None
        if self.use_distance_awareness and distance_matrix is not None:
            dist_mat_to_use = distance_matrix[0:1, :].unsqueeze(0)

        attention_output = self.cross_attention(query, attention_context, dist_mat_to_use)
        
        globally_aware_embed = attention_output[0] if isinstance(attention_output, tuple) else attention_output

        num_candidates = candidate_noc_embeds.size(0)
        if num_candidates == 0:
            return torch.empty(0, device=globally_aware_embed.device)

        globally_aware_embed_expanded = globally_aware_embed.expand(num_candidates, -1)
        q_head_input = torch.cat([globally_aware_embed_expanded, candidate_noc_embeds], dim=-1)
        q_values = self.q_head(q_head_input).squeeze(-1)

        return q_values