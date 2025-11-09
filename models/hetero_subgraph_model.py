# FILE: models/gcpn_hetero_subgraph_model.py
# -*- coding: utf-8 -*-

from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData, Batch
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear, global_mean_pool

# === Module 1: Subgraph GNN Encoder ===

class HeteroGNNEncoder_Subgraph(nn.Module):
    """
    A GNN encoder for heterogeneous subgraphs.
    It learns node representations by performing multi-layer graph convolutions on local subgraphs.
    """
    
    def __init__(self, snn_node_dim: int, noc_node_dim: int, snn_edge_dim: int, hidden_dim: int, out_dim: int, heads: int = 4):
        super().__init__()
        
        self.convs = nn.ModuleList()
        
        self.convs.append(HeteroConv({
            ('snn', 'connects', 'snn'): GATv2Conv(snn_node_dim, hidden_dim // heads, heads=heads, edge_dim=snn_edge_dim, add_self_loops=False),
            ('noc', 'links', 'noc'): GATv2Conv(noc_node_dim, hidden_dim // heads, heads=heads, add_self_loops=False),
            ('snn', 'mapped_to', 'noc'): GATv2Conv((snn_node_dim, noc_node_dim), hidden_dim // heads, heads=heads, add_self_loops=False),
            ('noc', 'rev_mapped_to', 'snn'): GATv2Conv((noc_node_dim, snn_node_dim), hidden_dim // heads, heads=heads, add_self_loops=False),
        }, aggr='sum'))


        self.convs.append(HeteroConv({
            ('snn', 'connects', 'snn'): GATv2Conv(hidden_dim, out_dim, heads=1, concat=False, edge_dim=snn_edge_dim, add_self_loops=False),
            ('noc', 'links', 'noc'): GATv2Conv(hidden_dim, out_dim, heads=1, concat=False, add_self_loops=False),
            ('snn', 'mapped_to', 'noc'): GATv2Conv(hidden_dim, out_dim, heads=1, concat=False, add_self_loops=False),
            ('noc', 'rev_mapped_to', 'snn'): GATv2Conv(hidden_dim, out_dim, heads=1, concat=False, add_self_loops=False),
        }, aggr='sum'))

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        
        edge_attr_dict = data.edge_attr_dict if 'edge_attr_dict' in data and data.edge_attr_dict else None
        
        for i, conv in enumerate(self.convs):
            if edge_attr_dict:
                x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            else:
                x_dict = conv(x_dict, edge_index_dict)
                
            if i < len(self.convs) - 1:
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}
                
        return x_dict


# === Module 2: Cross Attention ===

class CrossAttentionModule(nn.Module):
    """
    A cross-attention module.
    Allows a query node from the subgraph to attend to a global context (key/value) to gain global awareness.
    """
    def __init__(self, embed_dim: int, heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        if context is None or context.numel() == 0:
            return query

        if query.ndim == 1: query = query.unsqueeze(0) # [D] -> [1, D]
        
        q = query.unsqueeze(1)  # [B, D] -> [B, 1, D]
        
        if context.ndim == 2: context = context.unsqueeze(0).expand(q.size(0), -1, -1)
            
        k, v = context, context

        attn_output, _ = self.mha(q, k, v) # attn_output: [B, 1, D]

        return self.norm(query + attn_output.squeeze(1))


# === Module 3: PPO Actor-Critic Policy ===

class SubgraphHeteroActorCriticPolicy(nn.Module):
    """
    PPO Actor-Critic policy for heterogeneous subgraphs.
    Combines the GNN encoder, cross-attention, an actor head, and a critic head.
    """
    def __init__(self, snn_node_dim: int, noc_node_dim: int, snn_edge_dim: int, gnn_hidden_dim: int, gnn_out_dim: int, gnn_heads: int, ac_hidden_dim: int):
        super().__init__()
        self.gnn = HeteroGNNEncoder_Subgraph(snn_node_dim, noc_node_dim, snn_edge_dim, gnn_hidden_dim, gnn_out_dim, gnn_heads)
        self.cross_attention = CrossAttentionModule(gnn_out_dim, gnn_heads)
        self.actor_head = nn.Sequential(
            Linear(gnn_out_dim * 2, ac_hidden_dim), nn.ReLU(),
            Linear(ac_hidden_dim, 1)
        )
        self.critic_head = nn.Sequential(
            Linear(gnn_out_dim, ac_hidden_dim), nn.ReLU(),
            Linear(ac_hidden_dim, 1)
        )

    def forward(self, data: HeteroData, center_snn_idx: int, candidate_noc_indices: torch.Tensor,
                attention_context: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        
        node_embeds = self.gnn(data)
        snn_embeds, noc_embeds = node_embeds['snn'], node_embeds['noc']
        snn_node_to_place_embed = snn_embeds[center_snn_idx]
        globally_aware_embed = self.cross_attention(snn_node_to_place_embed, attention_context)
        
        value = self.critic_head(globally_aware_embed).squeeze()
        
        candidate_noc_embeds = noc_embeds[candidate_noc_indices]
        num_candidates = candidate_noc_embeds.size(0)
        globally_aware_embed_expanded = globally_aware_embed.expand(num_candidates, -1)
        actor_input = torch.cat([globally_aware_embed_expanded, candidate_noc_embeds], dim=-1)
        logits = self.actor_head(actor_input).squeeze(-1)
        
        return logits, value


    def evaluate_from_embeds(self, node_embeds: Dict[str, torch.Tensor], center_snn_idx: int, 
                             candidate_noc_indices: torch.Tensor, attention_context: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        A helper method to evaluate actions directly from pre-computed GNN embeddings and the latest attention context.
        """
        snn_embeds, noc_embeds = node_embeds['snn'], node_embeds['noc']
        snn_node_to_place_embed = snn_embeds[center_snn_idx]
        globally_aware_embed = self.cross_attention(snn_node_to_place_embed, attention_context)
        
        value = self.critic_head(globally_aware_embed).squeeze()
        
        candidate_noc_embeds = noc_embeds[candidate_noc_indices]
        num_candidates = candidate_noc_embeds.size(0)
        globally_aware_embed_expanded = globally_aware_embed.expand(num_candidates, -1)
        actor_input = torch.cat([globally_aware_embed_expanded, candidate_noc_embeds], dim=-1)
        logits = self.actor_head(actor_input).squeeze(-1)
        
        return logits, value


# === Module 4: DQN Q-Value Network ===

class SubgraphHeteroDQNNet(nn.Module):
    """
    DQN Q-value network for heterogeneous subgraphs.
    Combines the GNN encoder, cross-attention, and a Q-value head.
    """
    def __init__(self, snn_node_dim: int, noc_node_dim: int, snn_edge_dim: int, gnn_hidden_dim: int, gnn_out_dim: int, gnn_heads: int, head_hidden_dim: int):
        super().__init__()
        self.gnn = HeteroGNNEncoder_Subgraph(snn_node_dim, noc_node_dim, snn_edge_dim, gnn_hidden_dim, gnn_out_dim, gnn_heads)
        self.cross_attention = CrossAttentionModule(gnn_out_dim, gnn_heads)
        
        # Q-head input: concatenated(globally-aware SNN embed, NoC core embed)
        self.q_head = nn.Sequential(
            Linear(gnn_out_dim * 2, head_hidden_dim), nn.ReLU(),
            Linear(head_hidden_dim, 1)
        )

    def forward(self, data: HeteroData, center_snn_idx: int, candidate_noc_indices: torch.Tensor,
                attention_context: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Processes a single subgraph to compute Q-values for candidate actions.
        """
        node_embeds = self.gnn(data)
        snn_embeds, noc_embeds = node_embeds['snn'], node_embeds['noc']
        
        snn_node_to_place_embed = snn_embeds[center_snn_idx]

        globally_aware_embed = self.cross_attention(snn_node_to_place_embed, attention_context)

        candidate_noc_embeds = noc_embeds[candidate_noc_indices]
        num_candidates = candidate_noc_embeds.size(0)
        if num_candidates == 0:
            return torch.empty(0, device=globally_aware_embed.device)
            
        globally_aware_embed_expanded = globally_aware_embed.expand(num_candidates, -1)
        q_head_input = torch.cat([globally_aware_embed_expanded, candidate_noc_embeds], dim=-1)
        q_values = self.q_head(q_head_input).squeeze(-1)
        
        return q_values
        