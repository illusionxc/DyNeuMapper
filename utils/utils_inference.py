
# FILE: utils/gcpn_utils_inference.py
# -*- coding: utf-8 -*-
"""
Utility functions for GCPN algorithms (Inference Version).
"""
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import torch
import networkx as nx
from torch_geometric.data import HeteroData

if TYPE_CHECKING:
    from env.NoCMappingEnv_inference import NoCMappingEnvInference


def build_hetero_subgraph_from_env_enhanced(
    env: 'NoCMappingEnvInference',
    snn_node_to_idx: Dict[str, int],
    full_snn_node_features: torch.Tensor,
    snn_edge_index: torch.Tensor,
    snn_edge_features: torch.Tensor,
    noc_edge_index: torch.Tensor,
    activity_graph_nx: nx.Graph,
    num_hops: int
) -> Optional[Tuple[HeteroData, int, torch.Tensor]]:
    """
    Builds a PyG HeteroData subgraph object based on the current environment state.

    The subgraph includes the k-hop neighborhood around the current neuron to be placed,
    the entire NoC graph, and the existing mappings between them.
    """
    current_neuron_id = env._get_current_neuron_id()
    if current_neuron_id is None:
        return None

    snn_center_global_idx = snn_node_to_idx[current_neuron_id]

    # Create k-hop neighborhood
    subgraph_nodes_set = set(nx.ego_graph(activity_graph_nx, current_neuron_id, radius=num_hops).nodes())
    subgraph_nodes_global_indices_list = [snn_node_to_idx[nid] for nid in subgraph_nodes_set if nid in snn_node_to_idx]
    snn_global_to_local_map = {global_idx: i for i, global_idx in enumerate(subgraph_nodes_global_indices_list)}

    if snn_center_global_idx not in snn_global_to_local_map:
        return None
    center_node_subgraph_idx = snn_global_to_local_map[snn_center_global_idx]

    # Build HeteroData object
    data = HeteroData()
    subgraph_nodes_global_indices_tensor = torch.tensor(subgraph_nodes_global_indices_list, dtype=torch.long, device=env.device)
    data['snn'].x = full_snn_node_features[subgraph_nodes_global_indices_tensor]

    # Filter SNN edges
    mask = torch.isin(snn_edge_index[0], subgraph_nodes_global_indices_tensor) & \
           torch.isin(snn_edge_index[1], subgraph_nodes_global_indices_tensor)
    sub_edge_index_global = snn_edge_index[:, mask]

    if sub_edge_index_global.numel() > 0:
        remapping_tensor = torch.full((full_snn_node_features.shape[0],), -1, dtype=torch.long, device=env.device)
        remapping_tensor[subgraph_nodes_global_indices_tensor] = torch.arange(len(subgraph_nodes_global_indices_tensor), device=env.device)
        data['snn', 'connects', 'snn'].edge_index = remapping_tensor[sub_edge_index_global]
    else:
        data['snn', 'connects', 'snn'].edge_index = torch.empty((2, 0), dtype=torch.long, device=env.device)
    
    data['snn', 'connects', 'snn'].edge_attr = snn_edge_features[mask]

    # Add NoC graph
    noc_load = torch.tensor(env.core_current_load, dtype=torch.float32, device=env.device)
    data['noc'].x = (noc_load / env.core_capacity).unsqueeze(1)
    data['noc', 'links', 'noc'].edge_index = noc_edge_index

    # Add mapping edges
    mapped_snn_local_indices, mapped_noc_indices = [], []
    for snn_id, noc_idx in env.mapping_neuron_to_core_idx.items():
        snn_global_idx = snn_node_to_idx.get(snn_id)
        if snn_global_idx is not None and snn_global_idx in snn_global_to_local_map:
            mapped_snn_local_indices.append(snn_global_to_local_map[snn_global_idx])
            mapped_noc_indices.append(noc_idx)

    if mapped_snn_local_indices:
        edge_idx = torch.tensor([mapped_snn_local_indices, mapped_noc_indices], dtype=torch.long, device=env.device)
        data['snn', 'mapped_to', 'noc'].edge_index = edge_idx
        data['noc', 'rev_mapped_to', 'snn'].edge_index = edge_idx.flip(0)

    candidate_noc_core_indices = torch.tensor(
        [i for i, avail in enumerate(env.get_available_actions_mask()) if avail],
        dtype=torch.long, device=env.device
    )

    return data, center_node_subgraph_idx, candidate_noc_core_indices