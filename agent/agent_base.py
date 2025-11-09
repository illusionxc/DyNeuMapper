
# FILE: agent/gcpn_agent_base.py
# -*- coding: utf-8 -*-
"""
Abstract Base Class for GCPN-series Agents (Inference Version).
"""
import itertools
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, List, TYPE_CHECKING

import torch
import numpy as np
import networkx as nx

# Use TYPE_CHECKING to prevent circular imports at runtime
if TYPE_CHECKING:
    from env.NoCMappingEnv_inference import NoCMappingEnvInference


class BaseGCPAgent(ABC):
    """
    Abstract base class for Graph-based Placement agents.
    Handles common initialization tasks like graph pre-computation.
    """

    def __init__(self, activity_graph_nx: nx.DiGraph, noc_dims: Tuple[int, int],
                 device: torch.device, env: 'NoCMappingEnvInference', agent_config: dict):
        self.env = env
        self.activity_graph_nx = activity_graph_nx
        self.snn_node_list = sorted(list(activity_graph_nx.nodes()))
        self.snn_node_to_idx = {node_id: i for i, node_id in enumerate(self.snn_node_list)}
        self.num_snn_nodes = len(self.snn_node_list)
        self.noc_dims = noc_dims
        self.num_noc_cores = int(noc_dims[0] * noc_dims[1])
        self.core_capacity = agent_config.get('rl_core_capacity', 1)
        self.device = device
        self.agent_config = agent_config

        self.detailed_spike_data = {
            'spike_times_per_sample': activity_graph_nx.graph.get('spike_times_per_sample'),
            'num_samples': activity_graph_nx.graph.get('num_samples'),
            'total_time_steps': activity_graph_nx.graph.get('total_time_steps')
        }

        self.node_features_to_use = agent_config.get('gcpn_node_features', [])
        self.edge_features_to_use = agent_config.get('gcpn_edge_features', [])

        self._precompute_static_graph_and_features()
        self._create_models_and_optimizer()

    def _precompute_static_graph_and_features(self):
        """Pre-computes and normalizes static graph features (nodes and edges)."""
        # Node features
        if self.node_features_to_use:
            node_features_raw = [
                [float(self.activity_graph_nx.nodes.get(node_id, {}).get(feat, 0.0)) for feat in self.node_features_to_use]
                for node_id in self.snn_node_list
            ]
            node_features_np = np.array(node_features_raw, dtype=np.float32)
            if node_features_np.shape[0] > 0:
                mean, std = np.mean(node_features_np, axis=0), np.std(node_features_np, axis=0)
                std[std < 1e-9] = 1.0
                self.snn_node_features = torch.from_numpy((node_features_np - mean) / std).to(self.device)
            else:
                self.snn_node_features = torch.empty((0, len(self.node_features_to_use)), dtype=torch.float32, device=self.device)
        else:
            self.snn_node_features = torch.ones(self.num_snn_nodes, 1, dtype=torch.float32, device=self.device)

        # Edge features
        snn_edges, edge_features_raw = [], []
        for u_str, v_str, data in self.activity_graph_nx.edges(data=True):
            if u_str in self.snn_node_to_idx and v_str in self.snn_node_to_idx:
                snn_edges.append((self.snn_node_to_idx[u_str], self.snn_node_to_idx[v_str]))
                if self.edge_features_to_use:
                    edge_features_raw.append([float(data.get(feat, 0.0)) for feat in self.edge_features_to_use])

        self.snn_edge_index = torch.tensor(snn_edges, dtype=torch.long, device=self.device).t().contiguous() if snn_edges else torch.empty((2, 0), dtype=torch.long, device=self.device)

        if self.edge_features_to_use:
            edge_features_np = np.array(edge_features_raw, dtype=np.float32) if edge_features_raw else np.empty((0, len(self.edge_features_to_use)))
            if edge_features_np.shape[0] > 0:
                mean, std = np.mean(edge_features_np, axis=0), np.std(edge_features_np, axis=0)
                std[std < 1e-9] = 1.0
                self.snn_edge_features = torch.from_numpy((edge_features_np - mean) / std).to(self.device)
            else:
                self.snn_edge_features = torch.empty((0, len(self.edge_features_to_use)), dtype=torch.float32, device=self.device)
        else:
            self.snn_edge_features = torch.ones(self.snn_edge_index.shape[1], 1, dtype=torch.float32, device=self.device)
        
        # NoC graph structure
        noc_graph = nx.grid_2d_graph(self.noc_dims[0], self.noc_dims[1])
        coords_in_order = sorted(list(itertools.product(range(self.noc_dims[0]), range(self.noc_dims[1]))))
        noc_node_idx_mapping = {coord: i for i, coord in enumerate(coords_in_order)}
        noc_edges = [(noc_node_idx_mapping[u], noc_node_idx_mapping[v]) for u, v in noc_graph.edges()]
        noc_edge_index_one_dir = torch.tensor(noc_edges, dtype=torch.long, device=self.device).t().contiguous()
        self.noc_edge_index = torch.cat([noc_edge_index_one_dir, noc_edge_index_one_dir.flip(0)], dim=1)

    @abstractmethod
    def _create_models_and_optimizer(self):
        pass

    @abstractmethod
    def select_action(self, env, evaluation_mode=False):
        pass

    @abstractmethod
    def _get_model_parts_for_saving(self) -> Dict[str, torch.nn.Module]:
        pass

    def _reconstruct_env_from_snapshot(self, placement_snap: Dict[str, int], neuron_id_snap: Optional[str]) -> 'NoCMappingEnvInference':
        """Reconstructs a temporary environment from a state snapshot."""
        # Local import to avoid circular dependency at module load time
        from env.NoCMappingEnv_inference import NoCMappingEnvInference

        temp_env = NoCMappingEnvInference(
            activity_graph=self.activity_graph_nx,
            neurons_to_place_ordered=self.snn_node_list,
            noc_dims=self.noc_dims,
            device=self.device,
            core_capacity=self.core_capacity
        )

        temp_env.mapping_neuron_to_core_idx = placement_snap.copy()
        temp_env.core_current_load = [0] * self.num_noc_cores
        for core_idx in placement_snap.values():
            if 0 <= core_idx < self.num_noc_cores:
                temp_env.core_current_load[core_idx] += 1

        if neuron_id_snap:
            try:
                temp_env.current_neuron_idx_in_order = self.snn_node_list.index(neuron_id_snap)
            except ValueError:
                temp_env.current_neuron_idx_in_order = len(self.snn_node_list)
        else:
            temp_env.current_neuron_idx_in_order = len(self.snn_node_list)

        return temp_env