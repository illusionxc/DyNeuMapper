# FILE: agent/dyneumapper_agent_dqn_inference.py
# -*- coding: utf-8 -*-
"""
Inference-only DyneuMapper Agent.

This version is streamlined for inference, removing all components related to
training, such as the learning loop, optimizer, replay buffer, and target network.
It retains only the logic for model creation, loading, and action selection.
"""
import os
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from rich import print

from agent.agent_base import BaseGCPAgent
from models.dyneumapper_model import DyneuMapperNet_DQN
from env.NoCMappingEnv_inference import NoCMappingEnvInference
from utils.noc_utils import manhattan_distance_torch
from utils.utils_inference import build_hetero_subgraph_from_env_enhanced


class DyneuMapperAgentInference(BaseGCPAgent):
    """Inference-only version of the DyneuMapper Agent."""

    def _create_models_and_optimizer(self):
        """
        Creates only the necessary models for inference.
        No optimizers or target networks are created.
        """
        config = self.agent_config
        self.subgraph_radius = config['subgraph_radius']

        snn_node_dim = self.snn_node_features.shape[1]
        snn_edge_dim = self.snn_edge_features.shape[1]
        noc_node_dim = 1
        gnn_out_dim = config['gcpn_gnn_hidden_dim']
        noc_embed_dim = gnn_out_dim
        self.context_embed_dim = config.get('dyneumapper_context_embed_dim', gnn_out_dim)

        model_args = (
            snn_node_dim, noc_node_dim, snn_edge_dim,
            config['gcpn_gnn_hidden_dim'], gnn_out_dim, config['gcpn_gat_n_heads'],
            snn_node_dim, noc_embed_dim, self.context_embed_dim,
            config['gcpn_head_hidden_dim'], config
        )

        self.policy_net = DyneuMapperNet_DQN(*model_args).to(self.device)
        self.noc_core_base_embeddings = nn.Embedding(self.num_noc_cores, noc_embed_dim).to(self.device)

        self.snn_base_embedding = None
        if not self.agent_config.get('dyneumapper_use_dynamic_context', True):
            self.snn_base_embedding = torch.nn.Linear(snn_node_dim, self.context_embed_dim).to(self.device)

        self.dynamic_context_pool = torch.zeros(self.num_snn_nodes, self.context_embed_dim, device=self.device)
        self.core_coords_list_torch = torch.tensor(self.env.core_coords_list, dtype=torch.float, device=self.device)

    def select_action(self, env: NoCMappingEnvInference, evaluation_mode: bool = True) -> Tuple[int, None, None]:
        """
        Selects the best action based on the current policy network (always in evaluation mode).
        """
        self.policy_net.eval()

        current_neuron_id = env._get_current_neuron_id()
        if not current_neuron_id:
            return -1, None, None

        snn_global_idx = self.snn_node_to_idx[current_neuron_id]
        query_static_feature = self.snn_node_features[snn_global_idx]

        available_cores_mask = env.get_available_actions_mask()
        candidate_noc_indices = torch.tensor([i for i, avail in enumerate(available_cores_mask) if avail], device=self.device, dtype=torch.long)
        if candidate_noc_indices.numel() == 0:
            return -1, None, None
        candidate_noc_embeds = self.noc_core_base_embeddings(candidate_noc_indices)

        placement_step = env.current_neuron_idx_in_order
        attention_context, distance_matrix = self._prepare_attention_inputs(env, candidate_noc_indices)

        with torch.no_grad():
            q_values = self.policy_net(query_static_feature, candidate_noc_embeds, attention_context, placement_step, distance_matrix)

        if q_values.numel() == 0:
            return -1, None, None

        action_in_candidates = q_values.argmax().item()
        return candidate_noc_indices[action_in_candidates].item(), None, None

    def _prepare_attention_inputs(self, env: NoCMappingEnvInference, candidate_noc_indices: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Prepares the context and distance matrix for the attention mechanism."""
        mapped_snn_ids = list(env.mapping_neuron_to_core_idx.keys())
        if not mapped_snn_ids:
            return None, None

        mapped_global_indices = torch.tensor([self.snn_node_to_idx[nid] for nid in mapped_snn_ids], dtype=torch.long, device=self.device)

        if self.agent_config.get('dyneumapper_use_dynamic_context', True):
            attention_context = self.dynamic_context_pool[mapped_global_indices]
        else:
            mapped_static_features = self.snn_node_features[mapped_global_indices]
            attention_context = self.snn_base_embedding(mapped_static_features)

        distance_matrix = None
        if self.agent_config.get('dyneumapper_use_distance_awareness', True):
            mapped_core_indices = torch.tensor([env.mapping_neuron_to_core_idx[nid] for nid in mapped_snn_ids], device=self.device)
            mapped_core_coords = self.core_coords_list_torch[mapped_core_indices]
            candidate_coords = self.core_coords_list_torch[candidate_noc_indices]
            distance_matrix = manhattan_distance_torch(candidate_coords, mapped_core_coords)

        return attention_context, distance_matrix

    def _update_dynamic_context(self, state_snapshot: tuple, action: int, net: nn.Module, noc_embed_layer: nn.Embedding):
        """Updates the dynamic context embedding for a placed neuron."""
        net.eval()
        placement_map, neuron_id, _ = state_snapshot

        temp_env = self._reconstruct_env_from_snapshot(placement_map, neuron_id)
        build_result = build_hetero_subgraph_from_env_enhanced(
            temp_env, self.snn_node_to_idx, self.snn_node_features,
            self.snn_edge_index, self.snn_edge_features, self.noc_edge_index,
            self.activity_graph_nx, self.subgraph_radius
        )
        if build_result is None:
            return
        subgraph, center_snn_idx, _ = build_result

        with torch.no_grad():
            node_embeds = net.gnn(subgraph.to(self.device))
        snn_gnn_embed = node_embeds['snn'][center_snn_idx]

        noc_core_idx_tensor = torch.tensor([action], device=self.device)
        noc_embed = noc_embed_layer(noc_core_idx_tensor).squeeze(0)

        fused_input = torch.cat([snn_gnn_embed, noc_embed])
        dynamic_embed = net.context_fusion_mlp(fused_input)

        if self.agent_config.get('dyneumapper_use_temporal_encoding', True):
            step = len(placement_map)
            dynamic_embed = net.temporal_encoder_context(dynamic_embed, step)

        snn_global_idx = self.snn_node_to_idx[neuron_id]
        self.dynamic_context_pool[snn_global_idx] = dynamic_embed.detach()

    def reset_episode_specific_state(self):
        """Resets state that changes with each episode."""
        if self.agent_config.get('dyneumapper_use_dynamic_context', True):
            self.dynamic_context_pool.zero_()

    def _get_model_parts_for_saving(self) -> Dict[str, nn.Module]:
        """Returns a dictionary of model components to be saved or loaded."""
        parts = {
            'policy_net': self.policy_net,
            'noc_embeds': self.noc_core_base_embeddings
        }
        if self.snn_base_embedding is not None:
            parts['snn_base_embedding'] = self.snn_base_embedding
        return parts

    def load_models(self, path_prefix: str):
        """Loads model weights from specified files."""
        print(f"  Loading models from '{path_prefix}_*.pth'...")
        model_parts = self._get_model_parts_for_saving()
        loaded_count = 0
        for name, model in model_parts.items():
            if model is not None:
                model_path = f"{path_prefix}_{name}.pth"
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    print(f"    - Successfully loaded '{name}'.")
                    loaded_count += 1
                else:
                    print(f"    - [yellow]Warning: Model file not found, skipping '{name}': {model_path}[/yellow]")
        if loaded_count == 0:
            raise FileNotFoundError(f"Error: No model files found with prefix '{path_prefix}'. Please check the path.")