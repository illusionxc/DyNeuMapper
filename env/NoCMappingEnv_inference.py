
# FILE: env/NoCMappingEnv_inference.py
# -*- coding: utf-8 -*-
"""
Inference-only NoC Mapping Environment.

This version is streamlined for inference by removing all logic related to
reward calculation. The step() function only updates the state and checks for completion.
"""
from typing import Dict, List, Optional, Tuple

import torch
import networkx as nx

from utils.noc_utils import get_noc_core_coordinates


class NoCMappingEnvInference:
    """
    An environment for simulating the process of mapping SNN neurons to NoC cores.
    This version is optimized for inference and does not calculate rewards.
    """
    def __init__(self,
                 activity_graph: nx.DiGraph,
                 neurons_to_place_ordered: List[str],
                 noc_dims: Tuple[int, int],
                 device: torch.device,
                 core_capacity: int):
        self.activity_graph = activity_graph
        self.neurons_to_place_ordered = neurons_to_place_ordered
        self.num_total_neurons_to_place = len(neurons_to_place_ordered)
        self.noc_dims = noc_dims
        self.num_cores = int(noc_dims[0] * noc_dims[1])
        self.core_coords_list = get_noc_core_coordinates(noc_dims)
        self.device = device
        self.core_capacity = core_capacity

        self.current_neuron_idx_in_order: int = 0
        self.mapping_neuron_to_core_idx: Dict[str, int] = {}
        self.core_current_load: List[int] = []
        self.reset()

    def reset(self):
        """Resets the environment to its initial state."""
        self.current_neuron_idx_in_order = 0
        self.mapping_neuron_to_core_idx = {}
        self.core_current_load = [0] * self.num_cores
        return None

    def step(self, action_core_idx: int) -> Tuple[None, bool]:
        """
        Executes an action (placing a neuron) and updates the environment state.

        Returns:
            A tuple (None, done_flag).
        """
        current_neuron_id = self._get_current_neuron_id()
        if current_neuron_id is None:
            return None, True

        if self._is_core_available_for_placement(action_core_idx):
            self.mapping_neuron_to_core_idx[current_neuron_id] = action_core_idx
            self.core_current_load[action_core_idx] += 1
            self.current_neuron_idx_in_order += 1

        done = self.current_neuron_idx_in_order >= self.num_total_neurons_to_place
        return None, done

    def _get_current_neuron_id(self) -> Optional[str]:
        """Returns the ID of the neuron currently being placed."""
        if self.current_neuron_idx_in_order < self.num_total_neurons_to_place:
            return self.neurons_to_place_ordered[self.current_neuron_idx_in_order]
        return None

    def _is_core_available_for_placement(self, core_idx: int) -> bool:
        """Checks if a core is available based on its capacity."""
        return self.core_current_load[core_idx] < self.core_capacity

    def get_available_actions_mask(self) -> List[bool]:
        """Returns a boolean mask of available cores (actions)."""
        return [self._is_core_available_for_placement(i) for i in range(self.num_cores)]

    def get_state_snapshot(self) -> Tuple[Dict[str, int], Optional[str], List[bool]]:
        """Returns a lightweight snapshot of the current environment state for agent use."""
        return (
            self.mapping_neuron_to_core_idx.copy(),
            self._get_current_neuron_id(),
            self.get_available_actions_mask()
        )