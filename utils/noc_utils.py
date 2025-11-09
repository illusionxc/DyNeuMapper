  
# FILE: utils/noc_utils.py
# -*- coding: utf-8 -*-
"""
NoC Performance Metrics Calculation Module.

Provides tools for calculating various Network-on-Chip (NoC) performance metrics,
including communication cost, link load, congestion, energy, latency, and more.
"""
from collections import defaultdict
from typing import Dict, Tuple, List, Optional

import torch
import networkx as nx
import numpy as np


def manhattan_distance(coord1: tuple[int, int], coord2: tuple[int, int]) -> int:
    """Calculates the Manhattan distance between two 2D coordinates."""
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])


def get_noc_core_coordinates(noc_dims: tuple[int, int]) -> list[tuple[int, int]]:
    """Generates a list of NoC core coordinates in row-major order."""
    rows, cols = noc_dims
    return [(r, c) for r in range(rows) for c in range(cols)]


def manhattan_distance_torch(coords1: torch.Tensor, coords2: torch.Tensor) -> torch.Tensor:
    """
    Efficiently computes the Manhattan distance matrix between two sets of
    coordinate points using PyTorch broadcasting.
    """
    diff = coords1.unsqueeze(-2) - coords2.unsqueeze(-3)
    return torch.abs(diff).sum(dim=-1)


class NoCUtils:
    """
    A utility class for calculating NoC performance metrics.
    Provides a suite of static methods to evaluate an SNN-to-NoC mapping.
    """

    @staticmethod
    def calculate_all_noc_metrics(
        activity_graph: nx.DiGraph,
        mapping_coords: Dict[str, Tuple[int, int]],
        noc_dims: Tuple[int, int],
        simulation_params: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Unified entry point to calculate all NoC performance metrics.

        Args:
            activity_graph: SNN activity graph with communication weights.
            mapping_coords: A dictionary mapping neuron IDs to (row, col) coordinates.
            noc_dims: The dimensions of the NoC grid (rows, cols).
            simulation_params: A dictionary of simulation parameters.

        Returns:
            A dictionary containing all calculated performance metrics.
        """
        if not mapping_coords:
            return defaultdict(lambda: float('inf'))

        sim_params = simulation_params or {}
        all_metrics = {}

        comm_cost, total_inter_core_volume, avg_hops = NoCUtils.calculate_communication_metrics(
            activity_graph, mapping_coords
        )
        all_metrics['comm_cost'] = comm_cost
        all_metrics['throughput'] = total_inter_core_volume
        all_metrics['average_weighted_hops'] = avg_hops

        link_loads = NoCUtils.calculate_link_loads(activity_graph, mapping_coords, noc_dims)
        all_metrics.update(NoCUtils.get_congestion_metrics(link_loads))

        total_energy = NoCUtils.calculate_energy_consumption(comm_cost, total_inter_core_volume, sim_params)
        all_metrics['total_energy_consumption'] = total_energy
        all_metrics['average_energy_consumption'] = NoCUtils.calculate_average_energy_consumption(total_energy, activity_graph)

        avg_latency, max_latency = NoCUtils.calculate_latency(activity_graph, mapping_coords, link_loads, sim_params)
        all_metrics['avg_packet_latency'] = avg_latency
        all_metrics['max_packet_latency'] = max_latency

        all_metrics['saturation_throughput'] = NoCUtils.calculate_saturation_throughput(activity_graph, mapping_coords, noc_dims, sim_params)

        burst_metrics = NoCUtils.analyze_bursty_congestion(activity_graph, mapping_coords, noc_dims, sim_params)
        if burst_metrics:
            all_metrics.update(burst_metrics)

        return all_metrics

    @staticmethod
    def calculate_communication_metrics(
        activity_graph: nx.DiGraph,
        mapping_coords: Dict[str, Tuple[int, int]]
    ) -> Tuple[float, float, float]:
        """Calculates total communication cost, volume, and average weighted hops."""
        comm_cost = 0.0
        total_volume = 0.0
        for u, v, data in activity_graph.edges(data=True):
            if u in mapping_coords and v in mapping_coords and mapping_coords[u] != mapping_coords[v]:
                volume = float(data.get('source_activity', 0.0))
                dist = manhattan_distance(mapping_coords[u], mapping_coords[v])
                comm_cost += volume * dist
                total_volume += volume
        avg_hops = comm_cost / (total_volume + 1e-9)
        return comm_cost, total_volume, avg_hops

    @staticmethod
    def calculate_link_loads(
        activity_graph: nx.DiGraph,
        mapping_coords: Dict[str, Tuple[int, int]],
        noc_dims: Tuple[int, int]
    ) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]:
        """Calculates the load on each directed link using XY routing."""
        link_loads = NoCUtils._initialize_link_loads(noc_dims)
        if not mapping_coords:
            return link_loads

        for u, v, data in activity_graph.edges(data=True):
            comm_volume = float(data.get('source_activity', 0.0))
            if comm_volume <= 1e-6 or u not in mapping_coords or v not in mapping_coords:
                continue

            start_coord, end_coord = mapping_coords[u], mapping_coords[v]
            if start_coord == end_coord:
                continue

            curr_r, curr_c = start_coord
            end_r, end_c = end_coord

            while curr_c != end_c:
                next_c = curr_c + (1 if end_c > curr_c else -1)
                link_loads[((curr_r, curr_c), (curr_r, next_c))] += comm_volume
                curr_c = next_c
            
            while curr_r != end_r:
                next_r = curr_r + (1 if end_r > curr_r else -1)
                link_loads[((curr_r, curr_c), (next_r, curr_c))] += comm_volume
                curr_r = next_r
                
        return link_loads

    @staticmethod
    def get_congestion_metrics(link_loads: Dict) -> Dict[str, float]:
        """Calculates congestion-related metrics from link loads."""
        if not link_loads:
            return {"max_link_load": 0.0, "avg_link_load": 0.0, "load_variance": 0.0, "num_links_used": 0.0}
        
        load_values = np.array(list(link_loads.values()))
        return {
            "max_link_load": float(np.max(load_values)) if load_values.size > 0 else 0.0,
            "avg_link_load": float(np.mean(load_values)) if load_values.size > 0 else 0.0,
            "load_variance": float(np.var(load_values)) if load_values.size > 0 else 0.0,
            "num_links_used": float(np.sum(load_values > 1e-9))
        }

    @staticmethod
    def calculate_energy_consumption(
        communication_cost: float,
        total_inter_core_volume: float,
        simulation_params: Optional[Dict] = None
    ) -> float:
        """Estimates total communication energy consumption."""
        sim_params = simulation_params or {}
        e_link = sim_params.get('energy_per_bit_link', 0.5)
        e_router = sim_params.get('energy_per_bit_router', 1.0)
        bits_per_flit = sim_params.get('bits_per_flit', 64)
        
        return (communication_cost * bits_per_flit * e_link) + (total_inter_core_volume * bits_per_flit * e_router)

    @staticmethod
    def calculate_latency(
        activity_graph: nx.DiGraph,
        mapping_coords: Dict,
        link_loads: Dict,
        simulation_params: Optional[Dict] = None
    ) -> Tuple[float, float]:

        sim_params = simulation_params or {}
        router_delay = sim_params.get('router_pipeline_delay', 2) 
        link_bandwidth = sim_params.get('link_bandwidth', 1.0)  
        avg_packet_length = sim_params.get('avg_packet_length', 5) 

        num_samples = activity_graph.graph.get('num_samples')
        total_time_steps = activity_graph.graph.get('total_time_steps')


        if not (num_samples and total_time_steps):
            total_hops = 0
            num_comm_events = 0
            for u, v, data in activity_graph.edges(data=True):
                 if u in mapping_coords and v in mapping_coords and mapping_coords[u] != mapping_coords[v]:
                    total_hops += manhattan_distance(mapping_coords[u], mapping_coords[v])
                    num_comm_events += 1
            avg_hops = total_hops / num_comm_events if num_comm_events > 0 else 0
            return avg_hops * router_delay, ((activity_graph.graph.get('noc_rows', 4)-1) + (activity_graph.graph.get('noc_cols', 4)-1)) * router_delay

        total_packet_latency = 0.0
        max_packet_latency = 0.0
        num_inter_core_communications = 0

        for u, v, data in activity_graph.edges(data=True):
            if u not in mapping_coords or v not in mapping_coords: continue
            
            start_coord, end_coord = mapping_coords[u], mapping_coords[v]
            if start_coord == end_coord: continue
            
            hops = manhattan_distance(start_coord, end_coord)

            zero_load_latency = (avg_packet_length - 1) + hops * router_delay
            
            path = NoCUtils._get_xy_routing_path(start_coord, end_coord)
            path_congestion_delay = 0.0
            
            for i in range(len(path) - 1):
                link = (path[i], path[i+1])
                link_load = link_loads.get(link, 0.0)

                arrival_rate = link_load / (num_samples * total_time_steps)
                service_rate = link_bandwidth
                utilization = arrival_rate / service_rate

                if utilization < 1.0:
                    wait_time_per_flit = utilization / (1.0 - utilization)
                    path_congestion_delay += avg_packet_length * wait_time_per_flit
                else:
                    path_congestion_delay += 1000.0 * avg_packet_length 

            packet_latency = zero_load_latency + path_congestion_delay
            
            total_packet_latency += packet_latency
            max_packet_latency = max(max_packet_latency, packet_latency)
            num_inter_core_communications += 1

        # 4. 计算最终的平均延迟
        avg_packet_latency = total_packet_latency / num_inter_core_communications if num_inter_core_communications > 0 else 0.0
        
        return avg_packet_latency, max_packet_latency

    @staticmethod
    def calculate_saturation_throughput(
        activity_graph: nx.DiGraph,
        mapping_coords: Dict[str, Tuple[int, int]],
        noc_dims: Tuple[int, int],
        simulation_params: Optional[Dict] = None
    ) -> float:
        """Estimates the network's saturation throughput via bisection analysis."""
        if not mapping_coords: return 0.0

        sim_params = simulation_params or {}
        link_bandwidth = sim_params.get('link_bandwidth', 1.0)
        num_samples = activity_graph.graph.get('num_samples')
        total_time_steps = activity_graph.graph.get('total_time_steps')
        if not (num_samples and total_time_steps): return 0.0
        
        total_simulation_time = num_samples * total_time_steps
        if total_simulation_time == 0: return 0.0
            
        rows, cols = noc_dims
        bisection_col_idx, bisection_row_idx = cols // 2, rows // 2
        
        vertical_load, horizontal_load, total_volume = 0.0, 0.0, 0.0
        for u, v, data in activity_graph.edges(data=True):
            if u in mapping_coords and v in mapping_coords:
                start, end = mapping_coords[u], mapping_coords[v]
                if start == end: continue
                volume = float(data.get('source_activity', 0.0))
                total_volume += volume
                if (start[1] < bisection_col_idx and end[1] >= bisection_col_idx) or \
                   (start[1] >= bisection_col_idx and end[1] < bisection_col_idx):
                    vertical_load += volume
                if start[1] == end[1]:
                    if (start[0] < bisection_row_idx and end[0] >= bisection_row_idx) or \
                       (start[0] >= bisection_row_idx and end[0] < bisection_row_idx):
                        horizontal_load += volume
        
        vert_rate = vertical_load / total_simulation_time
        horiz_rate = horizontal_load / total_simulation_time
        
        vert_bw = rows * link_bandwidth
        horiz_bw = cols * link_bandwidth
        
        vert_ratio = vert_rate / (vert_bw + 1e-9)
        horiz_ratio = horiz_rate / (horiz_bw + 1e-9)
        
        max_ratio = max(vert_ratio, horiz_ratio)
        total_avg_rate = total_volume / total_simulation_time
        
        return total_avg_rate / max_ratio if max_ratio > 1e-6 else total_avg_rate

    @staticmethod
    def analyze_bursty_congestion(
        activity_graph: nx.DiGraph,
        mapping_coords: Dict,
        noc_dims: Tuple,
        simulation_params: Optional[Dict] = None
    ) -> Optional[Dict[str, float]]:
        """Analyzes bursty congestion using a sliding window approach."""
        sim_params = simulation_params or {}
        window_size = sim_params.get('burst_window_size', 10)
        stride = sim_params.get('burst_stride', max(1, window_size // 2))

        spike_times_per_sample = activity_graph.graph.get('spike_times_per_sample')
        num_samples = activity_graph.graph.get('num_samples')
        total_time_steps = activity_graph.graph.get('total_time_steps')
        
        if not all([isinstance(d, v) for d, v in [(spike_times_per_sample, dict), (num_samples, int), (total_time_steps, int)]]) or \
           total_time_steps < window_size:
            return None
        
        all_window_max_loads = []
        graph_edges = list(activity_graph.edges())
        
        for start_time in range(0, total_time_steps - window_size + 1, stride):
            end_time = start_time + window_size
            source_activity_in_window = defaultdict(float)
            for sample_id in range(num_samples):
                sample_spikes = spike_times_per_sample.get(sample_id, {})
                for neuron_id, spike_times in sample_spikes.items():
                    start_idx = np.searchsorted(spike_times, start_time, side='left')
                    end_idx = np.searchsorted(spike_times, end_time, side='left')
                    if (spikes_in_win := end_idx - start_idx) > 0:
                        source_activity_in_window[neuron_id] += spikes_in_win
            
            if not source_activity_in_window:
                all_window_max_loads.append(0.0)
                continue
            
            window_graph = nx.DiGraph()
            window_graph.add_nodes_from(activity_graph.nodes())
            edges_to_add = []
            for u_str, v_str in graph_edges:
                try:
                    numeric_part = ''.join(filter(str.isdigit, u_str))
                    source_id_int = int(numeric_part) if numeric_part else -1
                except (ValueError, TypeError): continue
                if source_id_int != -1 and (activity := source_activity_in_window.get(source_id_int, 0.0)) > 0:
                    edges_to_add.append((u_str, v_str, {'source_activity': activity}))
            
            if edges_to_add: window_graph.add_edges_from(edges_to_add)
            
            window_link_loads = NoCUtils.calculate_link_loads(window_graph, mapping_coords, noc_dims)
            all_window_max_loads.append(NoCUtils.get_congestion_metrics(window_link_loads)['max_link_load'])
        
        if not all_window_max_loads: return None
            
        peak_max_load = np.max(all_window_max_loads)
        avg_max_load = np.mean(all_window_max_loads)
        return {
            "peak_max_link_load": peak_max_load,
            "avg_max_link_load_over_windows": avg_max_load,
            "congestion_burstiness_ratio": peak_max_load / (avg_max_load + 1e-9)
        }

    @staticmethod
    def calculate_average_energy_consumption(total_energy: float, graph: nx.DiGraph) -> float:
        """Calculates the average communication energy per neuron."""
        return total_energy / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0.0

    @staticmethod
    def _initialize_link_loads(noc_dims: Tuple[int, int]) -> Dict:
        """Initializes a dictionary for link loads with all loads set to 0.0."""
        rows, cols = noc_dims
        link_loads = {}
        for r in range(rows):
            for c in range(cols):
                if c + 1 < cols:
                    link_loads[((r, c), (r, c + 1))] = 0.0
                    link_loads[((r, c + 1), (r, c))] = 0.0
                if r + 1 < rows:
                    link_loads[((r, c), (r + 1, c))] = 0.0
                    link_loads[((r + 1, c), (r, c))] = 0.0
        return link_loads

    @staticmethod
    def _get_xy_routing_path(start_coord, end_coord):
        """Returns the list of coordinates for an XY-routed path."""
        path = [start_coord]
        curr_r, curr_c = start_coord
        end_r, end_c = end_coord
        while curr_c != end_c:
            curr_c += 1 if end_c > curr_c else -1
            path.append((curr_r, curr_c))
        while curr_r != end_r:
            curr_r += 1 if end_r > curr_r else -1
            path.append((curr_r, curr_c))
        return path