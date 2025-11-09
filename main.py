# FILE: main_visualize.py
# -*- coding: utf-8 -*-
"""
DyneuMapper Inference, Visualization, and Multi-Run Evaluation Script.

Functionality:
1. Loads a pre-trained DyneuMapper Agent.
2. Runs the full inference process multiple times to get robust metrics.
3. For the best-performing run:
    a. Collects attention data and generates an aggregated visualization.
    b. Generates a final NoC load distribution map.
    c. Saves its detailed performance metrics to a JSON file.
4. Calculates and saves the average performance metrics over all runs.
"""
import argparse
import os
import json
from collections import defaultdict
from typing import Optional, Tuple, List, Dict
import copy

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import seaborn as sns
from rich import print

# --- Project Module Imports ---
from agent.dyneumapper_agent_dqn_inference import DyneuMapperAgentInference
from env.NoCMappingEnv_inference import NoCMappingEnvInference
from utils.utils_inference import build_hetero_subgraph_from_env_enhanced
from models.dyneumapper_model import BatchDistanceAwareCrossAttention
from utils.noc_utils import NoCUtils

# --- Plotting Style Configuration ---
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300


# --- Visualization and Helper Classes/Functions ---

class VisBatchDistanceAwareCrossAttention(BatchDistanceAwareCrossAttention):
    """An extension of BatchDistanceAwareCrossAttention that also returns attention weights for visualization."""
    def forward(self, query: torch.Tensor, context: Optional[torch.Tensor], dist_matrix: Optional[torch.Tensor] = None,
                context_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if context is None or context.numel() == 0:
            return query, torch.empty(0, device=query.device)
        B, N_ctx, D = context.shape
        q, k, v = self.q_proj(query), self.k_proj(context), self.v_proj(context)
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N_ctx, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N_ctx, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if self.distance_lambda != 0 and dist_matrix is not None:
            scores = scores + (dist_matrix.unsqueeze(1) * self.distance_lambda)
        if context_mask is not None:
            scores = scores.masked_fill(context_mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, 1, D)
        output = self.out_proj(attn_output.squeeze(1))
        final_output = self.norm(query + output)
        return final_output, attn_weights

# --- Plotting Functions (Beautified Version) ---

def plot_single_head_attention_heatmap(
    ax: plt.Axes,
    title: str,
    noc_dims: Tuple[int, int],
    core_coords_list: List[Tuple[int, int]],
    mapping_neuron_to_core_idx: Dict[str, int],
    context_neuron_ids: List[str],
    attention_weights_for_head: np.ndarray,
    vmin: float,
    vmax: float
):

    noc_rows, noc_cols = noc_dims
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')

    heatmap_grid = np.zeros((noc_rows, noc_cols))
    weight_map = {neuron_id: weight for neuron_id, weight in zip(context_neuron_ids, attention_weights_for_head)}

    for neuron_id, core_idx in mapping_neuron_to_core_idx.items():
        r, c = core_coords_list[core_idx]
        heatmap_grid[r, c] += weight_map.get(neuron_id, 0)

    sns.heatmap(
        data=np.flipud(heatmap_grid),
        ax=ax,
        cmap='tab20',
        linewidths=0.5,
        linecolor='white',
        cbar=False,
        vmin=vmin,
        vmax=vmax
    )

    ax.set_xticks(np.arange(noc_cols) + 0.5)
    ax.set_yticks(np.arange(noc_rows) + 0.5)
    ax.set_xticklabels(np.arange(noc_cols), fontsize=20, fontweight='bold')
    ax.set_yticklabels(np.arange(noc_rows)[::-1], fontsize=20, fontweight='bold')
    ax.tick_params(length=0)





def plot_multi_step_aggregated_attention(vis_data, noc_dims, core_coords, output_path):
    """Generates and saves a figure with attention heatmaps from multiple steps."""
    if not vis_data:
        return

    num_plots = len(vis_data)
    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 5), constrained_layout=True)
    axes_flat = [axes] if num_plots == 1 else axes.flatten()

    all_weights = [d['aggregated_weights'].max() for d in vis_data if d['aggregated_weights'].size > 0]
    global_vmax = max(all_weights) if all_weights else 1.0

    for i, data_point in enumerate(vis_data):
        ax = axes_flat[i]
        plot_single_head_attention_heatmap(
            ax, f"Step {data_point['step']}\nQuery: '{data_point['current_neuron_id']}'",
            noc_dims, core_coords, data_point['mapping_snapshot'],
            data_point['context_neuron_ids'], data_point['aggregated_weights'], 0, global_vmax
        )

    sm = plt.cm.ScalarMappable(cmap='tab20', norm=plt.Normalize(vmin=0, vmax=global_vmax))

    cbar = fig.colorbar(sm, ax=axes_flat, location='bottom', shrink=0.7, aspect=40, pad=0.1)
    cbar.set_label('STCA Weight on Core', fontsize=26, weight='bold')
    cbar.ax.tick_params(labelsize=16)
    plt.savefig(output_path, format='pdf', dpi=300)
    plt.close(fig)
    print(f"\n[bold green]STCA attention map saved to: {output_path}[/bold green]")




def plot_final_mapping_load(output_path, mapping, noc_dims, core_capacity):
    fig, ax = plt.subplots(figsize=(max(8, noc_dims[1]), max(8, noc_dims[0])))
    core_loads = defaultdict(int)
    for core_idx in mapping.values(): core_loads[core_idx] += 1
    cmap = plt.cm.get_cmap('YlGn')
    for r in range(noc_dims[0]):
        for c in range(noc_dims[1]):
            core_idx = r * noc_dims[1] + c; load = core_loads.get(core_idx, 0)
            norm_load = min(load / core_capacity, 1.0) if core_capacity > 0 else 0
            rect = patches.Rectangle((c, r), 1, 1, linewidth=1, edgecolor='black', facecolor=cmap(norm_load))
            ax.add_patch(rect)
            text_color = 'black' if norm_load < 0.6 else 'white'
            ax.text(c + 0.5, r + 0.5, f"({r},{c})\n{load}/{core_capacity}", ha='center', va='center', fontsize=9, color=text_color)
    ax.set_xlim(0, noc_dims[1]); ax.set_ylim(0, noc_dims[0])
    ax.set_xticks(np.arange(noc_dims[1]) + 0.5, labels=[str(i) for i in range(noc_dims[1])])
    ax.set_yticks(np.arange(noc_dims[0]) + 0.5, labels=[str(i) for i in range(noc_dims[0])])
    ax.set_xlabel("NoC Columns"); ax.set_ylabel("NoC Rows")
    ax.invert_yaxis(); ax.set_title("Final Mapping Load Distribution", fontsize=14)
    ax.set_aspect('equal', adjustable='box')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=core_capacity))
    fig.colorbar(sm, ax=ax, label=f'Core Load (Capacity: {core_capacity})')
    plt.tight_layout(); plt.savefig(output_path, format='pdf'); plt.close(fig)
    print(f"[bold green]Final mapping load distribution saved to: {output_path}[/bold green]")


# --- Core Logic Functions ---

def run_single_inference(agent: DyneuMapperAgentInference, env: NoCMappingEnvInference, 
                         steps_to_visualize: set = None) -> Tuple[Dict[str, int], List[Dict]]:
    """Run a single full inference process from start to finish."""
    if steps_to_visualize is None:
        steps_to_visualize = set()
        
    local_env = copy.deepcopy(env)
    local_env.reset()
    agent.reset_episode_specific_state()
    
    visualization_data = []

    for step in range(local_env.num_total_neurons_to_place):
        current_neuron_id = local_env._get_current_neuron_id()
        if current_neuron_id is None: break

        action, _, _ = agent.select_action(local_env, evaluation_mode=True)
        
        if step in steps_to_visualize:
            print(f"  - Collecting attention data for step {step}...")
            with torch.no_grad():
                attention_context, _ = agent._prepare_attention_inputs(local_env, torch.tensor([action], device=agent.device))
                if attention_context is not None and agent.policy_net.cross_attention.__class__.__name__ == 'VisBatchDistanceAwareCrossAttention':
                    snn_global_idx = agent.snn_node_to_idx[current_neuron_id]
                    query_static_feature = agent.snn_node_features[snn_global_idx]
                    query = agent.policy_net.static_feature_projector(query_static_feature).unsqueeze(0)
                    _, attention_w = agent.policy_net.cross_attention(query, attention_context.unsqueeze(0))
                    
                    if attention_w.numel() > 0:
                        weights_np = np.max(attention_w.squeeze().cpu().numpy(), axis=0)
                        visualization_data.append({
                            "step": step, "current_neuron_id": current_neuron_id,
                            "context_neuron_ids": list(local_env.mapping_neuron_to_core_idx.keys()),
                            "aggregated_weights": weights_np,
                            "mapping_snapshot": local_env.mapping_neuron_to_core_idx.copy(),
                        })

        if action == -1:
            print(f"[bold red]Error: Agent could not select a valid action at step {step}. Aborting run.[/bold red]")
            break

        local_env.step(action)
        if agent.agent_config.get('dyneumapper_use_dynamic_context', True):
            agent._update_dynamic_context(local_env.get_state_snapshot(), action, agent.policy_net, agent.noc_core_base_embeddings)

    return local_env.mapping_neuron_to_core_idx, visualization_data



def evaluate_and_save_metrics(output_path: str, mapping: Dict, graph: nx.DiGraph, env: NoCMappingEnvInference, args: argparse.Namespace):
    """Calculates all performance metrics for a given mapping and saves them to a JSON file."""
    print(f"\n[yellow]Calculating performance metrics...[/yellow]")
    if not mapping:
        print(f"[red]Error: Mapping is empty. Cannot calculate metrics.[/red]")
        return

    mapping_coords = {nid: env.core_coords_list[cidx] for nid, cidx in mapping.items()}
    sim_params = {
        'link_bandwidth': args.link_bandwidth, 'router_pipeline_delay': args.router_pipeline_delay,
        'bits_per_flit': args.bits_per_flit, 'avg_packet_length': args.avg_packet_length,
        'energy_per_bit_link': args.energy_per_bit_link, 'energy_per_bit_router': args.energy_per_bit_router,
        'burst_window_size': args.burst_window_size,
    }

    all_metrics = NoCUtils.calculate_all_noc_metrics(activity_graph=graph, mapping_coords=mapping_coords,
                                                   noc_dims=env.noc_dims, simulation_params=sim_params)
    
    results_to_save = {
        "model_path": args.model_load_path, "graph_file": args.activity_graph_gexf_file,
        "performance_metrics": all_metrics,
        "mapping_details": {
            "num_neurons_mapped": len(mapping),
            "noc_utilization": len(set(mapping.values())) / env.num_cores if env.num_cores > 0 else 0,
            "mapping": {str(k): int(v) for k, v in mapping.items()}
        }
    }
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=4)
        print(f"[bold green]Performance metrics saved to: {output_path}[/bold green]")
    except Exception as e:
        print(f"[bold red]Error saving metrics to JSON: {e}[/bold red]")

def load_configs_from_result_file(model_load_path, args):
    """Loads experiment configuration from a 'best_mapping_info.json' file."""
    try:
        exp_dir = os.path.dirname(os.path.abspath(model_load_path))
        result_file_path = os.path.join(exp_dir, 'best_mapping_info.json')
        if not os.path.exists(result_file_path):
            print(f"[yellow]Warning: 'best_mapping_info.json' not found in '{exp_dir}'. Using default arguments.[/yellow]")
            return args
        print(f"[bold blue]Loading experiment config from '{result_file_path}'...[/bold blue]")
        with open(result_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        params_start_index = next((i for i, line in enumerate(lines) if '# ----- all parameters -----' in line), -1)
        if params_start_index == -1: return args
        params_loaded_count = 0
        for line in lines[params_start_index + 1:]:
            line = line.strip()
            if line.startswith('# -----') or not line: continue
            if line.startswith('#'): line = line.lstrip('#').strip()
            parts = line.split(':', 1)
            if len(parts) != 2: continue
            key, value_str = parts[0].strip(), parts[1].strip()
            try: value = eval(value_str)
            except: value = value_str
            setattr(args, key, value)
            params_loaded_count += 1
        print(f"[green]Successfully loaded and applied {params_loaded_count} parameters.[/green]")
        return args
    except Exception as e:
        print(f"[bold red]Error loading config from result file: {e}[/bold red]")
        return args

def main():
    parser = argparse.ArgumentParser(description='DyneuMapper Inference & Visualization')
    parser.add_argument('--model_load_path', type=str, default='data/DyNeuMapper_dqn_agent_model', help='Path prefix for the pre-trained model.')
    parser.add_argument('--activity_graph_gexf_file', type=str, default='result/snn_analysis_output/deepsnn_layers128_64_64_32_tau2.0_T8_lr0.001_e1_cifar10/peak_joint_activity_graph_deepsnn_layers128646432_tau2.0_cifar10_T100_strategy_static_full_topo_peak_delay0-1.gexf', help='Path to the SNN activity graph GEXF file.')
    parser.add_argument('--steps_to_visualize', type=int, nargs='+', default=[10, 50, 100, 150, 200, 250], help='List of steps to visualize attention for.')
    parser.add_argument('--viz_output_dir', type=str, default='visualizations', help='Output directory for visualization results.')
    parser.add_argument('--device', type=str, default='cpu', help='Computation device (cpu or cuda).')
    
    parser.add_argument('--link_bandwidth', type=float, default=1.0)
    parser.add_argument('--router_pipeline_delay', type=int, default=2)
    parser.add_argument('--bits_per_flit', type=int, default=64)
    parser.add_argument('--avg_packet_length', type=int, default=5)
    parser.add_argument('--energy_per_bit_link', type=float, default=0.5)
    parser.add_argument('--energy_per_bit_router', type=float, default=1.0)
    parser.add_argument('--burst_window_size', type=int, default=10)
    
    args, _ = parser.parse_known_args()
    args = load_configs_from_result_file(args.model_load_path, args)
    
    model_name_dir = os.path.basename(os.path.dirname(os.path.abspath(args.model_load_path)))
    output_dir = os.path.join(args.viz_output_dir, model_name_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n[yellow]1. Initializing Environment and Agent[/yellow]")
    activity_graph = nx.read_gexf(args.activity_graph_gexf_file)
    if 'num_samples' not in activity_graph.graph: activity_graph.graph['num_samples'] = 10
    if 'total_time_steps' not in activity_graph.graph: activity_graph.graph['total_time_steps'] = 1000
    
    neurons_to_place_ordered = sorted(list(activity_graph.nodes()))
    device = torch.device(args.device)
    env = NoCMappingEnvInference(activity_graph=activity_graph, neurons_to_place_ordered=neurons_to_place_ordered,
                               noc_dims=(args.noc_rows, args.noc_cols), core_capacity=args.rl_core_capacity, device=device)
    agent = DyneuMapperAgentInference(activity_graph_nx=activity_graph, noc_dims=(args.noc_rows, args.noc_cols),
                                     device=device, env=env, agent_config=vars(args))
    
    print("\n[yellow]2. Loading Pre-trained Model and Preparing for Visualization[/yellow]")
    agent.load_models(args.model_load_path)
    # Replace attention module for visualization purposes
    vis_agent = copy.deepcopy(agent)
    vis_attention_module = VisBatchDistanceAwareCrossAttention(
        embed_dim=vis_agent.policy_net.cross_attention.embed_dim, heads=vis_agent.policy_net.cross_attention.num_heads,
        distance_lambda=vis_agent.policy_net.cross_attention.distance_lambda).to(vis_agent.device)
    vis_attention_module.load_state_dict(vis_agent.policy_net.cross_attention.state_dict())
    vis_agent.policy_net.cross_attention = vis_attention_module
    vis_agent.policy_net.eval()
    
    print("\n[yellow]3. Starting Inference Run[/yellow]")
    final_mapping, visualization_data = run_single_inference(vis_agent, env, steps_to_visualize=set(args.steps_to_visualize))
    
    if not final_mapping:
        print("[bold red]Inference failed to produce a valid mapping. Exiting.[/bold red]")
        return
        
    print("\n[yellow]4. Generating Visualizations[/yellow]")
    plot_multi_step_aggregated_attention(visualization_data, env.noc_dims, env.core_coords_list, os.path.join(output_dir, 'attention_map.pdf'))
    plot_final_mapping_load(os.path.join(output_dir, 'load_distribution.pdf'), final_mapping, env.noc_dims, env.core_capacity)

    print("\n[yellow]5. Evaluating and Saving Metrics[/yellow]")
    evaluate_and_save_metrics(os.path.join(output_dir, 'performance_metrics.json'),
                              final_mapping, activity_graph, env, args)

    print("\n[bold blue]All tasks complete.[/bold blue]")

if __name__ == '__main__':
    main()