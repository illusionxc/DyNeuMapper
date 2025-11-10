# DyNeuMapper: A Reinforcement Learning Framework for Dynamic SNN-to-NoC Mapping

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-%2300BCD4.svg?style=flat)](https://pyg.org/)

> ## **Note for Reviewers**
>
> This is the anonymous open-source repository for our paper. We provide the **inference version** of our code to demonstrate the core components and facilitate the reproduction of our results.
>
> This repository includes:
> * The full implementation of our core **DyNeuMapper model**.
> * The definition of the NoC simulation environment.
> * The full suite for calculating performance metrics.
>
> To ensure easy reproducibility, you can directly run the main script `main.py`. It will automatically load our provided pre-trained model (trained on mlp-4 with $\lambda_{distance}=0$) and perform inference. **This will generate the attention visualizations (corresponding to Fig. 5a in the paper), the final mapping load distribution plot, and detailed performance metrics in JSON files，thus achieving the effect of "w/o Spatial" as shown in Fig. 5 of the paper.**
>
> The complete code, including the training pipeline, will be released publicly upon the paper's acceptance.

------


## Key Features

- **High-Performance RL Agent:** A pre-trained DQN-based agent for efficient SNN-to-NoC mapping.

- **Comprehensive Evaluation Suite:** A suite of metrics are calculated, including comm_cost, throughput, max_link_load, avg_packet_latency, total_energy_consumption, and saturation_throughput.

- **Advanced Visualization:** Built-in tools to generate publication-quality visualizations for the final NoC core load distribution and the inner workings of the Spatio-Temporal Cross Attention mechanism.

  

## One-Click Reproduction: Visualizations & Results

**Run the `main.py` script once to automatically generate all the following results, fully replicating the complete outputs of the best-performing inference experiment:**

### 1. Final Mapping Load Distribution

**The script automatically generates a visualization of the final neuron placement on the NoC grid.** The color of each core indicates its load relative to its capacity, providing a clear view of resource utilization and placement balance.

<img src="visualizations/load_distribution.png" alt="image-20250728155130029" style="zoom: 20%;" />

*Caption: Final load distribution on an 6x6 NoC. Each cell shows the `(row, col)` coordinate and the number of neurons mapped to it versus its capacity.*

### 2. Spatio-Temporal Cross Attention (STCA) Heatmaps

**With one click, the script generates attention visualizations that reveal the agent’s decision logic.** For a given neuron being placed (the "Query"), the colors show which previously placed neurons (the "Context") the agent paid the most attention to. This reveals the agent's reasoning process for its placement decision.

<img src="visualizations/attention_map.png" alt="image-20250728155304276" style="zoom:75%;" />
*Caption: Aggregated attention weights at different placement steps (10, 50, 100, etc.). The agent places the 'Query' neuron by focusing on the locations of high-importance context neurons.*

### 3. Performance Metrics

**The `main.py` script also auto-generates a `performance_metrics.json` file. A snippet of the output is shown below:**

```json
{
    "model_path": "data/DyNeuMapper_dqn_agent_model",
    "graph_file": "result/snn_analysis_output/deepsnn_layers128_64_64_32_tau2.0_T8_lr0.001_e1_cifar10/peak_joint_activity_graph_deepsnn_layers128646432_tau2.0_cifar10_T100_strategy_static_full_topo_peak_delay0-1.gexf",
    "performance_metrics": {
        "comm_cost": 240300733.0,
        "throughput": 86680535.0,
        "average_weighted_hops": 2.7722571509278295,
        "max_link_load": 14011404.0,
        "avg_link_load": 2002506.1083333334,
        "load_variance": 10051352347413.379,
        "num_links_used": 69.0,
        "total_energy_consumption": 13237177696.0,
        "average_energy_consumption": 44420059.38255034,
        "avg_packet_latency": 13625.39147367689,
        "max_packet_latency": 35018.0,
        "saturation_throughput": 13.320908538513937
    },
```



## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/DyNeuMapper.git
    cd DyNeuMapper
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    This project requires PyTorch and PyTorch Geometric (PyG). Please follow the official installation instructions for your specific CUDA version.

    - **Install PyTorch:** [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
    - **Install PyG:** [https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

    Then, install the remaining packages:
    ```bash
    pip install -r requirements.txt
    ```
    (A `requirements.txt` file would typically contain `networkx`, `numpy`, `matplotlib`, `seaborn`, `rich`).

## Usage

The main entry point for running inference, evaluation, and visualization is `main.py`.

### One-Click Reproduction

**To reproduce the main results from the paper, simply run the following command. It will use the provided pre-trained model and sample graph.**

```bash
python main.py
```

### Full Command-Line Options

For more control, you can use the following arguments:

```bash
python main_visualize.py \
    --model_load_path path/to/model/dsta_net_dqn_agent_model \
    --activity_graph_gexf_file path/to/snn_activity_graph.gexf \
    --steps_to_visualize 10 50 100 150 200 250 \
    --viz_output_dir visualizations/ \
    --device cpu
```

### Key Arguments

-   `--model_load_path`: **(Required)** Path prefix to the pre-trained model files. The script will automatically load associated configuration files if found in the same directory.
-   `--activity_graph_gexf_file`: **(Required)** Path to the SNN activity graph in GEXF format.
-   `--steps_to_visualize`: A list of placement step numbers at which to capture and plot attention data.
-   `--viz_output_dir`: Directory where all output files (metrics as JSON, plots as PDF) will be saved.
-   `--device`: The computation device to use (`cpu` or `cuda`).

## Project Structure

```
.
├── agent/
│   ├── dyneumapper_agent_dqn_inference.py # The core DyNeuMapper agent logic
│   └── agent_base.py                 # Abstract base class for the agent
├── data/
│   ├── best_mapping_info.json             # Experiment config loaded by main.py
│   ├── DyNeuMapper_dqn_agent_model_noc_embeds.pth # Pre-trained NoC embeddings
│   └── DyNeuMapper_dqn_agent_model_policy_net.pth # Pre-trained policy network
├── env/
│   └── NoCMappingEnv_inference.py         # The NoC mapping environment
├── models/
│   ├── dyneumapper_model.py               # DyNeuMapper network architecture (STCA)
│   └── hetero_subgraph_model.py      # The Hetero-Subgraph GNN encoder
├── result/
│   └── snn_analysis_output/               # Contains sample SNN activity graphs
├── utils/
│   ├── utils_inference.py            # Utilities for building subgraphs
│   └── noc_utils.py                       # Comprehensive NoC performance metrics calculation
├── visualizations/data/
│   ├── attention_map.pdf                  # Generated attention heatmap
│   ├── load_distribution.pdf              # Generated load distribution map
│   └── performance_metrics.json           # Generated metrics file
├── main.py                                # Main script for inference, evaluation, and visualization
├── README.md                              # This file
└── requirements.txt                       # Python dependencies
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
