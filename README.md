# Hierarchical Reinforcement Learning for AP Cluster Composition-Aware Task Offloading in User-Centric MEC with Wireless Fronthaul

A hierarchical multi-agent reinforcement learning implementation for
joint access-point selection, computation offloading, and resource allocation.

## Project Structure

```text
UCMEC/
├── algorithms/              # MAPPO/RMAPPO and hierarchical policy implementations
├── envs/                    # UCMEC environments and environment wrappers
├── eval_experiments/        # Evaluation configs, scripts, outputs, and paper notebook
├── figures/                 # Selected figures
├── results/                 # Pretrained checkpoints and training summaries
├── runner/                  # Training rollout and policy update loops
├── scripts/                 # Plotting, inspection, and calibration utilities
├── train/                   # Main training entry point
├── utils/                   # Replay buffers and shared utilities
├── config.py                # Training arguments and project defaults
├── eval_script.py           # Core model evaluation program
├── eval_model_manifest.txt  # List of model checkpoint included in repository
├── requirements.txt         # Python dependencies
└── README.md                # Installation and operation guide
```
For an overview of the codebase architecture and module responsibilities, see the [Code Architecture and Module Guide](https://hackmd.io/@NYxxxUJHS0CbH0fXRVVbew/S1WmMef8Mg).
## Table of Contents

1. [Requirements](#1-requirements)
2. [Installation](#2-installation)
3. [Training](#3-training)
4. [Model Evaluation](#4-model-evaluation)
5. [Paper Results](#5-paper-results)



## 1. Requirements

Before installation, make sure the following tools are available:

- Linux or Windows Subsystem for Linux (WSL)
- Git
- Conda
- Python 3.10

All commands must be executed from the project root directory.

## 2. Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/ethan6115/UCMEC.git
cd UCMEC
```

### Step 2: Create the Python Environment

```bash
conda create -n ucmec python=3.10 -y
conda activate ucmec
```

### Step 3: Install PyTorch

Install a PyTorch build compatible with the target system. This project was developed and verified with PyTorch 2.7.1 and CUDA 11.8:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

For a different CUDA configuration, select the appropriate installation command from the [official PyTorch installation guide](https://pytorch.org/get-started/previous-versions/).

### Step 4: Install the Remaining Dependencies

```bash
python -m pip install -r requirements.txt
python -m pip check
```

PyTorch is installed separately because the appropriate build depends on the CUDA environment of the target system.

### Step 5: Verify the Installation

```bash
python -c "import torch, cvxpy; assert torch.cuda.is_available(), 'CUDA is not available'; assert 'CLARABEL' in cvxpy.installed_solvers(), 'CLARABEL is not available'; print(f'Installation successful | PyTorch {torch.__version__} | CUDA {torch.version.cuda} | GPU {torch.cuda.get_device_name(0)}')"
```

If the command prints `Installation successful`, the environment is ready.

## 3. Training

All training commands must be executed from the project root directory.

### Step 1: Check the Training Environment

The default hierarchical training environment in `envs/env_discrete.py` is configured for the proposed method. Different baseline and ablation environments require changing the corresponding import before training.

See the [Experiment Operation Guide](https://hackmd.io/@NYxxxUJHS0CbH0fXRVVbew/rJYklAWIMe) for all environment configurations.
### Step 2: Train the Proposed Model

```bash
bash train/run_training.sh \
  --mode hierarchical-pair \
  --experiment_name hierarchical_pair_scorer_noglobal \
  --scenario_name nlos_cluster_high_ablation_v2
```

The `experiment_name` and `scenario_name` arguments determine how training outputs are organized. They do not select the actual training environment.

Training outputs are saved under:

```text
results/MyEnv/<scenario_name>/rmappo/<experiment_name>/run<N>/
```

The `run<N>` directory is created automatically for each new run.

Commands for all baselines and ablation studies are provided in the [Experiment Operation Guide](https://hackmd.io/@NYxxxUJHS0CbH0fXRVVbew/rJYklAWIMe).

## 4. Model Evaluation

Run all evaluation commands from the project root directory.

```bash
# Main performance evaluation and sensitivity analysis
# Produces results for user count, AP count, and blockage density
python eval_experiments/scripts/eval_sweep.py
# Output: eval_experiments/outputs/sweep_outputs/<timestamp>/

# Ablation studies
# Produces cluster-size, candidate-size, and architecture ablation results
python eval_experiments/scripts/eval_ablation.py
# Output: eval_experiments/outputs/ablation_outputs/<timestamp>/

# User-speed sensitivity analysis
python eval_experiments/scripts/eval_speed_sweep.py
# Output: eval_experiments/outputs/speed_sweep_outputs/<timestamp>/

# High-level update interval comparison (H=10 and H=5)
python eval_experiments/scripts/eval_interval.py
# Output: eval_experiments/outputs/interval_outputs/<timestamp>/

# AP-selection quality and selected-rank distribution
python eval_experiments/scripts/analyze_ap_selection.py
# Output: eval_experiments/outputs/ap_selection_outputs/
```

## 5. Paper Results

### Evaluation Tables and Figures

Open the following notebook in VS Code or another Jupyter-compatible editor:

```text
eval_experiments/notebooks/paper_results.ipynb
```

Select the `ucmec` Conda environment as the notebook kernel and run all cells. The required evaluation results are included in the repository.

Generated tables are displayed directly in the notebook. Generated figures are saved in both PNG and PDF formats under:

```text
eval_experiments/outputs/paper_results/generated_figures/
```

The notebook generates the following figures:

- User-count sensitivity
- AP-count sensitivity
- Blockage-density sensitivity
- User-speed sensitivity

### Training Reward Curves

Run the following command from the project root directory:

```bash
python scripts/plot_training_rewards.py
```

The required `reward.mat` files are included with the retained training runs. Generated figures are saved as:

```text
figures/training_rewards_drl_methods.png
figures/training_rewards_drl_methods.pdf
```

### Environment Visualization

Run the following command from the project root directory:

```bash
python scripts/draw_env.py
```

This command generates and displays the default UCMEC environment layout, including users, access points, and the central processing unit.

## Reference Implementation

The main code structure and implementation in this project were developed with reference to the [qlt315/UCMEC-mmWave-Fronthaul](https://github.com/qlt315/UCMEC-mmWave-Fronthaul) repository.

The UCMEC environment used in this project includes corrections and modifications to the original implementation. A detailed comparison of the environment implementations, including the identified issues, corrections, and experimental adjustments, is provided in the [UCMEC Environment Implementation Comparison](https://hackmd.io/@NYxxxUJHS0CbH0fXRVVbew/By4O4KTWfg).