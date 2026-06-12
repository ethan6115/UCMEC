import os
from pathlib import Path

import numpy as np
import scipy.io as scio

_mpl_cfg = os.path.expanduser("~/.config/matplotlib")
if not os.access(_mpl_cfg, os.W_OK):
    _fallback_cfg = os.path.join("/tmp", "matplotlib")
    os.makedirs(_fallback_cfg, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", _fallback_cfg)

import matplotlib.pyplot as plt


METHOD_LABELS = {
    "Proposed_HDRL": "Proposed HDRL",
    "FlatDRL": "Flat DRL",
    "NoPairScorer": "w/o Pair Scorer",
    "PairConcat": "w/o Pair Interaction",
}

METHOD_COLORS = {
    "Proposed_HDRL": "#1f77b4",
    "FlatDRL": "#9467bd",
    "NoPairScorer": "#ff7f0e",
    "PairConcat": "#2ca02c",
}

TRAINING_REWARD_CONFIGS = [
    {
        "method": "Proposed_HDRL",
        "runs": [
            "results/MyEnv/nlos_cluster_high_ablation_v2/rmappo/hierarchical_pair_scorer_noglobal/run1/reward.mat",
            "results/MyEnv/nlos_cluster_high_ablation_v2/rmappo/hierarchical_pair_scorer_noglobal/run2/reward.mat",
            "results/MyEnv/nlos_cluster_high_ablation_v2/rmappo/hierarchical_pair_scorer_noglobal/run3/reward.mat",
        ],
    },
    {
        "method": "FlatDRL",
        "runs": [
            "results/MyEnv/nlos_cluster_v2/rmappo/flat_drl/run1/reward.mat",
            "results/MyEnv/nlos_cluster_v2/rmappo/flat_drl/run2/reward.mat",
            "results/MyEnv/nlos_cluster_v2/rmappo/flat_drl/run3/reward.mat",
        ],
    },
    {
        "method": "NoPairScorer",
        "runs": [
            "results/MyEnv/nlos_cluster_high_ablation_v2/rmappo/hierarchical_highlow/run1/reward.mat",
            "results/MyEnv/nlos_cluster_high_ablation_v2/rmappo/hierarchical_highlow/run2/reward.mat",
            "results/MyEnv/nlos_cluster_high_ablation_v2/rmappo/hierarchical_highlow/run3/reward.mat",
        ],
    },
    {
        "method": "PairConcat",
        "runs": [
            "results/MyEnv/nlos_cluster_high_ablation_v2/rmappo/hierarchical_pair_scorer_pairconcat/run1/reward.mat",
            "results/MyEnv/nlos_cluster_high_ablation_v2/rmappo/hierarchical_pair_scorer_pairconcat/run2/reward.mat",
            "results/MyEnv/nlos_cluster_high_ablation_v2/rmappo/hierarchical_pair_scorer_pairconcat/run3/reward.mat",
        ],
    },
]


def load_reward(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing reward file: {path}")
    data = scio.loadmat(path)
    if "reward" in data:
        return np.asarray(data["reward"], dtype=np.float64).flatten()
    if "high_reward" in data:
        return np.asarray(data["high_reward"], dtype=np.float64).flatten()
    raise KeyError(f"{path} contains no 'reward' or 'high_reward'. Keys: {list(data.keys())}")


def smooth_curve(y, window):
    y = np.asarray(y, dtype=np.float64)
    if window <= 1 or len(y) < window:
        return y
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    y_padded = np.pad(y, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(y_padded, kernel, mode="valid")


def group_rewards():
    grouped = {}
    for cfg in TRAINING_REWARD_CONFIGS:
        method = cfg["method"]
        for path_str in cfg["runs"]:
            path = Path(path_str)
            reward = load_reward(path)
            grouped.setdefault(method, []).append(
                {
                    "run": path.parent.name,
                    "path": path,
                    "reward": reward,
                }
            )
    return grouped


def plot_training_rewards(
    window=10,
    output_dir="figures",
    show=True,
    band="std",
    ylim=(-0.62, -0.10),
):
    grouped = group_rewards()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), num="Training Rewards", clear=True)

    for cfg in TRAINING_REWARD_CONFIGS:
        method = cfg["method"]
        if method not in grouped:
            continue

        runs = grouped[method]
        color = METHOD_COLORS.get(method, None)
        label = METHOD_LABELS.get(method, method)

        smoothed = []
        for item in runs:
            y = smooth_curve(item["reward"], window)
            smoothed.append(y)

        min_len = min(len(y) for y in smoothed)
        stack = np.stack([y[:min_len] for y in smoothed], axis=0)
        mean = stack.mean(axis=0)
        x = np.arange(min_len)
        if band == "std":
            spread = stack.std(axis=0)
            lower = mean - spread
            upper = mean + spread
        elif band == "minmax":
            lower = stack.min(axis=0)
            upper = stack.max(axis=0)
        else:
            raise ValueError("band must be 'std' or 'minmax'")
        ax.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=0.16,
            linewidth=0,
            label="_nolegend_",
        )
        ax.plot(
            x,
            mean,
            color=color,
            linewidth=2.6,
            label=label,
        )

    ax.set_xlabel("Training Iterations", fontsize=14)
    ax.set_ylabel("Average Low-Level Reward", fontsize=14)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=12)
    fig.tight_layout()

    png_path = output_dir / "training_rewards_drl_methods.png"
    pdf_path = output_dir / "training_rewards_drl_methods.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    plot_training_rewards(window=10, show=True)
