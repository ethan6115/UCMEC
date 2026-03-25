"""
Diagnostic: Does per-user optimal cluster size heterogeneity exist?

Method:
  For each (seed, k) pair, reset the env with that seed and run a full episode
  with ALL users fixed at cluster size k. Record per-user uplink delay, total
  delay, and uplink rate.

  After collecting data for k=1..K_MAX, for each user in each seed find which k
  gives the lowest total delay. Plot:
    1. Distribution of optimal k across all users/seeds
    2. optimal k vs user's best-AP beta (channel quality proxy for distance)
    3. Per-user uplink delay vs k, grouped by near/far users

  If per-user heterogeneity exists, we should see:
    - Near users (high best_beta) → optimal k is small (1-2)
    - Far users  (low best_beta)  → optimal k is larger (3-5+)
    - The curves for near vs far cross somewhere around k=2-3

Usage:
  cd UCMEC-mmWave-Fronthaul
  python scripts/sweep_peruser_optimal_k.py --model_low <path_to_actor.pt>
"""
import sys
import os
import argparse
import copy
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.MA_UCMEC_dyna_noncoop_hierarchical_peruser_hotspot import (
    MA_UCMEC_dyna_noncoop_hierarchical_peruser as HierarchicalEnv,
)
from algorithms.algorithm.r_actor_critic import R_Actor
from config import get_config

# ─── Configuration ────────────────────────────────────────────────────────────
K_SWEEP   = [1, 2, 3, 4, 5, 7]   # cluster sizes to test
N_SEEDS   = 30                           # number of seeds
SEEDS     = list(range(N_SEEDS))
N_EPISODES_PER_SEED = 1                  # episodes per (seed, k) combo

# Path to a trained low-level actor (cluster2 or randcluster)
DEFAULT_MODEL_LOW = (
    r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\hotspotEnv\peruser_clustersize"
    r"\rmappo\noncoop_rnn\hotspot_cluster2\models\actor_999.pt"
)
# ──────────────────────────────────────────────────────────────────────────────


def run_episode_fixed_k(env, actor, args, k_val, rng_seed):
    """Run one episode with all users fixed at cluster size k_val.

    Returns per-user arrays:
        uplink_delay_ms   shape (M_sim,)
        total_delay_ms    shape (M_sim,)
        uplink_rate_Mbps  shape (M_sim,)
        best_beta_dB      shape (M_sim,)  - top-1 AP channel gain per user
        front_delay_ms    shape (M_sim,)
    """
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    env.seed(rng_seed)
    obs = env.reset()

    M_sim = env.M_sim
    recurrent_N  = args.recurrent_N
    hidden_size  = args.hidden_size
    rnn_states = np.zeros((M_sim, recurrent_N, hidden_size), dtype=np.float32)
    masks      = np.ones((M_sim, 1), dtype=np.float32)

    # fixed cluster action: index = k_val - 1 for all users
    action_id = np.full(M_sim, k_val - 1, dtype=np.int32)

    # accumulators
    uplink_sum = np.zeros(M_sim, dtype=np.float64)
    total_sum  = np.zeros(M_sim, dtype=np.float64)
    front_sum  = np.zeros(M_sim, dtype=np.float64)
    rate_sum   = np.zeros(M_sim, dtype=np.float64)
    step_count = 0
    dones      = [False] * M_sim

    while not all(dones):
        # advance channel once per step (same as eval_script)
        if hasattr(env, "advance_channel"):
            env.advance_channel()

        # apply fixed cluster at each hierarchical decision point
        if step_count % 10 == 0:
            env.apply_high_action(action_id)

        obs_batch = np.stack(obs)
        with torch.no_grad():
            actions, _, rnn_states = actor(obs_batch, rnn_states, masks, deterministic=True)
        # Actor outputs integer indices (M_sim, 1); env.step() expects one-hot (M_sim, action_dim)
        # Same conversion as env_runner.py line 475:
        #   actions_env = np.squeeze(np.eye(action_space.n)[actions], 2)
        action_dim = env.action_dim
        action_idx = actions.cpu().numpy().astype(int).reshape(M_sim)
        action_onehot = np.eye(action_dim)[action_idx]  # (M_sim, action_dim)

        obs, rewards, dones, info = env.step(action_onehot)

        # per-user delay: all env attributes are in seconds → convert to ms
        td = np.asarray(env.delay_last,           dtype=np.float64).reshape(M_sim) * 1000.0
        ul = np.asarray(env._segment_avg_uplink,  dtype=np.float64) * 1000.0
        fd = np.asarray(env._segment_avg_front,   dtype=np.float64) * 1000.0
        # uplink rate from info[0] (only agent 0 carries the dict)
        ur_avg = info[0].get("avg_uplink_rate_Mbps", 0.0) if isinstance(info[0], dict) else 0.0
        ur = np.full(M_sim, ur_avg, dtype=np.float64)

        uplink_sum += ul
        total_sum  += td
        front_sum  += fd
        rate_sum   += ur
        step_count += 1

    n = max(1, step_count)

    # best-AP beta for each user (top-1 entry from get_global_obs beta block)
    # We call get_global_obs to trigger _top10_ap_idx construction
    global_obs = env.get_global_obs()           # (M_sim, 52)
    # beta_norm occupies dims 0..9 (10 dims), take dim 0 = top-1 AP
    beta_norm_top1 = global_obs[:, 0]           # normalised to [0,1]
    # convert back to dB: clip range was (-121, -95.5)
    beta_dB = beta_norm_top1 * (121.0 - 95.5) - 121.0

    return {
        "uplink_delay_ms":  uplink_sum / n,
        "total_delay_ms":   total_sum  / n,
        "front_delay_ms":   front_sum  / n,
        "uplink_rate_Mbps": rate_sum   / n,
        "best_beta_dB":     beta_dB,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_low", default=DEFAULT_MODEL_LOW)
    parser.add_argument("--out_dir",   default="scripts/sweep_output")
    cli = parser.parse_args()

    os.makedirs(cli.out_dir, exist_ok=True)

    # ── load low-level actor ──────────────────────────────────────────────────
    cfg_parser = get_config()
    args = cfg_parser.parse_args([])
    args.use_recurrent_policy      = True
    args.use_naive_recurrent_policy = False

    device = torch.device("cpu")
    dummy_env = HierarchicalEnv(render=False, seed=1)
    obs_space = dummy_env.observation_space[0]
    act_space = dummy_env.action_space[0]

    actor = R_Actor(args, obs_space, act_space, device)
    if os.path.exists(cli.model_low):
        actor.load_state_dict(torch.load(cli.model_low, map_location=device))
        print(f"Loaded low-level model: {cli.model_low}")
    else:
        print(f"[WARNING] model not found: {cli.model_low}. Using random weights.")
    actor.eval()

    M_sim = dummy_env.M_sim

    # ── sweep ─────────────────────────────────────────────────────────────────
    # results[k][seed] = dict of per-user arrays (M_sim,)
    results = {k: [] for k in K_SWEEP}

    for k in K_SWEEP:
        print(f"\n=== k = {k} ===")
        for seed in SEEDS:
            env = HierarchicalEnv(render=False, seed=seed)
            r   = run_episode_fixed_k(env, actor, args, k, rng_seed=seed)
            results[k].append(r)
            print(f"  seed={seed:3d}  avg_total={r['total_delay_ms'].mean():.1f}ms"
                  f"  avg_uplink={r['uplink_delay_ms'].mean():.1f}ms"
                  f"  avg_front={r['front_delay_ms'].mean():.1f}ms")

    # ── analysis ──────────────────────────────────────────────────────────────
    # For each (seed, user), find optimal k (lowest total delay)
    # Shape: (N_SEEDS, M_sim, len(K_SWEEP))
    total_delay_arr  = np.stack(
        [np.stack([results[k][s]["total_delay_ms"]  for s in range(N_SEEDS)]) for k in K_SWEEP],
        axis=2
    )   # (N_SEEDS, M_sim, len(K_SWEEP))
    uplink_delay_arr = np.stack(
        [np.stack([results[k][s]["uplink_delay_ms"] for s in range(N_SEEDS)]) for k in K_SWEEP],
        axis=2
    )
    front_delay_arr  = np.stack(
        [np.stack([results[k][s]["front_delay_ms"]  for s in range(N_SEEDS)]) for k in K_SWEEP],
        axis=2
    )
    best_beta_arr    = np.stack(
        [results[K_SWEEP[0]][s]["best_beta_dB"] for s in range(N_SEEDS)]
    )   # (N_SEEDS, M_sim), use k=K_SWEEP[0] as representative

    # optimal k index per (seed, user)
    optimal_k_idx = np.argmin(total_delay_arr, axis=2)   # (N_SEEDS, M_sim)
    optimal_k_val = np.array(K_SWEEP)[optimal_k_idx]     # (N_SEEDS, M_sim)

    # ── print summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Global average total delay per k:")
    for i, k in enumerate(K_SWEEP):
        mean_d = total_delay_arr[:, :, i].mean()
        print(f"  k={k}: {mean_d:.2f} ms")

    print("\nDistribution of optimal k across all (seed, user) pairs:")
    flat_opt_k = optimal_k_val.flatten()
    for k in K_SWEEP:
        frac = np.mean(flat_opt_k == k)
        print(f"  k={k}: {frac*100:.1f}%")

    # Heterogeneity indicator: std of optimal k within each seed
    opt_k_std_per_seed = optimal_k_val.std(axis=1)  # (N_SEEDS,)
    print(f"\nMean std of optimal k within a seed: {opt_k_std_per_seed.mean():.3f}")
    print("  (0 = all users prefer same k, >0 = heterogeneity exists)")

    # ── correlation: best_beta vs optimal_k ───────────────────────────────────
    flat_beta = best_beta_arr.flatten()
    flat_k    = optimal_k_val.flatten()
    corr = np.corrcoef(flat_beta, flat_k)[0, 1]
    print(f"\nPearson corr(best_beta_dB, optimal_k): {corr:.4f}")
    print("  (negative → weaker channel → larger optimal k)")

    # ── oracle upper bound ────────────────────────────────────────────────────
    # For each (seed, user), take the minimum delay across all k values.
    # This is the best possible total delay if high-level assigns k perfectly.
    oracle_delay = total_delay_arr.min(axis=2)          # (N_SEEDS, M_sim)
    oracle_mean  = oracle_delay.mean()
    oracle_std   = oracle_delay.std()

    k2_idx = K_SWEEP.index(2) if 2 in K_SWEEP else None
    k1_idx = K_SWEEP.index(1) if 1 in K_SWEEP else None
    cluster2_mean = total_delay_arr[:, :, k2_idx].mean() if k2_idx is not None else float("nan")
    cluster1_mean = total_delay_arr[:, :, k1_idx].mean() if k1_idx is not None else float("nan")

    print(f"\n{'='*60}")
    print("Oracle upper bound analysis:")
    print(f"  Fixed k=1  avg delay : {cluster1_mean:.2f} ms")
    print(f"  Fixed k=2  avg delay : {cluster2_mean:.2f} ms")
    print(f"  Oracle (per-user opt): {oracle_mean:.2f} ms  (std={oracle_std:.2f})")
    margin = cluster2_mean - oracle_mean
    print(f"  Oracle margin over k=2: {margin:.2f} ms  ({margin/cluster2_mean*100:.1f}%)")
    if margin > 10:
        print("  → Perfect hierarchical CAN beat fixed k=2 by a meaningful margin.")
        print("    If current model still loses, the problem is in training quality,")
        print("    not a fundamental environment limitation.")
    elif margin > 0:
        print("  → Perfect hierarchical beats k=2 only marginally.")
        print("    Training overhead may erase this gain in practice.")
    else:
        print("  → Oracle cannot beat fixed k=2 even with perfect assignment.")
        print("    Heterogeneity exists but the gain is cancelled by interference.")

    # ── plot 1: delay vs k, averaged over all users ───────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    k_arr = np.array(K_SWEEP)

    for ax, arr, title in zip(
        axes,
        [total_delay_arr, uplink_delay_arr, front_delay_arr],
        ["Total delay (ms)", "Uplink delay (ms)", "Front delay (ms)"],
    ):
        mean_d = arr.mean(axis=(0, 1))  # (len(K_SWEEP),)
        std_d  = arr.std(axis=(0, 1))
        ax.plot(k_arr, mean_d, "o-")
        ax.fill_between(k_arr, mean_d - std_d, mean_d + std_d, alpha=0.2)
        ax.set_xlabel("Cluster size k")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(k_arr)
        ax.grid(True)
    plt.tight_layout()
    out1 = os.path.join(cli.out_dir, "delay_vs_k_global.png")
    plt.savefig(out1, dpi=150)
    print(f"\nSaved: {out1}")
    plt.close()

    # ── plot 2: near vs far users ─────────────────────────────────────────────
    # Split users into near (top 50% beta) and far (bottom 50% beta) per seed
    median_beta = np.median(best_beta_arr, axis=1, keepdims=True)  # (N_SEEDS, 1)
    near_mask   = best_beta_arr >= median_beta  # (N_SEEDS, M_sim)
    far_mask    = ~near_mask

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, arr, title in zip(
        axes,
        [total_delay_arr, uplink_delay_arr],
        ["Total delay (ms)", "Uplink delay (ms)"],
    ):
        for mask, label, color in [
            (near_mask, "Near users (top 50% beta)", "tab:blue"),
            (far_mask,  "Far users (bot 50% beta)",  "tab:orange"),
        ]:
            vals = []
            for i, k in enumerate(K_SWEEP):
                # gather values for this group across all seeds
                slice_k = arr[:, :, i]          # (N_SEEDS, M_sim)
                v = slice_k[mask]               # flat array of matching entries
                vals.append(v)
            means = [v.mean() for v in vals]
            stds  = [v.std()  for v in vals]
            ax.plot(k_arr, means, "o-", label=label, color=color)
            ax.fill_between(
                k_arr,
                np.array(means) - np.array(stds),
                np.array(means) + np.array(stds),
                alpha=0.15, color=color,
            )
        ax.set_xlabel("Cluster size k")
        ax.set_ylabel(title)
        ax.set_title(f"{title}: near vs far users")
        ax.legend()
        ax.set_xticks(k_arr)
        ax.grid(True)
    plt.tight_layout()
    out2 = os.path.join(cli.out_dir, "delay_vs_k_near_far.png")
    plt.savefig(out2, dpi=150)
    print(f"Saved: {out2}")
    plt.close()

    # ── plot 3: scatter best_beta vs optimal_k ────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(flat_beta, flat_k, alpha=0.3, s=10)
    # add binned mean
    bins   = np.percentile(flat_beta, np.linspace(0, 100, 11))
    bin_idx = np.digitize(flat_beta, bins[1:-1])
    bin_k_mean = [flat_k[bin_idx == b].mean() if (bin_idx == b).any() else np.nan
                  for b in range(len(bins) - 1)]
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    ax.plot(bin_centers, bin_k_mean, "r-o", linewidth=2, label="binned mean")
    ax.set_xlabel("Best-AP beta (dB)  [higher = closer to AP]")
    ax.set_ylabel("Optimal cluster size k")
    ax.set_title(f"Channel quality vs optimal k  (corr={corr:.3f})")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    out3 = os.path.join(cli.out_dir, "beta_vs_optimal_k.png")
    plt.savefig(out3, dpi=150)
    print(f"Saved: {out3}")
    plt.close()

    # ── summary verdict ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("VERDICT:")
    std_threshold = 0.5
    corr_threshold = -0.15
    has_hetero = opt_k_std_per_seed.mean() > std_threshold or corr < corr_threshold
    if has_hetero:
        print("  Heterogeneity EXISTS.")
        print("  → Per-user optimal k varies meaningfully.")
        print("  → Hierarchical training should be able to outperform fixed k.")
        print("  → Problem is likely in reward signal or training, not environment.")
    else:
        print("  Heterogeneity is WEAK or ABSENT.")
        print("  → All users tend to prefer the same k.")
        print("  → Fixed k=2 is a strong baseline; hierarchical adds little value.")
        print("  → Consider modifying the environment (e.g., fronthaul blockage,")
        print("     larger area, sparser APs) to create meaningful diversity.")
    print("="*60)


if __name__ == "__main__":
    main()
