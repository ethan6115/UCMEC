import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
import torch


def build_default_args():
    from config import get_config

    parser = get_config()
    args = parser.parse_args([])
    # Match eval behavior for low-level actor.
    args.use_recurrent_policy = True
    args.use_naive_recurrent_policy = False
    return args


def make_env(seed):
    from envs.MA_UCMEC_dyna_noncoop_hierarchical_peruser_apselect import (
        MA_UCMEC_dyna_noncoop_hierarchical_peruser,
    )

    env = MA_UCMEC_dyna_noncoop_hierarchical_peruser(render=True, seed=seed)
    env.seed(seed)
    return env


def load_low_actor(model_path, device, args, env):
    from algorithms.algorithm.r_actor_critic import R_Actor

    obs_space = env.observation_space[0]
    act_space = env.action_space[0]
    actor = R_Actor(args, obs_space, act_space, device)
    state_dict = torch.load(model_path, map_location=device)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor, act_space


def make_action_family(m_sim, bits):
    fam = {}

    def repeat_mask(mask_1d):
        mask = np.array(mask_1d, dtype=np.int32)
        return np.tile(mask[None, :], (m_sim, 1))

    fam["top1"] = repeat_mask([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    fam["top2"] = repeat_mask([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    fam["top3"] = repeat_mask([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    fam["top5"] = repeat_mask([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    fam["alt135"] = repeat_mask([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    fam["tail67"] = repeat_mask([0, 0, 0, 0, 0, 1, 1, 0, 0, 0])
    fam["all_zero_fallback"] = np.zeros((m_sim, bits), dtype=np.int32)

    rng = np.random.default_rng(12345)
    rand2 = np.zeros((m_sim, bits), dtype=np.int32)
    rand3 = np.zeros((m_sim, bits), dtype=np.int32)
    for u in range(m_sim):
        rand2[u, rng.choice(bits, size=2, replace=False)] = 1
        rand3[u, rng.choice(bits, size=3, replace=False)] = 1
    fam["rand_k2"] = rand2
    fam["rand_k3"] = rand3
    return fam


def run_one_interval(env, actor, act_space, low_args, interval_len, obs):
    rnn_states = np.zeros((env.n_agents, low_args.recurrent_N, low_args.hidden_size), dtype=np.float32)
    masks = np.ones((env.n_agents, 1), dtype=np.float32)

    interval_total_delays = []
    interval_cluster_sizes = []

    for _ in range(interval_len):
        if hasattr(env, "advance_channel"):
            env.advance_channel()

        obs_batch = np.stack(obs)
        with torch.no_grad():
            actions, _, rnn_states = actor(obs_batch, rnn_states, masks, deterministic=True)

        action_indices = actions.cpu().numpy().flatten()
        actions_env = np.eye(act_space.n)[action_indices]

        next_obs, rewards, next_dones, infos = env.step(actions_env)
        obs = next_obs
        dones = np.array(next_dones)

        if hasattr(env, "delay_last") and env.delay_last is not None:
            interval_total_delays.extend(env.delay_last[: env.M_sim, 0].tolist())
        if hasattr(env, "current_cluster_size"):
            interval_cluster_sizes.append(env.current_cluster_size.copy())

        masks[dones == True] = 0.0
        if np.all(dones):
            break

    delays = np.array(interval_total_delays, dtype=np.float32) if interval_total_delays else np.array([], dtype=np.float32)
    reward = float(env.compute_interval_reward()) if hasattr(env, "compute_interval_reward") else np.nan

    out = {
        "reward": reward,
        "n_delay_samples": int(delays.size),
        "delay_mean": float(np.mean(delays)) if delays.size > 0 else np.nan,
        "delay_p95": float(np.percentile(delays, 95)) if delays.size > 0 else np.nan,
        "delay_dsr": float(np.mean(delays <= env.tau_c)) if delays.size > 0 else np.nan,
        "cluster_size_mean": float(np.mean(interval_cluster_sizes)) if interval_cluster_sizes else np.nan,
        "cluster_size_last": interval_cluster_sizes[-1].tolist() if interval_cluster_sizes else None,
    }
    return out


def apply_manual_high_action(env, action_mask):
    # Build top10 candidate mapping before apply.
    env.get_global_obs()
    env.apply_high_action(action_mask)

    selected = []
    if hasattr(env, "_top10_ap_idx") and env._top10_ap_idx is not None:
        for u in range(min(2, env.M_sim)):
            mask_u = action_mask[u].astype(bool)
            top10 = env._top10_ap_idx[u]
            if not np.any(mask_u):
                aps = [int(top10[0])]  # fallback behavior in apply_high_action
            else:
                aps = [int(x) for x in top10[mask_u]]
            selected.append({"user": u, "top10": top10.tolist(), "selected": aps})
    return selected


def main():
    parser = argparse.ArgumentParser(description="Test high-level reward sensitivity to manual high actions.")
    parser.add_argument("--low-model", required=True, type=str, help="Path to low-level actor checkpoint (actor_*.pt)")
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    args_cli = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root))

    device = torch.device(args_cli.device)
    low_args = build_default_args()

    # Build one env for space introspection / actor construction.
    env_ref = make_env(args_cli.seed)
    actor, act_space = load_low_actor(args_cli.low_model, device, low_args, env_ref)
    m_sim = env_ref.M_sim
    env_ref.close() if hasattr(env_ref, "close") else None

    candidates = make_action_family(m_sim, 10)

    print(f"Low model: {args_cli.low_model}")
    print(f"Seed: {args_cli.seed}, Interval: {args_cli.interval}")
    print(f"Candidates: {list(candidates.keys())}")

    results = []
    for name, action_mask in candidates.items():
        np.random.seed(args_cli.seed)
        torch.manual_seed(args_cli.seed)
        env = make_env(args_cli.seed)

        # Reset and prime channel/top10, then apply fixed high action.
        obs = env.reset()
        if hasattr(env, "advance_channel"):
            env.advance_channel()
        selected_info = apply_manual_high_action(env, action_mask)
        metrics = run_one_interval(env, actor, act_space, low_args, args_cli.interval, obs)
        metrics["name"] = name
        metrics["selected_preview"] = selected_info
        results.append(metrics)

    print("\n=== Reward Sensitivity Results (one interval) ===")
    for r in sorted(results, key=lambda x: x["reward"], reverse=True):
        print(
            f"{r['name']:>16} | R_high={r['reward']:+.6f} | "
            f"DSR={r['delay_dsr']:.3f} | meanD={r['delay_mean']:.4f}s | p95D={r['delay_p95']:.4f}s | "
            f"cluster_mean={r['cluster_size_mean']:.2f}"
        )
        if r["selected_preview"]:
            for item in r["selected_preview"]:
                print(
                    f"  user{item['user']} selected={item['selected']} (top10 first5={item['top10'][:5]})"
                )


if __name__ == "__main__":
    main()
