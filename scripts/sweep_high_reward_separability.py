import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_seed_spec(spec: str) -> List[int]:
    seeds: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if hi < lo:
                lo, hi = hi, lo
            seeds.extend(range(lo, hi + 1))
        else:
            seeds.append(int(token))
    seeds = sorted(set(seeds))
    if not seeds:
        raise ValueError("No valid seeds parsed from --seeds")
    return seeds


def build_low_args():
    from config import get_config

    parser = get_config()
    args = parser.parse_args([])
    args.use_recurrent_policy = True
    args.use_naive_recurrent_policy = False
    return args


def make_env(seed: int):
    from envs.MA_UCMEC_dyna_noncoop_hierarchical_peruser_hotspot import (
        MA_UCMEC_dyna_noncoop_hierarchical_peruser,
    )

    env = MA_UCMEC_dyna_noncoop_hierarchical_peruser(render=False, seed=seed)
    env.seed(seed)
    return env


def load_low_actor(model_path: str, device: torch.device, low_args, env):
    from algorithms.algorithm.r_actor_critic import R_Actor

    obs_space = env.observation_space[0]
    act_space = env.action_space[0]
    actor = R_Actor(low_args, obs_space, act_space, device)
    state_dict = torch.load(model_path, map_location=device)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor, act_space


def run_fixed_k_episode(env, actor, act_space, low_args, k: int, interval: int, slots: int) -> Dict[str, np.ndarray]:
    obs = env.reset()
    rnn_states = np.zeros((env.n_agents, low_args.recurrent_N, low_args.hidden_size), dtype=np.float32)
    masks = np.ones((env.n_agents, 1), dtype=np.float32)

    interval_rewards: List[float] = []
    step_delay_ms: List[float] = []
    interval_cluster_mean: List[float] = []

    action_id = np.full((env.M_sim,), int(k - 1), dtype=np.int32)

    for step in range(slots):
        if hasattr(env, "advance_channel"):
            env.advance_channel()

        if step % interval == 0:
            if hasattr(env, "apply_high_action"):
                env.apply_high_action(action_id)
            else:
                env.set_high_action(action_id)

        obs_batch = np.stack(obs)
        with torch.no_grad():
            actions, _, rnn_states = actor(obs_batch, rnn_states, masks, deterministic=True)

        action_indices = actions.cpu().numpy().flatten()
        actions_env = np.eye(act_space.n)[action_indices]
        next_obs, _, next_dones, infos = env.step(actions_env)
        obs = next_obs

        if isinstance(infos, (list, tuple)) and len(infos) > 0 and isinstance(infos[0], dict):
            if "avg_total_delay_ms" in infos[0]:
                step_delay_ms.append(float(infos[0]["avg_total_delay_ms"]))

        if hasattr(env, "current_cluster_size"):
            interval_cluster_mean.append(float(np.mean(env.current_cluster_size)))

        dones = np.array(next_dones)
        masks[dones == True] = 0.0

        interval_end = ((step + 1) % interval == 0)
        if interval_end and hasattr(env, "compute_interval_reward"):
            r_vec = env.compute_interval_reward()
            interval_rewards.append(float(np.mean(np.asarray(r_vec, dtype=np.float32))))

        if np.all(dones):
            break

    # Flush partial interval if needed.
    if hasattr(env, "_segment_delay_count") and getattr(env, "_segment_delay_count") > 0 and hasattr(env, "compute_interval_reward"):
        r_vec = env.compute_interval_reward()
        interval_rewards.append(float(np.mean(np.asarray(r_vec, dtype=np.float32))))

    return {
        "interval_reward": np.asarray(interval_rewards, dtype=np.float64),
        "step_delay_ms": np.asarray(step_delay_ms, dtype=np.float64),
        "cluster_mean": np.asarray(interval_cluster_mean, dtype=np.float64),
    }


def mean_std(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(np.mean(x)), "std": float(np.std(x))}


def main():
    parser = argparse.ArgumentParser(
        description="Sweep fixed high-level cluster size k and measure high-reward separability."
    )
    parser.add_argument("--low-model", type=str, required=True, help="Path to low-level actor checkpoint")
    parser.add_argument("--seeds", type=str, default="0-19", help="Seed list/range, e.g. 0-29 or 0,2,4")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per seed per k")
    parser.add_argument("--slots", type=int, default=200, help="Low-level slots per episode")
    parser.add_argument("--interval", type=int, default=10, help="High-level decision interval")
    parser.add_argument("--k-min", type=int, default=1, help="Min cluster size to sweep")
    parser.add_argument("--k-max", type=int, default=10, help="Max cluster size to sweep")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--output",
        type=str,
        default="results/high_reward_separability.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    if not os.path.exists(args.low_model):
        raise FileNotFoundError(f"--low-model not found: {args.low_model}")
    if args.k_min < 1 or args.k_max > 10 or args.k_min > args.k_max:
        raise ValueError("k range must satisfy 1 <= k_min <= k_max <= 10")

    seeds = parse_seed_spec(args.seeds)
    k_values = list(range(args.k_min, args.k_max + 1))
    device = torch.device(args.device)
    low_args = build_low_args()

    env0 = make_env(seeds[0])
    actor, act_space = load_low_actor(args.low_model, device, low_args, env0)
    if hasattr(env0, "close"):
        env0.close()

    print(f"[sweep] low model: {args.low_model}")
    print(f"[sweep] seeds={seeds}")
    print(f"[sweep] k={k_values}, episodes={args.episodes}, slots={args.slots}, interval={args.interval}")

    per_k_reward_samples: Dict[int, List[float]] = {k: [] for k in k_values}
    per_k_delay_samples: Dict[int, List[float]] = {k: [] for k in k_values}
    per_k_seed_means: Dict[int, List[float]] = {k: [] for k in k_values}
    per_k_delay_seed_means: Dict[int, List[float]] = {k: [] for k in k_values}

    for k in k_values:
        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            env = make_env(seed)

            reward_ep_means: List[float] = []
            delay_ep_means: List[float] = []

            for _ in range(args.episodes):
                out = run_fixed_k_episode(
                    env=env,
                    actor=actor,
                    act_space=act_space,
                    low_args=low_args,
                    k=k,
                    interval=args.interval,
                    slots=args.slots,
                )
                r = out["interval_reward"]
                d = out["step_delay_ms"]

                if r.size > 0:
                    per_k_reward_samples[k].extend(r.tolist())
                    reward_ep_means.append(float(np.mean(r)))
                if d.size > 0:
                    per_k_delay_samples[k].extend(d.tolist())
                    delay_ep_means.append(float(np.mean(d)))

            if reward_ep_means:
                per_k_seed_means[k].append(float(np.mean(reward_ep_means)))
            if delay_ep_means:
                per_k_delay_seed_means[k].append(float(np.mean(delay_ep_means)))

            if hasattr(env, "close"):
                env.close()

    summary = {"per_k": {}}
    reward_means = []
    reward_stds = []

    for k in k_values:
        reward_arr = np.asarray(per_k_reward_samples[k], dtype=np.float64)
        delay_arr = np.asarray(per_k_delay_samples[k], dtype=np.float64)
        seed_reward_arr = np.asarray(per_k_seed_means[k], dtype=np.float64)
        seed_delay_arr = np.asarray(per_k_delay_seed_means[k], dtype=np.float64)

        r_stat = mean_std(reward_arr)
        d_stat = mean_std(delay_arr)
        r_seed_stat = mean_std(seed_reward_arr)
        d_seed_stat = mean_std(seed_delay_arr)

        summary["per_k"][str(k)] = {
            "interval_reward_all": r_stat,
            "interval_reward_seed_mean": r_seed_stat,
            "avg_total_delay_ms_all": d_stat,
            "avg_total_delay_ms_seed_mean": d_seed_stat,
            "n_interval_samples": int(reward_arr.size),
            "n_step_delay_samples": int(delay_arr.size),
            "n_seed_means_reward": int(seed_reward_arr.size),
            "n_seed_means_delay": int(seed_delay_arr.size),
        }

        if not np.isnan(r_seed_stat["mean"]):
            reward_means.append(r_seed_stat["mean"])
            reward_stds.append(max(r_seed_stat["std"], 1e-12))

    reward_gap = float(np.max(reward_means) - np.min(reward_means)) if reward_means else float("nan")
    reward_noise = float(np.mean(reward_stds)) if reward_stds else float("nan")
    reward_snr = float(reward_gap / max(reward_noise, 1e-12)) if reward_stds else float("nan")

    summary["separability"] = {
        "reward_gap_over_k": reward_gap,
        "reward_noise_mean_std": reward_noise,
        "reward_snr_gap_over_noise": reward_snr,
        "interpretation": (
            "SNR < 0.5 weak, 0.5~1.0 marginal, > 1.0 usually learnable"
        ),
    }
    summary["config"] = {
        "low_model": args.low_model,
        "seeds": seeds,
        "episodes": args.episodes,
        "slots": args.slots,
        "interval": args.interval,
        "k_values": k_values,
    }

    print("\n[sweep] per-k (seed-mean stats)")
    for k in k_values:
        item = summary["per_k"][str(k)]
        r = item["interval_reward_seed_mean"]
        d = item["avg_total_delay_ms_seed_mean"]
        print(
            f"  k={k:2d} | highR mean={r['mean']:+.6f} std={r['std']:.6f} | "
            f"delay mean={d['mean']:.3f}ms std={d['std']:.3f}"
        )

    sep = summary["separability"]
    print("\n[sweep] separability")
    print(f"  reward_gap_over_k: {sep['reward_gap_over_k']:.6f}")
    print(f"  reward_noise_mean_std: {sep['reward_noise_mean_std']:.6f}")
    print(f"  reward_snr_gap_over_noise: {sep['reward_snr_gap_over_noise']:.6f}")
    print(f"  note: {sep['interpretation']}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[sweep] saved: {out_path}")


if __name__ == "__main__":
    main()
