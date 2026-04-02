import argparse
import copy
import json
import io
import sys
from pathlib import Path
from contextlib import nullcontext, redirect_stderr, redirect_stdout

import numpy as np

# Ensure repository root is importable when this script is run from scripts/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_seed_spec(spec: str):
    seeds = []
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


def phase_name(interval_idx: int, total_intervals: int) -> str:
    if total_intervals <= 0:
        return "all"
    frac = interval_idx / max(1, total_intervals - 1)
    if frac < 1.0 / 3.0:
        return "early"
    if frac < 2.0 / 3.0:
        return "mid"
    return "late"


def evenly_spaced_indices(total: int, k: int):
    if total <= 0:
        return set()
    if k >= total:
        return set(range(total))
    idx = np.linspace(0, total - 1, num=k)
    return set(int(round(x)) for x in idx.tolist())


def make_dummy_low_action(env):
    actions = np.zeros((env.M_sim, env.action_dim), dtype=np.float32)
    actions[:, 0] = 1.0
    return actions


def make_env(seed: int):
    from envs.MA_UCMEC_dyna_noncoop_hierarchical_peruser_hotspot_nlos_obs57_heurlow import (
        MA_UCMEC_dyna_noncoop_hierarchical_peruser,
    )

    env = MA_UCMEC_dyna_noncoop_hierarchical_peruser(render=False, seed=seed)
    env.seed(seed)
    return env


def snapshot_env(env):
    return {
        "rng_state": copy.deepcopy(env.rng.__getstate__()),
        "locations_users": env.locations_users.copy(),
        "user_dest": env.user_dest.copy() if env.user_dest is not None else None,
        "user_speed": env.user_speed.copy() if env.user_speed is not None else None,
        "distance_matrix": env.distance_matrix.copy(),
        "beta": env.beta.copy(),
        "PL": env.PL.copy(),
        "mu": env.mu.copy(),
        "h": env.h.copy(),
        "link_type": env.link_type.copy(),
        "G": env.G.copy(),
        "Task_size": env.Task_size.copy(),
        "Task_density": env.Task_density.copy(),
        "omega_last": env.omega_last.copy(),
        "p_last": env.p_last.copy(),
        "p_idx_last": env.p_idx_last.copy(),
        "delay_last": env.delay_last.copy(),
        "delay_last_clip": env.delay_last_clip.copy(),
        "uplink_rate_access_b": env.uplink_rate_access_b.copy(),
        "step_num": int(env.step_num),
        "_channel_ready": env._channel_ready,
        "theta_current": env.theta_current.copy() if env.theta_current is not None else None,
        "cluster_matrix": env.cluster_matrix.copy() if env.cluster_matrix is not None else None,
        "current_cluster_size": env.current_cluster_size.copy(),
        "cpu_front_quality": env.cpu_front_quality.copy(),
        "_segment_delay_sum": env._segment_delay_sum.copy(),
        "_segment_uplink_sum": env._segment_uplink_sum.copy(),
        "_segment_front_sum": env._segment_front_sum.copy(),
        "_segment_front_log_sum": env._segment_front_log_sum.copy(),
        "_segment_delay_count": env._segment_delay_count,
        "_segment_avg_delay": env._segment_avg_delay.copy(),
        "_segment_avg_uplink": env._segment_avg_uplink.copy(),
        "_segment_avg_front": env._segment_avg_front.copy(),
        "_segment_offload_count": env._segment_offload_count.copy(),
        "_segment_offload_success_count": env._segment_offload_success_count.copy(),
        "_segment_offload_success_ratio": env._segment_offload_success_ratio.copy(),
        "_segment_cpu_counts": env._segment_cpu_counts.copy(),
        "_segment_local_counts": env._segment_local_counts.copy(),
        "_interval_delays": list(env._interval_delays),
        "_pending_high_action": env._pending_high_action.copy() if env._pending_high_action is not None else None,
        "_top10_ap_idx": env._top10_ap_idx.copy() if env._top10_ap_idx is not None else None,
    }


def restore_env(env, snap):
    env.rng.__setstate__(copy.deepcopy(snap["rng_state"]))
    env.locations_users = snap["locations_users"].copy()
    env.user_dest = snap["user_dest"].copy() if snap["user_dest"] is not None else None
    env.user_speed = snap["user_speed"].copy() if snap["user_speed"] is not None else None
    env.distance_matrix = snap["distance_matrix"].copy()
    env.beta = snap["beta"].copy()
    env.PL = snap["PL"].copy()
    env.mu = snap["mu"].copy()
    env.h = snap["h"].copy()
    env.link_type = snap["link_type"].copy()
    env.G = snap["G"].copy()
    env.Task_size = snap["Task_size"].copy()
    env.Task_density = snap["Task_density"].copy()
    env.omega_last = snap["omega_last"].copy()
    env.p_last = snap["p_last"].copy()
    env.p_idx_last = snap["p_idx_last"].copy()
    env.delay_last = snap["delay_last"].copy()
    env.delay_last_clip = snap["delay_last_clip"].copy()
    env.uplink_rate_access_b = snap["uplink_rate_access_b"].copy()
    env.step_num = snap["step_num"]
    env._channel_ready = snap["_channel_ready"]
    env.theta_current = snap["theta_current"].copy() if snap["theta_current"] is not None else None
    env.cluster_matrix = snap["cluster_matrix"].copy() if snap["cluster_matrix"] is not None else None
    env.current_cluster_size = snap["current_cluster_size"].copy()
    env.cpu_front_quality = snap["cpu_front_quality"].copy()
    env._segment_delay_sum = snap["_segment_delay_sum"].copy()
    env._segment_uplink_sum = snap["_segment_uplink_sum"].copy()
    env._segment_front_sum = snap["_segment_front_sum"].copy()
    env._segment_front_log_sum = snap["_segment_front_log_sum"].copy()
    env._segment_delay_count = snap["_segment_delay_count"]
    env._segment_avg_delay = snap["_segment_avg_delay"].copy()
    env._segment_avg_uplink = snap["_segment_avg_uplink"].copy()
    env._segment_avg_front = snap["_segment_avg_front"].copy()
    env._segment_offload_count = snap["_segment_offload_count"].copy()
    env._segment_offload_success_count = snap["_segment_offload_success_count"].copy()
    env._segment_offload_success_ratio = snap["_segment_offload_success_ratio"].copy()
    env._segment_cpu_counts = snap["_segment_cpu_counts"].copy()
    env._segment_local_counts = snap["_segment_local_counts"].copy()
    env._interval_delays = list(snap["_interval_delays"])
    env._pending_high_action = snap["_pending_high_action"].copy() if snap["_pending_high_action"] is not None else None
    env._top10_ap_idx = snap["_top10_ap_idx"].copy() if snap["_top10_ap_idx"] is not None else None


def extract_interval_terms(env):
    count = max(1, int(env._segment_delay_count))
    avg_total_user = np.asarray(env._segment_delay_sum / count, dtype=np.float64)
    offload_den = np.maximum(np.asarray(env._segment_offload_count, dtype=np.float64), 1.0)
    avg_front_user = np.asarray(env._segment_front_sum, dtype=np.float64) / offload_den
    avg_front_log_user = np.asarray(env._segment_front_log_sum, dtype=np.float64) / offload_den
    reward_user = -(avg_total_user + float(env.lambda_front) * avg_front_log_user)
    return {
        "avg_total_user": avg_total_user,
        "avg_front_user": avg_front_user,
        "avg_front_log_user": avg_front_log_user,
        "reward_user": reward_user,
    }


def run_one_interval(env, high_action, interval, low_action, quiet_env_prints=True):
    env.apply_high_action(np.asarray(high_action, dtype=np.int32))
    for t in range(interval):
        if t > 0:
            env.advance_channel()
        if quiet_env_prints:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                env.step(low_action)
        else:
            env.step(low_action)

    reward_vec = np.asarray(env.compute_interval_reward(), dtype=np.float64).reshape(-1)
    terms = extract_interval_terms(env)

    front_raw_scaled = terms["avg_front_user"] / max(float(env.front_ref), 1e-12)
    front_log_recomputed = np.log1p(front_raw_scaled)
    front_log_compression = (
        float(np.std(front_log_recomputed) / (np.std(front_raw_scaled) + 1e-12))
        if front_raw_scaled.size > 1
        else 0.0
    )

    return {
        "reward_mean": float(np.mean(reward_vec)),
        "reward_std_user": float(np.std(reward_vec)),
        "avg_total_mean": float(np.mean(terms["avg_total_user"])),
        "avg_front_mean": float(np.mean(terms["avg_front_user"])),
        "avg_front_log_mean": float(np.mean(terms["avg_front_log_user"])),
        "front_component_mean": float(env.lambda_front * np.mean(terms["avg_front_log_user"])),
        "delay_component_mean": float(np.mean(terms["avg_total_user"])),
        "front_log_compression_user": front_log_compression,
    }


def build_action_set(rng, env, n_random, include_actions):
    action_dim = int(getattr(env, "high_action_dim", len(env._ap_combos)))
    m_sim = int(env.M_sim)
    pool = set()
    for a in include_actions:
        pool.add(tuple(int(x) for x in np.asarray(a).reshape(-1)[:m_sim]))
    while len(pool) < len(include_actions) + n_random:
        a = tuple(int(x) for x in rng.integers(0, action_dim, size=m_sim).tolist())
        pool.add(a)
    return [np.asarray(a, dtype=np.int32) for a in sorted(pool)]


def summarize_state_sweep(state_rows):
    rewards = np.asarray([x["reward_mean"] for x in state_rows], dtype=np.float64)
    totals = np.asarray([x["delay_component_mean"] for x in state_rows], dtype=np.float64)
    fronts = np.asarray([x["front_component_mean"] for x in state_rows], dtype=np.float64)
    front_raw = np.asarray([x["avg_front_mean"] for x in state_rows], dtype=np.float64)
    front_log = np.asarray([x["avg_front_log_mean"] for x in state_rows], dtype=np.float64)

    order = np.argsort(-rewards)
    best = float(rewards[order[0]])
    second = float(rewards[order[1]]) if rewards.size > 1 else best

    x = front_raw / max(1e-12, float(np.mean(front_raw) + 1e-12))
    y = np.log1p(x)
    compression = float(np.std(y) / (np.std(x) + 1e-12))

    return {
        "n_actions": int(rewards.size),
        "reward_best": best,
        "reward_second": second,
        "reward_top12_gap": float(best - second),
        "reward_spread_maxmin": float(np.max(rewards) - np.min(rewards)),
        "reward_spread_p90_p10": float(np.percentile(rewards, 90) - np.percentile(rewards, 10)),
        "reward_std_across_actions": float(np.std(rewards)),
        "total_std_across_actions": float(np.std(totals)),
        "front_std_across_actions": float(np.std(fronts)),
        "front_raw_std_across_actions": float(np.std(front_raw)),
        "front_log_std_across_actions": float(np.std(front_log)),
        "front_log_compression_across_actions": compression,
    }


def aggregate_phase(rows, key):
    if not rows:
        return float("nan")
    vals = [r[key] for r in rows]
    return float(np.mean(vals))


def main():
    parser = argparse.ArgumentParser(
        description="Validate high-level A/B hypotheses in obs57_heurlow in one run."
    )
    parser.add_argument("--seeds", type=str, default="0-2", help="seed range/list, e.g. 0-4 or 1,3,5")
    parser.add_argument("--episodes", type=int, default=1, help="episodes per seed")
    parser.add_argument("--episode-length", type=int, default=200, help="env steps per episode")
    parser.add_argument("--interval", type=int, default=10, help="high decision interval")
    parser.add_argument("--a-states-per-episode", type=int, default=6, help="how many interval states to probe for A")
    parser.add_argument("--a-random-actions", type=int, default=64, help="random high actions sampled per probed state")
    parser.add_argument(
        "--trajectory-policy",
        type=str,
        default="random",
        choices=["random", "fixed0"],
        help="policy for collecting trajectory states used by A/B",
    )
    parser.add_argument(
        "--top12-gap-threshold",
        type=float,
        default=0.02,
        help="state considered weakly separable if top1-top2 reward gap is below this",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/high_ab_validation_heurlow.json",
        help="output JSON path",
    )
    parser.add_argument(
        "--show-env-prints",
        action="store_true",
        default=False,
        help="show raw environment debug prints during validation",
    )
    args = parser.parse_args()

    seeds = parse_seed_spec(args.seeds)
    rng = np.random.default_rng(20260401)

    state_summaries = []
    trajectory_rows = []

    quiet_env_prints = not args.show_env_prints

    for seed in seeds:
        for ep in range(args.episodes):
            env_seed = seed * 1000 + ep
            env = make_env(env_seed)
            env.reset()

            total_intervals = args.episode_length // args.interval
            probe_set = evenly_spaced_indices(total_intervals, args.a_states_per_episode)
            low_action = make_dummy_low_action(env)
            action_dim = int(getattr(env, "high_action_dim", len(env._ap_combos)))

            for interval_idx in range(total_intervals):
                env.advance_channel()
                snap = snapshot_env(env)
                phase = phase_name(interval_idx, total_intervals)

                if args.trajectory_policy == "fixed0":
                    traj_action = np.zeros((env.M_sim,), dtype=np.int32)
                else:
                    traj_action = rng.integers(0, action_dim, size=env.M_sim, dtype=np.int32)

                if interval_idx in probe_set:
                    candidates = build_action_set(
                        rng=rng,
                        env=env,
                        n_random=args.a_random_actions,
                        include_actions=[traj_action, np.zeros((env.M_sim,), dtype=np.int32)],
                    )
                    rows = []
                    for cand in candidates:
                        restore_env(env, snap)
                        res = run_one_interval(
                            env=env,
                            high_action=cand,
                            interval=args.interval,
                            low_action=low_action,
                            quiet_env_prints=quiet_env_prints,
                        )
                        res["action"] = cand.tolist()
                        rows.append(res)

                    state_sum = summarize_state_sweep(rows)
                    state_sum.update(
                        {
                            "seed": int(seed),
                            "episode": int(ep),
                            "interval_idx": int(interval_idx),
                            "phase": phase,
                            "trajectory_policy": args.trajectory_policy,
                        }
                    )
                    state_summaries.append(state_sum)

                restore_env(env, snap)
                traj_res = run_one_interval(
                    env=env,
                    high_action=traj_action,
                    interval=args.interval,
                    low_action=low_action,
                    quiet_env_prints=quiet_env_prints,
                )
                traj_res.update(
                    {
                        "seed": int(seed),
                        "episode": int(ep),
                        "interval_idx": int(interval_idx),
                        "phase": phase,
                    }
                )
                trajectory_rows.append(traj_res)

            if hasattr(env, "close"):
                env.close()

    phase_names = ["early", "mid", "late"]
    state_by_phase = {p: [r for r in state_summaries if r["phase"] == p] for p in phase_names}
    traj_by_phase = {p: [r for r in trajectory_rows if r["phase"] == p] for p in phase_names}

    weak_states = [
        r for r in state_summaries if r["reward_top12_gap"] < float(args.top12_gap_threshold)
    ]

    summary = {
        "config": {
            "seeds": seeds,
            "episodes": int(args.episodes),
            "episode_length": int(args.episode_length),
            "interval": int(args.interval),
            "a_states_per_episode": int(args.a_states_per_episode),
            "a_random_actions": int(args.a_random_actions),
            "trajectory_policy": args.trajectory_policy,
            "top12_gap_threshold": float(args.top12_gap_threshold),
        },
        "A_action_separability": {
            "n_probed_states": int(len(state_summaries)),
            "weak_state_ratio_top12_gap": (
                float(len(weak_states) / max(1, len(state_summaries)))
            ),
            "overall_mean_reward_spread_maxmin": aggregate_phase(state_summaries, "reward_spread_maxmin"),
            "overall_mean_reward_spread_p90_p10": aggregate_phase(state_summaries, "reward_spread_p90_p10"),
            "overall_mean_top12_gap": aggregate_phase(state_summaries, "reward_top12_gap"),
            "phase": {
                p: {
                    "n_states": int(len(state_by_phase[p])),
                    "mean_reward_spread_maxmin": aggregate_phase(state_by_phase[p], "reward_spread_maxmin"),
                    "mean_reward_spread_p90_p10": aggregate_phase(state_by_phase[p], "reward_spread_p90_p10"),
                    "mean_top12_gap": aggregate_phase(state_by_phase[p], "reward_top12_gap"),
                    "mean_reward_std_across_actions": aggregate_phase(state_by_phase[p], "reward_std_across_actions"),
                }
                for p in phase_names
            },
        },
        "B_reward_shaping": {
            "state_sweep_decomposition": {
                "overall_mean_total_std_across_actions": aggregate_phase(state_summaries, "total_std_across_actions"),
                "overall_mean_front_std_across_actions": aggregate_phase(state_summaries, "front_std_across_actions"),
                "overall_mean_front_log_compression_across_actions": aggregate_phase(
                    state_summaries, "front_log_compression_across_actions"
                ),
                "phase": {
                    p: {
                        "mean_total_std_across_actions": aggregate_phase(state_by_phase[p], "total_std_across_actions"),
                        "mean_front_std_across_actions": aggregate_phase(state_by_phase[p], "front_std_across_actions"),
                        "mean_front_log_compression_across_actions": aggregate_phase(
                            state_by_phase[p], "front_log_compression_across_actions"
                        ),
                    }
                    for p in phase_names
                },
            },
            "trajectory_decomposition": {
                "overall_mean_delay_component": aggregate_phase(trajectory_rows, "delay_component_mean"),
                "overall_mean_front_component": aggregate_phase(trajectory_rows, "front_component_mean"),
                "overall_mean_reward": aggregate_phase(trajectory_rows, "reward_mean"),
                "phase": {
                    p: {
                        "n_intervals": int(len(traj_by_phase[p])),
                        "mean_delay_component": aggregate_phase(traj_by_phase[p], "delay_component_mean"),
                        "mean_front_component": aggregate_phase(traj_by_phase[p], "front_component_mean"),
                        "mean_reward": aggregate_phase(traj_by_phase[p], "reward_mean"),
                        "mean_front_log_compression_user": aggregate_phase(
                            traj_by_phase[p], "front_log_compression_user"
                        ),
                    }
                    for p in phase_names
                },
            },
        },
        "raw": {
            "state_summaries": state_summaries,
            "trajectory_rows": trajectory_rows,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== A: Action Separability ===")
    print("probed states:", summary["A_action_separability"]["n_probed_states"])
    print(
        "weak_state_ratio(top1-top2 < {:.4f}): {:.3f}".format(
            args.top12_gap_threshold, summary["A_action_separability"]["weak_state_ratio_top12_gap"]
        )
    )
    print(
        "mean spread max-min: {:.6f}".format(
            summary["A_action_separability"]["overall_mean_reward_spread_maxmin"]
        )
    )
    print(
        "mean spread p90-p10: {:.6f}".format(
            summary["A_action_separability"]["overall_mean_reward_spread_p90_p10"]
        )
    )

    print("\n=== B: Reward Shaping / Credit ===")
    print(
        "overall delay component mean: {:.6f}".format(
            summary["B_reward_shaping"]["trajectory_decomposition"]["overall_mean_delay_component"]
        )
    )
    print(
        "overall front component mean: {:.6f}".format(
            summary["B_reward_shaping"]["trajectory_decomposition"]["overall_mean_front_component"]
        )
    )
    print(
        "overall front-log compression (state-sweep): {:.6f}".format(
            summary["B_reward_shaping"]["state_sweep_decomposition"][
                "overall_mean_front_log_compression_across_actions"
            ]
        )
    )
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
