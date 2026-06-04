import csv
import contextlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from algorithms.algorithm.high_actor_critic import HighActor
from algorithms.algorithm.r_actor_critic import R_Actor
from config import get_config
with contextlib.redirect_stderr(io.StringIO()):
    from envs.MA_UCMEC_dyna_noncoop_hierarchical_peruser_hotspot_nlos_obs57 import (
        MA_UCMEC_dyna_noncoop_hierarchical_peruser,
    )
from model_configs import MODEL_CONFIGS


SEEDS = [
    18, 62, 53, 14, 58,
    161, 37, 4, 95, 150,
    11, 1, 29, 189, 198,
    153, 26, 59, 365, 84,
    946, 56, 99, 75, 263,
    776, 94, 71, 735, 64,
]
EPISODES_PER_SEED = 1
HIERARCHICAL_INTERVAL = 10
EXPECTED_CANDIDATE_N = 10
EXPECTED_K_FIXED = 2
OUTPUT_DIR = Path("ap_selection_outputs")
MAIN_SWEEP_CSV = Path("sweep_outputs/20260518_210501/eval_sweep_results.csv")

METHOD_NAMES = ("Access-Greedy", "Fronthaul-Greedy", "Proposed HDRL")
MAIN_TOTAL_DELAY_METHODS = {
    "Proposed_HDRL": "Proposed HDRL",
    "AccessGreedy": "Access-Greedy",
    "FrontGreedyTopL": "Fronthaul-Greedy",
}


def _proposed_configs():
    configs = [cfg for cfg in MODEL_CONFIGS if cfg["method"] == "Proposed_HDRL"]
    if not configs:
        raise RuntimeError("No Proposed_HDRL configs found in model_configs.py")
    return configs


def _make_args():
    parser = get_config()
    args = parser.parse_args([])
    args.use_recurrent_policy = True
    args.use_naive_recurrent_policy = False
    high_args = parser.parse_args([])
    high_args.use_recurrent_policy = True
    high_args.use_naive_recurrent_policy = False
    high_args.hidden_size = 128
    return args, high_args


def _configure_high_args_from_checkpoint(high_args, env, state_dict):
    keys = list(state_dict.keys())
    has_pair_scorer = any(
        k.startswith("ap_mlp.")
        or k.startswith("scorer.")
        or k.startswith("g_pre.")
        or k.startswith("_ap_indices")
        for k in keys
    )
    high_args.high_actor_type = "pair_scorer" if has_pair_scorer else "mlp"
    high_args.use_recurrent_policy = any(k.startswith("rnn.") for k in keys)
    high_args.use_naive_recurrent_policy = False

    if high_args.high_actor_type == "pair_scorer":
        high_args.high_pair_repr = "concat" if any(k.startswith("concat_proj.") for k in keys) else "sdp"
        has_global_ctx = any(k.startswith("g_pre.") or k.startswith("g_proj.") for k in keys)
        high_args.high_no_global_ctx = not has_global_ctx

    if "rnn.rnn.weight_hh_l0" in state_dict:
        high_args.hidden_size = int(state_dict["rnn.rnn.weight_hh_l0"].shape[1])
    elif "g_pre.weight" in state_dict:
        high_args.hidden_size = int(state_dict["g_pre.weight"].shape[0])
    elif "g_proj.weight" in state_dict:
        high_args.hidden_size = int(state_dict["g_proj.weight"].shape[1])
    elif "logits.weight" in state_dict:
        high_args.hidden_size = int(state_dict["logits.weight"].shape[1])

    high_args.candidate_n = int(env.candidate_n)
    high_args.k_fixed = int(env.k_fixed)
    high_args.num_cpus = int(env.K)
    return high_args


def _load_actors(env, cfg, device):
    args, high_args = _make_args()
    low_path = Path(cfg["env"]["EVAL_MODEL_LOW"])
    high_path = Path(cfg["env"]["EVAL_MODEL_HIGH"])
    if not low_path.exists():
        raise FileNotFoundError(low_path)
    if not high_path.exists():
        raise FileNotFoundError(high_path)

    low_actor = R_Actor(args, env.observation_space[0], env.action_space[0], device)
    low_actor.load_state_dict(torch.load(low_path, map_location=device))
    low_actor.eval()

    high_state = torch.load(high_path, map_location=device)
    high_args = _configure_high_args_from_checkpoint(high_args, env, high_state)
    high_actor = HighActor(high_args, env.high_observation_space, env.high_action_space, device)
    high_actor.load_state_dict(high_state)
    high_actor.eval()
    return args, high_args, low_actor, high_actor


def _front_bottleneck_quality(env, user_idx, combo_idx):
    combo = env._ap_combos[int(combo_idx)]
    ap_idx = env._top10_ap_idx[user_idx][list(combo)]
    lo, hi = env.front_db_clip
    best_quality = -np.inf
    for cpu in range(env.K):
        pl_values = []
        for ap in ap_idx:
            dist = max(float(env.distance_matrix_front[ap, cpu]), 1e-6)
            alpha = env.alpha_los if env.link_type[ap, cpu] == 0 else env.alpha_nlos
            gain = max(float(env.G[ap, cpu]), 1e-12)
            pl_values.append(gain * pow(dist, -alpha))
        min_pl_db = 10.0 * math.log10(min(pl_values) + 1e-12)
        quality = float(np.clip((min_pl_db - lo) / (hi - lo), 0.0, 1.0))
        best_quality = max(best_quality, quality)
    return best_quality


def _best_front_actions(env):
    actions = np.zeros(env.M_sim, dtype=np.int32)
    for user_idx in range(env.M_sim):
        best_score = -np.inf
        best_combo = 0
        for combo_idx in range(len(env._ap_combos)):
            score = _front_bottleneck_quality(env, user_idx, combo_idx)
            if score > best_score:
                best_score = score
                best_combo = combo_idx
        actions[user_idx] = best_combo
    return actions


def _init_stats():
    return {
        method: {
            "count": 0,
            "same_pair": 0,
            "one_ap_overlap": 0,
            "access_rank_sum": 0.0,
            "front_quality_sum": 0.0,
        }
        for method in METHOD_NAMES
    }


def _record_actions(stats, rank_hist, env, method, actions, access_actions):
    for user_idx, action in enumerate(np.asarray(actions, dtype=np.int32).reshape(-1)):
        action = int(action)
        access_action = int(access_actions[user_idx])
        combo = tuple(env._ap_combos[action])
        access_combo = tuple(env._ap_combos[access_action])
        overlap = len(set(combo).intersection(access_combo))
        stats[method]["count"] += 1
        stats[method]["same_pair"] += int(set(combo) == set(access_combo))
        stats[method]["one_ap_overlap"] += int(overlap == 1)
        stats[method]["access_rank_sum"] += float(np.mean(combo))
        stats[method]["front_quality_sum"] += _front_bottleneck_quality(env, user_idx, action)
        if method == "Proposed HDRL":
            rank_hist[combo] += 1


def _low_action_env(actor, act_space, obs, rnn_states, masks):
    obs_batch = np.stack(obs)
    with torch.no_grad():
        actions, _, rnn_states = actor(obs_batch, rnn_states, masks, deterministic=True)
    if act_space.__class__.__name__ == "MultiDiscrete":
        actions_env = actions.cpu().numpy().astype(np.int32)
    else:
        action_indices = actions.cpu().numpy().flatten()
        actions_env = np.eye(act_space.n)[action_indices]
    return actions_env, rnn_states


def _high_rnn_initial_state(env, high_args):
    if getattr(high_args, "high_actor_type", "mlp") == "pair_scorer":
        return np.zeros((1, env.M_sim, high_args.recurrent_N, high_args.hidden_size), dtype=np.float32)
    return np.zeros((1, high_args.recurrent_N, high_args.hidden_size), dtype=np.float32)


def analyze_run(cfg, device):
    env = MA_UCMEC_dyna_noncoop_hierarchical_peruser(render=False, seed=0)
    if env.candidate_n != EXPECTED_CANDIDATE_N or env.k_fixed != EXPECTED_K_FIXED:
        raise RuntimeError(
            f"Expected candidate_n={EXPECTED_CANDIDATE_N}, k_fixed={EXPECTED_K_FIXED}, "
            f"but env has candidate_n={env.candidate_n}, k_fixed={env.k_fixed}. "
            "Please restore the default obs57 environment before running this analysis."
        )

    args, high_args, low_actor, high_actor = _load_actors(env, cfg, device)
    act_space = env.action_space[0]
    stats = _init_stats()
    rank_hist = Counter()

    for seed in SEEDS:
        for _ in range(EPISODES_PER_SEED):
            env.seed(seed)
            obs = env.reset()
            dones = [False] * env.n_agents
            step_count = 0
            rnn_states = np.zeros((env.M_sim, args.recurrent_N, args.hidden_size), dtype=np.float32)
            masks = np.ones((env.M_sim, 1), dtype=np.float32)
            high_rnn_states = _high_rnn_initial_state(env, high_args)
            high_masks = np.ones((1, 1), dtype=np.float32)

            while not all(dones):
                env.advance_channel()
                if step_count % HIERARCHICAL_INTERVAL == 0:
                    global_obs = np.expand_dims(env.get_global_obs(), axis=0)
                    with torch.no_grad():
                        proposed_action, _, high_rnn_states = high_actor(
                            global_obs, high_rnn_states, high_masks, deterministic=True
                        )
                    proposed_action = proposed_action.cpu().numpy().astype(np.int32).squeeze(0).reshape(-1)
                    proposed_action = np.clip(proposed_action, 0, len(env._ap_combos) - 1)
                    access_actions = np.zeros(env.M_sim, dtype=np.int32)
                    front_actions = _best_front_actions(env)

                    _record_actions(stats, rank_hist, env, "Access-Greedy", access_actions, access_actions)
                    _record_actions(stats, rank_hist, env, "Fronthaul-Greedy", front_actions, access_actions)
                    _record_actions(stats, rank_hist, env, "Proposed HDRL", proposed_action, access_actions)
                    env.apply_high_action(proposed_action)

                actions_env, rnn_states = _low_action_env(low_actor, act_space, obs, rnn_states, masks)
                with contextlib.redirect_stdout(io.StringIO()):
                    obs, _, dones, _ = env.step(actions_env)
                masks = np.ones((env.M_sim, 1), dtype=np.float32)
                masks[np.asarray(dones, dtype=bool)] = 0.0
                step_count += 1

    return stats, rank_hist


def _summarize_stats(stats):
    summary = {}
    for method, values in stats.items():
        count = max(1, values["count"])
        summary[method] = {
            "samples": values["count"],
            "same_pair_rate": values["same_pair"] / count,
            "one_ap_overlap_rate": values["one_ap_overlap"] / count,
            "avg_selected_access_rank": values["access_rank_sum"] / count,
            "avg_front_bottleneck_quality_norm": values["front_quality_sum"] / count,
        }
    return summary


def _merge_run_summaries(run_summaries):
    methods = METHOD_NAMES
    metric_names = [
        "same_pair_rate",
        "one_ap_overlap_rate",
        "avg_selected_access_rank",
        "avg_front_bottleneck_quality_norm",
    ]
    merged = {}
    for method in methods:
        merged[method] = {}
        for metric in metric_names:
            vals = np.asarray([run[method][metric] for run in run_summaries], dtype=np.float64)
            merged[method][metric] = {
                "mean": float(vals.mean()),
                "se": float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0,
            }
        merged[method]["samples"] = int(sum(run[method]["samples"] for run in run_summaries))
    return merged


def _main_total_delay():
    totals = defaultdict(list)
    if not MAIN_SWEEP_CSV.exists():
        return {}
    with MAIN_SWEEP_CSV.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = MAIN_TOTAL_DELAY_METHODS.get(row.get("method", ""))
            if method is None:
                continue
            if row.get("sweep") != "M_sim":
                continue
            if str(row.get("value")) != "10":
                continue
            if int(float(row.get("m_sim", 0))) != 10:
                continue
            if int(float(row.get("n_sim", 0))) != 50:
                continue
            if abs(float(row.get("epsilon", 0.0)) - 3e-3) > 1e-12:
                continue
            totals[method].append(float(row["avg_total_delay_ms_mean"]))
    out = {}
    for method, vals in totals.items():
        arr = np.asarray(vals, dtype=np.float64)
        out[method] = {
            "mean": float(arr.mean()),
            "se": float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0,
        }
    return out


def _write_outputs(run_outputs, merged, rank_hist, total_delay):
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary_path = OUTPUT_DIR / "ap_selection_summary.json"
    table_path = OUTPUT_DIR / "ap_selection_quality_table.csv"
    hist_path = OUTPUT_DIR / "proposed_rank_pair_distribution.csv"

    payload = {
        "settings": {
            "seeds": SEEDS,
            "episodes_per_seed": EPISODES_PER_SEED,
            "hierarchical_interval": HIERARCHICAL_INTERVAL,
            "candidate_n": EXPECTED_CANDIDATE_N,
            "k_fixed": EXPECTED_K_FIXED,
            "front_quality": "max_cpu min_selected_ap normalized AP-CPU fronthaul pathloss quality",
            "unit": "one sample per user per high-level decision interval",
        },
        "runs": run_outputs,
        "merged": merged,
        "main_total_delay_ms": total_delay,
    }
    with summary_path.open("w") as f:
        json.dump(payload, f, indent=2)

    with table_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", *METHOD_NAMES])
        for metric in [
            "same_pair_rate",
            "one_ap_overlap_rate",
            "avg_selected_access_rank",
            "avg_front_bottleneck_quality_norm",
        ]:
            writer.writerow(
                [metric]
                + [
                    f"{merged[method][metric]['mean']:.6f}"
                    for method in METHOD_NAMES
                ]
            )
        writer.writerow(
            ["avg_total_delay_ms"]
            + [
                f"{total_delay.get(method, {}).get('mean', float('nan')):.6f}"
                for method in METHOD_NAMES
            ]
        )

    total_hist = sum(rank_hist.values())
    with hist_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank_pair", "count", "frequency"])
        for combo, count in rank_hist.most_common():
            writer.writerow([str(combo), count, count / max(1, total_hist)])

    return summary_path, table_path, hist_path


def _print_summary(merged, total_delay, rank_hist):
    print("\nAP selection quality comparison")
    print("Metric                               Access-Greedy   Fronthaul-Greedy   Proposed HDRL")
    print("-" * 86)
    labels = [
        ("same_pair_rate", "Same pair rate"),
        ("one_ap_overlap_rate", "One-AP overlap rate"),
        ("avg_selected_access_rank", "Avg. selected access rank"),
        ("avg_front_bottleneck_quality_norm", "Avg. front bottleneck quality"),
    ]
    for key, label in labels:
        vals = [merged[method][key]["mean"] for method in METHOD_NAMES]
        print(f"{label:<36}{vals[0]:>10.4f}{vals[1]:>19.4f}{vals[2]:>16.4f}")
    vals = [total_delay.get(method, {}).get("mean", float("nan")) for method in METHOD_NAMES]
    print(f"{'Avg. total delay (ms)':<36}{vals[0]:>10.2f}{vals[1]:>19.2f}{vals[2]:>16.2f}")

    print("\nTop Proposed HDRL rank pairs")
    total = sum(rank_hist.values())
    for combo, count in rank_hist.most_common(10):
        print(f"  {combo}: {count / max(1, total):.4f} ({count}/{total})")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_outputs = []
    total_rank_hist = Counter()

    for cfg in _proposed_configs():
        print(f"Running {cfg['name']}...")
        stats, rank_hist = analyze_run(cfg, device)
        summary = _summarize_stats(stats)
        run_outputs.append({"name": cfg["name"], "run": cfg["run"], "summary": summary})
        total_rank_hist.update(rank_hist)
        print(f"  done: {cfg['name']}")

    merged = _merge_run_summaries([r["summary"] for r in run_outputs])
    total_delay = _main_total_delay()
    summary_path, table_path, hist_path = _write_outputs(run_outputs, merged, total_rank_hist, total_delay)
    _print_summary(merged, total_delay, total_rank_hist)
    print("\nSaved files:")
    print(f"  {summary_path}")
    print(f"  {table_path}")
    print(f"  {hist_path}")


if __name__ == "__main__":
    main()
