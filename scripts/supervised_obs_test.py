"""
Supervised learnability test for high-level AP-combo policy.

Goal:
  Test whether current high-level observation contains enough signal to
  predict strong AP-combo actions (oracle labels), independent of RL noise.

Key design:
  1) Oracle label horizon is configurable (default: interval horizon).
  2) Data collection behavior policy is configurable (baseline/random/oracle/mixed).
  3) Final metric uses closed-loop rollout (MLP vs baseline vs oracle),
     i.e., each policy drives its own trajectory.

Usage:
  python scripts/supervised_obs_test.py --n_seeds 10 --n_steps 500 --interval 10
"""

import argparse
import contextlib
import copy
import os
import sys
import time

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from envs.MA_UCMEC_dyna_noncoop_hierarchical_peruser_hotspot_nlos_obs57_heurlow import (
    MA_UCMEC_dyna_noncoop_hierarchical_peruser as HeurEnv,
)


def _snapshot(env):
    """Capture env state for rollback."""
    snap = {
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
        "last_cluster_size": env.last_cluster_size.copy(),
        "last_selected_ap_mask": env.last_selected_ap_mask.copy(),
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
    return snap


def _restore(env, snap):
    """Restore env to snapshot."""
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
    env.last_cluster_size = snap["last_cluster_size"].copy()
    env.last_selected_ap_mask = snap["last_selected_ap_mask"].copy()
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


def _dummy_actions(env):
    """Return dummy one-hot actions (heuristic low-level overrides them)."""
    acts = []
    for _ in range(env.M_sim):
        a = np.zeros(env.action_dim, dtype=np.float32)
        a[0] = 1.0
        acts.append(a)
    return acts


def _rollout_interval_delay(env, combo_action, interval):
    """Apply one high-level action and simulate one interval.

    Assumption:
      channel for t=0 has already been advanced by caller.
    """
    env.apply_high_action(combo_action.astype(np.int32))
    delay_sum = np.zeros(env.M_sim, dtype=np.float64)
    for t in range(interval):
        if t > 0:
            env.advance_channel()
        env.step_low(_dummy_actions(env))
        delay_sum += env.delay_last_clip[:env.M_sim, 0]
    return delay_sum / max(1, interval)


def _sample_behavior_action(env, oracle_combo, behavior_policy):
    """Behavior policy used to move environment during data collection."""
    n_combos = len(env._ap_combos)
    if behavior_policy == "baseline":
        return np.zeros(env.M_sim, dtype=np.int32)
    if behavior_policy == "random":
        return env.rng.integers(0, n_combos, size=env.M_sim, dtype=np.int32)
    if behavior_policy == "oracle":
        return oracle_combo.copy().astype(np.int32)
    if behavior_policy == "mixed":
        # Episode-level mixture: baseline / random / oracle (1/3 each).
        u = float(env.rng.random())
        if u < (1.0 / 3.0):
            return np.zeros(env.M_sim, dtype=np.int32)
        if u < (2.0 / 3.0):
            return env.rng.integers(0, n_combos, size=env.M_sim, dtype=np.int32)
        return oracle_combo.copy().astype(np.int32)
    raise ValueError(f"Unknown behavior_policy={behavior_policy}")


def oracle_search(env, snap, interval=1, mode="greedy"):
    """Per-user oracle with configurable horizon/objective.

    mode='greedy':
      greedy sequential (user 0..M-1), same family as eval_script oracle.

    mode='indep':
      each user optimized independently with others fixed at baseline combo 0.
    """
    if mode not in {"greedy", "indep"}:
        raise ValueError(f"Unknown oracle mode: {mode}")

    n_combos = len(env._ap_combos)
    M = env.M_sim
    current_combo = np.zeros(M, dtype=np.int32)
    per_user_scores = np.zeros((M, n_combos), dtype=np.float64)

    for user_i in range(M):
        best_score = float("inf")
        best_combo_idx = current_combo[user_i]

        for combo_idx in range(n_combos):
            trial = np.zeros(M, dtype=np.int32) if mode == "indep" else current_combo.copy()
            trial[user_i] = combo_idx

            _restore(env, snap)
            avg_delay = _rollout_interval_delay(env, trial, interval=interval)

            score = float(np.mean(avg_delay))
            per_user_scores[user_i, combo_idx] = score

            if score < best_score:
                best_score = score
                best_combo_idx = combo_idx

        current_combo[user_i] = best_combo_idx

    _restore(env, snap)
    return current_combo, per_user_scores


def collect_data(n_seeds=10, n_steps=500, interval=10,
                 behavior_policy="mixed", oracle_horizon=None,
                 oracle_mode="greedy"):
    """Collect (high_obs, oracle_best_combo) pairs across seeds and intervals.

    Returns:
      obs, labels, delays, seed_ids
    """
    if oracle_horizon is None:
        oracle_horizon = interval

    all_obs = []
    all_labels = []
    all_delays_list = []
    all_seed_ids = []

    for seed_i, seed in enumerate(range(n_seeds)):
        print(f"\n[Seed {seed_i + 1}/{n_seeds}] seed={seed}")
        env = HeurEnv(seed=seed)
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            env.reset()

        n_intervals = n_steps // interval
        for iv in range(n_intervals):
            env.advance_channel()

            high_obs = env.get_global_obs()
            snap = _snapshot(env)

            best_combos, combo_delays = oracle_search(
                env,
                snap,
                interval=oracle_horizon,
                mode=oracle_mode,
            )

            all_obs.append(high_obs)
            all_labels.append(best_combos)
            all_delays_list.append(combo_delays)
            all_seed_ids.append(np.full(env.M_sim, seed, dtype=np.int32))

            _restore(env, snap)
            behavior_action = _sample_behavior_action(env, best_combos, behavior_policy)
            _rollout_interval_delay(env, behavior_action, interval=interval)

            if (iv + 1) % 10 == 0:
                labels_so_far = np.concatenate(all_labels)
                n_unique = len(np.unique(labels_so_far))
                n_combos_env = len(env._ap_combos)
                top1_combo = np.bincount(labels_so_far, minlength=n_combos_env).argmax()
                top1_frac = np.mean(labels_so_far == top1_combo)
                baseline_frac = np.mean(labels_so_far == 0)
                print(
                    f"  interval {iv + 1}/{n_intervals}  "
                    f"unique_labels={n_unique}  "
                    f"most_common=combo{top1_combo}({top1_frac:.1%})  "
                    f"baseline_optimal={baseline_frac:.1%}"
                )

    obs_arr = np.concatenate(all_obs, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    delays_arr = np.concatenate(all_delays_list, axis=0)
    seed_ids_arr = np.concatenate(all_seed_ids, axis=0)

    tmp_env = HeurEnv(seed=0)
    print("\n" + "=" * 60)
    print(
        f"Collected {obs_arr.shape[0]} samples "
        f"(obs_dim={obs_arr.shape[1]}, n_classes={len(tmp_env._ap_combos)})"
    )
    return obs_arr, labels_arr, delays_arr, seed_ids_arr


def _eval_policy_closed_loop(policy_name, model, seeds, n_steps, interval,
                             oracle_horizon, oracle_mode):
    """Closed-loop rollout mean delay (ms) for one policy."""
    import torch

    delays = []
    for seed in seeds:
        env = HeurEnv(seed=seed)
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            env.reset()

        n_intervals = n_steps // interval
        for _ in range(n_intervals):
            env.advance_channel()
            high_obs = env.get_global_obs()

            if policy_name == "baseline":
                action = np.zeros(env.M_sim, dtype=np.int32)
            elif policy_name == "mlp":
                with torch.no_grad():
                    logits = model(torch.FloatTensor(high_obs))
                    action = logits.argmax(dim=1).numpy().astype(np.int32)
            elif policy_name == "oracle":
                snap = _snapshot(env)
                action, _ = oracle_search(
                    env,
                    snap,
                    interval=oracle_horizon,
                    mode=oracle_mode,
                )
            else:
                raise ValueError(f"Unknown policy_name={policy_name}")

            avg_delay = _rollout_interval_delay(env, action, interval=interval)
            delays.append(avg_delay.copy())

    return float(np.mean(np.concatenate(delays)) * 1000.0)


def joint_rollout_eval(model, test_seeds, n_steps, interval,
                       oracle_horizon, oracle_mode):
    """Closed-loop comparison: baseline vs MLP vs oracle."""
    model.eval()

    bl = _eval_policy_closed_loop(
        "baseline", model, test_seeds, n_steps, interval, oracle_horizon, oracle_mode
    )
    ml = _eval_policy_closed_loop(
        "mlp", model, test_seeds, n_steps, interval, oracle_horizon, oracle_mode
    )
    ol = _eval_policy_closed_loop(
        "oracle", model, test_seeds, n_steps, interval, oracle_horizon, oracle_mode
    )

    gap_closed = 1.0 - (ml - ol) / max(bl - ol, 1e-12)

    print(f"\n  Closed-loop delay (ms) on test seeds {list(test_seeds)}:")
    print(f"    baseline:    {bl:.1f}")
    print(f"    MLP:         {ml:.1f}")
    print(f"    oracle:      {ol:.1f}")
    print(f"    gap closed:  {gap_closed:.1%}")

    return bl, ml, ol


def train_and_eval(obs, labels, delays, seed_ids, n_classes=None,
                   n_steps=500, interval=10,
                   oracle_horizon=None, oracle_mode="greedy"):
    """Train MLP, evaluate accuracy (per-seed split), then closed-loop rollout."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    if oracle_horizon is None:
        oracle_horizon = interval

    obs_dim = obs.shape[1]
    if n_classes is None:
        tmp = HeurEnv(seed=0)
        n_classes = len(tmp._ap_combos)

    unique_seeds = np.array(sorted(set(seed_ids.tolist())), dtype=np.int32)
    rng = np.random.default_rng(0)
    rng.shuffle(unique_seeds)

    n_train = max(1, int(0.8 * len(unique_seeds)))
    train_seeds = set(unique_seeds[:n_train].tolist())
    test_seeds = set(unique_seeds[n_train:].tolist())
    if len(test_seeds) == 0:
        test_seeds = {int(unique_seeds[-1])}
        train_seeds = set(unique_seeds[:-1].tolist())

    train_mask = np.isin(seed_ids, list(train_seeds))
    test_mask = np.isin(seed_ids, list(test_seeds))

    X_train = torch.FloatTensor(obs[train_mask])
    y_train = torch.LongTensor(labels[train_mask])
    X_test = torch.FloatTensor(obs[test_mask])
    y_test = torch.LongTensor(labels[test_mask])
    delays_test = delays[test_mask]

    print(f"\nSplit by seed: train_seeds={sorted(train_seeds)}, test_seeds={sorted(test_seeds)}")
    print(f"  train={int(train_mask.sum())}, test={int(test_mask.sum())}")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)

    model = nn.Sequential(
        nn.Linear(obs_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, n_classes),
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining MLP: {obs_dim} -> 256 -> 256 -> {n_classes}")

    best_test_acc = 0.0
    for epoch in range(200):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item() * xb.size(0))

        if (epoch + 1) % 20 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                logits_test = model(X_test)
                probs = torch.softmax(logits_test, dim=1).numpy()
                preds = probs.argmax(axis=1)
                y_np = y_test.numpy()

                top1 = float(np.mean(preds == y_np))
                top3_idx = np.argsort(probs, axis=1)[:, -3:]
                top3 = float(np.mean([y_np[i] in top3_idx[i] for i in range(len(y_np))]))
                top5_idx = np.argsort(probs, axis=1)[:, -5:]
                top5 = float(np.mean([y_np[i] in top5_idx[i] for i in range(len(y_np))]))
                best_test_acc = max(best_test_acc, top1)

            print(
                f"  epoch {epoch + 1:3d}  "
                f"loss={total_loss / max(1, int(train_mask.sum())):.4f}  "
                f"top1={top1:.1%}  top3={top3:.1%}  top5={top5:.1%}"
            )

    model.eval()
    with torch.no_grad():
        logits_test = model(X_test)
        probs = torch.softmax(logits_test, dim=1).numpy()
        preds = probs.argmax(axis=1)

    y_np = y_test.numpy()
    top1 = float(np.mean(preds == y_np))
    top3_idx = np.argsort(probs, axis=1)[:, -3:]
    top3 = float(np.mean([y_np[i] in top3_idx[i] for i in range(len(y_np))]))
    top5_idx = np.argsort(probs, axis=1)[:, -5:]
    top5 = float(np.mean([y_np[i] in top5_idx[i] for i in range(len(y_np))]))

    print("\n" + "=" * 60)
    print(f"ACCURACY RESULTS (test set, n={len(y_np)}, split by seed)")
    print("=" * 60)
    print(f"  Random baseline:    {1.0 / n_classes:.1%}")
    print(f"  MLP top-1:          {top1:.1%}")
    print(f"  MLP top-3:          {top3:.1%}")
    print(f"  MLP top-5:          {top5:.1%}")
    print(f"  Best test accuracy: {best_test_acc:.1%}")

    print("\nLabel distribution (full dataset):")
    combo_counts = np.bincount(labels, minlength=n_classes)
    combo_frac = combo_counts / max(1, combo_counts.sum())
    top_combos = np.argsort(combo_counts)[::-1][:10]
    tmp_env = HeurEnv(seed=0)
    for rank, c in enumerate(top_combos):
        aps = tmp_env._ap_combos[c]
        print(f"  #{rank + 1}: combo {c} (cand {aps[0]},{aps[1]}) = {combo_frac[c]:.1%}")
    print(f"  Baseline (combo 0, cand 0,1) is optimal: {combo_frac[0]:.1%}")

    # Reference only: independent delay lookup on label table.
    mlp_delay_indep = np.array([delays_test[i, preds[i]] for i in range(len(preds))])
    oracle_delay_indep = np.array([delays_test[i, y_np[i]] for i in range(len(y_np))])
    baseline_delay_indep = delays_test[:, 0]
    print("\n  Independent delay lookup (reference, not closed-loop):")
    print(f"    baseline: {np.mean(baseline_delay_indep) * 1000:.1f} ms")
    print(f"    MLP:      {np.mean(mlp_delay_indep) * 1000:.1f} ms")
    print(f"    oracle:   {np.mean(oracle_delay_indep) * 1000:.1f} ms")

    wrong_mask = preds != y_np
    if int(wrong_mask.sum()) > 0:
        wrong_gap = mlp_delay_indep[wrong_mask] - oracle_delay_indep[wrong_mask]
        near_optimal = (
            np.abs(wrong_gap)
            / (np.abs(oracle_delay_indep[wrong_mask]) + 1e-12)
            < 0.05
        )
        print(f"\n  When MLP is wrong ({int(wrong_mask.sum())} samples):")
        print(f"    median extra delay: {np.median(wrong_gap) * 1000:.2f} ms")
        print(f"    mean extra delay:   {np.mean(wrong_gap) * 1000:.2f} ms")
        print(f"    within 5% of oracle: {np.mean(near_optimal):.1%}")

    print("\n" + "=" * 60)
    print("CLOSED-LOOP ROLLOUT EVALUATION")
    print("=" * 60)
    joint_rollout_eval(
        model,
        sorted(test_seeds),
        n_steps=n_steps,
        interval=interval,
        oracle_horizon=oracle_horizon,
        oracle_mode=oracle_mode,
    )

    return top1, top3, top5


def main():
    parser = argparse.ArgumentParser(description="Supervised obs->combo learnability test")
    parser.add_argument("--n_seeds", type=int, default=10, help="Number of env seeds")
    parser.add_argument("--n_steps", type=int, default=500, help="Steps per seed")
    parser.add_argument("--interval", type=int, default=10, help="High-level interval")
    parser.add_argument("--oracle_horizon", type=int, default=None,
                        help="Oracle scoring horizon in low-steps (default: interval)")
    parser.add_argument("--oracle_mode", type=str, default="greedy",
                        choices=["greedy", "indep"],
                        help="Per-user oracle mode")
    parser.add_argument("--behavior_policy", type=str, default="mixed",
                        choices=["baseline", "random", "oracle", "mixed"],
                        help="Data collection behavior policy")
    parser.add_argument("--data_file", type=str, default=None,
                        help="Load pre-collected data (skips collection)")
    args = parser.parse_args()

    if args.oracle_horizon is None:
        args.oracle_horizon = args.interval

    data_path = args.data_file or os.path.join(_SCRIPT_DIR, "supervised_obs_data.npz")

    if args.data_file and os.path.exists(args.data_file):
        print(f"Loading data from {args.data_file}")
        data = np.load(args.data_file)
        obs, labels, delays = data["obs"], data["labels"], data["delays"]
        if "seed_ids" in data:
            seed_ids = data["seed_ids"]
        else:
            tmp = HeurEnv(seed=0)
            samples_per_seed = (args.n_steps // args.interval) * tmp.M_sim
            n_seeds_inferred = len(obs) // max(1, samples_per_seed)
            seed_ids = np.repeat(
                np.arange(n_seeds_inferred, dtype=np.int32),
                samples_per_seed,
            )[:len(obs)]
            print(f"  (seed_ids missing, inferred {n_seeds_inferred} seeds)")
        print(f"Loaded: {obs.shape[0]} samples, obs_dim={obs.shape[1]}")
    else:
        t0 = time.time()
        obs, labels, delays, seed_ids = collect_data(
            n_seeds=args.n_seeds,
            n_steps=args.n_steps,
            interval=args.interval,
            behavior_policy=args.behavior_policy,
            oracle_horizon=args.oracle_horizon,
            oracle_mode=args.oracle_mode,
        )
        elapsed = time.time() - t0
        print(f"Data collection took {elapsed:.0f}s")

        np.savez(data_path, obs=obs, labels=labels, delays=delays, seed_ids=seed_ids)
        print(f"Saved to {data_path}")

    train_and_eval(
        obs,
        labels,
        delays,
        seed_ids,
        n_steps=args.n_steps,
        interval=args.interval,
        oracle_horizon=args.oracle_horizon,
        oracle_mode=args.oracle_mode,
    )


if __name__ == "__main__":
    main()
