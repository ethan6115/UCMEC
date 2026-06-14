"""
Pure environment test: does AP selection significantly affect front delay?

No trained model needed. Uses a fixed heuristic low-level policy
(best CPU by fronthaul pathloss, max power) to isolate the effect of
AP selection on fronthaul delay.

Compares:
  baseline: top-2 APs by beta (default)
  oracle:   best C(10,2) combo per user (greedy, minimize front delay)

Usage:
    python scripts/test_ap_front_gap.py
"""

import contextlib
import copy
import itertools
import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from envs.ucmec_hierarchical import (
    UCMEC_hierarchical_env as Obs57Env,
)

# ── config ────────────────────────────────────────────────────────────────────
SEEDS = [18, 62, 53, 14, 58]
CANDIDATE_N = 8
K_FIXED = 2
N_EPISODES = 1
SLOTS_PER_EPISODE = 200
INTERVAL = 10


def heuristic_action(env, cluster_matrix):
    """
    Heuristic low-level policy: for each user, pick the CPU with best
    average fronthaul pathloss to their selected APs, use max power.
    Returns one-hot action arrays (M_sim x action_dim).
    """
    actions = []
    for i in range(env.M_sim):
        ap_idx = np.where(cluster_matrix[i, :] == 1)[0]
        if len(ap_idx) == 0:
            # fallback: CPU1, max power
            best_cpu = 0
        else:
            # pick CPU with best (lowest) average distance to selected APs
            avg_dist = np.zeros(env.K)
            for cpu in range(env.K):
                avg_dist[cpu] = np.mean(env.distance_matrix_front[ap_idx, cpu])
            best_cpu = int(np.argmin(avg_dist))

        # max power (p_level 3) for chosen CPU
        # mask_local=True: action = cpu_id * 3 + 2 (power level 3)
        # mask_local=False: action = cpu_id * 3 + 3
        if env.mask_local:
            act_idx = best_cpu * 3 + 2  # 0-8
        else:
            act_idx = best_cpu * 3 + 3  # 1-9 (0=local)

        one_hot = np.zeros(env.action_dim)
        one_hot[act_idx] = 1.0
        actions.append(one_hot)
    return actions


def build_cluster_matrix(env, top_idx_all, combo_per_user):
    """Build cluster_matrix from per-user AP combo selections."""
    cm = np.zeros((env.M_sim, env.N_sim), dtype=int)
    for i in range(env.M_sim):
        for c in combo_per_user[i]:
            ap = int(top_idx_all[i, c])
            cm[i, ap] = 1
    return cm


def run_interval_with_cluster(env, snap, cluster_matrix, n_steps):
    """Run n_steps with a fixed cluster_matrix, return avg front delay per user."""
    restore_env(env, snap)
    env.cluster_matrix = cluster_matrix.copy()
    env.current_cluster_size = np.sum(cluster_matrix, axis=1).astype(np.int32)

    front_sum = np.zeros(env.M_sim)
    for _ in range(n_steps):
        env.advance_channel()
        # keep cluster_matrix fixed
        env.cluster_matrix = cluster_matrix.copy()
        actions = heuristic_action(env, cluster_matrix)
        env.step_low(actions)
        front_sum += env.front_delay_last[:, 0]

    return front_sum / n_steps


def snapshot_env(env):
    """Capture dynamic state for replay."""
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
    }
    return snap


def restore_env(env, snap):
    """Restore env to snapshot."""
    env.rng.__setstate__(copy.deepcopy(snap["rng_state"]))
    env.locations_users = snap["locations_users"].copy()
    if snap["user_dest"] is not None:
        env.user_dest = snap["user_dest"].copy()
    if snap["user_speed"] is not None:
        env.user_speed = snap["user_speed"].copy()
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


def main():
    ap_combos = list(itertools.combinations(range(CANDIDATE_N), K_FIXED))
    n_combos = len(ap_combos)
    print(f"C({CANDIDATE_N},{K_FIXED}) = {n_combos} combos per user")
    print(f"Seeds: {SEEDS}")
    print()

    all_base_front = []
    all_oracle_front = []
    all_base_total = []
    all_oracle_total = []

    for seed in SEEDS:
        print(f"{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        env = Obs57Env(seed=seed)
        env.candidate_n = CANDIDATE_N
        env.k_fixed = K_FIXED
        env._ap_combos = ap_combos

        seed_base_front = []
        seed_oracle_front = []
        seed_base_total = []
        seed_oracle_total = []

        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            env.reset()

            n_intervals = SLOTS_PER_EPISODE // INTERVAL
            for iv in range(n_intervals):
                # advance channel once to get current beta for AP ranking
                env.advance_channel()
                beta_sim = env.beta[:env.M_sim, :env.N_sim]
                top_idx_all = np.zeros((env.M_sim, CANDIDATE_N), dtype=np.int32)
                for i in range(env.M_sim):
                    top_idx_all[i] = np.argsort(beta_sim[i])[::-1][:CANDIDATE_N]

                # take snapshot before interval
                snap = snapshot_env(env)

                # --- baseline: top-k by beta ---
                baseline_combo = list(range(K_FIXED))  # indices 0,1 = top-2
                baseline_per_user = [baseline_combo] * env.M_sim
                cm_base = build_cluster_matrix(env, top_idx_all, baseline_per_user)
                base_front = run_interval_with_cluster(env, snap, cm_base, INTERVAL)

                # also get total delay from last run
                restore_env(env, snap)
                env.cluster_matrix = cm_base.copy()
                env.current_cluster_size = np.sum(cm_base, axis=1).astype(np.int32)
                base_total_sum = np.zeros(env.M_sim)
                for _ in range(INTERVAL):
                    env.advance_channel()
                    env.cluster_matrix = cm_base.copy()
                    actions = heuristic_action(env, cm_base)
                    env.step_low(actions)
                    base_total_sum += env.delay_last_clip[:, 0]
                base_total = base_total_sum / INTERVAL

                # --- oracle: greedy per-user sweep ---
                best_combo_per_user = [baseline_combo] * env.M_sim
                best_front_per_user = base_front.copy()

                for user_i in range(env.M_sim):
                    for combo_idx, combo in enumerate(ap_combos):
                        if list(combo) == baseline_combo:
                            continue
                        test_per_user = list(best_combo_per_user)
                        test_per_user[user_i] = list(combo)
                        cm_test = build_cluster_matrix(env, top_idx_all, test_per_user)
                        test_front = run_interval_with_cluster(env, snap, cm_test, INTERVAL)
                        if test_front[user_i] < best_front_per_user[user_i]:
                            best_front_per_user[user_i] = test_front[user_i]
                            best_combo_per_user[user_i] = list(combo)

                # run oracle final to get total delay
                cm_oracle = build_cluster_matrix(env, top_idx_all, best_combo_per_user)
                restore_env(env, snap)
                env.cluster_matrix = cm_oracle.copy()
                env.current_cluster_size = np.sum(cm_oracle, axis=1).astype(np.int32)
                oracle_total_sum = np.zeros(env.M_sim)
                for _ in range(INTERVAL):
                    env.advance_channel()
                    env.cluster_matrix = cm_oracle.copy()
                    actions = heuristic_action(env, cm_oracle)
                    env.step_low(actions)
                    oracle_total_sum += env.delay_last_clip[:, 0]
                oracle_total = oracle_total_sum / INTERVAL

                avg_base_front = float(np.mean(base_front) * 1000)
                avg_oracle_front = float(np.mean(best_front_per_user) * 1000)
                avg_base_total = float(np.mean(base_total) * 1000)
                avg_oracle_total = float(np.mean(oracle_total) * 1000)

                seed_base_front.append(avg_base_front)
                seed_oracle_front.append(avg_oracle_front)
                seed_base_total.append(avg_base_total)
                seed_oracle_total.append(avg_oracle_total)

                # advance env to end of interval for next iteration
                restore_env(env, snap)
                env.cluster_matrix = cm_base.copy()
                env.current_cluster_size = np.sum(cm_base, axis=1).astype(np.int32)
                for _ in range(INTERVAL):
                    env.advance_channel()
                    env.cluster_matrix = cm_base.copy()
                    actions = heuristic_action(env, cm_base)
                    env.step_low(actions)

        # print per-interval results
        for iv in range(len(seed_base_front)):
            gap_f = seed_base_front[iv] - seed_oracle_front[iv]
            gap_t = seed_base_total[iv] - seed_oracle_total[iv]
            print(
                f"  interval {iv+1:3d}/{len(seed_base_front)}  "
                f"base_front={seed_base_front[iv]:7.1f}ms  "
                f"oracle_front={seed_oracle_front[iv]:7.1f}ms  "
                f"gap_front={gap_f:+7.1f}ms  |  "
                f"base_total={seed_base_total[iv]:7.1f}ms  "
                f"oracle_total={seed_oracle_total[iv]:7.1f}ms  "
                f"gap_total={gap_t:+7.1f}ms"
            )

        avg_bf = np.mean(seed_base_front)
        avg_of = np.mean(seed_oracle_front)
        avg_bt = np.mean(seed_base_total)
        avg_ot = np.mean(seed_oracle_total)
        print(f"\n  Seed {seed} summary:")
        print(f"    front:  baseline={avg_bf:.1f}ms  oracle={avg_of:.1f}ms  gap={avg_bf-avg_of:+.1f}ms ({(avg_bf-avg_of)/max(avg_bf,1)*100:+.1f}%)")
        print(f"    total:  baseline={avg_bt:.1f}ms  oracle={avg_ot:.1f}ms  gap={avg_bt-avg_ot:+.1f}ms ({(avg_bt-avg_ot)/max(avg_bt,1)*100:+.1f}%)")
        print()

        all_base_front.extend(seed_base_front)
        all_oracle_front.extend(seed_oracle_front)
        all_base_total.extend(seed_base_total)
        all_oracle_total.extend(seed_oracle_total)

    # aggregate
    print(f"{'='*60}")
    print(f"AGGREGATE ({len(SEEDS)} seeds, heuristic low-level, mask_local)")
    print(f"{'='*60}")
    abf = np.mean(all_base_front)
    aof = np.mean(all_oracle_front)
    abt = np.mean(all_base_total)
    aot = np.mean(all_oracle_total)
    print(f"  front delay:  baseline={abf:.1f}ms  oracle={aof:.1f}ms  gap={abf-aof:+.1f}ms ({(abf-aof)/max(abf,1)*100:+.1f}%)")
    print(f"  total delay:  baseline={abt:.1f}ms  oracle={aot:.1f}ms  gap={abt-aot:+.1f}ms ({(abt-aot)/max(abt,1)*100:+.1f}%)")


if __name__ == "__main__":
    main()
