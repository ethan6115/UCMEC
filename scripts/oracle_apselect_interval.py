"""
Oracle AP-selection evaluation for cluster_rand environment.

Compares two strategies per interval, using a trained low-level policy:
  1. baseline_topk  – top-k APs by beta (what the env does by default)
  2. oracle_greedy  – sequential per-user sweep over all C(10, k) combos from
                      each user's top-10 APs, picking the combo that minimises
                      that user's avg total delay over the interval.

Output shows theoretical improvement space for a high-level AP-selection policy.

Usage:
    python scripts/oracle_apselect_interval.py
"""

import contextlib
import os
import sys
import copy
import itertools

import numpy as np
import torch

# ── path setup ───────────────────────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

# ── config ────────────────────────────────────────────────────────────────────
MODEL_LOW = (
    r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul"
    r"\results\hotspotEnv\nlos_cluster\rmappo\noncoop_rnn"
    r"\hotspot_cluster2\models\actor_999.pt"
)

SEEDS = [18, 62, 53, 14, 58]


CLUSTER_K           = 2      # fixed cluster size to evaluate
TOP_K_CANDIDATES    = 8     # how many top-beta APs to consider for oracle
HIERARCHICAL_INTERVAL = 10   # low-level steps between high-level decisions
EPISODE_LOW_STEPS   = 200    # total low-level steps per episode
N_EPISODES          = 1      # episodes per seed

# ── imports ───────────────────────────────────────────────────────────────────
try:
    from envs.MA_UCMEC_dyna_noncoop_cluster_rand import (
        MA_UCMEC_dyna_noncoop_cluster_rand as ClusterRandEnv,
    )
    from algorithms.algorithm.r_actor_critic import R_Actor
    from config import get_config
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


# ── injectable subclass ───────────────────────────────────────────────────────
class OracleEnv(ClusterRandEnv):
    """cluster_rand with an optional forced cluster matrix for oracle search."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._forced_cluster_matrix = None

    def force_cluster(self, matrix: np.ndarray):
        """Set a fixed cluster matrix (M_sim × N_sim) to use instead of beta sort."""
        self._forced_cluster_matrix = matrix

    def release_cluster(self):
        """Revert to default top-k-by-beta clustering."""
        self._forced_cluster_matrix = None

    def cluster(self):
        if self._forced_cluster_matrix is not None:
            return self._forced_cluster_matrix
        return super().cluster()


# ── env state snapshot / restore ─────────────────────────────────────────────
def _copy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x.copy()
    if isinstance(x, list):
        return list(x)
    return copy.copy(x)


def snapshot_env(env: OracleEnv) -> dict:
    """Capture all dynamic state needed to replay an interval identically."""
    snap = {
        "rng_state":       copy.deepcopy(env.rng.__getstate__()),
        # mobility
        "locations_users": env.locations_users.copy(),
        "user_dest":       _copy(env.user_dest),
        "user_speed":      _copy(env.user_speed),
        "distance_matrix": env.distance_matrix.copy(),
        # channel
        "beta":            env.beta.copy(),
        "PL":              env.PL.copy(),
        "mu":              env.mu.copy(),
        "h":               env.h.copy(),
        "access_chan":     env.access_chan.copy(),
        "link_type":       env.link_type.copy() if hasattr(env, "link_type") else None,
        # tasks
        "Task_size":       env.Task_size.copy(),
        "Task_density":    env.Task_density.copy(),
        # last-step outputs
        "omega_last":      env.omega_last.copy(),
        "p_last":          env.p_last.copy(),
        "delay_last":      env.delay_last.copy(),
        "delay_last_clip": _copy(getattr(env, "delay_last_clip", None)),
        "front_delay_last":  _copy(getattr(env, "front_delay_last", None)),
        "uplink_delay_last": _copy(getattr(env, "uplink_delay_last", None)),
        "actual_process_delay_last": _copy(getattr(env, "actual_process_delay_last", None)),
        "uplink_rate_access_b": env.uplink_rate_access_b.copy(),
        # cluster_rand extras
        "cluster_size":    int(env.cluster_size),
        # misc
        "step_num":        int(env.step_num),
    }
    return snap


def restore_env(env: OracleEnv, snap: dict):
    """Restore env to a previously taken snapshot."""
    env.rng.__setstate__(copy.deepcopy(snap["rng_state"]))
    env.locations_users  = snap["locations_users"].copy()
    if snap["user_dest"] is not None:
        env.user_dest    = snap["user_dest"].copy()
    if snap["user_speed"] is not None:
        env.user_speed   = snap["user_speed"].copy()
    env.distance_matrix  = snap["distance_matrix"].copy()
    env.beta             = snap["beta"].copy()
    env.PL               = snap["PL"].copy()
    env.mu               = snap["mu"].copy()
    env.h                = snap["h"].copy()
    env.access_chan      = snap["access_chan"].copy()
    if snap["link_type"] is not None:
        env.link_type    = snap["link_type"].copy()
    env.Task_size        = snap["Task_size"].copy()
    env.Task_density     = snap["Task_density"].copy()
    env.omega_last       = snap["omega_last"].copy()
    env.p_last           = snap["p_last"].copy()
    env.delay_last       = snap["delay_last"].copy()
    if snap["delay_last_clip"] is not None:
        env.delay_last_clip = snap["delay_last_clip"].copy()
    if snap["front_delay_last"] is not None:
        env.front_delay_last = snap["front_delay_last"].copy()
    if snap["uplink_delay_last"] is not None:
        env.uplink_delay_last = snap["uplink_delay_last"].copy()
    if snap["actual_process_delay_last"] is not None:
        env.actual_process_delay_last = snap["actual_process_delay_last"].copy()
    env.uplink_rate_access_b = snap["uplink_rate_access_b"].copy()
    env.cluster_size     = snap["cluster_size"]
    env.step_num         = snap["step_num"]


# ── low-level policy runner ───────────────────────────────────────────────────
class LowPolicyRunner:
    def __init__(self, actor: R_Actor, args, act_space, device):
        self.actor     = actor
        self.args      = args
        self.act_space = act_space
        self.device    = device

    def run_interval(self, env: OracleEnv, obs, rnn_states, masks, n_steps: int):
        """
        Run n_steps low-level steps.

        Returns (obs, rnn_states, masks, delay_buf, front_buf)
          delay_buf: list of shape-(M_sim,) arrays, one per step (seconds)
          front_buf: same for front_delay
        """
        delay_buf = []
        front_buf = []

        for _ in range(n_steps):
            obs_batch = np.stack(obs)
            with torch.no_grad():
                actions, _, rnn_states = self.actor(
                    obs_batch, rnn_states, masks, deterministic=True
                )
            action_indices = actions.cpu().numpy().flatten()
            actions_env    = np.eye(self.act_space.n)[action_indices]

            with open(os.devnull, "w") as _dev, contextlib.redirect_stdout(_dev):
                next_obs, _, next_dones, _ = env.step(actions_env)

            if hasattr(env, "delay_last") and env.delay_last is not None:
                delay_buf.append(env.delay_last[:env.M_sim, 0].copy())
            if hasattr(env, "front_delay_last") and env.front_delay_last is not None:
                front_buf.append(env.front_delay_last[:env.M_sim, 0].copy())

            masks = np.array([[0.0] if d else [1.0] for d in next_dones],
                             dtype=np.float32)
            obs   = next_obs

            if all(next_dones):
                break

        # R_Actor returns rnn_states as a Tensor; convert back to numpy so
        # callers can safely call .copy() on it.
        if isinstance(rnn_states, torch.Tensor):
            rnn_states = rnn_states.detach().cpu().numpy()

        return obs, rnn_states, masks, delay_buf, front_buf


# ── helper: build cluster matrix from top-K candidates ───────────────────────
def build_cluster_matrix(env: OracleEnv, combo_per_user: list[np.ndarray]) -> np.ndarray:
    """
    combo_per_user: list of M_sim arrays, each containing the AP indices
                    (into N_sim space) to assign to that user.
    Returns cluster_matrix of shape (M_sim, N_sim).
    """
    cm = np.zeros((env.M_sim, env.N_sim), dtype=int)
    for i, ap_list in enumerate(combo_per_user):
        for ap in ap_list:
            cm[i, int(ap)] = 1
    return cm


def topk_ap_indices(env: OracleEnv, user_i: int, k: int) -> np.ndarray:
    """Return indices of top-k APs by beta for user_i (into N_sim space)."""
    return np.argsort(env.beta[user_i, :env.N_sim])[::-1][:k]


def top_candidates(env: OracleEnv, user_i: int, n_cand: int) -> np.ndarray:
    """Return top n_cand AP indices by beta for user_i."""
    return np.argsort(env.beta[user_i, :env.N_sim])[::-1][:n_cand]


# ── single episode evaluation ─────────────────────────────────────────────────
def evaluate_episode(env: OracleEnv, runner: LowPolicyRunner, args) -> dict:
    obs = env.reset()
    rnn_states = np.zeros((env.n_agents, args.recurrent_N, args.hidden_size),
                          dtype=np.float32)
    masks      = np.ones((env.n_agents, 1), dtype=np.float32)

    n_intervals = EPISODE_LOW_STEPS // HIERARCHICAL_INTERVAL
    all_combos  = list(itertools.combinations(range(TOP_K_CANDIDATES), CLUSTER_K))

    baseline_delay_list = []
    baseline_front_list = []
    baseline_dsr_list   = []
    oracle_delay_list   = []
    oracle_front_list   = []
    oracle_dsr_list     = []

    for interval_idx in range(n_intervals):
        # snapshot before this interval so every trial is identical
        snap       = snapshot_env(env)
        obs_snap   = [o.copy() for o in obs]
        rnn_snap   = rnn_states.copy()
        masks_snap = masks.copy()

        # precompute each user's top-candidate AP indices (from current beta)
        candidates = [top_candidates(env, u, TOP_K_CANDIDATES)
                      for u in range(env.M_sim)]

        # ── 1. baseline: top-k by beta ────────────────────────────────────
        baseline_combo = [topk_ap_indices(env, u, CLUSTER_K)
                          for u in range(env.M_sim)]
        cm_base = build_cluster_matrix(env, baseline_combo)

        restore_env(env, snap)
        env.force_cluster(cm_base)
        _, _, _, delay_b, front_b = runner.run_interval(
            env, [o.copy() for o in obs_snap],
            rnn_snap.copy(), masks_snap.copy(), HIERARCHICAL_INTERVAL
        )
        env.release_cluster()

        b_delay = float(np.mean(delay_b)) * 1000.0 if delay_b else float("nan")
        b_front = float(np.mean(front_b)) * 1000.0 if front_b else float("nan")
        flat_b  = np.concatenate(delay_b) if delay_b else np.array([])
        b_dsr   = float(np.mean(flat_b <= env.tau_c)) if flat_b.size else float("nan")

        baseline_delay_list.append(b_delay)
        baseline_front_list.append(b_front)
        baseline_dsr_list.append(b_dsr)

        # ── 2. oracle greedy: improve one user at a time ──────────────────
        # start from the baseline combo and greedily improve per user
        oracle_aps = [arr.copy() for arr in baseline_combo]

        for user_i in range(env.M_sim):
            best_delay_u = float("inf")
            best_ap_list = oracle_aps[user_i].copy()

            for combo in all_combos:
                # map combo positions (0..9) → actual AP indices via candidates
                trial_aps       = candidates[user_i][list(combo)]
                trial_oracle    = [arr.copy() for arr in oracle_aps]
                trial_oracle[user_i] = trial_aps

                cm_trial = build_cluster_matrix(env, trial_oracle)

                restore_env(env, snap)
                env.force_cluster(cm_trial)
                _, _, _, delay_t, _ = runner.run_interval(
                    env, [o.copy() for o in obs_snap],
                    rnn_snap.copy(), masks_snap.copy(), HIERARCHICAL_INTERVAL
                )
                env.release_cluster()

                if delay_t:
                    # Score by ALL-user mean delay so the greedy does not
                    # sacrifice other users to benefit user_i alone.
                    delay_u = float(np.mean(delay_t))
                else:
                    delay_u = float("inf")

                if delay_u < best_delay_u:
                    best_delay_u = delay_u
                    best_ap_list = trial_aps.copy()

            oracle_aps[user_i] = best_ap_list

        # evaluate final oracle combo
        cm_oracle = build_cluster_matrix(env, oracle_aps)
        restore_env(env, snap)
        env.force_cluster(cm_oracle)
        _, _, _, delay_o, front_o = runner.run_interval(
            env, [o.copy() for o in obs_snap],
            rnn_snap.copy(), masks_snap.copy(), HIERARCHICAL_INTERVAL
        )
        env.release_cluster()

        o_delay = float(np.mean(delay_o)) * 1000.0 if delay_o else float("nan")
        o_front = float(np.mean(front_o)) * 1000.0 if front_o else float("nan")
        flat_o  = np.concatenate(delay_o) if delay_o else np.array([])
        o_dsr   = float(np.mean(flat_o <= env.tau_c)) if flat_o.size else float("nan")

        oracle_delay_list.append(o_delay)
        oracle_front_list.append(o_front)
        oracle_dsr_list.append(o_dsr)

        print(
            f"  interval {interval_idx+1:3d}/{n_intervals}"
            f"  base={b_delay:6.1f}ms dsr={b_dsr:.3f}"
            f"  oracle={o_delay:6.1f}ms dsr={o_dsr:.3f}"
            f"  gap={b_delay - o_delay:+.1f}ms",
            flush=True,
        )

        # ── advance env along baseline trajectory for next interval ───────
        restore_env(env, snap)
        env.force_cluster(cm_base)
        obs, rnn_states, masks, _, _ = runner.run_interval(
            env, [o.copy() for o in obs_snap],
            rnn_snap.copy(), masks_snap.copy(), HIERARCHICAL_INTERVAL
        )
        env.release_cluster()

    return {
        "baseline_delay": baseline_delay_list,
        "baseline_front": baseline_front_list,
        "baseline_dsr":   baseline_dsr_list,
        "oracle_delay":   oracle_delay_list,
        "oracle_front":   oracle_front_list,
        "oracle_dsr":     oracle_dsr_list,
    }


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = get_config()
    args   = parser.parse_args([])
    args.use_recurrent_policy       = True
    args.use_naive_recurrent_policy = False
    device = torch.device("cpu")

    print(f"Oracle AP-select  k={CLUSTER_K}  top-{TOP_K_CANDIDATES} candidates"
          f"  C({TOP_K_CANDIDATES},{CLUSTER_K})={len(list(itertools.combinations(range(TOP_K_CANDIDATES), CLUSTER_K)))} combos/user"
          f"  {EPISODE_LOW_STEPS} steps/ep  interval={HIERARCHICAL_INTERVAL}")

    # build throwaway env for obs/act spaces
    _tmp = OracleEnv(render=False, seed=0)
    obs_space = _tmp.observation_space[0]
    act_space = _tmp.action_space[0]
    del _tmp

    # load low-level actor
    actor = R_Actor(args, obs_space, act_space, device)
    if not os.path.exists(MODEL_LOW):
        print(f"Model not found: {MODEL_LOW}")
        sys.exit(1)
    actor.load_state_dict(torch.load(MODEL_LOW, map_location=device))
    actor.eval()
    print(f"Loaded: {MODEL_LOW}\n")

    runner = LowPolicyRunner(actor, args, act_space, device)

    all_base_delay   = []
    all_oracle_delay = []
    all_base_dsr     = []
    all_oracle_dsr   = []
    all_base_front   = []
    all_oracle_front = []

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")
        np.random.seed(seed)
        torch.manual_seed(seed)

        env = OracleEnv(render=False, seed=seed)
        env.seed(seed)
        # fix cluster size for oracle evaluation
        env.cluster_k_min = CLUSTER_K
        env.cluster_k_max = CLUSTER_K

        ep_base_delay   = []
        ep_oracle_delay = []
        ep_base_dsr     = []
        ep_oracle_dsr   = []
        ep_base_front   = []
        ep_oracle_front = []

        for ep in range(N_EPISODES):
            print(f"  Episode {ep+1}/{N_EPISODES}")
            result = evaluate_episode(env, runner, args)
            ep_base_delay.extend(result["baseline_delay"])
            ep_oracle_delay.extend(result["oracle_delay"])
            ep_base_dsr.extend(result["baseline_dsr"])
            ep_oracle_dsr.extend(result["oracle_dsr"])
            ep_base_front.extend(result["baseline_front"])
            ep_oracle_front.extend(result["oracle_front"])

        s_base  = float(np.nanmean(ep_base_delay))
        s_ora   = float(np.nanmean(ep_oracle_delay))
        s_bd    = float(np.nanmean(ep_base_dsr))
        s_od    = float(np.nanmean(ep_oracle_dsr))
        s_bf    = float(np.nanmean(ep_base_front))
        s_of    = float(np.nanmean(ep_oracle_front))

        all_base_delay.append(s_base)
        all_oracle_delay.append(s_ora)
        all_base_dsr.append(s_bd)
        all_oracle_dsr.append(s_od)
        all_base_front.append(s_bf)
        all_oracle_front.append(s_of)

        print(f"\n  Seed {seed} summary:")
        print(f"    baseline : delay={s_base:.2f}ms  front={s_bf:.2f}ms  DSR={s_bd:.4f}")
        print(f"    oracle   : delay={s_ora:.2f}ms  front={s_of:.2f}ms  DSR={s_od:.4f}")
        print(f"    gap (↓)  : delay={s_base - s_ora:+.2f}ms  DSR={s_od - s_bd:+.4f}")

    # aggregate
    print(f"\n{'='*60}")
    print(f"AGGREGATE  ({len(SEEDS)} seeds, k={CLUSTER_K}, "
          f"C({TOP_K_CANDIDATES},{CLUSTER_K}) combos/user)")
    print(f"{'='*60}")
    agg_bd = float(np.nanmean(all_base_delay))
    agg_od = float(np.nanmean(all_oracle_delay))
    agg_bs = float(np.nanmean(all_base_dsr))
    agg_os = float(np.nanmean(all_oracle_dsr))
    agg_bf = float(np.nanmean(all_base_front))
    agg_of = float(np.nanmean(all_oracle_front))
    print(f"                 avg delay (ms)   front (ms)    DSR")
    print(f"  baseline_topk :  {agg_bd:8.2f}       {agg_bf:8.2f}    {agg_bs:.4f}")
    print(f"  oracle_greedy :  {agg_od:8.2f}       {agg_of:8.2f}    {agg_os:.4f}")
    print(f"  oracle margin :  {agg_bd - agg_od:+8.2f} ms delay   {agg_os - agg_bs:+.4f} DSR")
    print(f"\n  per-seed:")
    print(f"  {'seed':>6}  {'base(ms)':>10}  {'oracle(ms)':>10}  {'gap(ms)':>8}  {'b_dsr':>6}  {'o_dsr':>6}")
    for i, seed in enumerate(SEEDS):
        print(f"  {seed:>6}  {all_base_delay[i]:>10.2f}  {all_oracle_delay[i]:>10.2f}"
              f"  {all_base_delay[i] - all_oracle_delay[i]:>+8.2f}"
              f"  {all_base_dsr[i]:>6.4f}  {all_oracle_dsr[i]:>6.4f}")


if __name__ == "__main__":
    main()
