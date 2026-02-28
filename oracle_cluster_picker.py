import os
import sys
import copy
import math
import numpy as np
import torch

# ========= User Config (edit only this block) =========
MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\MyEnv\rmappo\noncoop_rnn\IPPO_cluster5\models/actor_999.pt"
DEVICE = "cpu"  # "cuda" or "cpu"
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
NUM_EPISODES = 1
MAX_STEPS = 200
CLUSTER_CANDIDATES = [1, 3, 5, 7, 9]
CLUSTER_INTERVAL = 10
DETERMINISTIC = True
USE_RECURRENT_POLICY = True
RANDOM_POLICY_FALLBACK = False
FIXED_CLUSTERS = [1, 3, 5, 7, 9]

# Oracle objective:
#   "sum"  -> J_sum = (avg_u + avg_f + avg_p) * N_off   (comparable across N_off)
#   "avg"  -> J_avg = (avg_u + avg_f + avg_p)           (per-offloading-user average)
#   "total"-> avg_total_delay_ms                        (use total delay to pick cluster)
ORACLE_OBJECTIVE = "sum"
# ============================================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from envs.MA_UCMEC_dyna_noncoop_test import MA_UCMEC_dyna_noncoop
    from algorithms.algorithm.r_actor_critic import R_Actor
    from config import get_config
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

try:
    import pandas as pd
except Exception:
    pd = None

SNAPSHOT_ATTRS = [
    "step_num",
    "locations_users",
    "user_dest",
    "user_speed",
    "distance_matrix",
    "PL",
    "mu",
    "beta",
    "Task_size",
    "Task_density",
    "omega_last",
    "p_last",
    "p_idx_last",
    "delay_last",
    "cluster_size",
    "cluster_matrix",
    "front_delay_last",
    "uplink_rate_access_b",
    "P_los",
    "link_type",
    "G",
    "h",
    "h_real",
    "h_imag",
]


def _clone_value(val):
    if isinstance(val, np.ndarray):
        return np.copy(val)
    return copy.deepcopy(val)


def snapshot_env(env):
    state = {}
    state["rng_state"] = copy.deepcopy(env.rng.bit_generator.state)
    for name in SNAPSHOT_ATTRS:
        if hasattr(env, name):
            state[name] = _clone_value(getattr(env, name))
    if hasattr(env, "opt_vars"):
        opt_vals = []
        for var in env.opt_vars:
            opt_vals.append(None if var.value is None else np.copy(var.value))
        state["opt_var_values"] = opt_vals
    return state


def restore_env(env, state):
    env.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
    for name in SNAPSHOT_ATTRS:
        if name in state:
            setattr(env, name, _clone_value(state[name]))
    if "opt_var_values" in state and hasattr(env, "opt_vars"):
        for var, val in zip(env.opt_vars, state["opt_var_values"]):
            var.value = None if val is None else np.copy(val)


def compute_offloading_terms(info0):
    """
    Returns offloading-related terms from env.info[0].
    avg_* are assumed to be averages over offloading users (as in logging).

    J_avg = avg_u + avg_f + avg_p                      (per-offloading-user average)
    J_sum = (avg_u + avg_f + avg_p) * N_off            (comparable across N_off)
    """
    if not info0:
        return {
            "n_off": 0,
            "avg_u": 0.0,
            "avg_f": 0.0,
            "avg_p": 0.0,
            "j_avg": 0.0,
            "j_sum": 0.0,
        }

    n_off = int(info0.get("num_offloading_users", 0))
    avg_u = float(info0.get("avg_uplink_delay_ms", 0.0))
    avg_f = float(info0.get("avg_front_delay_ms", 0.0))
    avg_p = float(info0.get("avg_actual_process_delay_ms", 0.0))

    if n_off <= 0:
        return {
            "n_off": 0,
            "avg_u": avg_u,
            "avg_f": avg_f,
            "avg_p": avg_p,
            "j_avg": 0.0,
            "j_sum": 0.0,
        }

    j_avg = avg_u + avg_f + avg_p
    j_sum = j_avg * n_off
    return {
        "n_off": n_off,
        "avg_u": avg_u,
        "avg_f": avg_f,
        "avg_p": avg_p,
        "j_avg": j_avg,
        "j_sum": j_sum,
    }


def oracle_score(info0):
    """
    Score used for oracle cluster picking.
    IMPORTANT: if n_off == 0, set score = +inf to avoid trivial "no-offloading" exploit.
    """
    if ORACLE_OBJECTIVE == "total":
        return float(info0.get("avg_total_delay_ms", 0.0)) if info0 else math.inf

    t = compute_offloading_terms(info0)
    if t["n_off"] <= 0:
        return math.inf

    if ORACLE_OBJECTIVE == "sum":
        return float(t["j_sum"])
    if ORACLE_OBJECTIVE == "avg":
        return float(t["j_avg"])

    raise ValueError(f"Unknown ORACLE_OBJECTIVE: {ORACLE_OBJECTIVE}")


class PolicyWrapper:
    def __init__(self, actor, device, action_dim, args=None):
        self.actor = actor
        self.device = device
        self.action_dim = action_dim
        self.args = args
        self.rnn_states = None
        self.masks = None
        self.mode = None

    def reset(self, n_agents):
        if self.actor is None:
            self.rnn_states = None
            self.masks = None
            return
        if USE_RECURRENT_POLICY and self.args is not None:
            self.rnn_states = np.zeros(
                (n_agents, self.args.recurrent_N, self.args.hidden_size), dtype=np.float32
            )
        else:
            self.rnn_states = None
        self.masks = np.ones((n_agents, 1), dtype=np.float32)

    def _act_r_actor(self, obs_batch):
        actions, _, rnn_states = self.actor(
            obs_batch, self.rnn_states, self.masks, deterministic=DETERMINISTIC
        )
        action_indices = actions.cpu().numpy().flatten().astype(int)
        return action_indices, rnn_states

    def _act_logits(self, obs_batch):
        obs_t = torch.as_tensor(obs_batch, device=self.device, dtype=torch.float32)
        out = self.actor(obs_t)
        if isinstance(out, (tuple, list)):
            out = out[0]
        action_indices = torch.argmax(out, dim=-1).cpu().numpy().astype(int)
        return action_indices, self.rnn_states

    def act(self, obs_list, update_state=True):
        n_agents = len(obs_list)
        obs_batch = np.stack(obs_list)
        if self.actor is None:
            action_indices = np.random.randint(0, self.action_dim, size=n_agents, dtype=int)
            actions_oh = np.eye(self.action_dim, dtype=np.float32)[action_indices]
            return [actions_oh[i] for i in range(n_agents)]

        with torch.no_grad():
            if self.mode is None:
                try:
                    action_indices, rnn_states = self._act_r_actor(obs_batch)
                    self.mode = "r_actor"
                except Exception:
                    action_indices, rnn_states = self._act_logits(obs_batch)
                    self.mode = "logits"
            elif self.mode == "r_actor":
                action_indices, rnn_states = self._act_r_actor(obs_batch)
            else:
                action_indices, rnn_states = self._act_logits(obs_batch)

        if update_state and rnn_states is not None:
            self.rnn_states = rnn_states

        actions_oh = np.eye(self.action_dim, dtype=np.float32)[action_indices]
        return [actions_oh[i] for i in range(n_agents)]


def load_policy(env):
    device = torch.device(DEVICE)
    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = torch.device("cpu")

    if not os.path.exists(MODEL_LOW):
        msg = f"Model not found: {MODEL_LOW}"
        if RANDOM_POLICY_FALLBACK:
            print(f"{msg}. Using random policy.")
            return PolicyWrapper(None, device, env.action_space[0].n)
        raise FileNotFoundError(msg)

    obj = torch.load(MODEL_LOW, map_location=device)
    actor = None
    args = None
    if isinstance(obj, torch.nn.Module):
        actor = obj
    elif isinstance(obj, dict):
        state_dict = None
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state_dict = obj["state_dict"]
        elif "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            state_dict = obj["model_state_dict"]
        else:
            state_dict = obj

        parser = get_config()
        args = parser.parse_args([])
        args.use_recurrent_policy = bool(USE_RECURRENT_POLICY)
        args.use_naive_recurrent_policy = False
        args.use_set_encoder = False
        obs_space = env.observation_space[0]
        act_space = env.action_space[0]
        actor = R_Actor(args, obs_space, act_space, device)
        actor.load_state_dict(state_dict)
    else:
        msg = f"Unsupported model format: {type(obj)}"
        if RANDOM_POLICY_FALLBACK:
            print(f"{msg}. Using random policy.")
            return PolicyWrapper(None, device, env.action_space[0].n)
        raise RuntimeError(msg)

    actor.eval()
    return PolicyWrapper(actor, device, env.action_space[0].n, args=args)


def make_env(seed):
    env = MA_UCMEC_dyna_noncoop(render=False, seed=seed)
    env.seed(seed)
    return env


def run_episode(env, policy, mode_name, fixed_cluster=None):
    obs = env.reset()
    policy.reset(env.n_agents)
    if fixed_cluster is not None:
        env.cluster_size = fixed_cluster

    step_count = 0
    done = [0] * env.n_agents
    off_javg_sum = 0.0
    off_jsum_sum = 0.0
    off_javg_cond_sum = 0.0
    off_steps = 0
    total_delay_sum = 0.0
    off_users_sum = 0.0
    steps = 0
    cluster_hist = {c: 0 for c in CLUSTER_CANDIDATES}

    while (not all(done)) and step_count < MAX_STEPS:
        next_step = env.step_num + 1
        if mode_name == "oracle" and (next_step == 1 or next_step % CLUSTER_INTERVAL == 0):
            action = policy.act(obs, update_state=True)
            snapshot = snapshot_env(env)
            best_cand = None
            best_delay = math.inf
            best_total = math.inf
            best_total_cand = None
            for cand in CLUSTER_CANDIDATES:
                restore_env(env, snapshot)
                env.cluster_size = cand
                _, _, _, info = env.step(action)
                info0 = info[0] if info else {}
                total_d = float(info0.get("avg_total_delay_ms", 0.0))
                if total_d < best_total:
                    best_total = total_d
                    best_total_cand = cand

                score = oracle_score(info0)
                if score < best_delay:
                    best_delay = score
                    best_cand = cand

            if best_cand is None:
                best_cand = best_total_cand if best_total_cand is not None else CLUSTER_CANDIDATES[0]
            restore_env(env, snapshot)
            env.cluster_size = best_cand
            cluster_hist[best_cand] += 1
            obs, _, done, info = env.step(action)
        else:
            action = policy.act(obs, update_state=True)
            obs, _, done, info = env.step(action)

        info0 = info[0] if info else {}
        t = compute_offloading_terms(info0)
        off_javg_sum += float(t["j_avg"])
        off_jsum_sum += float(t["j_sum"])
        if t["n_off"] > 0:
            off_javg_cond_sum += float(t["j_avg"])
            off_steps += 1
        total_delay_sum += float(info0.get("avg_total_delay_ms", 0.0))
        off_users_sum += float(t["n_off"])
        steps += 1
        step_count += 1

    mean_off_javg = off_javg_sum / steps if steps > 0 else 0.0
    mean_off_jsum = off_jsum_sum / steps if steps > 0 else 0.0
    mean_off_javg_cond = off_javg_cond_sum / off_steps if off_steps > 0 else 0.0
    off_step_ratio = float(off_steps) / float(steps) if steps > 0 else 0.0
    mean_total_delay = total_delay_sum / steps if steps > 0 else 0.0
    avg_off_users = off_users_sum / steps if steps > 0 else 0.0
    return {
        "mean_offloading_delay_ms": mean_off_javg,
        "mean_offloading_delay_cond_ms": mean_off_javg_cond,
        "mean_offloading_obj_ms": mean_off_jsum,
        "offloading_step_ratio": off_step_ratio,
        "mean_total_delay_ms": mean_total_delay,
        "avg_offloading_users": avg_off_users,
        "cluster_hist": cluster_hist,
        "steps": steps,
    }


def run_all():
    summary = {
        "oracle": {
            "mean_offloading_delay_ms": [],
            "mean_offloading_delay_cond_ms": [],
            "mean_offloading_obj_ms": [],
            "offloading_step_ratio": [],
            "mean_total_delay_ms": [],
            "avg_offloading_users": [],
        },
        **{
            f"fixed_c{c}": {
                "mean_offloading_delay_ms": [],
                "mean_offloading_delay_cond_ms": [],
                "mean_offloading_obj_ms": [],
                "offloading_step_ratio": [],
                "mean_total_delay_ms": [],
                "avg_offloading_users": [],
            }
            for c in FIXED_CLUSTERS
        },
    }

    for seed in SEEDS:
        np.random.seed(seed)
        torch.manual_seed(seed)

        for ep in range(NUM_EPISODES):
            env = make_env(seed)
            policy = load_policy(env)
            oracle_res = run_episode(env, policy, mode_name="oracle")
            summary["oracle"]["mean_offloading_delay_ms"].append(
                oracle_res["mean_offloading_delay_ms"]
            )
            summary["oracle"]["mean_offloading_delay_cond_ms"].append(
                oracle_res["mean_offloading_delay_cond_ms"]
            )
            summary["oracle"]["mean_offloading_obj_ms"].append(
                oracle_res["mean_offloading_obj_ms"]
            )
            summary["oracle"]["offloading_step_ratio"].append(
                oracle_res["offloading_step_ratio"]
            )
            summary["oracle"]["mean_total_delay_ms"].append(
                oracle_res["mean_total_delay_ms"]
            )
            summary["oracle"]["avg_offloading_users"].append(
                oracle_res["avg_offloading_users"]
            )
            if "cluster_hist_list" not in summary["oracle"]:
                summary["oracle"]["cluster_hist_list"] = []
            summary["oracle"]["cluster_hist_list"].append(oracle_res["cluster_hist"])
            print(
                f"[Seed {seed} Ep {ep}] Oracle "
                f"obj={ORACLE_OBJECTIVE}, "
                f"J_avg(all)={oracle_res['mean_offloading_delay_ms']:.4f}, "
                f"J_avg(cond)={oracle_res['mean_offloading_delay_cond_ms']:.4f}, "
                f"J_sum={oracle_res['mean_offloading_obj_ms']:.4f}, "
                f"off_step_ratio={oracle_res['offloading_step_ratio']:.3f}, "
                f"mean_total_delay_ms={oracle_res['mean_total_delay_ms']:.4f}, "
                f"cluster_hist={oracle_res['cluster_hist']}"
            )

            for c in FIXED_CLUSTERS:
                env = make_env(seed)
                policy = load_policy(env)
                fixed_res = run_episode(env, policy, mode_name="fixed", fixed_cluster=c)
                key = f"fixed_c{c}"
                summary[key]["mean_offloading_delay_ms"].append(
                    fixed_res["mean_offloading_delay_ms"]
                )
                summary[key]["mean_offloading_delay_cond_ms"].append(
                    fixed_res["mean_offloading_delay_cond_ms"]
                )
                summary[key]["mean_offloading_obj_ms"].append(
                    fixed_res["mean_offloading_obj_ms"]
                )
                summary[key]["offloading_step_ratio"].append(
                    fixed_res["offloading_step_ratio"]
                )
                summary[key]["mean_total_delay_ms"].append(
                    fixed_res["mean_total_delay_ms"]
                )
                summary[key]["avg_offloading_users"].append(
                    fixed_res["avg_offloading_users"]
                )
                print(
                    f"[Seed {seed} Ep {ep}] Fixed c={c} "
                    f"J_avg(all)={fixed_res['mean_offloading_delay_ms']:.4f}, "
                    f"J_avg(cond)={fixed_res['mean_offloading_delay_cond_ms']:.4f}, "
                    f"J_sum={fixed_res['mean_offloading_obj_ms']:.4f}, "
                    f"off_step_ratio={fixed_res['offloading_step_ratio']:.3f}"
                )

    rows = []
    for name, metrics in summary.items():
        vals = metrics["mean_offloading_delay_ms"]
        arr = np.array(vals, dtype=np.float64) if len(vals) else np.array([0.0])
        cond_vals = metrics.get("mean_offloading_delay_cond_ms", [])
        cond_arr = np.array(cond_vals, dtype=np.float64) if len(cond_vals) else np.array([0.0])
        obj_vals = metrics.get("mean_offloading_obj_ms", [])
        obj_arr = np.array(obj_vals, dtype=np.float64) if len(obj_vals) else np.array([0.0])
        ratio_vals = metrics.get("offloading_step_ratio", [])
        ratio_arr = np.array(ratio_vals, dtype=np.float64) if len(ratio_vals) else np.array([0.0])
        total_vals = metrics["mean_total_delay_ms"]
        total_arr = (
            np.array(total_vals, dtype=np.float64) if len(total_vals) else np.array([0.0])
        )
        off_users_vals = metrics["avg_offloading_users"]
        off_users_arr = (
            np.array(off_users_vals, dtype=np.float64)
            if len(off_users_vals)
            else np.array([0.0])
        )
        rows.append(
            {
                "name": name,
                "mean_offloading_delay_ms": float(arr.mean()),
                "mean_offloading_delay_cond_ms": float(cond_arr.mean()),
                "mean_offloading_obj_ms": float(obj_arr.mean()),
                "offloading_step_ratio": float(ratio_arr.mean()),
                "mean_total_delay_ms": float(total_arr.mean()),
                "std_over_seeds": float(arr.std()),
                "avg_offloading_users": float(off_users_arr.mean()),
            }
        )

    if pd is not None:
        df = pd.DataFrame(rows).set_index("name")
        print("\n=== Summary ===")
        print(df)
        if "cluster_hist_list" in summary["oracle"]:
            for idx, hist in enumerate(summary["oracle"]["cluster_hist_list"]):
                seed = SEEDS[idx // NUM_EPISODES]
                ep = idx % NUM_EPISODES
                print(
                    f"[Seed {seed} Ep {ep}] Oracle "
                    f"mean_offloading_delay_ms={summary['oracle']['mean_offloading_delay_ms'][idx]:.4f}, "
                    f"mean_total_delay_ms={summary['oracle']['mean_total_delay_ms'][idx]:.4f}, "
                    f"cluster_hist={hist}"
                )
    else:
        print("\n=== Summary ===")
        for row in rows:
            print(
                f"{row['name']}: mean_offloading_delay_ms={row['mean_offloading_delay_ms']:.4f}, "
                f"mean_total_delay_ms={row['mean_total_delay_ms']:.4f}, "
                f"std_over_seeds={row['std_over_seeds']:.4f}, "
                f"avg_offloading_users={row['avg_offloading_users']:.4f}"
            )
        if "cluster_hist_list" in summary["oracle"]:
            for idx, hist in enumerate(summary["oracle"]["cluster_hist_list"]):
                seed = SEEDS[idx // NUM_EPISODES]
                ep = idx % NUM_EPISODES
                print(
                    f"[Seed {seed} Ep {ep}] Oracle "
                    f"mean_offloading_delay_ms={summary['oracle']['mean_offloading_delay_ms'][idx]:.4f}, "
                    f"mean_total_delay_ms={summary['oracle']['mean_total_delay_ms'][idx]:.4f}, "
                    f"cluster_hist={hist}"
                )


def main():
    run_all()


if __name__ == "__main__":
    main()
