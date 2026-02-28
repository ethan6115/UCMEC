import os
import sys
import numpy as np
import torch

# ===== User config =====
MODEL_PATH = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\MyEnv\MyEnv\mappo\noncoop_paper_baseline\git_interval10\models/actor.pt"
SEEDS = [4]
NUM_EPISODES = 1
DEVICE = "cpu"  # "cpu" or "cuda"
USE_RECURRENT = False
DETERMINISTIC = True
# =======================

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.MA_UCMEC_stat_noncoop import MA_UCMEC_stat_noncoop
#from envs.MA_UCMEC_dyna_noncoop_paperver import MA_UCMEC_dyna_noncoop
from algorithms.algorithm.r_actor_critic import R_Actor
from config import get_config


def build_actor(env, model_path, device):
    parser = get_config()
    args = parser.parse_args([])
    args.use_recurrent_policy = bool(USE_RECURRENT)
    args.use_naive_recurrent_policy = False

    obs_space = env.observation_space[0]
    act_space = env.action_space[0]

    actor = R_Actor(args, obs_space, act_space, device)
    state_dict = torch.load(model_path, map_location=device)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor, args, act_space


def eval_one_seed(actor, args, act_space, seed):
    env = MA_UCMEC_stat_noncoop(render=False)
    env.seed(seed)

    per_ep = {
        "avg_total_delay_ms": [],
        "deadline_satisfaction_ratio": [],
        "avg_uplink_rate_Mbps": [],
        "avg_offloading_users": [],
        "avg_reward": [],
        "power_dist": [],
    }

    for _ in range(NUM_EPISODES):
        np.random.seed(seed)
        torch.manual_seed(seed)

        obs = env.reset()
        rnn_states = np.zeros((env.n_agents, args.recurrent_N, args.hidden_size), dtype=np.float32)
        masks = np.ones((env.n_agents, 1), dtype=np.float32)
        done = np.zeros(env.n_agents, dtype=bool)

        sum_delay_ms = 0.0
        sum_deadline_ok = 0.0
        sum_uplink_rate = 0.0
        sum_off_users = 0.0
        sum_reward = 0.0
        step_count = 0
        power_hist = np.zeros(4, dtype=np.float64)
        power_hist_steps = 0

        while not np.all(done):
            obs_batch = np.stack(obs).astype(np.float32)
            with torch.no_grad():
                actions, _, rnn_states = actor(obs_batch, rnn_states, masks, deterministic=DETERMINISTIC)

            action_idx = actions.cpu().numpy().astype(int).flatten()
            actions_env = np.eye(act_space.n, dtype=np.float32)[action_idx]
            next_obs, rewards, next_done, _ = env.step(actions_env)

            delay = env.delay_last[:env.M_sim, 0]
            omega = env.omega_last[:env.M_sim].astype(int)
            off_mask = omega != 0

            sum_delay_ms += float(np.mean(delay) * 1000.0)
            sum_deadline_ok += float(np.mean(delay <= env.tau_c))
            sum_off_users += float(np.count_nonzero(off_mask))
            if np.any(off_mask):
                sum_uplink_rate += float(np.mean(env.uplink_rate_access_b[off_mask, 0]) / 1e6)
            sum_reward += float(np.mean(rewards))

            if env.p_last is not None:
                p_level = env.P_max / 4.0
                idx = np.clip(np.rint(env.p_last[:env.M_sim] / max(p_level, 1e-12)).astype(int), 0, 3)
                cnt = np.bincount(idx, minlength=4)
                if cnt.sum() > 0:
                    power_hist += cnt / cnt.sum()
                    power_hist_steps += 1

            obs = next_obs
            done = np.array(next_done, dtype=bool)
            step_count += 1
            masks = np.ones((env.n_agents, 1), dtype=np.float32)

        denom = max(1, step_count)
        per_ep["avg_total_delay_ms"].append(sum_delay_ms / denom)
        per_ep["deadline_satisfaction_ratio"].append(sum_deadline_ok / denom)
        per_ep["avg_uplink_rate_Mbps"].append(sum_uplink_rate / denom)
        per_ep["avg_offloading_users"].append(sum_off_users / denom)
        per_ep["avg_reward"].append(sum_reward / denom)
        if power_hist_steps > 0:
            per_ep["power_dist"].append(power_hist / power_hist_steps)

    out = {
        "avg_total_delay_ms": float(np.mean(per_ep["avg_total_delay_ms"])),
        "deadline_satisfaction_ratio": float(np.mean(per_ep["deadline_satisfaction_ratio"])),
        "avg_uplink_rate_Mbps": float(np.mean(per_ep["avg_uplink_rate_Mbps"])),
        "avg_offloading_users": float(np.mean(per_ep["avg_offloading_users"])),
        "avg_reward": float(np.mean(per_ep["avg_reward"])),
        "power_dist": np.mean(np.stack(per_ep["power_dist"], axis=0), axis=0)
        if per_ep["power_dist"]
        else np.zeros(4, dtype=np.float64),
    }
    return out


def main():
    if DEVICE == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    warm_env = MA_UCMEC_stat_noncoop(render=False)
    actor, args, act_space = build_actor(warm_env, MODEL_PATH, device)

    all_res = []
    for seed in SEEDS:
        res = eval_one_seed(actor, args, act_space, int(seed))
        all_res.append(res)
        print(
            f"[seed {seed}] "
            f"delay={res['avg_total_delay_ms']:.4f} ms, "
            f"deadline_ok={res['deadline_satisfaction_ratio']:.4f}, "
            f"uplink={res['avg_uplink_rate_Mbps']:.4f} Mbps, "
            f"off_users={res['avg_offloading_users']:.4f}, "
            f"reward={res['avg_reward']:.4f}"
        )

    keys = [
        "avg_total_delay_ms",
        "deadline_satisfaction_ratio",
        "avg_uplink_rate_Mbps",
        "avg_offloading_users",
        "avg_reward",
    ]

    print("\nSummary over seeds (mean +/- std):")
    for k in keys:
        vals = np.array([x[k] for x in all_res], dtype=np.float64)
        print(f"  {k}: {vals.mean():.4f} +/- {vals.std():.4f}")

    p = np.stack([x["power_dist"] for x in all_res], axis=0)
    p_mean = p.mean(axis=0)
    p_std = p.std(axis=0)
    print("  power_dist (level0..3):")
    for i in range(4):
        print(f"    p{i}: {p_mean[i]:.4f} +/- {p_std[i]:.4f}")


if __name__ == "__main__":
    main()
