import os
import sys
import copy
from collections import defaultdict

import numpy as np
import torch

# Adjust these settings.
MODEL_PATH = r"C:\\DCNLab\\UCMEC\\UCMEC-mmWave-Fronthaul\\results\\smallEnv\\MyEnv\\rmappo\\noncoop_rnn\\IPPO_cluster5_interval20\\models\\actor_999.pt"
SEEDS = [3]
#SEEDS = [11, 2, 3, 4, 5, 6, 7, 8, 9, 10]
EPISODES = 1
USE_RECURRENT = True
METRIC_NAME = "avg_total_delay_ms"  # e.g. avg_total_delay_ms, deadline_satisfaction_ratio
METRIC_HIGHER_IS_BETTER = False
CLUSTER_SIZES = list(range(1, 11))
DEVICE = "cpu"

current_path = os.getcwd()
sys.path.append(os.path.join(current_path, "UCMEC-mmWave-Fronthaul"))

try:
    from envs.MA_UCMEC_dyna_noncoop_test import MA_UCMEC_dyna_noncoop
    from algorithms.algorithm.r_actor_critic import R_Actor
    from config import get_config
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def _build_actor():
    parser = get_config()
    args = parser.parse_args([])
    if USE_RECURRENT:
        args.use_recurrent_policy = True
        args.use_naive_recurrent_policy = False
    device = torch.device(DEVICE)
    # Build a dummy env to get spaces.
    env = MA_UCMEC_dyna_noncoop(render=False, seed=0)
    obs_space = env.observation_space[0]
    act_space = env.action_space[0]
    actor = R_Actor(args, obs_space, act_space, device)
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=device)
        actor.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    actor.eval()
    return actor, args, act_space


def _make_env(seed, cluster_size):
    env = MA_UCMEC_dyna_noncoop(render=False, seed=seed)
    env.cluster_size = int(cluster_size)
    return env


def _metric_better(a, b):
    return a > b if METRIC_HIGHER_IS_BETTER else a < b


def run_interval_compare():
    actor, args, act_space = _build_actor()

    overall_wins = defaultdict(int)
    overall_intervals = []
    overall_diff_list = []

    for seed in SEEDS:
        for ep in range(EPISODES):
            episode_seed = int(seed) + ep
            envs = []
            obs_list = []
            rnn_states_list = []
            masks_list = []
            done_list = []

            for cs in CLUSTER_SIZES:
                env = _make_env(episode_seed, cs)
                envs.append(env)
                obs = env.reset()
                obs_list.append(obs)
                rnn_states = np.zeros((env.n_agents, args.recurrent_N, args.hidden_size), dtype=np.float32)
                masks = np.ones((env.n_agents, 1), dtype=np.float32)
                rnn_states_list.append(rnn_states)
                masks_list.append(masks)
                done_list.append(False)

            interval_wins = []
            interval_index = 0
            max_steps = 200

            for step in range(max_steps):
                for idx, env in enumerate(envs):
                    if done_list[idx]:
                        continue
                    obs_batch = np.stack(obs_list[idx])
                    with torch.no_grad():
                        actions, _, rnn_states = actor(obs_batch, rnn_states_list[idx], masks_list[idx], deterministic=True)
                    action_indices = actions.cpu().numpy().flatten()
                    actions_env = np.eye(act_space.n)[action_indices]
                    next_obs, rewards, next_dones, infos = env.step(actions_env)

                    obs_list[idx] = next_obs
                    rnn_states_list[idx] = rnn_states

                    if np.all(next_dones):
                        done_list[idx] = True
                        masks_list[idx] = np.zeros((env.n_agents, 1), dtype=np.float32)
                    else:
                        masks_list[idx] = np.ones((env.n_agents, 1), dtype=np.float32)

                if all(done_list):
                    break

                if envs[0].step_num % envs[0].cluster_interval == 0:
                    # Compare cluster sizes on this interval.
                    interval_scores = []
                    for idx, env in enumerate(envs):
                        metrics = env.last_interval_metrics
                        if metrics is None:
                            score = None
                        else:
                            score = metrics.get(METRIC_NAME, None)
                        interval_scores.append(score)

                    valid_scores = [(cs, sc) for cs, sc in zip(CLUSTER_SIZES, interval_scores) if sc is not None]
                    if valid_scores:
                        best_cs, best_score = valid_scores[0]
                        second_score = None
                        second_cs = None
                        for cs, sc in valid_scores[1:]:
                            if _metric_better(sc, best_score):
                                second_cs, second_score = best_cs, best_score
                                best_cs, best_score = cs, sc
                            else:
                                if second_score is None or _metric_better(sc, second_score):
                                    second_cs, second_score = cs, sc
                        if second_score is None:
                            diff = 0.0
                        else:
                            diff = (best_score - second_score) if METRIC_HIGHER_IS_BETTER else (second_score - best_score)
                        overall_wins[best_cs] += 1
                        interval_wins.append((interval_index, best_cs, diff))
                        overall_intervals.append((episode_seed, interval_index, best_cs, diff))
                        overall_diff_list.append(float(diff))
                        print(f"Episode {ep} Interval {interval_index}: best cluster_size={best_cs}, diff={diff}")
                        interval_index += 1

            # Episode summary
            print("Episode summary")
            for cs in CLUSTER_SIZES:
                wins = sum(1 for _, w, _ in interval_wins if w == cs)
                print(f"cluster_size={cs} wins={wins}")
            for idx, cs, diff in interval_wins:
                print(f"interval {idx}: winner={cs}, diff={diff}")

    # Overall summary
    print("Overall summary")
    for cs in CLUSTER_SIZES:
        print(f"cluster_size={cs} wins={overall_wins[cs]}")
    if overall_wins:
        sorted_wins = sorted(overall_wins.items(), key=lambda x: x[1], reverse=True)
        best_cs, _ = sorted_wins[0]
        if overall_diff_list:
            diff_mean = float(np.mean(overall_diff_list))
            diff_std = float(np.std(overall_diff_list))
            diff_p95 = float(np.percentile(overall_diff_list, 95))
        else:
            diff_mean = 0.0
            diff_std = 0.0
            diff_p95 = 0.0
        print(f"Best cluster_size={best_cs}, diff_mean={diff_mean}, diff_std={diff_std}, diff_p95={diff_p95}")


if __name__ == "__main__":
    run_interval_compare()
