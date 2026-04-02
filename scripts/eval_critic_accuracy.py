"""Evaluate high-level critic accuracy: predicted value vs actual MC return.

Usage:
    conda run -n ucmec_fast python scripts/eval_critic_accuracy.py

This script:
1. Loads the trained high-level actor + critic from run2 checkpoint
2. Runs eval episodes mimicking the runner's hierarchical loop
3. At each high-level decision point, records:
   - Critic's predicted value (in normalized space)
   - The chosen action and its log-prob
   - The actual interval reward (from compute_interval_reward)
4. After each episode, computes MC returns (with gamma=0.904)
5. Reports: value prediction error by timestep, by combo, by return quartile
"""

import sys
import os
import json
import itertools
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.algorithm.high_policy import HighPolicy
from utils.valuenorm import ValueNorm


def make_high_args():
    """Reconstruct high_args matching run2 training command."""
    return argparse.Namespace(
        lr=3e-4,
        critic_lr=3e-4,
        hidden_size=128,
        layer_N=2,
        recurrent_N=1,
        use_orthogonal=True,
        use_ReLU=True,
        gain=0.01,
        use_naive_recurrent_policy=False,
        use_recurrent_policy=True,
        use_popart=False,
        use_valuenorm=True,
        use_set_encoder=True,
        use_high_peruser_credit=True,
        num_agents=10,
        ppo_epoch=5,
        num_mini_batch=10,
        clip_param=0.2,
        entropy_coef=0.01,
        value_loss_coef=1.0,
        max_grad_norm=10.0,
        data_chunk_length=5,
        use_gae=True,
        gamma=0.99 ** 10,  # ~0.904
        gae_lambda=0.95,
        use_proper_time_limits=False,
        opti_eps=1e-5,
        weight_decay=0,
        episode_length=20,
        use_feature_normalization=True,
        high_num_heads=4,
    )


def build_env(seed=42):
    from envs.MA_UCMEC_dyna_noncoop_hierarchical_peruser_hotspot_nlos_obs57_heurlow import (
        MA_UCMEC_dyna_noncoop_hierarchical_peruser as HierEnv,
    )
    return HierEnv(seed=seed)


def run_eval(model_dir, n_episodes=10, seeds=None, deterministic=True):
    if seeds is None:
        seeds = list(range(n_episodes))

    device = torch.device("cpu")
    high_args = make_high_args()
    gamma = high_args.gamma

    # Build temp env for spaces
    tmp_env = build_env(seed=0)
    high_obs_space = tmp_env.high_observation_space
    high_act_space = tmp_env.high_action_space
    M_sim = tmp_env.M_sim
    interval = 10
    episode_length = 200
    high_steps = episode_length // interval
    del tmp_env

    # Load policy
    policy = HighPolicy(high_args, high_obs_space, high_obs_space, high_act_space, device=device)
    policy.actor.load_state_dict(
        torch.load(os.path.join(model_dir, "actor_high.pt"), map_location=device)
    )
    policy.critic.load_state_dict(
        torch.load(os.path.join(model_dir, "critic_high.pt"), map_location=device)
    )
    policy.actor.eval()
    policy.critic.eval()

    # We build a ValueNorm and warm it up from eval rewards (pass 1),
    # then use it to denormalize critic predictions.
    # But a cleaner approach: collect normalized preds + raw returns,
    # then fit normalizer from the returns.

    all_records = []

    for ep_idx, seed in enumerate(seeds):
        env = build_env(seed=seed)
        env.reset()

        # RNN states: [batch, recurrent_N, hidden_size] — matches runner's squeeze(1)
        rnn_h = np.zeros((1, high_args.recurrent_N, high_args.hidden_size), dtype=np.float32)
        rnn_hc = np.zeros((1, high_args.recurrent_N, high_args.hidden_size), dtype=np.float32)
        masks = np.ones((1, 1), dtype=np.float32)  # [batch, 1] — after runner's squeeze(1)

        episode_records = []

        for h_step in range(high_steps):
            # Advance channel (mimics runner: advance_channel called every low step,
            # but for first step of interval it makes obs available)
            env.advance_channel()
            high_obs = env.get_global_obs()  # [M_sim, obs_dim]
            high_obs_t = torch.FloatTensor(high_obs).unsqueeze(0)  # [1, M_sim, obs_dim]

            with torch.no_grad():
                values_norm, actions, action_log_probs, rnn_h_new, rnn_hc_new = (
                    policy.get_actions(
                        high_obs_t,
                        high_obs_t,
                        torch.FloatTensor(rnn_h),
                        torch.FloatTensor(rnn_hc),
                        torch.FloatTensor(masks),
                        deterministic=deterministic,
                    )
                )

            values_norm_np = values_norm.cpu().numpy().reshape(M_sim)  # [M_sim]
            actions_np = actions.cpu().numpy().reshape(M_sim).astype(int)  # [M_sim]
            logp_np = action_log_probs.cpu().numpy().reshape(M_sim)  # [M_sim]

            rnn_h = rnn_h_new.cpu().numpy()
            rnn_hc = rnn_hc_new.cpu().numpy()

            # Apply high-level action
            env.set_high_action(actions_np)

            # Run interval low-level steps
            for low_step in range(interval):
                # advance_channel for steps after the first
                if low_step > 0:
                    env.advance_channel()
                dummy_action = [[0] for _ in range(env.agent_num)]
                env.step(dummy_action)

            # Get interval reward
            interval_reward = np.asarray(env.compute_interval_reward(), dtype=np.float64)  # [M_sim]

            episode_records.append({
                "h_step": h_step,
                "value_pred_norm": values_norm_np.tolist(),
                "action_ids": actions_np.tolist(),
                "action_log_probs": logp_np.tolist(),
                "interval_reward": interval_reward.tolist(),
            })

        # MC returns: return[t] = reward[t] + gamma * return[t+1]
        mc_returns = np.zeros((high_steps, M_sim), dtype=np.float64)
        mc_returns[-1] = np.array(episode_records[-1]["interval_reward"])
        for t in reversed(range(high_steps - 1)):
            mc_returns[t] = (
                np.array(episode_records[t]["interval_reward"]) + gamma * mc_returns[t + 1]
            )

        for t in range(high_steps):
            episode_records[t]["mc_return"] = mc_returns[t].tolist()
            episode_records[t]["seed"] = seed
            episode_records[t]["episode"] = ep_idx
            all_records.append(episode_records[t])

        mean_rew = np.mean([np.mean(r["interval_reward"]) for r in episode_records])
        print(f"Episode {ep_idx} (seed={seed}): mean_interval_reward={mean_rew:.4f}")

    # Fit ValueNorm from MC returns (matching training: r_mappo.py line 88
    # updates ValueNorm with return_batch, NOT single-step rewards).
    vnorm = ValueNorm((M_sim, 1), device=device)
    for rec in all_records:
        ret = torch.FloatTensor(np.array(rec["mc_return"]).reshape(1, M_sim, 1))
        vnorm.update(ret)

    mean_v, var_v = vnorm.running_mean_var()
    mean_np = mean_v.cpu().numpy().reshape(M_sim)
    std_np = np.sqrt(var_v.cpu().numpy().reshape(M_sim))

    print(f"\nValueNorm stats (fitted from MC returns, matching training convention):")
    print(f"  mean per user: {mean_np}")
    print(f"  std  per user: {std_np}")

    # Denormalize critic outputs and compute errors
    for rec in all_records:
        v_norm = np.array(rec["value_pred_norm"])
        v_denorm = v_norm * std_np + mean_np
        mc_ret = np.array(rec["mc_return"])
        rec["value_pred_denorm"] = v_denorm.tolist()
        rec["value_error"] = (v_denorm - mc_ret).tolist()
        rec["abs_error"] = np.abs(v_denorm - mc_ret).tolist()

    # Also report normalized-space comparison (avoids denorm bias entirely)
    print(f"\n--- Normalized-space comparison (no denorm needed) ---")
    all_v_norm = []
    all_ret_norm = []
    for rec in all_records:
        v_n = np.array(rec["value_pred_norm"])
        mc_r = np.array(rec["mc_return"])
        ret_normalized = (mc_r - mean_np) / std_np
        all_v_norm.extend(v_n.tolist())
        all_ret_norm.extend(ret_normalized.tolist())
        rec["mc_return_norm"] = ret_normalized.tolist()
        rec["value_error_norm"] = (v_n - ret_normalized).tolist()
        rec["abs_error_norm"] = np.abs(v_n - ret_normalized).tolist()
    all_v_norm = np.array(all_v_norm)
    all_ret_norm = np.array(all_ret_norm)
    norm_errors = all_v_norm - all_ret_norm
    corr_norm = np.corrcoef(all_v_norm, all_ret_norm)[0, 1] if len(all_v_norm) > 1 else 0.0
    ev_norm = 1.0 - np.var(norm_errors) / np.var(all_ret_norm) if np.var(all_ret_norm) > 1e-8 else 0.0
    print(f"  Critic (norm) : mean={np.mean(all_v_norm):.4f}, std={np.std(all_v_norm):.4f}")
    print(f"  Return (norm) : mean={np.mean(all_ret_norm):.4f}, std={np.std(all_ret_norm):.4f}")
    print(f"  Error (norm)  : mean={np.mean(norm_errors):.4f}, std={np.std(norm_errors):.4f}")
    print(f"  MAE (norm)    : {np.mean(np.abs(norm_errors)):.4f}")
    print(f"  Corr (norm)   : {corr_norm:.4f}")
    print(f"  Explained Var : {ev_norm:.4f}")

    return all_records


def analyze_results(records, M_sim=10):
    print("\n" + "=" * 70)
    print("CRITIC ACCURACY ANALYSIS")
    print("=" * 70)

    all_errors = np.array([e for r in records for e in r["value_error"]])
    all_abs_errors = np.array([e for r in records for e in r["abs_error"]])
    all_returns = np.array([e for r in records for e in r["mc_return"]])
    all_preds = np.array([e for r in records for e in r["value_pred_denorm"]])
    all_rewards = np.array([e for r in records for e in r["interval_reward"]])

    # 1. Overall
    n = len(all_errors)
    print(f"\n--- Overall ({n} data points: {len(records)} steps x {M_sim} users) ---")
    print(f"  MC Return    : mean={np.mean(all_returns):.4f}, std={np.std(all_returns):.4f}, "
          f"range=[{np.min(all_returns):.4f}, {np.max(all_returns):.4f}]")
    print(f"  Critic Pred  : mean={np.mean(all_preds):.4f}, std={np.std(all_preds):.4f}, "
          f"range=[{np.min(all_preds):.4f}, {np.max(all_preds):.4f}]")
    print(f"  Error (pred-ret): mean={np.mean(all_errors):.4f}, std={np.std(all_errors):.4f}")
    print(f"  MAE          : {np.mean(all_abs_errors):.4f}")
    ret_range = np.max(all_returns) - np.min(all_returns)
    if ret_range > 1e-8:
        nrmse = np.sqrt(np.mean(all_errors ** 2)) / ret_range
        print(f"  NRMSE        : {nrmse:.4f}")
    corr = np.corrcoef(all_returns, all_preds)[0, 1] if n > 1 else 0.0
    print(f"  Corr(pred, return): {corr:.4f}")

    # 2. By timestep
    print(f"\n--- By High-Level Timestep ---")
    print(f"  {'step':>4s} | {'MAE':>8s} | {'bias':>8s} | {'ret_mean':>9s} | {'pred_mean':>9s} | {'n':>4s}")
    for s in sorted(set(r["h_step"] for r in records)):
        recs = [r for r in records if r["h_step"] == s]
        errs = np.array([e for r in recs for e in r["value_error"]])
        abs_e = np.array([e for r in recs for e in r["abs_error"]])
        rets = np.array([e for r in recs for e in r["mc_return"]])
        preds = np.array([e for r in recs for e in r["value_pred_denorm"]])
        print(f"  {s:4d} | {np.mean(abs_e):8.4f} | {np.mean(errs):+8.4f} | "
              f"{np.mean(rets):9.4f} | {np.mean(preds):9.4f} | {len(errs):4d}")

    # 3. By action combo
    print(f"\n--- By Action Combo (top 15 by frequency) ---")
    combo_data = {}
    for rec in records:
        for u in range(M_sim):
            cid = rec["action_ids"][u]
            if cid not in combo_data:
                combo_data[cid] = {"errors": [], "abs_errors": [], "returns": [], "rewards": []}
            combo_data[cid]["errors"].append(rec["value_error"][u])
            combo_data[cid]["abs_errors"].append(rec["abs_error"][u])
            combo_data[cid]["returns"].append(rec["mc_return"][u])
            combo_data[cid]["rewards"].append(rec["interval_reward"][u])

    total_count = sum(len(v["errors"]) for v in combo_data.values())
    print(f"  {'combo':>5s} | {'count':>5s} | {'freq%':>5s} | {'MAE':>8s} | {'bias':>8s} | "
          f"{'rew_mean':>9s} | {'ret_mean':>9s}")
    sorted_combos = sorted(combo_data.keys(), key=lambda k: -len(combo_data[k]["errors"]))
    for cid in sorted_combos[:15]:
        d = combo_data[cid]
        cnt = len(d["errors"])
        freq = 100.0 * cnt / total_count
        print(f"  {cid:5d} | {cnt:5d} | {freq:5.1f} | {np.mean(d['abs_errors']):8.4f} | "
              f"{np.mean(d['errors']):+8.4f} | {np.mean(d['rewards']):9.4f} | "
              f"{np.mean(d['returns']):9.4f}")

    # 4. By return quartile
    print(f"\n--- By MC Return Quartile ---")
    q25, q50, q75 = np.percentile(all_returns, [25, 50, 75])
    quartiles = [
        ("Q1(worst)", all_returns <= q25),
        ("Q2", (all_returns > q25) & (all_returns <= q50)),
        ("Q3", (all_returns > q50) & (all_returns <= q75)),
        ("Q4(best)", all_returns > q75),
    ]
    print(f"  {'quartile':>10s} | {'n':>5s} | {'MAE':>8s} | {'bias':>8s} | "
          f"{'ret_range':>18s} | {'pred_range':>18s}")
    for label, mask in quartiles:
        if mask.sum() == 0:
            continue
        print(f"  {label:>10s} | {mask.sum():5d} | {np.mean(all_abs_errors[mask]):8.4f} | "
              f"{np.mean(all_errors[mask]):+8.4f} | "
              f"[{all_returns[mask].min():.3f},{all_returns[mask].max():.3f}] | "
              f"[{all_preds[mask].min():.3f},{all_preds[mask].max():.3f}]")

    # 5. Advantage quality (in normalized space — denorm-invariant)
    print(f"\n--- Advantage Quality (normalized space) ---")
    rank_corrs = []
    sign_agreements = []
    for rec in records:
        ret_n = np.array(rec["mc_return_norm"])
        v_n = np.array(rec["value_pred_norm"])
        advs = ret_n - v_n  # advantage in normalized space
        ret_centered = ret_n - np.mean(ret_n)
        if np.std(ret_n) > 1e-8 and np.std(advs) > 1e-8:
            rank_corrs.append(np.corrcoef(ret_n, advs)[0, 1])
        sign_agreements.append(np.mean(np.sign(ret_centered) == np.sign(advs)))

    if rank_corrs:
        print(f"  Per-step corr(return_norm, advantage): mean={np.mean(rank_corrs):.4f}, "
              f"std={np.std(rank_corrs):.4f}, min={np.min(rank_corrs):.4f}")
    print(f"  Sign agreement (above/below avg return vs adv sign): "
          f"mean={np.mean(sign_agreements):.4f}")

    # 6. Per-user
    print(f"\n--- Per-User Critic Accuracy ---")
    print(f"  {'user':>4s} | {'MAE':>8s} | {'bias':>8s} | {'ret_std':>8s} | {'pred_std':>8s} | {'corr':>6s}")
    for u in range(M_sim):
        u_err = np.array([r["value_error"][u] for r in records])
        u_abs = np.array([r["abs_error"][u] for r in records])
        u_ret = np.array([r["mc_return"][u] for r in records])
        u_pred = np.array([r["value_pred_denorm"][u] for r in records])
        c = np.corrcoef(u_ret, u_pred)[0, 1] if np.std(u_ret) > 1e-8 else 0.0
        print(f"  {u:4d} | {np.mean(u_abs):8.4f} | {np.mean(u_err):+8.4f} | "
              f"{np.std(u_ret):8.4f} | {np.std(u_pred):8.4f} | {c:6.3f}")

    # 7. Critic variance vs return variance
    print(f"\n--- Signal-to-Noise Summary ---")
    print(f"  Return std (across all):  {np.std(all_returns):.4f}")
    print(f"  Pred std (across all):    {np.std(all_preds):.4f}")
    print(f"  Error std:                {np.std(all_errors):.4f}")
    print(f"  Reward (single-step) std: {np.std(all_rewards):.4f}")
    if np.std(all_returns) > 1e-8:
        explained_var = 1.0 - np.var(all_errors) / np.var(all_returns)
        print(f"  Explained variance:       {explained_var:.4f}")


def main():
    model_dir = (
        "c:/DCNLab/UCMEC/UCMEC-mmWave-Fronthaul/results/"
        "hotspotEnv/nlos_cluster/rmappo/hierarchical_hotspot_heuristic/run2/models"
    )
    n_episodes = 10
    seeds = list(range(n_episodes))

    print(f"Model dir: {model_dir}")
    print(f"Episodes: {n_episodes}, Seeds: {seeds}")
    print(f"High gamma: {0.99**10:.6f}")
    print()

    records = run_eval(model_dir, n_episodes, seeds, deterministic=True)
    analyze_results(records)

    out_path = "c:/DCNLab/UCMEC/UCMEC-mmWave-Fronthaul/results/critic_accuracy_eval.json"
    with open(out_path, "w") as f:
        json.dump({"records": records}, f, indent=2)
    print(f"\nRaw data saved to {out_path}")


if __name__ == "__main__":
    main()
