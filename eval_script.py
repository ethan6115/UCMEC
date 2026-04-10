import torch
import numpy as np
import sys
import os
import copy
# 將 UCMEC-mmWave-Fronthaul 資料夾加入系統路徑，以確保能匯入 envs 和 algorithms
# 假設此腳本位於 UCMEC-mmWave-Fronthaul 資料夾的上一層或同層
current_path = os.getcwd()
sys.path.append(os.path.join(current_path, "UCMEC-mmWave-Fronthaul"))
# Toggle here to switch evaluation mode without CLI args.
USE_HIERARCHICAL = True
PER_USER = True
HIERARCHICAL_INTERVAL = 10
#HIERARCHICAL_INTERVAL = 10
USE_RECURRENT = True
DEBUG_HIGH_ACTION_PROBS = False  # Print Bernoulli bit probs / AP mapping at high-level decision steps.
USE_PIVOTAL_STATS = False
# High-level policy for hierarchical eval:
#   "trained": use MODEL_HIGH
#   "baseline_topk": always pick combo (0,1) in top-candidate list
HIGH_POLICY_MODE = "baseline_topk"  # "trained" | "baseline_topk" | "oracle" | "best_front"
BASELINE_TOPK_COMBO = (0, 1)

SEEDS = [18, 62, 53, 14, 58,
         161, 37, 3, 95, 150,
         11, 1, 29, 189, 198,
         153, 26, 59, 365, 84,
         946, 56, 99, 75, 263,
         776, 94, 71, 735, 64]

#SEEDS = [18, 62, 53, 14, 58]
#SEEDS = [99, 95, 58, 776, 153, 11, 84, 94, 189, 735] #win
def make_env(seed):
    if USE_HIERARCHICAL:
        if PER_USER:
            return MA_UCMEC_dyna_noncoop_hierarchical_peruser(render=True, seed=seed)
        else:
            return MA_UCMEC_dyna_noncoop_hierarchical_alluser(render=True, seed=seed)
    return MA_UCMEC_dyna_noncoop(render=True, seed=seed)
#IPPO
#MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\hotspotEnv\peruser_clustersize\rmappo\noncoop_rnn\hotspot_randcluster\models/actor_999.pt"
#MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\hotspotEnv\nlos_cluster\rmappo\noncoop_rnn\hotspot_cluster2\models/actor_999.pt"
#peruser
#MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\hotspotEnv\peruser_clustersize\rmappo\hierarchical_hotspot_v1\hotspot_hierarchical_peruser\models/actor_999.pt"
#MODEL_HIGH = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\hotspotEnv\peruser_clustersize\rmappo\hierarchical_hotspot_v1\hotspot_hierarchical_peruser\models/actor_high.pt"
#stageBC
#MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\hotspotEnv\peruser_clustersize\rmappo\hierarchical_hotspot_stageBC\run1/models/actor_999.pt"
#MODEL_HIGH = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\hotspotEnv\peruser_clustersize\rmappo\hierarchical_hotspot_stageBC\run1/models/actor_high.pt"

#nlos
MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\hotspotEnv\nlos_cluster\rmappo\noncoop_rnn\hotspot_cluster2_cpuobs8\models/actor_999.pt"
#nlos hierarchical peruser
#MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\hotspotEnv\nlos_cluster\rmappo\hierarchical_hotspot_v1\hotspot_hierarchical_peruser\models/actor_999.pt"
#MODEL_HIGH = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\hotspotEnv\nlos_cluster\rmappo\hierarchical_hotspot_v1\hotspot_hierarchical_peruser\models/actor_high.pt"
#hierarchical local
#MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\hotspotEnv\nlos_cluster\rmappo\hierarchical_hotspot_local\run1\models/actor_999.pt"
#MODEL_HIGH = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\hotspotEnv\nlos_cluster\rmappo\hierarchical_hotspot_local\run1\models/actor_high.pt"
#hierarchical heuristic low
#MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\hotspotEnv\nlos_cluster\rmappo\hierarchical_hotspot_heuristic\run2\models/actor_999.pt"
#MODEL_HIGH = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\hotspotEnv\nlos_cluster\rmappo\hierarchical_hotspot_heuristic\run2\models/actor_high.pt"
#MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\hotspotEnv\nlos_cluster\rmappo\hierarchical_hotspot_heuristic\run5\models/actor_499.pt"
#MODEL_HIGH = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\hotspotEnv\nlos_cluster\rmappo\hierarchical_hotspot_heuristic\run5\models/actor_high.pt"
#nlos noattn ablation
#MODEL_HIGH = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\hotspotEnv\nlos_cluster\rmappo\hierarchical_hotspot_heuristic_noattn_lp\run1\models/actor_high.pt"
#MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\hotspotEnv\nlos_cluster\rmappo\hierarchical_hotspot_noattn-mp_low\run2\models/actor_800.pt"
#MODEL_HIGH = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\hotspotEnv\nlos_cluster\rmappo\hierarchical_hotspot_heuristic_noattn-mp\run1\models/actor_high.pt"
# new actor
MODEL_HIGH = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\hotspotEnv\nlos_cluster\rmappo\hierarchical_hotspot_heuristic_pair_scorer\run2\models/actor_high.pt"

#stageBC
#MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\hotspotEnv\nlos_cluster\rmappo\hierarchical_hotspot_stageBC\run1/models/actor_700.pt"
#MODEL_HIGH = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\hotspotEnv\nlos_cluster\rmappo\hierarchical_hotspot_stageBC\run1/models/actor_high.pt"


try:
    #from envs.MA_UCMEC_dyna_noncoop import MA_UCMEC_dyna_noncoop
    #from envs.MA_UCMEC_dyna_noncoop_cluster_rand import MA_UCMEC_dyna_noncoop_cluster_rand as MA_UCMEC_dyna_noncoop
    from envs.MA_UCMEC_dyna_noncoop_big_3_2_nlos_cpuobs import MA_UCMEC_dyna_noncoop_big_3_2_nlos_cpuobs as MA_UCMEC_dyna_noncoop
    from envs.MA_UCMEC_dyna_coop import MA_UCMEC_dyna_coop
    from envs.MA_UCMEC_dyna_noncoop_hierarchical_alluser_front import MA_UCMEC_dyna_noncoop_hierarchical_alluser
    #from envs.MA_UCMEC_dyna_noncoop_hierarchical_alluser import MA_UCMEC_dyna_noncoop_hierarchical_alluser
    from envs.MA_UCMEC_dyna_noncoop_hierarchical_peruser_hotspot_nlos_obs57_testpara import MA_UCMEC_dyna_noncoop_hierarchical_peruser
    from algorithms.algorithm.r_actor_critic import R_Actor
    from algorithms.algorithm.high_actor_critic import HighActor
    from config import get_config
except ImportError as e:
    print("匯入模組失敗，請確認 'UCMEC-mmWave-Fronthaul' 資料夾是否在當前目錄下。")
    print(f"錯誤訊息: {e}")
    sys.exit(1)
# ── Oracle helpers ────────────────────────────────────────────────────────────
def _snapshot_env(env):
    """Capture env dynamic state for oracle rollback."""
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
def _restore_env(env, snap):
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
def _build_dummy_actions_env(act_space, n_agents):
    """Build a fixed one-hot low-level action batch.

    For heurlow heuristic-low eval, low actions are overridden by env logic,
    so any valid one-hot action is equivalent.
    """
    idx = np.zeros((n_agents,), dtype=np.int64)
    return np.eye(act_space.n, dtype=np.float32)[idx]


def _run_interval_sim(env, actor, act_space, obs, rnn_states, masks, n_steps):
    """Run n_steps with current env state, return mean total_delay_clip per user."""
    delay_sum = np.zeros(env.M_sim, dtype=np.float64)
    if actor is None:
        rnn_tmp = None
    elif isinstance(rnn_states, torch.Tensor):
        rnn_tmp = rnn_states.clone()
    else:
        rnn_tmp = rnn_states.copy()
    # Avoid mutating caller-side observation list reference across trials.
    sim_obs = [o.copy() for o in obs]
    for t in range(n_steps):
        if t > 0:
            env.advance_channel()
        if actor is None:
            actions_env = _build_dummy_actions_env(act_space, env.n_agents)
        else:
            obs_batch = np.stack(sim_obs)
            with torch.no_grad():
                actions_t, _, rnn_tmp = actor(obs_batch, rnn_tmp, masks, deterministic=True)
            action_indices = actions_t.cpu().numpy().flatten()
            actions_env = np.eye(act_space.n)[action_indices]
        sim_obs, _, _, _ = env.step(actions_env)
        delay_sum += env.delay_last_clip[:env.M_sim, 0]
    return delay_sum / max(1, n_steps)
def _oracle_search(env, actor, act_space, obs, rnn_states, masks, n_steps):
    """Greedy per-user oracle: for each user, try all combos (others fixed),
    pick the combo minimising all-user mean total_delay_clip.
    Same logic as oracle_apselect_interval.py:
      - Start from baseline combo 0 (top-2 by beta) for all users
      - Iterate users: try all C(n,k) combos for that user, keep others fixed
      - Score = all-user mean total_delay (not just front, not just that user)
      - Greedy: once a user's best combo is found, fix it for subsequent users
    """
    n_combos = len(env._ap_combos)
    snap = _snapshot_env(env)
    # Start from baseline (combo 0 = top-2 by beta) for all users
    current_combo = np.zeros(env.M_sim, dtype=np.int32)
    for user_i in range(env.M_sim):
        best_score = float("inf")
        best_combo_idx = current_combo[user_i]
        for combo_idx in range(n_combos):
            trial_combo = current_combo.copy()
            trial_combo[user_i] = combo_idx
            _restore_env(env, snap)
            env.apply_high_action(trial_combo)
            avg_delay = _run_interval_sim(
                env, actor, act_space, obs, rnn_states, masks, n_steps
            )
            # Score = all-user mean total_delay_clip (跟主指標一致，避免犧牲其他 user)
            score = float(np.mean(avg_delay))
            if score < best_score:
                best_score = score
                best_combo_idx = combo_idx
        current_combo[user_i] = best_combo_idx
    # Restore env to original state
    _restore_env(env, snap)
    return current_combo

# ── Best-fronthaul heuristic ─────────────────────────────────────────────────
def _best_front_action(env):
    """Per-user heuristic: pick the AP combo with the best fronthaul quality.

    For each user, for each of the C(n,k) combos, compute
    max_cpu(bottleneck_front_quality) and select the combo that maximises it.
    Uses the same pathloss formula as env._compute_cpu_front_quality().
    No simulation rollout needed — purely channel-based.
    """
    n_combos = len(env._ap_combos)
    lo, hi = env.front_db_clip
    best_combo = np.zeros(env.M_sim, dtype=np.int32)
    for i in range(env.M_sim):
        best_score = -np.inf
        for combo_idx in range(n_combos):
            combo = env._ap_combos[combo_idx]
            ap_idx = env._top10_ap_idx[i][list(combo)]
            # Compute bottleneck fronthaul quality for each CPU
            max_cpu_quality = -np.inf
            for cpu in range(env.K):
                pl_values = []
                for ap in ap_idx:
                    dist = max(env.distance_matrix_front[ap, cpu], 1e-6)
                    alpha = env.alpha_los if env.link_type[ap, cpu] == 0 else env.alpha_nlos
                    g = max(env.G[ap, cpu], 1e-12)
                    pl = g * pow(dist, -alpha)
                    pl_values.append(pl)
                min_pl_db = 10.0 * np.log10(min(pl_values) + 1e-12)
                quality = float(np.clip((min_pl_db - lo) / (hi - lo), 0.0, 1.0))
                if quality > max_cpu_quality:
                    max_cpu_quality = quality
            if max_cpu_quality > best_score:
                best_score = max_cpu_quality
                best_combo[i] = combo_idx
    return best_combo

def evaluate(model_path):
    # 1. 取得設定參數 (Arguments)
    # 使用 config.py 中的預設參數
    parser = get_config()
    # 如果訓練參數有大幅修改，在這裡透過參數覆蓋，或是確保 config.py 是正確的
    args = parser.parse_args([])
    high_args = copy.deepcopy(args)
    high_args.hidden_size = 128
    # Reason: match evaluation actor to recurrent checkpoint if needed.
    if USE_RECURRENT:
        args.use_recurrent_policy = True
        args.use_naive_recurrent_policy = False
        high_args.use_recurrent_policy = True
        high_args.use_naive_recurrent_policy = False
    high_args.use_set_encoder = True
    
    # 設定使用 CPU 進行推論 (除非你有 GPU 且想用)
    device = torch.device("cpu")
    
    # 2. 建立環境
    print("正在初始化環境...")
    if USE_HIERARCHICAL:
        if PER_USER:
            env = MA_UCMEC_dyna_noncoop_hierarchical_peruser(render=True)
        else:
            env = MA_UCMEC_dyna_noncoop_hierarchical_alluser(render=True)
    else:
        env = MA_UCMEC_dyna_noncoop(render=True) # render=True 可能不會顯示畫面，視環境實作而定
    # 3. 初始化 Actor 網路
    # 取得單一 Agent 的 observation 和 action space
    # 根據環境程式碼，observation_space 是 Tuple(Box(...), ...)
    obs_space = env.observation_space[0] 
    act_space = env.action_space[0]
    
    use_dummy_low_policy = bool(
        USE_HIERARCHICAL
        and PER_USER
        and hasattr(env, "low_heuristic_only")
        and getattr(env, "low_heuristic_only", False)
        and getattr(env, "low_heuristic_cpu", False)
        and getattr(env, "low_heuristic_power", "") == "max"
    )

    actor = None
    if use_dummy_low_policy:
        print("[eval] Heurlow heuristic-low detected: skip loading low model and use fixed dummy low actions.")
    else:
        actor = R_Actor(args, obs_space, act_space, device)

        # 4. 載入訓練好的模型權重
        print(f"Loading model: {model_path}")
        if os.path.exists(model_path):
            # load state_dict
            state_dict = torch.load(model_path, map_location=device)
            actor.load_state_dict(state_dict)
            print("Model loaded.")
        else:
            print(f"Error: model not found {model_path}")
            return
        # 切換到評估模式
        actor.eval()
    high_actor = None
    if USE_HIERARCHICAL:
        if HIGH_POLICY_MODE == "trained":
            high_obs_space = env.high_observation_space
            high_act_space = env.high_action_space
            print(f"Loading model: {MODEL_HIGH}")
            if os.path.exists(MODEL_HIGH):
                state_dict = torch.load(MODEL_HIGH, map_location=device)
                if PER_USER:
                    ckpt_keys = list(state_dict.keys())

                    # Auto-match high actor type for backward compatibility:
                    #   - old/new MLP heads
                    #   - pair_scorer heads
                    has_pair_scorer = any(
                        k.startswith("ap_mlp.")
                        or k.startswith("scorer.")
                        or k.startswith("g_pre.")
                        or k.startswith("_ap_indices")
                        for k in ckpt_keys
                    )
                    inferred_actor_type = "pair_scorer" if has_pair_scorer else "mlp"
                    configured_actor_type = getattr(high_args, "high_actor_type", "mlp")
                    if configured_actor_type != inferred_actor_type:
                        print(
                            f"[eval] Override high_actor_type: {configured_actor_type} -> "
                            f"{inferred_actor_type} (from checkpoint keys)."
                        )
                    high_args.high_actor_type = inferred_actor_type

                    # Infer high hidden size from checkpoint RNN weights if available.
                    # Keeps eval compatible with runs trained using different high_hidden_size.
                    inferred_hidden_size = None
                    if "rnn.rnn.weight_hh_l0" in state_dict:
                        inferred_hidden_size = int(state_dict["rnn.rnn.weight_hh_l0"].shape[1])
                    elif "g_pre.weight" in state_dict:
                        inferred_hidden_size = int(state_dict["g_pre.weight"].shape[0])
                    elif "logits.weight" in state_dict:
                        inferred_hidden_size = int(state_dict["logits.weight"].shape[1])
                    if inferred_hidden_size is not None:
                        if int(high_args.hidden_size) != inferred_hidden_size:
                            print(
                                f"[eval] Override high hidden_size: {high_args.hidden_size} -> "
                                f"{inferred_hidden_size} (from checkpoint shape)."
                            )
                        high_args.hidden_size = inferred_hidden_size

                    # Keep pair_scorer combo construction aligned with env.
                    if hasattr(env, "candidate_n"):
                        high_args.candidate_n = int(env.candidate_n)
                    if hasattr(env, "k_fixed"):
                        high_args.k_fixed = int(env.k_fixed)
                    if hasattr(env, "K"):
                        high_args.num_cpus = int(env.K)

                    # MLP-only encoder type auto-match (noattn / set ablations).
                    if inferred_actor_type == "mlp":
                        has_attn = any(k.startswith("encoder.attn.") or k.startswith("encoder.attn_norm.") for k in ckpt_keys)
                        has_pool = any(k.startswith("encoder.pool_score.") for k in ckpt_keys)
                        inferred_encoder_type = "set" if has_attn else ("noattn-lp" if has_pool else "noattn-mp")
                        configured_encoder_type = getattr(high_args, "high_encoder_type", "set")
                        if configured_encoder_type != inferred_encoder_type:
                            print(
                                f"[eval] Override high_encoder_type: {configured_encoder_type} -> "
                                f"{inferred_encoder_type} (from checkpoint keys)."
                            )
                        high_args.high_encoder_type = inferred_encoder_type

                    high_actor = HighActor(high_args, high_obs_space, high_act_space, device)
                else:
                    high_actor = R_Actor(high_args, high_obs_space, high_act_space, device)
                high_actor.load_state_dict(state_dict)
                print("Model loaded.")
            else:
                print(f"Error: model not found {MODEL_HIGH}")
                return
            high_actor.eval()
        elif HIGH_POLICY_MODE == "baseline_topk":
            print("[eval] HIGH_POLICY_MODE=baseline_topk, skip loading MODEL_HIGH.")
        elif HIGH_POLICY_MODE == "oracle":
            print(f"[eval] HIGH_POLICY_MODE=oracle, exhaustive C({env.candidate_n},{env.k_fixed})={len(env._ap_combos)} combos per interval.")
        elif HIGH_POLICY_MODE == "best_front":
            print("[eval] HIGH_POLICY_MODE=best_front, select AP pair with best fronthaul quality per user.")
        else:
            raise ValueError(f"Unsupported HIGH_POLICY_MODE: {HIGH_POLICY_MODE}")
    # 5. Evaluate with multiple seeds
    num_episodes = 1
    seed_results = {
        "avg_total_delay_ms": [],
        "avg_local_delay_ms": [],
        "avg_uplink_delay_ms": [],
        "total_delay_ms_max": [],
        "total_delay_ms_p95": [],
        "uplink_delay_ms_max": [],
        "uplink_delay_ms_p95": [],
        "front_delay_ms_max": [],
        "front_delay_ms_p95": [],
        "avg_front_delay_ms": [],
        "avg_front_rate_Mbps": [],
        "front_rate_Mbps_max": [],
        "front_rate_Mbps_p95": [],
        "avg_actual_process_delay_ms": [],
        "avg_uplink_rate_Mbps": [],
        "avg_offloading_users": [],
        "deadline_satisfaction_ratio": [],
        "offloading_deadline_satisfaction_ratio": [],
        "p_front_bottleneck_offload": [],
        "p_front_bottleneck_offload_fail": [],
        "fail_bottleneck_uplink_ratio": [],
        "fail_bottleneck_front_ratio": [],
        "fail_bottleneck_process_ratio": [],
        "power_dist": [],
        "cpu_select_ratio": [],
        "cpu_select_entropy": [],
        "avg_cpu_used_per_step": [],
        "front_log_cap_hit_ratio": [],
    }
    if USE_HIERARCHICAL and PER_USER:
        seed_results["high_combo_hist"] = []           # combo index distribution
        seed_results["high_combo_mean_best_cfq"] = []  # avg max(cpu_front_quality) per decision
        seed_results["high_cpu_match_rate"] = []        # low-level CPU matches best cfq CPU
        seed_results["high_action_entropy_norm"] = []   # mean normalized entropy over high decisions
    if USE_PIVOTAL_STATS:
        seed_results.update({
            "pivotal_uplink": [],
            "pivotal_front": [],
            "pivotal_both": [],
            "pivotal_none": [],
            "pivotal_viol_count": [],
            "offloading_violation_rate": [],
        })
    # accumulators for combo-action diagnostic
    _diag_best_cfq_all = []   # per high-decision: mean max(cpu_front_quality)
    _diag_front_delay_all = []  # per seed: avg front_delay
    for seed in SEEDS:
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = make_env(seed)
        env.seed(seed)
        allsum_avg_total_delay = 0.0
        allsum_avg_local_delay = 0.0
        allsum_avg_uplink_delay = 0.0
        allsum_total_delay_max = 0.0
        allsum_total_delay_p95 = 0.0
        allsum_uplink_delay_max = 0.0
        allsum_uplink_delay_p95 = 0.0
        allsum_front_delay_max = 0.0
        allsum_front_delay_p95 = 0.0
        allsum_avg_front_delay = 0.0
        allsum_avg_front_rate = 0.0
        allsum_front_rate_max = 0.0
        allsum_front_rate_p95 = 0.0
        allsum_avg_actual_process_delay = 0.0
        allsum_avg_uplink_rate = 0.0
        allsum_num_offloading_users = 0.0
        allsum_deadline_satisfaction_ratio = 0.0
        allsum_offloading_deadline_satisfaction_ratio = 0.0
        allsum_offload_count = 0
        allsum_offload_fail_count = 0
        allsum_front_bottleneck_offload_count = 0
        allsum_front_bottleneck_offload_fail_count = 0
        allsum_fail_bottleneck_uplink_count = 0
        allsum_fail_bottleneck_front_count = 0
        allsum_fail_bottleneck_process_count = 0
        allsum_power_dist = np.zeros(4, dtype=np.float64)
        allsum_cpu_counts = None
        allsum_offload = 0
        allsum_cpu_used_steps = 0.0
        allsum_cpu_used_count = 0
        allsum_high_combo_hist = None
        allsum_best_cfq_sum = 0.0
        allsum_best_cfq_count = 0
        allsum_cpu_match_sum = 0.0
        allsum_cpu_match_count = 0
        allsum_high_action_entropy_sum = 0.0
        allsum_high_action_entropy_count = 0
        allsum_front_log_cap_hit_ratio = 0.0
        pivotal_off_count = 0
        pivotal_viol_count = 0
        pivotal_u_only_count = 0
        pivotal_f_only_count = 0
        pivotal_either_count = 0
        pivotal_need_both_count = 0
        pivotal_other_count = 0
        pivotal_u_count = 0
        pivotal_f_count = 0
        pivotal_both_count = 0
        pivotal_none_count = 0
        for _ in range(num_episodes):
            obs = env.reset()
            if use_dummy_low_policy:
                rnn_states = None
                masks = None
            else:
                rnn_states = np.zeros((env.n_agents, args.recurrent_N, args.hidden_size), dtype=np.float32)
                masks = np.ones((env.n_agents, 1), dtype=np.float32)
            sum_avg_total_delay = 0.0
            sum_avg_local_delay = 0.0
            sum_avg_uplink_delay = 0.0
            sum_total_delay_max = 0.0
            sum_total_delay_p95 = 0.0
            sum_uplink_delay_max = 0.0
            sum_uplink_delay_p95 = 0.0
            sum_front_delay_max = 0.0
            sum_front_delay_p95 = 0.0
            sum_avg_front_delay = 0.0
            sum_avg_front_rate = 0.0
            sum_front_rate_max = 0.0
            sum_front_rate_p95 = 0.0
            sum_avg_actual_process_delay = 0.0
            sum_avg_uplink_rate = 0.0
            sum_num_offloading_users = 0.0
            sum_deadline_satisfaction_ratio = 0.0
            sum_agent_steps = 0
            sum_offloading_deadline_satisfaction_ratio = 0.0
            sum_offload_steps = 0
            sum_offload_count = 0
            sum_offload_fail_count = 0
            sum_front_bottleneck_offload_count = 0
            sum_front_bottleneck_offload_fail_count = 0
            sum_fail_bottleneck_uplink_count = 0
            sum_fail_bottleneck_front_count = 0
            sum_fail_bottleneck_process_count = 0
            sum_power_dist = np.zeros(4, dtype=np.float64)
            metric_steps = 0
            dist_steps = 0
            cpu_counts = None
            offload_counts = 0
            cpu_used_sum = 0.0
            cpu_used_steps = 0
            interval_pivotal_off_count = 0
            interval_pivotal_viol_count = 0
            interval_pivotal_u_only_count = 0
            interval_pivotal_f_only_count = 0
            interval_pivotal_either_count = 0
            interval_pivotal_need_both_count = 0
            interval_pivotal_other_count = 0
            sum_high_combo_hist = None
            sum_best_cfq_sum = 0.0
            sum_best_cfq_count = 0
            sum_cpu_match_sum = 0.0
            sum_cpu_match_count = 0
            sum_high_action_entropy_sum = 0.0
            sum_high_action_entropy_count = 0
            sum_front_log_cap_hit_ratio = 0.0
            dones = [False] * env.n_agents
            if USE_HIERARCHICAL:
                if HIGH_POLICY_MODE == "trained":
                    high_rnn_states = np.zeros((1, high_args.recurrent_N, high_args.hidden_size), dtype=np.float32)
                    high_masks = np.ones((1, 1), dtype=np.float32)
                attn_max_list = []
                attn_entropy_list = []
                prev_high_global_obs_dbg = None
            step_count = 0
            while not all(dones):
                if USE_HIERARCHICAL and hasattr(env, "advance_channel"):
                    env.advance_channel()
                if USE_HIERARCHICAL and (step_count % HIERARCHICAL_INTERVAL == 0):
                    global_obs = env.get_global_obs()
                    global_obs = np.expand_dims(global_obs, axis=0)
                    if HIGH_POLICY_MODE == "trained":
                        if isinstance(high_rnn_states, torch.Tensor):
                            high_rnn_states_dbg = high_rnn_states.clone()
                        else:
                            high_rnn_states_dbg = high_rnn_states.copy()
                        with torch.no_grad():
                            high_action, _, high_rnn_states = high_actor(
                                global_obs, high_rnn_states, high_masks, deterministic=True
                            )
                        if hasattr(high_actor, "encoder"):
                            enc = high_actor.encoder
                            if getattr(enc, "last_attn_max", None) is not None:
                                attn_max_list.append(enc.last_attn_max)
                            if getattr(enc, "last_attn_entropy", None) is not None:
                                attn_entropy_list.append(enc.last_attn_entropy)
                        if PER_USER:
                            high_action = high_action.cpu().numpy().astype(int).squeeze(0)
                        else:
                            high_action = int(high_action.cpu().numpy().flatten()[0])
                    elif HIGH_POLICY_MODE == "oracle":
                        # Myopic oracle: evaluate on current channel state only (1 step).
                        # No future lookahead — same info available to a learned policy.
                        high_action = _oracle_search(
                            env, actor, act_space, obs, rnn_states, masks, n_steps=1
                        )
                    elif HIGH_POLICY_MODE == "best_front":
                        # Per-user heuristic: pick AP pair with best fronthaul quality.
                        high_action = _best_front_action(env)
                    else:
                        # baseline_topk: fixed combo among top candidates, same for all users
                        if hasattr(env, "_ap_combos"):
                            try:
                                baseline_combo_idx = int(env._ap_combos.index(BASELINE_TOPK_COMBO))
                            except ValueError:
                                baseline_combo_idx = 0
                        else:
                            baseline_combo_idx = 0
                        if PER_USER:
                            high_action = np.full((env.M_sim,), baseline_combo_idx, dtype=np.int32)
                        else:
                            high_action = baseline_combo_idx
                    if USE_HIERARCHICAL and PER_USER:
                        high_action_np = np.asarray(high_action, dtype=np.int32).reshape(-1)
                        # combo action: index 0..C(n,k)-1 maps to AP combo
                        n_combos = len(env._ap_combos) if hasattr(env, "_ap_combos") else int(env.high_action_space.nvec[0])
                        high_action_np = np.clip(high_action_np, 0, n_combos - 1)
                        if sum_high_combo_hist is None:
                            sum_high_combo_hist = np.zeros(n_combos, dtype=np.float64)
                        action_hist = np.bincount(high_action_np, minlength=n_combos).astype(np.float64)
                        sum_high_combo_hist += action_hist
                        # Per high-step action entropy across users (normalized to [0,1]).
                        probs = action_hist / max(1.0, action_hist.sum())
                        nz = probs > 0
                        if n_combos > 1 and np.any(nz):
                            ent_norm = float(-np.sum(probs[nz] * np.log(probs[nz])) / np.log(n_combos))
                        else:
                            ent_norm = 0.0
                        sum_high_action_entropy_sum += ent_norm
                        sum_high_action_entropy_count += 1
                    # ===== DEBUG_HIGH_ACTION_PROBS BEGIN (safe to delete this whole block later) =====
                    if USE_HIERARCHICAL and PER_USER and DEBUG_HIGH_ACTION_PROBS and hasattr(high_actor, "encoder"):
                        print(f"[HIGH-DEBUG] step={step_count}")
                        with torch.no_grad():
                            obs_t = torch.as_tensor(global_obs, dtype=torch.float32)
                            rnn_t = torch.as_tensor(high_rnn_states_dbg, dtype=torch.float32)
                            mask_t = torch.as_tensor(high_masks, dtype=torch.float32)
                            h_ctx_dbg, g_dbg = high_actor.encoder(obs_t)
                            if high_args.use_recurrent_policy or high_args.use_naive_recurrent_policy:
                                g_dbg, _ = high_actor.rnn(g_dbg, rnn_t, mask_t)
                            g_expand_dbg = g_dbg.unsqueeze(1).expand_as(h_ctx_dbg)
                            actor_in_dbg = torch.cat([h_ctx_dbg, g_expand_dbg], dim=-1)
                            flat_dbg = actor_in_dbg.reshape(-1, actor_in_dbg.shape[-1])
                            feat_dbg = high_actor.actor_mlp(flat_dbg)
                            logits_dbg = high_actor.logits(feat_dbg).reshape(obs_t.shape[0], obs_t.shape[1], -1)
                            bit_probs = torch.sigmoid(logits_dbg).cpu().numpy().squeeze(0)  # [M_sim, 10]
                            logits_np = logits_dbg.cpu().numpy().squeeze(0)  # [M_sim, 10]
                            print("[HIGH-DEBUG] logits mean/std:", float(logits_np.mean()), float(logits_np.std()))
                            print("[HIGH-DEBUG] logits user0:", np.round(logits_np[0], 3))
                            if env.M_sim > 1:
                                print("[HIGH-DEBUG] logits user1:", np.round(logits_np[1], 3))
                        
                        print(
                            "[HIGH-DEBUG] obs diff u0-u1 max:",
                            float(np.max(np.abs(global_obs[0, 0] - global_obs[0, 1])))
                            if env.M_sim > 1 else 0.0,
                        )
                        if prev_high_global_obs_dbg is not None:
                            print(
                                "[HIGH-DEBUG] obs diff prev-high-step max:",
                                float(np.max(np.abs(global_obs - prev_high_global_obs_dbg))),
                            )
                        prev_high_global_obs_dbg = global_obs.copy()
                        print("[HIGH-DEBUG] bit_probs user0:", np.round(bit_probs[0], 3))
                        if env.M_sim > 1:
                            print("[HIGH-DEBUG] bit_probs user1:", np.round(bit_probs[1], 3))
                        print("[HIGH-DEBUG] bit_probs mean:", np.round(bit_probs.mean(axis=0), 3))
                        print("[HIGH-DEBUG] probs std:", float(bit_probs.std()))
                        print("[HIGH-DEBUG] action user0:", high_action[0])
                        if env.M_sim > 1:
                            print("[HIGH-DEBUG] action user1:", high_action[1])
                        if hasattr(env, "_top10_ap_idx") and env._top10_ap_idx is not None:
                            for u in range(min(2, env.M_sim)):
                                top10 = env._top10_ap_idx[u]
                                mask_u = high_action[u].astype(bool)
                                if not np.any(mask_u):
                                    # apply_high_action() fallback: choose top-1 AP when all-zero
                                    selected = np.array([top10[0]], dtype=np.int32)
                                else:
                                    selected = top10[mask_u]
                                print(f"[HIGH-DEBUG] user{u} top10:", top10.tolist())
                                print(f"[HIGH-DEBUG] user{u} selected_APs:", selected.tolist())
                    # ===== DEBUG_HIGH_ACTION_PROBS END =====
                    if hasattr(env, "apply_high_action"):
                        env.apply_high_action(high_action)
                    else:
                        env.set_high_action(high_action)
                    # --- combo quality diagnostic: best cpu_front_quality ---
                    # Must be AFTER apply_high_action() so cpu_front_quality reflects current interval
                    if USE_HIERARCHICAL and PER_USER:
                        if hasattr(env, "cpu_front_quality") and env.cpu_front_quality is not None:
                            best_cfq = np.max(env.cpu_front_quality, axis=1)  # (M_sim,)
                            sum_best_cfq_sum += float(np.mean(best_cfq))
                            sum_best_cfq_count += 1
                            _diag_best_cfq_all.append(float(np.mean(best_cfq)))
                if use_dummy_low_policy:
                    actions_env = _build_dummy_actions_env(act_space, env.n_agents)
                else:
                    obs_batch = np.stack(obs)
                    with torch.no_grad():
                        actions, _, rnn_states = actor(obs_batch, rnn_states, masks, deterministic=True)
                    action_indices = actions.cpu().numpy().flatten()
                    actions_env = np.eye(act_space.n)[action_indices]
                next_obs, rewards, next_dones, infos = env.step(actions_env)
                if USE_PIVOTAL_STATS:
                    if (
                        hasattr(env, "delay_last")
                        and hasattr(env, "omega_last")
                        and hasattr(env, "uplink_rate_access_b")
                        and hasattr(env, "front_delay_last")
                    ):
                        delay_last = env.delay_last
                        omega_last = env.omega_last
                        rates = env.uplink_rate_access_b
                        front_delay = env.front_delay_last
                        if (
                            delay_last is not None
                            and omega_last is not None
                            and rates is not None
                            and front_delay is not None
                        ):
                            offload_mask = omega_last[:env.M_sim] != 0
                            if np.any(offload_mask):
                                T = delay_last[:env.M_sim, 0]
                                task = env.Task_size[0, :env.M_sim]
                                rate_off = rates[offload_mask, 0]
                                task_off = task[offload_mask]
                                U = task_off / np.maximum(rate_off, 1e-12)
                                F = front_delay[:env.M_sim, 0][offload_mask]
                                T_off = T[offload_mask]
                                D = env.tau_c
                                viol = T_off > D
                                interval_pivotal_viol_count += int(viol.sum())
                                u_ok = (T_off - U <= D)
                                f_ok = (T_off - F <= D)
                                uf_ok = (T_off - U - F <= D)
                                u_only = viol & u_ok & ~f_ok
                                f_only = viol & f_ok & ~u_ok
                                either = viol & u_ok & f_ok
                                need_both = viol & ~u_ok & ~f_ok & uf_ok
                                other = viol & ~uf_ok
                                if USE_PIVOTAL_STATS:
                                    assert int(viol.sum()) == int(
                                        u_only.sum()
                                        + f_only.sum()
                                        + either.sum()
                                        + need_both.sum()
                                        + other.sum()
                                    )
                                interval_pivotal_off_count += int(offload_mask.sum())
                                interval_pivotal_u_only_count += int(u_only.sum())
                                interval_pivotal_f_only_count += int(f_only.sum())
                                interval_pivotal_either_count += int(either.sum())
                                interval_pivotal_need_both_count += int(need_both.sum())
                                interval_pivotal_other_count += int(other.sum())
                if USE_PIVOTAL_STATS and (
                    ((step_count + 1) % HIERARCHICAL_INTERVAL == 0) or all(next_dones)
                ):
                    pivotal_off_count += interval_pivotal_off_count
                    pivotal_viol_count += interval_pivotal_viol_count
                    pivotal_u_only_count += interval_pivotal_u_only_count
                    pivotal_f_only_count += interval_pivotal_f_only_count
                    pivotal_either_count += interval_pivotal_either_count
                    pivotal_need_both_count += interval_pivotal_need_both_count
                    pivotal_other_count += interval_pivotal_other_count
                    pivotal_u_count += interval_pivotal_u_only_count + interval_pivotal_either_count
                    pivotal_f_count += interval_pivotal_f_only_count + interval_pivotal_either_count
                    pivotal_both_count += interval_pivotal_need_both_count
                    pivotal_none_count += interval_pivotal_other_count
                    interval_pivotal_off_count = 0
                    interval_pivotal_viol_count = 0
                    interval_pivotal_u_only_count = 0
                    interval_pivotal_f_only_count = 0
                    interval_pivotal_either_count = 0
                    interval_pivotal_need_both_count = 0
                    interval_pivotal_other_count = 0
                if hasattr(env, "omega_last"):
                    if cpu_counts is None:
                        cpu_counts = np.zeros(env.K, dtype=np.int64)
                    omega = env.omega_last.astype(int)
                    offload_mask = omega > 0
                    offload_counts += int(offload_mask.sum())
                    for k in range(env.K):
                        cpu_counts[k] += int(np.sum(omega == (k + 1)))
                    if np.any(offload_mask):
                        used_cpus = np.unique(omega[offload_mask])
                        cpu_used_sum += float(used_cpus.size)
                        cpu_used_steps += 1
                    # CPU match rate: does low-level pick best-cfq CPU?
                    if hasattr(env, "cpu_front_quality") and env.cpu_front_quality is not None:
                        cfq = env.cpu_front_quality
                        best_cpu = np.argmax(cfq, axis=1)  # 0-indexed
                        chosen_cpu = omega[:env.M_sim] - 1  # omega=1→0, 2→1, 3→2
                        off_mask = omega[:env.M_sim] > 0
                        if np.any(off_mask):
                            sum_cpu_match_sum += float(np.sum(best_cpu[off_mask] == chosen_cpu[off_mask]))
                            sum_cpu_match_count += int(off_mask.sum())
                if isinstance(infos, (list, tuple)) and len(infos) > 0 and isinstance(infos[0], dict):
                    info0 = infos[0]
                    if "avg_total_delay_ms" in info0:
                        sum_avg_total_delay += info0["avg_total_delay_ms"]
                        sum_avg_local_delay += info0["avg_local_delay_ms"]
                        sum_avg_uplink_delay += info0["avg_uplink_delay_ms"]
                        sum_avg_front_delay += info0["avg_front_delay_ms"]
                        sum_avg_actual_process_delay += info0["avg_actual_process_delay_ms"]
                        sum_avg_uplink_rate += info0["avg_uplink_rate_Mbps"]
                        sum_num_offloading_users += info0["num_offloading_users"]
                        sum_front_log_cap_hit_ratio += float(info0.get("front_log_cap_hit_ratio", 0.0))
                        # Compute uplink delay stats from env internals for this step.
                        if hasattr(env, "uplink_rate_access_b") and hasattr(env, "omega_last"):
                            rates = env.uplink_rate_access_b
                            omega = env.omega_last
                            if rates is not None and omega is not None:
                                offload_mask = omega != 0
                                if np.any(offload_mask):
                                    task = env.Task_size[0, :env.M_sim]
                                    rates_off = rates[offload_mask, 0]
                                    task_off = task[offload_mask]
                                    uplink_delay_ms = np.where(
                                        rates_off > 0,
                                        (task_off / rates_off) * 1000.0,
                                        0.0,
                                    )
                                    sum_uplink_delay_max += float(np.max(uplink_delay_ms))
                                    sum_uplink_delay_p95 += float(np.percentile(uplink_delay_ms, 95))
                        # Compute front delay stats from env internals for this step.
                        if hasattr(env, "front_delay_last") and hasattr(env, "omega_last"):
                            front_delay = env.front_delay_last
                            omega = env.omega_last
                            if front_delay is not None and omega is not None:
                                offload_mask = omega != 0
                                if np.any(offload_mask):
                                    front_delay_s = front_delay[offload_mask, 0]
                                    front_delay_ms = front_delay_s * 1000.0
                                    sum_front_delay_max += float(np.max(front_delay_ms))
                                    sum_front_delay_p95 += float(np.percentile(front_delay_ms, 95))
                                    task = env.Task_size[0, :env.M_sim]
                                    task_off = task[offload_mask]
                                    front_rate_mbps = np.where(
                                        front_delay_s > 0,
                                        (task_off / np.maximum(front_delay_s, 1e-12)) / 1e6,
                                        0.0,
                                    )
                                    sum_avg_front_rate += float(np.mean(front_rate_mbps))
                                    sum_front_rate_max += float(np.max(front_rate_mbps))
                                    sum_front_rate_p95 += float(np.percentile(front_rate_mbps, 95))
                        metric_steps += 1
                        # Per-agent delay probability (all agents vs offloading only).
                        if hasattr(env, "delay_last") and hasattr(env, "omega_last"):
                            delay_last = env.delay_last
                            omega_last = env.omega_last
                            if delay_last is not None and omega_last is not None:
                                delay_ms = delay_last[:env.M_sim, 0] * 1000.0
                                sum_total_delay_max += float(np.max(delay_ms))
                                sum_total_delay_p95 += float(np.percentile(delay_ms, 95))
                                sum_deadline_satisfaction_ratio += float(np.sum(delay_ms <= env.tau_c * 1000.0))
                                sum_agent_steps += int(delay_ms.size)
                                offload_mask = omega_last != 0
                                if np.any(offload_mask):
                                    offload_delay_ms = delay_ms[offload_mask]
                                    sum_offloading_deadline_satisfaction_ratio += float(np.sum(offload_delay_ms <= env.tau_c * 1000.0))
                                    sum_offload_steps += int(offload_delay_ms.size)
                        # Bottleneck stats for offloading users.
                        if (
                            hasattr(env, "uplink_delay_last")
                            and hasattr(env, "front_delay_last")
                            and hasattr(env, "actual_process_delay_last")
                            and hasattr(env, "omega_last")
                            and hasattr(env, "delay_last")
                        ):
                            omega_last = env.omega_last
                            offload_mask = omega_last != 0
                            if np.any(offload_mask):
                                uplink_delay = env.uplink_delay_last[:env.M_sim, 0]
                                front_delay = env.front_delay_last[:env.M_sim, 0]
                                process_delay = env.actual_process_delay_last[:env.M_sim, 0]
                                comp = np.stack([uplink_delay, front_delay, process_delay], axis=0)
                                bottleneck_idx = np.argmax(comp, axis=0)  # 0=uplink,1=front,2=process
                                sum_offload_count += int(offload_mask.sum())
                                sum_front_bottleneck_offload_count += int(np.sum((bottleneck_idx == 1) & offload_mask))
                                total_delay = env.delay_last[:env.M_sim, 0]
                                fail_mask = (total_delay > env.tau_c) & offload_mask
                                if np.any(fail_mask):
                                    sum_offload_fail_count += int(fail_mask.sum())
                                    sum_front_bottleneck_offload_fail_count += int(
                                        np.sum((bottleneck_idx == 1) & fail_mask)
                                    )
                                    fail_idx = bottleneck_idx[fail_mask]
                                    sum_fail_bottleneck_uplink_count += int(np.sum(fail_idx == 0))
                                    sum_fail_bottleneck_front_count += int(np.sum(fail_idx == 1))
                                    sum_fail_bottleneck_process_count += int(np.sum(fail_idx == 2))
                # Power distribution over all agents (including local=0).
                if hasattr(env, "p_idx_last") and env.p_idx_last is not None:
                    levels = np.clip(env.p_idx_last.astype(int), 0, 3)
                    counts = np.bincount(levels, minlength=4)
                    total = counts.sum()
                    if total > 0:
                        sum_power_dist += counts / total
                        dist_steps += 1
                obs = next_obs
                dones = next_dones
                step_count += 1
                masks = np.ones((env.n_agents, 1), dtype=np.float32)
            if metric_steps > 0:
                allsum_avg_total_delay += sum_avg_total_delay / metric_steps
                allsum_avg_local_delay += sum_avg_local_delay / metric_steps
                allsum_avg_uplink_delay += sum_avg_uplink_delay / metric_steps
                allsum_total_delay_max += sum_total_delay_max / metric_steps
                allsum_total_delay_p95 += sum_total_delay_p95 / metric_steps
                allsum_uplink_delay_max += sum_uplink_delay_max / metric_steps
                allsum_uplink_delay_p95 += sum_uplink_delay_p95 / metric_steps
                allsum_front_delay_max += sum_front_delay_max / metric_steps
                allsum_front_delay_p95 += sum_front_delay_p95 / metric_steps
                allsum_avg_front_delay += sum_avg_front_delay / metric_steps
                allsum_avg_front_rate += sum_avg_front_rate / metric_steps
                allsum_front_rate_max += sum_front_rate_max / metric_steps
                allsum_front_rate_p95 += sum_front_rate_p95 / metric_steps
                allsum_avg_actual_process_delay += sum_avg_actual_process_delay / metric_steps
                allsum_avg_uplink_rate += sum_avg_uplink_rate / metric_steps
                allsum_num_offloading_users += sum_num_offloading_users / metric_steps
                allsum_front_log_cap_hit_ratio += sum_front_log_cap_hit_ratio / metric_steps
                if sum_agent_steps > 0:
                    allsum_deadline_satisfaction_ratio += sum_deadline_satisfaction_ratio / sum_agent_steps
                if sum_offload_steps > 0:
                    allsum_offloading_deadline_satisfaction_ratio += sum_offloading_deadline_satisfaction_ratio / sum_offload_steps
            allsum_offload_count += sum_offload_count
            allsum_offload_fail_count += sum_offload_fail_count
            allsum_front_bottleneck_offload_count += sum_front_bottleneck_offload_count
            allsum_front_bottleneck_offload_fail_count += sum_front_bottleneck_offload_fail_count
            allsum_fail_bottleneck_uplink_count += sum_fail_bottleneck_uplink_count
            allsum_fail_bottleneck_front_count += sum_fail_bottleneck_front_count
            allsum_fail_bottleneck_process_count += sum_fail_bottleneck_process_count
            if dist_steps > 0:
                allsum_power_dist += sum_power_dist / dist_steps
            if cpu_counts is None:
                cpu_counts = np.zeros(env.K, dtype=np.int64)
            if allsum_cpu_counts is None:
                allsum_cpu_counts = np.zeros(env.K, dtype=np.int64)
            allsum_cpu_counts += cpu_counts
            allsum_offload += offload_counts
            allsum_cpu_used_steps += cpu_used_sum
            allsum_cpu_used_count += cpu_used_steps
            if USE_HIERARCHICAL and PER_USER and sum_high_combo_hist is not None:
                if allsum_high_combo_hist is None:
                    allsum_high_combo_hist = np.zeros_like(sum_high_combo_hist)
                allsum_high_combo_hist += sum_high_combo_hist
                allsum_best_cfq_sum += sum_best_cfq_sum
                allsum_best_cfq_count += sum_best_cfq_count
                allsum_cpu_match_sum += sum_cpu_match_sum
                allsum_cpu_match_count += sum_cpu_match_count
            if USE_HIERARCHICAL and PER_USER and sum_high_action_entropy_count > 0:
                allsum_high_action_entropy_sum += sum_high_action_entropy_sum
                allsum_high_action_entropy_count += sum_high_action_entropy_count
            if USE_HIERARCHICAL and attn_max_list:
                attn_max_arr = np.array(attn_max_list, dtype=np.float32)
                attn_ent_arr = np.array(attn_entropy_list, dtype=np.float32)
                print(
                    f"attn_max: mean={attn_max_arr.mean():.4f} p5={np.percentile(attn_max_arr, 5):.4f} min={attn_max_arr.min():.4f}"
                )
                print(
                    f"attn_entropy: mean={attn_ent_arr.mean():.4f} p5={np.percentile(attn_ent_arr, 5):.4f} min={attn_ent_arr.min():.4f}"
                )
        seed_results["avg_total_delay_ms"].append(allsum_avg_total_delay / num_episodes)
        seed_results["avg_local_delay_ms"].append(allsum_avg_local_delay / num_episodes)
        seed_results["avg_uplink_delay_ms"].append(allsum_avg_uplink_delay / num_episodes)
        seed_results["total_delay_ms_max"].append(allsum_total_delay_max / num_episodes)
        seed_results["total_delay_ms_p95"].append(allsum_total_delay_p95 / num_episodes)
        seed_results["uplink_delay_ms_max"].append(allsum_uplink_delay_max / num_episodes)
        seed_results["uplink_delay_ms_p95"].append(allsum_uplink_delay_p95 / num_episodes)
        seed_results["front_delay_ms_max"].append(allsum_front_delay_max / num_episodes)
        seed_results["front_delay_ms_p95"].append(allsum_front_delay_p95 / num_episodes)
        seed_results["avg_front_delay_ms"].append(allsum_avg_front_delay / num_episodes)
        seed_results["avg_front_rate_Mbps"].append(allsum_avg_front_rate / num_episodes)
        seed_results["front_rate_Mbps_max"].append(allsum_front_rate_max / num_episodes)
        seed_results["front_rate_Mbps_p95"].append(allsum_front_rate_p95 / num_episodes)
        seed_results["avg_actual_process_delay_ms"].append(allsum_avg_actual_process_delay / num_episodes)
        seed_results["avg_uplink_rate_Mbps"].append(allsum_avg_uplink_rate / num_episodes)
        seed_results["avg_offloading_users"].append(allsum_num_offloading_users / num_episodes)
        seed_results["front_log_cap_hit_ratio"].append(allsum_front_log_cap_hit_ratio / num_episodes)
        seed_results["deadline_satisfaction_ratio"].append(allsum_deadline_satisfaction_ratio / num_episodes)
        seed_results["offloading_deadline_satisfaction_ratio"].append(allsum_offloading_deadline_satisfaction_ratio / num_episodes)
        offload_den = max(1, int(allsum_offload_count))
        offload_fail_den = max(1, int(allsum_offload_fail_count))
        seed_results["p_front_bottleneck_offload"].append(
            allsum_front_bottleneck_offload_count / offload_den
        )
        seed_results["p_front_bottleneck_offload_fail"].append(
            allsum_front_bottleneck_offload_fail_count / offload_fail_den
        )
        seed_results["fail_bottleneck_uplink_ratio"].append(
            allsum_fail_bottleneck_uplink_count / offload_fail_den
        )
        seed_results["fail_bottleneck_front_ratio"].append(
            allsum_fail_bottleneck_front_count / offload_fail_den
        )
        seed_results["fail_bottleneck_process_ratio"].append(
            allsum_fail_bottleneck_process_count / offload_fail_den
        )
        seed_results["power_dist"].append(allsum_power_dist / num_episodes)
        if allsum_cpu_counts is None:
            allsum_cpu_counts = np.zeros(env.K, dtype=np.int64)
        cpu_total = max(1, int(allsum_offload))
        cpu_ratio = allsum_cpu_counts / cpu_total
        cpu_entropy = 0.0
        if cpu_ratio.sum() > 0:
            cpu_entropy = float(-(cpu_ratio * np.log(cpu_ratio + 1e-12)).sum() / np.log(len(cpu_ratio)))
        seed_results["cpu_select_ratio"].append(cpu_ratio)
        seed_results["cpu_select_entropy"].append(cpu_entropy)
        if allsum_cpu_used_count > 0:
            seed_results["avg_cpu_used_per_step"].append(allsum_cpu_used_steps / allsum_cpu_used_count)
        else:
            seed_results["avg_cpu_used_per_step"].append(0.0)
        if USE_HIERARCHICAL and PER_USER:
            if allsum_high_combo_hist is not None:
                hist_ratio = allsum_high_combo_hist / float(max(1, allsum_high_combo_hist.sum()))
                seed_results["high_combo_hist"].append(hist_ratio)
            else:
                n_combos = len(env._ap_combos) if hasattr(env, "_ap_combos") else 28
                seed_results["high_combo_hist"].append(np.zeros(n_combos, dtype=np.float64))
            mean_cfq = allsum_best_cfq_sum / max(1, allsum_best_cfq_count)
            seed_results["high_combo_mean_best_cfq"].append(mean_cfq)
            cpu_mr = allsum_cpu_match_sum / max(1, allsum_cpu_match_count)
            seed_results["high_cpu_match_rate"].append(cpu_mr)
            ent_norm = allsum_high_action_entropy_sum / max(1, allsum_high_action_entropy_count)
            seed_results["high_action_entropy_norm"].append(ent_norm)
        if USE_PIVOTAL_STATS:
            piv_den = max(1, pivotal_viol_count)
            seed_results["pivotal_uplink"].append(pivotal_u_count / piv_den)
            seed_results["pivotal_front"].append(pivotal_f_count / piv_den)
            seed_results["pivotal_both"].append(pivotal_both_count / piv_den)
            seed_results["pivotal_none"].append(pivotal_none_count / piv_den)
            viol_rate = pivotal_viol_count / max(1, pivotal_off_count)
            seed_results["offloading_violation_rate"].append(viol_rate)
            seed_results["pivotal_viol_count"].append(float(pivotal_viol_count))
            print(f"  offloading_violation_rate: {viol_rate:.4f}")
            print(f"  pivotal_viol_count: {pivotal_viol_count}")
    # ── combo quality diagnostic ──
    if USE_HIERARCHICAL and PER_USER and _diag_best_cfq_all:
        cfq_arr = np.array(_diag_best_cfq_all, dtype=np.float32)
        print(f"\n  [Diagnostic] best_cfq per high-decision: mean={cfq_arr.mean():.4f}, std={cfq_arr.std():.4f}")
        print(f"    Higher = high-level selects AP combos with better fronthaul to at least one CPU.")
        print(f"    Compare with random-high baseline to judge learning.")
    print("Summary over seeds (mean +/- std):")
    for key, vals in seed_results.items():
        if key == "deadline_satisfaction_ratio":
            vals = np.array(vals, dtype=np.float32)
            print("  deadline_satisfaction_ratio per seed:", vals)
            print(f"  {key}: {vals.mean():.4f} ? {vals.std():.4f}")
            continue
        if USE_PIVOTAL_STATS and key in {
            "pivotal_uplink",
            "pivotal_front",
            "pivotal_both",
            "pivotal_none",
            "pivotal_viol_count",
            "offloading_violation_rate",
        }:
            vals = np.array(vals, dtype=np.float32)
            print(f"  {key} per seed:", vals)
            print(f"  {key}: {vals.mean():.4f} ? {vals.std():.4f}")
            continue
        if key == "power_dist":
            vals = np.stack(vals, axis=0)
            mean = vals.mean(axis=0)
            std = vals.std(axis=0)
            print("  power_dist (p_level=0..3):")
            for i in range(4):
                print(f"    p{i}: {mean[i]:.4f} +/- {std[i]:.4f}")
            continue
        if key == "cpu_select_ratio":
            vals = np.stack(vals, axis=0)
            mean = vals.mean(axis=0)
            std = vals.std(axis=0)
            print("  cpu_select_ratio (cpu1..K):")
            for i in range(mean.size):
                print(f"    cpu{i + 1}: {mean[i]:.4f} +/- {std[i]:.4f}")
            continue
        if key == "high_combo_hist":
            vals = np.stack(vals, axis=0)
            mean = vals.mean(axis=0)
            std = vals.std(axis=0)
            # Print combo index → AP pair mapping for readability
            if hasattr(env, "_ap_combos"):
                print(f"  high_combo_hist (C({env.candidate_n},{env.k_fixed})={len(env._ap_combos)} combos):")
                top_k = min(10, len(mean))
                top_idx = np.argsort(mean)[::-1][:top_k]
                for ci in top_idx:
                    combo = env._ap_combos[ci]
                    print(f"    combo {ci} {combo}: {mean[ci]:.3f} +/- {std[ci]:.3f}")
            else:
                print("  high_combo_hist:")
                print("   ", np.array2string(mean, precision=3, separator=" "))
            continue
        vals = np.array(vals, dtype=np.float32)
        print(f"  {key}: {vals.mean():.4f} ? {vals.std():.4f}")
        if key == "avg_front_delay_ms" or key == "avg_uplink_delay_ms":
            print(vals)
if __name__ == "__main__":
    #model_file = "C:/DCNLab/UCMEC/UCMEC-mmWave-Fronthaul/results/MyEnv/MyEnv/mappo/noncoop_paper_baseline/paper_interval10/models/actor.pt" 
    model_file = globals().get("MODEL_LOW", None)
    
    # 檢查路徑是否已設定
    if model_file is None:
        print("[eval] MODEL_LOW 未定義，將以無 low model 模式執行（僅適用 heuristic-low 覆蓋情境）。")
        evaluate("")
    elif "請替換" in model_file:
        print("提示: 請編輯程式碼底部的 'model_file' 變數，設定正確的 actor.pt 路徑。")
    else:
        evaluate(model_file)
