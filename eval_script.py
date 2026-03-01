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
USE_HIERARCHICAL = False
PER_USER = False
HIERARCHICAL_INTERVAL = 10
#HIERARCHICAL_INTERVAL = 10
USE_RECURRENT = True
DEBUG_HIGH_ACTION_PROBS = True  # Print Bernoulli bit probs / AP mapping at high-level decision steps.
USE_PIVOTAL_STATS = False
#SEEDS = [11, 2, 3, 4, 5, 6, 7, 8, 9, 10]

SEEDS = [18, 62, 53, 14, 58,
         161, 37, 88, 95, 150,
         11, 17, 29, 189, 198,
         153, 26, 59, 365, 84,
         946, 56, 99, 75, 263,
         776, 94, 71, 735, 64]
#SEEDS = [1, 1001, 2001, 3001, 4001, 5001, 6001, 7001, 8001, 9001]
#SEEDS = [3]

def make_env(seed):
    if USE_HIERARCHICAL:
        if PER_USER:
            return MA_UCMEC_dyna_noncoop_hierarchical_peruser(render=True, seed=seed)
        else:
            return MA_UCMEC_dyna_noncoop_hierarchical_alluser(render=True, seed=seed)
    return MA_UCMEC_dyna_noncoop(render=True, seed=seed)
#IPPO
#MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\MyEnv\rmappo\noncoop_rnn\IPPO_cluster5_oldobs\models/actor_999.pt"
#MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\APselect\rmappo\noncoop_rnn\IPPO_clsuter5_meter\models/actor_999.pt"
#MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\APselect\rmappo\noncoop_rnn\IPPO_clsuter5_meter_edge\models/actor_999.pt"
MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\APselect\rmappo\noncoop_rnn\IPPO_randcluster_meter_edge\models/actor_999.pt"

#MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\MyEnv\rmappo\noncoop_rnn\IPPO_cluster5\models/actor_999.pt"
#MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\smallEnv\MyEnv\rmappo\noncoop_rnn\IPPO_cluster5_interval20\models/actor_999.pt"

#peruser
#MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\APselect\rmappo\hierarchical_noncoop_rnn\hierarchical_IPPO_select/models/actor_999.pt"
#MODEL_HIGH = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\APselect\rmappo\hierarchical_noncoop_rnn\hierarchical_IPPO_select/models/actor_high.pt"
#MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\APselect\rmappo\hierarchical_noncoop_rnn\hierarchical_IPPO_select_newreward2/models/actor_999.pt"
#MODEL_HIGH = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\APselect\rmappo\hierarchical_noncoop_rnn\hierarchical_IPPO_select_newreward2/models/actor_high.pt"
#MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\APselect\rmappo\hierarchical_noncoop_rnn\hierarchical_IPPO_select_newma_rewardA/models/actor_999.pt"
#MODEL_HIGH = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\APselect\rmappo\hierarchical_noncoop_rnn\hierarchical_IPPO_select_newma_rewardA/models/actor_high.pt"

#peruser_perreward
#MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\APselect\rmappo\hierarchical_noncoop_rnn_perreward\hierarchical_IPPO_select_perreward2/models/actor_999.pt"
#MODEL_HIGH = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\APselect\rmappo\hierarchical_noncoop_rnn_perreward\hierarchical_IPPO_select_perreward/models/actor_high.pt"

#test
#MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\APselect\rmappo\hierarchical_config_test\run6/models/actor_99.pt"
#MODEL_HIGH = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\APselect\rmappo\hierarchical_config_test\run6/models/actor_high.pt"

#alluser
#MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\MyEnv\rmappo\hierarchical_noncoop_rnn\oldobs\hierarchical_IPPO_alluser_front_attn/models/actor_999.pt"
#MODEL_HIGH = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\MyEnv\rmappo\hierarchical_noncoop_rnn\oldobs\hierarchical_IPPO_alluser_front_attn/models/actor_high.pt"
#MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\smallEnv\MyEnv\rmappo\hierarchical_noncoop_rnn\hierarchical_IPPO_alluser_interval20_clusterobs/models/actor_999.pt"
#MODEL_HIGH = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\smallEnv\MyEnv\rmappo\hierarchical_noncoop_rnn\hierarchical_IPPO_alluser_interval20_clusterobs/models/actor_high.pt"

#MAPPO
#MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\MyEnv\rmappo\coop_rnn\MAPPO_cluster5_newobs\models/actor_999.pt"

#MAPPO peruser
#MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\MyEnv\rmappo\hierarchical_coop_rnn\hierarchical_MAPPO_peruser_clusterobs/models/actor_999.pt"
#MODEL_HIGH = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\MyEnv\rmappo\hierarchical_coop_rnn\hierarchical_MAPPO_peruser_clusterobs/models/actor_high.pt"

#fixlow peruser
#MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\MyEnv\rmappo\hierarchical_fixlow\hierarchical_fixIPPO_peruser_ratio/models/actor_999.pt"
#MODEL_HIGH = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\newEnv\MyEnv\rmappo\hierarchical_fixlow\hierarchical_fixIPPO_peruser_ratio/models/actor_high.pt"

# 匯入必要的模組
try:
    #from envs.MA_UCMEC_dyna_noncoop import MA_UCMEC_dyna_noncoop
    from envs.MA_UCMEC_dyna_noncoop_cluster_rand import MA_UCMEC_dyna_noncoop_cluster_rand as MA_UCMEC_dyna_noncoop
    from envs.MA_UCMEC_dyna_coop import MA_UCMEC_dyna_coop
    from envs.MA_UCMEC_dyna_noncoop_hierarchical_alluser_front import MA_UCMEC_dyna_noncoop_hierarchical_alluser
    #from envs.MA_UCMEC_dyna_noncoop_hierarchical_alluser import MA_UCMEC_dyna_noncoop_hierarchical_alluser
    from envs.MA_UCMEC_dyna_noncoop_hierarchical_peruser_apselect import MA_UCMEC_dyna_noncoop_hierarchical_peruser
    from algorithms.algorithm.r_actor_critic import R_Actor
    from algorithms.algorithm.high_actor_critic import HighActor
    from config import get_config
except ImportError as e:
    print("匯入模組失敗，請確認 'UCMEC-mmWave-Fronthaul' 資料夾是否在當前目錄下。")
    print(f"錯誤訊息: {e}")
    sys.exit(1)

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

    if USE_HIERARCHICAL:
        high_obs_space = env.high_observation_space
        high_act_space = env.high_action_space
        if PER_USER:
            high_actor = HighActor(high_args, high_obs_space, high_act_space, device)
        else:
            high_actor = R_Actor(high_args, high_obs_space, high_act_space, device)
        print(f"Loading model: {MODEL_HIGH}")
        if os.path.exists(MODEL_HIGH):
            state_dict = torch.load(MODEL_HIGH, map_location=device)
            high_actor.load_state_dict(state_dict)
            print("Model loaded.")
        else:
            print(f"Error: model not found {MODEL_HIGH}")
            return
        high_actor.eval()

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
    }
    if USE_PIVOTAL_STATS:
        seed_results.update({
            "pivotal_uplink": [],
            "pivotal_front": [],
            "pivotal_both": [],
            "pivotal_none": [],
            "pivotal_viol_count": [],
            "offloading_violation_rate": [],
        })

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

            dones = [False] * env.n_agents

            if USE_HIERARCHICAL:
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
                                    front_delay_ms = front_delay[offload_mask, 0] * 1000.0
                                    sum_front_delay_max += float(np.max(front_delay_ms))
                                    sum_front_delay_p95 += float(np.percentile(front_delay_ms, 95))
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
                allsum_avg_actual_process_delay += sum_avg_actual_process_delay / metric_steps
                allsum_avg_uplink_rate += sum_avg_uplink_rate / metric_steps
                allsum_num_offloading_users += sum_num_offloading_users / metric_steps
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
        seed_results["avg_actual_process_delay_ms"].append(allsum_avg_actual_process_delay / num_episodes)
        seed_results["avg_uplink_rate_Mbps"].append(allsum_avg_uplink_rate / num_episodes)
        seed_results["avg_offloading_users"].append(allsum_num_offloading_users / num_episodes)
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
        vals = np.array(vals, dtype=np.float32)
        print(f"  {key}: {vals.mean():.4f} ? {vals.std():.4f}")
        if key == "avg_front_delay_ms" or key == "avg_uplink_delay_ms":
            print(vals)

if __name__ == "__main__":
    #model_file = "C:/DCNLab/UCMEC/UCMEC-mmWave-Fronthaul/results/MyEnv/MyEnv/mappo/noncoop_paper_baseline/paper_interval10/models/actor.pt" 
    model_file = MODEL_LOW 
    
    # 檢查路徑是否已設定
    if "請替換" in model_file:
        print("提示: 請編輯程式碼底部的 'model_file' 變數，設定正確的 actor.pt 路徑。")
    else:
        evaluate(model_file)
