import torch
import numpy as np
import sys
import os

# 將 UCMEC-mmWave-Fronthaul 資料夾加入系統路徑，以確保能匯入 envs 和 algorithms
# 假設此腳本位於 UCMEC-mmWave-Fronthaul 資料夾的上一層或同層
current_path = os.getcwd()
sys.path.append(os.path.join(current_path, "UCMEC-mmWave-Fronthaul"))

# Toggle here to switch evaluation mode without CLI args.
USE_HIERARCHICAL = False
PER_USER = False
HIERARCHICAL_INTERVAL = 10
USE_RECURRENT = True
#SEEDS = [1, 2, 3, 4, 5]
SEEDS = [6, 7, 8, 9, 10]
#SEEDS = [3]

def make_env(seed):
    if USE_HIERARCHICAL:
        if PER_USER:
            return MA_UCMEC_dyna_noncoop_hierarchical_peruser(render=True, seed=seed)
        else:
            return MA_UCMEC_dyna_noncoop_hierarchical_alluser(render=True, seed=seed)
    return MA_UCMEC_dyna_noncoop(render=True, seed=seed)
#IPPO
MODEL_LOW = "C:/DCNLab/UCMEC/UCMEC-mmWave-Fronthaul/results/MyEnv/MyEnv/rmappo/noncoop_rnn_test/IPPO_cluster5/models/actor_999.pt"
#peruser
#MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\MyEnv\MyEnv\rmappo\hierarchical_noncoop_rnn_test\hierarchical_IPPO_peruser/models/actor_999.pt"
#MODEL_HIGH = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\MyEnv\MyEnv\rmappo\hierarchical_noncoop_rnn_test\hierarchical_IPPO_peruser/models/actor_high.pt"
#MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\MyEnv\MyEnv\rmappo\hierarchical_noncoop_rnn_test\hierarchical_IPPO_peruser_commlim_newreward/models/actor_999.pt"
#MODEL_HIGH = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\MyEnv\MyEnv\rmappo\hierarchical_noncoop_rnn_test\hierarchical_IPPO_peruser_commlim_newreward/models/actor_high.pt"
#alluser
#MODEL_LOW = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\MyEnv\MyEnv\rmappo\hierarchical_noncoop_rnn_test\hierarchical_IPPO/models/actor_999.pt"
#MODEL_HIGH = r"C:\DCNLab\UCMEC\UCMEC-mmWave-Fronthaul\results\MyEnv\MyEnv\rmappo\hierarchical_noncoop_rnn_test\hierarchical_IPPO/models/actor_high.pt"

# 匯入必要的模組
try:
    from envs.MA_UCMEC_dyna_noncoop import MA_UCMEC_dyna_noncoop
    from envs.MA_UCMEC_dyna_noncoop_hierarchical_alluser import MA_UCMEC_dyna_noncoop_hierarchical_alluser
    from envs.MA_UCMEC_dyna_noncoop_hierarchical_peruser import MA_UCMEC_dyna_noncoop_hierarchical_peruser
    from algorithms.algorithm.r_actor_critic import R_Actor
    from config import get_config
except ImportError as e:
    print("匯入模組失敗，請確認 'UCMEC-mmWave-Fronthaul' 資料夾是否在當前目錄下。")
    print(f"錯誤訊息: {e}")
    sys.exit(1)

def evaluate(model_path):
    # 1. 取得設定參數 (Arguments)
    # 使用 config.py 中的預設參數
    parser = get_config()
    # 如果你的訓練參數有大幅修改（例如 hidden_size），請在這裡透過參數覆蓋，或是確保 config.py 是正確的
    args = parser.parse_args([])
    # Reason: match evaluation actor to recurrent checkpoint if needed.
    if USE_RECURRENT:
        args.use_recurrent_policy = True
        args.use_naive_recurrent_policy = False
    
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
        high_actor = R_Actor(args, high_obs_space, high_act_space, device)
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
    num_episodes = 10
    seed_results = {
        "avg_total_delay_ms": [],
        "avg_local_delay_ms": [],
        "avg_uplink_delay_ms": [],
        "uplink_delay_ms_max": [],
        "uplink_delay_ms_p95": [],
        "avg_front_delay_ms": [],
        "avg_actual_process_delay_ms": [],
        "avg_uplink_rate_Mbps": [],
        "avg_offloading_users": [],
        "power_dist": [],
    }

    for seed in SEEDS:
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = make_env(seed)
        env.seed(seed)

        allsum_avg_total_delay = 0.0
        allsum_avg_local_delay = 0.0
        allsum_avg_uplink_delay = 0.0
        allsum_uplink_delay_max = 0.0
        allsum_uplink_delay_p95 = 0.0
        allsum_avg_front_delay = 0.0
        allsum_avg_actual_process_delay = 0.0
        allsum_avg_uplink_rate = 0.0
        allsum_num_offloading_users = 0.0
        allsum_power_dist = np.zeros(4, dtype=np.float64)

        for _ in range(num_episodes):
            obs = env.reset()
            rnn_states = np.zeros((env.n_agents, args.recurrent_N, args.hidden_size), dtype=np.float32)
            masks = np.ones((env.n_agents, 1), dtype=np.float32)

            sum_avg_total_delay = 0.0
            sum_avg_local_delay = 0.0
            sum_avg_uplink_delay = 0.0
            sum_uplink_delay_max = 0.0
            sum_uplink_delay_p95 = 0.0
            sum_avg_front_delay = 0.0
            sum_avg_actual_process_delay = 0.0
            sum_avg_uplink_rate = 0.0
            sum_num_offloading_users = 0.0
            sum_power_dist = np.zeros(4, dtype=np.float64)
            metric_steps = 0
            dist_steps = 0

            dones = [False] * env.n_agents

            if USE_HIERARCHICAL:
                high_rnn_states = np.zeros((1, args.recurrent_N, args.hidden_size), dtype=np.float32)
                high_masks = np.ones((1, 1), dtype=np.float32)

            step_count = 0
            while not all(dones):
                if USE_HIERARCHICAL and (step_count % HIERARCHICAL_INTERVAL == 0):
                    global_obs = env.get_global_obs().reshape(1, -1)
                    with torch.no_grad():
                        high_action, _, high_rnn_states = high_actor(
                            global_obs, high_rnn_states, high_masks, deterministic=True
                        )
                    if PER_USER:
                        high_action = high_action.cpu().numpy()
                        num_classes = len(env.cluster_size_candidates)
                        if high_action.ndim >= 2 and high_action.shape[-1] == num_classes:
                            # one-hot -> index
                            high_action = np.argmax(high_action, axis=-1)
                        high_action = high_action.astype(int).squeeze(0)
                    else:
                        high_action = int(high_action.cpu().numpy().flatten()[0])
                    env.set_high_action(high_action)
                    
                obs_batch = np.stack(obs)
                with torch.no_grad():
                    actions, _, rnn_states = actor(obs_batch, rnn_states, masks, deterministic=True)

                action_indices = actions.cpu().numpy().flatten()
                actions_env = np.eye(act_space.n)[action_indices]

                next_obs, rewards, next_dones, infos = env.step(actions_env)

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
                        metric_steps += 1

                # Power distribution over all agents (including local=0).
                if hasattr(env, "p_last") and env.p_last is not None:
                    p_level = env.P_max / 4.0
                    levels = np.rint(env.p_last / p_level).astype(int)
                    levels = np.clip(levels, 0, 3)
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
                allsum_uplink_delay_max += sum_uplink_delay_max / metric_steps
                allsum_uplink_delay_p95 += sum_uplink_delay_p95 / metric_steps
                allsum_avg_front_delay += sum_avg_front_delay / metric_steps
                allsum_avg_actual_process_delay += sum_avg_actual_process_delay / metric_steps
                allsum_avg_uplink_rate += sum_avg_uplink_rate / metric_steps
                allsum_num_offloading_users += sum_num_offloading_users / metric_steps
            if dist_steps > 0:
                allsum_power_dist += sum_power_dist / dist_steps

        seed_results["avg_total_delay_ms"].append(allsum_avg_total_delay / num_episodes)
        seed_results["avg_local_delay_ms"].append(allsum_avg_local_delay / num_episodes)
        seed_results["avg_uplink_delay_ms"].append(allsum_avg_uplink_delay / num_episodes)
        seed_results["uplink_delay_ms_max"].append(allsum_uplink_delay_max / num_episodes)
        seed_results["uplink_delay_ms_p95"].append(allsum_uplink_delay_p95 / num_episodes)
        seed_results["avg_front_delay_ms"].append(allsum_avg_front_delay / num_episodes)
        seed_results["avg_actual_process_delay_ms"].append(allsum_avg_actual_process_delay / num_episodes)
        seed_results["avg_uplink_rate_Mbps"].append(allsum_avg_uplink_rate / num_episodes)
        seed_results["avg_offloading_users"].append(allsum_num_offloading_users / num_episodes)
        seed_results["power_dist"].append(allsum_power_dist / num_episodes)

    print("Summary over seeds (mean +/- std):")
    for key, vals in seed_results.items():
        if key == "power_dist":
            vals = np.stack(vals, axis=0)
            mean = vals.mean(axis=0)
            std = vals.std(axis=0)
            print("  power_dist (p_level=0..3):")
            for i in range(4):
                print(f"    p{i}: {mean[i]:.4f} +/- {std[i]:.4f}")
            continue
        vals = np.array(vals, dtype=np.float32)
        print(f"  {key}: {vals.mean():.4f} ? {vals.std():.4f}")

if __name__ == "__main__":
    #model_file = "C:/DCNLab/UCMEC/UCMEC-mmWave-Fronthaul/results/MyEnv/MyEnv/mappo/noncoop_paper_baseline/paper_interval10/models/actor.pt" 
    model_file = MODEL_LOW 
    
    # 檢查路徑是否已設定
    if "請替換" in model_file:
        print("提示: 請編輯程式碼底部的 'model_file' 變數，設定正確的 actor.pt 路徑。")
    else:
        evaluate(model_file)
