'''
from MA_UCMEC_dyna_noncoop import MA_UCMEC_dyna_noncoop
import numpy as np

def make_env():
    env = MA_UCMEC_dyna_noncoop(render=False)
    # 建議測試時先關掉移動，避免結果太亂
    env.is_mobile = False
    return env

def make_action_all_local(env):
    # 回傳形狀: [M_sim, 10]
    act = []
    for i in range(env.M_sim):
        a = np.zeros(10, dtype=int)
        a[0] = 1  # local
        act.append(a)
    
    return act

def make_action_all_offload_cpu1_maxP(env):
    act = []
    a = np.zeros(10, dtype=int)
    a[9] = 1  # local
    act.append(a)
    for i in range(env.M_sim-1):
        a = np.zeros(10, dtype=int)
        a[0] = 1  # local
        act.append(a)
    return act

def make_action_all_offload_cpu2_maxP(env):
    act = []
    for i in range(env.M_sim-5):
        a = np.zeros(10, dtype=int)
        a[0] = 1  # omega=1, p=3
        act.append(a)
    for i in range(env.M_sim-5):
        a = np.zeros(10, dtype=int)
        a[9] = 1  # omega=2, p=3
        act.append(a)
    return act

def make_action_all_offload_cpu3_maxP(env):
    act = []
    for i in range(env.M_sim):
        a = np.zeros(10, dtype=int)
        a[9] = 1  # omega=3, p=3
        act.append(a)
    return act

def eval_fixed_policy(env, action_fn, num_episodes=20, max_steps=50):
    all_delays = []

    for ep in range(num_episodes):
        obs = env.reset()
        for t in range(max_steps):
            action = action_fn(env)
            obs, rewards, dones, infos = env.step(action)
            # infos: list of length M_sim, 每個是 dict
            for info in infos:
                all_delays.append(info["total_delay"])
            # 如果你不想讓 max_steps 生效，也可以用 dones 來 break
            # if all(dones):
            #     break
    all_delays = np.array(all_delays)
    return all_delays.mean(), all_delays.std()

if __name__ == "__main__":
    env = make_env()

    policies = {
        "all_local": make_action_all_local,
        "offload_cpu1_maxP": make_action_all_offload_cpu1_maxP,
        "offload_cpu2_maxP": make_action_all_offload_cpu2_maxP,
        "offload_cpu3_maxP": make_action_all_offload_cpu3_maxP,
    }

    for name, fn in policies.items():
        mean_delay, std_delay = eval_fixed_policy(env, fn, num_episodes=10, max_steps=50)
        print(f"Policy {name}: mean delay = {mean_delay*1000:.2f} ms, std = {std_delay*1000:.2f} ms")
'''
'''調整xdb
import numpy as np
from MA_UCMEC_dyna_noncoop import MA_UCMEC_dyna_noncoop

def estimate_avg_K(x_db, num_episodes=10, steps_per_ep=200):
    """
    測試給定 x_db 門檻下，平均會選出多少顆 AP (K)。
    """
    env = MA_UCMEC_dyna_noncoop(render=False)
    
    Ks = []
    print(f"正在測試 x_db = {x_db} dB ... ", end="", flush=True)
    
    for ep in range(num_episodes):
        env.reset()
        for t in range(steps_per_ep):
            # 1. 生成隨機動作 (整數 tuple)
            raw_actions = env.action_space.sample()
            
            # 2. 【修正點】將整數動作轉換為環境需要的 One-Hot 向量格式
            one_hot_actions = []
            for act_idx in raw_actions:
                # 動作空間大小為 10
                oh_vec = np.zeros(10)
                oh_vec[act_idx] = 1
                one_hot_actions.append(oh_vec)
            
            # 3. 傳入 One-Hot 動作讓環境 Step
            env.step(one_hot_actions)

            # 4. 取得當前的 Large-scale Fading
            # env.beta 形狀通常是 (M, N)，我們只取模擬用的前 N_sim 個 AP
            beta = env.beta[:, :env.N_sim]
            
            for i in range(env.M_sim):
                beta_i = beta[i]
                
                # --- Threshold-based Clustering 核心邏輯 ---
                # 找出最強 AP (dB)
                best_val = np.max(beta_i)
                best_db  = 10 * np.log10(best_val + 1e-12)
                
                # 算出所有 AP 與最強 AP 的差值
                beta_db  = 10 * np.log10(beta_i + 1e-12)
                diff_db  = best_db - beta_db
                
                # 篩選差距在 x_db 內的候選人
                cand_idx = np.where(diff_db <= x_db)[0]
                num_cand = len(cand_idx)
                
                # 套用 K_min / K_max 強制邊界
                K_min, K_max = 2, 8
                
                if num_cand < K_min:
                    chosen_K = K_min
                elif num_cand > K_max:
                    chosen_K = K_max
                else:
                    chosen_K = num_cand
                    
                Ks.append(chosen_K)
    
    avg_k = np.mean(Ks)
    # 統計分佈：10% (較少AP的情況), 50% (中位數), 90% (較多AP的情況)
    percentiles = np.percentile(Ks, [10, 50, 90])
    print(f"Done. Avg K = {avg_k:.2f}")
    
    return avg_k, percentiles

if __name__ == "__main__":
    # 設定您的 Baseline K 值 (例如原本固定 K=5)
    TARGET_K = 5.0
    
    # 設定要測試的門檻範圍
    test_values = [20]
    
    best_x = None
    min_diff = float('inf')
    
    print(f"\n{'='*60}")
    print(f"開始參數校準 (Target Avg K = {TARGET_K})")
    print(f"{'='*60}")
    print(f"{'Threshold (dB)':<15} | {'Avg K':<10} | {'Distribution [10%, 50%, 90%]'}")
    print(f"{'-'*60}")
    
    for x in test_values:
        avg_k, stats = estimate_avg_K(x_db=x, num_episodes=10, steps_per_ep=200)
        
        # 顯示結果
        stats_str = f"[{stats[0]:.0f}, {stats[1]:.0f}, {stats[2]:.0f}]"
        print(f"{x:<15} | {avg_k:<10.2f} | {stats_str}")
        
        # 紀錄最接近目標的 x
        diff = abs(avg_k - TARGET_K)
        if diff < min_diff:
            min_diff = diff
            best_x = x
            
    print(f"{'='*60}")
    print(f"建議設定結果：")
    print(f"最佳 x_db = {best_x} dB")
    if best_x is not None:
        print(f"最小誤差: {min_diff:.4f}")
    print(f"{'='*60}")
'''
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import math
from MA_UCMEC_dyna_noncoop import MA_UCMEC_dyna_noncoop

# ==========================================
# 1. 定義可自訂策略的評估環境
# ==========================================
class EvalEnv(MA_UCMEC_dyna_noncoop):
    """
    繼承原始環境，但允許從外部注入 Clustering 策略與間隔。
    """
    def __init__(self, cluster_func, interval):
        super().__init__(render=False)
        self.custom_cluster_func = cluster_func # 外部定義的 Clustering 函數
        self.cluster_interval = interval        # Clustering 更新間隔 (1, 10, 20...)

    # 覆寫 step 函數以支援自訂 Clustering 頻率
    def step(self, action):
        self.step_num += 1

        # --- 1. 用戶移動 (保持原樣) ---
        if self.is_mobile:
            max_speed = 15 * self.tau_c
            min_speed = 5 * self.tau_c
            destination_users = np.random.random_sample([self.M, 2]) * 900
            user_speed = np.random.uniform(min_speed, max_speed, [self.M, 1])
            for i in range(self.M):
                dx = destination_users[i, 0] - self.locations_users[i, 0]
                dy = destination_users[i, 1] - self.locations_users[i, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist == 0: continue
                step_dist = user_speed[i, 0] / dist  
                self.locations_users[i, 0] += dx * step_dist
                self.locations_users[i, 1] += dy * step_dist
            self.locations_users = np.clip(self.locations_users, 0, 900)

        # --- 2. 更新 Channel & Path Loss (保持原樣) ---
        diff = self.locations_users[:, np.newaxis, :] - self.locations_aps[np.newaxis, :, :]
        self.distance_matrix = np.sqrt(np.sum(diff**2, axis=2))
        
        d_km = self.distance_matrix / 1000.0
        far_mask  = self.distance_matrix > self.d_1
        mid_mask  = (self.distance_matrix >= self.d_0) & (self.distance_matrix <= self.d_1)
        near_mask = self.distance_matrix < self.d_0
        
        if np.any(far_mask): self.PL[far_mask] = -self.L - 35 * np.log10(d_km[far_mask])
        if np.any(mid_mask): self.PL[mid_mask] = -self.L - 10 * np.log10((self.d_1/1000)**1.5 * (d_km[mid_mask]**2))
        if np.any(near_mask):self.PL[near_mask] = -self.L - 10 * np.log10((self.d_1/1000)**1.5 * (self.d_0/1000)**2)

        
        self.beta = np.power(10, self.PL / 10.0) * np.power(10, (self.sigma_s * self.mu) / 10.0)
        
        h_real = np.random.normal(loc=0, scale=0.5, size=(self.M, self.N, self.varsig))
        h_imag = np.random.normal(loc=0, scale=0.5, size=(self.M, self.N, self.varsig))
        self.h = h_real + 1j * h_imag
        self.access_chan = np.sqrt(self.beta)[:, :, np.newaxis] * self.h
        
        theta = (self.tau_p * self.P_max * (self.beta ** 2)) / (self.tau_p * self.P_max * self.beta + self.noise_access)

        # --- 3. 執行 Clustering (關鍵修改點) ---
        # 使用自訂的 interval 和 strategy
        if self.step_num == 1 or self.step_num % self.cluster_interval == 0:
            # 呼叫注入的策略函數，傳入 self (env) 以取得 beta 等資訊
            self.cluster_matrix = self.custom_cluster_func(self)
        
        cluster_matrix = self.cluster_matrix

        # --- 4. 計算速率與延遲 (保持原樣) ---
        omega_current = np.zeros([self.M_sim])
        p_current = np.zeros([self.M_sim])
        p_level = self.P_max / 4

        for i in range(self.M_sim):
            omega_current[i], p_current[i] = self.action_mapping(action[i])
            p_current[i] = (p_current[i] + 1) * p_level

        uplink_rate_access = self.uplink_rate_cal(p_current, omega_current, cluster_matrix, theta)
        front_rate_user = self.front_rate_cal(omega_current, cluster_matrix)
        self.uplink_rate_access_b = uplink_rate_access

        # Delays
        local_delay = np.zeros([self.M_sim, 1])
        for i in range(self.M_sim):
            if omega_current[i] == 0:
                local_delay[i, 0] = self.Task_density[0, i] * self.Task_size[0, i] / self.C_user[0, i]

        uplink_delay = np.zeros([self.M_sim, 1])
        for i in range(self.M_sim):
            if omega_current[i] != 0 and uplink_rate_access[i, 0] > 0:
                uplink_delay[i, 0] = self.Task_size[0, i] / uplink_rate_access[i, 0]

        front_delay = np.zeros([self.M_sim, 1])
        for i in range(self.M_sim):
            if omega_current[i] != 0:
                ap_idx = np.where(cluster_matrix[i, :] == 1)[0]
                delays = []
                for j in ap_idx:
                    if front_rate_user[i, j] > 0:
                        delays.append(self.Task_size[0, i] / front_rate_user[i, j])
                if len(delays) > 0:
                    front_delay[i, 0] = np.max(delays)

        # CPU Processing
        task_mat = np.zeros([self.M_sim, self.K])
        for i in range(self.M_sim):
            if omega_current[i] != 0:
                CPU_id = int(omega_current[i] - 1)
                task_mat[i, CPU_id] = self.Task_size[0, i] * self.Task_density[0, i]

        actual_C = np.zeros([self.M_sim, self.K])
        for i in range(self.K):
            serve_user_id = []
            serve_user_task = []
            _local_delay = []
            _uplink_delay = []
            for j in range(self.M_sim):
                if task_mat[j, i] != 0:
                    serve_user_id.append(j)
                    serve_user_task.append(task_mat[j, i])
                    _local_delay.append(local_delay[j, 0])
                    _uplink_delay.append(uplink_delay[j, 0])
            if len(serve_user_id) == 0: continue
            
            C = cp.Variable(len(serve_user_id))
            _process_delay = cp.multiply(serve_user_task, cp.inv_pos(C))
            _local_delay = np.array(_local_delay)
            _uplink_delay = np.array(_uplink_delay)
            func = cp.Minimize(cp.sum(cp.maximum(_local_delay, _uplink_delay + _process_delay)))
            cons = [0 <= C, cp.sum(C) <= self.C_edge[i, 0]]
            prob = cp.Problem(func, cons)
            prob.solve(solver=cp.SCS, verbose=False)
            for k in range(len(serve_user_id)):
                _C = C.value
                if _C is not None: actual_C[serve_user_id[k], i] = _C[k]
                else: actual_C[serve_user_id[k], i] = self.C_edge[i, 0] / len(serve_user_id)

        actual_process_delay = np.zeros([self.M_sim, 1])
        for i in range(self.M_sim):
            if omega_current[i] != 0:
                CPU_id = int(omega_current[i] - 1)
                actual_process_delay[i, 0] = task_mat[i, CPU_id] / np.sum(actual_C[i, :])

        total_delay = np.zeros([self.M_sim, 1])
        for i in range(self.M_sim):
            total_delay[i, 0] = np.maximum(local_delay[i, 0],
                                           front_delay[i, 0] + uplink_delay[i, 0] + actual_process_delay[i, 0])
        
        # --- 5. 收集回傳資訊 (重要：將 Delay 傳出去) ---
        # 我們這裡直接把每個 user 的 delay 都回傳，方便統計 CDF
        info = {
            "total_delays": total_delay.flatten(),  # 每個 User 的 delay
            "avg_delay": np.mean(total_delay)
        }
        
        # Dummy returns for gym compatibility
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = [False]*self.M_sim
        sub_agent_info = [info]*self.M_sim # 讓外部能拿到
        
        return sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info

# ==========================================
# 2. 定義不同的 Clustering 策略函數
# ==========================================

def strategy_fixed_k5(env):
    """
    原始策略：固定選訊號最強的 5 個 AP
    """
    cluster_matrix = np.zeros([env.M_sim, env.N_sim], dtype=int)
    fixed_k = 5
    beta = env.beta[:, :env.N_sim]
    for i in range(env.M_sim):
        sorted_idx = np.argsort(beta[i])[::-1]
        chosen = sorted_idx[:fixed_k]
        cluster_matrix[i, chosen] = 1
    return cluster_matrix

def strategy_rule_based_dynamic(env):
    """
    Proposed 策略：基於相對強度 (Dominance) 的動態 K
    參數設定：x_db = 20 (基於之前的校準)
    """
    cluster_matrix = np.zeros([env.M_sim, env.N_sim], dtype=int)
    threshold_db = 20.0
    K_min = 2
    K_max = 8
    beta = env.beta[:, :env.N_sim]

    for i in range(env.M_sim):
        beta_i = beta[i]
        best_idx = np.argmax(beta_i)
        best_db  = 10 * np.log10(beta_i[best_idx] + 1e-12)
        beta_db  = 10 * np.log10(beta_i + 1e-12)
        diff_db  = best_db - beta_db
        
        cand_idx = np.where(diff_db <= threshold_db)[0]
        
        if len(cand_idx) < K_min:
            sorted_idx = np.argsort(beta_i)[::-1]
            chosen = sorted_idx[:K_min]
        elif len(cand_idx) > K_max:
            sorted_cand = cand_idx[np.argsort(beta_i[cand_idx])[::-1]]
            chosen = sorted_cand[:K_max]
        else:
            chosen = cand_idx
            
        cluster_matrix[i, chosen] = 1
    return cluster_matrix

# ==========================================
# 3. 執行評估迴圈
# ==========================================

def run_evaluation(policy_name, cluster_func, interval, num_episodes=20, steps_per_ep=200):
    print(f"[{policy_name}] 開始評估... (Interval={interval})")
    env = EvalEnv(cluster_func, interval)
    
    all_delays = [] # 收集所有 step、所有 user 的 delay
    
    for ep in range(num_episodes):
        env.reset()
        for t in range(steps_per_ep):
            # 隨機動作 (One-Hot)
            raw_actions = env.action_space.sample()
            one_hot_actions = []
            for act_idx in raw_actions:
                oh_vec = np.zeros(10)
                oh_vec[act_idx] = 1
                one_hot_actions.append(oh_vec)
            
            # 執行 Step
            obs, rew, done, info_list = env.step(one_hot_actions)
            
            # 從 info 收集 delay (取第一個 agent 的 info 即可，因為我們塞了一樣的 array)
            current_delays = info_list[0]['total_delays']
            all_delays.extend(current_delays)
            
    return np.array(all_delays) * 1000 # 轉成 ms

# ==========================================
# 4. 主程式：比較三種策略並畫圖
# ==========================================

if __name__ == "__main__":
    # 設定參數
    EPISODES = 30
    STEPS = 200
    
    # 1. 執行 Baseline: Fixed K=5, Interval=20 (原始設定)
    delays_baseline = run_evaluation(
        "Baseline (Fixed K=5, Int=10)", 
        strategy_fixed_k5, 
        interval=10, 
        num_episodes=EPISODES, steps_per_ep=STEPS
    )
    
    # 2. 執行 Frequent: Fixed K=5, Interval=1 (高頻固定)
    delays_frequent = run_evaluation(
        "Frequent (Fixed K=5, Int=1)", 
        strategy_fixed_k5, 
        interval=1, 
        num_episodes=EPISODES, steps_per_ep=STEPS
    )
    
    # 3. 執行 Proposed: Dynamic K, Interval=1 (高頻動態)
    delays_proposed = run_evaluation(
        "Proposed (Dynamic K, Int=10)", 
        strategy_rule_based_dynamic, 
        interval=10, 
        num_episodes=EPISODES, steps_per_ep=STEPS
    )
    
    # --- 統計與顯示 ---
    print("\n" + "="*60)
    print(f"{'Policy':<30} | {'Avg (ms)':<10} | {'10% (ms)':<10} | {'50% (ms)':<10} | {'95% (ms)':<10} | {'99% (ms)':<10}")
    print("-" * 80)
    
    datasets = [
        ("Baseline (Int=10)", delays_baseline),
        ("Frequent (Int=1)", delays_frequent),
        ("Proposed (Dynamic)", delays_proposed)
    ]
    
    colors = ['gray', 'blue', 'red']
    linestyles = ['--', ':', '-']
    
    plt.figure(figsize=(10, 6))
    
    for (name, data), color, ls in zip(datasets, colors, linestyles):
        # 計算統計量
        avg = np.mean(data)
        p10 = np.percentile(data, 10)
        p50 = np.percentile(data, 50)
        p95 = np.percentile(data, 95)
        p99 = np.percentile(data, 99)
        
        print(f"{name:<30} | {avg:<10.2f} | {p10:<10.2f} | {p50:<10.2f} | {p95:<10.2f} | {p99:<10.2f}")
        
        # 繪製 CDF
        sorted_data = np.sort(data)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        plt.plot(sorted_data, yvals, label=name, color=color, linestyle=ls, linewidth=2)

    print("="*60)
    
    # 圖表美化
    plt.title("Total Delay CDF Comparison (Evaluation)")
    plt.xlabel("Total Delay (ms)")
    plt.ylabel("Cumulative Probability (CDF)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, 100) # 根據您的數據範圍調整，通常 100ms 內是重點
    
    # 儲存圖表
    plt.savefig("delay_cdf_comparison.png", dpi=300)
    print("\n圖表已儲存為 'delay_cdf_comparison.png'")
    plt.show()