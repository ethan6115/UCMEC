import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
# from stable_baselines3.common.env_checker import check_env
import cvxpy as cp

class MA_UCMEC_dyna_coop(object):
    def __init__(self, render: bool = False, seed=None):
        
        # Initialization
        self.is_mobile = True
        gym.logger.set_level(40)
        self.M = 50  # number of users
        self.N = 200  # number of APs
        self.varsig = 16  # number of antennas of each AP
        self.K = 3  # number of CPUs
        self.P_max = 0.1  # maximum transmit power of user / pilot power
        self.M_sim = 10  # number of users for simulation
        self.N_sim = 50  # number of APs for simulation 50
        self.Task_size = np.zeros([1, self.M])
        self.Task_density = np.zeros([1, self.M])
        self.cluster_matrix = None

        #加速h計算
        self.h_real = np.empty((self.M, self.N, self.varsig), dtype=np.float64)
        self.h_imag = np.empty_like(self.h_real)
        self.rng = np.random.default_rng(seed)

        # locations of users and APs
        self.locations_users = self.rng.random([self.M, 2]) * 900  # 2-D location of users
        self.locations_aps = self.rng.random([self.N, 2]) * 900  # 2-D location of APs
        # mobility
        self.user_dest = None
        self.user_speed = None


        # location of 3 CPUs
        self.locations_cpu = np.zeros([3, 2])
        self.locations_cpu[0, 0] = 300
        self.locations_cpu[0, 1] = 300
        self.locations_cpu[1, 0] = 600
        self.locations_cpu[1, 1] = 300
        self.locations_cpu[2, 0] = 450
        self.locations_cpu[2, 1] = 600
        # self.locations_cpu[3, 0] = 600
        # self.locations_cpu[3, 1] = 600

        # calculate distance between APs and users MxN matrix
        self.distance_matrix = np.zeros([self.M, self.N])
        self.distance_matrix_front = np.zeros([self.N, self.K])
        for i in range(self.M):
            for j in range(self.N):
                self.distance_matrix[i, j] = math.sqrt((self.locations_users[i, 0] - self.locations_aps[j, 0]) ** 2
                                                       + (self.locations_users[i, 1] - self.locations_aps[j, 1]) ** 2)

        for i in range(self.N):
            for j in range(self.K):
                self.distance_matrix_front[i, j] = math.sqrt((self.locations_aps[i, 0] - self.locations_cpu[j, 0]) ** 2
                                                             + (self.locations_aps[i, 1] - self.locations_cpu[
                    j, 1]) ** 2)

        # edge computing parameter
        # user parameter
        #self.C_user = self.rng.uniform(2e8, 5e8, [1, self.M])  # 根據論文修改為2e9, 5e9
        self.C_user = self.rng.uniform(2e9, 5e9, [1, self.M])  # computing resource of users  in Hz
        self.cluster_size = 5  

        # edge server parameter
        self.C_edge = self.rng.uniform(10e9, 20e9, [self.K, 1])  # computing resource of edge server in CPU #由10~20增加五倍

        # access channel parameter
        self.tau_c = 0.1  # coherence time = 100ms
        self.L = 140.7
        self.d_0 = 10  # path-loss distance threshold
        self.d_1 = 15  # path-loss distance threshold，從50改為論文的15
        self.PL = np.zeros([self.M, self.N])  # path-loss in dB
        self.beta = np.zeros([self.M, self.N])  # large scale fading
        self.gamma = np.zeros([self.M, self.N])
        self.sigma_s = 8  # standard deviation of shadow fading (dB)
        self.delta = 0.5  # parameter in Eq. (5)
        self.mu = np.zeros([self.M, self.N])  # shadow fading parameter
        self.h = np.zeros([self.M, self.N, self.varsig], dtype=complex)  # small scale fading
        self.bandwidth_a = 20e6  # bandwidth of access channel，從2改為論文的20
        self.noise_access = 3.9810717055349565e-21 * self.bandwidth_a  # noise of access channel -> -174 dbm/Hz
        self.f_carrier = 1.9e9  # carrier frequency in Hz
        self.h_ap = 15  # antenna height of AP
        self.h_user = 1.65  # antenna height of user
        # L = 46.3 + 33.9 * np.log10(f_carrier / 1000) - 13.82 * np.log10(h_ap) - (
        #        1.11 * np.log10(f_carrier / 1000) - 0.7) * h_user + 1.56 * np.log10(f_carrier / 1000) - 0.8
        self.access_chan = np.zeros([self.M, self.N, self.varsig], dtype=complex)  # complex channel

        # pathloss
        d_km = self.distance_matrix / 1000.0
        PL = np.empty_like(self.distance_matrix)
        # 建立布林遮罩
        far_mask  = self.distance_matrix > self.d_1
        mid_mask  = (self.distance_matrix >= self.d_0) & (self.distance_matrix <= self.d_1)
        near_mask = self.distance_matrix < self.d_0
        
        if np.any(far_mask):    #1. d > d1
            self.PL[far_mask] = -self.L - 35 * np.log10(d_km[far_mask])
        if np.any(mid_mask):    #2. d0 <= d <= d1
            term = (self.d_1 / 1000.0) ** 1.5 * (d_km[mid_mask] ** 2)
            self.PL[mid_mask] = -self.L - 10 * np.log10(term)
        if np.any(near_mask):   #3. d < d0
            term = (self.d_1 / 1000.0) ** 1.5 * (self.d_0 / 1000.0) ** 2
            self.PL[near_mask] = -self.L - 10 * np.log10(term)
        '''舊版
        for i in range(self.M):
            for j in range(self.N):
                # three slope path-loss model
                if self.distance_matrix[i, j] > self.d_1:
                    self.PL[i, j] = -self.L - 35 * np.log10(self.distance_matrix[i, j] / 1000)
                elif self.d_0 <= self.distance_matrix[i, j] <= self.d_1:
                    self.PL[i, j] = -self.L - 10 * np.log10(
                        (self.d_1 / 1000) ** 1.5 * (self.distance_matrix[i, j] / 1000) ** 2)
                else:
                    self.PL[i, j] = -self.L - 10 * np.log10((self.d_1 / 1000) ** 1.5 * (self.d_0 / 1000) ** 2)
        '''

        # fronthaul channel parameter
        # fronthaul channel
        # front_chan = np.zeros([N, K])
        self.bandwidth_f = 2e9  # bandwidth of fronthaul channel 2GHz?
        self.epsilon = 6e-4  # blockage density
        self.p_ap = 1  # transmit power of APs (30 dBm = 1 W)
        self.alpha_los = 2.5  # path-loss exponent for LOS links
        self.alpha_nlos = 4  # path-loss exponent for NLOS links
        self.psi_los = 3  # Nakagami fading parameter for LOS links
        self.psi_nlos = 2  # Nakagami fading parameter for NLOS links
        self.noise_front = 1.380649 * 10e-23 * 290 * 9 * self.bandwidth_f  # fronthaul channel noise variance
        self.G = np.zeros([self.N, self.K])  # random antenna gain
        self.fai = math.pi / 6  # Main lobe beamwidth
        self.Gm = 63.1  # Directivity gain of main lobes
        self.Gs = 0.631  # Directivity gain of side lobes
        self.Gain = np.array(
            [self.Gs * self.Gs, self.Gm * self.Gm, self.Gm * self.Gs])  # random antenna gain in Eq. (7)
        self.Gain_pro = np.array(
            [(self.fai / (2 * math.pi)) ** 2, 2 * self.fai * (2 * math.pi - self.fai) / (2 * math.pi) ** 2,
             ((2 * math.pi - self.fai) / (2 * math.pi)) ** 2])

        self.P_los = np.zeros([self.N, self.K])  # probability of LOS links
        self.link_type = np.zeros([self.N, self.K])  # type of fronthaul links
        for i in range(self.N):
            for j in range(self.K):
                self.P_los[i, j] = np.exp(-self.epsilon * self.distance_matrix_front[i, j] / 1000)
                self.link_type[i, j] = self.rng.choice([0, 1], p=[self.P_los[i, j],
                                                                   1 - self.P_los[i, j]])  # 0 for LOS, 1 for NLOS
                # if link_type[i, j] == 0:  # LOS link
                #     front_chan[i, j] = np.random.gamma(2, 1 / psi_los)  # Nakagami channel gain
                # else:  # NLOS link
                #     front_chan[i, j] = np.random.gamma(2, 1 / psi_nlos)  # Nakagami channel gain
                self.G[i, j] = self.rng.choice(self.Gain, p=self.Gain_pro.ravel())

        # pilot assignment
        self.tau_p = self.M  # length of pilot symbol
        self.pilot_matrix = np.zeros([self.M, self.tau_p])
        for i in range(self.M):
            self.pilot_index = i
            self.pilot_matrix[i, self.pilot_index] = 1

        # parameter init
        self.n_agents = self.M_sim
        self.agent_num = self.n_agents
        # === [coop修改 1: 擴大觀察空間維度] ===
        self.obs_dim = 5 + (self.M_sim - 1) * 3  # set the observation dimension of agents
        self.action_dim = 10
        self._render = render
        # action space: [omega_1,omega_2,...,omega_K,p]  K+1 continuous vector for each agent
        # a in {0,1,2,3,4}, p in {0, 1, 2, 3, 4} (totally 5 levels (p+1)/5*100 mW)
        self.omega_last = np.zeros([self.M_sim])
        self.p_last = np.zeros([self.M_sim])
        self.delay_last = np.zeros([self.M_sim, 1])
        self.action_space = spaces.Tuple(tuple([spaces.Discrete(10)] * self.n_agents))
        # state space: [r_1(t-1),r_2(t-1),...,r_M(t-1)]  1xM continuous vector. -> uplink rate
        # r in [0, 10e8]
        # === [coop修改 2: 擴充正規化因子] ===
        base_norm = [819200.0, 1000.0, 3.0, self.P_max, 2.0]   #對obs做正規化用的，根據論文修改100000改為819200
        neighbor_norm = [3.0, self.P_max, 2.0] * (self.M_sim - 1)   # 鄰居的資訊: [omega, p, delay] * (M_sim - 1) 人
        self.norm_factor = np.array(base_norm + neighbor_norm)
        self.obs_low = np.zeros(self.obs_dim)  # [0, 0, 0, 0, 0]
        self.obs_high = np.ones(self.obs_dim)  # [1, 1, 1, 1, 1]

        # obs = {task data size, task computing density, action index, total delay of last time slot}
        self.observation_space = spaces.Tuple(tuple(
            [spaces.Box(low=self.obs_low, high=self.obs_high, shape=(self.obs_dim,),
                        dtype=np.float32)] * self.n_agents))
        # self.np_random = None
        self.uplink_rate_access_b = np.zeros([self.M_sim, 1])
        self.step_num = 0
        
        #預先定義求解器問題
        self.opt_probs = []
        self.opt_params = []
        self.opt_vars = []
        # 預先為每個 CPU 定義一個優化問題
        for k in range(self.K):
            # 假設每個 CPU 最多可能服務所有 M_sim 個用戶
            # 定義 Parameter (數值容器)
            p_tasks = cp.Parameter(self.M_sim, nonneg=True)  # 任務大小
            p_local = cp.Parameter(self.M_sim, nonneg=True)  # 本地延遲
            p_uplink = cp.Parameter(self.M_sim, nonneg=True) # 上傳延遲
            
            # 定義 Variable (固定大小)
            C_scaled = cp.Variable(self.M_sim, nonneg=True)
            
            # 目標函數
            # 使用 multiply(p_mask, ...) 來讓沒分配的 user 不影響 Objective
            # 為了避免除以 0，分母加上微小值 1e-6
            process_delay = cp.multiply(p_tasks, cp.inv_pos(C_scaled + 1e-6))
            total_delay = cp.maximum(p_local, p_uplink + process_delay)
            
            # 這裡加入 alpha 懲罰項
            alpha = 1e-3
            objective = cp.Minimize(
                cp.sum(total_delay) + 
                alpha * cp.sum(process_delay)
            )
            
            # 限制條件：只有被 mask 選中的 user 消耗的 CPU 總量受限
            # 實際上只要限制 sum(C_scaled) 即可，因為最佳解會讓沒用到的 C 趨近 0
            max_c_scaled = self.C_edge[k, 0] / 1e9 # 假設 1e9 是你的 SCALE_FACTOR
            constraints = [cp.sum(C_scaled) <= max_c_scaled]
            
            prob = cp.Problem(objective, constraints)
            
            # 存起來備用
            self.opt_probs.append(prob)
            self.opt_params.append((p_tasks, p_local, p_uplink))
            self.opt_vars.append(C_scaled)

    def action_mapping(self, action_agent):
        omega_agent = 0
        p_agent = 0
        # Transform the action space form MultiDiscrete to Discrete (1+3*3=10 cases)
        if action_agent[0] == 1:  # local processing
            omega_agent = 0
            p_agent = 0
        elif action_agent[1] == 1:
            omega_agent = 1
            p_agent = 1
        elif action_agent[2] == 1:
            omega_agent = 1
            p_agent = 2
        elif action_agent[3] == 1:
            omega_agent = 1
            p_agent = 3
        elif action_agent[4] == 1:
            omega_agent = 2
            p_agent = 1
        elif action_agent[5] == 1:
            omega_agent = 2
            p_agent = 2
        elif action_agent[6] == 1:
            omega_agent = 2
            p_agent = 3
        elif action_agent[7] == 1:
            omega_agent = 3
            p_agent = 1
        elif action_agent[8] == 1:
            omega_agent = 3
            p_agent = 2
        elif action_agent[9] == 1:
            omega_agent = 3
            p_agent = 3
        return omega_agent, p_agent
    
    def cluster(self):  #改
        cluster_matrix = np.zeros([self.M_sim, self.N_sim], dtype=int)
        ap_index_list = np.zeros([self.M_sim, self.cluster_size], dtype=int)

        for i in range(self.M_sim):
            # 只對第 i 個 user 的前 N_sim 個 AP 做排序
            # 注意：beta 的大小是 M x N (full)，我們要取前 N_sim
            sorted_idx = np.argsort(self.beta[i, :self.N_sim])  # 小到大
            sorted_idx = sorted_idx[::-1]                       # 反轉成大到小
            chosen = sorted_idx[:self.cluster_size]             # 取前 cluster_size 個 AP index
            ap_index_list[i, :] = chosen
            for k_idx in chosen:
                cluster_matrix[i, int(k_idx)] = 1
        return cluster_matrix

    def uplink_rate_cal(self, p, omega, cluster_matrix, theta):  # calculate the uplink transmit rate in Eq. (12) 改成新版

        uplink_rate_access = np.zeros([self.M_sim, 1])
        for i in range(self.M_sim):
            if omega[i] == 0:
                continue
            # useful signal and noise accumulation
            sum_theta = 0.0
            noise_term = 0.0
            for j in range(self.N_sim):
                if cluster_matrix[i, j] == 1:
                    sum_theta += theta[i, j]
                    noise_term += self.noise_access * theta[i, j]

            # useful (magnitude)^2 * p * varsig  (保留原公式)
            useful = (sum_theta ** 2) * p[i] * self.varsig

            # interference from other users
            inter_term = 0.0
            for k in range(self.M_sim):
                if k == i or omega[k] == 0:
                    continue
                for j in range(self.N_sim):
                    if cluster_matrix[i, j] == 1:
                        inter_term += theta[i, j] * self.beta[k, j] * p[k]

            SINR = useful / (inter_term + noise_term)
            raw_rate = self.bandwidth_a * np.log2(1 + SINR)
            uplink_rate_access[i, 0] = max(raw_rate, 1e-9)  # clip，避免 0

        return uplink_rate_access

    def front_rate_cal(self, omega, cluster_matrix):
        chi = np.zeros([self.N_sim, self.K])  # whether an AP transmit symbol to a CPU or not
        SINR_front = np.zeros([self.N_sim, self.K])  # SINR in Eq. (7)
        front_rate = np.zeros([self.N_sim, self.K])  # Eq. (12)
        front_rate_user = np.zeros([self.M_sim, self.N_sim])
        I_sum = 0  # total sum of fronthaul interference
        for i in range(self.M_sim):
            if omega[i] == 0:
                continue
            CPU_id = int(omega[i] - 1)
            for j in range(self.N_sim):
                if cluster_matrix[i, j] == 1:  # This AP is belonged to the cluster of user i
                    chi[j, CPU_id] = 1

        for i in range(self.N_sim):
            for j in range(self.K):
                if chi[i, j] == 1:
                    if self.link_type[i, j] == 0:  # LOS link 從[j, j] 改成[i, j]
                        I_sum = I_sum + self.p_ap * pow(self.distance_matrix_front[i, j] / 1000, -self.alpha_los)
                    else:
                        I_sum = I_sum + self.p_ap * pow(self.distance_matrix_front[i, j] / 1000, -self.alpha_nlos)
                else:
                    pass

        for i in range(self.N_sim):
            for j in range(self.K):
                if chi[i, j] == 1:
                    if self.link_type[i, j] == 0:  # LOS link
                        SINR_front_mole = self.p_ap * self.G[i, j] * pow(self.distance_matrix_front[i, j] / 1000,
                                                                         -self.alpha_los)
                    else:
                        SINR_front_mole = self.p_ap * self.G[i, j] * pow(self.distance_matrix_front[i, j] / 1000,
                                                                         -self.alpha_nlos)  #改成負號
                    SINR_front[i, j] = SINR_front_mole / (I_sum - SINR_front_mole / self.G[i, j] + self.noise_front)
                    front_rate[i, j] = self.bandwidth_f * np.log2(1 + SINR_front[i, j])

        for i in range(self.M_sim):
            if omega[i] == 0:
                pass
            else:
                CPU_id = int(omega[i] - 1)
                for j in range(self.N_sim):
                    if cluster_matrix[i, j] == 1:
                        front_rate_user[i, j] = front_rate[j, CPU_id]

        return front_rate_user

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.rng = np.random.default_rng(seed)  #固定亂數產生的seed
        return [seed]

    def render(self, mode='human'):
        pass

    def reset(self):
        '''
        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_obs = self.rng.uniform(low=self.obs_low, high=self.obs_high, size=(self.obs_dim,))
            sub_agent_obs.append(sub_obs)
        '''
        #重製mobility
        if self.is_mobile:
            max_speed = 20 * self.tau_c     #根據論文改成10~20
            min_speed = 10 * self.tau_c

            # 每個 user 的當前 waypoint
            self.user_dest = self.rng.random((self.M, 2)) * 900  

            # 每個 user 的速度（整段 waypoint 期間固定）
            self.user_speed = self.rng.uniform(min_speed, max_speed, (self.M, 1))
        else:
            self.user_dest = None
            self.user_speed = None
        '''
        #每episode固定shadowing
        kappa_1 = self.rng.standard_normal((1, self.N))
        kappa_2 = self.rng.standard_normal((self.M, 1))
        self.mu = np.sqrt(self.delta) * kappa_1 + np.sqrt(1 - self.delta) * kappa_2
        '''
        # 回傳observation
        self.step_num = 0 
        self.Task_size = self.rng.uniform(409600, 819200, [1, self.M])  # 單位從KB改成bits，根據論文修改
        #self.Task_size = self.rng.uniform(50000, 100000, [1, self.M])
        self.Task_density = self.rng.uniform(500, 1000, [1, self.M])
        
        sub_agent_obs = []
        for i in range(self.agent_num):
            # === [coop修改 3: 初始化 Observation 補零] ===
            raw_obs = np.zeros(self.obs_dim)
            raw_obs[0] = self.Task_size[0, i]
            raw_obs[1] = self.Task_density[0, i]
            norm_obs = raw_obs / self.norm_factor #正規化
            sub_agent_obs.append(norm_obs)
            
        return sub_agent_obs

    def step(self, action):
        self.step_num += 1

        # mobility 改成正規RWP
        if self.is_mobile:
        # 根據論文修改速度 5–15 改成 10–20 m/s
            max_speed = 20 * self.tau_c  # (m / tau_c second)
            min_speed = 10 * self.tau_c  

            # 正統 RWP：每個 user 往 waypoint 走，走到才換新 waypoint
            for i in range(self.M):
                # 目前 waypoint 與位置的位移向量
                dx = self.user_dest[i, 0] - self.locations_users[i, 0]
                dy = self.user_dest[i, 1] - self.locations_users[i, 1]
                dist = math.sqrt(dx * dx + dy * dy)

                # 該 user 本 step 可走的距離（速度單位 = m / tau_c，一個 step 就走這麼多）
                step_dist = self.user_speed[i, 0]

                if dist <= 1e-6:
                    # 已經在 waypoint 上（或數值上非常接近）：直接抽新 waypoint + 速度，下一步再走
                    self.user_dest[i, :] = self.rng.random(2) * 900
                    self.user_speed[i, 0] = self.rng.uniform(min_speed, max_speed)
                    continue

                if step_dist >= dist:
                    # 這一步就能走到 waypoint，直接拉到目的地
                    self.locations_users[i, 0] = self.user_dest[i, 0]
                    self.locations_users[i, 1] = self.user_dest[i, 1]

                    # 抵達後，立刻為「下一個段落」抽新的 waypoint 和速度
                    self.user_dest[i, :] = self.rng.random(2) * 900
                    self.user_speed[i, 0] = self.rng.uniform(min_speed, max_speed)
                else:
                    # 還沒到 waypoint，沿著目前方向走 step_dist
                    dir_x = dx / dist
                    dir_y = dy / dist
                    self.locations_users[i, 0] += dir_x * step_dist
                    self.locations_users[i, 1] += dir_y * step_dist

            # 把 user 位置拉回900內
            self.locations_users = np.clip(self.locations_users, 0, 900)

        # distance
        diff = self.locations_users[:, np.newaxis, :] - self.locations_aps[np.newaxis, :, :]
        self.distance_matrix = np.sqrt(np.sum(diff**2, axis=2))
        '''舊版
        for i in range(self.M):
            for j in range(self.N):
                self.distance_matrix[i, j] = math.sqrt((self.locations_users[i, 0] - self.locations_aps[j, 0]) ** 2
                                                       + (self.locations_users[i, 1] - self.locations_aps[j, 1]) ** 2)
        '''
        # pathloss
        d_km = self.distance_matrix / 1000.0
        PL = np.empty_like(self.distance_matrix)
        # 建立布林遮罩
        far_mask  = self.distance_matrix > self.d_1
        mid_mask  = (self.distance_matrix >= self.d_0) & (self.distance_matrix <= self.d_1)
        near_mask = self.distance_matrix < self.d_0

        if np.any(far_mask):    #1. d > d1
            self.PL[far_mask] = -self.L - 35 * np.log10(d_km[far_mask])
        if np.any(mid_mask):    #2. d0 <= d <= d1
            term = (self.d_1 / 1000.0) ** 1.5 * (d_km[mid_mask] ** 2)
            self.PL[mid_mask] = -self.L - 10 * np.log10(term)
        if np.any(near_mask):   #3. d < d0
            term = (self.d_1 / 1000.0) ** 1.5 * (self.d_0 / 1000.0) ** 2
            self.PL[near_mask] = -self.L - 10 * np.log10(term)
        


        '''舊版
        for i in range(self.M):
            for j in range(self.N):
                # three slope path-loss model
                if self.distance_matrix[i, j] > self.d_1:
                    self.PL[i, j] = -self.L - 35 * np.log10(self.distance_matrix[i, j] / 1000)
                elif self.d_0 <= self.distance_matrix[i, j] <= self.d_1:
                    self.PL[i, j] = -self.L - 10 * np.log10(
                        (self.d_1 / 1000) ** 1.5 * (self.distance_matrix[i, j] / 1000) ** 2)
                else:
                    self.PL[i, j] = -self.L - 10 * np.log10((self.d_1 / 1000) ** 1.5 * (self.d_0 / 1000) ** 2)
        '''
        # access channel
        '''時間瓶頸，改掉
        kappa_1 = self.rng.random(1, self.N)  # parameter in Eq. (5)
        kappa_2 = self.rng.random(1, self.M)  # parameter in Eq. (5)
        
        for i in range(self.M):
            for j in range(self.N):
                # Eq. (5) shadow fading computation
                self.mu[i, j] = math.sqrt(self.delta) * kappa_1[0, j] + math.sqrt(1 - self.delta) * kappa_2[
                    0, i]  # MxN matrix as Eq. (5)

                # Eq. (2) channel computation
                self.beta[i, j] = pow(10, self.PL[i, j] / 10) * pow(10, (self.sigma_s * self.mu[i, j]) / 10)
                for k in range(self.varsig):
                    self.h[i, j, k] = self.rng.normal(loc=0, scale=0.5) + 1j * self.rng.normal(loc=0, scale=0.5)
                    self.access_chan[i, j, k] = np.sqrt(self.beta[i, j]) * self.h[i, j, k]
        '''
        #測試固定shadowing
        
        # 1. 生成隨機參數 (一次生成整個矩陣，取代迴圈內生成)
        kappa_1 = self.rng.standard_normal((1, self.N))  # 形狀: (1, N)    從rand改成randn，才符合論文的公式
        kappa_2 = self.rng.standard_normal((self.M, 1))  # 形狀: (M, 1)，轉置以便廣播  從rand改成randn，才符合論文的公式

        # 2. 計算 Shadow Fading (mu) - 利用 Broadcasting
        # (1, N) 與 (M, 1) 運算會自動廣播成 (M, N) 矩陣
        self.mu = np.sqrt(self.delta) * kappa_1 + np.sqrt(1 - self.delta) * kappa_2
        
        # 3. 計算 Large Scale Fading (beta) - 矩陣直接運算
        # self.PL 和 self.mu 都是 (M, N) 矩陣，直接進行元素級運算
        self.beta = np.power(10, self.PL / 10.0) * np.power(10, (self.sigma_s * self.mu) / 10.0)


        # 4. 計算 Small Scale Fading (h) - 一次生成所有亂數
        # 形狀: (M, N, varsig)
        #self.h_real = self.rng.normal(loc=0, scale=0.5, size=(self.M, self.N, self.varsig))
        #self.h_imag = self.rng.normal(loc=0, scale=0.5, size=(self.M, self.N, self.varsig))
        #加速h計算
        self.rng.standard_normal(size=(self.M, self.N, self.varsig), dtype=np.float64, out=self.h_real)
        self.rng.standard_normal(size=(self.M, self.N, self.varsig), dtype=np.float64, out=self.h_imag)
        self.h_real *= 0.5
        self.h_imag *= 0.5

        self.h = self.h_real + 1j * self.h_imag

        

        # 5. 計算 Access Channel - 利用 Broadcasting
        # self.beta 形狀是 (M, N)，需要擴展維度變成 (M, N, 1) 才能跟 (M, N, varsig) 的 h 相乘
        self.access_chan = np.sqrt(self.beta)[:, :, np.newaxis] * self.h
        
        # MMSE channel estimation(一樣改為numpy版本)
        '''
        theta = np.zeros([self.M, self.N])
        for i in range(self.M):
            for j in range(self.N):
                theta[i, j] = self.tau_p * self.P_max * (self.beta[i, j] ** 2) / (
                        self.tau_p * self.P_max * self.beta[i, j] + self.noise_access)
        '''
        theta = (self.tau_p * self.P_max * (self.beta ** 2)) / (self.tau_p * self.P_max * self.beta + self.noise_access)

        
        

        # 根據論文將cluster改為每10個time slot做一次
        if self.step_num == 1 or self.step_num % 10 == 0:
            self.cluster_matrix = self.cluster()
        cluster_matrix = self.cluster_matrix
        
        #cluster_matrix = self.cluster()

        # obtain the action
        omega_current = np.zeros([self.M_sim])
        p_current = np.zeros([self.M_sim])
        p_level = self.P_max / 4
        

        for i in range(self.M_sim):
            omega_current[i], p_current_idx = self.action_mapping(action[i])
            # --- [修正：確保數據一致性] ---
            if omega_current[i] == 0:
                p_current[i] = 0.0  # 讓 Agent 明確看到 "0"
            else:
                p_current[i] = (p_current_idx + 1) * p_level
        # print("Chosen CPU ID:", omega_current)
        # print("Power:", p_current)

        #計算速率
        uplink_rate_access = self.uplink_rate_cal(p_current, omega_current, cluster_matrix, theta)
        front_rate_user = self.front_rate_cal(omega_current, cluster_matrix)
        self.uplink_rate_access_b = uplink_rate_access
        # print("Fronthaul Rate", front_rate_user)
        # print("Uplink Rate (Mbps):", uplink_rate_access / 10e6)
        # print("Average Uplink Rate (Mbps):", np.sum(uplink_rate_access) / (np.count_nonzero(omega_current) * 10e6))

        # local computing delay
        local_delay = np.zeros([self.M_sim, 1])
        for i in range(self.M_sim):
            if omega_current[i] == 0:
                local_delay[i, 0] = self.Task_density[0, i] * self.Task_size[0, i] / self.C_user[0, i]

        # uplink delay
        uplink_delay = np.zeros([self.M_sim, 1])
        for i in range(self.M_sim): #原寫法不合理
            if omega_current[i] != 0 and uplink_rate_access[i, 0] > 0:
                uplink_delay[i, 0] = self.Task_size[0, i] / uplink_rate_access[i, 0]
            else:
                uplink_delay[i, 0] = 0.0

        # fronthaul delay
        front_delay = np.zeros([self.M_sim, 1])
        for i in range(self.M_sim): #原寫法不合理
            if omega_current[i] != 0:
                ap_idx = np.where(cluster_matrix[i, :] == 1)[0]
                # 算出每個 AP 的 fronthaul delay
                delays = []
                for j in ap_idx:
                    if front_rate_user[i, j] > 0:
                        delays.append(self.Task_size[0, i] / front_rate_user[i, j])
                if len(delays) > 0:
                    front_delay[i, 0] = np.max(delays)
                else:
                    front_delay[i, 0] = 0.0

        # processing delay calculation
        # solve convex problem according to Eq. (24)
        task_mat = np.zeros([self.M_sim, self.K])
        for i in range(self.M_sim):
            if omega_current[i] != 0:
                CPU_id = int(omega_current[i] - 1)
                task_mat[i, CPU_id] = self.Task_size[0, i] * self.Task_density[0, i]

        # Each CPU solves a resource allocation optimization problem
        actual_C = np.zeros([self.M_sim, self.K])
        # 在 step 迴圈中替換原有程式碼
        SCALE_FACTOR = 1e9
        for i in range(self.K):
            # 1. 準備數據 (Vectorization)
            mask_val = np.zeros(self.M_sim)
            task_val = np.zeros(self.M_sim)
            local_val = np.zeros(self.M_sim)
            uplink_val = np.zeros(self.M_sim)
            
            # 填入目前分配給 CPU i 的 user 數據
            # 這裡假設你有個邏輯能快速找出哪些 user 在這個 CPU (例如透過 omega_current)
            # 範例邏輯：
            has_user = False
            for u_idx in range(self.M_sim):
                if task_mat[u_idx, i] > 0: # 判斷該 user 是否分配給此 CPU
                    mask_val[u_idx] = 1.0
                    task_val[u_idx] = task_mat[u_idx, i] / SCALE_FACTOR
                    local_val[u_idx] = local_delay[u_idx, 0]
                    uplink_val[u_idx] = uplink_delay[u_idx, 0]
                    has_user = True
                    
            if not has_user:
                continue

            p_tasks, p_local, p_uplink = self.opt_params[i]
            #利用 mask_val 把無效用戶的數據全部歸零
            task_val = task_val * mask_val
            local_val = local_val * mask_val
            uplink_val = uplink_val * mask_val
            # 2. 更新 Parameter
            
            p_tasks.value = task_val
            p_local.value = local_val
            p_uplink.value = uplink_val
            
            # 3. 求解 (Warm Start 加速)
            try:
                # 放寬 tolerance 到 1e-2 或 1e-3 對 RL 訓練通常沒影響，但速度快很多
                self.opt_probs[i].solve(solver=cp.CLARABEL, warm_start=True, 
                                        tol_gap_abs=1e-2, tol_gap_rel=1e-2, verbose=False)
                
                # 4. 取回結果
                c_res = self.opt_vars[i].value
                if c_res is not None:
                    actual_C[:, i] = c_res * SCALE_FACTOR
            except cp.error.SolverError:
                # Fallback 邏輯
                pass

        actual_process_delay = np.zeros([self.M_sim, 1])
        for i in range(self.M_sim):
            if omega_current[i] != 0:
                CPU_id = int(omega_current[i] - 1)
                actual_process_delay[i, 0] = task_mat[i, CPU_id] / np.sum(actual_C[i, :])
        '''
        process_delay = cp.max(cp.multiply(task_mat, cp.inv_pos(C)))  # Mx1
        func = cp.Minimize(cp.sum(cp.maximum(local_delay, front_delay + uplink_delay + process_delay)))
        # func = cp.Minimize(cp.sum(cp.maximum(local_delay, process_delay)))
        cons = [0 <= C]
        for i in range(K):
            cons += [cp.sum(C[:, i]) <= C_edge[i, 0]]

        prob = cp.Problem(func, cons)
        prob.solve(solver=cp.SCS, verbose=False)
        actual_C = C.value
        actual_process_delay = np.max(task_mat / actual_C, axis=1)
        # print(actual_process_delay)
        # print(C.value)
        '''

        # # reward calculation
        # print("Uplink Delay:", uplink_delay)
        # print("Local Delay:", local_delay)
        # print("Front Delay:", front_delay)
        # print("Edge Processing Delay:", actual_process_delay)
        # print("Offloading Delay:", front_delay + uplink_delay + actual_process_delay)
        total_delay = np.zeros([self.M_sim, 1])
        for i in range(self.M_sim):
            total_delay[i, 0] = np.maximum(local_delay[i, 0],
                                           front_delay[i, 0] + uplink_delay[i, 0] + actual_process_delay[i, 0])
        max_delay = 2.0  # 超過 2 秒視為「同樣很爛」，避免 reward 爆
        total_delay = np.minimum(total_delay, max_delay)


        if self.step_num >= 200:    #>20改>=200
            done = [1] * self.M_sim
        else:
            done = [0] * self.M_sim

        # === [coop修改 4: 更改 Reward 計算 (合作模式)] ===
        avg_system_delay = np.mean(total_delay)
        reward = np.zeros([self.M_sim, 1])
        for i in range(self.M_sim):
            reward[i, 0] = -0.9 * avg_system_delay + 0.1 * (self.tau_c - avg_system_delay)  #原來的reward
        
        # === 每個 time step 的統計量 (之後會塞進 info) ===
        # Average Total Delay (所有 user)
        avg_total_delay_ms = float(np.mean(total_delay) * 1000.0)

        # Local user：omega_current == 0
        local_mask = (omega_current == 0)
        if np.any(local_mask):
            avg_local_delay_ms = float(np.mean(local_delay[local_mask, 0]) * 1000.0)
        else:
            avg_local_delay_ms = 0.0
        # Offloading user：omega_current != 0
        offload_mask = (omega_current != 0)
        active = int(np.count_nonzero(offload_mask))  # Offloading user 數量
        if active > 0:
            avg_uplink_delay_ms = float(np.mean(uplink_delay[offload_mask, 0]) * 1000.0)
            avg_front_delay_ms = float(np.mean(front_delay[offload_mask, 0]) * 1000.0)
            avg_actual_process_delay_ms = float(np.mean(actual_process_delay[offload_mask, 0]) * 1000.0)
            avg_uplink_rate_Mbps = float(np.mean(uplink_rate_access[offload_mask, 0]) / 1e6)
        else:
            avg_uplink_delay_ms = 0.0
            avg_front_delay_ms = 0.0
            avg_actual_process_delay_ms = 0.0
            avg_uplink_rate_Mbps = 0.0

        # 保留原本每 200 step print 的 debug（用上面已經算好的統計量）
        if self.step_num % 200 == 0:
            print("Step Index:", self.step_num)
            print("Average Total Delay (ms):", avg_total_delay_ms)

            if np.any(local_mask):
                print("Average Local Delay (ms):", avg_local_delay_ms)
            else:
                print("Average Local Delay (ms): 0 (no local users)")

            if active > 0:
                print("Average Uplink Delay (ms):", avg_uplink_delay_ms)
                print("Average Front Delay (ms):", avg_front_delay_ms)
                print("Average Actual Process Delay (ms):", avg_actual_process_delay_ms)
                print("Average Uplink Rate (Mbps):", avg_uplink_rate_Mbps)
                print("Offloading user", active)
            else:
                print("No Offloading Users")

        # task parameter
        #Task_size_next = self.rng.uniform(50000, 100000, [1, self.M])  # task size in bit
        Task_size_next = self.rng.uniform(409600, 819200, [1, self.M])  # 單位從KB改成bits，根據論文修改
        Task_density_next = self.rng.uniform(500, 1000, [1, self.M])  # task density cpu cycles per bit
        # Task_max_delay = self.rng.uniform(2, 5, [1, M])  # task max delay in second
        # 更新 self，給下一次 step 用
        self.Task_size = Task_size_next
        self.Task_density = Task_density_next

        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        self.delay_last = total_delay
        self.omega_last = omega_current
        self.p_last = p_current
        metrics_info = {
            "avg_total_delay_ms": avg_total_delay_ms,
            "avg_local_delay_ms": avg_local_delay_ms,
            "avg_uplink_delay_ms": avg_uplink_delay_ms,
            "avg_front_delay_ms": avg_front_delay_ms,
            "avg_actual_process_delay_ms": avg_actual_process_delay_ms,
            "avg_uplink_rate_Mbps": avg_uplink_rate_Mbps,
            "num_offloading_users": active
        }
        for i in range(self.agent_num):
            # === [coop修改 5: 構建包含鄰居資訊的 Observation] ===
            # 自己的資訊
            my_obs = [              
                self.Task_size[0, i],
                self.Task_density[0, i],
                self.omega_last[i],
                self.p_last[i],
                self.delay_last[i, 0]
            ]
            # 其他人的資訊
            neighbor_obs = []
            for j in range(self.agent_num):
                if i == j:
                    continue
                # 加入其他人的 omega, p, delay
                neighbor_obs.extend([
                    self.omega_last[j],
                    self.p_last[j],
                    self.delay_last[j, 0]
                ])
            #合併後正規化
            raw_obs = np.array(my_obs + neighbor_obs)
            norm_obs = raw_obs / self.norm_factor #正規化
            sub_agent_obs.append(norm_obs)
            sub_agent_reward.append(reward[i])
            sub_agent_done.append(done[i])
            #sub_agent_info.append({})
            # 只在第 0 個 agent 的 info 塞統計量，其它保持空 dict
            if i == 0:
                sub_agent_info.append(metrics_info)
            else:
                sub_agent_info.append({})

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]


if __name__ == "__main__":
    env = MA_UCMEC_dyna_coop(render=False)
    # check_env(env)
    obs = env.reset()
    episode = 5
    for _ in range(episode):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if np.all(done):
            obs = env.reset()
        # print(f"state: {obs} \n")
        print(f"action : {action}, reward : {reward}")
