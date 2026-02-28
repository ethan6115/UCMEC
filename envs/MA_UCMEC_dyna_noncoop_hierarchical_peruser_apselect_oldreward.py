import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
import cvxpy as cp

class MA_UCMEC_dyna_noncoop_hierarchical_peruser(object):
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
        # self.h_real = np.empty((self.M, self.N, self.varsig), dtype=np.float64)
        # self.h_imag = np.empty_like(self.h_real)
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
        self.current_cluster_size = np.full(self.M_sim, 5, dtype=np.int32)

        # edge server parameter
        self.C_edge = self.rng.uniform(30e9, 50e9, [self.K, 1])  # computing resource of edge server in CPU #由10~20增加五倍

        # access channel parameter
        self.tau_c = 0.1  # coherence time = 100ms
        self.L = 140.7
        self.d_0 = 10  # path-loss distance threshold
        self.d_1 = 200  # path-loss distance threshold，從50改為論文的15
        self.PL = np.zeros([self.M, self.N])  # path-loss in dB
        self.beta = np.zeros([self.M, self.N])  # large scale fading
        self.gamma = np.zeros([self.M, self.N])
        self.sigma_s = 8  # standard deviation of shadow fading (dB)
        self.delta = 0.5  # parameter in Eq. (5)
        self.mu = np.zeros([self.M, self.N])  # shadow fading parameter
        self.h = np.zeros([self.M, self.N, self.varsig], dtype=complex)  # small scale fading
        self.bandwidth_a = 20e6  # bandwidth of access channel，從2改為論文的20 #嘗試調整成comm limit，20改為10
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
        
        # fronthaul channel parameter
        # fronthaul channel
        # front_chan = np.zeros([N, K])
        self.bandwidth_f = 2e9  # bandwidth of fronthaul channel 2GHz?  #嘗試調整成comm limit，2改為1
        self.epsilon = 6e-4  # blockage density
        self.p_ap = 1  # transmit power of APs (30 dBm = 1 W)
        self.alpha_los = 2.5  # path-loss exponent for LOS links
        self.alpha_nlos = 4  # path-loss exponent for NLOS links
        self.psi_los = 3  # Nakagami fading parameter for LOS links
        self.psi_nlos = 2  # Nakagami fading parameter for NLOS links
        self.noise_front = 1.380649 * 1e-23 * 290 * 9 * self.bandwidth_f  # fronthaul channel noise variance
        self.G = np.zeros([self.N, self.K])  # random antenna gain
        self.fai = math.pi / 6  # Main lobe beamwidth
        self.Gm = 63.1  # Directivity gain of main lobes
        self.Gs = 0.631  # Directivity gain of side lobes
        self.Gain = np.array(
            [self.Gm * self.Gm, self.Gm * self.Gs, self.Gs * self.Gs])  # random antenna gain in Eq. (7)
        self.Gain_pro = np.array(
            [(self.fai / (2 * math.pi)) ** 2, 2 * self.fai * (2 * math.pi - self.fai) / (2 * math.pi) ** 2,
             ((2 * math.pi - self.fai) / (2 * math.pi)) ** 2])

        self.P_los = np.zeros([self.N, self.K])  # probability of LOS links
        self.link_type = np.zeros([self.N, self.K])  # type of fronthaul links
        '''
        for i in range(self.N):
            for j in range(self.K):
                self.P_los[i, j] = np.exp(-self.epsilon * self.distance_matrix_front[i, j])
                self.link_type[i, j] = self.rng.choice([0, 1], p=[self.P_los[i, j],
                                                                   1 - self.P_los[i, j]])  # 0 for LOS, 1 for NLOS
                # if link_type[i, j] == 0:  # LOS link
                #     front_chan[i, j] = np.random.gamma(2, 1 / psi_los)  # Nakagami channel gain
                # else:  # NLOS link
                #     front_chan[i, j] = np.random.gamma(2, 1 / psi_nlos)  # Nakagami channel gain
                self.G[i, j] = self.rng.choice(self.Gain, p=self.Gain_pro.ravel())
        '''
        # pilot assignment
        self.tau_p = self.M  # length of pilot symbol
        self.pilot_matrix = np.zeros([self.M, self.tau_p])
        for i in range(self.M):
            self.pilot_index = i
            self.pilot_matrix[i, self.pilot_index] = 1

        # parameter init
        self.n_agents = self.M_sim
        self.agent_num = self.n_agents
        self.obs_dim = 6  # set the observation dimension of agents 
        self.action_dim = 10
        self._render = render

        # High-level observation: top-10 beta per agent + last cluster size + per-user delay/uplink/front + position/speed.
        self.max_delay = 1.0
        self.cluster_size_candidates = list(range(1, 11))
        # High-level action: per-user binary mask over top-10 APs.
        self.high_action_space = spaces.MultiBinary((self.M_sim, 10))
        self.high_action_dim = 10
        self.high_obs_dim = 48
        self.high_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.M_sim, self.high_obs_dim), dtype=np.float32
        )
        self._top10_ap_idx = None
        self._segment_delay_sum = np.zeros((self.M_sim,), dtype=np.float32)
        self._segment_uplink_sum = np.zeros((self.M_sim,), dtype=np.float32)
        self._segment_front_sum = np.zeros((self.M_sim,), dtype=np.float32)
        self._segment_delay_count = 0
        self._segment_avg_delay = np.zeros((self.M_sim,), dtype=np.float32)
        self._segment_avg_uplink = np.zeros((self.M_sim,), dtype=np.float32)
        self._segment_avg_front = np.zeros((self.M_sim,), dtype=np.float32)
        self._segment_offload_count = 0
        self._segment_offload_success_count = 0
        self._segment_offload_success_ratio = 0.0
        self._pending_high_action = None #記住高層動作
        self._channel_ready = False
        self.theta_current = None
        self.last_cluster_size = self.current_cluster_size.copy()
        # action space: [omega_1,omega_2,...,omega_K,p]  K+1 continuous vector for each agent
        # a in {0,1,2,3,4}, p in {0, 1, 2, 3, 4} (totally 5 levels (p+1)/5*100 mW)
        self.omega_last = np.zeros([self.M_sim])
        self.p_last = np.zeros([self.M_sim])
        self.p_idx_last = np.zeros([self.M_sim], dtype=np.int32)
        self.delay_last = np.zeros([self.M_sim, 1])
        self.action_space = spaces.Tuple(tuple([spaces.Discrete(10)] * self.n_agents))
        # state space: [r_1(t-1),r_2(t-1),...,r_M(t-1)]  1xM continuous vector. -> uplink rate
        # r in [0, 10e8]
        self.norm_factor = np.array([819200.0*5, 1000.0/5, 3.0, self.P_max, 2.0, 10.0])   #對obs做正規化用的，根據論文修改100000改為819200
        self.obs_low = np.zeros(self.obs_dim)  # [0, 0, 0, 0, 0, 0]
        self.obs_high = np.ones(self.obs_dim)  # [1, 1, 1, 1, 1, 1]
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
            
            objective = cp.Minimize(cp.sum(total_delay))    #不用懲罰項
            
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
    
    # [新增] 上層控制接口
    def set_high_action(self, action_id):
        # Reason: accept per-user binary masks over top-10 APs.
        if isinstance(action_id, (list, np.ndarray)):
            action_id = np.asarray(action_id, dtype=np.int32)
        else:
            action_id = np.full((self.M_sim, 10), int(action_id), dtype=np.int32)
        self._pending_high_action = action_id

    def apply_high_action(self, action_id=None):
        if action_id is None:
            if self._pending_high_action is None:
                return
            action_id = self._pending_high_action
        if self._top10_ap_idx is None:
            return
        action_id = np.asarray(action_id, dtype=np.int32)
        if action_id.ndim == 1:
            action_id = action_id.reshape(self.M_sim, 10)
        cluster_matrix = np.zeros((self.M_sim, self.N_sim), dtype=int)
        for i in range(self.M_sim):
            mask = action_id[i] != 0
            if not np.any(mask):
                mask[0] = True
            ap_idx = self._top10_ap_idx[i][mask]
            for ap in ap_idx:
                cluster_matrix[i, int(ap)] = 1
        self.cluster_matrix = cluster_matrix
        self.current_cluster_size = np.sum(cluster_matrix, axis=1).astype(np.int32)
        self.last_cluster_size = self.current_cluster_size.copy()
        self._segment_delay_sum = np.zeros((self.M_sim,), dtype=np.float32)
        self._segment_uplink_sum = np.zeros((self.M_sim,), dtype=np.float32)
        self._segment_front_sum = np.zeros((self.M_sim,), dtype=np.float32)
        self._segment_delay_count = 0
        self._segment_avg_delay = np.zeros((self.M_sim,), dtype=np.float32)
        self._segment_avg_uplink = np.zeros((self.M_sim,), dtype=np.float32)
        self._segment_avg_front = np.zeros((self.M_sim,), dtype=np.float32)
        self._segment_offload_count = 0
        self._segment_offload_success_count = 0
        self._segment_offload_success_ratio = 0.0
        self._pending_high_action = None

    def advance_channel(self):
        # Update positions, pathloss, and channel for the current slot.
        if self.is_mobile:
            max_speed = 20 * self.tau_c
            min_speed = 10 * self.tau_c
            for i in range(self.M):
                dx = self.user_dest[i, 0] - self.locations_users[i, 0]
                dy = self.user_dest[i, 1] - self.locations_users[i, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                step_dist = self.user_speed[i, 0]
                if dist <= 1e-6:
                    self.user_dest[i, :] = self.rng.random(2) * 900
                    self.user_speed[i, 0] = self.rng.uniform(min_speed, max_speed)
                    continue
                if step_dist >= dist:
                    self.locations_users[i, 0] = self.user_dest[i, 0]
                    self.locations_users[i, 1] = self.user_dest[i, 1]
                    self.user_dest[i, :] = self.rng.random(2) * 900
                    self.user_speed[i, 0] = self.rng.uniform(min_speed, max_speed)
                else:
                    dir_x = dx / dist
                    dir_y = dy / dist
                    self.locations_users[i, 0] += dir_x * step_dist
                    self.locations_users[i, 1] += dir_y * step_dist
            self.locations_users = np.clip(self.locations_users, 0, 900)

        diff = self.locations_users[:, np.newaxis, :] - self.locations_aps[np.newaxis, :, :]
        self.distance_matrix = np.sqrt(np.sum(diff**2, axis=2))

        d_km = self.distance_matrix / 1000.0
        PL = np.empty_like(self.distance_matrix)
        far_mask = self.distance_matrix > self.d_1
        mid_mask = (self.distance_matrix >= self.d_0) & (self.distance_matrix <= self.d_1)
        near_mask = self.distance_matrix < self.d_0
        if np.any(far_mask):
            self.PL[far_mask] = -self.L - 35 * np.log10(d_km[far_mask])
        if np.any(mid_mask):
            term = (self.d_1 / 1000.0) ** 1.5 * (d_km[mid_mask] ** 2)
            self.PL[mid_mask] = -self.L - 10 * np.log10(term)
        if np.any(near_mask):
            term = (self.d_1 / 1000.0) ** 1.5 * (self.d_0 / 1000.0) ** 2
            self.PL[near_mask] = -self.L - 10 * np.log10(term)

        self.beta = np.power(10, self.PL / 10.0) * np.power(10, (self.sigma_s * self.mu) / 10.0)
        '''沒有用到
        self.rng.standard_normal(size=(self.M, self.N, self.varsig), dtype=np.float64, out=self.h_real)
        self.rng.standard_normal(size=(self.M, self.N, self.varsig), dtype=np.float64, out=self.h_imag)
        self.h_real *= 0.5
        self.h_imag *= 0.5
        self.h = self.h_real + 1j * self.h_imag
        
        self.access_chan = np.sqrt(self.beta)[:, :, np.newaxis] * self.h
        '''
        self.theta_current = (self.tau_p * self.P_max * (self.beta ** 2)) / (
            self.tau_p * self.P_max * self.beta + self.noise_access
        )
        self._channel_ready = True
        return self.get_global_obs()
        

    # [新增] 全域觀測接口
    def get_global_obs(self):
        beta_top10 = []
        top10_idx_all = np.zeros((self.M_sim, 10), dtype=np.int32)
        front_stats = np.zeros((self.M_sim, 30), dtype=np.float32)
        for i in range(self.M_sim):
            beta_row = self.beta[i, :self.N_sim]
            top_idx = np.argsort(beta_row)[::-1][:10]
            top10_idx_all[i] = top_idx
            beta_top10.append(beta_row[top_idx])
            ap_idx = top_idx
            for cpu in range(self.K):
                dist_km = self.distance_matrix_front[ap_idx, cpu]
                dist_km = np.maximum(dist_km, 1e-6)
                alpha = np.where(self.link_type[ap_idx, cpu] == 0, self.alpha_los, self.alpha_nlos)
                pathloss = np.power(dist_km, -alpha)
                pl_db = 10.0 * np.log10(pathloss + 1e-12)
                pl_db = np.clip(pl_db, -110.0, -50.0)
                pl_norm = (pl_db + 110.0) / 60.0
                start = cpu * 10
                front_stats[i, start:start + 10] = pl_norm.astype(np.float32)

        self._top10_ap_idx = top10_idx_all
        beta_top10 = np.array(beta_top10, dtype=np.float32)
        #beta normalization
        beta_db = 10.0 * np.log10(beta_top10 + 1e-12)
        beta_db = np.clip(beta_db, -120.0, -80.0)
        beta_norm = (beta_db + 120.0) / 40.0
        cluster_norm = (self.last_cluster_size / 10.0).reshape(self.M_sim, 1).astype(np.float32)
        delay_norm = (self._segment_avg_delay / self.max_delay).reshape(self.M_sim, 1)
        uplink_norm = (self._segment_avg_uplink / self.max_delay).reshape(self.M_sim, 1)
        front_norm = (self._segment_avg_front / self.max_delay).reshape(self.M_sim, 1)
        offload_satisfy = np.full(
            (self.M_sim, 1), float(self._segment_offload_success_ratio), dtype=np.float32
        )
        pos_norm = (self.locations_users[:self.M_sim, :2] / 900.0).astype(np.float32)
        max_speed = 20 * self.tau_c
        speed_norm = (self.user_speed[:self.M_sim, 0] / max_speed).reshape(self.M_sim, 1).astype(np.float32)
        obs = np.concatenate(
            [
                beta_norm,
                cluster_norm,
                delay_norm,
                uplink_norm,
                front_norm,
                offload_satisfy,
                pos_norm,
                speed_norm,
                front_stats,
            ],
            axis=1,
        )
        return obs
    
    def cluster(self):
        if self.cluster_matrix is not None:
            return self.cluster_matrix
        cluster_matrix = np.zeros([self.M_sim, self.N_sim], dtype=int)
        for i in range(self.M_sim):
            sorted_idx = np.argsort(self.beta[i, :self.N_sim])[::-1]
            chosen = sorted_idx[:1]
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

        #  計算每個ap傳給幾個cpu
        L = chi.sum(axis=1).astype(np.int32)  # shape: (N_sim,)
        #建立干擾池
        I_total = np.zeros(self.K)
        for ap in range(self.N_sim):
            if L[ap] == 0:
                continue
            for cpu_j  in range(self.K):
                if chi[ap, cpu_j] != 1:
                    continue
                for cpu_k in range(self.K):
                    G_int = self.G[ap, cpu_k]
                    if self.link_type[ap, cpu_k] == 0:  
                        I_total[cpu_k] += self.p_ap * G_int * pow(self.distance_matrix_front[ap, cpu_k]  , -self.alpha_los)
                    else:
                        I_total[cpu_k] += self.p_ap * G_int * pow(self.distance_matrix_front[ap, cpu_k]  , -self.alpha_nlos)
        #計算sinr
        G_sig = (self.Gm ** 2)
        for ap in range(self.N_sim):
            for cpu in range(self.K):
                if chi[ap, cpu] == 1:
                    if self.link_type[ap, cpu] == 0:  # LOS link
                        p1 = self.p_ap * pow(self.distance_matrix_front[ap, cpu] ,
                                                                         -self.alpha_los)
                    else:
                        p1 = self.p_ap * pow(self.distance_matrix_front[ap, cpu] ,
                                                                         -self.alpha_nlos)  #改正為負號
                    SINR_front_mole = p1 * G_sig    #有用訊號功率
                    I_self = p1 * self.G[ap, cpu]
                    I = (I_total[cpu] - I_self) + self.noise_front
                    SINR_front[ap, cpu] = SINR_front_mole / I
                    front_rate[ap, cpu] = self.bandwidth_f * np.log2(1 + SINR_front[ap, cpu])

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
        #重製mobility
        if self.is_mobile:
            max_speed = 20 * self.tau_c     #根據論文改成10~20
            min_speed = 10 * self.tau_c

            # 每個 user 的當前 waypoint
            self.user_dest = self.rng.random((self.M, 2)) * 900  
            # 每個 user 的速度（整段 waypoint 期間固定）
            self.user_speed = self.rng.uniform(min_speed, max_speed, (self.M, 1))
            #重抽用戶位置
            self.locations_users = self.rng.random([self.M, 2]) * 900
        else:
            self.user_dest = None
            self.user_speed = None
        #重算距離
        diff = self.locations_users[:, np.newaxis, :] - self.locations_aps[np.newaxis, :, :]
        self.distance_matrix = np.sqrt(np.sum(diff**2, axis=2))
        d_km = self.distance_matrix / 1000.0
        # 重算 PL (Path Loss)
        far_mask  = self.distance_matrix > self.d_1
        mid_mask  = (self.distance_matrix >= self.d_0) & (self.distance_matrix <= self.d_1)
        near_mask = self.distance_matrix < self.d_0
        
        if np.any(far_mask):
            self.PL[far_mask] = -self.L - 35 * np.log10(d_km[far_mask])
        if np.any(mid_mask):
            term = (self.d_1 / 1000.0) ** 1.5 * (d_km[mid_mask] ** 2)
            self.PL[mid_mask] = -self.L - 10 * np.log10(term)
        if np.any(near_mask):
            term = (self.d_1 / 1000.0) ** 1.5 * (self.d_0 / 1000.0) ** 2
            self.PL[near_mask] = -self.L - 10 * np.log10(term)
        

        # 重抽front通道狀態
        for i in range(self.N):
            for j in range(self.K):
                self.P_los[i, j] = np.exp(-self.epsilon * self.distance_matrix_front[i, j])
                self.link_type[i, j] = self.rng.choice([0, 1], p=[self.P_los[i, j],
                                                                   1 - self.P_los[i, j]])  # 0 for LOS, 1 for NLOS
                self.G[i, j] = self.rng.choice(self.Gain, p=self.Gain_pro.ravel())

        #每episode固定shadowing
        kappa_1 = self.rng.standard_normal((1, self.N))
        kappa_2 = self.rng.standard_normal((self.M, 1))
        self.mu = np.sqrt(self.delta) * kappa_1 + np.sqrt(1 - self.delta) * kappa_2

        # 回傳observation
        self.step_num = 0 
        self._channel_ready = False
        self.theta_current = None
        self._pending_high_action = None
        self._segment_delay_sum = np.zeros((self.M_sim,), dtype=np.float32)
        self._segment_uplink_sum = np.zeros((self.M_sim,), dtype=np.float32)
        self._segment_front_sum = np.zeros((self.M_sim,), dtype=np.float32)
        self._segment_delay_count = 0
        self._segment_avg_delay = np.zeros((self.M_sim,), dtype=np.float32)
        self._segment_avg_uplink = np.zeros((self.M_sim,), dtype=np.float32)
        self._segment_avg_front = np.zeros((self.M_sim,), dtype=np.float32)
        '''
        self.Task_size = self.rng.uniform(409600, 819200, [1, self.M])  # 單位從KB改成bits，根據論文修改
        #self.Task_size = self.rng.uniform(50000, 100000, [1, self.M])
        self.Task_density = self.rng.uniform(500, 1000, [1, self.M])
        '''
        #調整task大小跟密度試試
        self.Task_size = self.rng.uniform(409600*5, 819200*5, [1, self.M])  # 單位從KB改成bits，根據論文修改
        self.Task_density = self.rng.uniform(100, 200, [1, self.M])
        
        sub_agent_obs = []
        for i in range(self.agent_num):
            raw_obs = np.array([    #obs改為一次全部正規化
            self.Task_size[0, i],
            self.Task_density[0, i],
            0,
            0,
            0,
            self.current_cluster_size[i]
            ])
            norm_obs = raw_obs / self.norm_factor #正規化
            sub_agent_obs.append(norm_obs)

        return sub_agent_obs

    def step_low(self, action):
        self.step_num += 1

        if not self._channel_ready:
            self.advance_channel()

        theta = self.theta_current

        if self._pending_high_action is not None:
            self.apply_high_action()
        elif self.cluster_matrix is None:
            self.cluster_matrix = self.cluster()

        # obtain the action
        omega_current = np.zeros([self.M_sim])
        p_current = np.zeros([self.M_sim])
        p_level = self.P_max / 4
        p_current_idx_record = np.zeros([self.M_sim], dtype=np.int32)

        for i in range(self.M_sim):

            omega_current[i], p_current_idx = self.action_mapping(action[i])
            p_current_idx_record[i] = p_current_idx
            # Ensure correct power mapping.
            if omega_current[i] == 0:
                p_current[i] = 0.0  # Ensure local action uses zero power.
            else:
                p_current[i] = (p_current_idx + 1) * p_level
        # print("Chosen CPU ID:", omega_current)
        # print("Power:", p_current)
        
        #計算速率
        uplink_rate_access = self.uplink_rate_cal(p_current, omega_current, self.cluster_matrix, theta)
        front_rate_user = self.front_rate_cal(omega_current, self.cluster_matrix)
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
                ap_idx = np.where(self.cluster_matrix[i, :] == 1)[0]
                # 算出每個 AP 的 fronthaul delay
                delays = []
                for j in ap_idx:
                    if front_rate_user[i, j] > 0:
                        delays.append(self.Task_size[0, i] / front_rate_user[i, j])
                if len(delays) > 0:
                    front_delay[i, 0] = np.max(delays)
                else:
                    front_delay[i, 0] = 0.0

        # store front delay for evaluation
        self.front_delay_last = front_delay

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
                    uplink_val[u_idx] = uplink_delay[u_idx, 0] + front_delay[u_idx, 0] #加入front delay
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
        max_delay = 1.0  # 超過 1 秒視為「同樣很爛」，避免 reward 爆
        total_delay = np.minimum(total_delay, max_delay)

        # update per-user segment averages for high-level observation
        segment_delay = total_delay[:self.M_sim, 0]
        segment_uplink = uplink_delay[:self.M_sim, 0]
        segment_front = front_delay[:self.M_sim, 0]
        self._segment_delay_sum += segment_delay.astype(np.float32)
        self._segment_uplink_sum += segment_uplink.astype(np.float32)
        self._segment_front_sum += segment_front.astype(np.float32)
        self._segment_delay_count += 1
        count = max(1, self._segment_delay_count)
        self._segment_avg_delay = self._segment_delay_sum / count
        self._segment_avg_uplink = self._segment_uplink_sum / count
        self._segment_avg_front = self._segment_front_sum / count
        offload_mask = (omega_current != 0)
        if np.any(offload_mask):
            self._segment_offload_count += int(offload_mask.sum())
            success_mask = (total_delay[:self.M_sim, 0] <= self.tau_c) & offload_mask
            self._segment_offload_success_count += int(success_mask.sum())
        if self._segment_offload_count > 0:
            self._segment_offload_success_ratio = (
                self._segment_offload_success_count / self._segment_offload_count
            )

        
        if self.step_num >= 200:    #>20改>=200
            done = [1] * self.M_sim
        else:
            done = [0] * self.M_sim

        reward = np.zeros([self.M_sim, 1])
        for i in range(self.M_sim):
            reward[i, 0] = -0.9 * total_delay[i, 0] + 0.1 * (self.tau_c - total_delay[i, 0])  #原來的reward
        
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
        '''old reward
        # High-level reward: total delay + deadline exceed penalty.
        exceed = np.maximum(total_delay - self.tau_c, 0.0)
        deadline_penalty = float(np.mean(exceed))
        self.high_reward_step = -(
            avg_total_delay_ms / 1000.0
            + deadline_penalty
        )
        '''
        # 用DSR做reward試試
        D = total_delay[:self.M_sim, 0]
        overall_DSR = float(np.mean(D <= self.tau_c))

        mean_norm = float(np.mean(D) / self.tau_c)

        p95 = float(np.percentile(D, 95))
        p95_norm = p95 / self.tau_c
        p95_norm_clip = float(np.clip(p95_norm, 0.0, 5.0))

        self.high_reward_step = (
            1.0 * overall_DSR
            - 0.30 * mean_norm
            - 0.20 * p95_norm_clip
        )
        
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
                print("cluster size", self.current_cluster_size)
            else:
                print("No Offloading Users")

        # task parameter
        '''
        #Task_size_next = self.rng.uniform(50000, 100000, [1, self.M])  # task size in bit
        Task_size_next = self.rng.uniform(409600, 819200, [1, self.M])  # 單位從KB改成bits，根據論文修改
        Task_density_next = self.rng.uniform(500, 1000, [1, self.M])  # task density cpu cycles per bit
        '''
        #調整task大小跟密度試試
        Task_size_next = self.rng.uniform(409600*5, 819200*5, [1, self.M])  # 單位從KB改成bits，根據論文修改
        Task_density_next = self.rng.uniform(100, 200, [1, self.M])

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
        self.p_idx_last = np.asarray(p_current_idx_record, dtype=np.int32)
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
            raw_obs = np.array([    #obs改為一次全部正規化
            self.Task_size[0, i],
            self.Task_density[0, i],
            self.omega_last[i],
            self.p_last[i],
            self.delay_last[i, 0],
            self.current_cluster_size[i]
            ])
            norm_obs = raw_obs / self.norm_factor # normalize
            sub_agent_obs.append(norm_obs)

            sub_agent_reward.append(reward[i])
            sub_agent_done.append(done[i])
            #sub_agent_info.append({})
            # 只在第 0 個 agent 的 info 塞統計量，其它保持空 dict
            if i == 0:
                sub_agent_info.append(metrics_info)
            else:
                sub_agent_info.append({})

        self._channel_ready = False
        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]

    def step(self, action):
        if not self._channel_ready:
            self.advance_channel()
        return self.step_low(action)


if __name__ == "__main__":
    env = MA_UCMEC_dyna_noncoop_hierarchical_peruser(render=False, seed=5)
    obs = env.reset()
    step_idx = 0
    # run a single episode
    while True:
        n_actions = env.action_space[0].n
        action_idx = np.random.randint(0, n_actions, size=env.n_agents)
        action = np.eye(n_actions, dtype=np.int32)[action_idx]
        obs, reward, done, info = env.step(action)
        step_idx += 1
        if np.all(done):
            break
