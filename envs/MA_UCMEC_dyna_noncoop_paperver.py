import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
# from stable_baselines3.common.env_checker import check_env
import cvxpy as cp


class MA_UCMEC_dyna_noncoop(object):
    def __init__(self, render: bool = False):

        # Initialization
        self.is_mobile = True
        gym.logger.set_level(40)
        np.random.seed(3)
        self.M = 50  # number of users
        self.N = 200  # number of APs
        self.varsig = 16  # number of antennas of each AP
        self.K = 3  # number of CPUs
        self.P_max = 0.1  # maximum transmit power of user / pilot power
        self.M_sim = 10  # number of users for simulation
        self.N_sim = 50  # number of APs for simulation
        self.Task_size = np.zeros([1, self.M])
        self.Task_density = np.zeros([1, self.M])
        self.cluster_matrix = None

        # locations of users and APs
        self.locations_users = np.random.random_sample([self.M, 2]) * 900  # 2-D location of users
        self.locations_aps = np.random.random_sample([self.N, 2]) * 900  # 2-D location of APs

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
        #self.C_user = np.random.uniform(2e8, 5e8, [1, self.M])  # 根據論文修改為2e9, 5e9
        self.C_user = np.random.uniform(2e9, 5e9, [1, self.M])  # computing resource of users  in Hz
        self.cluster_size = 5  # AP cluster size

        # edge server parameter
        self.C_edge = np.random.uniform(10e9, 20e9, [self.K, 1])  # computing resource of edge server in CPU

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
                self.link_type[i, j] = np.random.choice([0, 1], p=[self.P_los[i, j],
                                                                   1 - self.P_los[i, j]])  # 0 for LOS, 1 for NLOS
                # if link_type[i, j] == 0:  # LOS link
                #     front_chan[i, j] = np.random.gamma(2, 1 / psi_los)  # Nakagami channel gain
                # else:  # NLOS link
                #     front_chan[i, j] = np.random.gamma(2, 1 / psi_nlos)  # Nakagami channel gain
                self.G[i, j] = np.random.choice(self.Gain, p=self.Gain_pro.ravel())

        # pilot assignment
        self.tau_p = self.M  # length of pilot symbol
        self.pilot_matrix = np.zeros([self.M, self.tau_p])
        for i in range(self.M):
            self.pilot_index = i
            self.pilot_matrix[i, self.pilot_index] = 1

        # parameter init
        self.n_agents = self.M_sim
        self.agent_num = self.n_agents
        self.obs_dim = 5  # set the observation dimension of agents
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
        self.norm_factor = np.array([819200.0, 1000.0, 3.0, self.P_max, 2.0])   #對obs做正規化用的，根據論文修改100000改為819200
        self.obs_low = np.zeros(self.obs_dim)  # [0, 0, 0, 0, 0]
        self.obs_high = np.ones(self.obs_dim)  # [1, 1, 1, 1, 1]
        # obs = {task data size, task computing density, action index, total delay of last time slot}
        self.observation_space = spaces.Tuple(tuple(
            [spaces.Box(low=self.obs_low, high=self.obs_high, shape=(self.obs_dim,),
                        dtype=np.float32)] * self.n_agents))
        # self.np_random = None
        self.uplink_rate_access_b = np.zeros([self.M_sim, 1])
        self.step_num = 0

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
        return [seed]

    def render(self, mode='human'):
        pass

    def reset(self):
        '''
        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_obs = np.random.uniform(low=self.obs_low, high=self.obs_high, size=(self.obs_dim,))
            sub_agent_obs.append(sub_obs)
        '''
        self.step_num = 0 
        self.Task_size = np.random.uniform(409600, 819200, [1, self.M])  # 單位從KB改成bits，根據論文修改
        #self.Task_size = np.random.uniform(50000, 100000, [1, self.M])
        self.Task_density = np.random.uniform(500, 1000, [1, self.M])
        
        sub_agent_obs = []
        for i in range(self.agent_num):
            raw_obs = np.array([    #obs改為一次全部正規化
            self.Task_size[0, i],
            self.Task_density[0, i],
            0,
            0,
            0])
            #norm_obs = raw_obs / self.norm_factor  #論文原本code沒做正規化
            sub_agent_obs.append(raw_obs)
            
        return sub_agent_obs

    def step(self, action):
        self.step_num += 1

        if self.is_mobile:
            #根據論文修改速度5-15改成10-20
            max_speed = 20 * self.tau_c  # maximum speed of users (m / tau_c second)
            min_speed = 10 * self.tau_c  # maximum speed of users (m / tau_c second)
            destination_users = np.random.random_sample([self.M, 2]) * 900
            user_speed = np.random.uniform(min_speed, max_speed, [self.M, 1])
            for i in range(self.M):
                dx = destination_users[i, 0] - self.locations_users[i, 0]
                dy = destination_users[i, 1] - self.locations_users[i, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist == 0:
                    continue
                # 正確方向：用符號，不要 abs
                step = user_speed[i, 0] / dist  
                self.locations_users[i, 0] += dx * step
                self.locations_users[i, 1] += dy * step
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
        kappa_1 = np.random.rand(1, self.N)  # parameter in Eq. (5)
        kappa_2 = np.random.rand(1, self.M)  # parameter in Eq. (5)
        
        for i in range(self.M):
            for j in range(self.N):
                # Eq. (5) shadow fading computation
                self.mu[i, j] = math.sqrt(self.delta) * kappa_1[0, j] + math.sqrt(1 - self.delta) * kappa_2[
                    0, i]  # MxN matrix as Eq. (5)

                # Eq. (2) channel computation
                self.beta[i, j] = pow(10, self.PL[i, j] / 10) * pow(10, (self.sigma_s * self.mu[i, j]) / 10)
                for k in range(self.varsig):
                    self.h[i, j, k] = np.random.normal(loc=0, scale=0.5) + 1j * np.random.normal(loc=0, scale=0.5)
                    self.access_chan[i, j, k] = np.sqrt(self.beta[i, j]) * self.h[i, j, k]
        '''
        # 1. 生成隨機參數 (一次生成整個矩陣，取代迴圈內生成)
        kappa_1 = np.random.rand(1, self.N)  # 形狀: (1, N)
        kappa_2 = np.random.rand(self.M, 1)  # 形狀: (M, 1)，轉置以便廣播

        # 2. 計算 Shadow Fading (mu) - 利用 Broadcasting
        # (1, N) 與 (M, 1) 運算會自動廣播成 (M, N) 矩陣
        self.mu = np.sqrt(self.delta) * kappa_1 + np.sqrt(1 - self.delta) * kappa_2

        # 3. 計算 Large Scale Fading (beta) - 矩陣直接運算
        # self.PL 和 self.mu 都是 (M, N) 矩陣，直接進行元素級運算
        self.beta = np.power(10, self.PL / 10.0) * np.power(10, (self.sigma_s * self.mu) / 10.0)

        # 4. 計算 Small Scale Fading (h) - 一次生成所有亂數
        # 形狀: (M, N, varsig)
        h_real = np.random.normal(loc=0, scale=0.5, size=(self.M, self.N, self.varsig))
        h_imag = np.random.normal(loc=0, scale=0.5, size=(self.M, self.N, self.varsig))
        self.h = h_real + 1j * h_imag

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
        
        
        # obtain the action
        omega_current = np.zeros([self.M_sim])
        p_current = np.zeros([self.M_sim])
        p_level = self.P_max / 4
        

        for i in range(self.M_sim):
            omega_current[i], p_current[i] = self.action_mapping(action[i])
            p_current[i] = (p_current[i] + 1) * p_level
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
        for i in range(self.K):
            serve_user_id = []
            serve_user_task = []
            _local_delay = []
            _front_delay = []
            _uplink_delay = []

            for j in range(self.M_sim):
                if task_mat[j, i] != 0:
                    serve_user_id.append(j)
                    serve_user_task.append(task_mat[j, i])
                    _local_delay.append(local_delay[j, 0])
                    _uplink_delay.append(uplink_delay[j, 0])
            if len(serve_user_id) == 0:
                continue
            
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
                if _C is not None: 
                    actual_C[serve_user_id[k], i] = _C[k]
                else:
                    # 處理失敗的邏輯，避免崩潰。使用平均分配：
                    actual_C[serve_user_id[k], i] = self.C_edge[i, 0] / len(serve_user_id)

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
        reward = np.zeros([self.M_sim, 1])
        for i in range(self.M_sim):
            reward[i, 0] = -0.9 * total_delay[i, 0] + 0.1 * (self.tau_c - total_delay[i, 0])  #原來的reward
            
        if self.step_num % 200 == 0:
            print("Step Index:", self.step_num)
            print("Average Total Delay (ms):", np.sum(total_delay) * 1000 / self.M_sim)
            
            # --- 平均 local delay（只算沒 offload 的 user）---
            local_mask = (omega_current == 0)
            if np.any(local_mask):
                avg_local_delay_ms = np.mean(local_delay[local_mask, 0]) * 1000
                print("Average Local Delay (ms):", avg_local_delay_ms)
            else:
                print("Average Local Delay (ms): 0 (no local users)")

            # --- 平均 offloading delay, uplink rate（uplink + fronthaul + processing）---
            offload_mask = (omega_current != 0)
            active = np.count_nonzero(offload_mask)
            if active > 0:
                offload_delay = uplink_delay + front_delay + actual_process_delay  # 單位：秒
                avg_offload_delay_ms = np.mean(offload_delay[offload_mask, 0]) * 1000
                avg_uplink_rate_Mbps = np.mean(uplink_rate_access[offload_mask, 0]) / 1e6
                print("Average Offloading Delay (ms):", avg_offload_delay_ms)
                print("Average Uplink Rate (Mbps):", avg_uplink_rate_Mbps)
                print("Offloading user", active)
            else:
                print("Average Offloading Delay (ms): 0 (no offloading users)")
                print("Average Uplink Rate (Mbps): 0 (no offloading users)")


        # task parameter
        #Task_size_next = np.random.uniform(50000, 100000, [1, self.M])  # task size in bit
        Task_size_next = np.random.uniform(409600, 819200, [1, self.M])  # 單位從KB改成bits，根據論文修改
        Task_density_next = np.random.uniform(500, 1000, [1, self.M])  # task density cpu cycles per bit
        # Task_max_delay = np.random.uniform(2, 5, [1, M])  # task max delay in second
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
        for i in range(self.agent_num):
            raw_obs = np.array([    #obs改為一次全部正規化
            self.Task_size[0, i],
            self.Task_density[0, i],
            self.omega_last[i],
            self.p_last[i],
            self.delay_last[i, 0]
            ])
            # norm_obs = raw_obs / self.norm_factor #論文原本code沒做正規化
            sub_agent_obs.append(raw_obs)

            sub_agent_reward.append(reward[i])
            sub_agent_done.append(done[i])
            sub_agent_info.append({})
            
        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]


if __name__ == "__main__":
    env = MA_UCMEC_dyna_noncoop(render=False)
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
