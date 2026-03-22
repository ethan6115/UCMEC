import numpy as np
from gym import spaces

from envs.MA_UCMEC_dyna_noncoop_big_3_2 import MA_UCMEC_dyna_noncoop


class MA_UCMEC_dyna_noncoop_cluster_rand(MA_UCMEC_dyna_noncoop):
    """
    Low-level training environment variant:
    - Keep base non-coop dynamics.
    - Sample one episode-level cluster size k in [1, 10].
    - Append cluster_size_norm (= k/10) to low-level obs.
    - Optional hybrid mode: fixed k=2 with AP selection:
      (1) best beta overall, (2) best beta among APs that are LOS to all CPUs.
    """

    def __init__(self, render=False, seed=None):
        super().__init__(render=render, seed=seed)
        self.cluster_k_min = 1
        self.cluster_k_max = 10
        self.use_hybrid_cluster2_los = False
        self.hybrid_fixed_cluster_size = 2
        self.hybrid_obs_cluster_size = 1

        # Base env builds 5-dim low-level obs. This variant appends cluster_size_norm.
        self.obs_dim = 6
        self.obs_low = np.zeros(self.obs_dim, dtype=np.float32)
        self.obs_high = np.ones(self.obs_dim, dtype=np.float32)
        self.observation_space = spaces.Tuple(
            tuple(
                [
                    spaces.Box(
                        low=self.obs_low,
                        high=self.obs_high,
                        shape=(self.obs_dim,),
                        dtype=np.float32,
                    )
                ]
                * self.n_agents
            )
        )

    def _append_cluster_size_norm(self, obs_list):
        obs_cluster_size = self.cluster_size
        if self.use_hybrid_cluster2_los:
            obs_cluster_size = self.hybrid_obs_cluster_size
        cluster_norm = np.array([obs_cluster_size / 10.0], dtype=np.float32)
        out = []
        for obs in obs_list:
            obs_np = np.asarray(obs, dtype=np.float32)
            out.append(np.concatenate([obs_np, cluster_norm], axis=0))
        return out

    def _cluster_hybrid_fixed2_los(self):
        cluster_matrix = np.zeros([self.M_sim, self.N_sim], dtype=int)
        los_all_cpu_mask = np.all(self.link_type[: self.N_sim, : self.K] == 0, axis=1)

        for i in range(self.M_sim):
            beta_i = self.beta[i, : self.N_sim]

            best_overall = int(np.argmax(beta_i))
            selected = [best_overall]

            los_candidates = np.where(los_all_cpu_mask)[0]
            los_candidates = los_candidates[los_candidates != best_overall]
            if los_candidates.size > 0:
                los_best = int(los_candidates[np.argmax(beta_i[los_candidates])])
                selected.append(los_best)
            else:
                all_sorted = np.argsort(beta_i)[::-1]
                fallback = next((int(idx) for idx in all_sorted if int(idx) != best_overall), best_overall)
                selected.append(fallback)

            for ap_idx in selected:
                cluster_matrix[i, ap_idx] = 1

        return cluster_matrix

    def cluster(self):
        if self.use_hybrid_cluster2_los:
            return self._cluster_hybrid_fixed2_los()
        return super().cluster()

    def reset(self):
        # Sample one cluster size per episode.
        if self.use_hybrid_cluster2_los:
            self.cluster_size = int(self.hybrid_fixed_cluster_size)
        else:
            self.cluster_size = int(self.rng.integers(self.cluster_k_min, self.cluster_k_max + 1))
        obs = super().reset()
        return self._append_cluster_size_norm(obs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        obs = self._append_cluster_size_norm(obs)
        return [obs, reward, done, info]
