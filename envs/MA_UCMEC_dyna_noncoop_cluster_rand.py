import numpy as np
from gym import spaces

from envs.MA_UCMEC_dyna_noncoop import MA_UCMEC_dyna_noncoop


class MA_UCMEC_dyna_noncoop_cluster_rand(MA_UCMEC_dyna_noncoop):
    """
    Low-level training environment variant:
    - Keep base non-coop dynamics.
    - Sample one episode-level cluster size k in [1, 10].
    - Append cluster_size_norm (= k/10) to low-level obs.
    """

    def __init__(self, render=False, seed=None):
        super().__init__(render=render, seed=seed)
        self.cluster_k_min = 1
        self.cluster_k_max = 6

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
        cluster_norm = np.array([self.cluster_size / 10.0], dtype=np.float32)
        out = []
        for obs in obs_list:
            obs_np = np.asarray(obs, dtype=np.float32)
            out.append(np.concatenate([obs_np, cluster_norm], axis=0))
        return out

    def reset(self):
        # Sample one cluster size per episode.
        self.cluster_size = int(self.rng.integers(self.cluster_k_min, self.cluster_k_max + 1))
        obs = super().reset()
        return self._append_cluster_size_norm(obs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        obs = self._append_cluster_size_norm(obs)
        return [obs, reward, done, info]
