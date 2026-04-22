import numpy as np
from gym import spaces

from envs.MA_UCMEC_dyna_coop_nlos import MA_UCMEC_dyna_coop_nlos


class MA_UCMEC_dyna_coop_nlos_cpuobs8(MA_UCMEC_dyna_coop_nlos):
    """
    Low-level environment with 8-dim observation:
      [task_size, task_density, omega_last, p_last, delay_last_clip,
       cpu_front_quality_0, cpu_front_quality_1, cpu_front_quality_2]

    Notes:
    - Inherits all dynamics/reward/channel/process from coop_nlos base env.
    - Only low-level observation construction is changed.
    """

    def __init__(self, render=False, seed=None):
        super().__init__(render=render, seed=seed)

        self.obs_dim = 8
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

        self.front_db_clip = (-112.0, -29.0)
        self.cpu_front_quality = np.zeros((self.M_sim, self.K), dtype=np.float32)

    def _compute_cpu_front_quality(self):
        """
        Per-user, per-CPU bottleneck fronthaul quality in [0, 1].
        """
        quality = np.zeros((self.M_sim, self.K), dtype=np.float32)
        if self.cluster_matrix is None:
            self.cpu_front_quality = quality
            return

        lo, hi = self.front_db_clip
        span = max(hi - lo, 1e-9)

        for i in range(self.M_sim):
            ap_idx = np.where(self.cluster_matrix[i, :] == 1)[0]
            if ap_idx.size == 0:
                continue

            for cpu in range(self.K):
                pl_values = []
                for ap in ap_idx:
                    dist = max(float(self.distance_matrix_front[ap, cpu]), 1e-6)
                    alpha = self.alpha_los if self.link_type[ap, cpu] == 0 else self.alpha_nlos
                    g = max(float(self.G[ap, cpu]), 1e-12)
                    pl_values.append(g * pow(dist, -alpha))

                min_pl_db = 10.0 * np.log10(min(pl_values) + 1e-12)
                quality[i, cpu] = float(np.clip((min_pl_db - lo) / span, 0.0, 1.0))

        self.cpu_front_quality = quality

    def _append_cpu_front_quality(self, obs_list):
        self._compute_cpu_front_quality()
        out = []
        for i, obs in enumerate(obs_list):
            obs_np = np.asarray(obs, dtype=np.float32)
            cpu_q = self.cpu_front_quality[i].astype(np.float32)
            out.append(np.concatenate([obs_np, cpu_q], axis=0))
        return out

    def reset(self):
        obs = super().reset()
        return self._append_cpu_front_quality(obs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        obs = self._append_cpu_front_quality(obs)
        return [obs, reward, done, info]
