import numpy as np
from gym import spaces

from envs.MA_UCMEC_dyna_noncoop_cluster_rand import MA_UCMEC_dyna_noncoop_cluster_rand


class MA_UCMEC_dyna_noncoop_big_3_2_nlos_cpuobs(MA_UCMEC_dyna_noncoop_cluster_rand):
    """
    Cluster-rand style low-level environment with obs aligned to
    hotspot_nlos_obs57_heurlow low-level obs:
      [task_size, task_density, omega_last, p_last, delay_last_clip,
       cluster_size_norm, cpu_front_quality_0, cpu_front_quality_1, cpu_front_quality_2]

    Notes:
    - Dynamics, channel, delay, reward, and cluster policy are inherited.
    - Only low-level observation construction is changed.
    """

    def __init__(self, render=False, seed=None):
        super().__init__(render=render, seed=seed)

        # Match heurlow low-level observation layout and normalization style.
        self.obs_dim = 9
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

        # Use the same clipping range as obs57_heurlow for CPU fronthaul quality.
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

    def _append_cluster_size_norm(self, obs_list):
        # Keep cluster-rand behavior for the cluster-size feature.
        obs_cluster_size = self.cluster_size
        if self.use_hybrid_cluster2_los:
            obs_cluster_size = self.hybrid_obs_cluster_size
        cluster_norm = np.array([obs_cluster_size / 10.0], dtype=np.float32)

        # Refresh quality from the current selected AP cluster.
        self._compute_cpu_front_quality()

        out = []
        for i, obs in enumerate(obs_list):
            obs_np = np.asarray(obs, dtype=np.float32)
            cpu_q = self.cpu_front_quality[i].astype(np.float32)
            out.append(np.concatenate([obs_np, cluster_norm, cpu_q], axis=0))
        return out
