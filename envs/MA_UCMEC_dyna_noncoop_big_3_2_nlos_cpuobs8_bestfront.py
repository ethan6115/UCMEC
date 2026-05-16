import itertools

import numpy as np

from envs.MA_UCMEC_dyna_noncoop_big_3_2_nlos_cpuobs8 import (
    MA_UCMEC_dyna_noncoop_big_3_2_nlos_cpuobs8,
)


class MA_UCMEC_dyna_noncoop_big_3_2_nlos_cpuobs8_bestfront(
    MA_UCMEC_dyna_noncoop_big_3_2_nlos_cpuobs8
):
    """
    Low-level training environment with the same best-front AP selection rule
    used by eval_script.py HIGH_POLICY_MODE="best_front".

    AP selection:
      top-10 access-link candidates -> enumerate C(10, 2) pairs ->
      choose the pair maximizing max_cpu min_ap fronthaul quality.
    """

    def __init__(self, render=False, seed=None):
        super().__init__(render=render, seed=seed)
        self.candidate_n = 10
        self.k_fixed = 2
        self._ap_combos = list(itertools.combinations(range(self.candidate_n), self.k_fixed))
        self._top10_ap_idx = None

    def cluster(self):
        cluster_matrix = np.zeros([self.M_sim, self.N_sim], dtype=int)
        top10_idx_all = np.zeros((self.M_sim, self.candidate_n), dtype=np.int32)
        lo, hi = self.front_db_clip

        for i in range(self.M_sim):
            beta_row = self.beta[i, : self.N_sim]
            top_idx = np.argsort(beta_row)[::-1][: self.candidate_n]
            top10_idx_all[i] = top_idx

            best_score = -np.inf
            best_combo = self._ap_combos[0]
            for combo in self._ap_combos:
                ap_idx = top_idx[list(combo)]
                max_cpu_quality = -np.inf

                for cpu in range(self.K):
                    pl_values = []
                    for ap in ap_idx:
                        dist = max(float(self.distance_matrix_front[ap, cpu]), 1e-6)
                        alpha = self.alpha_los if self.link_type[ap, cpu] == 0 else self.alpha_nlos
                        g = max(float(self.G[ap, cpu]), 1e-12)
                        pl_values.append(g * pow(dist, -alpha))

                    min_pl_db = 10.0 * np.log10(min(pl_values) + 1e-12)
                    quality = float(np.clip((min_pl_db - lo) / (hi - lo), 0.0, 1.0))
                    if quality > max_cpu_quality:
                        max_cpu_quality = quality

                if max_cpu_quality > best_score:
                    best_score = max_cpu_quality
                    best_combo = combo

            selected_ap = top_idx[list(best_combo)]
            for ap in selected_ap:
                cluster_matrix[i, int(ap)] = 1

        self._top10_ap_idx = top10_idx_all
        return cluster_matrix
