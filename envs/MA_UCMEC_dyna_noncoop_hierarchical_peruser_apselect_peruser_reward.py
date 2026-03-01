import numpy as np

from envs.MA_UCMEC_dyna_noncoop_hierarchical_peruser_apselect import (
    MA_UCMEC_dyna_noncoop_hierarchical_peruser,
)


class MA_UCMEC_dyna_noncoop_hierarchical_peruser_peruser_reward(
    MA_UCMEC_dyna_noncoop_hierarchical_peruser
):
    """Per-user interval reward variant for high-level per-user credit."""

    def compute_interval_reward(self):
        if len(self._interval_delays) == 0:
            return np.zeros((self.M_sim,), dtype=np.float32)

        delay_flat = np.array(self._interval_delays, dtype=np.float32)
        if delay_flat.size % self.M_sim != 0:
            # Fallback to legacy scalar when interval buffer is malformed.
            D_interval = delay_flat
            overall_DSR = float(np.mean(D_interval <= self.tau_c))
            p95_norm_clip = float(
                np.clip(np.percentile(D_interval, 95) / self.tau_c, 0.0, 5.0)
            )
            reward_scalar = 1.0 * overall_DSR - 0.05 * p95_norm_clip
            self._interval_delays = []
            return np.full((self.M_sim,), reward_scalar, dtype=np.float32)

        D_steps = delay_flat.reshape(-1, self.M_sim)  # [interval_steps, M_sim]
        dsr_user = np.mean(D_steps <= self.tau_c, axis=0).astype(np.float32)
        p95_user = np.percentile(D_steps, 95, axis=0).astype(np.float32)
        p95_norm_clip_user = np.clip(p95_user / self.tau_c, 0.0, 5.0)

        offload_ratio_user = self._segment_offload_count / max(1, self._segment_delay_count)
        offload_eff_user = self._segment_offload_success_ratio * offload_ratio_user

        reward_user = (
            1.0 * dsr_user
            + 0.5 * offload_eff_user
            - 0.05 * p95_norm_clip_user
        ).astype(np.float32)

        self._interval_delays = []
        return reward_user
