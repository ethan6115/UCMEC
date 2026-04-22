import numpy as np
from gym import spaces

from envs.MA_UCMEC_dyna_noncoop_hierarchical_peruser_hotspot_nlos_obs57 import (
    MA_UCMEC_dyna_noncoop_hierarchical_peruser,
)


class MA_UCMEC_dyna_noncoop_hierarchical_peruser_fixedpower(
    MA_UCMEC_dyna_noncoop_hierarchical_peruser
):
    """
    Hierarchical env with fixed low-level transmit power.

    Low-level action space is reduced to 4 one-hot actions:
      0: local
      1: offload to cpu1
      2: offload to cpu2
      3: offload to cpu3

    Offloading power index is fixed:
      - mode='min' -> p_idx=1
      - mode='max' -> p_idx=3
    """

    def __init__(self, render=False, seed=None, fixed_power_mode="max"):
        super().__init__(render=render, seed=seed)

        if fixed_power_mode not in ("min", "max"):
            raise ValueError("fixed_power_mode must be 'min' or 'max'")

        # User said mask_local=True is not needed for this ablation.
        self.mask_local = False

        self.fixed_power_mode = fixed_power_mode
        self.fixed_power_idx = 1 if fixed_power_mode == "min" else 3

        # local / cpu1 / cpu2 / cpu3
        self.action_dim = 4
        self.action_space = spaces.Tuple(
            tuple([spaces.Discrete(self.action_dim)] * self.n_agents)
        )

    def action_mapping(self, action_agent):
        act_idx = int(np.argmax(action_agent))

        if act_idx == 0:
            # local
            return 0, 0
        if act_idx == 1:
            return 1, self.fixed_power_idx
        if act_idx == 2:
            return 2, self.fixed_power_idx
        if act_idx == 3:
            return 3, self.fixed_power_idx

        # fallback for malformed one-hot action
        return 0, 0


class MA_UCMEC_dyna_noncoop_hierarchical_peruser_fixedpower_min(
    MA_UCMEC_dyna_noncoop_hierarchical_peruser_fixedpower
):
    def __init__(self, render=False, seed=None):
        super().__init__(render=render, seed=seed, fixed_power_mode="min")


class MA_UCMEC_dyna_noncoop_hierarchical_peruser_fixedpower_max(
    MA_UCMEC_dyna_noncoop_hierarchical_peruser_fixedpower
):
    def __init__(self, render=False, seed=None):
        super().__init__(render=render, seed=seed, fixed_power_mode="max")
