import numpy as np
from gym import spaces

from envs.MA_UCMEC_dyna_noncoop_hierarchical_peruser_hotspot_nlos_obs57 import (
    MA_UCMEC_dyna_noncoop_hierarchical_peruser,
)


class MA_UCMEC_dyna_noncoop_hierarchical_peruser_flat(
    MA_UCMEC_dyna_noncoop_hierarchical_peruser
):
    """
    Flat joint-policy environment for a single-policy PPO baseline.

    The policy emits one MultiDiscrete action per user:
      [AP-pair index in C(top-10, 2), low action index].

    AP-pair actions are applied only at H-slot boundaries, while low-level
    offloading / CPU / power actions are applied every slot. The environment
    reuses the hierarchical env's candidate construction, AP-pair mapping,
    channel dynamics, delay calculation, and per-slot reward.
    """

    def __init__(self, render=False, seed=None, hierarchical_interval=10):
        super().__init__(render=render, seed=seed)

        self.hierarchical_interval = int(hierarchical_interval)
        self.low_obs_dim = int(self.obs_dim)
        self.low_action_dim = int(self.action_dim)
        self.flat_phase_dim = 2
        self.flat_obs_dim = self.low_obs_dim + self.high_obs_dim + self.flat_phase_dim

        action_spaces = []
        for _ in range(self.n_agents):
            action_space = spaces.MultiDiscrete([self.high_action_dim, self.low_action_dim])
            action_space.joint_logprob_sum = True
            action_spaces.append(action_space)
        self.action_space = spaces.Tuple(tuple(action_spaces))
        self.obs_dim = self.flat_obs_dim
        self.obs_low = np.full(self.flat_obs_dim, -np.inf, dtype=np.float32)
        self.obs_high = np.full(self.flat_obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Tuple(
            tuple(
                [
                    spaces.Box(
                        low=self.obs_low,
                        high=self.obs_high,
                        shape=(self.flat_obs_dim,),
                        dtype=np.float32,
                    )
                ]
                * self.n_agents
            )
        )

    def _is_high_step(self):
        return self.step_num == 0 or self.step_num % self.hierarchical_interval == 0

    def _phase_features(self):
        high_flag = 1.0 if self._is_high_step() else 0.0
        phase = (self.step_num % self.hierarchical_interval) / float(self.hierarchical_interval)
        return np.array([high_flag, phase], dtype=np.float32)

    def _build_flat_obs(self, low_obs):
        high_obs = self.get_global_obs().astype(np.float32)
        phase = self._phase_features()
        flat_obs = []
        for i in range(self.M_sim):
            low_i = np.asarray(low_obs[i], dtype=np.float32)
            flat_obs.append(np.concatenate([low_i, high_obs[i], phase], axis=0))
        return flat_obs

    def _split_joint_action(self, action):
        action = np.asarray(action)

        if action.ndim == 3 and action.shape[-1] == self.high_action_dim + self.low_action_dim:
            ap_action = np.argmax(action[:, :, : self.high_action_dim], axis=-1)
            low_action = np.argmax(action[:, :, self.high_action_dim :], axis=-1)
            action = np.stack([ap_action, low_action], axis=-1)

        if action.ndim == 2 and action.shape[1] == self.high_action_dim + self.low_action_dim:
            ap_action = np.argmax(action[:, : self.high_action_dim], axis=-1)
            low_action = np.argmax(action[:, self.high_action_dim :], axis=-1)
        elif action.ndim == 2 and action.shape[1] == 2:
            ap_action = action[:, 0]
            low_action = action[:, 1]
        elif action.ndim == 3 and action.shape[0] == 1 and action.shape[2] == 2:
            ap_action = action[0, :, 0]
            low_action = action[0, :, 1]
        else:
            raise ValueError(
                "Flat joint action must be shaped as [M, 2] indices or "
                "[M, high_action_dim + low_action_dim] one-hot vectors."
            )

        ap_action = np.asarray(ap_action, dtype=np.int32).reshape(-1)[: self.M_sim]
        low_action = np.asarray(low_action, dtype=np.int32).reshape(-1)[: self.M_sim]
        ap_action = np.clip(ap_action, 0, self.high_action_dim - 1)
        low_action = np.clip(low_action, 0, self.low_action_dim - 1)
        return ap_action, low_action

    def reset(self):
        low_obs = super().reset()
        return self._build_flat_obs(low_obs)

    def step(self, action):
        ap_action, low_action = self._split_joint_action(action)

        if not self._channel_ready:
            self.advance_channel()

        if self._is_high_step():
            self.apply_high_action(ap_action)
        elif self.cluster_matrix is None:
            self.apply_high_action(np.zeros((self.M_sim,), dtype=np.int32))

        low_onehot = np.eye(self.low_action_dim, dtype=np.float32)[low_action]
        low_obs, reward, done, info = self.step_low(low_onehot)
        self.high_reward_step = 0.0
        return [self._build_flat_obs(low_obs), reward, done, info]
