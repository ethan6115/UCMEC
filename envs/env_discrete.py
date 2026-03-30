"""
# @Time    : 2021/7/2 5:22 下午
# @Author  : hezhiqiang
# @Email   : tinyzqh@163.com
# @File    : env_discrete.py
"""

import gym
from gym import spaces
import numpy as np
from envs.env_core import EnvCore
from envs.MA_UCMEC_stat_coop import MA_UCMEC_stat_coop
from envs.MA_UCMEC_stat_noncoop import MA_UCMEC_stat_noncoop
from envs.MA_CBO_stat_coop import MA_CBO_stat_coop
from envs.MA_CBO_stat_noncoop import MA_CBO_stat_noncoop
from envs.MA_MPO_stat_coop import MA_MPO_stat_coop
from envs.MA_MPO_stat_noncoop import MA_MPO_stat_noncoop

from envs.MA_UCMEC_dyna_coop import MA_UCMEC_dyna_coop
from envs.MA_UCMEC_dyna_noncoop import MA_UCMEC_dyna_noncoop
from envs.MA_UCMEC_dyna_noncoop_cluster_rand import MA_UCMEC_dyna_noncoop_cluster_rand
from envs.MA_UCMEC_dyna_noncoop_origin import MA_UCMEC_dyna_noncoop_origin
from envs.MA_CBO_dyna_coop import MA_CBO_dyna_coop
from envs.MA_CBO_dyna_noncoop import MA_CBO_dyna_noncoop
from envs.MA_MPO_dyna_coop import MA_MPO_dyna_coop
from envs.MA_MPO_dyna_noncoop import MA_MPO_dyna_noncoop
#切換hierarchical版本
from envs.MA_UCMEC_dyna_noncoop_hierarchical_alluser_front_small_clusterobs import MA_UCMEC_dyna_noncoop_hierarchical_alluser as MA_UCMEC_dyna_noncoop_hierarchical
from envs.MA_UCMEC_dyna_noncoop_hierarchical_peruser_hotspot_nlos_obs57_heurlow import (
    MA_UCMEC_dyna_noncoop_hierarchical_peruser as MA_UCMEC_dyna_noncoop_hierarchical_peruser,
)
from envs.MA_UCMEC_dyna_noncoop_hierarchical_peruser_apselect_peruser_reward import (
    MA_UCMEC_dyna_noncoop_hierarchical_peruser_peruser_reward as MA_UCMEC_dyna_noncoop_hierarchical_peruser_peruser_reward,
)
#from envs.MA_UCMEC_dyna_noncoop_hierarchical_alluser import MA_UCMEC_dyna_noncoop_hierarchical_alluser as MA_UCMEC_dyna_noncoop_hierarchical
#from envs.MA_UCMEC_dyna_noncoop_hierarchical_peruser import MA_UCMEC_dyna_noncoop_hierarchical_peruser as MA_UCMEC_dyna_noncoop_hierarchical



class DiscreteActionEnv(object):
    """
    对于离散动作环境的封装
    Wrapper for discrete action environment.
    """

    def __init__(self, all_args=None):
        # Decide env here so action/obs semantics stay inside the environment.
        use_hierarchical = getattr(all_args, "use_hierarchical", False) if all_args is not None else False
        use_high_peruser = getattr(all_args, "use_high_peruser", False) if all_args is not None else False
        use_high_peruser_credit = getattr(all_args, "use_high_peruser_credit", False) if all_args is not None else False
        use_low_cluster_randomization = getattr(all_args, "use_low_cluster_randomization", False) if all_args is not None else False
        seed = getattr(all_args, "seed", None) if all_args is not None else None
        if use_hierarchical:
            if use_high_peruser:
                # Hotspot per-user env already exposes interval reward in per-user
                # form, so keep a single env path regardless of credit setting.
                self.env = MA_UCMEC_dyna_noncoop_hierarchical_peruser(seed=seed)
            else:
                self.env = MA_UCMEC_dyna_noncoop_hierarchical(seed=seed)
        else:
            if use_low_cluster_randomization:
                self.env = MA_UCMEC_dyna_noncoop_cluster_rand(seed=seed)
            else:
                self.env = MA_UCMEC_dyna_noncoop(seed=seed)
        self.num_agent = self.env.agent_num
        self.signal_obs_dim = self.env.obs_dim
        self.signal_action_dim = self.env.action_dim
        # Reason: keep single-layer envs working by defaulting to None when fields are absent.
        self.high_action_space = getattr(self.env, "high_action_space", None)
        self.high_observation_space = getattr(self.env, "high_observation_space", None)

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False

        self.movable = True

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        share_obs_dim = 0
        total_action_space = []
        for agent_idx in range(self.num_agent):
            # physical action space
            u_action_space = spaces.Discrete(self.signal_action_dim)  # 5个离散的动作

            # if self.movable:
            total_action_space.append(u_action_space)

            # total action space
            # if len(total_action_space) > 1:
            #     # all action spaces are discrete, so simplify to MultiDiscrete action space
            #     if all(
            #         [
            #             isinstance(act_space, spaces.Discrete)
            #             for act_space in total_action_space
            #         ]
            #     ):
            #         act_space = MultiDiscrete(
            #             [[0, act_space.n - 1] for act_space in total_action_space]
            #         )
            #     else:
            #         act_space = spaces.Tuple(total_action_space)
            # self.action_space.append(act_space)
            # else:
            self.action_space.append(total_action_space[agent_idx])

            # observation space
            share_obs_dim += self.signal_obs_dim
            self.observation_space.append(
                spaces.Box(
                    low=-np.inf,
                    high=+np.inf,
                    shape=(self.signal_obs_dim,),
                    dtype=np.float32,
                )
            )  # [-inf,inf]

        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)
            for _ in range(self.num_agent)
        ]

    def step(self, actions):
        """
        输入actions维度假设：
        # actions shape = (5, 2, 5)
        # 5个线程的环境，里面有2个智能体，每个智能体的动作是一个one_hot的5维编码
        Input actions dimension assumption:
        # actions shape = (5, 2, 5)
        # 5 threads of the environment, with 2 intelligent agents inside, and each intelligent agent's action is a 5-dimensional one_hot encoding
        """

        results = self.env.step(actions)
        obs, rews, dones, infos = results
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        obs = self.env.reset()
        return np.stack(obs)

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass

    def seed(self, seed):
        return self.env.seed(seed)
    
    def get_global_obs(self):
        return self.env.get_global_obs()

    def set_high_action(self, action_id):
        return self.env.set_high_action(action_id)


class MultiDiscrete:
    """
    - The multi-discrete action space consists of a series of discrete action spaces with different parameters
    - It can be adapted to both a Discrete action space or a continuous (Box) action space
    - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
    - It is parametrized by passing an array of arrays containing [min, max] for each discrete action space
       where the discrete action space can take any integers from `min` to `max` (both inclusive)
    Note: A value of 0 always need to represent the NOOP action.
    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:
        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
    - Can be initialized as
        MultiDiscrete([ [0,4], [0,1], [0,1] ])
    """

    def __init__(self, array_of_param_array):
        super().__init__()
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]
        self.n = np.sum(self.high) + 2

    def sample(self):
        """Returns a array with one sample from each discrete action space"""
        # For each row: round(random .* (max - min) + min, 0)
        random_array = np.random.rand(self.num_discrete_space)
        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.0), random_array) + self.low)]

    def contains(self, x):
        return (
            len(x) == self.num_discrete_space
            and (np.array(x) >= self.low).all()
            and (np.array(x) <= self.high).all()
        )

    @property
    def shape(self):
        return self.num_discrete_space

    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)

    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)


if __name__ == "__main__":
    DiscreteActionEnv().step(actions=None)
