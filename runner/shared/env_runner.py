"""
# @Time    : 2021/7/1 7:15 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : env_runner.py
"""

import time
import numpy as np
import torch
from runner.shared.base_runner import Runner
from scipy.io import savemat


# import imageio


def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""

    def __init__(self, config):
        super(EnvRunner, self).__init__(config)
        # Reason: accumulate high-level rewards across intervals before inserting.
        self._high_reward_acc = None
        self._high_reward_count = 0
        self._high_pending = False
        self._high_transition = {}
        

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        reward_list = np.zeros([episodes, 1])
        high_reward_list = np.zeros([episodes, 1]) if self.use_hierarchical else None
        for episode in range(episodes):

            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            if self.use_hierarchical:
                high_episode_reward_sum = np.zeros((self.n_rollout_threads, 1), dtype=np.float32)
                high_episode_reward_count = 0

            for step in range(self.episode_length):

                # Update channel state before any decision in this slot.
                if self.use_hierarchical:
                    if hasattr(self.envs, "envs"):
                        for env in self.envs.envs:
                            src = env.env if hasattr(env, "env") else env
                            if hasattr(src, "advance_channel"):
                                src.advance_channel()
                    else:
                        if hasattr(self.envs, "advance_channel"):
                            self.envs.advance_channel()


                #每 10 步（或你設定的 interval）取得高層動作，直接餵給環境
                if self.use_hierarchical and (step % self.hierarchical_interval == 0):
                    self.high_trainer.prep_rollout()
                    global_obs = self._get_global_obs_batch()
                    value_h, action_h, logp_h, rnn_h, rnn_hc = self.high_trainer.policy.get_actions(
                        global_obs, global_obs,
                        # Reason: use high_buffer's own step counter to avoid index drift.
                        self.high_buffer.rnn_states[self.high_buffer.step].squeeze(1),
                        self.high_buffer.rnn_states_critic[self.high_buffer.step].squeeze(1),
                        self.high_buffer.masks[self.high_buffer.step].squeeze(1),
                    )
                    action_h = _t2n(action_h)
                    if self.use_high_peruser:
                        space_name = getattr(self.high_action_space, "__class__", None).__name__
                        if space_name == "MultiDiscrete":
                            action_id = action_h.astype(int)
                        elif space_name == "MultiBinary":
                            action_id = action_h.astype(int)
                        else:
                            action_id = action_h.squeeze(-1).astype(int)
                    elif action_h.ndim == 2 and action_h.shape[1] == 1:
                        # Discrete high-level: shape [n_threads, 1]
                        action_id = action_h.squeeze(-1).astype(int)
                    else:
                        # MultiDiscrete: shape [n_threads, M_sim]
                        action_id = action_h.astype(int)
                    # Reason: runner does not define semantics; pass raw indices to env.
                    if hasattr(self.envs, "envs"):
                        for env, a in zip(self.envs.envs, action_id):
                            src = env.env if hasattr(env, "env") else env
                            if hasattr(src, "apply_high_action"):
                                src.apply_high_action(a)
                            else:
                                src.set_high_action(a)
                    else:
                        if hasattr(self.envs, "apply_high_action"):
                            self.envs.apply_high_action(action_id[0])
                        else:
                            self.envs.set_high_action(action_id[0])
                    # Reason: store decision-time transition for interval-avg reward.
                    self._high_transition = {
                        "global_obs": global_obs,
                        "rnn_h": _t2n(rnn_h),
                        "rnn_hc": _t2n(rnn_hc),
                        "action_h": action_h,
                        "logp_h": _t2n(logp_h),
                        "value_h": _t2n(value_h),
                    }
                    self._high_reward_acc = np.zeros((self.n_rollout_threads, 1), dtype=np.float32)
                    self._high_reward_count = 0
                    self._high_pending = True

                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = (
                    obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)

                #把高層的狀態/動作/回饋寫進高層 buffer
                if self.use_hierarchical and self._high_pending:
                    step_reward = None
                    if hasattr(self.envs, "envs"):
                        step_reward = np.zeros((self.n_rollout_threads, 1), dtype=np.float32)
                        valid = False
                        for t, env in enumerate(self.envs.envs):
                            src = env.env if hasattr(env, "env") else env
                            if hasattr(src, "high_reward_step"):
                                step_reward[t, 0] = float(src.high_reward_step)
                                valid = True
                        if not valid:
                            step_reward = None
                    else:
                        if hasattr(self.envs, "high_reward_step"):
                            step_reward = np.array([[float(self.envs.high_reward_step)]], dtype=np.float32)
                    if step_reward is None:
                        step_reward = np.mean(rewards, axis=1)
                    interval_end = (step % self.hierarchical_interval) == (self.hierarchical_interval - 1)
                    episode_end = step == (self.episode_length - 1)
                    if interval_end or episode_end:
                        if self.use_high_peruser and hasattr(self.envs, "envs"):
                            for t, env in enumerate(self.envs.envs):
                                src = env.env if hasattr(env, "env") else env
                                if hasattr(src, "compute_interval_reward"):
                                    step_reward[t, 0] = float(src.compute_interval_reward())
                        elif hasattr(self.envs, "compute_interval_reward"):
                            step_reward = np.array(
                                [[float(self.envs.compute_interval_reward())]], dtype=np.float32
                            )
                    self._high_reward_acc += step_reward
                    self._high_reward_count += 1
                    if interval_end or episode_end:
                        sum_reward = self._high_reward_acc
                        masks_h = np.ones((self.n_rollout_threads, 1, 1), dtype=np.float32)
                        masks_h[dones.all(axis=1)] = 0.0
                        t = self._high_transition
                        if self.use_high_peruser:
                            self.high_buffer.insert(
                                t["global_obs"],
                                t["global_obs"],
                                t["rnn_h"][:, None, ...],
                                t["rnn_hc"][:, None, ...],
                                t["action_h"],
                                t["logp_h"],
                                t["value_h"][:, None, :],
                                sum_reward.reshape(self.n_rollout_threads, 1, 1),
                                masks_h,
                            )
                        else:
                            self.high_buffer.insert(
                                t["global_obs"][:, None, ...],
                                t["global_obs"][:, None, ...],
                                t["rnn_h"][:, None, ...], t["rnn_hc"][:, None, ...],
                                # Reason: store model output directly; no runner-side mapping.
                                t["action_h"][:, None, :], t["logp_h"][:, None, :],
                                t["value_h"][:, None, :],
                                sum_reward[:, None, :],
                                masks_h
                            )
                        self._high_pending = False
                    high_episode_reward_sum += step_reward
                    high_episode_reward_count += 1


            # compute return and update network
            self.compute()
            train_infos = self.train()
            train_infos_high = {}

            #對高層 PPO 做 GAE / 更新
            if self.use_hierarchical and not self.freeze_high:
                # compute return
                # Reason: compute returns from high_buffer contents like low-level.
                self.high_trainer.prep_rollout()
                if self.use_high_peruser:
                    next_values = self.high_trainer.policy.get_values(
                        self.high_buffer.share_obs[-1],
                        self.high_buffer.rnn_states_critic[-1].squeeze(1),
                        self.high_buffer.masks[-1].squeeze(1),
                    )
                    next_values = _t2n(next_values)[:, None, :]
                else:
                    next_values = self.high_trainer.policy.get_values(
                        np.concatenate(self.high_buffer.share_obs[-1]),
                        np.concatenate(self.high_buffer.rnn_states_critic[-1]),
                        np.concatenate(self.high_buffer.masks[-1]),
                    )
                    next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
                self.high_buffer.compute_returns(next_values, self.high_trainer.value_normalizer)
                # train
                self.high_trainer.prep_training()
                train_infos_high = self.high_trainer.train(self.high_buffer)
                self.high_buffer.after_update()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save(episode, save_high_final=(episode == episodes - 1))
            #accumulate high reward
            if self.use_hierarchical and high_reward_list is not None and high_episode_reward_count > 0:
                high_reward_list[episode, 0] = float(
                    np.mean(high_episode_reward_sum / max(1, high_episode_reward_count))
                )
            # periodic reward snapshots (overwrite same filenames)
            if episode % 100 == 0:
                file_name = 'reward.mat'
                savemat(file_name, {'reward': reward_list[:episode + 1]})
                file_name = str(self.run_dir / 'reward.mat')
                savemat(file_name, {'reward': reward_list[:episode + 1]})
                if self.use_hierarchical and high_reward_list is not None:
                    file_name = 'high_reward.mat'
                    savemat(file_name, {'high_reward': high_reward_list[:episode + 1]})
                    file_name = str(self.run_dir / 'high_reward.mat')
                    savemat(file_name, {'high_reward': high_reward_list[:episode + 1]})

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        self.all_args.scenario_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )
                # if self.env_name == "MPE":
                #     env_infos = {}
                #     for agent_id in range(self.num_agents):
                #         idv_rews = []
                #         for info in infos:
                #             if 'individual_reward' in info[agent_id].keys():
                #                 idv_rews.append(info[agent_id]['individual_reward'])
                #         agent_k = 'agent%i/individual_rewards' % agent_id
                #         env_infos[agent_k] = idv_rews

                # train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards)
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                if self.use_hierarchical and high_reward_list is not None:
                    train_infos["average_episode_rewards_high"] = float(high_reward_list[episode, 0])
                    print("average high-level rewards is {}".format(train_infos["average_episode_rewards_high"]))
                if self.use_hierarchical and train_infos_high:
                    for k, v in train_infos_high.items():
                        train_infos[f"high/{k}"] = v
                self.log_train(train_infos, total_num_steps)
                reward_list[episode, 0] = np.mean(self.buffer.rewards)
                # self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

        file_name = 'reward.mat'
        savemat(file_name, {'reward': reward_list})
        #多存一個在結果路徑
        file_name = str(self.run_dir / 'reward.mat')
        savemat(file_name, {'reward': reward_list})
        if self.use_hierarchical and high_reward_list is not None:
            file_name = 'high_reward.mat'
            savemat(file_name, {'high_reward': high_reward_list})
            file_name = str(self.run_dir / 'high_reward.mat')
            savemat(file_name, {'high_reward': high_reward_list})

    def warmup(self):
        # reset env
        obs = self.envs.reset()  # shape = [env_num, agent_num, obs_dim]

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)  # shape = [env_num, agent_num * obs_dim]
            share_obs = np.expand_dims(share_obs, 1).repeat(
                self.num_agents, axis=1
            )  # shape = shape = [env_num, agent_num， agent_num * obs_dim]
        else:
            share_obs = obs
        #hierarchical
        if self.use_hierarchical:
            global_obs = self._get_global_obs_batch()
            if self.use_high_peruser:
                self.high_buffer.share_obs[0] = global_obs.copy()
                self.high_buffer.obs[0] = global_obs.copy()
            else:
                self.high_buffer.share_obs[0] = global_obs[:, None, ...].copy()
                self.high_buffer.obs[0] = global_obs[:, None, ...].copy()
            # Reason: reset interval-avg reward tracking at episode start.
            self._high_reward_acc = np.zeros((self.n_rollout_threads, 1), dtype=np.float32)
            self._high_reward_count = 0
            self._high_pending = False
            self._high_transition = {}

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        (
            value,
            action,
            action_log_prob,
            rnn_states,
            rnn_states_critic,
        ) = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
        )
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))  # [env_num, agent_num, 1]
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))  # [env_num, agent_num, action_dim]
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )  # [env_num, agent_num, 1]
        rnn_states = np.array(
            np.split(_t2n(rnn_states), self.n_rollout_threads)
        )  # [env_num, agent_num, 1, hidden_size]
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads)
        )  # [env_num, agent_num, 1, hidden_size]
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == "MultiDiscrete":
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == "Discrete":
            # actions  --> actions_env : shape:[10, 1] --> [5, 2, 5]
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            # TODO 这里改造成自己环境需要的形式即可
            # TODO Here, you can change the shape of actions_env to fit your environment
            actions_env = actions
            # raise NotImplementedError

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
        )

    def insert(self, data):
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
        )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]),
            dtype=np.float32,
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True,
            )
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            if self.eval_envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i] + 1)[
                        eval_actions[:, :, i]
                    ]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == "Discrete":
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos["eval_average_episode_rewards"] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos["eval_average_episode_rewards"])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs

        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render("rgb_array")[0][0]
                all_frames.append(image)
            else:
                envs.render("human")

            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            episode_rewards = []

            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True,
                )
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == "Discrete":
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render("rgb_array")[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render("human")

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        # if self.all_args.save_gifs:
        #     imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
