import numpy as np


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


class HighReplayBuffer(object):
    def __init__(self, args, obs_space):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits
        self.use_high_peruser_credit = getattr(args, "use_high_peruser_credit", False)

        obs_shape = obs_space.shape
        self.credit_users = obs_shape[0] if self.use_high_peruser_credit else 1

        self.share_obs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, *obs_shape), dtype=np.float32
        )
        self.obs = np.zeros_like(self.share_obs)

        self.num_users = obs_shape[0]
        # pair_scorer uses per-user hidden states; mlp uses a single shared state
        self.actor_rnn_slots = (
            self.num_users if getattr(args, 'high_actor_type', 'mlp') == 'pair_scorer' else 1
        )
        self.rnn_states = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads,
             self.actor_rnn_slots, self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        # critic always uses a single global hidden state
        self.rnn_states_critic = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads,
             1, self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )

        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.credit_users, 1), dtype=np.float32
        )
        self.returns = np.zeros_like(self.value_preds)

        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, obs_shape[0], 1), dtype=np.int64
        )
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, obs_shape[0], 1), dtype=np.float32
        )
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, self.credit_users, 1), dtype=np.float32
        )

        self.masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, self.credit_users, 1), dtype=np.float32
        )
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0

    def insert(
        self,
        share_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
        bad_masks=None,
        active_masks=None,
    ):
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states_actor.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]
                        ) * self.masks[step + 1] - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * gae * self.masks[step + 1]
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = (
                            self.rewards[step]
                            + self.gamma * self.value_preds[step + 1] * self.masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        self.returns[step] = (
                            self.returns[step + 1] * self.gamma * self.masks[step + 1]
                            + self.rewards[step]
                        ) * self.bad_masks[step + 1] + (
                            1 - self.bad_masks[step + 1]
                        ) * value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:
                        self.returns[step] = (
                            self.returns[step + 1] * self.gamma * self.masks[step + 1]
                            + self.rewards[step]
                        ) * self.bad_masks[step + 1] + (
                            1 - self.bad_masks[step + 1]
                        ) * self.value_preds[step]
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]
                        ) * self.masks[step + 1] - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = (
                            self.rewards[step]
                            + self.gamma * self.value_preds[step + 1] * self.masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = (
                        self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]
                    )

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        episode_length, n_rollout_threads, num_users, action_dim = self.actions.shape
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, batch_size, num_mini_batch)
            )
            mini_batch_size = batch_size // num_mini_batch

        rand = np.random.permutation(batch_size)
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size] for i in range(num_mini_batch)
        ]

        share_obs = self.share_obs[:-1].reshape(batch_size, *self.share_obs.shape[2:])
        obs = self.obs[:-1].reshape(batch_size, *self.obs.shape[2:])
        if self.actor_rnn_slots > 1:
            rnn_states = self.rnn_states[:-1].reshape(
                batch_size, self.actor_rnn_slots, self.recurrent_N, self.hidden_size)
        else:
            rnn_states = self.rnn_states[:-1].reshape(batch_size, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(batch_size, *self.rnn_states_critic.shape[3:])
        actions = self.actions.reshape(batch_size, num_users, action_dim)
        if self.use_high_peruser_credit:
            value_preds = self.value_preds[:-1].reshape(batch_size, self.credit_users, 1)
            returns = self.returns[:-1].reshape(batch_size, self.credit_users, 1)
            # Actor/critic RNN masks are sequence-level, keep scalar mask channel.
            masks = self.masks[:-1, :, :1, :].reshape(batch_size, 1)
            active_masks = self.active_masks[:-1].reshape(batch_size, self.credit_users, 1)
        else:
            value_preds = self.value_preds[:-1].reshape(batch_size, 1)
            returns = self.returns[:-1].reshape(batch_size, 1)
            masks = self.masks[:-1].reshape(batch_size, 1)
            active_masks = self.active_masks[:-1].reshape(batch_size, 1)
        action_log_probs = self.action_log_probs.reshape(batch_size, num_users, 1)
        if self.use_high_peruser_credit:
            adv_targ_all = advantages.reshape(batch_size, num_users, 1)
        else:
            advantages = advantages.reshape(batch_size, 1)
            adv_targ_all = np.broadcast_to(advantages[:, None, :], (batch_size, num_users, 1))

        for indices in sampler:
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            adv_targ = adv_targ_all[indices]

            yield (
                share_obs_batch,
                obs_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                None,
            )

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        episode_length, n_rollout_threads, num_users, action_dim = self.actions.shape
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length
        mini_batch_size = data_chunks // num_mini_batch

        rand = np.random.permutation(data_chunks)
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size] for i in range(num_mini_batch)
        ]

        share_obs = self.share_obs[:-1].transpose(1, 0, 2, 3).reshape(batch_size, *self.share_obs.shape[2:])
        obs = self.obs[:-1].transpose(1, 0, 2, 3).reshape(batch_size, *self.obs.shape[2:])
        actions = self.actions.transpose(1, 0, 2, 3).reshape(batch_size, num_users, action_dim)
        action_log_probs = self.action_log_probs.transpose(1, 0, 2, 3).reshape(
            batch_size, num_users, 1
        )
        if self.use_high_peruser_credit:
            value_preds = self.value_preds[:-1].transpose(1, 0, 2, 3).reshape(
                batch_size, self.credit_users, 1
            )
            returns = self.returns[:-1].transpose(1, 0, 2, 3).reshape(
                batch_size, self.credit_users, 1
            )
            masks = self.masks[:-1, :, :1, :].transpose(1, 0, 2, 3).reshape(batch_size, 1)
            active_masks = self.active_masks[:-1].transpose(1, 0, 2, 3).reshape(
                batch_size, self.credit_users, 1
            )
        else:
            value_preds = self.value_preds[:-1].transpose(1, 0, 2, 3).reshape(batch_size, 1)
            returns = self.returns[:-1].transpose(1, 0, 2, 3).reshape(batch_size, 1)
            masks = self.masks[:-1].transpose(1, 0, 2, 3).reshape(batch_size, 1)
            active_masks = self.active_masks[:-1].transpose(1, 0, 2, 3).reshape(batch_size, 1)
        if self.use_high_peruser_credit:
            advantages_user = advantages.transpose(1, 0, 2, 3).reshape(batch_size, num_users, 1)
        else:
            advantages = advantages.transpose(1, 0, 2, 3).reshape(batch_size, 1)
            advantages_user = np.broadcast_to(advantages[:, None, :], (batch_size, num_users, 1))
        if self.actor_rnn_slots > 1:
            rnn_states = self.rnn_states[:-1].transpose(1, 0, 2, 3, 4).reshape(
                batch_size, self.actor_rnn_slots, self.recurrent_N, self.hidden_size)
        else:
            rnn_states = self.rnn_states[:-1].transpose(1, 0, 2, 3, 4).reshape(
                batch_size, 1, self.recurrent_N, self.hidden_size)
        rnn_states_critic = self.rnn_states_critic[:-1].transpose(1, 0, 2, 3, 4).reshape(
            batch_size, 1, self.recurrent_N, self.hidden_size
        )

        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for index in indices:
                ind = index * data_chunk_length
                share_obs_batch.append(share_obs[ind : ind + data_chunk_length])
                obs_batch.append(obs[ind : ind + data_chunk_length])
                actions_batch.append(actions[ind : ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind : ind + data_chunk_length])
                return_batch.append(returns[ind : ind + data_chunk_length])
                masks_batch.append(masks[ind : ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind : ind + data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind : ind + data_chunk_length])
                adv_targ.append(advantages_user[ind : ind + data_chunk_length])
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            obs_batch = np.stack(obs_batch, axis=1)
            actions_batch = np.stack(actions_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)

            if self.actor_rnn_slots > 1:
                rnn_states_batch = np.stack(rnn_states_batch).reshape(
                    N, self.actor_rnn_slots, self.recurrent_N, self.hidden_size)
            else:
                rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(
                N, *self.rnn_states_critic.shape[3:])

            share_obs_batch = _flatten(L, N, share_obs_batch)
            obs_batch = _flatten(L, N, obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)

            yield (
                share_obs_batch,
                obs_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                None,
            )

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        episode_length, n_rollout_threads, num_users, action_dim = self.actions.shape
        batch_size = n_rollout_threads
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(n_rollout_threads, num_mini_batch)
        )
        num_envs_per_batch = batch_size // num_mini_batch
        perm = np.random.permutation(batch_size)

        share_obs = self.share_obs[:-1].transpose(1, 0, 2, 3)
        obs = self.obs[:-1].transpose(1, 0, 2, 3)
        actions = self.actions.transpose(1, 0, 2, 3)
        action_log_probs = self.action_log_probs.transpose(1, 0, 2, 3)
        if self.use_high_peruser_credit:
            value_preds = self.value_preds[:-1].transpose(1, 0, 2, 3).reshape(
                n_rollout_threads, self.episode_length, self.credit_users, 1
            )
            returns = self.returns[:-1].transpose(1, 0, 2, 3).reshape(
                n_rollout_threads, self.episode_length, self.credit_users, 1
            )
            masks = self.masks[:-1, :, :1, :].transpose(1, 0, 2, 3).reshape(
                n_rollout_threads, self.episode_length, 1
            )
            active_masks = self.active_masks[:-1].transpose(1, 0, 2, 3).reshape(
                n_rollout_threads, self.episode_length, self.credit_users, 1
            )
        else:
            value_preds = self.value_preds[:-1].transpose(1, 0, 2, 3).reshape(
                n_rollout_threads, self.episode_length, 1
            )
            returns = self.returns[:-1].transpose(1, 0, 2, 3).reshape(
                n_rollout_threads, self.episode_length, 1
            )
            masks = self.masks[:-1].transpose(1, 0, 2, 3).reshape(
                n_rollout_threads, self.episode_length, 1
            )
            active_masks = self.active_masks[:-1].transpose(1, 0, 2, 3).reshape(
                n_rollout_threads, self.episode_length, 1
            )
        if self.use_high_peruser_credit:
            advantages_user = advantages.transpose(1, 0, 2, 3).reshape(
                n_rollout_threads, self.episode_length, num_users, 1
            )
        else:
            advantages = advantages.transpose(1, 0, 2, 3).reshape(n_rollout_threads, self.episode_length, 1)
            advantages_user = np.broadcast_to(
                advantages[:, :, None, :], (n_rollout_threads, self.episode_length, num_users, 1)
            )

        for start_ind in range(0, batch_size, num_envs_per_batch):
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                share_obs_batch.append(share_obs[ind])
                obs_batch.append(obs[ind])
                actions_batch.append(actions[ind])
                value_preds_batch.append(value_preds[ind])
                return_batch.append(returns[ind])
                masks_batch.append(masks[ind])
                active_masks_batch.append(active_masks[ind])
                old_action_log_probs_batch.append(action_log_probs[ind])
                adv_targ.append(advantages_user[ind])
                rnn_states_batch.append(self.rnn_states[0:1, ind])
                rnn_states_critic_batch.append(self.rnn_states_critic[0:1, ind])

            T, N = self.episode_length, num_envs_per_batch
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            obs_batch = np.stack(obs_batch, axis=1)
            actions_batch = np.stack(actions_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)

            if self.actor_rnn_slots > 1:
                rnn_states_batch = np.stack(rnn_states_batch).reshape(
                    N, self.actor_rnn_slots, self.recurrent_N, self.hidden_size)
            else:
                rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(
                N, *self.rnn_states_critic.shape[3:])

            share_obs_batch = _flatten(T, N, share_obs_batch)
            obs_batch = _flatten(T, N, obs_batch)
            actions_batch = _flatten(T, N, actions_batch)
            value_preds_batch = _flatten(T, N, value_preds_batch)
            return_batch = _flatten(T, N, return_batch)
            masks_batch = _flatten(T, N, masks_batch)
            active_masks_batch = _flatten(T, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(T, N, old_action_log_probs_batch)
            adv_targ = _flatten(T, N, adv_targ)

            yield (
                share_obs_batch,
                obs_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                None,
            )
