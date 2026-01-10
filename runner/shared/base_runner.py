import os
import math
from copy import deepcopy
import numpy as np
import torch
import gym
from tensorboardX import SummaryWriter
from utils.shared_buffer import SharedReplayBuffer

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir / 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writter = SummaryWriter(self.log_dir)
        self.save_dir = str(self.run_dir / 'models')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        from algorithms.algorithm.r_mappo import RMAPPO as TrainAlgo
        from algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy as Policy

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        # policy network
        self.policy = Policy(self.all_args,
                            self.envs.observation_space[0],
                            share_observation_space,
                            self.envs.action_space[0],
                            device = self.device)

        if self.model_dir is not None:
            self.restore()

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)
        
        # buffer
        self.buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])
        #hierarchical
        self.use_hierarchical = getattr(self.all_args, "use_hierarchical", False)
        self.hierarchical_interval = getattr(self.all_args, "hierarchical_interval", 10)

        if self.use_hierarchical:
            # Reason: high-level params are separate while keeping single-layer behavior intact.
            high_args = deepcopy(self.all_args)
            high_args.lr = self.all_args.high_lr
            high_args.critic_lr = self.all_args.high_critic_lr
            high_args.hidden_size = self.all_args.high_hidden_size
            high_args.layer_N = self.all_args.high_layer_N
            high_args.recurrent_N = self.all_args.high_recurrent_N
            high_args.data_chunk_length = self.all_args.high_data_chunk_length
            high_args.ppo_epoch = self.all_args.high_ppo_epoch
            high_args.clip_param = self.all_args.high_clip_param
            high_args.num_mini_batch = self.all_args.high_num_mini_batch
            high_args.entropy_coef = self.all_args.high_entropy_coef
            high_args.value_loss_coef = self.all_args.high_value_loss_coef
            high_args.max_grad_norm = self.all_args.high_max_grad_norm
            high_args.gamma = self.all_args.high_gamma
            high_args.gae_lambda = self.all_args.high_gae_lambda
            high_args.episode_length = int(math.ceil(self.episode_length / self.hierarchical_interval))

            env0 = self.envs.envs[0] if hasattr(self.envs, "envs") else self.envs
            # Reason: spaces are defined in env, runner only consumes them.
            self.high_obs_space = getattr(env0, "high_observation_space", None)
            self.high_action_space = getattr(env0, "high_action_space", None)
            if self.high_obs_space is None or self.high_action_space is None:
                raise ValueError("Hierarchical env must define high_observation_space and high_action_space.")

            self.high_policy = Policy(
                high_args,
                self.high_obs_space,
                self.high_obs_space,
                self.high_action_space,
                device=self.device,
            )
            self.high_trainer = TrainAlgo(high_args, self.high_policy, device=self.device)
            self.high_buffer = SharedReplayBuffer(
                high_args, 1,
                self.high_obs_space, self.high_obs_space,
                self.high_action_space
            )

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        self.buffer.after_update()
        return train_infos

    def save(self, episode=None):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer.policy.actor
        policy_critic = self.trainer.policy.critic
        
        # 如果有傳入 episode，就加在檔名後面，否則維持原樣
        if episode is not None:
            actor_name = "/actor_%d.pt" % episode
            critic_name = "/critic_%d.pt" % episode
        else:
            actor_name = "/actor.pt"
            critic_name = "/critic.pt"
            
        torch.save(policy_actor.state_dict(), str(self.save_dir) + actor_name)
        torch.save(policy_critic.state_dict(), str(self.save_dir) + critic_name)
        if self.use_hierarchical:
            torch.save(self.high_trainer.policy.actor.state_dict(), str(self.save_dir) + "/actor_high.pt")
            torch.save(self.high_trainer.policy.critic.state_dict(), str(self.save_dir) + "/critic_high.pt")

    def restore(self):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)
            if self.use_hierarchical:
                high_actor = torch.load(str(self.model_dir) + "/actor_high.pt")
                self.high_policy.actor.load_state_dict(high_actor)
                if not self.all_args.use_render:
                    high_critic = torch.load(str(self.model_dir) + "/critic_high.pt")
                    self.high_policy.critic.load_state_dict(high_critic)
 
    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    #hierarchical
    def _get_global_obs_batch(self):
        if hasattr(self.envs, "envs"):
            return np.stack([env.get_global_obs() for env in self.envs.envs])
        return np.expand_dims(self.envs.get_global_obs(), axis=0)
