import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from utils.shared_buffer import SharedReplayBuffer
import gym

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
        self.low_model_dir = getattr(self.all_args, "low_model_dir", None)
        self.high_model_dir = getattr(self.all_args, "high_model_dir", None)
        self.freeze_low = getattr(self.all_args, "freeze_low", False)
        self.freeze_high = getattr(self.all_args, "freeze_high", False)

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

        if self.model_dir is not None or self.low_model_dir is not None or self.high_model_dir is not None:
            self.restore()

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)
        
        # buffer
        self.buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])
        
        # ----- hierarchical config -----
        self.use_hierarchical = getattr(self.all_args, "use_hierarchical", False)
        self.hierarchical_interval = getattr(self.all_args, "hierarchical_interval", 1)
        self.use_high_peruser = getattr(self.all_args, "use_high_peruser", False)
        self.cluster_size_candidates = getattr(
            self.all_args, "cluster_size_candidates", [1, 3, 5, 10]
        )

        if self.use_hierarchical:
            # 從第一個子環境拿 global obs shape
            env0 = self.envs.envs[0] if hasattr(self.envs, "envs") else self.envs
            global_obs = np.asarray(env0.get_global_obs(), dtype=np.float32)
            self.high_obs_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=global_obs.shape, dtype=np.float32
            )
            if self.use_high_peruser and hasattr(env0, "high_action_space") and env0.high_action_space is not None:
                self.high_action_space = env0.high_action_space
            else:
                self.high_action_space = gym.spaces.Discrete(len(self.cluster_size_candidates))

            if self.use_high_peruser:
                from algorithms.algorithm.high_policy import HighPolicy
                from utils.high_buffer import HighReplayBuffer

                self.high_policy = HighPolicy(
                    self.all_args,
                    self.high_obs_space,
                    self.high_obs_space,
                    self.high_action_space,
                    device=self.device,
                )
                self.high_trainer = TrainAlgo(self.all_args, self.high_policy, device=self.device)
                self.high_buffer = HighReplayBuffer(self.all_args, self.high_obs_space)
            else:
                self.high_policy = Policy(
                    self.all_args,
                    self.high_obs_space,
                    self.high_obs_space,
                    self.high_action_space,
                    device=self.device,
                )
                self.high_trainer = TrainAlgo(self.all_args, self.high_policy, device=self.device)

                self.high_buffer = SharedReplayBuffer(
                    self.all_args,
                    1,
                    self.high_obs_space,
                    self.high_obs_space,
                    self.high_action_space,
                )
        if self.freeze_low:
            for p in self.policy.actor.parameters():
                p.requires_grad = False
            for p in self.policy.critic.parameters():
                p.requires_grad = False
        if self.use_hierarchical and self.freeze_high:
            for p in self.high_policy.actor.parameters():
                p.requires_grad = False
            for p in self.high_policy.critic.parameters():
                p.requires_grad = False



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
        if self.freeze_low:
            return {}
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        self.buffer.after_update()
        return train_infos

    def save(self, episode=None, save_high_final=False):
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
        # save()
        if self.use_hierarchical:
            if save_high_final or episode is None:
                torch.save(self.high_trainer.policy.actor.state_dict(), str(self.save_dir) + "/actor_high.pt")
                torch.save(self.high_trainer.policy.critic.state_dict(), str(self.save_dir) + "/critic_high.pt")
            elif episode % 100 == 0:
                torch.save(
                    self.high_trainer.policy.actor.state_dict(),
                    str(self.save_dir) + f"/actor_high_{episode}.pt",
                )
                torch.save(
                    self.high_trainer.policy.critic.state_dict(),
                    str(self.save_dir) + f"/critic_high_{episode}.pt",
                )

        


    def restore(self):
        """Restore policy's networks from a saved model."""
        model_dir = self.low_model_dir or self.model_dir
        if model_dir is None:
            model_dir = self.model_dir
        policy_actor_state_dict = torch.load(str(model_dir) + '/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(str(model_dir) + '/critic.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)
            # restore()
            if self.use_hierarchical:
                high_dir = self.high_model_dir or self.model_dir
                high_actor = torch.load(str(high_dir) + "/actor_high.pt")
                self.high_policy.actor.load_state_dict(high_actor)
                if not self.all_args.use_render:
                    high_critic = torch.load(str(high_dir) + "/critic_high.pt")
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

    def _get_global_obs_batch(self):
        # 回傳 shape: [n_rollout_threads, obs_dim]
        if hasattr(self.envs, "envs"):
            return np.stack([env.get_global_obs() for env in self.envs.envs])
        return np.expand_dims(self.envs.get_global_obs(), axis=0)

    def _set_cluster_size_batch(self, sizes):
        # sizes: [n_rollout_threads] or scalar
        if not hasattr(self.envs, "envs"):
            self.envs.set_cluster_size(int(sizes))
            return
        for env, size in zip(self.envs.envs, sizes):
            env.set_cluster_size(int(size))
