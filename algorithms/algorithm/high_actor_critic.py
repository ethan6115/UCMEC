import torch
import torch.nn as nn

from algorithms.utils.util import init, check
from algorithms.utils.popart import PopArt
from algorithms.utils.high_context_encoder import HighContextEncoder
from algorithms.utils.rnn import RNNLayer


class HighActor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(HighActor, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._layer_N = args.layer_N
        self._gain = args.gain
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_dim = obs_space.shape[-1]
        if action_space.__class__.__name__ == "MultiBinary":
            # MultiBinary can be defined with a tuple shape, e.g. (M_sim, 10).
            self.action_dim = int(action_space.shape[-1])
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.action_dim = int(action_space.nvec[0])
        else:
            self.action_dim = int(action_space.n)

        self.encoder = HighContextEncoder(args, obs_dim)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
        self.actor_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU() if self._use_ReLU else nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU() if self._use_ReLU else nn.Tanh(),
        )
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=self._gain)

        self.logits = init_(nn.Linear(self.hidden_size, self.action_dim))
        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        h_ctx, g = self.encoder(obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            g, rnn_states = self.rnn(g, rnn_states, masks)
        g_expand = g.unsqueeze(1).expand_as(h_ctx)
        actor_in = torch.cat([h_ctx, g_expand], dim=-1)
        flat = actor_in.reshape(-1, actor_in.shape[-1])
        feat = self.actor_mlp(flat)
        logits = self.logits(feat).reshape(obs.shape[0], obs.shape[1], -1)

        if available_actions is not None:
            avail = check(available_actions).to(**self.tpdv)
            logits = logits.masked_fill(avail <= 0.0, -1e10)

        dist = torch.distributions.Bernoulli(logits=logits)
        if deterministic:
            actions = (dist.probs >= 0.5).to(logits.dtype)
        else:
            actions = dist.sample()
        # Keep per-user log-prob (sum over bits only). User dimension is handled
        # later by the high-level replay buffer / PPO sample expansion.
        action_log_probs = dist.log_prob(actions).sum(-1, keepdim=True)
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        h_ctx, g = self.encoder(obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            g, _ = self.rnn(g, rnn_states, masks)
        g_expand = g.unsqueeze(1).expand_as(h_ctx)
        actor_in = torch.cat([h_ctx, g_expand], dim=-1)
        flat = actor_in.reshape(-1, actor_in.shape[-1])
        feat = self.actor_mlp(flat)
        logits = self.logits(feat).reshape(obs.shape[0], obs.shape[1], -1)

        if available_actions is not None:
            avail = check(available_actions).to(**self.tpdv)
            logits = logits.masked_fill(avail <= 0.0, -1e10)

        dist = torch.distributions.Bernoulli(logits=logits)
        # Keep per-user log-prob (sum over bits only).
        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)
        # Per-user entropy (sum over bits); reduce to scalar with masks below to
        # preserve compatibility with the shared PPO trainer interface.
        dist_entropy = dist.entropy().sum(-1, keepdim=True)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
            if active_masks.dim() == 2:
                active_masks = active_masks.unsqueeze(-1)
            if active_masks.shape[1] != dist_entropy.shape[1]:
                if active_masks.shape[1] == 1:
                    active_masks = active_masks.expand(-1, dist_entropy.shape[1], -1)
                else:
                    # Fallback: reduce and broadcast if caller provides an
                    # unexpected shape.
                    active_masks = active_masks.mean(dim=1, keepdim=True).expand(
                        -1, dist_entropy.shape[1], -1
                    )
            dist_entropy = (dist_entropy * active_masks).sum() / active_masks.sum()
        else:
            dist_entropy = dist_entropy.mean()
        return action_log_probs, dist_entropy


class HighCritic(nn.Module):
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(HighCritic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._layer_N = args.layer_N
        self._use_popart = args.use_popart
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_high_peruser_credit = getattr(args, "use_high_peruser_credit", False)
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_dim = cent_obs_space.shape[-1]
        self.encoder = HighContextEncoder(args, obs_dim)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.critic_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU() if self._use_ReLU else nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU() if self._use_ReLU else nn.Tanh(),
        )

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        h_ctx, g = self.encoder(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            g, rnn_states = self.rnn(g, rnn_states, masks)
        if self._use_high_peruser_credit:
            g_expand = g.unsqueeze(1).expand_as(h_ctx)
            critic_in = torch.cat([h_ctx, g_expand], dim=-1)
            flat = critic_in.reshape(-1, critic_in.shape[-1])
            feat = self.critic_mlp(flat)
            values = self.v_out(feat).reshape(cent_obs.shape[0], cent_obs.shape[1], 1)
        else:
            values = self.v_out(g)
        return values, rnn_states
