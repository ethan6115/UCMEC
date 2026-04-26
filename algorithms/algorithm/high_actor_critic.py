import itertools

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
        self._actor_type = getattr(args, "high_actor_type", "mlp")

        obs_dim = obs_space.shape[-1]
        self._is_multibinary = action_space.__class__.__name__ == "MultiBinary"
        if self._is_multibinary:
            # MultiBinary can be defined with a tuple shape, e.g. (M_sim, 10).
            self.action_dim = int(action_space.shape[-1])
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.action_dim = int(action_space.nvec[0])
        else:
            self.action_dim = int(action_space.n)

        if self._actor_type == "pair_scorer":
            # --- Structured Pair Scorer ---
            # Obs layout: beta(cn) | mask(cn) | delay,uplink,front,satisfy(4) | pos(2) | speed(1) | front_stats(K*cn)
            self._candidate_n = getattr(args, "candidate_n", 8)
            self._k_fixed = getattr(args, "k_fixed", 2)
            self._num_cpus = getattr(args, "num_cpus", 3)
            self._ap_combos = list(itertools.combinations(range(self._candidate_n), self._k_fixed))
            # Ablation flags
            self._pair_repr_type = getattr(args, "high_pair_repr", "sdp")   # "sdp" | "concat"
            self._use_global_ctx = not getattr(args, "high_no_global_ctx", False)
            # Pre-compute obs indices for per-AP features: [beta, mask, front_cpu0, front_cpu1, front_cpu2]
            ap_feat_dim = 2 + self._num_cpus  # beta + mask + front per cpu
            self._ap_feat_dim = ap_feat_dim
            # Indices into obs for each AP a: beta[a], mask[cn+a], front_stats[23+k*cn+a] for k=0..K-1
            ap_indices = []
            front_base = 2 * self._candidate_n + 7  # after beta(cn) + mask(cn) + 4 + 2 + 1
            for a in range(self._candidate_n):
                idx = [a, self._candidate_n + a]  # beta[a], mask[a]
                for k in range(self._num_cpus):
                    idx.append(front_base + k * self._candidate_n + a)
                ap_indices.append(idx)
            # Register as buffer so it moves with .to(device)
            self.register_buffer("_ap_indices", torch.tensor(ap_indices, dtype=torch.long))  # [cn, ap_feat_dim]
            # Combo pair indices
            combo_i = [c[0] for c in self._ap_combos]
            combo_j = [c[1] for c in self._ap_combos]
            self.register_buffer("_combo_i", torch.tensor(combo_i, dtype=torch.long))
            self.register_buffer("_combo_j", torch.tensor(combo_j, dtype=torch.long))
            # User context indices: delay, uplink, front, satisfy, pos_x, pos_y, speed
            # These 7 fields start right after beta(cn) + mask(cn) = 2*cn
            ctx_start = 2 * self._candidate_n
            self.register_buffer("_user_ctx_indices", torch.tensor(list(range(ctx_start, ctx_start + 7)), dtype=torch.long))
            self._user_ctx_dim = 7

            assert len(self._ap_combos) == self.action_dim, (
                f"ap_combos ({len(self._ap_combos)}) != action_dim ({self.action_dim}); "
                f"check candidate_n={self._candidate_n} k_fixed={self._k_fixed}"
            )
            assert front_base + self._num_cpus * self._candidate_n <= obs_dim, (
                f"obs index out of range: front_base={front_base} + "
                f"num_cpus={self._num_cpus}*candidate_n={self._candidate_n} > obs_dim={obs_dim}"
            )

            # AP embedding MLP (shared across all APs)
            ap_embed_dim = 32
            self._ap_embed_dim = ap_embed_dim
            act_fn = nn.ReLU() if self._use_ReLU else nn.Tanh()
            init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
            gain = nn.init.calculate_gain(['tanh', 'relu'][self._use_ReLU])

            def init_(m):
                return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

            def init_final_(m):
                return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=self._gain)

            self.ap_mlp = nn.Sequential(
                init_(nn.Linear(ap_feat_dim, ap_embed_dim)),
                act_fn,
                init_(nn.Linear(ap_embed_dim, ap_embed_dim)),
                act_fn,
            )
            # Pair representation dim: both paths output 96D for fair capacity comparison.
            # sdp: sum+|diff|+prod directly gives 96D.
            # concat: [e_i, e_j]=64D -> linear projection to 96D (no activation, capacity alignment only).
            pair_repr_dim = ap_embed_dim * 3  # 96
            if self._pair_repr_type == "concat":
                self.concat_proj = nn.Linear(ap_embed_dim * 2, pair_repr_dim, bias=False)
                init_method(self.concat_proj.weight, gain=gain)
            # Global context branch (ablation: disabled by --high_no_global_ctx)
            # RNN must use hidden_size to match buffer rnn_states shape.
            self._g_proj_dim = 16
            if self._use_global_ctx:
                self.g_pre = init_(nn.Linear(ap_embed_dim, self.hidden_size))
                if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                    self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
                self.g_proj = init_(nn.Linear(self.hidden_size, self._g_proj_dim))
            # Scorer MLP input dim
            scorer_in_dim = pair_repr_dim + self._user_ctx_dim
            if self._use_global_ctx:
                scorer_in_dim += self._g_proj_dim
            self.scorer = nn.Sequential(
                init_(nn.Linear(scorer_in_dim, 64)),
                act_fn,
                init_final_(nn.Linear(64, 1)),
            )
        else:
            # --- Original MLP actor head ---
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

    def _compute_pair_logits(self, obs):
        """Compute combo logits using structured pair scoring."""
        B, M, _ = obs.shape
        # Extract per-AP features: [B, M, cn, ap_feat_dim]
        ap_feats = obs[:, :, self._ap_indices]  # fancy index with [cn, feat_dim]
        # Shared AP embedding
        ap_embeds = self.ap_mlp(ap_feats)  # [B, M, cn, 32]
        # Pair interaction for each combo
        e_i = ap_embeds[:, :, self._combo_i]  # [B, M, 28, 32]
        e_j = ap_embeds[:, :, self._combo_j]  # [B, M, 28, 32]
        if self._pair_repr_type == "sdp":
            pair_repr = torch.cat([e_i + e_j, torch.abs(e_i - e_j), e_i * e_j], dim=-1)  # [B, M, 28, 96]
        else:
            raw = torch.cat([e_i, e_j], dim=-1)                          # [B, M, 28, 64]
            B2, M2, C2, _ = raw.shape
            pair_repr = self.concat_proj(raw.reshape(B2 * M2 * C2, -1)).reshape(B2, M2, C2, -1)  # [B, M, 28, 96]
        # User context
        user_ctx = obs[:, :, self._user_ctx_indices]  # [B, M, 7]
        user_ctx_expand = user_ctx.unsqueeze(2).expand(B, M, len(self._ap_combos), self._user_ctx_dim)
        # Global context from AP embeds: per-user mean over APs only (decentralized)
        g = ap_embeds.mean(dim=2)  # [B, M, 32]  mean over APs, keep user dim
        return pair_repr, user_ctx_expand, g, ap_embeds

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if self._actor_type == "pair_scorer":
            pair_repr, user_ctx_expand, g, _ = self._compute_pair_logits(obs)
            B, M = obs.shape[0], obs.shape[1]
            num_combos = len(self._ap_combos)
            if self._use_global_ctx:
                # g: [B, M, 32] -> flatten to [B*M, 32] for shared MLP/RNN
                g_flat = g.reshape(B * M, -1)                          # [B*M, 32]
                g_flat = self.g_pre(g_flat)                            # [B*M, hidden_size]
                if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                    # rnn_states: rollout=[B, M, recN, H], training=[N, M, recN, H]
                    N_rnn = rnn_states.shape[0]
                    rnn_states_flat = rnn_states.reshape(N_rnn * M, *rnn_states.shape[2:])  # [N_rnn*M, recN, H]
                    # masks: [B, 1] -> expand per-user -> [B*M, 1]
                    masks_flat = masks.unsqueeze(1).expand(B, M, 1).reshape(B * M, 1)
                    g_flat, rnn_states_out = self.rnn(g_flat, rnn_states_flat, masks_flat)
                    rnn_states = rnn_states_out.reshape(N_rnn, M, *rnn_states_out.shape[1:])  # [N_rnn, M, recN, H]
                g_proj = self.g_proj(g_flat).reshape(B, M, -1)         # [B, M, g_proj_dim]
                g_expand = g_proj.unsqueeze(2).expand(B, M, num_combos, self._g_proj_dim)
                scorer_input = torch.cat([pair_repr, user_ctx_expand, g_expand], dim=-1)
            else:
                # global ctx disabled: rnn_states passed through unchanged
                scorer_input = torch.cat([pair_repr, user_ctx_expand], dim=-1)
            logits = self.scorer(scorer_input).squeeze(-1)          # [B, M, 45]
        else:
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

        if self._is_multibinary:
            dist = torch.distributions.Bernoulli(logits=logits)
            if deterministic:
                actions = (dist.probs >= 0.5).to(logits.dtype)
            else:
                actions = dist.sample()
            # Keep per-user log-prob (sum over bits only). User dimension is handled
            # later by the high-level replay buffer / PPO sample expansion.
            action_log_probs = dist.log_prob(actions).sum(-1, keepdim=True)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            if deterministic:
                actions = torch.argmax(logits, dim=-1)
            else:
                actions = dist.sample()
            action_log_probs = dist.log_prob(actions).unsqueeze(-1)
            actions = actions.unsqueeze(-1)
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if self._actor_type == "pair_scorer":
            pair_repr, user_ctx_expand, g, _ = self._compute_pair_logits(obs)
            B, M = obs.shape[0], obs.shape[1]
            num_combos = len(self._ap_combos)
            if self._use_global_ctx:
                g_flat = g.reshape(B * M, -1)                          # [B*M, 32]
                g_flat = self.g_pre(g_flat)                            # [B*M, hidden_size]
                if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                    N_rnn = rnn_states.shape[0]
                    rnn_states_flat = rnn_states.reshape(N_rnn * M, *rnn_states.shape[2:])
                    masks_flat = masks.unsqueeze(1).expand(B, M, 1).reshape(B * M, 1)
                    g_flat, _ = self.rnn(g_flat, rnn_states_flat, masks_flat)
                g_proj = self.g_proj(g_flat).reshape(B, M, -1)
                g_expand = g_proj.unsqueeze(2).expand(B, M, num_combos, self._g_proj_dim)
                scorer_input = torch.cat([pair_repr, user_ctx_expand, g_expand], dim=-1)
            else:
                # global ctx disabled: rnn_states not used
                scorer_input = torch.cat([pair_repr, user_ctx_expand], dim=-1)
            logits = self.scorer(scorer_input).squeeze(-1)
        else:
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

        if self._is_multibinary:
            dist = torch.distributions.Bernoulli(logits=logits)
            # Keep per-user log-prob (sum over bits only).
            action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)
            # Per-user entropy (sum over bits); reduce to scalar with masks below to
            # preserve compatibility with the shared PPO trainer interface.
            dist_entropy = dist.entropy().sum(-1, keepdim=True)
        else:
            action = action.squeeze(-1).long()
            dist = torch.distributions.Categorical(logits=logits)
            action_log_probs = dist.log_prob(action).unsqueeze(-1)
            dist_entropy = dist.entropy().unsqueeze(-1)
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
