import torch
import torch.nn as nn

from .mlp import MLPLayer


class HighContextEncoder(nn.Module):
    def __init__(self, args, obs_dim):
        super(HighContextEncoder, self).__init__()
        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size
        self._encoder_type = getattr(args, "high_encoder_type", "set")

        self.tau = 2.0

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.embed = MLPLayer(
            obs_dim,
            self.hidden_size,
            self._layer_N,
            self._use_orthogonal,
            self._use_ReLU,
        )

        if self._encoder_type == "set":
            num_heads = getattr(args, "high_num_heads", 4)
            if self.hidden_size % num_heads != 0:
                num_heads = 1
            self.attn = nn.MultiheadAttention(
                embed_dim=self.hidden_size, num_heads=num_heads, batch_first=True
            )
            self.attn_norm = nn.LayerNorm(self.hidden_size)

        if self._encoder_type in ("set", "noattn-lp"):
            self.pool_score = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)
        batch_size, set_size, feat_dim = x.shape
        x = x.reshape(batch_size * set_size, feat_dim)
        x = self.embed(x)
        x = x.reshape(batch_size, set_size, -1)

        if self._encoder_type == "set":
            attn_out, _ = self.attn(x, x, x, need_weights=False)
            h_ctx = self.attn_norm(x + attn_out)
        else:
            h_ctx = x

        if self._encoder_type in ("set", "noattn-lp"):
            scores = self.pool_score(h_ctx).squeeze(-1)
            weights = torch.softmax(scores / self.tau, dim=1).unsqueeze(-1)
            g = torch.sum(h_ctx * weights, dim=1)
        else:
            g = torch.mean(h_ctx, dim=1)

        return h_ctx, g
