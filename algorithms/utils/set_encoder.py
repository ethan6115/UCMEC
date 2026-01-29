import torch
import torch.nn as nn
from .mlp import MLPLayer


class SetEncoder(nn.Module):
    def __init__(self, args, obs_shape):
        super(SetEncoder, self).__init__()
        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        self.tau = 2.0

        obs_dim = obs_shape[-1]

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(obs_dim, self.hidden_size,
                            self._layer_N, self._use_orthogonal, self._use_ReLU)
        num_heads = getattr(args, "set_num_heads", getattr(args, "high_num_heads", 4))
        if self.hidden_size % num_heads != 0:
            num_heads = 1
        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size, num_heads=num_heads, batch_first=True
        )
        self.attn_score = nn.Linear(self.hidden_size, 1)
        #print max, entropy
        self.last_attn_max = None
        self.last_attn_entropy = None

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if self._use_feature_normalization:
            x = self.feature_norm(x)
        batch_size, set_size, feat_dim = x.shape
        x = x.reshape(batch_size * set_size, feat_dim)
        x = self.mlp(x)
        x = x.reshape(batch_size, set_size, -1)
        #x = x.mean(dim=1) #平均版deepset
        x, _ = self.attn(x, x, x, need_weights=False)
        scores = self.attn_score(x).squeeze(-1)
        weights = torch.softmax(scores / self.tau, dim=1).unsqueeze(-1)
        #print max, entropy
        with torch.no_grad():
            w = weights.squeeze(-1)
            self.last_attn_max = float(w.max(dim=1).values.mean().item())
            ent = -(w * torch.log(w + 1e-8)).sum(dim=1).mean()
            self.last_attn_entropy = float(ent.item())
        x = torch.sum(x * weights, dim=1)
        return x
