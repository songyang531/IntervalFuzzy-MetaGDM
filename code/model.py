import math

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_CONTEXT_HIDDEN_DIM = 64
DEFAULT_SOCIAL_HIDDEN_DIM = 256
DEFAULT_Z_DIM = 8
DEFAULT_NUM_HEADS = 4
DEFAULT_CONTEXT_LAYERS = 2


class ContextEncoder(nn.Module):
    def __init__(
            self,
            input_dim=3,
            hidden_dim=DEFAULT_CONTEXT_HIDDEN_DIM,
            z_dim=DEFAULT_Z_DIM,
            num_heads=DEFAULT_NUM_HEADS,
            num_layers=DEFAULT_CONTEXT_LAYERS):
        super(ContextEncoder, self).__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.z_mean = nn.Linear(hidden_dim, z_dim)
        self.z_log_std = nn.Linear(hidden_dim, z_dim)

    def forward(self, history):
        x = F.relu(self.input_fc(history))
        x = x + self._sinusoidal_positional_encoding(
            seq_len=x.size(1),
            hidden_dim=x.size(2),
            device=x.device,
            dtype=x.dtype
        )
        padding_mask = torch.all(torch.abs(history) < 1e-8, dim=-1)
        all_padding = padding_mask.all(dim=1)
        if all_padding.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_padding, -1] = False
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        h_ctx = x[:, -1, :]
        mu = self.z_mean(h_ctx)
        log_std = self.z_log_std(h_ctx)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        return mu, std

    @staticmethod
    def _sinusoidal_positional_encoding(seq_len, hidden_dim, device, dtype):
        position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / hidden_dim)
        )
        pe = torch.zeros(seq_len, hidden_dim, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        return pe.unsqueeze(0)

    def sample_z(self, mu, std):
        epsilon = torch.randn_like(std)
        return mu + epsilon * std


class SocialAttention(nn.Module):
    def __init__(self, feature_dim, hidden_dim=DEFAULT_SOCIAL_HIDDEN_DIM, num_heads=DEFAULT_NUM_HEADS):
        super(SocialAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.feature_encoder = nn.Linear(feature_dim, hidden_dim)
        self.global_query = nn.Parameter(torch.randn(1, num_heads, 1, self.head_dim))
        self.q_layer = nn.Linear(hidden_dim, hidden_dim)
        self.k_layer = nn.Linear(hidden_dim, hidden_dim)
        self.v_layer = nn.Linear(hidden_dim, hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.last_attention_weights = None

    def forward(self, inputs):
        batch_size = inputs.size(0)
        num_agents = inputs.size(1)
        u = F.relu(self.feature_encoder(inputs))
        group_summary = u.mean(dim=1)
        q = self.q_layer(group_summary).view(batch_size, self.num_heads, 1, self.head_dim)
        q = q + self.global_query.expand(batch_size, -1, -1, -1)
        k = self.k_layer(u).view(batch_size, num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_layer(u).view(batch_size, num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        self.last_attention_weights = attn_weights.detach()
        attended_context = torch.matmul(attn_weights, v).squeeze(2)
        mean_context = group_summary.view(batch_size, self.num_heads, self.head_dim)
        context = (attended_context + mean_context).contiguous().view(batch_size, -1)
        context = self.context_norm(context)
        return self.out_proj(context)


class MeanSocialPooling(nn.Module):
    def __init__(self, feature_dim, hidden_dim=DEFAULT_SOCIAL_HIDDEN_DIM, num_heads=DEFAULT_NUM_HEADS):
        super(MeanSocialPooling, self).__init__()
        self.feature_encoder = nn.Linear(feature_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.last_attention_weights = None

    def forward(self, inputs):
        batch_size, num_agents, _ = inputs.shape
        u = F.relu(self.feature_encoder(inputs))
        pooled = u.mean(dim=1)
        self.last_attention_weights = torch.full(
            (batch_size, 1, 1, num_agents),
            1.0 / num_agents,
            device=inputs.device,
            dtype=inputs.dtype
        )
        return self.out_proj(pooled)


class StandardAttentionPooling(nn.Module):
    def __init__(self, feature_dim, hidden_dim=DEFAULT_SOCIAL_HIDDEN_DIM, num_heads=DEFAULT_NUM_HEADS):
        super(StandardAttentionPooling, self).__init__()
        self.feature_encoder = nn.Linear(feature_dim, hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.last_attention_weights = None

    def forward(self, inputs):
        u = F.relu(self.feature_encoder(inputs))
        attn_out, attn_weights = self.self_attn(
            u,
            u,
            u,
            need_weights=True,
            average_attn_weights=False
        )
        self.last_attention_weights = attn_weights.detach()
        return self.out_proj(attn_out.mean(dim=1))


def build_social_module(
        feature_dim,
        hidden_dim=DEFAULT_SOCIAL_HIDDEN_DIM,
        num_heads=DEFAULT_NUM_HEADS,
        aggregation="social_attention"):
    if aggregation == "social_attention":
        return SocialAttention(feature_dim, hidden_dim, num_heads)
    if aggregation == "mean_pooling":
        return MeanSocialPooling(feature_dim, hidden_dim, num_heads)
    if aggregation == "standard_mha":
        return StandardAttentionPooling(feature_dim, hidden_dim, num_heads)
    raise ValueError(f"Unknown aggregation type: {aggregation}")
