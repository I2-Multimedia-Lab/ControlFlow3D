"""Conditioning, cross-attention and shared-prototype building blocks."""

import torch
import torch.nn as nn
from torch.nn import functional as F


class ConditionalAffineConv1d(nn.Module):
    """Apply a context-conditioned affine transform to a 1x1 convolution.

    The context predicts a channel gate and bias:
        y = Conv1d(x) * sigmoid(gate) + bias
    Optional GroupNorm follows the affine transform.
    """
    def __init__(self, dim_in, dim_out, dim_ctx, norm=False, cond=True):
        super(ConditionalAffineConv1d, self).__init__()
        self._layer = nn.Conv1d(dim_in, dim_out, 1, 1, 0)
        self.cond = cond
        if cond is True:
            self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
            self._hyper_gate = nn.Linear(dim_ctx, dim_out)
        self.norm = norm
        if norm is True:
            self.bn = nn.GroupNorm(32, dim_out)

    def forward(self, ctx, x):
        if self.cond is True:
            # ctx: [B, C_ctx] -> gate/bias: [B, C_out]
            gate = torch.sigmoid(self._hyper_gate(ctx))
            bias = self._hyper_bias(ctx)
            if x.dim() == 3:
                # Broadcast channel modulation over the point dimension.
                gate = gate.unsqueeze(-1)
                bias = bias.unsqueeze(-1)
            if self.norm is True:
                ret = self.bn(self._layer(x) * gate + bias)
            else:
                ret = self._layer(x) * gate + bias
        else:
            # Apply an unconditional 1x1 convolution with optional normalization.
            if self.norm is True:
                ret = self.bn(self._layer(x))
            else:
                ret = self._layer(x)
        return ret


class FlowMatchingCrossBlock(nn.Module):
    """Inject conditions through residual cross-attention and a feed-forward layer."""

    def __init__(self, feat_dim, cond_dim, time_dim):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=4,
            kdim=cond_dim,
            vdim=cond_dim,
            batch_first=True
        )
        self.ca_norm_q = nn.LayerNorm(feat_dim)
        self.ca_norm_out = nn.LayerNorm(feat_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 4),
            nn.GELU(),
            nn.Linear(feat_dim * 4, feat_dim),
        )
        self.ffn_norm = nn.LayerNorm(feat_dim)
        self.cond_ca = ConditionalAffineConv1d(feat_dim, feat_dim, time_dim, norm=False)
        self.cond_ffn = ConditionalAffineConv1d(feat_dim, feat_dim, time_dim, norm=False)

    def forward(self, x, time_emb, cond_tokens):
        q = x.transpose(1, 2).contiguous()  # [B, N, C]
        q_normed = self.ca_norm_q(q)
        attn_out, _ = self.cross_attn(q_normed, cond_tokens, cond_tokens)
        q = q + attn_out
        q = self.ca_norm_out(q)
        feat_ca = q.transpose(1, 2).contiguous()  # [B, C, N]
        feat_ca = self.cond_ca(time_emb, feat_ca)

        q = feat_ca.transpose(1, 2).contiguous()
        q = q + self.ffn(self.ffn_norm(q))
        feat_out = q.transpose(1, 2).contiguous()
        feat_out = self.cond_ffn(time_emb, feat_out)
        return feat_out


class SharedPrototypeClusterer(nn.Module):
    """Cluster image and optional target tokens into shared semantic prototypes.

    Image and target tokens share cluster centers when targets are provided.
    Query tokens receive soft assignments to those centers. Without targets,
    prototypes are constructed from image tokens only.
    """

    def __init__(
        self,
        image_dim,
        query_dim,
        target_dim,
        hidden_dim,
        prototype_dim=128,
        num_prototypes=8,
        kmeans_iters=4,
        assign_temperature=6.0,
        enable_query_assign=True,
    ):
        super().__init__()
        self.prototype_dim = int(prototype_dim)
        self.num_prototypes = max(int(num_prototypes), 1)
        self.kmeans_iters = max(int(kmeans_iters), 1)
        self.assign_temperature = float(assign_temperature)
        self.enable_query_assign = bool(enable_query_assign)

        self.prototype_seeds = nn.Parameter(torch.randn(self.num_prototypes, self.prototype_dim))
        self.image_cluster_proj = nn.Linear(image_dim, self.prototype_dim)
        self.query_cluster_proj = (
            nn.Conv1d(query_dim, self.prototype_dim, 1)
            if self.enable_query_assign else None
        )
        self.target_cluster_proj = nn.Conv1d(target_dim, self.prototype_dim, 1)
        self.target_value_proj = nn.Conv1d(target_dim, hidden_dim, 1)
        self.prototype_value_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _normalize(tokens):
        return F.normalize(tokens, dim=-1, eps=1e-6)

    def _soft_assign(self, tokens, centers):
        logits = self.assign_temperature * torch.matmul(tokens, centers.transpose(1, 2))
        return torch.softmax(logits, dim=-1)

    @staticmethod
    def _aggregate(assign, values):
        proto = torch.einsum('btk,btd->bkd', assign, values)
        denom = assign.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)
        return proto / denom

    def _run_soft_kmeans(self, tokens):
        centers = self._normalize(self.prototype_seeds).unsqueeze(0).expand(tokens.shape[0], -1, -1)
        for _ in range(self.kmeans_iters):
            assign = self._soft_assign(tokens, centers)
            centers = self._normalize(self._aggregate(assign, tokens))
        return centers

    def forward(self, image_tokens, image_value_tokens, query_tokens, target_tokens=None):
        image_cluster = self._normalize(self.image_cluster_proj(image_tokens.float()))
        query_assign = None
        if self.query_cluster_proj is not None:
            query_cluster = self._normalize(
                self.query_cluster_proj(query_tokens.float()).transpose(1, 2).contiguous()
            )

        target_cluster = None
        cluster_source = image_cluster
        if target_tokens is not None:
            target_cluster = self._normalize(
                self.target_cluster_proj(target_tokens.float()).transpose(1, 2).contiguous()
            )
            cluster_source = torch.cat([image_cluster, target_cluster], dim=1)

        centers = self._run_soft_kmeans(cluster_source)
        img_assign = self._soft_assign(image_cluster, centers)
        if self.query_cluster_proj is not None:
            query_assign = self._soft_assign(query_cluster, centers)
        image_prototypes_raw = self._aggregate(img_assign, image_value_tokens.float())
        prototype_usage = img_assign.mean(dim=1)
        image_prototypes = image_prototypes_raw

        target_prototypes = None
        if target_tokens is not None:
            target_assign = self._soft_assign(target_cluster, centers)
            target_values = self.target_value_proj(target_tokens.float()).transpose(1, 2).contiguous()
            target_prototypes = self._aggregate(target_assign, target_values)
            prototype_usage = 0.5 * (prototype_usage + target_assign.mean(dim=1))
            image_prototypes = 0.5 * (image_prototypes_raw + target_prototypes)

        image_prototypes = self.prototype_value_norm(image_prototypes)
        return {
            'image_prototypes': image_prototypes,
            'query_assign': (
                query_assign.transpose(1, 2).contiguous()
                if query_assign is not None else None
            ),
            'image_prototypes_raw': image_prototypes_raw,
            'target_prototypes': target_prototypes,
            'prototype_usage': prototype_usage,
        }
