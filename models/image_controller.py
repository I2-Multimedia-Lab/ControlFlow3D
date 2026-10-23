"""Image-conditioned flow controller (VGGT features)."""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.blocks import SharedPrototypeClusterer


class MultiViewImageFlowController(nn.Module):
    """Apply prototype-conditioned affine control to latent flow velocity.

    Shared image/target prototypes predict per-channel, per-point scale g and
    bias b, yielding v_ctrl = g * v_init + b. Hidden-channel scaling modulates
    prototype context. The point gate w is an auxiliary statistic and does not
    multiply the output velocity.
    """

    def __init__(
        self,
        flow_dim,
        image_dim=512,
        hidden_dim=256,
        time_dim=3,
        gate_bias=-1.0,
        max_tokens_per_view=32,
        num_prototypes=8,
        coarse_num_prototypes=4,
        prototype_dim=128,
        prototype_kmeans_iters=4,
        prototype_assign_temperature=6.0,
        use_query_tokens_as_infer_target=False,
    ):
        super().__init__()
        self.flow_dim = flow_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.gate_bias = gate_bias
        self.max_tokens_per_view = max(int(max_tokens_per_view), 0)
        self.num_prototypes = max(int(num_prototypes), 1)
        self.coarse_num_prototypes = max(int(coarse_num_prototypes), 1)
        self.prototype_assign_temperature = float(prototype_assign_temperature)
        self.use_query_tokens_as_infer_target = bool(use_query_tokens_as_infer_target)

        self.context_proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.token_proj = nn.Linear(image_dim, hidden_dim)
        self.time_query_proj = nn.Linear(time_dim, hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.local_query_proj = nn.Conv1d(flow_dim + 3 + time_dim, hidden_dim, 1)
        self.local_query_norm = nn.LayerNorm(hidden_dim)
        self.local_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
        )
        self.prototype_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
        )
        self.coarse_prototype_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
        )
        self.local_out_norm = nn.LayerNorm(hidden_dim)
        self.prototype_ctx_norm = nn.LayerNorm(hidden_dim)
        self.gb_proj = nn.Linear(hidden_dim + time_dim, flow_dim * 2)
        self.prototype_scale_proj = nn.Sequential(
            nn.Linear(hidden_dim + time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.prototype_clusterer = SharedPrototypeClusterer(
            image_dim=image_dim,
            query_dim=flow_dim + 3 + time_dim,
            target_dim=flow_dim + 3,
            hidden_dim=hidden_dim,
            prototype_dim=prototype_dim,
            num_prototypes=self.num_prototypes,
            kmeans_iters=prototype_kmeans_iters,
            assign_temperature=self.prototype_assign_temperature,
            enable_query_assign=False,
        )
        self.coarse_prototype_clusterer = SharedPrototypeClusterer(
            image_dim=image_dim,
            query_dim=flow_dim + 3 + time_dim,
            target_dim=flow_dim + 3,
            hidden_dim=hidden_dim,
            prototype_dim=prototype_dim,
            num_prototypes=self.coarse_num_prototypes,
            kmeans_iters=prototype_kmeans_iters,
            assign_temperature=self.prototype_assign_temperature,
            enable_query_assign=True,
        )
        self.semantic_scale_proj = nn.Sequential(
            nn.Conv1d(hidden_dim * 2 + time_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
        )
        self.point_w_proj = nn.Sequential(
            nn.Conv1d(hidden_dim * 2 + time_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 1, 1),
        )

    @staticmethod
    def _compute_prototype_consistency(image_prototypes_raw, target_prototypes):
        if image_prototypes_raw is None or target_prototypes is None:
            return None
        image_proto = F.normalize(image_prototypes_raw.float(), dim=-1, eps=1e-6)
        target_proto = F.normalize(target_prototypes.float(), dim=-1, eps=1e-6)
        return 1.0 - (image_proto * target_proto).sum(dim=-1).mean(dim=-1)

    def _flatten_image_tokens(self, image_cond):
        """Flatten image features to tokens [B, T_total, C_img].

        Accepts [B, C_img], [B, V, C_img] or [B, V, T, C_img].
        Multi-view tokens are flattened across view and token dimensions.
        """
        if image_cond.dim() == 2:
            return image_cond.unsqueeze(1)
        if image_cond.dim() == 3:
            return image_cond
        if image_cond.dim() == 4:
            if self.max_tokens_per_view > 0 and image_cond.shape[2] > self.max_tokens_per_view:
                raise RuntimeError(
                    'image_cond token count exceeds the controller limit.'
                    f' current tokens per view={image_cond.shape[2]},'
                    f' controller limit={self.max_tokens_per_view}.'
                    ' Please re-compress the image_cond cache with build_memmap_dataset_cache.py first,'
                    'rather than doing runtime pooling during training.'
                )
            return image_cond.reshape(image_cond.shape[0], -1, image_cond.shape[-1])
        raise RuntimeError(
            f'image_cond only supports [B,C] / [B,V,C] / [B,V,T,C], but got {tuple(image_cond.shape)}'
        )

    def forward(self, flow_pred, time_emb, query_xyz, image_cond=None, target_prototypes=None, return_aux=False):
        if image_cond is None:
            if return_aux:
                zero_gate = torch.zeros(
                    flow_pred.shape[0],
                    device=flow_pred.device,
                    dtype=flow_pred.dtype,
                )
                zero_scale = torch.zeros(
                    flow_pred.shape[0],
                    self.hidden_dim,
                    device=flow_pred.device,
                    dtype=flow_pred.dtype,
                )
                return flow_pred, {
                    'gate_mean': zero_gate,
                    'prototype_scale_mean': zero_scale,
                    'semantic_scale_mean': zero_scale,
                    'proto_consistency': zero_gate,
                    'fine_proto_consistency': zero_gate,
                    'coarse_proto_consistency': zero_gate,
                }
            return flow_pred

        img_tokens = self._flatten_image_tokens(image_cond)
        if img_tokens.shape[0] != flow_pred.shape[0]:
            raise RuntimeError(
                f'image_cond batch dimension mismatch: image={img_tokens.shape[0]}, flow={flow_pred.shape[0]}'
            )
        if img_tokens.shape[-1] != self.image_dim:
            raise RuntimeError(
                f'image_cond last dimension must be {self.image_dim}, but got {tuple(image_cond.shape)}'
            )

        # Pool image tokens with time-conditioned attention to form a global context.
        token_feat = self.token_norm(self.token_proj(img_tokens.float()))
        time_query = self.time_query_proj(time_emb.float()).unsqueeze(1)
        # Score tokens by their alignment with the time query.
        attn_logits = (token_feat * time_query).sum(dim=-1) / math.sqrt(float(self.hidden_dim))
        attn_weights = torch.softmax(attn_logits, dim=1)
        img_ctx = torch.sum(img_tokens.float() * attn_weights.unsqueeze(-1), dim=1)
        ctrl_ctx = self.context_proj(img_ctx)

        if query_xyz is None:
            raise RuntimeError('the image_cond path requires query_xyz; the None fallback branch is no longer supported.')
        if query_xyz.dim() != 3 or query_xyz.shape[1] != 3:
            raise RuntimeError(
                f'query_xyz shape must be [B,3,N], but got {tuple(query_xyz.shape)}'
            )
        n_query = query_xyz.shape[2]
        if n_query != flow_pred.shape[2]:
            raise RuntimeError(
                f'query_xyz point count does not match the flow features: xyz={n_query}, flow={flow_pred.shape[2]}'
            )
        time_token = time_emb.float().unsqueeze(-1).expand(-1, -1, n_query)
        query_cluster_tokens = torch.cat([flow_pred.float(), query_xyz.float(), time_token], dim=1)
        target_cluster_tokens = None
        if target_prototypes is not None:
            if target_prototypes.dim() != 3:
                raise RuntimeError(
                    f'target_prototypes shape must be [B,C,N], but got {tuple(target_prototypes.shape)}'
                )
            if target_prototypes.shape[0] != flow_pred.shape[0] or target_prototypes.shape[2] != n_query:
                raise RuntimeError(
                    'target_prototypes batch or point count does not match flow_pred/query_xyz: '
                    f'target={tuple(target_prototypes.shape)}, flow={tuple(flow_pred.shape)}'
                )
            target_cluster_tokens = torch.cat([target_prototypes.float(), query_xyz.float()], dim=1)
        elif self.use_query_tokens_as_infer_target:
            target_cluster_tokens = torch.cat([flow_pred.float(), query_xyz.float()], dim=1)

        # Cluster image and optional target tokens at fine and coarse scales.
        fine_pack = self.prototype_clusterer(
            image_tokens=img_tokens.float(),
            image_value_tokens=token_feat,
            query_tokens=query_cluster_tokens,
            target_tokens=target_cluster_tokens,
        )
        coarse_pack = self.coarse_prototype_clusterer(
            image_tokens=img_tokens.float(),
            image_value_tokens=token_feat,
            query_tokens=query_cluster_tokens,
            target_tokens=target_cluster_tokens,
        )
        # Prototype values: [B, K, H] at each scale.
        fine_image_prototypes = fine_pack['image_prototypes']
        coarse_image_prototypes = coarse_pack['image_prototypes']
        # Coarse query assignments: [B, K_coarse, N].
        coarse_query_proto_weights = coarse_pack['query_assign']

        # Compare image and target prototype directions at each scale.
        fine_proto_consistency = self._compute_prototype_consistency(
            fine_pack.get('image_prototypes_raw'),
            fine_pack.get('target_prototypes'),
        )
        coarse_proto_consistency = self._compute_prototype_consistency(
            coarse_pack.get('image_prototypes_raw'),
            coarse_pack.get('target_prototypes'),
        )
        # Average consistency terms for scales with target prototypes.
        proto_consistency_terms = [
            term for term in (fine_proto_consistency, coarse_proto_consistency)
            if term is not None
        ]
        if proto_consistency_terms:
            proto_consistency = torch.stack(proto_consistency_terms, dim=0).mean(dim=0)
        else:
            # Use zero consistency loss when target prototypes are absent.
            proto_consistency = torch.zeros(
                flow_pred.shape[0],
                device=flow_pred.device,
                dtype=flow_pred.dtype,
            )

        # Fuse raw image tokens and prototype context through local cross-attention.
        local_query = query_cluster_tokens
        local_query = self.local_query_proj(local_query).transpose(1, 2).contiguous()
        local_query = self.local_query_norm(local_query)
        raw_local_ctx, _ = self.local_cross_attn(local_query, token_feat, token_feat)
        fine_proto_local_ctx, _ = self.prototype_cross_attn(
            local_query,
            fine_image_prototypes,
            fine_image_prototypes,
        )
        coarse_proto_local_ctx, _ = self.coarse_prototype_cross_attn(
            local_query,
            coarse_image_prototypes,
            coarse_image_prototypes,
        )
        proto_local_ctx = 0.65 * fine_proto_local_ctx + 0.35 * coarse_proto_local_ctx
        local_ctx = self.local_out_norm(local_query + 0.5 * (raw_local_ctx + proto_local_ctx))
        local_ctx_cf = local_ctx.transpose(1, 2).contiguous()
        raw_local_ctx_cf = raw_local_ctx.transpose(1, 2).contiguous()
        proto_local_ctx_cf = proto_local_ctx.transpose(1, 2).contiguous()

        # Map coarse prototype controls to points using their soft assignments.
        proto_ctx = self.prototype_ctx_norm(coarse_image_prototypes + ctrl_ctx.unsqueeze(1))
        proto_time = time_emb.float().unsqueeze(1).expand(-1, self.coarse_num_prototypes, -1)
        proto_scale = 1.0 + 0.5 * torch.tanh(self.prototype_scale_proj(torch.cat([proto_ctx, proto_time], dim=-1)))
        proto_ctx_aniso = proto_ctx * proto_scale
        gb = self.gb_proj(torch.cat([proto_ctx_aniso, proto_time], dim=-1))
        gb = gb.view(flow_pred.shape[0], self.coarse_num_prototypes, 2, self.flow_dim)
        g_proto = 1.0 + 0.35 * torch.tanh(gb[:, :, 0, :])
        b_proto = 0.35 * torch.tanh(gb[:, :, 1, :])
        g = torch.einsum('bkc,bkn->bcn', g_proto, coarse_query_proto_weights)
        b = torch.einsum('bkc,bkn->bcn', b_proto, coarse_query_proto_weights)

        # Scale prototype context along hidden feature channels.
        semantic_in = torch.cat([raw_local_ctx_cf, proto_local_ctx_cf, time_token], dim=1)
        semantic_scale = 1.0 + 0.5 * torch.tanh(self.semantic_scale_proj(semantic_in))
        proto_local_ctx_aniso = proto_local_ctx_cf * semantic_scale

        # Track per-point gate strength as an auxiliary statistic.
        gate_in = torch.cat([local_ctx_cf, proto_local_ctx_aniso, time_token], dim=1)
        w = torch.sigmoid(self.point_w_proj(gate_in) + self.gate_bias)

        g = g.to(dtype=flow_pred.dtype)
        b = b.to(dtype=flow_pred.dtype)
        w = w.to(dtype=flow_pred.dtype)

        ctrl_flow = g * flow_pred + b
        if return_aux:
            return ctrl_flow, {
                'gate_mean': w.mean(dim=[1, 2]),
                'prototype_scale_mean': proto_scale.mean(dim=1),
                'semantic_scale_mean': semantic_scale.mean(dim=2),
                'proto_consistency': proto_consistency.to(dtype=flow_pred.dtype),
                'fine_proto_consistency': (
                    fine_proto_consistency.to(dtype=flow_pred.dtype)
                    if fine_proto_consistency is not None else torch.zeros_like(proto_consistency)
                ),
                'coarse_proto_consistency': (
                    coarse_proto_consistency.to(dtype=flow_pred.dtype)
                    if coarse_proto_consistency is not None else torch.zeros_like(proto_consistency)
                ),
            }
        return ctrl_flow
