"""Conditional flow-matching velocity network with prototype control."""

import torch
import torch.nn as nn

from models.blocks import ConditionalAffineConv1d, FlowMatchingCrossBlock
from models.image_controller import MultiViewImageFlowController


class ConditionalFlowMatchingNetwork(nn.Module):
    """Predict latent feature velocity with conditional cross-attention.

    By default, four blocks inject query coordinates, local conditions and
    broadcast global conditions. FiLM layers provide time conditioning;
    the optional image controller modulates the predicted velocity.
    """

    def __init__(
        self,
        feat_dim=128,
        cond_dim=515,
        time_dim=None,
        num_layers=4,
        image_cond_dim=512,
        enable_image_controller=True,
        image_cond_max_tokens_per_view=32,
        n_parts=8,
        num_prototypes=8,
        coarse_num_prototypes=4,
        prototype_dim=128,
        prototype_kmeans_iters=4,
        prototype_assign_temperature=6.0,
        use_query_tokens_as_infer_target=False,
    ):
        super(ConditionalFlowMatchingNetwork, self).__init__()
        self.feat_dim = feat_dim
        self.cond_dim = cond_dim
        self.time_dim = 3 if time_dim is None else time_dim
        self.num_layers = num_layers
        self.enable_image_controller = bool(enable_image_controller)
        self.n_parts = max(int(n_parts), 1)
        self.num_prototypes = max(int(num_prototypes), 1)
        self.coarse_num_prototypes = max(int(coarse_num_prototypes), 1)
        self.prototype_assign_temperature = float(prototype_assign_temperature)
        self.use_query_tokens_as_infer_target = bool(use_query_tokens_as_infer_target)

        self.cond_in = ConditionalAffineConv1d(feat_dim, feat_dim, self.time_dim, norm=False)
        self.blocks = nn.ModuleList([
            FlowMatchingCrossBlock(feat_dim, cond_dim, time_dim=self.time_dim)
            for _ in range(self.num_layers)
        ])
        self.out_proj = nn.Conv1d(feat_dim, feat_dim, 1)
        self.image_controller = (
            MultiViewImageFlowController(
                flow_dim=feat_dim,
                image_dim=image_cond_dim,
                hidden_dim=max(feat_dim * 2, 256),
                time_dim=self.time_dim,
                max_tokens_per_view=image_cond_max_tokens_per_view,
                num_prototypes=self.num_prototypes,
                coarse_num_prototypes=self.coarse_num_prototypes,
                prototype_dim=prototype_dim,
                prototype_kmeans_iters=prototype_kmeans_iters,
                prototype_assign_temperature=self.prototype_assign_temperature,
                use_query_tokens_as_infer_target=self.use_query_tokens_as_infer_target,
            )
            if self.enable_image_controller else None
        )

    def forward(
        self,
        x_t,
        t,
        global_cond=None,
        local_cond=None,
        query_xyz=None,
        cond_tokens=None,
        image_cond=None,
        target_prototypes=None,
        return_aux=False,
    ):
        """Predict velocity for the current flow state.

        Args:
            x_t: Noisy latent features [B, C_flow, N_query].
            t: Flow times [B].
            global_cond: Global conditions [B, C_global].
            local_cond: Query-aligned local conditions [B, C_local, N_query].
            query_xyz: Query coordinates [B, 3, N_query].
            cond_tokens: Prebuilt conditions [B, C_cond, N_query] or
                [B, N_query, C_cond]; replaces global/local condition assembly.
            image_cond: Optional image features for the flow controller.
            target_prototypes: Optional target features [B, C_flow, N_query].
            return_aux: Whether to return controller statistics.

        Returns:
            Velocity [B, C_flow, N_query], optionally paired with auxiliary statistics.
        """
        t = t.unsqueeze(-1).float()  # [B, 1]
        time_emb = torch.cat([t, torch.sin(t), torch.cos(t)], dim=-1)  # [B, 3]

        if cond_tokens is None:
            if global_cond is None or local_cond is None:
                raise RuntimeError('condition_flow_matching requires global_cond + local_cond, or cond_tokens provided directly.')
            if local_cond.dim() != 3:
                raise RuntimeError(f'local_cond shape must be [B, 256, N_query(=256)], but got {tuple(local_cond.shape)}')
            n_query = local_cond.shape[2]
            if query_xyz is None:
                query_xyz = torch.zeros(
                    x_t.shape[0], 3, n_query,
                    device=x_t.device,
                    dtype=x_t.dtype
                )
            if query_xyz.dim() != 3 or query_xyz.shape[1] != 3 or query_xyz.shape[2] != n_query:
                raise RuntimeError(
                    f'query_xyz shape must be [B, 3, {n_query}], but got {tuple(query_xyz.shape)}'
                )
            global_tokens = global_cond.unsqueeze(-1).expand(-1, -1, n_query)
            cond_tokens = torch.cat(
                [
                    query_xyz.float(),
                    local_cond.float(),
                    global_tokens.float(),
                ],
                dim=1
            )

        if cond_tokens.dim() != 3:
            raise RuntimeError(f'cond_tokens must be a 3D tensor, but got {tuple(cond_tokens.shape)}')
        if cond_tokens.shape[1] == self.cond_dim:
            cond_tokens = cond_tokens.transpose(1, 2).contiguous()
        elif cond_tokens.shape[2] != self.cond_dim:
            raise RuntimeError(
                f'cond_tokens last dimension must be {self.cond_dim}, but got {tuple(cond_tokens.shape)}'
            )

        feat = self.cond_in(time_emb, x_t)
        for block in self.blocks:
            feat = block(feat, time_emb, cond_tokens)

        flow_pred = self.out_proj(feat)
        zero_gate = torch.zeros(
            flow_pred.shape[0],
            device=flow_pred.device,
            dtype=flow_pred.dtype,
        )
        aux = {'gate_mean': zero_gate}
        if self.image_controller is not None and image_cond is not None:
            if return_aux:
                flow_pred, ctrl_aux = self.image_controller(
                    flow_pred,
                    time_emb,
                    query_xyz=query_xyz,
                    image_cond=image_cond,
                    target_prototypes=target_prototypes,
                    return_aux=True,
                )
                aux.update(ctrl_aux)
            else:
                flow_pred = self.image_controller(
                    flow_pred,
                    time_emb,
                    query_xyz=query_xyz,
                    image_cond=image_cond,
                    target_prototypes=target_prototypes,
                    return_aux=False,
                )

        if return_aux:
            return flow_pred, aux
        return flow_pred
