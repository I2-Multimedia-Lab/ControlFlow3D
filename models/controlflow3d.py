"""Latent flow matching, geometry decoding and training objectives."""

import os
import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from models.condition import PointNetMSGConditionExtractor
from models.flow_matching import ConditionalFlowMatchingNetwork
from models.mamba_decoder import MambaGeometryDecoder
from einops import rearrange
from models.utils import FPS, FPS_with_idx, fps_indices, index_points
from extensions.chamfer_dist import chamfer_3DDist
from extensions.Pointnet2.pointnet2.pointnet2_utils import three_nn, three_interpolate

class ControlFlow3D(nn.Module):
    _printed_refiner_routes_global = set()
    _printed_infer_image_ignored = False

    def __init__(
        self,
        n_parts=8,
        flow_feature_dim=128,
        condition_feature_dim=256,
        image_condition_dim=512,
        use_image_controller=True,
        image_cond_max_tokens_per_view=32,
        num_prototypes=8,
        coarse_num_prototypes=4,
        prototype_dim=128,
        prototype_kmeans_iters=4,
        prototype_assign_temperature=6.0,
        fm_ctrl_weight=1.0,
        prototype_consistency_weight=0.05,
        local_loss_weight=0.25,
        local_loss_final_stage_weight=0.0,
        use_query_tokens_as_infer_target=False,
    ):
        super().__init__()
        self.n_parts = n_parts
        self.use_partwise_inference = True
        self.decode_aux_weight = 0.0
        self.flow_token_count = 256
        self.flow_feature_dim = flow_feature_dim
        self.condition_feature_dim = condition_feature_dim
        self.image_condition_dim = image_condition_dim
        self.image_cond_max_tokens_per_view = max(int(image_cond_max_tokens_per_view), 0)
        self.use_image_controller = bool(use_image_controller)
        self.num_prototypes = max(int(num_prototypes), 1)
        self.coarse_num_prototypes = max(int(coarse_num_prototypes), 1)
        self.prototype_dim = int(prototype_dim)
        self.prototype_kmeans_iters = max(int(prototype_kmeans_iters), 1)
        self.prototype_assign_temperature = float(prototype_assign_temperature)
        self.use_query_tokens_as_infer_target = bool(use_query_tokens_as_infer_target)
        self.x0_noise_std = 0.01
        self.fm_ctrl_weight = float(fm_ctrl_weight)
        self.prototype_consistency_weight = float(prototype_consistency_weight)
        self.local_loss_weight = float(local_loss_weight)
        self.local_loss_final_stage_weight = float(local_loss_final_stage_weight)
        self.inference_refiner_index_mode = 'part'

        # Extract global and local conditions from coordinates with MSG SA/FP.
        # Use external sparse features only to initialize flow tokens.
        self.condition_extractor = PointNetMSGConditionExtractor(out_dim=condition_feature_dim)
        self.condition_flow_matching = ConditionalFlowMatchingNetwork(
            feat_dim=flow_feature_dim,
            cond_dim=3 + condition_feature_dim + condition_feature_dim,
            num_layers=4,
            image_cond_dim=image_condition_dim,
            enable_image_controller=self.use_image_controller,
            image_cond_max_tokens_per_view=self.image_cond_max_tokens_per_view,
            n_parts=self.n_parts,
            num_prototypes=self.num_prototypes,
            coarse_num_prototypes=self.coarse_num_prototypes,
            prototype_dim=self.prototype_dim,
            prototype_kmeans_iters=self.prototype_kmeans_iters,
            prototype_assign_temperature=self.prototype_assign_temperature,
            use_query_tokens_as_infer_target=self.use_query_tokens_as_infer_target,
        )

        pred_hidden_dim = max(flow_feature_dim // 2, 48)
        self.pred_x1_mlp = nn.Sequential(
            nn.Conv1d(flow_feature_dim, pred_hidden_dim, 1),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(pred_hidden_dim, flow_feature_dim, 1),
            nn.BatchNorm1d(flow_feature_dim),
            nn.ReLU(inplace=True),
        )

        self.refiner = MambaGeometryDecoder(
            up_factors=(2, 2),
            query_local_dim=condition_feature_dim,
            cond_flow_dim=flow_feature_dim,
        )
        self.chamfer_loss = chamfer_3DDist()

    def encode_category_ids(self, categories, device=None):
        return None

    @staticmethod
    def _three_interp(dst_xyz, src_xyz, src_feat):
        """Interpolate source features at target coordinates with three neighbors.

        Args:
            dst_xyz: Target coordinates [B, 3, N_dst].
            src_xyz: Source coordinates [B, 3, N_src].
            src_feat: Source features [B, C, N_src].

        Returns:
            Interpolated features [B, C, N_dst].
        """
        dst_xyz  = dst_xyz.float()
        src_xyz  = src_xyz.float()
        src_feat = src_feat.float()
        q = dst_xyz.transpose(1, 2).contiguous()
        s = src_xyz.transpose(1, 2).contiguous()
        dist, idx = three_nn(q, s)
        dist_recip = 1.0 / (dist + 1e-8)
        weight = dist_recip / dist_recip.sum(dim=2, keepdim=True)
        return three_interpolate(src_feat.contiguous(), idx, weight)

    def _compute_local_cd(self, pred_pts: torch.Tensor, gt_xyz: torch.Tensor):
        """Compute per-sample Chamfer loss over contiguous parts.

        Split predictions and GT into equal-count parts, then evaluate all parts
        in one batched Chamfer call. Returns one loss per sample [B].
        """
        N_pred = pred_pts.shape[1]
        N_gt = gt_xyz.shape[1]
        if N_pred % self.n_parts != 0:
            raise RuntimeError(f"pred point count {N_pred} is not divisible by the number of parts {self.n_parts}.")
        if N_gt % self.n_parts != 0:
            raise RuntimeError(f"gt point count {N_gt} is not divisible by the number of parts {self.n_parts}.")
        B = pred_pts.shape[0]
        part_pred_n = N_pred // self.n_parts
        part_gt_n = N_gt // self.n_parts
        pred_part = pred_pts.view(B, self.n_parts, part_pred_n, 3).reshape(B * self.n_parts, part_pred_n, 3)
        gt_part = gt_xyz.view(B, self.n_parts, part_gt_n, 3).reshape(B * self.n_parts, part_gt_n, 3)
        d1_p, d2_p, _, _ = self.chamfer_loss(pred_part.float(), gt_part.float())
        part_loss = torch.mean(d1_p, dim=1) + torch.mean(d2_p, dim=1)
        return part_loss.view(B, self.n_parts).mean(dim=1)

    def _compute_patch_loss(self, pred_pts: torch.Tensor, gt_xyz: torch.Tensor, gt_patch_idx: torch.Tensor):
        """Compute per-sample Chamfer loss over aligned patches.

        Predicted patches occupy consecutive equal-length blocks. GT patches are
        gathered with gt_patch_idx. Returns one loss per sample [B].
        """
        if gt_patch_idx.dim() != 3:
            raise RuntimeError(
                f"gt_patch_idx shape must be [B, n_parts, K], but got {tuple(gt_patch_idx.shape)}"
            )
        B = pred_pts.shape[0]
        if gt_patch_idx.shape[0] != B:
            raise RuntimeError(
                f"gt_patch_idx batch mismatch: idx={gt_patch_idx.shape[0]}, B={B}"
            )
        if int(gt_patch_idx.shape[1]) != int(self.n_parts):
            raise RuntimeError(
                f"gt_patch_idx n_parts={gt_patch_idx.shape[1]} is inconsistent with the model config n_parts={self.n_parts}."
            )
        N_pred = pred_pts.shape[1]
        if N_pred % self.n_parts != 0:
            raise RuntimeError(f"pred point count {N_pred} is not divisible by the number of patches {self.n_parts}.")
        pred_part_n = N_pred // self.n_parts
        pred_part = pred_pts.view(B, self.n_parts, pred_part_n, 3).reshape(B * self.n_parts, pred_part_n, 3)

        idx_flat = gt_patch_idx.reshape(B, -1)
        gt_part = self._gather_bnc(gt_xyz, idx_flat)
        gt_part = gt_part.view(B, self.n_parts, -1, 3).reshape(B * self.n_parts, -1, 3)

        d1_p, d2_p, _, _ = self.chamfer_loss(pred_part.float(), gt_part.float())
        part_loss = torch.mean(d1_p, dim=1) + torch.mean(d2_p, dim=1)
        return part_loss.view(B, self.n_parts).mean(dim=1)

    def _compute_local_cd_patch(self, pred_pts: torch.Tensor, gt_xyz: torch.Tensor, gt_patch_idx: torch.Tensor):
        # Backward-compatible alias kept for existing call sites.
        return self._compute_patch_loss(pred_pts, gt_xyz, gt_patch_idx)

    def _log_refiner_route_once(self, route_mode: str, phase: str, device: torch.device):
        """Log the part, patch or fallback decoder route once per phase.

        Only the main rank emits the message.
        """
        rank = int(os.environ.get('RANK', '0'))
        if rank != 0:
            return
        route = str(route_mode or "global").strip().lower()
        key = f"{phase}:{route}"
        if key in ControlFlow3D._printed_refiner_routes_global:
            return
        if route == "part":
            mode = "PART-WISE"
        elif route == "patch":
            mode = "PATCH-WISE"
        else:
            mode = "GLOBAL-FALLBACK"
        print(f"[ControlFlow3D] Refiner route ({phase}): {mode}")
        ControlFlow3D._printed_refiner_routes_global.add(key)

    @staticmethod
    def _gather_bnc(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Gather point-first features [B, N, C] with indices [B, S]."""
        if points.dim() != 3 or idx.dim() != 2:
            raise RuntimeError(
                f"_gather_bnc expects points=[B,N,C], idx=[B,S], got {tuple(points.shape)} and {tuple(idx.shape)}"
            )
        idx_expand = idx.unsqueeze(-1).expand(-1, -1, points.shape[-1])
        return torch.gather(points, dim=1, index=idx_expand)

    def _prepare_gt_flow_targets(self, gt_feat, gt_xyz):
        """Sample GT features and coordinates at the configured flow token count.

        Apply the same FPS indices to features and coordinates to preserve their
        pointwise correspondence.
        """
        target_n = self.flow_token_count
        if gt_feat.dim() != 3 or gt_feat.shape[-1] != self.flow_feature_dim:
            raise RuntimeError(
                f"GT feature shape must be [B, N, {self.flow_feature_dim}], but got {tuple(gt_feat.shape)}"
            )
        if gt_feat.shape[1] != gt_xyz.shape[1]:
            raise RuntimeError(
                f"GT feat/xyz point count mismatch: feat={gt_feat.shape[1]}, xyz={gt_xyz.shape[1]}"
            )
        if gt_xyz.shape[1] < target_n:
            raise RuntimeError(
                f"GT point count too small: got {gt_xyz.shape[1]}, expected at least {target_n}."
            )

        # Convert to channel-first coordinates for FPS.
        sample_idx = fps_indices(gt_xyz.transpose(1, 2).contiguous(), target_n)
        # Use the same indices to keep sampled features and coordinates aligned.
        gt_feat = self._gather_bnc(gt_feat, sample_idx)
        gt_query_xyz = self._gather_bnc(gt_xyz, sample_idx)
        return gt_feat, gt_query_xyz

    def _align_gt_to_query(self, gt_feat: torch.Tensor, gt_xyz: torch.Tensor, query_xyz: torch.Tensor):
        """Interpolate dense GT features onto sparse query coordinates.

        Args:
            gt_feat: Dense target features [B, N_gt, C_flow].
            gt_xyz: Dense target coordinates [B, N_gt, 3].
            query_xyz: Sparse query coordinates [B, 3, N_query].

        Returns:
            Query-aligned target features [B, C_flow, N_query].
        """
        if gt_feat.dim() != 3 or gt_feat.shape[-1] != self.flow_feature_dim:
            raise RuntimeError(
                f"GT feature shape must be [B, N, {self.flow_feature_dim}], but got {tuple(gt_feat.shape)}"
            )
        if gt_xyz.dim() != 3 or gt_xyz.shape[-1] != 3:
            raise RuntimeError(
                f"GT coordinate shape must be [B, N, 3], but got {tuple(gt_xyz.shape)}"
            )
        if gt_feat.shape[:2] != gt_xyz.shape[:2]:
            raise RuntimeError(
                f"GT feat/xyz point count mismatch: feat={tuple(gt_feat.shape)}, xyz={tuple(gt_xyz.shape)}"
            )
        if query_xyz.dim() != 3 or query_xyz.shape[1] != 3:
            raise RuntimeError(
                f"query_xyz shape must be [B, 3, N_query], but got {tuple(query_xyz.shape)}"
            )

        gt_feat_cf = gt_feat.transpose(1, 2).contiguous()
        gt_xyz_cf = gt_xyz.transpose(1, 2).contiguous()
        return self._three_interp(query_xyz, gt_xyz_cf, gt_feat_cf)

    def _extract_condition(self, sparse_in, sparse_feat):
        """Extract coordinate conditions and sample flow tokens with FPS.

        Args:
            sparse_in: Sparse coordinates [B, 3, N_sparse].
            sparse_feat: External point features [B, N_sparse, C_flow].

        Returns:
            global_cond: Global conditions [B, C_cond].
            local_cond: Query conditions [B, C_cond, N_query].
            local_cond_ref: Decoder conditions [B, C_cond, N_sparse].
            token_xyz: Flow query coordinates [B, 3, N_query].
            token_feat: External features for x_0 [B, C_flow, N_query].
        """
        if sparse_feat.dim() != 3:
            raise RuntimeError(
                f"sparse_feat shape must be [B, N_sparse, {self.flow_feature_dim}], but got {tuple(sparse_feat.shape)}"
            )
        if sparse_feat.shape[1] != sparse_in.shape[2] or sparse_feat.shape[2] != self.flow_feature_dim:
            raise RuntimeError(
                f"sparse_feat must be one-to-one aligned with the sparse points, expected [B, {sparse_in.shape[2]}, {self.flow_feature_dim}], "
                f"but got {tuple(sparse_feat.shape)}"
            )

        # Extract global [B, C_cond] and local [B, C_cond, N_sparse] conditions.
        global_cond, local_cond_ref = self.condition_extractor(
            sparse_in,
            return_pointwise=True,
        )

        # Select query coordinates with FPS on the sparse input.
        token_xyz, token_idx = FPS_with_idx(sparse_in, self.flow_token_count)
        # Gather query-aligned local conditions: [B, C_cond, N_query].
        local_cond = index_points(local_cond_ref, token_idx).contiguous()
        # Convert external features to channel-first layout for token gathering.
        sparse_feat_cf = sparse_feat.transpose(1, 2).contiguous()  # [B, 128, N_sparse]
        # Gather external features for x_0: [B, C_flow, N_query].
        token_feat = index_points(sparse_feat_cf, token_idx).contiguous()

        return global_cond, local_cond, local_cond_ref, token_xyz, token_feat

    def _build_x0(self, token_xyz, token_feat, target_xyz, add_noise=True, noise=None):
        """Initialize flow features and interpolate them to target coordinates.

        Args:
            token_xyz: Flow query coordinates [B, 3, N_query].
            token_feat: External query features [B, C_flow, N_query].
            target_xyz: Target coordinates [B, 3, N_target].
            add_noise: Whether to add Gaussian noise before interpolation.
            noise: Optional noise [B, C_flow, N_query]; sampled when needed if absent.

        Returns:
            Initial flow state [B, C_flow, N_target].
        """
        if noise is not None:
            if noise.shape != token_feat.shape:
                raise RuntimeError(
                    f'x0 noise shape must match token_feat: got {tuple(noise.shape)}, expected {tuple(token_feat.shape)}'
                )
            src_feat = token_feat + self.x0_noise_std * noise
        elif add_noise:
            src_feat = token_feat + self.x0_noise_std * torch.randn_like(token_feat)
        else:
            src_feat = token_feat

        return self._three_interp(target_xyz, token_xyz, src_feat)

    def _partwise_refine(
        self,
        sparse_in,
        cond_flow_query,
        cond_sparse_feat,
        patch_indices=None,
        normalize_patches=False,
    ):
        """Decode dense coordinates independently for each part or indexed patch.

        Each group receives sparse local conditions and shared flow query features.
        The decoder returns a 2x first stage and a 4x final prediction.

        Args:
            sparse_in: Sparse coordinates [B, 3, N_sparse].
            cond_flow_query: Flow query features [B, C_flow, N_query].
            cond_sparse_feat: Sparse local conditions [B, C_cond, N_sparse].
            patch_indices: Optional groups [B, P, K]; otherwise use contiguous parts.
            normalize_patches: Whether to center and scale each indexed patch.

        Returns:
            Stage predictions, each with shape [B, N_stage, 3].
        """
        B = sparse_in.shape[0]
        N_sparse = sparse_in.shape[2]
        actual_parts = self.n_parts
        patch_centroid = None
        patch_scale = None
        if patch_indices is not None:
            if patch_indices.dim() != 3:
                raise RuntimeError(
                    f"patch_indices shape must be [B, n_parts, K], but got {tuple(patch_indices.shape)}"
                )
            if patch_indices.shape[0] != B:
                raise RuntimeError(
                    f"patch_indices batch mismatch: idx={patch_indices.shape[0]}, B={B}"
                )
            n_parts = patch_indices.shape[1]
            patch_size = patch_indices.shape[2]
            actual_parts = int(n_parts)
            patch_indices = patch_indices.long()
            if patch_indices.device != sparse_in.device:
                patch_indices = patch_indices.to(device=sparse_in.device)
            idx_flat = patch_indices.reshape(B, -1)
            idx_expand_xyz = idx_flat.unsqueeze(1).expand(-1, 3, -1)
            sparse_part_flat = torch.gather(sparse_in, 2, idx_expand_xyz)
            sparse_part_flat = sparse_part_flat.view(B, 3, n_parts, patch_size).permute(0, 2, 1, 3).contiguous()
            sparse_part_flat = sparse_part_flat.reshape(B * n_parts, 3, patch_size)

            idx_expand_feat = idx_flat.unsqueeze(1).expand(-1, cond_sparse_feat.shape[1], -1)
            feat_sparse_part = torch.gather(cond_sparse_feat, 2, idx_expand_feat)
            feat_sparse_part = feat_sparse_part.view(B, cond_sparse_feat.shape[1], n_parts, patch_size).permute(
                0, 2, 1, 3
            ).contiguous()
            cond_sparse_feat_flat = feat_sparse_part.reshape(B * n_parts, cond_sparse_feat.shape[1], patch_size)

            if normalize_patches:
                patch_centroid = sparse_part_flat.mean(dim=2, keepdim=True)
                centered = sparse_part_flat - patch_centroid
                patch_scale = torch.norm(centered, p=2, dim=1, keepdim=True).amax(dim=2, keepdim=True)
                patch_scale = patch_scale.clamp_min(1e-6)
                sparse_part_flat = centered / patch_scale
        else:
            if N_sparse % self.n_parts != 0:
                raise RuntimeError(
                    f"N_sparse={N_sparse} is not divisible by n_parts={self.n_parts}, cannot split evenly into parts by index."
                )
            part_size_sparse = N_sparse // self.n_parts

            # Convert to point-first layout before partitioning.
            sparse_xyz_t = sparse_in.transpose(1, 2).contiguous()  # [B, N_sparse, 3]
            # Split consecutive points into equally sized parts.
            sparse_part_xyz = sparse_xyz_t.view(B, self.n_parts, part_size_sparse, 3).contiguous()
            # Partition local conditions with the same point ordering.
            feat_sparse_part = cond_sparse_feat.view(
                B, cond_sparse_feat.shape[1], self.n_parts, part_size_sparse
            ).permute(0, 2, 1, 3).contiguous()

            # Flatten batch and part dimensions for a shared decoder pass.
            sparse_part_flat = sparse_part_xyz.reshape(-1, part_size_sparse, 3).transpose(1, 2).contiguous()
            cond_sparse_feat_flat = feat_sparse_part.reshape(-1, cond_sparse_feat.shape[1], part_size_sparse).contiguous()

        # Repeat flow query features for each part.
        cond_flow_feat_flat = cond_flow_query.unsqueeze(1).expand(
            -1, actual_parts, -1, -1
        ).reshape(-1, cond_flow_query.shape[1], cond_flow_query.shape[2]).contiguous()

        refined_part_outputs = self.refiner(
            sparse_part_flat,
            cond_flow_feat_flat,
            cond_sparse_feat_flat
        )

        refined_outputs = []
        for stage_xyz in refined_part_outputs:
            if patch_centroid is not None and patch_scale is not None:
                stage_xyz = stage_xyz * patch_scale.transpose(1, 2) + patch_centroid.transpose(1, 2)
            refined_outputs.append(
                stage_xyz.view(B, actual_parts, -1, 3).reshape(B, -1, 3)
            )
        return refined_outputs

    def _build_inference_patch_indices(self, sparse_in, patch_size=256, patch_rate=3.0, patch_num=0):
        B = sparse_in.shape[0]
        N_sparse = sparse_in.shape[2]
        patch_size = int(patch_size)
        if patch_size <= 0:
            raise RuntimeError(f"patch_size must be a positive integer, but got {patch_size}")
        if patch_size > N_sparse:
            raise RuntimeError(f"patch_size={patch_size} cannot exceed N_sparse={N_sparse}")
        if int(patch_num) > 0:
            sample_num = int(patch_num)
        else:
            sample_num = int(N_sparse / float(patch_size) * float(patch_rate))
        sample_num = max(1, min(sample_num, N_sparse))
        seed_xyz = FPS(sparse_in, sample_num)
        pts_trans = rearrange(sparse_in, 'b c n -> b n c').contiguous()
        seed_xyz_trans = rearrange(seed_xyz, 'b c m -> b m c').contiguous()
        # Build KNN patches around FPS seeds, merge predictions, then resample with FPS.
        dists = torch.cdist(seed_xyz_trans, pts_trans)  # [B, sample_num, N_sparse]
        patch_indices = torch.topk(dists, k=patch_size, dim=-1, largest=False).indices.long()
        if patch_indices.shape != (B, sample_num, patch_size):
            raise RuntimeError(
                f"unexpected dynamic patch index shape: got={tuple(patch_indices.shape)}, "
                f"expected={(B, sample_num, patch_size)}"
            )
        return patch_indices

    def _rollout_sparse_only(
        self,
        sparse_xyz,
        sparse_feat,
        steps=5,
        category_ids=None,
        image_cond=None,
        sparse_patch_idx=None,
        add_x0_noise=False,
        x0_noise=None,
        action_noise_std=0.0,
        action_noise=None,
        return_trace=False,
        return_refined_outputs=False,
        refiner_index_mode=None,
        inference_patch_size=256,
        inference_patch_rate=3.0,
        inference_patch_num=0,
        inference_patch_normalize=True,
    ):
        """Run Euler flow integration and geometry decoding without GT features.

        Sparse coordinates and external features define the flow state and
        conditions. Optional image conditions are passed to the velocity network.
        """
        if steps <= 0:
            raise RuntimeError(f'steps must be a positive integer, but got {steps}')
        sparse_xyz = sparse_xyz.float()
        sparse_feat = sparse_feat.float()
        device = sparse_xyz.device
        B = sparse_xyz.shape[0]
        sparse_in = sparse_xyz.transpose(1, 2).contiguous()  # [B, 3, N_sparse]

        global_cond, local_cond, local_cond_ref, token_xyz, token_feat = self._extract_condition(
            sparse_in,
            sparse_feat,
        )

        query_pts = FPS(sparse_in, self.flow_token_count)  # [B, 3, N_query]
        local_cond_query = self._three_interp(query_pts, token_xyz, local_cond)
        x_t = self._build_x0(
            token_xyz,
            token_feat,
            query_pts,
            add_noise=add_x0_noise,
            noise=x0_noise,
        )
        n_sparse = sparse_in.shape[2]
        dt = 1.0 / float(steps)

        trace = None
        if return_trace:
            trace = {
                'state_steps': [],
                'mean_velocity_steps': [],
                'sampled_velocity_steps': [],
                'time_steps': [],
                'query_pts': query_pts,
                'global_cond': global_cond,
                'local_cond_query': local_cond_query,
                'token_xyz': token_xyz,
                'token_feat': token_feat,
                'sparse_in': sparse_in,
                'local_cond_ref': local_cond_ref,
            }

        if action_noise is not None:
            expected_shape = (B, steps, self.flow_feature_dim, self.flow_token_count)
            if tuple(action_noise.shape) != expected_shape:
                raise RuntimeError(
                    f'action_noise shape must be {expected_shape}, but got {tuple(action_noise.shape)}'
                )

        for step_idx in range(steps):
            t = torch.full((B,), step_idx / float(steps), device=device, dtype=x_t.dtype)
            pred_velocity = self.condition_flow_matching(
                x_t,
                t,
                global_cond=global_cond,
                local_cond=local_cond_query,
                query_xyz=query_pts,
                image_cond=image_cond,
            )
            if action_noise is not None:
                sampled_velocity = pred_velocity + action_noise_std * action_noise[:, step_idx]
            elif action_noise_std > 0:
                sampled_velocity = pred_velocity + action_noise_std * torch.randn_like(pred_velocity)
            else:
                sampled_velocity = pred_velocity

            if return_trace:
                trace['state_steps'].append(x_t)
                trace['mean_velocity_steps'].append(pred_velocity)
                trace['sampled_velocity_steps'].append(sampled_velocity)
                trace['time_steps'].append(t)

            x_t = x_t + sampled_velocity * dt

        pred_x1 = x_t + self.pred_x1_mlp(x_t)

        use_partwise = bool(getattr(self, 'use_partwise_inference', True))
        if not use_partwise:
            raise RuntimeError(
                f"only the patch-wise Refiner is currently supported. Got use_partwise_inference={use_partwise}"
            )
        index_mode = str(refiner_index_mode or self.inference_refiner_index_mode or 'part').strip().lower()
        if index_mode not in ('part', 'patch'):
            raise RuntimeError(f"refiner_index_mode must be 'part' or 'patch', got {index_mode!r}")
        if index_mode == 'patch':
            refine_indices = self._build_inference_patch_indices(
                sparse_in,
                patch_size=inference_patch_size,
                patch_rate=inference_patch_rate,
                patch_num=inference_patch_num,
            )
            normalize_refine_patches = bool(inference_patch_normalize)
        else:
            refine_indices = None
            normalize_refine_patches = False

        refined_outputs = self._partwise_refine(
            sparse_in,
            pred_x1,
            local_cond_ref,
            patch_indices=refine_indices,
            normalize_patches=normalize_refine_patches,
        )
        final = refined_outputs[-1]
        target_n = n_sparse * math.prod(self.refiner.up_factors)
        if index_mode == 'patch' or final.shape[1] != target_n:
            final = FPS(final.transpose(1, 2).contiguous(), target_n).transpose(1, 2).contiguous()
        if final.shape[1] != target_n:
            raise RuntimeError(
                f"unexpected Refiner output point count: got {final.shape[1]}, expected {target_n}. Please check the patch indices or up_factors configuration."
            )

        if not return_trace and not return_refined_outputs:
            return final

        if return_refined_outputs and not return_trace:
            return final, refined_outputs

        trace['state_steps'] = torch.stack(trace['state_steps'], dim=1)
        trace['mean_velocity_steps'] = torch.stack(trace['mean_velocity_steps'], dim=1)
        trace['sampled_velocity_steps'] = torch.stack(trace['sampled_velocity_steps'], dim=1)
        trace['time_steps'] = torch.stack(trace['time_steps'], dim=1)
        trace['pred_x1'] = pred_x1
        trace['refined_outputs'] = refined_outputs
        return final, trace

    @torch.no_grad()
    def inference(
        self,
        sparse_xyz,
        sparse_feat,
        steps=5,
        category_ids=None,
        image_cond=None,
        sparse_patch_idx=None,
        refiner_index_mode=None,
        inference_patch_size=256,
        inference_patch_rate=3.0,
        inference_patch_num=0,
        inference_patch_normalize=True,
    ):
        """Upsample sparse points with flow integration and two-stage decoding.

        Args:
            sparse_xyz: Sparse coordinates [B, N_sparse, 3].
            sparse_feat: Aligned point features [B, N_sparse, C_flow].
            steps: Number of Euler integration steps.
            category_ids: Unused placeholder for category labels.
            sparse_patch_idx: Optional predefined patch indices [B, P, K].
            image_cond: Optional image features [B, C], [B, V, C] or [B, V, T, C].
            refiner_index_mode: Decoder grouping mode, part or patch.
            inference_patch_size: Number of sparse points per dynamic patch.
            inference_patch_rate: Coverage factor used to choose the patch count.
            inference_patch_num: Explicit dynamic patch count; zero selects it automatically.
            inference_patch_normalize: Whether to center and scale dynamic patches.

        Returns:
            Dense coordinates [B, N_out, 3], normally with N_out = 4 * N_sparse.
        """
        index_mode = str(refiner_index_mode or self.inference_refiner_index_mode or 'part').strip().lower()
        if index_mode not in ('part', 'patch'):
            raise RuntimeError(f"refiner_index_mode must be 'part' or 'patch', got {index_mode!r}")
        self._log_refiner_route_once(index_mode, "inference", sparse_xyz.device)
        return self._rollout_sparse_only(
            sparse_xyz,
            sparse_feat,
            steps=steps,
            image_cond=image_cond,
            sparse_patch_idx=sparse_patch_idx,
            add_x0_noise=False,
            action_noise_std=0.0,
            return_trace=False,
            refiner_index_mode=refiner_index_mode,
            inference_patch_size=inference_patch_size,
            inference_patch_rate=inference_patch_rate,
            inference_patch_num=inference_patch_num,
            inference_patch_normalize=inference_patch_normalize,
        )

    def forward(self, sparse_xyz, sparse_feat=None, gt_feat=None, gt_xyz=None,
                sparse_patch_idx=None, gt_patch_idx=None,
                image_cond=None,
                category_ids=None,
                train=True,
                return_loss_breakdown=False,
                refiner_index_mode=None,
                inference_patch_size=256,
                inference_patch_rate=3.0,
                inference_patch_num=0,
                inference_patch_normalize=True):
        """Compute training losses or return an inference prediction.

        Args:
            sparse_xyz: Sparse coordinates [B, N_sparse, 3].
            sparse_feat: Aligned external features [B, N_sparse, C_flow].
            gt_feat: Dense target features [B, N_gt, C_flow].
            gt_xyz: Dense GT coordinates [B, N_gt, 3].
            sparse_patch_idx: Optional sparse patch indices [B, P, K_sparse].
            gt_patch_idx: Optional GT patch indices [B, P, K_gt].
            image_cond: Optional image features [B, C], [B, V, C] or [B, V, T, C].
            category_ids: Unused placeholder for category labels.
            train: Whether to compute training losses instead of running inference.
            return_loss_breakdown: Whether to return all six training loss terms.
            refiner_index_mode: Decoder grouping mode, part or patch.
            inference_patch_size: Number of sparse points per dynamic patch.
            inference_patch_rate: Coverage factor used to choose the patch count.
            inference_patch_num: Explicit dynamic patch count; zero selects it automatically.
            inference_patch_normalize: Whether to center and scale dynamic patches.

        Returns:
            Training: total, Chamfer and flow losses, each [B]. With
            return_loss_breakdown=True, also return prototype, global Chamfer
            and local Chamfer losses. Inference: dense coordinates [B, N_out, 3].
        """
        sparse_xyz = sparse_xyz.float()

        if train:
            device = sparse_xyz.device
            B = sparse_xyz.shape[0]
            sparse_in = sparse_xyz.transpose(1, 2).contiguous()  # [B, 3, N_sparse]
            sparse_feat = sparse_feat.float()
            gt_feat = gt_feat.float()
            gt_xyz = gt_xyz.float()

            # Extract conditions from sparse coordinates.
            global_cond, local_cond, local_cond_ref, token_xyz, token_feat = self._extract_condition(
                sparse_in,
                sparse_feat,
            )

            # Use sparse-input FPS queries for both training and inference.
            query_pts = FPS(sparse_in, self.flow_token_count)  # [B, 3, N_query]

            # Gather query conditions and initialize x_0 from external features.
            local_cond_query = self._three_interp(query_pts, token_xyz, local_cond)
            x_0 = self._build_x0(token_xyz, token_feat, query_pts, add_noise=True)

            # Interpolate dense GT features onto sparse query coordinates for x_1.
            gt_feat_query = self._align_gt_to_query(gt_feat, gt_xyz, query_pts)
            x_1 = gt_feat_query

            # Sample time with cosine reparameterization along the linear flow path.
            t = torch.rand(B, device=device)
            t = 1 - torch.cos(t * math.pi / 2)
            t_expanded = t.view(B, 1, 1)

            x_t = t_expanded * x_1 + (1 - t_expanded) * x_0
            target_flow = x_1 - x_0

            pred_flow_ctrl, flow_aux = self.condition_flow_matching(
                x_t, t,
                global_cond=global_cond,
                local_cond=local_cond_query,
                query_xyz=query_pts,
                image_cond=image_cond,
                target_prototypes=gt_feat_query,
                return_aux=True,
            )  # [B, C_flow, N_query]

            loss_fm_ctrl = F.mse_loss(
                pred_flow_ctrl.float(), target_flow.float(),
                reduction='none'
            ).mean(dim=[1, 2])  # [B]

            if image_cond is not None and self.use_image_controller:
                loss_proto = flow_aux.get('proto_consistency', torch.zeros_like(loss_fm_ctrl)).float()
            else:
                loss_proto = torch.zeros_like(loss_fm_ctrl)

            # Estimate x_1 from the predicted velocity in one step.
            pred_x1 = x_t + (1 - t_expanded) * pred_flow_ctrl  # [B, C_flow, N_query]
            pred_x1 = pred_x1 + self.pred_x1_mlp(pred_x1)  # [B, C_flow, N_query]

            # Decode a 2N-point first stage and a 4N-point final output.
            N_sparse = sparse_in.shape[2]
            index_mode = str(refiner_index_mode or 'part').strip().lower()
            if index_mode not in ('part', 'patch'):
                raise RuntimeError(f"refiner_index_mode must be 'part' or 'patch', got {index_mode!r}")
            # Part mode uses contiguous partitions; patch mode can use dataset indices.
            if index_mode == 'part':
                sparse_patch_idx = None
                gt_patch_idx = None
            elif sparse_patch_idx is None or gt_patch_idx is None:
                raise RuntimeError(
                    "patch mode training requires both sparse_patch_idx and gt_patch_idx; they are missing in the current batch. "
                    "Please check the dataset/cache, or switch to --index_mode part."
                )

            use_partwise = True
            if sparse_patch_idx is None and N_sparse % self.n_parts != 0:
                raise RuntimeError(
                    f"the training stage only supports the patch-wise Refiner, N_sparse must be divisible by n_parts. "
                    f"Got N_sparse={N_sparse}, n_parts={self.n_parts}"
                )
            self._log_refiner_route_once(index_mode, "train", device)
            refined_outputs = self._partwise_refine(
                sparse_in,
                pred_x1,
                local_cond_ref,
                patch_indices=(sparse_patch_idx if index_mode == 'patch' else None),
            )
            expected_final_n = gt_xyz.shape[1]
            final_refined_pre = refined_outputs[-1]
            final_refined = final_refined_pre
            if final_refined.shape[1] != expected_final_n:
                final_refined = FPS(
                    final_refined.transpose(1, 2).contiguous(),
                    expected_final_n,
                ).transpose(1, 2).contiguous()
            if final_refined.shape[1] != expected_final_n:
                raise RuntimeError(
                    f"the Refiner final output configuration is inconsistent with the training target: "
                    f"refiner={final_refined.shape[1]}, target={expected_final_n}."
                )

            def _compute_local_loss_for_stage(pred_pts: torch.Tensor) -> torch.Tensor:
                if index_mode == 'patch':
                    return self._compute_patch_loss(pred_pts, gt_xyz, gt_patch_idx)
                if (
                    pred_pts.shape[1] % self.n_parts == 0
                    and gt_xyz.shape[1] % self.n_parts == 0
                ):
                    return self._compute_local_cd(pred_pts, gt_xyz)
                return torch.zeros(B, device=device)

            loss_cd_global = torch.zeros(B, device=device)
            loss_local = torch.zeros(B, device=device)
            for stage_idx, pred_pts in enumerate(refined_outputs):
                pred_for_global = pred_pts
                if stage_idx == len(refined_outputs) - 1:
                    pred_for_global = final_refined
                d1, d2, _, _ = self.chamfer_loss(pred_for_global.float(), gt_xyz.float())
                global_cd = torch.mean(d1, dim=1) + torch.mean(d2, dim=1)
                loss_cd_global = loss_cd_global + global_cd
                stage_local_weight = 1.0
                if stage_idx == len(refined_outputs) - 1:
                    stage_local_weight = 1.0
                loss_local = loss_local + stage_local_weight * _compute_local_loss_for_stage(pred_pts)

            # Sum global CD, local CD, flow matching and prototype consistency losses.
            loss_cd = loss_cd_global + loss_local
            loss_total = loss_cd + loss_fm_ctrl + loss_proto
            if return_loss_breakdown:
                return (
                    loss_total,
                    loss_cd,
                    loss_fm_ctrl,
                    loss_proto,
                    loss_cd_global,
                    loss_local,
                )
            return loss_total, loss_cd, loss_fm_ctrl

        steps = getattr(self, 'inference_steps', 5)
        return self.inference(
            sparse_xyz,
            sparse_feat,
            steps=steps,
            image_cond=image_cond,
            sparse_patch_idx=sparse_patch_idx,
            refiner_index_mode=refiner_index_mode,
            inference_patch_size=inference_patch_size,
            inference_patch_rate=inference_patch_rate,
            inference_patch_num=inference_patch_num,
            inference_patch_normalize=inference_patch_normalize,
        )
