"""Point sampling, neighborhood grouping and feature propagation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import FPS, index_points, square_distance
from models.pointops.functions import pointops


def nearest_k_distances(src, dst, k):
    """Return nearest-neighbor squared distances and indices.

    Args:
        src: Query coordinates [B, N, C].
        dst: Reference coordinates [B, M, C].
        k: Requested neighbor count, capped at M.

    Returns:
        Unsorted distances and indices, both [B, N, min(k, M)].
    """
    dists = square_distance(src, dst)
    k = min(int(k), dists.shape[-1])
    dists, idx = torch.topk(dists, k=k, dim=-1, largest=False, sorted=False)
    return dists, idx


def index_points(points, idx):
    """Gather point-first values with clamped batched indices.

    Args:
        points: Features [B, N, C] or scalar values [B, N].
        idx: Indices [B, S] or [B, S, K].

    Returns:
        Values with shape idx.shape + [C] for features, or idx.shape for scalars.
    """
    device = points.device
    B = points.shape[0]
    N = points.shape[1]
    idx_safe = idx.clamp(0, N - 1)
    if points.dim() == 3:
        # points: [B, N, C]
        if idx_safe.dim() == 2:
            # [B, S]
            batch_indices = torch.arange(B, device=device).view(B, 1).expand(-1, idx_safe.shape[1])
            return points[batch_indices, idx_safe, :]
        elif idx_safe.dim() == 3:
            # [B, S, K]
            B, S, K = idx_safe.shape
            batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(-1, S, K)
            return points[batch_indices, idx_safe, :]
        else:
            raise ValueError(f"idx shape not supported: {idx_safe.shape}")
    elif points.dim() == 2:
        # points: [B, N]
        if idx_safe.dim() == 2:
            batch_indices = torch.arange(B, device=device).view(B, 1).expand(-1, idx_safe.shape[1])
            return points[batch_indices, idx_safe]
        elif idx_safe.dim() == 3:
            B, S, K = idx_safe.shape
            batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(-1, S, K)
            return points[batch_indices, idx_safe]
        else:
            raise ValueError(f"idx shape not supported: {idx_safe.shape}")
    else:
        raise ValueError(f"points shape not supported: {points.shape}")


def farthest_point_sample(xyz, npoint):
    """Select point indices by farthest-point sampling.

    Args:
        xyz: Coordinates [B, N, 3].
        npoint: Number of samples.

    Returns:
        Sampled indices [B, npoint].
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """Group points within a radius of each query center.

    Args:
        radius: Query radius.
        nsample: Maximum number of neighbors per center.
        xyz: Reference coordinates [B, N, 3].
        new_xyz: Query coordinates [B, S, 3].

    Returns:
        Grouped point indices [B, S, nsample].
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """Sample centers with FPS and gather radius-based neighborhoods.

    Args:
        npoint: Number of sampled centers.
        radius: Query radius.
        nsample: Maximum number of neighbors per center.
        xyz: Coordinates [B, N, 3].
        points: Optional point features [B, N, D].
        returnfps: Whether to also return grouped coordinates and FPS indices.

    Returns:
        Center coordinates [B, npoint, 3] and grouped relative coordinates/features
        [B, npoint, nsample, 3 + D]. D is zero when points is None.
        If returnfps is True, also return grouped coordinates
        [B, npoint, nsample, 3] and center indices [B, npoint].
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C + D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """Group all points around a zero-valued center.

    Args:
        xyz: Coordinates [B, N, 3].
        points: Optional features [B, N, D].

    Returns:
        Zero center [B, 1, 3] and grouped coordinates/features [B, 1, N, 3 + D].
        D is zero when points is None.
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """Sample centers and aggregate local point features.

        Args:
            xyz: Point coordinates [B, 3, N].
            points: Optional point features [B, D, N].

        Returns:
            Sampled coordinates [B, 3, S] and aggregated features [B, D_out, S].
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # Sampled coordinates: [B, S, 3].
        # Grouped relative coordinates and features: [B, S, K, 3 + D].
        new_points = new_points.permute(0, 3, 2, 1)  # [B, 3 + D, K, S]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        self.in_channel = in_channel
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """Sample centers and aggregate local point features.

        Args:
            xyz: Point coordinates [B, 3, N] or [B, N, 3].
            points: Optional point features [B, D, N].

        Returns:
            Sampled coordinates [B, 3, S] and aggregated features [B, D_out, S].
        """
        # Accept channel-first or point-first coordinates.
        grouped_points = None
        if xyz.shape[1] == 3 and xyz.shape[-1] != 3:
            xyz = xyz.permute(0, 2, 1)
        xyz = xyz[..., :3]  # Keep XYZ channels in point-first layout: [B, N, 3].
        if points is not None:
            if points.dim() == 4:
                points = points.mean(-1)
            assert points.dim() == 3, f"points should be 3D [B, C, N], but got {points.shape}"
            assert points.shape[1] == self.in_channel, f"Expected {self.in_channel} channels, got {points.shape[1]}"
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = FPS(xyz.permute(0, 2, 1), S)
        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = pointops.ballquery(radius, K, xyz, new_xyz)
            xyz_perm = xyz if xyz.shape[-1] == 3 else xyz[..., :3]
            assert xyz_perm.shape[-1] == 3, f"Input xyz for grouping must have last dim=3, got {xyz_perm.shape}"
            grouped_xyz = index_points(xyz_perm, group_idx)
            grouped_xyz = grouped_xyz - new_xyz.view(B, S, 1, 3)
            if grouped_points is not None:
                # Align grouped feature axes with coordinate groups.
                if grouped_points.shape[:3] != grouped_xyz.shape[:3]:
                    grouped_points = grouped_points.permute(0, 2, 1, 3).contiguous()
            if points is not None:
                points_perm = points  # [B, N, C]
                group_idx_safe = group_idx.clamp(0, points_perm.shape[1] - 1)
                grouped_points = index_points(points_perm, group_idx_safe)
                # Normalize grouped features to [B, S, K, C].
                if grouped_points.shape[:2] != grouped_xyz.shape[:2]:
                    grouped_points = grouped_points.permute(0, 2, 1, 3).contiguous()
                
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            if grouped_points.shape[-1] != grouped_points.shape[1]:
                if grouped_points.shape[1] == self.nsample_list[i]:
                    # [B, K, S, C] -> [B, S, K, C]
                    grouped_points = grouped_points.permute(0, 2, 1, 3).contiguous()
            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            conv_in_channels = self.conv_blocks[i][0].in_channels
            cur_channels = grouped_points.shape[1]
            if cur_channels != conv_in_channels:
                raise RuntimeError(
                    f"PointNetSetAbstractionMsg channel mismatch at scale={i}: "
                    f"got grouped_points C={cur_channels}, expected conv C={conv_in_channels}. "
                    f"radius={self.radius_list[i]}, nsample={self.nsample_list[i]}, "
                    f"grouped_points.shape={tuple(grouped_points.shape)}"
                )
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetSetAbstractionMsgAttn(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsgAttn, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()

        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)
        
        for i in range(len(mlp_list)):
            last_channel = in_channel + 3
            self.attn_blocks.append(AttentionModule(C_in1=mlp_list[i][-1], C_in2=last_channel, C1=mlp_list[i][-1], C2=mlp_list[i][-1], C_out=mlp_list[i][-1]))


    def forward(self, xyz, points):
        """Sample centers and aggregate local point features.

        Args:
            xyz: Point coordinates [B, 3, N] or [B, N, 3].
            points: Optional point features [B, D, N].

        Returns:
            Sampled coordinates [B, 3, S] and aggregated features [B, D_out, S].
        """
        # Accept channel-first or point-first coordinates.
        if xyz.shape[1] == 3 and xyz.shape[-1] != 3:
            xyz = xyz.permute(0, 2, 1)
        xyz = xyz[..., :3]  # Keep XYZ channels in point-first layout: [B, N, 3].
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = FPS(xyz.permute(0, 2, 1), S)
        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = pointops.ballquery(radius, K, xyz, new_xyz)
            xyz_perm = xyz if xyz.shape[-1] == 3 else xyz[..., :3]
            assert xyz_perm.shape[-1] == 3, f"Input xyz for grouping must have last dim=3, got {xyz_perm.shape}"
            grouped_xyz = index_points(xyz_perm, group_idx)
            grouped_xyz = grouped_xyz - new_xyz.view(B, S, 1, 3)
            if points is not None:
                grouped_points = index_points(points.permute(0, 2, 1), group_idx)
                grouped_points = grouped_points.permute(0, 2, 3, 1)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            grouped_points_before = grouped_points
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points = self.attn_blocks[i](new_points, grouped_points_before, grouped_points)
            
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """Interpolate coarse features and fuse optional fine-level features.

        Args:
            xyz1: Fine coordinates [B, 3, N].
            xyz2: Coarse coordinates [B, 3, S].
            points1: Optional fine features [B, D1, N].
            points2: Coarse features [B, D2, S].

        Returns:
            Propagated features [B, D_out, N].
        """
        # Convert channel-first coordinates to point-first layout for interpolation.
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)


        if points2.dim() == 4:
            points2 = points2.mean(-1)
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists, idx = nearest_k_distances(xyz1, xyz2, k=3)

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            # Gather neighboring features before weighted interpolation.
            a = index_points(points2, idx)
            interpolated_points = torch.sum(a * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            if points1.dim() == 4:
                points1 = points1.mean(-1)
            assert points1.dim() == 3, f"points1 should be 3D [B, C, N], but got {points1.shape}"
            points1 = points1.permute(0, 2, 1)
            if points1.shape[0] != interpolated_points.shape[0] or points1.shape[1] != interpolated_points.shape[1]:
                raise RuntimeError(
                    "PointNetFeaturePropagation shape mismatch before concat: "
                    f"points1.shape={tuple(points1.shape)}, "
                    f"interpolated_points.shape={tuple(interpolated_points.shape)}"
                )
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            conv_in_channels = conv.in_channels
            cur_channels = new_points.shape[1]
            if cur_channels != conv_in_channels:
                raise RuntimeError(
                    f"PointNetFeaturePropagation channel mismatch at mlp[{i}]: "
                    f"got input C={cur_channels}, expected conv C={conv_in_channels}. "
                    f"new_points.shape={tuple(new_points.shape)}"
                )
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class PointNetFeaturePropagation_v2(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation_v2, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """Interpolate coarse features and fuse optional fine-level features.

        Args:
            xyz1: Fine coordinates [B, 3, N].
            xyz2: Coarse coordinates [B, 3, S].
            points1: Optional fine features [B, D1, N].
            points2: Coarse features [B, D2, S].

        Returns:
            Propagated features [B, D_out, N].
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists, idx = nearest_k_distances(xyz1, xyz2, k=3)

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            a = index_points(points2.permute(0, 2, 1), idx)
            interpolated_points = torch.sum(a.permute(0, 2, 3, 1) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class PartialChannelGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels):
        super(PartialChannelGroupNorm, self).__init__()
        self.num_channels = num_channels - num_channels % num_groups
        self.num_groups = num_groups
        self.group_norm = nn.GroupNorm(self.num_groups, self.num_channels)
    def forward(self, x):
        # Normalize spatial channels of [B, C, H, W] features.
        if x.shape[1] == self.num_channels:
            out = self.group_norm(x)
        else:
            # Leave appended position channels unnormalized.
            x0 = x[:,0:self.num_channels,:,:]
            res = x[:,self.num_channels:,:,:]
            x0_out = self.group_norm(x0)
            out = torch.cat([x0_out, res], dim=1)
        return out


class AttentionModule(nn.Module):
    def __init__(self, C_in1, C_in2, C1, C2, C_out, attention_bn=True, transform_grouped_feat_out=True, last_activation=True):
        super(AttentionModule, self).__init__()
        C1 = max(C1, 32)
        C2 = max(C2, 32)
        self.feat_conv = nn.Conv2d(C_in1, C1, kernel_size=1, bias=False)
        self.grouped_feat_conv = nn.Conv2d(C_in2, C2, kernel_size=1, bias=False)

        inter_C = min(C1+C2, C_out)
        if attention_bn:
            self.weight_conv = nn.Sequential(
                        nn.ReLU(),
                        PartialChannelGroupNorm(min(32, C1+C2), C1+C2),
                        nn.Conv2d(C1+C2, inter_C, kernel_size=1, bias=False),
                        nn.ReLU(),
                        PartialChannelGroupNorm(min(32, inter_C), inter_C),
                        nn.Conv2d(inter_C, C_out,kernel_size=1, bias=False))
        else:
            self.weight_conv = nn.Sequential(
                        nn.ReLU(),
                        nn.Conv2d(C1+C2, inter_C, kernel_size=1, bias=False),
                        nn.ReLU(),
                        nn.Conv2d(inter_C, C_out,kernel_size=1, bias=False))

        self.transform_grouped_feat_out = transform_grouped_feat_out
        if transform_grouped_feat_out:
            self.feat_out_conv = [nn.Conv2d(C_out, C_out, kernel_size=1, bias=False)]
            if last_activation:
                if attention_bn:
                    self.feat_out_conv.append(PartialChannelGroupNorm(min(32, C_out), C_out))
                self.feat_out_conv.append(nn.ReLU())

            self.feat_out_conv = nn.Sequential(*self.feat_out_conv)


    def forward(self, feat, grouped_feat, grouped_feat_out):
        # Query features: [B, C_in1, N].
        # Key features: [B, C_in2, N, K].
        # Value features: [B, C_out, N, K].
        grouped_feat = grouped_feat.permute(0, 1, 3, 2).contiguous()
        grouped_feat_out = grouped_feat_out.permute(0, 1, 3, 2).contiguous()
        K = grouped_feat.shape[-1]
        feat1 = self.feat_conv(feat.unsqueeze(-1))  # [B, C1, N, 1]
        feat1 = feat1.repeat(1, 1, 1, K)  # [B, C1, N, K]

        grouped_feat1 = self.grouped_feat_conv(grouped_feat)  # [B, C2, N, K]

        total_feat = torch.cat([feat1, grouped_feat1], dim=1)  # [B, C1 + C2, N, K]
        scores = self.weight_conv(total_feat)  # [B, C_out, N, K]

        weight = F.softmax(scores, dim=-1).contiguous()  # [B, C_out, N, K]

        if self.transform_grouped_feat_out:
            grouped_feat_out = self.feat_out_conv(grouped_feat_out)  # [B, C_out, N, K]
        out = grouped_feat_out * weight  # [B, C_out, N, K]
        out = out.sum(dim=-1)  # [B, C_out, N]
        return out
