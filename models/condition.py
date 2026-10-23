"""PointNet++ MSG condition extractor."""

import torch
import torch.nn as nn

from models.pointnet2_util import (
    PointNetFeaturePropagation,
    PointNetSetAbstractionMsg,
)
from models.utils import get_knn_pts


class PointNetMSGConditionExtractor(nn.Module):
    """Extract global and pointwise conditions with a PointNet++ MSG pyramid.

    Default abstraction levels contain 1024, 256, 64 and 16 points.
    Feature propagation restores features to the input point count N.
    Forward accepts coordinates [B, 3, N] and returns global features [B, C_out].
    With return_pointwise=True, it also returns local features [B, C_out, N].
    """
    def __init__(self, out_dim=256, sa_npoints=(1024, 256, 64, 16)):
        super().__init__()
        self.out_dim = out_dim
        self.sa_npoints = sa_npoints

        self.sa1 = PointNetSetAbstractionMsg(
            self.sa_npoints[0], [0.05, 0.1], [16, 32], 3 * 16,
            [[16, 16, 32], [32, 32, 64]]
        )
        self.sa2 = PointNetSetAbstractionMsg(
            self.sa_npoints[1], [0.1, 0.2], [16, 32], 32 + 64,
            [[64, 64, 128], [64, 96, 128]]
        )
        self.sa3 = PointNetSetAbstractionMsg(
            self.sa_npoints[2], [0.2, 0.4], [16, 32], 128 + 128,
            [[128, 196, 256], [128, 196, 256]]
        )
        self.sa4 = PointNetSetAbstractionMsg(
            self.sa_npoints[3], [0.4, 0.8], [16, 32], 256 + 256,
            [[256, 256, 512], [256, 384, 512]]
        )

        self.fp4 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32 + 64 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, out_dim])
        self.global_proj = nn.Sequential(
            nn.Linear(1024, out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, return_pointwise=False):
        # x: [B, 3, N]
        bs = x.shape[0]
        n = x.shape[2]

        l0_xyz = x
        l0_k = min(16, n)
        l0_points = get_knn_pts(l0_k, l0_xyz, l0_xyz, return_idx=False)
        l0_points = l0_points.permute(0, 1, 3, 2).reshape(bs, -1, n)

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        pointwise_feat = l0_points
        global_feature = torch.max(l4_points, dim=2)[0]
        global_feature = self.global_proj(global_feature)

        if return_pointwise:
            return global_feature, pointwise_feat
        return global_feature
