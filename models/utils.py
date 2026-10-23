"""Point-cloud sampling, indexing, distances and training helpers."""

import torch
from einops import rearrange
from models.pointops.functions import pointops
import numpy as np
import random
from torch.autograd import grad

def square_distance(src, dst):
    """Compute pairwise squared Euclidean distances.

    Args:
        src: Source points [B, N, C].
        dst: Target points [B, M, C].

    Returns:
        Squared distances [B, N, M].
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return points_grad


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def index_points(pts, idx):
    """Gather channel-first features by batched point indices.

    Args:
        pts: Point features [B, C, N].
        idx: Indices [B, S] or [B, S, K].

    Returns:
        Gathered features [B, C, S] or [B, C, S, K].
    """
    batch_size = idx.shape[0]
    sample_num = idx.shape[1]
    fdim = pts.shape[1]
    reshape = False
    if len(idx.shape) == 3:
        reshape = True
        idx = idx.reshape(batch_size, -1)
    # [B, C, S * K]
    res = torch.gather(pts, 2, idx[:, None].repeat(1, fdim, 1))
    if reshape:
        res = rearrange(res, 'b c (s k) -> b c s k', s=sample_num)

    return res


def FPS(pts, fps_pts_num):
    # Input coordinates: [B, 3, N].
    sample_pts, _ = FPS_with_idx(pts, fps_pts_num)
    return sample_pts


def fps_indices(pts, fps_pts_num):
    # Input coordinates: [B, 3, N].
    pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
    return pointops.furthestsampling(pts_trans, fps_pts_num).long()


def FPS_with_idx(pts, fps_pts_num):
    # Input coordinates: [B, 3, N].
    sample_idx = fps_indices(pts, fps_pts_num)
    sample_pts = index_points(pts, sample_idx)
    return sample_pts, sample_idx


def get_knn_pts(k, pts, center_pts, return_idx=False):
    # Input coordinates: [B, 3, N]; query centers: [B, 3, M].

    # [B, N, 3]
    pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
    # [B, M, 3]
    center_pts_trans = rearrange(center_pts, 'b c m -> b m c').contiguous()
    # [B, M, K]
    knn_idx = pointops.knnquery_heap(k, pts_trans, center_pts_trans).long()
    # [B, 3, M, K]
    knn_pts = index_points(pts, knn_idx)

    if return_idx == False:
        return knn_pts
    else:
        return knn_pts, knn_idx


def add_noise(pts, sigma, clamp):
    # Add clipped Gaussian noise to coordinates: [B, 3, N].

    assert (clamp > 0)
    jittered_data = torch.clamp(sigma * torch.randn_like(pts), -1 * clamp, clamp).to(pts.device)
    jittered_data += pts

    return jittered_data


# Build inference patches around FPS seeds.

