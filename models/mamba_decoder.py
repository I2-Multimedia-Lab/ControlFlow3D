"""Two-stage Mamba geometry decoder for point-cloud upsampling.

The first stage fuses local conditions and flow tokens with cross-attention
and Hilbert-ordered Mamba scans. The second stage uses Z-order scans.
A refinement head combines both branches into a 4x prediction.
Keep module and parameter names stable for checkpoint compatibility.
"""

import ctypes
import importlib.util
import pathlib
import sys
import types

import torch
import torch.nn as nn
from timm.models.layers import DropPath
from torch import einsum
from torch.utils.checkpoint import checkpoint

from extensions.Pointnet2.pointnet2.pointnet2_utils import grouping_operation
from models.pointops.functions import pointops
from models.utils import square_distance


def _preload_torch_shared_libs():
    libdir = pathlib.Path(torch.__file__).resolve().parent / "lib"
    for name in (
        "libc10.so",
        "libtorch.so",
        "libtorch_cpu.so",
        "libtorch_python.so",
        "libc10_cuda.so",
        "libtorch_cuda.so",
    ):
        candidate = libdir / name
        if candidate.exists():
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)


def _load_mamba_class():
    # Load the Mamba block without eagerly importing unrelated package modules.
    _preload_torch_shared_libs()
    pkg_name = "mamba_ssm"
    if pkg_name not in sys.modules:
        spec = importlib.util.find_spec(pkg_name)
        if spec is None or not spec.submodule_search_locations:
            raise ImportError("mamba_ssm is not installed or discoverable.")
        pkg = types.ModuleType(pkg_name)
        pkg.__file__ = spec.origin
        pkg.__path__ = list(spec.submodule_search_locations)
        sys.modules[pkg_name] = pkg
    from mamba_ssm.modules.mamba_simple import Mamba as _Mamba

    return _Mamba


Mamba = _load_mamba_class()


def knn(x, k):
    """Find neighbor indices [B, N, K] for coordinates x [B, 3, N]."""
    with torch.cuda.amp.autocast(enabled=False):
        with torch.no_grad():
            x = x.float()
            x_t = x.transpose(2, 1).contiguous()
            if x_t.is_cuda:
                return pointops.knnquery_heap(k, x_t, x_t).long()
            dist = torch.cdist(x_t, x_t)
            return dist.topk(k=k, dim=-1, largest=False)[1]


def get_graph_feature(x, k=20, idx=None):
    """Build local graph features from x [B, C, N] for the feed-forward block."""
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).view(-1)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


def gather_bnc(points, idx):
    """Gather points [B, N, C] with indices [B, S] into [B, S, C]."""
    idx_expand = idx.unsqueeze(-1).expand(-1, -1, points.shape[-1])
    return torch.gather(points, dim=1, index=idx_expand)


def _binary_right_shift(binary, k=1, axis=-1):
    if binary.shape[axis] <= k:
        return torch.zeros_like(binary)
    slicing = [slice(None)] * len(binary.shape)
    slicing[axis] = slice(None, -k)
    return torch.nn.functional.pad(binary[tuple(slicing)], (k, 0), mode="constant", value=0)


def _gray_to_binary(gray, axis=-1):
    shift = int(2 ** (torch.tensor(float(gray.shape[axis]), device=gray.device).log2().ceil().item() - 1))
    while shift > 0:
        gray = torch.logical_xor(gray, _binary_right_shift(gray, shift))
        shift //= 2
    return gray


def hilbert_encode(grid_coord, num_dims=3, num_bits=10):
    """Encode nonnegative integer grid coordinates [M, 3] as Hilbert keys."""
    orig_shape = grid_coord.shape
    if orig_shape[-1] != num_dims:
        raise ValueError(f"hilbert_encode expects last dim {num_dims}, got {orig_shape[-1]}")
    if num_dims * num_bits > 63:
        raise ValueError("Hilbert code needs num_dims * num_bits <= 63.")

    bitpack_mask = 1 << torch.arange(0, 8, device=grid_coord.device)
    bitpack_mask_rev = bitpack_mask.flip(-1)

    locs_uint8 = grid_coord.long().view(torch.uint8).reshape((-1, num_dims, 8)).flip(-1)
    gray = (
        locs_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0)
        .byte()
        .flatten(-2, -1)[..., -num_bits:]
    )

    for bit in range(num_bits):
        for dim in range(num_dims):
            mask = gray[:, dim, bit]
            gray[:, 0, bit + 1:] = torch.logical_xor(gray[:, 0, bit + 1:], mask[:, None])
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]).repeat(1, gray.shape[2] - bit - 1),
                torch.logical_xor(gray[:, 0, bit + 1:], gray[:, dim, bit + 1:]),
            )
            gray[:, dim, bit + 1:] = torch.logical_xor(gray[:, dim, bit + 1:], to_flip)
            gray[:, 0, bit + 1:] = torch.logical_xor(gray[:, 0, bit + 1:], to_flip)

    gray = gray.swapaxes(1, 2).reshape((-1, num_bits * num_dims))
    hh_bin = _gray_to_binary(gray)

    padded = torch.nn.functional.pad(hh_bin, (64 - num_bits * num_dims, 0), "constant", 0)
    hh_uint8 = (
        (padded.flip(-1).reshape((-1, 8, 8)) * bitpack_mask)
        .sum(2)
        .squeeze()
        .type(torch.uint8)
    )
    return hh_uint8.view(torch.int64).squeeze()


class KeyLUT:
    def __init__(self):
        r256 = torch.arange(256, dtype=torch.int64)
        zero = torch.zeros(256, dtype=torch.int64)
        device = torch.device("cpu")
        self._encode = {
            device: (
                self.xyz2key(r256, zero, zero, 8),
                self.xyz2key(zero, r256, zero, 8),
                self.xyz2key(zero, zero, r256, 8),
            )
        }

    def encode_lut(self, device=torch.device("cpu")):
        if device not in self._encode:
            cpu = torch.device("cpu")
            self._encode[device] = tuple(e.to(device) for e in self._encode[cpu])
        return self._encode[device]

    @staticmethod
    def xyz2key(x, y, z, depth):
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key = (
                key
                | ((x & mask) << (2 * i + 2))
                | ((y & mask) << (2 * i + 1))
                | ((z & mask) << (2 * i + 0))
            )
        return key


_key_lut = KeyLUT()


def z_order_encode(grid_coord, depth=10):
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    ex, ey, ez = _key_lut.encode_lut(x.device)

    mask = 255 if depth > 8 else (1 << depth) - 1
    key = ex[x & mask] | ey[y & mask] | ez[z & mask]
    if depth > 8:
        mask = (1 << (depth - 8)) - 1
        key16 = ex[(x >> 8) & mask] | ey[(y >> 8) & mask] | ez[(z >> 8) & mask]
        key = key16 << 24 | key
    return key


def quantize_xyz_to_grid(xyz, depth=10):
    """Quantize coordinates [B, N, 3] to integer cells in [0, 2**depth - 1]."""
    xyz_min = xyz.amin(dim=1, keepdim=True)
    xyz_max = xyz.amax(dim=1, keepdim=True)
    scale = (2 ** depth - 1) / (xyz_max - xyz_min).clamp(min=1e-6)
    grid_coord = torch.round((xyz - xyz_min) * scale).long()
    return grid_coord.clamp_(min=0, max=2 ** depth - 1)


def build_serial_order(xyz, order="hilbert", depth=10):
    """Return point ordering indices [B, N] for coordinates [B, N, 3].

    Supported orders are hilbert, hilbert-reverse, z and z-reverse.
    """
    if xyz.dim() != 3 or xyz.shape[-1] != 3:
        raise RuntimeError(f"build_serial_order expects xyz=[B, N, 3], got {tuple(xyz.shape)}")

    batch_size, num_points, _ = xyz.shape
    grid_coord = quantize_xyz_to_grid(xyz.float(), depth=depth).reshape(-1, 3)
    batch = torch.arange(batch_size, device=xyz.device).repeat_interleave(num_points)

    if order in {"hilbert", "hilbert-reverse"}:
        code = hilbert_encode(grid_coord, num_dims=3, num_bits=depth)
        code = (batch.long() << (depth * 3)) | code
    elif order in {"z", "z-reverse"}:
        code = z_order_encode(grid_coord, depth=depth)
        code = (batch.long() << 48) | code
    else:
        raise ValueError(f"Unsupported scan order: {order}")

    flat_order = code.argsort(dim=0).view(batch_size, num_points)
    local_order = (flat_order % num_points).long()
    if order in {"hilbert-reverse", "z-reverse"}:
        local_order = local_order.flip(-1)
    return local_order


class RefinerPosEmbed(nn.Module):
    """Embed coordinates [B, N, 3] into positional features [B, N, C_out]."""

    def __init__(self, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.SiLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, xyz):
        return self.net(xyz)


class LocalGeometryFFN(nn.Module):
    """Apply a local feed-forward block to features [B, N, C] and coordinates [B, N, 3]."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., gcn_k=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features * 2, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.k = gcn_k

    def forward(self, x, center_xyz):
        idx = knn(center_xyz.transpose(2, 1), k=self.k)
        x_graph = get_graph_feature(x.permute(0, 2, 1), k=self.k, idx=idx).permute(0, 2, 3, 1)
        x_graph = self.fc1(x_graph)
        x_graph = self.act(x_graph)
        x_graph = self.drop(x_graph)
        x_graph = x_graph.max(dim=-2, keepdim=False)[0]
        x_graph = self.fc2(x_graph)
        x_graph = self.drop(x_graph)
        return x_graph


class MultiScanMamba(nn.Module):
    """Aggregate Mamba scans over multiple orders of the same point set."""

    def __init__(self, d_model, scan_depth=10, scan_orders=("hilbert", "hilbert-reverse", "z")):
        super().__init__()
        self.scan_depth = scan_depth
        self.scan_orders = tuple(scan_orders)
        self.mamba_layers = nn.ModuleList([Mamba(d_model=d_model) for _ in self.scan_orders])
        self.merge = nn.Sequential(
            nn.Linear(d_model * len(self.scan_orders), d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x, center_xyz):
        scan_outputs = []
        for order_name, layer in zip(self.scan_orders, self.mamba_layers):
            order = build_serial_order(center_xyz, order=order_name, depth=self.scan_depth)
            x_sorted = gather_bnc(x, order)
            out_sorted = layer(x_sorted)
            inv_order = order.argsort(dim=-1)
            scan_outputs.append(gather_bnc(out_sorted, inv_order))
        return self.merge(torch.cat(scan_outputs, dim=-1))


class MambaBlock(nn.Module):
    """Combine multi-order Mamba scans with a local feed-forward block."""

    def __init__(
        self,
        dim,
        mlp_ratio=2.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        gcn_k=5,
        use_checkpoint=True,
        scan_orders=("hilbert", "hilbert-reverse", "z"),
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(dim)
        self.attn = MultiScanMamba(d_model=dim, scan_orders=scan_orders)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = LocalGeometryFFN(in_features=dim, hidden_features=int(dim * mlp_ratio), gcn_k=gcn_k)

    def _forward_impl(self, x, center_xyz):
        x = x + self.drop_path(self.attn(self.norm1(x), center_xyz))
        x = x + self.drop_path(self.mlp(self.norm2(x), center_xyz))
        return x

    def forward(self, x, center_xyz):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, x, center_xyz, use_reentrant=False)
        return self._forward_impl(x, center_xyz)


class CrossAttentionBlock(nn.Module):  # Fuse query and condition features with cross-attention.

    def __init__(self, x_dim, y_dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, drop_path=0.1):
        super().__init__()
        self.bn1 = nn.LayerNorm(x_dim)
        self.bn2 = nn.LayerNorm(y_dim)
        self.internal_dim = out_dim
        self.query_proj = nn.LayerNorm(x_dim)
        self.kv_proj = nn.LayerNorm(y_dim)
        self.x_map = nn.Linear(x_dim, out_dim) if x_dim != out_dim else nn.Identity()
        self.y_map = nn.Linear(y_dim, out_dim) if y_dim != out_dim else nn.Identity()
        self.attn = nn.MultiheadAttention(out_dim, num_heads, dropout=attn_drop)
        # Keep residual paths deterministic without stochastic depth.
        self.drop_path = nn.Identity()
        self.bn3 = nn.LayerNorm(out_dim)
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x, y):
        x = self.bn1(x)
        y = self.bn2(y)
        x = self.x_map(self.query_proj(x))  # [B, N, out_dim]
        y = self.y_map(self.kv_proj(y))  # [B, M, out_dim]
        x_t = x.transpose(0, 1)
        y_t = y.transpose(0, 1)
        attn_out = self.attn(query=x_t, key=y_t, value=y_t)[0]
        # Add the attention residual in the projected feature space.
        x = x + self.drop_path(attn_out.transpose(0, 1))
        x = x + self.drop_path(self.ffn(self.bn3(x)))
        return x


class CrossMambaFusion(nn.Module):
    """Fuse 512-channel queries with flow tokens, then refine 256-channel features."""

    def __init__(self, x_dim, y_dim, out_dim=256, gcn_k=5, num_heads=4):
        super().__init__()
        self.crossformer_1 = CrossAttentionBlock(
            x_dim,
            y_dim,
            out_dim,
            num_heads=num_heads,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
        )
        self.mamba_self_attention = MambaBlock(
            out_dim,
            mlp_ratio=2.,
            gcn_k=gcn_k,
            scan_orders=("hilbert", "hilbert-reverse"),
        )

    def forward(self, feat_x, feat_y, center_xyz_x):
        feat_fused = self.crossformer_1(feat_x, feat_y)
        return self.mamba_self_attention(feat_fused, center_xyz_x)


class Stage1CrossMambaRefiner(nn.Module):
    """Generate the first 2x coordinate and feature branch.

    Forward inputs:
        coarse_xyz_cf: Sparse coordinates [B, 3, N].
        cond_local_feat: Local conditions [B, 256, N].
        cond_flow_feat: Flow query features [B, 128, M].

    Forward outputs:
        stage1_xyz_cf: First-stage coordinates [B, 3, 2N].
        stage1_feat: First-stage features [B, 2N, 256].
    """

    def __init__(self, gf_dim=512, up_factor=2, query_local_dim=256, cond_flow_dim=128, fusion_dim=256):
        super().__init__()
        if up_factor != 2:
            raise ValueError(f"Stage-1 refiner expects up_factor=2, got {up_factor}")
        self.up_factor = up_factor
        self.query_local_dim = query_local_dim
        self.cond_flow_dim = cond_flow_dim
        self.fusion_dim = fusion_dim

        self.pos_embed = RefinerPosEmbed(out_dim=query_local_dim)
        self.query_dim = query_local_dim * 2
        self.fusion = CrossMambaFusion(
            x_dim=self.query_dim,
            y_dim=self.cond_flow_dim,
            out_dim=self.fusion_dim,
        )
        self.child_feat_head = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.fusion_dim, self.fusion_dim),
        )
        self.coord_head = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.fusion_dim, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 3),
        )

    def forward(self, coarse_xyz_cf, cond_flow_feat, cond_local_feat):
        b, _, n_coarse = coarse_xyz_cf.shape
        coarse_xyz = coarse_xyz_cf[:, :3, :].transpose(2, 1).contiguous()  # [B, N, 3]

        if cond_local_feat.dim() != 3 or cond_local_feat.shape[1] != self.query_local_dim:
            raise RuntimeError(
                f"refiner local condition must be [B, {self.query_local_dim}, N], got {tuple(cond_local_feat.shape)}"
            )
        if cond_flow_feat.dim() != 3 or cond_flow_feat.shape[1] != self.cond_flow_dim:
            raise RuntimeError(
                f"refiner flow condition must be [B, {self.cond_flow_dim}, M], got {tuple(cond_flow_feat.shape)}"
            )
        if cond_local_feat.shape[-1] != n_coarse:
            raise RuntimeError(
                f"refiner local condition point count mismatch: cond={cond_local_feat.shape[-1]}, coarse={n_coarse}"
            )

        local_feat = cond_local_feat.transpose(2, 1).contiguous()  # [B, N, 256]
        pos_embed = self.pos_embed(coarse_xyz)  # [B, N, 256]
        query_feat = torch.cat([local_feat, pos_embed], dim=-1)  # [B, N, 512]
        cond_flow_tokens = cond_flow_feat.transpose(2, 1).contiguous()  # [B, M, 128]

        parent_feat = self.fusion(query_feat, cond_flow_tokens, coarse_xyz)  # [B, N, 256]
        child_feat = self.child_feat_head(parent_feat)  # [B, N, 256]
        child_xyz = coarse_xyz + self.coord_head(parent_feat)  # [B, N, 3]

        stage1_xyz = torch.cat([coarse_xyz, child_xyz], dim=1)  # [B, 2N, 3]
        stage1_feat = torch.cat([parent_feat, child_feat], dim=1)  # [B, 2N, 256]
        return stage1_xyz.transpose(1, 2).contiguous(), stage1_feat


class Stage2PureMambaRefiner(nn.Module):
    """Generate a second coordinate branch from first-stage features.

    Forward inputs:
        coarse_xyz_cf: First-stage coordinates [B, 3, 2N].
        coarse_feat: First-stage features [B, 2N, 256].

    Forward outputs:
        stage2_branch_xyz_cf: Branch coordinates [B, 3, 2N].
        stage2_branch_feat: Branch features [B, 2N, 256].
    """

    def __init__(self, gf_dim=512, up_factor=2, cond_flow_dim=128, cond_fused_dim=256, fusion_dim=256):
        super().__init__()
        if up_factor != 2:
            raise ValueError(f"Stage-2 refiner expects up_factor=2, got {up_factor}")
        self.up_factor = up_factor
        self.input_dim = cond_fused_dim
        self.fusion_dim = fusion_dim

        self.pos_embed = RefinerPosEmbed(out_dim=self.input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.fusion_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.fusion_dim, self.fusion_dim),
        )
        self.mamba_block = MambaBlock(
            dim=self.fusion_dim,
            mlp_ratio=2.,
            scan_orders=("z", "z-reverse"),
        )
        self.child_feat_head = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.fusion_dim, self.fusion_dim),
        )
        self.coord_head = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.fusion_dim, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 3),
        )

    def forward(self, coarse_xyz_cf, coarse_feat):
        coarse_xyz = coarse_xyz_cf[:, :3, :].transpose(1, 2).contiguous()  # [B, 2N, 3]
        if coarse_feat.dim() != 3 or coarse_feat.shape[-1] != self.input_dim:
            raise RuntimeError(
                f"stage2 coarse_feat must be [B, N, {self.input_dim}], got {tuple(coarse_feat.shape)}"
            )
        if coarse_feat.shape[1] != coarse_xyz.shape[1]:
            raise RuntimeError(
                f"stage2 coarse_feat/xyz point count mismatch: feat={coarse_feat.shape[1]}, xyz={coarse_xyz.shape[1]}"
            )

        pos_embed = self.pos_embed(coarse_xyz)  # [B, 2N, 256]
        feat = torch.cat([coarse_feat, pos_embed], dim=-1)  # [B, 2N, 512]
        feat = self.input_proj(feat)  # [B, 2N, 256]
        feat = self.mamba_block(feat, coarse_xyz)  # [B, 2N, 256]

        branch_feat = self.child_feat_head(feat)  # [B, 2N, 256]
        branch_xyz = coarse_xyz + self.coord_head(feat)  # [B, 2N, 3]
        return branch_xyz.transpose(1, 2).contiguous(), branch_feat


def query_knn_point(k, xyz, new_xyz):
    """Find K nearest reference points for each query.

    Args:
        k: Number of neighbors.
        xyz: Reference coordinates [B, N, 3].
        new_xyz: Query coordinates [B, S, 3].

    Returns:
        Neighbor indices [B, S, K].
    """
    dist = square_distance(new_xyz, xyz)
    _, group_idx = dist.topk(k, largest=False)
    return group_idx.int()


class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        self.af = nn.ReLU()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(self.af)
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)


class Transformer(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(Transformer, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, x, pos):
        """Apply local position-aware self-attention.

        Args:
            x: Point features [B, C_in, N].
            pos: Point coordinates [B, 3, N].

        Returns:
            Updated features [B, C_in, N].
        """

        # Aggregate local point features with position-aware attention.

        identity = x

        x = self.linear_start(x)
        b, dim, n = x.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()

        idx_knn = query_knn_point(self.n_knn, pos_flipped, pos_flipped)


        key = self.conv_key(x)

        value = self.conv_value(x)

        query = self.conv_query(x)

        key = grouping_operation(key, idx_knn)  # [B, C, N, K]

        qk_rel = query.reshape((b, -1, n, 1)) - key

        # Encode relative positions within each neighborhood.
        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # [B, 3, N, K]

        pos_embedding = self.pos_mlp(pos_rel)  # [B, C, N, K]

        attention = self.attn_mlp(qk_rel + pos_embedding)
        attention = torch.softmax(attention, -1)

        value = grouping_operation(value, idx_knn) + pos_embedding

        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # [B, C, N]
        y = self.linear_end(agg)

        return y+identity


class MambaGeometryDecoder(nn.Module):
    """Decode a 2x first-stage point cloud and a 4x final point cloud.

    The final prediction combines both 2x branches before coordinate refinement.
    """

    def __init__(self, up_factors=(2, 2), gf_dim=512, query_local_dim=256, cond_flow_dim=128, fusion_dim=256):
        super().__init__()
        self.up_factors = tuple(up_factors)
        if self.up_factors != (2, 2):
            raise ValueError(f"MambaGeometryDecoder now expects up_factors=(2, 2), got {self.up_factors}")

        self.MBA_Refiner_1 = Stage1CrossMambaRefiner(
            gf_dim=gf_dim,
            up_factor=self.up_factors[0],
            query_local_dim=query_local_dim,
            cond_flow_dim=cond_flow_dim,
            fusion_dim=fusion_dim,
        )
        self.MBA_Refiner_2 = Stage2PureMambaRefiner(
            gf_dim=gf_dim,
            up_factor=self.up_factors[1],
            cond_flow_dim=cond_flow_dim,
            cond_fused_dim=fusion_dim,
            fusion_dim=fusion_dim,
        )
        self.final_pos_embed = RefinerPosEmbed(out_dim=fusion_dim)
        self.final_refine = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim, 3),
        )

    def forward(self, coarse_xyz_cf, cond_flow_feat, cond_local_feat):
        stage1_xyz_cf, stage1_feat = self.MBA_Refiner_1(coarse_xyz_cf, cond_flow_feat, cond_local_feat)
        stage2_branch_xyz_cf, stage2_branch_feat = self.MBA_Refiner_2(stage1_xyz_cf, stage1_feat)

        stage1_xyz = stage1_xyz_cf.transpose(1, 2).contiguous()  # [B, 2N, 3]
        stage2_branch_xyz = stage2_branch_xyz_cf.transpose(1, 2).contiguous()  # [B, 2N, 3]

        candidate_xyz = torch.cat([stage1_xyz, stage2_branch_xyz], dim=1)  # [B, 4N, 3]
        candidate_feat = torch.cat([stage1_feat, stage2_branch_feat], dim=1)  # [B, 4N, 256]
        refine_in = torch.cat([candidate_feat, self.final_pos_embed(candidate_xyz)], dim=-1)  # [B, 4N, 512]
        final_xyz = candidate_xyz + torch.tanh(self.final_refine(refine_in))  # [B, 4N, 3]

        return [stage1_xyz, final_xyz]
