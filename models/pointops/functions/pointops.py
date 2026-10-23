"""CUDA point sampling and neighborhood-query wrappers."""

from typing import Tuple
import ctypes
import glob
import importlib.util
import os
import sys
import torch
from torch.autograd import Function


def _preload_torch_shared_libs():
    torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
    if not os.path.isdir(torch_lib_dir):
        return
    candidates = (
        "libc10.so",
        "libtorch.so",
        "libtorch_cpu.so",
        "libtorch_python.so",
        "libc10_cuda.so",
        "libtorch_cuda.so",
    )
    for name in candidates:
        path = os.path.join(torch_lib_dir, name)
        if os.path.isfile(path):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue


def _load_prebuilt_pointops_cuda():
    _preload_torch_shared_libs()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pattern_candidates = (
        os.path.join(base_dir, "src", "pointops_cuda*.so"),
        os.path.join(base_dir, "build", "lib.*", "pointops_cuda*.so"),
    )
    module_name = "pointops_cuda"
    for pattern in pattern_candidates:
        for path in sorted(glob.glob(pattern)):
            try:
                spec = importlib.util.spec_from_file_location(module_name, path)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                return module
            except Exception:
                sys.modules.pop(module_name, None)
                continue
    raise ImportError("Unable to load a prebuilt pointops_cuda extension.")


try:
    import pointops_cuda
except ImportError:
    import warnings
    from torch.utils.cpp_extension import load
    try:
        pointops_cuda = _load_prebuilt_pointops_cuda()
        warnings.warn("Loaded bundled pointops_cuda extension after resolving shared-library dependencies.")
    except ImportError:
        warnings.warn("Unable to load pointops_cuda binary extension; falling back to local JIT build.")
        pointops_cuda_src = os.path.join(os.path.dirname(__file__), "../src")
        pointops_cuda = load('pointops_cuda', [
            pointops_cuda_src + '/pointops_api.cpp',
            pointops_cuda_src + '/ballquery/ballquery_cuda.cpp',
            pointops_cuda_src + '/ballquery/ballquery_cuda_kernel.cu',
            pointops_cuda_src + '/knnquery_heap/knnquery_heap_cuda.cpp',
            pointops_cuda_src + '/knnquery_heap/knnquery_heap_cuda_kernel.cu',
            pointops_cuda_src + '/sampling/sampling_cuda.cpp',
            pointops_cuda_src + '/sampling/sampling_cuda_kernel.cu',
        ], build_directory=pointops_cuda_src, verbose=False)


def _same_device_as(tensor: torch.Tensor) -> torch.device:
    return tensor.device


class FurthestSampling(Function):
    @staticmethod
    def forward(ctx, xyz, m):
        """Select point indices by farthest-point sampling.

        Args:
            xyz: Contiguous coordinates [B, N, 3].
            m: Number of sampled points.

        Returns:
            Sampled indices [B, M] with dtype int32.
        """
        assert xyz.is_contiguous()
        b, n, _ = xyz.size()
        device = _same_device_as(xyz)
        idx = torch.empty(b, m, dtype=torch.int32, device=device)
        temp = torch.full((b, n), 1e10, dtype=torch.float32, device=device)
        pointops_cuda.furthestsampling_cuda(b, n, m, xyz, temp, idx)
        return idx

    @staticmethod
    def backward(xyz, a=None):
        return None, None

furthestsampling = FurthestSampling.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        """Find reference points within a radius of each query center.

        Args:
            radius: Query radius.
            nsample: Maximum number of neighbors per query.
            xyz: Reference coordinates [B, N, 3].
            new_xyz: Query centers [B, M, 3].

        Returns:
            Neighbor indices [B, M, nsample] with dtype int64.
        """
        xyz = xyz.contiguous()
        new_xyz = new_xyz.contiguous()
        b, n, _ = xyz.size()
        m = new_xyz.size(1)
        idx = torch.zeros(b, m, nsample, dtype=torch.int32, device=_same_device_as(xyz))
        pointops_cuda.ballquery_cuda(b, n, m, radius, nsample, new_xyz, xyz, idx)
        return idx.long()

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None

ballquery = BallQuery.apply


class KNNQuery_Heap(Function):
    @staticmethod
    def forward(ctx, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """Find nearest-neighbor indices with a heap-based CUDA query.

        Args:
            nsample: Number of neighbors per query.
            xyz: Contiguous reference coordinates [B, N, 3].
            new_xyz: Contiguous query coordinates [B, M, 3]; defaults to xyz.

        Returns:
            Neighbor indices [B, M, nsample] with dtype int32.
            Squared distances are computed internally but are not returned.
        """
        if new_xyz is None:
            new_xyz = xyz
        assert xyz.is_contiguous()
        assert new_xyz.is_contiguous()
        b, m, _ = new_xyz.size()
        n = xyz.size(1)
        device = _same_device_as(xyz)
        idx = torch.zeros(b, m, nsample, dtype=torch.int32, device=device)
        dist2 = torch.zeros(b, m, nsample, dtype=torch.float32, device=device)
        pointops_cuda.knnquery_heap_cuda(b, n, m, nsample, xyz, new_xyz, idx, dist2)
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None

knnquery_heap = KNNQuery_Heap.apply
