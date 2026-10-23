"""Model loading, evaluation metrics, batch adaptation and report export."""

import math
import json
import tempfile
from pathlib import Path
import os
import shutil
import types
import importlib.util
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from models.controlflow3d import ControlFlow3D

o3d = None
HAS_O3D = importlib.util.find_spec('open3d') is not None


def ensure_open3d():
    global o3d
    if not HAS_O3D:
        raise RuntimeError('open3d is required but not installed.')
    if o3d is None:
        import open3d as _open3d
        o3d = _open3d
    return o3d


def infer_num_workers():
    cpu_count = os.cpu_count() or 8
    return max(2, min(8, cpu_count // 2))


def maybe_cuda_sync(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def to_float(value):
    return float(value) if not isinstance(value, float) else value


def denormalize_points(points: torch.Tensor, centroid: torch.Tensor, furthest_distance: torch.Tensor) -> torch.Tensor:
    return points.float() * furthest_distance.float() + centroid.float()


def compute_cd_loss_batch(d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
    if d1.ndim != 2 or d2.ndim != 2:
        raise RuntimeError(f'Chamfer distances must be rank-2 tensors, got d1={d1.shape}, d2={d2.shape}')
    if d1.shape[0] != d2.shape[0]:
        raise RuntimeError(f'Chamfer batch sizes differ: d1={d1.shape}, d2={d2.shape}')
    return torch.mean(d1, dim=1) + torch.mean(d2, dim=1)


def normalize_points_for_metric_batch(points: torch.Tensor) -> torch.Tensor:
    points = points.float().contiguous()
    if points.ndim != 3 or points.shape[-1] != 3:
        raise RuntimeError(f'points must be [B, N, 3], got {tuple(points.shape)}')
    centroid = torch.mean(points, dim=1, keepdim=True)
    centered = points - centroid
    furthest_distance = torch.amax(torch.linalg.norm(centered, dim=2), dim=1, keepdim=True).unsqueeze(-1)
    furthest_distance = torch.clamp(furthest_distance, min=1e-8)
    return centered / furthest_distance


def compute_cd_hd_from_points(chamfer_fn, pred_points: torch.Tensor, gt_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pred_norm = normalize_points_for_metric_batch(pred_points)
    gt_norm = normalize_points_for_metric_batch(gt_points)
    d1, d2, _, _ = chamfer_fn(pred_norm, gt_norm)
    return compute_cd_loss_batch(d1, d2), compute_hd_from_squared_chamfer(d1, d2)


def save_ply(points: np.ndarray, filename: str):
    if not HAS_O3D:
        raise RuntimeError('open3d is required for PLY export.')
    open3d = ensure_open3d()
    if points.ndim == 2 and points.shape[0] == 3 and points.shape[1] > 3:
        points = points.T
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points.astype(np.float64))
    open3d.io.write_point_cloud(filename, pcd)


def parse_mesh_extensions(mesh_extensions: str) -> List[str]:
    exts = []
    for item in mesh_extensions.split(','):
        ext = item.strip().lower()
        if not ext:
            continue
        if not ext.startswith('.'):
            ext = f'.{ext}'
        exts.append(ext)
    return exts or ['.obj', '.off', '.ply']


def build_model_for_test(
    checkpoint_path: str,
    device: torch.device,
    steps: int,
    n_parts: int = 8,
    x0_mode: str = 'feature',
    x0_noise_scale: float = 1.0,
    wrap_data_parallel: bool = True,
):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_kwargs = dict(ckpt.get('model_kwargs', {}))
    required_model_keys = (
        'flow_feature_dim',
        'condition_feature_dim',
        'image_condition_dim',
        'image_cond_max_tokens_per_view',
        'num_prototypes',
        'coarse_num_prototypes',
        'prototype_dim',
        'prototype_kmeans_iters',
        'prototype_assign_temperature',
        'use_image_controller',
        'prototype_consistency_weight',
    )
    missing_keys = [key for key in required_model_keys if key not in model_kwargs]
    if missing_keys:
        raise RuntimeError(
            'Checkpoint is missing model_kwargs fields required by ControlFlow3D, '
            f'cannot load with the current structure: missing={missing_keys}'
        )
    model_kwargs['n_parts'] = int(n_parts)
    model = ControlFlow3D(**model_kwargs).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()

    if wrap_data_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model_core = model.module if isinstance(model, nn.DataParallel) else model
    sync_runtime_inference_params(model_core, steps)
    configure_runtime_x0_mode(model_core, x0_mode=x0_mode, x0_noise_scale=x0_noise_scale)
    return model, model_core, ckpt


def sync_runtime_inference_params(model_core, steps: int):
    model_core.inference_steps = int(steps)
    model_core.use_partwise_inference = True
    model_core.runtime_x0_mode = 'feature'
    model_core.runtime_x0_noise_scale = 1.0


def configure_runtime_x0_mode(model_core, x0_mode: str, x0_noise_scale: float):
    model_core.runtime_x0_mode = str(x0_mode)
    model_core.runtime_x0_noise_scale = float(x0_noise_scale)
    if x0_mode != 'pure_noise':
        return

    def _build_x0_pure_noise(self, token_xyz, token_feat, target_xyz, add_noise=True, noise=None):
        batch_size = target_xyz.shape[0]
        target_n = target_xyz.shape[2]
        if noise is None:
            src_feat = torch.randn(
                batch_size,
                self.flow_feature_dim,
                target_n,
                device=target_xyz.device,
                dtype=target_xyz.dtype,
            )
        else:
            src_feat = noise.to(device=target_xyz.device, dtype=target_xyz.dtype)
            if src_feat.shape[2] != target_n:
                src_feat = self._three_interp(target_xyz, token_xyz, src_feat)
        return float(x0_noise_scale) * src_feat

    model_core._build_x0 = types.MethodType(_build_x0_pure_noise, model_core)


def get_sample_dir(sample: Dict) -> str:
    gt_path = str(sample.get('gt_path', ''))
    if not gt_path:
        return ''
    return os.path.dirname(os.path.dirname(gt_path))


def _strip_part_suffix(value: str, marker: str = '__part') -> str:
    text = str(value)
    index = text.find(marker)
    if index > 0:
        return text[:index]
    return text


def normalize_sample_identifiers(category: str, sample_id: str) -> Tuple[str, str]:
    return _strip_part_suffix(category, marker='__part'), _strip_part_suffix(sample_id, marker='__part')


def build_eval_sample_stub(
    dataset_root: str,
    split: str,
    category: str,
    sample_id: str,
) -> Dict[str, str]:
    category, sample_id = normalize_sample_identifiers(category, sample_id)
    sample_dir = os.path.join(os.path.abspath(dataset_root), str(split), str(category), str(sample_id))
    gt_dir = os.path.join(sample_dir, 'gt')
    gt_candidates = [
        os.path.join(gt_dir, 'gt_xyz.npy'),
        os.path.join(gt_dir, 'gt_xyz.ply'),
        os.path.join(gt_dir, 'upsampled_point_cloud.npy'),
        os.path.join(gt_dir, 'upsampled_point_cloud.ply'),
    ]

    gt_path = ''
    for candidate in gt_candidates:
        if os.path.isfile(candidate):
            gt_path = candidate
            break
    if not gt_path:
        gt_path = gt_candidates[0]

    return {
        'category': category,
        'sample_id': sample_id,
        'gt_path': gt_path,
    }


def find_mesh_path(sample: Dict, mesh_root: str = '', mesh_extensions: str = 'obj,off,ply') -> str:
    exts = parse_mesh_extensions(mesh_extensions)
    category, sample_id = normalize_sample_identifiers(str(sample['category']), str(sample['sample_id']))
    sample_dir = get_sample_dir(sample)

    candidate_roots = [sample_dir]
    if mesh_root:
        mesh_root = os.path.abspath(mesh_root)
        candidate_roots.extend(
            [
                os.path.join(mesh_root, category, sample_id),
                os.path.join(mesh_root, category),
                mesh_root,
            ]
        )

    patterns = [
        os.path.join('gt', 'gt_mesh{ext}'),
        os.path.join('gt', 'mesh{ext}'),
        os.path.join('mesh', 'model_normalized{ext}'),
        os.path.join('models', 'model_normalized{ext}'),
        os.path.join('models', 'model{ext}'),
        os.path.join('{sample_id}{ext}'),
        os.path.join('model_normalized{ext}'),
        os.path.join('model{ext}'),
    ]

    checked = set()
    for root in candidate_roots:
        for pattern in patterns:
            for ext in exts:
                path = os.path.join(root, pattern.format(ext=ext, sample_id=sample_id))
                if path in checked:
                    continue
                checked.add(path)
                if os.path.isfile(path):
                    return path
    return ''


def resolve_mesh_path_for_sample(
    dataset_root: str,
    split: str,
    category: str,
    sample_id: str,
    mesh_root: str = '',
    mesh_extensions: str = 'obj,off,ply',
    copy_to_gt: bool = True,
    copied_mesh_basename: str = 'gt_mesh',
) -> Tuple[str, str]:
    sample = build_eval_sample_stub(
        dataset_root=dataset_root,
        split=split,
        category=category,
        sample_id=sample_id,
    )

    local_mesh_path = find_mesh_path(sample, mesh_root='', mesh_extensions=mesh_extensions)
    if local_mesh_path:
        return local_mesh_path, 'mesh_local'

    source_mesh_path = find_mesh_path(sample, mesh_root=mesh_root, mesh_extensions=mesh_extensions)
    if not source_mesh_path:
        return '', 'mesh_not_found'

    if not copy_to_gt:
        return source_mesh_path, f'mesh_source:{source_mesh_path}'

    gt_dir = os.path.dirname(str(sample['gt_path']))
    os.makedirs(gt_dir, exist_ok=True)
    mesh_ext = os.path.splitext(source_mesh_path)[1].lower() or '.obj'
    copied_mesh_path = os.path.join(gt_dir, f'{copied_mesh_basename}{mesh_ext}')

    if os.path.isfile(copied_mesh_path):
        return copied_mesh_path, f'mesh_cached:{copied_mesh_path}'

    try:
        shutil.copy2(source_mesh_path, copied_mesh_path)
        return copied_mesh_path, f'mesh_copied:{source_mesh_path}->{copied_mesh_path}'
    except Exception as exc:
        return source_mesh_path, f'mesh_copy_failed:{exc}'


def compute_hd_from_squared_chamfer(d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
    hd_pred = torch.max(d1, dim=1).values
    hd_gt = torch.max(d2, dim=1).values
    return hd_pred + hd_gt


def compute_point_to_mesh_distances(pred_points: np.ndarray, mesh_path: str) -> np.ndarray:
    if not HAS_O3D:
        raise RuntimeError('open3d is required for mesh-based P2F.')
    open3d = ensure_open3d()
    if not hasattr(open3d, 't') or not hasattr(open3d.t.geometry, 'RaycastingScene'):
        raise RuntimeError('open3d.t.geometry.RaycastingScene is unavailable in the current open3d build.')

    if hasattr(open3d.utility, 'get_verbosity_level') and hasattr(open3d.utility, 'set_verbosity_level'):
        prev_level = open3d.utility.get_verbosity_level()
        try:
            open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Error)
            mesh = open3d.io.read_triangle_mesh(mesh_path)
            if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
                raise RuntimeError(f'mesh has no valid triangles: {mesh_path}')

            # Remove invalid topology before constructing the raycasting mesh.
            mesh.remove_duplicated_vertices()
            mesh.remove_duplicated_triangles()
            mesh.remove_degenerate_triangles()
            mesh.remove_unreferenced_vertices()
            if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
                raise RuntimeError(f'mesh became invalid after cleanup: {mesh_path}')

            tmesh = open3d.t.geometry.TriangleMesh.from_legacy(mesh)
        finally:
            open3d.utility.set_verbosity_level(prev_level)
    else:
        mesh = open3d.io.read_triangle_mesh(mesh_path)
        if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            raise RuntimeError(f'mesh has no valid triangles: {mesh_path}')
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            raise RuntimeError(f'mesh became invalid after cleanup: {mesh_path}')
        tmesh = open3d.t.geometry.TriangleMesh.from_legacy(mesh)

    scene = open3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)
    query = open3d.core.Tensor(np.asarray(pred_points, dtype=np.float32))
    distances = scene.compute_distance(query).numpy()
    return np.asarray(distances, dtype=np.float32)


def compute_p2f_metrics_for_sample(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    mesh_path: str = '',
    p2f_mode: str = 'mesh',
) -> Tuple[float, float, str, str]:
    pred_points = np.asarray(pred_points, dtype=np.float32)
    del gt_points
    if p2f_mode != 'mesh':
        raise RuntimeError(f'Only mesh-based P2F is supported, got p2f_mode={p2f_mode!r}')
    if not mesh_path:
        return math.nan, math.nan, 'missing', 'mesh_not_found'
    try:
        mesh_dist = compute_point_to_mesh_distances(pred_points, mesh_path)
        return float(np.mean(mesh_dist)), float(np.std(mesh_dist)), 'mesh', mesh_path
    except Exception as exc:
        return math.nan, math.nan, 'missing', f'mesh_failed:{exc}'


def adapt_inference_batch(batch: Sequence[Any], use_image_condition: bool) -> Dict[str, Any]:
    if not isinstance(batch, (list, tuple)):
        raise RuntimeError(f'batch type must be list/tuple, got {type(batch)}')

    image_cond = None
    core_batch = batch

    if use_image_condition:
        if len(batch) < 10:
            raise RuntimeError('After --use_image_condition is enabled, the dataset must return an image_cond tensor.')
        candidate = batch[-1]
        if not torch.is_tensor(candidate) or candidate.dim() < 2:
            raise RuntimeError('After --use_image_condition is enabled, the last item of the batch must be an image_cond tensor.')
        image_cond = candidate
        core_batch = batch[:-1]

    if len(core_batch) < 8:
        raise RuntimeError(f'Inference batch must contain at least 8 items (+ optional 1 image_cond item), got {len(core_batch)} items.')

    sparse_xyz, sparse_feat, gt_feat, gt_xyz = core_batch[:4]
    rest = list(core_batch[4:])

    sparse_patch_idx = None
    gt_patch_idx = None
    if rest and torch.is_tensor(rest[0]) and rest[0].dtype in (
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
    ):
        sparse_patch_idx = rest.pop(0)
    if rest and torch.is_tensor(rest[0]) and rest[0].dtype in (
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
    ):
        gt_patch_idx = rest.pop(0)

    if len(rest) != 4:
        raise RuntimeError(f'Inference batch tail must contain 4 items (centroid, furthest, sample_ids, categories), got {len(rest)} items.')

    centroid, furthest_distance, sample_ids, categories = rest

    return {
        'sparse_xyz': sparse_xyz,
        'sparse_feat': sparse_feat,
        'gt_feat': gt_feat,
        'gt_xyz': gt_xyz,
        'centroid': centroid,
        'furthest_distance': furthest_distance,
        'sample_ids': sample_ids,
        'categories': categories,
        'image_cond': image_cond,
        'sparse_patch_idx': sparse_patch_idx,
        'gt_patch_idx': gt_patch_idx,
    }


def write_evaluation_report(directory, summary, per_category, per_sample, run_args):
    directory = Path(directory)
    # Validate all values before replacing any previous report file.
    documents = {
        'summary.json': json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
        'per_category.json': json.dumps(per_category, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
        'run_args.json': json.dumps(run_args, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
        'per_sample.jsonl': ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
                                   for row in per_sample),
    }
    directory.mkdir(parents=True, exist_ok=True)
    for name, contents in documents.items():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=directory,
                                             prefix=f'.{name}.', delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(contents)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, directory / name)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def validate_evaluation_samples(entries, expected_samples):
    """Require one saved artifact for every selected dataset sample."""
    expected = [(str(sample['category']), str(sample['sample_id'])) for sample in expected_samples]
    expected_set = set(expected)
    if len(expected_set) != len(expected):
        raise RuntimeError('Evaluation dataset contains duplicate category/sample IDs.')
    actual = {(str(category), str(sample_id)) for category, sample_id, _ in entries}
    missing = sorted(expected_set - actual)
    unexpected = sorted(actual - expected_set)
    if missing or unexpected:
        raise RuntimeError(
            f'Evaluation artifacts do not match the selected dataset: '
            f'expected={len(expected_set)}, actual={len(actual)}, '
            f'missing={len(missing)}, unexpected={len(unexpected)}. '
            f'First missing IDs: {missing[:8]}; first unexpected IDs: {unexpected[:8]}.'
        )
