"""Evaluate ControlFlow3D checkpoints on a dataset split."""

import argparse
import datetime
import json
import math
import os
import time
import warnings
from collections import defaultdict

from utils.runtime_utils import preset_cuda_visible_devices_from_argv

warnings.filterwarnings(
    'ignore',
    message=r".*param_schemas.*deprecated.*op_signature.*",
    category=FutureWarning,
)


from utils.dataset_feature_layout import (
    DEFAULT_IMAGE_COND_FILENAME, DEFAULT_IMAGE_COND_KEY,
    DEFAULT_IMAGE_COND_ARRAY_KEY, DEFAULT_IMAGE_COND_DIM,
)
DEFAULT_TEST_X0_MODE = 'feature'
DEFAULT_TEST_X0_NOISE_SCALE = 1.0
warnings.filterwarnings(
    'ignore',
    message=r"Importing from timm\.models\.layers is deprecated, please import via timm\.layers",
    category=FutureWarning,
)
warnings.filterwarnings(
    'ignore',
    message=r"The pynvml package is deprecated.*",
    category=FutureWarning,
)


def _early_log(message: str):
    """Report startup on the main rank, excluding loader worker imports."""
    if __name__ == '__main__' and os.environ.get('RANK', '0') == '0':
        print(message, flush=True)


def resolve_image_cond_dim(requested_dim):
    requested_dim = int(requested_dim)
    if requested_dim > 0:
        return requested_dim
    return DEFAULT_IMAGE_COND_DIM


# Set GPU visibility before importing PyTorch.
# Preserve environment visibility when --gpu_ids is absent.
preset_cuda_visible_devices_from_argv()

_early_log('[test] Loading dependencies...')
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from datasetsource.multimodal_pointcloud_dataset import MultimodalPointCloudDataset
from extensions.chamfer_dist import chamfer_3DDist
from utils.eval_utils import (
    adapt_inference_batch, write_evaluation_report, validate_evaluation_samples,
    HAS_O3D,
    build_model_for_test,
    compute_cd_hd_from_points,
    compute_p2f_metrics_for_sample,
    infer_num_workers,
    maybe_cuda_sync,
    resolve_mesh_path_for_sample,
    save_ply,
)


def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()


def maybe_destroy_process_group():
    if is_dist_initialized():
        dist.destroy_process_group()


def setup_distributed():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    distributed = world_size > 1

    if distributed and not torch.cuda.is_available():
        raise RuntimeError('DDP inference requires CUDA.')

    if torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device('cuda', local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device, local_rank, rank, world_size, distributed


def broadcast_string(value: str, rank: int, distributed: bool):
    if not distributed:
        return value
    payload = [value if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def gather_object_list(local_value, world_size: int):
    if world_size == 1:
        return [local_value]
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_value)
    return gathered


def flatten_gathered_rows(gathered_rows):
    merged = []
    for rows in gathered_rows:
        if not rows:
            continue
        merged.extend(rows)
    return merged


def round_nested_floats(value, ndigits=9):
    if isinstance(value, float):
        return round(value, ndigits)
    if isinstance(value, dict):
        return {key: round_nested_floats(sub_value, ndigits=ndigits) for key, sub_value in value.items()}
    if isinstance(value, list):
        return [round_nested_floats(item, ndigits=ndigits) for item in value]
    if isinstance(value, tuple):
        return tuple(round_nested_floats(item, ndigits=ndigits) for item in value)
    return value


def infer_sparse_only_partwise(model, sparse_xyz, sparse_feat, full_num_parts):
    if sparse_xyz.shape[1] % full_num_parts != 0:
        raise RuntimeError(
            f'part inference requires sparse_n={sparse_xyz.shape[1]} to be divisible by full_num_parts={full_num_parts}.'
        )
    if sparse_feat.shape[1] != sparse_xyz.shape[1]:
        raise RuntimeError(
            f'sparse_feat must be aligned with sparse_xyz, got feat_n={sparse_feat.shape[1]}, xyz_n={sparse_xyz.shape[1]}'
        )

    batch_size = sparse_xyz.shape[0]
    sparse_part_n = sparse_xyz.shape[1] // full_num_parts
    sparse_xyz_parts = sparse_xyz.view(batch_size, full_num_parts, sparse_part_n, 3).reshape(
        batch_size * full_num_parts,
        sparse_part_n,
        3,
    ).contiguous()
    sparse_feat_parts = sparse_feat.view(batch_size, full_num_parts, sparse_part_n, sparse_feat.shape[2]).reshape(
        batch_size * full_num_parts,
        sparse_part_n,
        sparse_feat.shape[2],
    ).contiguous()
    pred_parts = model(sparse_xyz_parts, sparse_feat_parts, category_ids=None, train=False)
    return pred_parts.view(batch_size, full_num_parts, pred_parts.shape[1], 3).reshape(batch_size, -1, 3).contiguous()


def infer_sparse_only_partwise_conditioned(rollout_fn, sparse_xyz, sparse_feat, full_num_parts, category_ids=None, image_cond=None):
    if sparse_xyz.shape[1] % full_num_parts != 0:
        raise RuntimeError(
            f'part inference requires sparse_n={sparse_xyz.shape[1]} to be divisible by full_num_parts={full_num_parts}.'
        )
    if sparse_feat.shape[1] != sparse_xyz.shape[1]:
        raise RuntimeError(
            f'sparse_feat must be aligned with sparse_xyz, got feat_n={sparse_feat.shape[1]}, xyz_n={sparse_xyz.shape[1]}'
        )

    batch_size = sparse_xyz.shape[0]
    sparse_part_n = sparse_xyz.shape[1] // full_num_parts
    sparse_xyz_parts = sparse_xyz.view(batch_size, full_num_parts, sparse_part_n, 3).reshape(
        batch_size * full_num_parts,
        sparse_part_n,
        3,
    ).contiguous()
    sparse_feat_parts = sparse_feat.view(batch_size, full_num_parts, sparse_part_n, sparse_feat.shape[2]).reshape(
        batch_size * full_num_parts,
        sparse_part_n,
        sparse_feat.shape[2],
    ).contiguous()

    part_category_ids = None
    if category_ids is not None:
        part_category_ids = category_ids.repeat_interleave(full_num_parts, dim=0)

    part_image_cond = None
    if image_cond is not None:
        part_image_cond = image_cond.repeat_interleave(full_num_parts, dim=0)

    pred_parts = rollout_fn(
        sparse_xyz_parts,
        sparse_feat_parts,
        part_category_ids,
        part_image_cond,
    )
    return pred_parts.view(batch_size, full_num_parts, pred_parts.shape[1], 3).reshape(batch_size, -1, 3).contiguous()


def aggregate_category_rows(per_sample_rows):
    grouped = defaultdict(list)
    for row in per_sample_rows:
        grouped[str(row['category'])].append(row)

    per_category_summary = {}
    for cat, rows in sorted(grouped.items()):
        count = len(rows)
        p2f_values = [float(row['p2f']) for row in rows if np.isfinite(row['p2f'])]
        p2f_std_values = [float(row['p2f_std']) for row in rows if np.isfinite(row['p2f_std'])]
        top3_samples = sorted(rows, key=lambda row: row['loss'])[:3]
        per_category_summary[cat] = {
            'samples': int(count),
            'avg_loss': float(np.mean([row['loss'] for row in rows])) if rows else math.nan,
            'avg_hd': float(np.mean([row['hd'] for row in rows])) if rows else math.nan,
            'avg_p2f': float(np.mean(p2f_values)) if p2f_values else None,
            'avg_p2f_std': float(np.mean(p2f_std_values)) if p2f_std_values else None,
            'p2f_valid_samples': int(len(p2f_values)),
            'p2f_mesh_samples': int(sum(1 for row in rows if row['p2f_mode'] == 'mesh')),
            'p2f_missing_samples': int(sum(1 for row in rows if row['p2f_mode'] == 'missing')),
            'avg_infer_time_sec_per_sample': float(np.mean([row['infer_time_sec'] for row in rows])) if rows else math.nan,
            'avg_infer_time_ms_per_sample': float(np.mean([row['infer_time_sec'] for row in rows]) * 1000.0) if rows else math.nan,
            'best_3_samples': top3_samples,
        }
    return per_category_summary


def get_artifact_path(artifact_root: str, category: str, sample_id: str) -> str:
    return os.path.join(artifact_root, str(category), str(sample_id), 'artifact.npz')


def save_inference_artifact(
    artifact_root: str,
    category: str,
    sample_id: str,
    sparse_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    pred_xyz: np.ndarray,
    centroid: np.ndarray,
    furthest_distance: np.ndarray,
    infer_time_sec: float,
    rank: int,
):
    artifact_path = get_artifact_path(artifact_root, category, sample_id)
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    np.savez(
        artifact_path,
        sparse_xyz=np.asarray(sparse_xyz, dtype=np.float32),
        gt_xyz=np.asarray(gt_xyz, dtype=np.float32),
        pred_xyz=np.asarray(pred_xyz, dtype=np.float32),
        centroid=np.asarray(centroid, dtype=np.float32),
        furthest_distance=np.asarray(furthest_distance, dtype=np.float32),
        infer_time_sec=np.asarray([infer_time_sec], dtype=np.float32),
        rank=np.asarray([rank], dtype=np.int32),
    )


def collect_artifact_entries(artifact_root: str):
    entries = []
    for category in sorted(os.listdir(artifact_root)):
        cat_dir = os.path.join(artifact_root, category)
        if not os.path.isdir(cat_dir):
            continue
        for sample_id in sorted(os.listdir(cat_dir)):
            sample_dir = os.path.join(cat_dir, sample_id)
            artifact_path = os.path.join(sample_dir, 'artifact.npz')
            if os.path.isfile(artifact_path):
                entries.append((str(category), str(sample_id), artifact_path))
    return entries


def evaluate_saved_artifacts(
    artifact_root: str,
    save_dir_abs: str,
    args,
    device: torch.device,
    distributed: bool,
    rank: int,
    world_size: int,
    expected_samples=None,
):
    entries = collect_artifact_entries(artifact_root)
    if expected_samples is not None:
        validate_evaluation_samples(entries, expected_samples)
    if not entries:
        raise RuntimeError(f'No inference artifacts found under {artifact_root}')

    local_entries = entries[rank::world_size] if distributed else entries

    chamfer_fn = chamfer_3DDist()
    local_rows = []
    local_total_count = 0
    local_total_loss = 0.0
    local_total_hd = 0.0
    local_total_p2f = 0.0
    local_total_p2f_std = 0.0
    local_total_p2f_count = 0
    local_total_p2f_mesh_count = 0
    local_total_p2f_missing_count = 0
    local_total_infer_time_sec = 0.0

    iterator = tqdm(local_entries, desc=f'Eval-Metrics-rank{rank}' if distributed else 'Eval-Metrics', disable=distributed and rank != 0)
    for cat, sid, artifact_path in iterator:
        with np.load(artifact_path, allow_pickle=False) as data:
            sparse_xyz = np.asarray(data['sparse_xyz'], dtype=np.float32)
            gt_xyz = np.asarray(data['gt_xyz'], dtype=np.float32)
            pred_xyz = np.asarray(data['pred_xyz'], dtype=np.float32)
            centroid = np.asarray(data['centroid'], dtype=np.float32)
            furthest_distance = np.asarray(data['furthest_distance'], dtype=np.float32)
            infer_time_sec = float(np.asarray(data['infer_time_sec']).reshape(-1)[0])

        sparse_raw = sparse_xyz * furthest_distance + centroid
        gt_raw = gt_xyz * furthest_distance + centroid
        pred_raw = pred_xyz * furthest_distance + centroid

        pred_t = torch.from_numpy(pred_xyz).unsqueeze(0).to(device=device, dtype=torch.float32)
        gt_t = torch.from_numpy(gt_xyz).unsqueeze(0).to(device=device, dtype=torch.float32)
        sample_global_cd_batch, sample_hd_batch = compute_cd_hd_from_points(chamfer_fn, pred_t, gt_t)
        sample_global_cd = float(sample_global_cd_batch.item())
        sample_hd = float(sample_hd_batch.item())
        sample_mesh_path, sample_mesh_note = resolve_mesh_path_for_sample(
            dataset_root=args.dataset_root,
            split=args.split,
            category=cat,
            sample_id=sid,
            mesh_root=args.mesh_root,
            mesh_extensions=args.mesh_extensions,
            copy_to_gt=args.copy_mesh_to_gt,
            copied_mesh_basename=args.copied_mesh_basename,
        )
        sample_p2f, sample_p2f_std, sample_p2f_mode, sample_p2f_note = compute_p2f_metrics_for_sample(
            pred_raw,
            gt_raw,
            mesh_path=sample_mesh_path,
        )

        local_total_count += 1
        local_total_loss += sample_global_cd
        local_total_hd += sample_hd
        local_total_infer_time_sec += infer_time_sec

        if np.isfinite(sample_p2f):
            local_total_p2f += float(sample_p2f)
            local_total_p2f_std += float(sample_p2f_std)
            local_total_p2f_count += 1
        if sample_p2f_mode == 'mesh':
            local_total_p2f_mesh_count += 1
        if sample_p2f_mode == 'missing':
            local_total_p2f_missing_count += 1

        local_rows.append({
            'sample_id': sid,
            'category': cat,
            'loss': sample_global_cd,
            'hd': sample_hd,
            'p2f': float(sample_p2f) if np.isfinite(sample_p2f) else math.nan,
            'p2f_std': float(sample_p2f_std) if np.isfinite(sample_p2f_std) else math.nan,
            'p2f_mode': sample_p2f_mode,
            'p2f_note': sample_p2f_note,
            'mesh_path': sample_mesh_path,
            'mesh_note': sample_mesh_note,
            'infer_time_sec': infer_time_sec,
        })

        if not args.no_save_ply:
            sample_dir = os.path.join(save_dir_abs, cat, sid)
            os.makedirs(sample_dir, exist_ok=True)
            save_ply(sparse_raw, os.path.join(sample_dir, 'input.ply'))
            save_ply(gt_raw, os.path.join(sample_dir, 'gt.ply'))
            save_ply(pred_raw, os.path.join(sample_dir, 'output.ply'))

        iterator.set_postfix({
            'CD': f'{sample_global_cd:.6f}',
            'HD': f'{sample_hd:.6f}',
            'ms/sample': f'{infer_time_sec * 1000.0:.2f}',
        })

    if distributed:
        gathered_rows = gather_object_list(local_rows, world_size)
        per_sample_rows = flatten_gathered_rows(gathered_rows)
        local_agg = {
            'count': int(local_total_count),
            'loss': float(local_total_loss),
            'hd': float(local_total_hd),
            'p2f': float(local_total_p2f),
            'p2f_std': float(local_total_p2f_std),
            'p2f_count': int(local_total_p2f_count),
            'p2f_mesh': int(local_total_p2f_mesh_count),
            'p2f_missing': int(local_total_p2f_missing_count),
            'infer_time': float(local_total_infer_time_sec),
        }
        gathered_agg = gather_object_list(local_agg, world_size)
    else:
        per_sample_rows = local_rows
        gathered_agg = [{
            'count': int(local_total_count),
            'loss': float(local_total_loss),
            'hd': float(local_total_hd),
            'p2f': float(local_total_p2f),
            'p2f_std': float(local_total_p2f_std),
            'p2f_count': int(local_total_p2f_count),
            'p2f_mesh': int(local_total_p2f_mesh_count),
            'p2f_missing': int(local_total_p2f_missing_count),
            'infer_time': float(local_total_infer_time_sec),
        }]

    if distributed and rank != 0:
        return None

    total_count = int(sum(item['count'] for item in gathered_agg))
    total_loss = float(sum(item['loss'] for item in gathered_agg))
    total_hd = float(sum(item['hd'] for item in gathered_agg))
    total_p2f = float(sum(item['p2f'] for item in gathered_agg))
    total_p2f_std = float(sum(item['p2f_std'] for item in gathered_agg))
    total_p2f_count = int(sum(item['p2f_count'] for item in gathered_agg))
    total_p2f_mesh_count = int(sum(item['p2f_mesh'] for item in gathered_agg))
    total_p2f_missing_count = int(sum(item['p2f_missing'] for item in gathered_agg))
    total_infer_time_sec = float(sum(item['infer_time'] for item in gathered_agg))

    if total_count <= 0:
        raise RuntimeError('No samples were evaluated during distributed metrics stage.')

    per_category_summary = aggregate_category_rows(per_sample_rows)
    p2f_definition = 'Mesh-based P2F on original-scale coordinates: average point-to-triangle-surface distance from each denormalized predicted point to the GT mesh (Open3D RaycastingScene).'

    overall_summary = {
        'samples': int(total_count),
        'cd_definition': 'CD on independently normalized pred/gt points = mean(dist1) + mean(dist2) on squared Chamfer distances.',
        'avg_loss': float(total_loss / total_count),
        'normalization': 'Dataset samples are normalized by sparse-input centroid/furthest-distance before inference. For metrics, CD/HD follow Evaluation/cd_hd_p2f.py: pred and gt are each normalized by their own centroid/furthest-distance before distance computation; P2F is computed after denormalizing predictions back to the original scale.',
        'hd_definition': 'HD on independently normalized pred/gt points = max_nn(pred->gt) + max_nn(gt->pred) on squared Chamfer distances.',
        'avg_hd': float(total_hd / total_count),
        'p2f_definition': p2f_definition,
        'avg_p2f': float(total_p2f / total_p2f_count) if total_p2f_count > 0 else None,
        'avg_p2f_std': float(total_p2f_std / total_p2f_count) if total_p2f_count > 0 else None,
        'p2f_valid_samples': int(total_p2f_count),
        'p2f_mesh_samples': int(total_p2f_mesh_count),
        'p2f_missing_samples': int(total_p2f_missing_count),
        'avg_infer_time_sec_per_sample': float(total_infer_time_sec / total_count),
        'avg_infer_time_ms_per_sample': float((total_infer_time_sec / total_count) * 1000.0),
        'total_infer_time_sec': float(total_infer_time_sec),
        'checkpoint': os.path.abspath(args.checkpoint),
        'dataset_root': os.path.abspath(args.dataset_root),
        'split': args.split,
        'steps': int(args.steps),
        'unit_mode': str(getattr(args, 'unit_mode', 'full')),
        'inference_index_mode': getattr(args, 'inference_index_mode', 'patch'),
        'inference_patch_size': int(getattr(args, 'inference_patch_size', 256)),
        'inference_patch_rate': float(getattr(args, 'inference_patch_rate', 3.0)),
        'inference_patch_num': int(getattr(args, 'inference_patch_num', 0)),
        'inference_patch_normalize': not bool(getattr(args, 'no_inference_patch_normalize', False)),
        'sparse_only_inference': True,
        'use_image_condition': bool(args.use_image_condition),
        'image_cond_filename': getattr(args, 'image_cond_file', DEFAULT_IMAGE_COND_FILENAME),
        'image_cond_key': getattr(args, 'image_cond_key', DEFAULT_IMAGE_COND_KEY),
        'image_cond_array_key': getattr(args, 'image_cond_array_key', DEFAULT_IMAGE_COND_ARRAY_KEY),
        'image_cond_dim': int(getattr(args, 'effective_image_cond_dim', DEFAULT_IMAGE_COND_DIM)),
        'infer_joint_query_prototypes': bool(getattr(args, 'infer_joint_query_prototypes', False)),
        'x0_mode': DEFAULT_TEST_X0_MODE,
        'x0_noise_inference': False,
        'model_n_parts': getattr(args, 'model_n_parts', None),
        'model_use_image_controller': getattr(args, 'model_use_image_controller', None),
        'model_coarse_num_prototypes': getattr(args, 'model_coarse_num_prototypes', None),
        'model_num_prototypes': getattr(args, 'model_num_prototypes', None),
        'model_prototype_kmeans_iters': getattr(args, 'model_prototype_kmeans_iters', None),
        'model_prototype_assign_temperature': getattr(args, 'model_prototype_assign_temperature', None),
        'model_prototype_consistency_weight': getattr(args, 'model_prototype_consistency_weight', None),
        'model_use_query_tokens_as_infer_target': getattr(args, 'model_use_query_tokens_as_infer_target', None),
        'p2f_mode': 'mesh',
        'mesh_root': os.path.abspath(args.mesh_root) if args.mesh_root else '',
        'mesh_extensions': str(args.mesh_extensions),
        'copy_mesh_to_gt': bool(args.copy_mesh_to_gt),
        'copied_mesh_basename': str(args.copied_mesh_basename),
        'distributed': bool(int(os.environ.get('WORLD_SIZE', '1')) > 1),
        'world_size': int(os.environ.get('WORLD_SIZE', '1')),
        'evaluation_mode': 'two_stage_ddp_infer_then_ddp_metrics_reduce',
        'artifact_root': artifact_root,
    }
    return overall_summary, per_category_summary, per_sample_rows


def parse_args():
    """Build and validate the inference CLI."""
    parser = argparse.ArgumentParser(description='ControlFlow3D sparse-only inference (DDP-ready)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='./data/ShapeNetPU',
    )
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--steps', type=int, default=5, help='Euler steps for inference')
    parser.add_argument(
        '--unit_mode',
        type=str,
        choices=['full', 'part'],
        default='full',
        help='full: run inference on the whole sparse input. part: split into contiguous blocks like 0-255/256-511/... and stitch outputs.',
    )
    parser.add_argument('--full_num_parts', type=int, default=8, help='Number of part/patch units configured in the trained checkpoint.')
    parser.add_argument('--inference_index_mode', type=str, default='patch', choices=['part', 'patch'], help='Refiner split mode for inference. patch builds FPS/KNN patches and resamples the merged output with FPS.')
    parser.add_argument('--inference_patch_size', type=int, default=256, help='Number of input points per KNN patch.')
    parser.add_argument('--inference_patch_rate', type=float, default=3.0, help='Patch coverage rate: patch_num=int(N/patch_size*patch_rate).')
    parser.add_argument('--inference_patch_num', type=int, default=0, help='If >0, override the automatically computed patch count.')
    parser.add_argument('--no_inference_patch_normalize', action='store_true', help='Disable per-patch normalization and scale restoration.')
    parser.add_argument('--use_image_condition', action='store_true', help='Enable image conditioning during inference (off by default).')
    parser.add_argument('--image_cond_file', type=str, default=DEFAULT_IMAGE_COND_FILENAME, help='Image condition npz filename under each sample input directory.')
    parser.add_argument('--image_cond_key', type=str, default=DEFAULT_IMAGE_COND_KEY, help='Key inside each image condition NPZ file.')
    parser.add_argument('--image_cond_array_key', type=str, default=DEFAULT_IMAGE_COND_ARRAY_KEY, help='Array name in the memmap cache (the NPZ key is --image_cond_key).' )
    parser.add_argument('--image_cond_dim', type=int, default=0, help='Override image condition dim. 0 uses the default 512 dimensions; set explicitly for custom features.')
    parser.add_argument(
        '--infer_joint_query_prototypes',
        action='store_true',
        help='During image-conditioned inference, cluster image tokens jointly with query 3D tokens instead of image-only prototypes.',
    )
    parser.add_argument('--gpu_ids', type=str, default='0', help='Visible GPU IDs, e.g. 1,2,3,0 or 0,1')
    parser.add_argument('--save_dir', type=str, default='results', help='Root directory for outputs')
    parser.add_argument('--no_save_ply', action='store_true', help='Skip saving PLY files')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--categories', type=str, default='', help='Comma-separated category IDs')
    parser.add_argument('--max_categories', type=int, default=0, help='If > 0 and --categories is empty, evaluate only first K categories')
    parser.add_argument('--max_samples_per_category', type=int, default=0, help='If > 0, limit number of samples for each category')
    parser.add_argument('--num_workers', type=int, default=0, help='0 means auto infer')
    parser.add_argument('--mesh_root', type=str, default='./data/ShapeNetCore.v1', help='Root directory for GT meshes, usually ShapeNetCore.v1.')
    parser.add_argument('--mesh_extensions', type=str, default='obj,off,ply', help='Comma-separated mesh extensions used for lookup.')
    parser.add_argument('--copied_mesh_basename', type=str, default='gt_mesh', help='Basename when copying mesh into sample gt folder.')
    parser.add_argument('--copy_mesh_to_gt', dest='copy_mesh_to_gt', action='store_true', help='Copy found mesh into each sample gt folder for test-time P2F.')
    parser.add_argument('--no_copy_mesh_to_gt', dest='copy_mesh_to_gt', action='store_false', help='Do not copy mesh; use source mesh path directly.')
    parser.set_defaults(copy_mesh_to_gt=True)
    args = parser.parse_args()
    args.effective_image_cond_dim = resolve_image_cond_dim(args.image_cond_dim)
    return args


def main():
    args = parse_args()

    startup_logs = []

    def log_stage(message: str, *, detail=False, summary=None):
        """Keep startup diagnostics for the report and show concise stage messages."""
        startup_logs.append(message)
        if not detail:
            tqdm.write(message if summary is None else summary)

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')
    if not args.no_save_ply and not HAS_O3D:
        raise RuntimeError('open3d is required for PLY export. Install open3d or pass --no_save_ply.')

    device, local_rank, rank, world_size, distributed = setup_distributed()
    is_main_process = rank == 0
    num_points = 2048
    num_workers = args.num_workers if args.num_workers > 0 else infer_num_workers()

    try:
        if is_main_process:
            log_stage(
                f'[startup] checkpoint={os.path.abspath(args.checkpoint)} '
                f'split={args.split} batch_size={args.batch_size} num_workers={num_workers} '
                f'image_cond={int(args.use_image_condition)} '
                f'infer_joint_query_prototypes={int(args.infer_joint_query_prototypes)} '
                f'unit_mode={args.unit_mode} x0_mode={DEFAULT_TEST_X0_MODE}',
                summary=f'[test] checkpoint={os.path.basename(args.checkpoint)} | split={args.split} | batch_size={args.batch_size}'
            )
            startup_t0 = time.perf_counter()

        dataset_t0 = time.perf_counter()
        dataset = MultimodalPointCloudDataset(
            root_dir=args.dataset_root,
            split=args.split,
            skip_bad_samples=False,
            num_points=num_points,
            return_category=True,
            return_sample_id=True,
            normalize_by_input=True,
            return_normalization_meta=True,
            load_gt_feat=False,
            load_image_cond=args.use_image_condition,
            image_cond_filename=args.image_cond_file,
            image_cond_key=args.image_cond_key,
            image_cond_array_key=args.image_cond_array_key,
            image_cond_dim=args.effective_image_cond_dim,
            point_index_mode='part',
            num_parts=args.full_num_parts,
        )
        if is_main_process:
            log_stage(f'[startup] dataset_ready in {time.perf_counter() - dataset_t0:.2f}s',
                detail=True
            )

        sample_records = dataset.samples
        available_categories = sorted({sample['category'] for sample in sample_records})
        selected_categories = None
        if args.categories.strip():
            selected_categories = [item.strip() for item in args.categories.split(',') if item.strip()]
        elif args.max_categories > 0:
            selected_categories = available_categories[:args.max_categories]

        if selected_categories is not None:
            selected_set = set(selected_categories)
            dataset.samples = [sample for sample in dataset.samples if sample['category'] in selected_set]

        if args.max_samples_per_category > 0:
            kept_samples = []
            count_by_cat = {}
            for sample in dataset.samples:
                cat = sample['category']
                current = count_by_cat.get(cat, 0)
                if current < args.max_samples_per_category:
                    kept_samples.append(sample)
                    count_by_cat[cat] = current + 1
            dataset.samples = kept_samples

        if len(dataset.samples) == 0:
            raise RuntimeError('No samples left after filtering.')

        dataloader_t0 = time.perf_counter()
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=False) if distributed else None
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        if is_main_process:
            log_stage(
                f'[startup] dataloader_ready in {time.perf_counter() - dataloader_t0:.2f}s '
                f'(samples={len(dataset.samples)}, batches={len(dataloader)})',
                summary=f'[data] samples={len(dataset.samples)} | batches={len(dataloader)} | source={"memmap" if dataset.memmap_backend is not None else "files"}'
            )

        model_t0 = time.perf_counter()
        model, model_core, _ckpt = build_model_for_test(
            checkpoint_path=args.checkpoint,
            device=device,
            steps=args.steps,
            n_parts=1 if args.unit_mode == 'part' else args.full_num_parts,
            x0_mode=DEFAULT_TEST_X0_MODE,
            x0_noise_scale=DEFAULT_TEST_X0_NOISE_SCALE,
            wrap_data_parallel=False,
        )
        if getattr(model_core, 'condition_flow_matching', None) is not None:
            image_controller = getattr(model_core.condition_flow_matching, 'image_controller', None)
            if image_controller is not None:
                image_controller.use_query_tokens_as_infer_target = bool(args.infer_joint_query_prototypes)
            model_core.condition_flow_matching.use_query_tokens_as_infer_target = bool(args.infer_joint_query_prototypes)
        model_core.use_query_tokens_as_infer_target = bool(args.infer_joint_query_prototypes)
        if is_main_process:
            log_stage(f'[startup] model_ready in {time.perf_counter() - model_t0:.2f}s',
                summary=f'[model] Checkpoint loaded in {time.perf_counter() - model_t0:.2f}s'
            )
            log_stage(
                f'[startup] model_effective n_parts={model_core.n_parts} '
                f'use_partwise_refiner={int(getattr(model_core, "use_partwise_inference", True))} '
                f'steps={model_core.inference_steps} '
                f'x0_mode={getattr(model_core, "runtime_x0_mode", "feature")} '
                f'image_controller={int(getattr(model_core, "use_image_controller", False))} '
                f'coarse_num_prototypes={getattr(model_core, "coarse_num_prototypes", -1)} '
                f'num_prototypes={getattr(model_core, "num_prototypes", -1)} '
                f'prototype_kmeans_iters={getattr(model_core, "prototype_kmeans_iters", -1)} '
                f'prototype_assign_temperature={getattr(model_core, "prototype_assign_temperature", -1.0)} '
                f'infer_joint_query_prototypes={int(args.infer_joint_query_prototypes)} '
                f'prototype_consistency_weight={getattr(model_core, "prototype_consistency_weight", -1.0)}',
                detail=True
            )
        args.model_n_parts = int(model_core.n_parts)
        args.model_use_image_controller = bool(getattr(model_core, 'use_image_controller', False))
        args.model_coarse_num_prototypes = int(getattr(model_core, 'coarse_num_prototypes', -1))
        args.model_num_prototypes = int(getattr(model_core, 'num_prototypes', -1))
        args.model_prototype_kmeans_iters = int(getattr(model_core, 'prototype_kmeans_iters', -1))
        args.model_prototype_assign_temperature = float(getattr(model_core, 'prototype_assign_temperature', -1.0))
        args.model_prototype_consistency_weight = float(getattr(model_core, 'prototype_consistency_weight', -1.0))
        args.model_use_query_tokens_as_infer_target = bool(getattr(model_core, 'use_query_tokens_as_infer_target', False))
        if distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
            model_core = model.module

        run_time = broadcast_string(
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S') if is_main_process else '',
            rank=rank,
            distributed=distributed,
        )
        save_dir_abs = os.path.abspath(os.path.join(args.save_dir, run_time))
        artifact_root = os.path.join(save_dir_abs, '_artifacts')
        if is_main_process:
            os.makedirs(artifact_root, exist_ok=True)
        # Synchronize all ranks after creating output directories.
        if distributed:
            dist.barrier()

        with torch.no_grad():
            if is_main_process:
                log_stage(f'[startup] entering_inference after {time.perf_counter() - startup_t0:.2f}s',
                    summary=f'[test] Starting inference | steps={args.steps} | grouping={args.inference_index_mode}'
                )
            iterator = tqdm(
                dataloader,
                total=len(dataloader),
                desc=f'Inference-rank{rank}' if distributed else 'Inference',
                disable=not is_main_process,
            )
            for batch in iterator:
                batch_data = adapt_inference_batch(batch, args.use_image_condition)
                image_cond = batch_data['image_cond']
                sparse_xyz = batch_data['sparse_xyz']
                sparse_feat = batch_data['sparse_feat']
                gt_xyz = batch_data['gt_xyz']
                centroid = batch_data['centroid']
                furthest_distance = batch_data['furthest_distance']
                sample_ids = batch_data['sample_ids']
                categories = batch_data['categories']
                sparse_patch_idx = batch_data.get('sparse_patch_idx')
                sparse_xyz = sparse_xyz.to(device, non_blocking=True).float()
                sparse_feat = sparse_feat.to(device, non_blocking=True).float()
                gt_xyz = gt_xyz.to(device, non_blocking=True).float()
                centroid = centroid.to(device, non_blocking=True).float()
                furthest_distance = furthest_distance.to(device, non_blocking=True).float()
                if sparse_patch_idx is not None:
                    sparse_patch_idx = sparse_patch_idx.to(device, non_blocking=True)
                if image_cond is not None:
                    image_cond = image_cond.to(device, non_blocking=True).float()

                maybe_cuda_sync(device)
                infer_t0 = time.perf_counter()
                if args.use_image_condition:
                    def _conditioned_rollout(local_sparse_xyz, local_sparse_feat, local_category_ids, local_image_cond):
                        return model_core._rollout_sparse_only(
                            sparse_xyz=local_sparse_xyz,
                            sparse_feat=local_sparse_feat,
                            steps=model_core.inference_steps,
                            category_ids=local_category_ids,
                            image_cond=local_image_cond,
                            sparse_patch_idx=sparse_patch_idx,
                            refiner_index_mode=args.inference_index_mode,
                            add_x0_noise=False,
                            action_noise_std=0.0,
                            return_trace=False,
                            inference_patch_size=args.inference_patch_size,
                            inference_patch_rate=args.inference_patch_rate,
                            inference_patch_num=args.inference_patch_num,
                            inference_patch_normalize=not args.no_inference_patch_normalize,
                        )

                    if args.unit_mode == 'part':
                        def _conditioned_rollout_part(local_sparse_xyz, local_sparse_feat, local_category_ids, local_image_cond):
                            return model_core._rollout_sparse_only(
                                sparse_xyz=local_sparse_xyz,
                                sparse_feat=local_sparse_feat,
                                steps=model_core.inference_steps,
                                category_ids=local_category_ids,
                                image_cond=local_image_cond,
                                sparse_patch_idx=None,
                                add_x0_noise=False,
                                action_noise_std=0.0,
                                return_trace=False,
                            )

                        final_pred = infer_sparse_only_partwise_conditioned(
                            _conditioned_rollout_part,
                            sparse_xyz,
                            sparse_feat,
                            args.full_num_parts,
                            category_ids=None,
                            image_cond=image_cond,
                        )
                    else:
                        final_pred = _conditioned_rollout(
                            sparse_xyz,
                            sparse_feat,
                            None,
                            image_cond,
                        )
                else:
                    if args.unit_mode == 'part':
                        final_pred = infer_sparse_only_partwise(
                            model,
                            sparse_xyz,
                            sparse_feat,
                            args.full_num_parts,
                        )
                    else:
                        final_pred = model(
                            sparse_xyz,
                            sparse_feat,
                            category_ids=None,
                            train=False,
                            sparse_patch_idx=sparse_patch_idx,
                            refiner_index_mode=args.inference_index_mode,
                            inference_patch_size=args.inference_patch_size,
                            inference_patch_rate=args.inference_patch_rate,
                            inference_patch_num=args.inference_patch_num,
                            inference_patch_normalize=not args.no_inference_patch_normalize,
                        )
                maybe_cuda_sync(device)
                infer_elapsed = time.perf_counter() - infer_t0

                batch_size = sparse_xyz.shape[0]
                infer_time_per_sample = infer_elapsed / max(batch_size, 1)

                if is_main_process:
                    iterator.set_postfix({
                        'ms/sample': f'{infer_time_per_sample * 1000.0:.2f}',
                    })

                pred_np = final_pred.detach().cpu().numpy()
                sparse_np = sparse_xyz.detach().cpu().numpy()
                gt_np = gt_xyz.detach().cpu().numpy()
                centroid_np = centroid.detach().cpu().numpy()
                furthest_np = furthest_distance.detach().cpu().numpy()

                for b_idx in range(batch_size):
                    sid = str(sample_ids[b_idx])
                    cat = str(categories[b_idx])
                    save_inference_artifact(
                        artifact_root=artifact_root,
                        category=cat,
                        sample_id=sid,
                        sparse_xyz=sparse_np[b_idx],
                        gt_xyz=gt_np[b_idx],
                        pred_xyz=pred_np[b_idx],
                        centroid=centroid_np[b_idx],
                        furthest_distance=furthest_np[b_idx],
                        infer_time_sec=float(infer_time_per_sample),
                        rank=rank,
                    )

        if distributed:
            dist.barrier()

        eval_outputs = evaluate_saved_artifacts(
            artifact_root=artifact_root,
            save_dir_abs=save_dir_abs,
            args=args,
            device=device,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            expected_samples=dataset.samples,
        )
        if eval_outputs is None:
            return
        overall_summary, per_category_summary, per_sample_rows = eval_outputs
        overall_summary = round_nested_floats(overall_summary, ndigits=9)
        per_category_summary = round_nested_floats(per_category_summary, ndigits=9)
        per_sample_rows = round_nested_floats(per_sample_rows, ndigits=9)
        run_args = {key: (str(value) if isinstance(value, os.PathLike) else value) for key, value in vars(args).items()}

        write_evaluation_report(save_dir_abs, overall_summary, per_category_summary,
                                per_sample_rows, run_args)

        log_path = os.path.join(save_dir_abs, 'evaluation_log.txt')
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('[startup]\n' + '\n'.join(startup_logs) + '\n\n')
            f.write('[summary]\n')
            f.write(f'samples: {overall_summary["samples"]}\n')
            f.write(f'CD: {overall_summary["avg_loss"]:.9f}\n')
            f.write(f'HD: {overall_summary["avg_hd"]:.9f}\n')
            if overall_summary['avg_p2f'] is None:
                f.write('P2F: N/A\n')
            else:
                f.write(f'P2F: {overall_summary["avg_p2f"]:.9f}\n')
            f.write(f'infer_time_ms_per_sample: {overall_summary["avg_infer_time_ms_per_sample"]:.2f}\n')
            f.write('\n')
            f.write('[args]\n')
            f.write(json.dumps(run_args, ensure_ascii=False, indent=2))
            f.write('\n\n')
            f.write('[overall]\n')
            f.write(json.dumps(overall_summary, ensure_ascii=False, indent=2))
            f.write('\n\n[per_category]\n')
            f.write(json.dumps(per_category_summary, ensure_ascii=False, indent=2))
            f.write('\n\n[per_sample]\n')
            for row in per_sample_rows:
                f.write(json.dumps(row, ensure_ascii=False))
                f.write('\n')

        print(f'[test] Complete | samples={overall_summary["samples"]}')
        print(f'CD: {overall_summary["avg_loss"]:.9f}')
        print(f'HD: {overall_summary["avg_hd"]:.9f}')
        if overall_summary['avg_p2f'] is None:
            print('P2F: N/A (no valid GT meshes)')
        else:
            print(
                f'P2F: {overall_summary["avg_p2f"]:.9f} '
                f'(mesh={overall_summary["p2f_mesh_samples"]}, '
                f'missing={overall_summary["p2f_missing_samples"]})'
            )
        print(f'Infer time per sample: {overall_summary["avg_infer_time_ms_per_sample"]:.2f} ms')
        print('Per-category metrics:')
        for cat, stats in sorted(per_category_summary.items()):
            p2f_str = 'N/A' if stats['avg_p2f'] is None else f'{stats["avg_p2f"]:.9f}'
            print(
                f'  {cat}: CD={stats["avg_loss"]:.9f}, HD={stats["avg_hd"]:.9f}, P2F={p2f_str}, '
                f'{stats["avg_infer_time_ms_per_sample"]:.2f} ms/sample, n={stats["samples"]}'
            )
        print(f'[output] {log_path}')
    finally:
        maybe_destroy_process_group()


if __name__ == '__main__':
    main()
