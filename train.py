"""Train ControlFlow3D with optional distributed execution."""

import argparse
import datetime
import gc
import math
import os
import sys
import tempfile
import warnings
import time
import glob

from utils.runtime_utils import (
    preset_cuda_visible_devices_from_argv, validation_intervals, should_validate,
)
from utils.dataset_feature_layout import (
    DEFAULT_IMAGE_COND_FILENAME, DEFAULT_IMAGE_COND_KEY,
    DEFAULT_IMAGE_COND_ARRAY_KEY, DEFAULT_IMAGE_COND_DIM,
)

DEFAULT_GPU_IDS = '0'

# Set GPU visibility before importing PyTorch.
preset_cuda_visible_devices_from_argv(DEFAULT_GPU_IDS, set_unconditionally=True)

warnings.filterwarnings(
    'ignore',
    message=r".*param_schemas.*deprecated.*op_signature.*",
    category=FutureWarning,
)
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

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasetsource.multimodal_pointcloud_dataset import MultimodalPointCloudDataset
from models.controlflow3d import ControlFlow3D


def split_data(batch, device):
    image_cond = None
    core_batch = batch
    if (
        len(batch) > 0
        and torch.is_tensor(batch[-1])
        and batch[-1].dim() >= 2
        and batch[-1].shape[-1] >= 64
    ):
        image_cond = batch[-1].to(device, non_blocking=True)
        core_batch = batch[:-1]

    if len(core_batch) < 4:
        raise ValueError(f'Unexpected batch length: {len(batch)} (core={len(core_batch)})')

    sparse, sparse_feat, gt_feat, gt = core_batch[:4]
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

    sample_ids = None
    categories = None
    if len(rest) >= 2 and torch.is_tensor(rest[0]) and rest[0].dim() == 3 and rest[0].shape[-1] == 3:
        # Centroid and normalization radius.
        rest = rest[2:]

    if len(rest) == 2:
        sample_ids, categories = rest
    elif len(rest) == 1:
        categories = rest[0]
    elif len(rest) != 0:
        raise ValueError(f'Unexpected batch tail length: {len(rest)} (core={len(core_batch)})')
    sparse = sparse.to(device, non_blocking=True)
    sparse_feat = sparse_feat.to(device, non_blocking=True)
    gt_feat = gt_feat.to(device, non_blocking=True)
    gt = gt.to(device, non_blocking=True)
    return sparse, sparse_feat, gt_feat, gt, image_cond, sample_ids, categories, sparse_patch_idx, gt_patch_idx


def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()


def maybe_destroy_process_group():
    if is_dist_initialized():
        dist.destroy_process_group()


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def reduce_tensor_mean(tensor, world_size):
    if world_size == 1:
        return tensor
    reduced = tensor.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= world_size
    return reduced


def reduce_metrics_sums(values, device, world_size):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def set_runtime_inference_params(model):
    model.inference_steps = 5
    model.use_partwise_inference = True
    model.runtime_x0_mode = 'feature'
    model.runtime_x0_noise_scale = 1.0


def compute_scheduled_weight(epoch_idx, total_epochs, start, end, mode, warmup_epochs=0):
    if total_epochs <= 1 or mode == 'fixed':
        return float(start)
    warmup_epochs = max(int(warmup_epochs), 0)
    if warmup_epochs > 0 and epoch_idx < warmup_epochs:
        return float(start)
    schedule_epochs = max(total_epochs - warmup_epochs, 1)
    progress = float(epoch_idx - warmup_epochs) / float(max(schedule_epochs - 1, 1))
    if mode == 'linear':
        return float(start + (end - start) * progress)
    if mode == 'cosine':
        return float(start + (end - start) * 0.5 * (1.0 - math.cos(math.pi * progress)))
    raise ValueError(f'Unknown loss weight schedule: {mode}')


def broadcast_run_time(local_value, distributed, rank):
    if not distributed:
        return local_value
    payload = [local_value if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def setup_distributed(args):
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    args.rank = int(os.environ.get('RANK', 0))
    args.world_size = int(os.environ.get('WORLD_SIZE', 1))
    args.distributed = args.world_size > 1

    if args.distributed and not torch.cuda.is_available():
        raise RuntimeError('DDP training requires CUDA.')

    if torch.cuda.is_available():
        if args.distributed:
            torch.cuda.set_device(args.local_rank)
            device = torch.device('cuda', args.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')
        else:
            torch.cuda.set_device(0)
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device


def infer_num_workers(world_size):
    cpu_count = os.cpu_count() or 8
    per_rank_budget = max(cpu_count // max(world_size, 1), 1)
    return max(2, min(8, per_rank_budget))


def _parse_cpu_list(cpu_list_str):
    cpus = set()
    for chunk in cpu_list_str.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '-' in chunk:
            start, end = chunk.split('-', 1)
            cpus.update(range(int(start), int(end) + 1))
        else:
            cpus.add(int(chunk))
    return sorted(cpus)


def _format_cpu_list(cpus):
    if not cpus:
        return ''
    ranges = []
    start = prev = int(cpus[0])
    for cpu in cpus[1:]:
        cpu = int(cpu)
        if cpu == prev + 1:
            prev = cpu
            continue
        ranges.append(f'{start}-{prev}' if start != prev else str(start))
        start = prev = cpu
    ranges.append(f'{start}-{prev}' if start != prev else str(start))
    return ','.join(ranges)


def discover_numa_cpu_sets():
    node_paths = sorted(glob.glob('/sys/devices/system/node/node[0-9]*'))
    cpu_sets = []
    for path in node_paths:
        cpulist_path = os.path.join(path, 'cpulist')
        try:
            with open(cpulist_path, 'r', encoding='utf-8') as f:
                cpus = _parse_cpu_list(f.read().strip())
            if cpus:
                cpu_sets.append(cpus)
        except OSError:
            continue
    return cpu_sets


def discover_gpu_local_cpu_sets():
    gpu_cpu_sets = {}
    gpu_paths = sorted(glob.glob('/proc/driver/nvidia/gpus/*'))
    for gpu_path in gpu_paths:
        bus_id = os.path.basename(gpu_path)
        device_minor = None
        info_path = os.path.join(gpu_path, 'information')
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('Device Minor:'):
                        device_minor = int(line.split(':', 1)[1].strip())
                    elif line.startswith('Bus Location:'):
                        bus_id = line.split(':', 1)[1].strip()
        except OSError:
            continue
        if device_minor is None:
            continue

        cpulist_path = os.path.join('/sys/bus/pci/devices', bus_id, 'local_cpulist')
        try:
            with open(cpulist_path, 'r', encoding='utf-8') as f:
                cpus = _parse_cpu_list(f.read().strip())
        except OSError:
            continue
        if cpus:
            gpu_cpu_sets[int(device_minor)] = cpus
    return gpu_cpu_sets


def parse_visible_gpu_ids(gpu_ids_arg):
    if gpu_ids_arg is None:
        gpu_ids_arg = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not gpu_ids_arg:
        return []

    physical_ids = []
    for token in str(gpu_ids_arg).split(','):
        token = token.strip()
        if not token:
            continue
        try:
            physical_ids.append(int(token))
        except ValueError:
            # Non-integer identifiers such as GPU UUIDs are left unresolved.
            return []
    return physical_ids


def get_rank_device_mapping(local_rank, gpu_ids_arg):
    physical_ids = parse_visible_gpu_ids(gpu_ids_arg)
    if physical_ids and 0 <= local_rank < len(physical_ids):
        return {
            'visible_cuda_index': int(local_rank),
            'physical_gpu_id': int(physical_ids[local_rank]),
            'visible_gpu_ids': physical_ids,
        }
    return {
        'visible_cuda_index': int(local_rank),
        'physical_gpu_id': None,
        'visible_gpu_ids': physical_ids,
    }


def maybe_set_rank_cpu_affinity(local_rank, mode, gpu_ids_arg=None):
    device_mapping = get_rank_device_mapping(local_rank, gpu_ids_arg)
    if mode == 'off' or not hasattr(os, 'sched_setaffinity'):
        return None, device_mapping

    selected = None
    physical_gpu_id = device_mapping['physical_gpu_id']
    if physical_gpu_id is not None:
        gpu_cpu_sets = discover_gpu_local_cpu_sets()
        selected = gpu_cpu_sets.get(int(physical_gpu_id))

    if not selected:
        cpu_sets = discover_numa_cpu_sets()
        if not cpu_sets:
            return None, device_mapping
        affinity_index = int(physical_gpu_id) if physical_gpu_id is not None else int(local_rank)
        selected = cpu_sets[affinity_index % len(cpu_sets)]

    os.sched_setaffinity(0, selected)
    return selected, device_mapping


def build_dataloader(
    dataset,
    batch_size,
    sampler,
    shuffle,
    num_workers,
    prefetch_factor,
    drop_last,
    persistent_workers,
    multiprocessing_context,
):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
    if num_workers > 0:
        kwargs['persistent_workers'] = persistent_workers
        if multiprocessing_context is not None:
            kwargs['multiprocessing_context'] = multiprocessing_context
        if prefetch_factor is not None:
            kwargs['prefetch_factor'] = prefetch_factor
    return DataLoader(**kwargs)


class ShardAwareDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas, rank, *, drop_last=True, seed=0):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        backend = getattr(dataset, 'memmap_backend', None)
        if backend is None or not getattr(backend, 'is_sharded', False):
            raise ValueError('ShardAwareDistributedSampler requires a sharded memmap backend.')
        self.shard_ranges = list(getattr(backend, 'shard_ranges', []))
        dataset_size = len(dataset)
        if self.drop_last:
            self.total_size = (dataset_size // self.num_replicas) * self.num_replicas
        else:
            self.total_size = int(math.ceil(dataset_size / self.num_replicas) * self.num_replicas)
        self.num_samples = self.total_size // self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        shard_order = torch.randperm(len(self.shard_ranges), generator=g).tolist()
        ordered_indices = []
        for shard_order_pos in shard_order:
            start, end, shard_id = self.shard_ranges[shard_order_pos]
            shard_len = int(end - start)
            if shard_len <= 0:
                continue
            # Rotate each shard's starting point while preserving sequential reads.
            offset = 0 if shard_len <= 1 else int((self.epoch * 9973 + int(shard_id) * 131) % shard_len)
            ordered_indices.extend(range(start + offset, end))
            if offset > 0:
                ordered_indices.extend(range(start, start + offset))

        if not self.drop_last and len(ordered_indices) < self.total_size:
            padding = self.total_size - len(ordered_indices)
            ordered_indices.extend(ordered_indices[:padding])
        else:
            ordered_indices = ordered_indices[:self.total_size]

        rank_start = self.rank * self.num_samples
        rank_end = rank_start + self.num_samples
        return iter(ordered_indices[rank_start:rank_end])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


def shutdown_dataloader_workers(loader):
    if loader is None:
        return
    iterator = getattr(loader, '_iterator', None)
    if iterator is None:
        return
    shutdown = getattr(iterator, '_shutdown_workers', None)
    if callable(shutdown):
        shutdown()
    loader._iterator = None


def _move_state_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _move_state_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_state_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_state_to_cpu(v) for v in obj)
    return obj


def save_checkpoint_atomic(path, state, *, device=None):
    if device is not None and device.type == 'cuda':
        torch.cuda.synchronize(device)
    cpu_state = _move_state_to_cpu(state)
    directory = os.path.dirname(path) or '.'
    fd, tmp_path = tempfile.mkstemp(prefix='.tmp_ckpt_', suffix='.pth', dir=directory)
    os.close(fd)
    try:
        torch.save(cpu_state, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        finally:
            raise


def parse_args():
    """Build and validate the training CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='./data/ShapeNetPU', help='Dataset root')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=128, help='Global training batch size across all ranks')
    parser.add_argument(
        '--val_batch_size',
        type=int,
        default=128,
        help='Global validation batch size across all ranks. 0 means use the same as --batch_size.',
    )
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--logdir', type=str, default='logs', help='Directory to save logs and checkpoints')
    parser.add_argument('--gpu_ids', type=str, default=DEFAULT_GPU_IDS, help='Visible GPU IDs, e.g. 1,2,3,0 or 0,1,2,3')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--num_workers', type=int, default=8, help='Per-rank train dataloader workers. Negative means auto.')
    parser.add_argument('--val_num_workers', type=int, default=0, help='Per-rank val dataloader workers. Negative means auto.')
    parser.add_argument(
        '--prefetch_factor',
        type=int,
        default=4,
        help='Per-worker dataloader prefetch depth. <= 0 means use the PyTorch default when num_workers > 0.',
    )
    parser.add_argument(
        '--rank_cpu_affinity',
        type=str,
        default='auto',
        choices=['auto', 'off'],
        help='Bind each local rank to the CPU set closest to its GPU. Recommended on multi-NUMA hosts.',
    )
    parser.add_argument('--log_every', type=int, default=50, help='Refresh tqdm/tensorboard every N train steps.')
    parser.add_argument('--val_freq', type=int, default=20, help='Initial validation interval, capped at 10/5 in later stages; the final epoch always validates.')
    parser.add_argument(
        '--disable_image_cond',
        action='store_true',
        help='Deprecated. Training now always requires 2D image conditioning.',
    )
    parser.add_argument(
        '--image_cond_file',
        type=str,
        default=DEFAULT_IMAGE_COND_FILENAME,
        help='Image conditioning NPZ filename under each sample input folder.',
    )
    parser.add_argument(
        '--image_cond_dim',
        type=int,
        default=512,
        help='Image conditioning embedding dimension.',
    )
    parser.add_argument(
        '--image_cond_key',
        type=str,
        default=DEFAULT_IMAGE_COND_KEY,
        help='Image conditioning key inside NPZ (e.g., global_cls or patch_tokens).',
    )
    parser.add_argument(
        '--image_feature_source',
        type=str,
        default='vggt',
        choices=['vggt', 'custom'],
        help='Image feature source preset. vggt auto-fills image_cond_array_key/image_cond_file/image_cond_dim; custom keeps manual values.',
    )
    parser.add_argument(
        '--image_cond_array_key',
        type=str,
        default=DEFAULT_IMAGE_COND_ARRAY_KEY,
        help='Memmap array key for image conditioning, e.g. vggt_image_cond.',
    )
    parser.add_argument(
        '--index_mode',
        type=str,
        default='part',
        choices=['part', 'patch'],
        help='Training refiner index source. part uses 8 contiguous part blocks by default; patch uses dataset patch indices.',
    )
    parser.add_argument(
        '--n_parts',
        type=int,
        default=0,
        help='Override refiner part/patch count. Default: 8 for part mode, 10 for patch mode.',
    )
    parser.add_argument(
        '--image_cond_max_tokens_per_view',
        type=int,
        default=32,
        help='If > 0 and image_cond is [V,T,C], average-pool each view to at most this many tokens before batching/controller.',
    )
    parser.add_argument(
        '--num_prototypes',
        type=int,
        default=8,
        help='Number of shared prototypes used by the image controller soft-KMeans.',
    )
    parser.add_argument(
        '--coarse_num_prototypes',
        type=int,
        default=4,
        help='Number of coarse shared prototypes used for global semantic modulation.',
    )
    parser.add_argument(
        '--prototype_kmeans_iters',
        type=int,
        default=4,
        help='Number of soft-KMeans refinement iterations for shared prototypes.',
    )
    parser.add_argument(
        '--prototype_assign_temperature',
        type=float,
        default=6.0,
        help='Soft assignment temperature for prototype clustering and query assignment.',
    )
    parser.add_argument(
        '--loss_fm_ctrl_weight',
        type=float,
        default=0.003,
        help='Weight for flow matching loss on image-controlled branch.',
    )
    parser.add_argument(
        '--loss_fm_ctrl_weight_end',
        type=float,
        default=0.0015,
        help='End weight for flow matching loss (enables scheduling when set).',
    )
    parser.add_argument(
        '--loss_proto_consistency_weight',
        type=float,
        default=0.01,
        help='Weight for image/3D prototype consistency loss.',
    )
    parser.add_argument(
        '--loss_proto_consistency_weight_end',
        type=float,
        default=0.005,
        help='End weight for prototype consistency loss (enables scheduling when set).',
    )
    parser.add_argument(
        '--local_loss_weight',
        type=float,
        default=0.35,
        help='Weight for local supervision loss in the total geometric loss.',
    )
    parser.add_argument(
        '--local_loss_final_stage_weight',
        type=float,
        default=0.0,
        help='Relative weight for local supervision on the final refined stage; 0 keeps local loss on earlier stages only.',
    )
    parser.add_argument(
        '--loss_weight_schedule',
        type=str,
        default='linear',
        choices=['fixed', 'linear', 'cosine'],
        help='Schedule for loss weights (fixed/linear/cosine).',
    )
    parser.add_argument(
        '--loss_weight_warmup_epochs',
        type=int,
        default=20,
        help='Warmup epochs to keep start weights before scheduling (<=0 uses default 20%% of epochs).',
    )
    parser.add_argument(
        '--save_last_every',
        type=int,
        default=1,
        help='Deprecated: last checkpoint is now saved every epoch; this flag is kept only for CLI compatibility.',
    )
    args = parser.parse_args()
    source_presets = {
        'vggt': {
            'array_key': DEFAULT_IMAGE_COND_ARRAY_KEY,
            'file': DEFAULT_IMAGE_COND_FILENAME,
            'dim': DEFAULT_IMAGE_COND_DIM,
        },
    }
    if args.image_feature_source in source_presets:
        preset = source_presets[args.image_feature_source]
        args.image_cond_array_key = preset['array_key']
        args.image_cond_file = preset['file']
        args.image_cond_dim = int(preset['dim'])
    if args.disable_image_cond:
        raise RuntimeError('Training must enable 2D image_cond; do not pass --disable_image_cond. Missing 2D image features will raise an error directly.')
    if args.loss_fm_ctrl_weight_end is None:
        args.loss_fm_ctrl_weight_end = args.loss_fm_ctrl_weight
    if args.loss_proto_consistency_weight_end is None:
        args.loss_proto_consistency_weight_end = args.loss_proto_consistency_weight
    if args.loss_weight_warmup_epochs is None or int(args.loss_weight_warmup_epochs) <= 0:
        args.loss_weight_warmup_epochs = max(int(round(args.epochs * 0.2)), 0)

    return args


def main():
    args = parse_args()

    device = setup_distributed(args)
    is_main_process = (args.rank == 0)
    bound_cpus, device_mapping = maybe_set_rank_cpu_affinity(
        args.local_rank,
        args.rank_cpu_affinity,
        gpu_ids_arg=args.gpu_ids,
    )

    if args.num_workers != 0 or args.val_num_workers != 0:
        mp.set_sharing_strategy('file_descriptor')

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.batch_size % args.world_size != 0:
        raise ValueError(f'batch_size={args.batch_size} must be divisible by world_size={args.world_size}.')
    num_points = 2048
    flow_feature_dim = 128
    condition_feature_dim = 256
    num_prototypes = max(int(args.num_prototypes), 1)
    coarse_num_prototypes = max(int(args.coarse_num_prototypes), 1)
    prototype_dim = 128
    prototype_kmeans_iters = max(int(args.prototype_kmeans_iters), 1)
    prototype_assign_temperature = float(args.prototype_assign_temperature)
    num_workers = infer_num_workers(args.world_size) if args.num_workers < 0 else int(args.num_workers)
    val_num_workers = num_workers if args.val_num_workers < 0 else int(args.val_num_workers)
    num_workers = max(num_workers, 0)
    val_num_workers = max(val_num_workers, 0)
    configured_prefetch_factor = int(args.prefetch_factor)
    prefetch_factor = configured_prefetch_factor if (num_workers > 0 and configured_prefetch_factor > 0) else None
    val_prefetch_factor = configured_prefetch_factor if (val_num_workers > 0 and configured_prefetch_factor > 0) else None
    prefetch_desc = '<torch-default>' if (num_workers > 0 and prefetch_factor is None) else prefetch_factor
    val_prefetch_desc = '<torch-default>' if (val_num_workers > 0 and val_prefetch_factor is None) else val_prefetch_factor
    dataloader_context = 'spawn'
    train_persistent_workers = num_workers > 0
    val_persistent_workers = False
    reload_train_loader_after_val = False
    empty_cache_after_val = False
    log_every = max(int(args.log_every), 1)
    val_freq = max(int(args.val_freq), 1)
    save_last_every = 1
    early_stage_val_freq, mid_stage_val_freq, late_stage_val_freq = validation_intervals(val_freq)
    mid_stage_start_epoch = int(args.epochs * 0.5) + 1
    late_stage_start_epoch = int(args.epochs * 0.7) + 1
    max_train_steps = 0

    local_batch_size = args.batch_size // args.world_size
    val_batch_size = int(args.val_batch_size) if int(args.val_batch_size) > 0 else int(args.batch_size)
    if val_batch_size % args.world_size != 0:
        val_batch_size = ((val_batch_size + args.world_size - 1) // args.world_size) * args.world_size
    local_val_batch_size = val_batch_size // args.world_size

    run_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') if is_main_process else None
    run_time = broadcast_run_time(run_time, args.distributed, args.rank)
    run_dir = os.path.join(args.logdir, run_time)
    os.makedirs(run_dir, exist_ok=True)

    tb = SummaryWriter(run_dir) if is_main_process else None
    log_fout = open(os.path.join(run_dir, 'log_train.txt'), 'w') if is_main_process else None

    def log_string(out_str, *, detail=False, summary=None):
        """Write full training logs and show concise main-rank console messages."""
        if not is_main_process:
            return
        if not detail:
            tqdm.write(out_str if summary is None else summary)
        if log_fout is not None:
            log_fout.write(out_str + '\n')
            log_fout.flush()

    if args.resume is not None and not os.path.isfile(args.resume):
        raise FileNotFoundError(f'Checkpoint not found: {args.resume}')

    model_n_parts = int(args.n_parts) if int(args.n_parts) > 0 else (8 if args.index_mode == 'part' else 10)

    log_string(
        f'Runtime config | visible_gpus={os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")} '
        f'train_workers={num_workers} val_workers={val_num_workers} '
        f'train_prefetch={prefetch_desc} val_prefetch={val_prefetch_desc} ctx={dataloader_context} '
        f'train_persistent={int(train_persistent_workers)} val_persistent={int(val_persistent_workers)} '
        f'reload_train_loader_after_val={int(reload_train_loader_after_val)} '
        f'empty_cache_after_val={int(empty_cache_after_val)} '
        f'cpu_affinity={args.rank_cpu_affinity} '
        f'local_batch={local_batch_size} local_val_batch={local_val_batch_size} '
        f'log_every={log_every} val_freq={val_freq} save_last_every={save_last_every} '
        f'mid_stage_start={mid_stage_start_epoch} mid_stage_val_freq={mid_stage_val_freq} '
        f'late_stage_start={late_stage_start_epoch} late_stage_val_freq={late_stage_val_freq} '
        f'early_stage_val_freq={early_stage_val_freq} '
        f'max_train_steps={max_train_steps} '
        f'image_feature_source={args.image_feature_source} '
        f'image_cond={int(not args.disable_image_cond)} image_cond_file={args.image_cond_file} '
        f'image_cond_array_key={args.image_cond_array_key} '
        f'image_cond_dim={args.image_cond_dim} image_cond_key={args.image_cond_key} '
        f'image_cond_max_tokens_per_view={max(int(args.image_cond_max_tokens_per_view), 0)} '
        f'index_mode={args.index_mode} n_parts={model_n_parts} '
        f'model_arch(n_parts/flow/cond/proto_dim/coarse/fine/kmeans/temp)='
        f'({model_n_parts},{flow_feature_dim},{condition_feature_dim},{prototype_dim},{coarse_num_prototypes},{num_prototypes},{prototype_kmeans_iters},{prototype_assign_temperature}) '
        f'loss_w(fm/proto)=({args.loss_fm_ctrl_weight}->{args.loss_fm_ctrl_weight_end},'
        f'{args.loss_proto_consistency_weight}->{args.loss_proto_consistency_weight_end},'
        f'local={args.local_loss_weight},local_final={args.local_loss_final_stage_weight},'
        f'{args.loss_weight_schedule},warmup={args.loss_weight_warmup_epochs})',
        summary=f'[train] epochs={args.epochs} | batch_size={args.batch_size} | ranks={args.world_size} | gpu_ids={args.gpu_ids}'
    )
    log_string(f'[output] {run_dir}')
    log_string('Coordinate normalization: enabled (sparse-input centroid/furthest-distance shared with gt_xyz)',
        detail=True
    )
    local_rank_mapping = {
        'rank': int(args.rank),
        'local_rank': int(args.local_rank),
        'visible_cuda_index': int(device_mapping['visible_cuda_index']),
        'physical_gpu_id': device_mapping['physical_gpu_id'],
        'visible_gpu_ids': device_mapping['visible_gpu_ids'],
        'cpu_affinity': None if bound_cpus is None else _format_cpu_list(bound_cpus),
        'cpu_affinity_count': 0 if bound_cpus is None else int(len(bound_cpus)),
    }
    gathered_rank_mappings = [None for _ in range(args.world_size)] if args.world_size > 1 else [local_rank_mapping]
    if args.world_size > 1:
        dist.all_gather_object(gathered_rank_mappings, local_rank_mapping)
    if is_main_process:
        for item in gathered_rank_mappings:
            rank_mapping_msg = (
                f'Rank mapping | rank={item["rank"]} local_rank={item["local_rank"]} '
                f'visible_cuda_index={item["visible_cuda_index"]} '
                f'physical_gpu_id={item["physical_gpu_id"]} '
                f'visible_gpu_ids={item["visible_gpu_ids"] or "<unresolved>"}'
            )
            if item['cpu_affinity'] is not None:
                rank_mapping_msg += f' cpu_affinity={item["cpu_affinity"]} ({item["cpu_affinity_count"]} CPUs)'
            log_string(rank_mapping_msg,
                detail=True
            )
    if is_main_process and bound_cpus is not None:
        log_string(f'Rank0 CPU affinity: {_format_cpu_list(bound_cpus)} ({len(bound_cpus)} CPUs)',
            detail=True
        )

    train_dataset = MultimodalPointCloudDataset(
        root_dir=args.dataset_root,
        split='train',
        skip_bad_samples=True,
        num_points=num_points,
        return_category=True,
        return_sample_id=False,
        normalize_by_input=True,
        load_gt_feat=True,
        load_image_cond=not args.disable_image_cond,
        image_cond_filename=args.image_cond_file,
        image_cond_key=args.image_cond_key,
        image_cond_array_key=args.image_cond_array_key,
        image_cond_dim=args.image_cond_dim,
        image_cond_max_tokens_per_view=args.image_cond_max_tokens_per_view,
        point_index_mode=args.index_mode,
        num_parts=model_n_parts,
    )
    val_dataset = MultimodalPointCloudDataset(
        root_dir=args.dataset_root,
        split='test',
        skip_bad_samples=False,
        num_points=num_points,
        return_category=False,
        return_sample_id=False,
        normalize_by_input=True,
        return_normalization_meta=False,
        load_gt_feat=False,
        # Validation follows inference policy: no image condition.
        load_image_cond=False,
        image_cond_filename=args.image_cond_file,
        image_cond_key=args.image_cond_key,
        image_cond_array_key=args.image_cond_array_key,
        image_cond_dim=args.image_cond_dim,
        image_cond_max_tokens_per_view=args.image_cond_max_tokens_per_view,
        point_index_mode=args.index_mode,
        num_parts=model_n_parts,
    )

    log_string(
        f'[data] train={len(train_dataset)} ({"memmap" if train_dataset.memmap_backend is not None else "source"}) | '
        f'validation={len(val_dataset)} ({"memmap" if val_dataset.memmap_backend is not None else "source"})'
    )

    model_kwargs = {
        'n_parts': model_n_parts,
        'flow_feature_dim': flow_feature_dim,
        'condition_feature_dim': condition_feature_dim,
        'image_condition_dim': args.image_cond_dim,
        'image_cond_max_tokens_per_view': args.image_cond_max_tokens_per_view,
        'num_prototypes': num_prototypes,
        'coarse_num_prototypes': coarse_num_prototypes,
        'prototype_dim': prototype_dim,
        'prototype_kmeans_iters': prototype_kmeans_iters,
        'prototype_assign_temperature': prototype_assign_temperature,
        'use_image_controller': (not args.disable_image_cond),
        'fm_ctrl_weight': args.loss_fm_ctrl_weight,
        'prototype_consistency_weight': args.loss_proto_consistency_weight,
        'local_loss_weight': args.local_loss_weight,
        'local_loss_final_stage_weight': args.local_loss_final_stage_weight,
    }

    use_shard_sampler = bool(
        args.distributed
        and train_dataset.memmap_backend is not None
        and getattr(train_dataset.memmap_backend, 'is_sharded', False)
        and getattr(train_dataset.memmap_backend, 'num_shards', 1) > 1
    )
    train_sampler = (
        ShardAwareDistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            drop_last=True,
        )
        if use_shard_sampler else
        DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
            drop_last=True,
        )
    ) if args.distributed else None
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=False,
        drop_last=False,
    ) if args.distributed else None

    def create_train_dataloader():
        return build_dataloader(
            dataset=train_dataset,
            batch_size=local_batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            drop_last=True,
            persistent_workers=train_persistent_workers,
            multiprocessing_context=dataloader_context,
        )

    def create_val_dataloader():
        return build_dataloader(
            dataset=val_dataset,
            batch_size=local_val_batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=val_num_workers,
            prefetch_factor=val_prefetch_factor,
            drop_last=False,
            persistent_workers=val_persistent_workers,
            multiprocessing_context=dataloader_context,
        )

    if is_main_process:
        log_string(f'Train sampler route: {"SHARD-AWARE-DISTRIBUTED" if use_shard_sampler else "STANDARD-DISTRIBUTED"}',
            detail=True
        )
        if not args.disable_image_cond:
            for split_name, dataset in (('train', train_dataset), ('test', val_dataset)):
                backend = getattr(dataset, 'memmap_backend', None)
                if backend is None:
                    log_string(
                        f'Image-cond cache status | split={split_name} source=npz '
                        f'(no matching memmap array)',
                        detail=True
                    )
                    continue
                if backend.image_cond_matches(
                    key=args.image_cond_key,
                    filename=args.image_cond_file,
                    array_key=args.image_cond_array_key,
                ):
                    log_string(
                        f'Image-cond cache status | split={split_name} source=memmap '
                        f'cache={args.image_cond_array_key}:{args.image_cond_key}',
                        detail=True
                    )
                else:
                    meta = getattr(backend, 'image_cond_meta', {}) or {}
                    log_string(
                        f'Image-cond cache status | split={split_name} source=npz-fallback '
                        f'requested={args.image_cond_array_key}:{args.image_cond_file}:{args.image_cond_key} '
                        f'cache={meta.get("array_key", "<none>")}:{meta.get("filename", "<none>")}:{meta.get("key", "<none>")}',
                        detail=True
                    )

    dataloader = create_train_dataloader()
    val_dataloader = create_val_dataloader()

    model = ControlFlow3D(**model_kwargs).to(device)
    raw_model = model
    if is_main_process:
        log_string(
            f'Model configuration | n_parts={raw_model.n_parts} '
            f'flow_feature_dim={raw_model.flow_feature_dim} condition_feature_dim={raw_model.condition_feature_dim} '
            f'image_condition_dim={raw_model.image_condition_dim} image_cond_max_tokens_per_view={raw_model.image_cond_max_tokens_per_view} '
            f'coarse_num_prototypes={raw_model.coarse_num_prototypes} num_prototypes={raw_model.num_prototypes} prototype_dim={raw_model.prototype_dim} '
            f'prototype_kmeans_iters={raw_model.prototype_kmeans_iters} '
            f'prototype_assign_temperature={raw_model.prototype_assign_temperature} '
            f'prototype_consistency_weight={raw_model.prototype_consistency_weight} '
            f'local_loss_weight={raw_model.local_loss_weight} '
            f'local_loss_final_stage_weight={raw_model.local_loss_final_stage_weight} '
            f'use_image_controller={int(raw_model.use_image_controller)}',
            detail=True
        )
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    if args.distributed:
        model = DDP(
            raw_model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
        if is_main_process:
            log_string('DDP config: find_unused_parameters=1 (enabled for optional controller/aux branches).',
                detail=True
            )

    start_epoch = 0
    best_cd = float('inf')

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        unwrap_model(model).load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_cd = checkpoint['best_cd']
        log_string(f'[resume] epoch={start_epoch} | best_CD={best_cd:.6f}')

    global_step = start_epoch * len(dataloader)

    def run_validation():
        model.eval()
        model_obj = unwrap_model(model)
        cd_module = model_obj.chamfer_loss
        set_runtime_inference_params(model_obj)
        # CD is the sum of directed mean squared nearest-neighbor distances.
        totals = torch.zeros(3, dtype=torch.float64, device=device)

        # Validate without image conditions.
        use_image_cond = False
        val_tag = 'no-image'
        iterator = tqdm(val_dataloader, desc=f'Validation({val_tag})', leave=False, disable=not is_main_process)
        with torch.no_grad():
            for batch in iterator:
                image_cond = None
                core_batch = batch
                if use_image_cond and (
                    len(batch) >= 6
                    and torch.is_tensor(batch[-1])
                    and batch[-1].dim() >= 2
                    and batch[-1].shape[-1] >= 64
                ):
                    image_cond = batch[-1].to(device, non_blocking=True)
                    core_batch = batch[:-1]

                if len(core_batch) < 4:
                    raise RuntimeError(f'Validation batch too short: {len(core_batch)}')

                sparse_xyz = core_batch[0].to(device, non_blocking=True)
                sparse_feat = core_batch[1].to(device, non_blocking=True)
                gt_xyz = core_batch[3].to(device, non_blocking=True)
                rest = list(core_batch[4:])
                sparse_patch_idx = None
                gt_patch_idx = None
                if rest and torch.is_tensor(rest[0]) and rest[0].dtype in (
                    torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
                ):
                    sparse_patch_idx = rest.pop(0).to(device, non_blocking=True)
                if rest and torch.is_tensor(rest[0]) and rest[0].dtype in (
                    torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
                ):
                    gt_patch_idx = rest.pop(0).to(device, non_blocking=True)

                if args.index_mode == 'part':
                    sparse_patch_idx = None
                    gt_patch_idx = None
                # Build dynamic inference patches instead of using dataset patch indices.
                if args.index_mode == 'patch':
                    sparse_patch_idx = None
                    gt_patch_idx = None

                if args.index_mode == 'patch':
                    pred_xyz, refined_outputs = model_obj._rollout_sparse_only(
                        sparse_xyz=sparse_xyz,
                        sparse_feat=sparse_feat,
                        steps=model_obj.inference_steps,
                        category_ids=None,
                        image_cond=image_cond,
                        sparse_patch_idx=sparse_patch_idx,
                        refiner_index_mode=args.index_mode,
                        add_x0_noise=False,
                        action_noise_std=0.0,
                        return_refined_outputs=True,
                    )
                    if pred_xyz.shape[1] % model_obj.n_parts != 0 or gt_xyz.shape[1] % model_obj.n_parts != 0:
                        local_loss = 0.0
                    else:
                        local_loss = model_obj._compute_local_cd(pred_xyz, gt_xyz).mean().item()
                else:
                    pred_xyz = model(
                        sparse_xyz,
                        sparse_feat,
                        category_ids=None,
                        train=False,
                        sparse_patch_idx=sparse_patch_idx,
                        image_cond=image_cond,
                        refiner_index_mode=args.index_mode,
                    )
                    if pred_xyz.shape[1] % model_obj.n_parts != 0 or gt_xyz.shape[1] % model_obj.n_parts != 0:
                        local_loss = 0.0
                    else:
                        local_loss = model_obj._compute_local_cd(pred_xyz, gt_xyz).mean().item()

                d1, d2, _, _ = cd_module(pred_xyz, gt_xyz)
                global_cd = (torch.mean(d1, dim=1) + torch.mean(d2, dim=1)).mean().item()

                totals[0] += global_cd
                totals[1] += local_loss
                totals[2] += 1.0

        if args.world_size > 1:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)

        model.train()
        denom = max(float(totals[2].item()), 1.0)
        return (
            float(totals[0].item() / denom),
            float(totals[1].item() / denom),
        )

    try:
        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            model_obj = unwrap_model(model)
            fm_weight = compute_scheduled_weight(
                epoch,
                args.epochs,
                args.loss_fm_ctrl_weight,
                args.loss_fm_ctrl_weight_end,
                args.loss_weight_schedule,
                warmup_epochs=args.loss_weight_warmup_epochs,
            )
            proto_weight = compute_scheduled_weight(
                epoch,
                args.epochs,
                args.loss_proto_consistency_weight,
                args.loss_proto_consistency_weight_end,
                args.loss_weight_schedule,
                warmup_epochs=0,
            )
            model_obj.fm_ctrl_weight = fm_weight
            model_obj.prototype_consistency_weight = proto_weight
            if is_main_process:
                log_string(
                    f'Epoch {epoch + 1} configured loss weights | fm={fm_weight:.6f} proto={proto_weight:.6f}',
                    detail=True
                )
            total_loss = 0.0
            # Report global CD separately from local supervision.
            total_cd = 0.0
            total_cd_total = 0.0
            total_cd_global = 0.0
            total_local_loss = 0.0
            total_fm_ctrl = 0.0
            total_proto = 0.0
            steps_this_epoch = 0

            pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{args.epochs}', disable=not is_main_process)
            for i, batch in enumerate(pbar):
                (
                    sparse_xyz,
                    sparse_feat,
                    gt_feat,
                    gt_xyz,
                    image_cond,
                    _,
                    categories,
                    sparse_patch_idx,
                    gt_patch_idx,
                ) = split_data(batch, device)

                if args.index_mode == 'part':
                    sparse_patch_idx = None
                    gt_patch_idx = None
                elif sparse_patch_idx is None or gt_patch_idx is None:
                    raise RuntimeError(
                        'Training in patch mode requires sparse_patch_idx and gt_patch_idx; they are missing from the current batch.'
                    )

                if gt_feat.numel() == 0:
                    raise RuntimeError('Training requires GT features, but the current batch is missing gt_feat data.')
                if (not args.disable_image_cond) and image_cond is None:
                    raise RuntimeError('Training has image_cond enabled, but the current batch does not provide image conditions.')

                loss, loss_cd, loss_fm_ctrl, loss_proto, loss_cd_global_batch, local_loss_batch = model(
                    sparse_xyz,
                    sparse_feat,
                    gt_feat,
                    gt_xyz,
                    sparse_patch_idx=sparse_patch_idx,
                    gt_patch_idx=gt_patch_idx,
                    image_cond=image_cond,
                    refiner_index_mode=args.index_mode,
                    return_loss_breakdown=True,
                )
                loss = loss.mean()
                loss_cd = loss_cd.mean()
                loss_cd_global_batch = loss_cd_global_batch.mean()
                local_loss_batch = local_loss_batch.mean()
                loss_fm_ctrl = loss_fm_ctrl.mean()
                loss_proto = loss_proto.mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                total_cd += loss_cd_global_batch.item()
                total_cd_total += loss_cd.item()
                total_cd_global += loss_cd_global_batch.item()
                total_local_loss += local_loss_batch.item()
                total_fm_ctrl += loss_fm_ctrl.item()
                total_proto += loss_proto.item()
                steps_this_epoch += 1

                if i % log_every == 0:
                    metrics = torch.stack([
                        loss.detach(),
                        loss_cd_global_batch.detach(),
                        loss_fm_ctrl.detach(),
                        loss_proto.detach(),
                        loss_cd.detach(),
                        local_loss_batch.detach(),
                    ])
                    metrics = reduce_tensor_mean(metrics, args.world_size)
                    if is_main_process:
                        pbar.set_postfix({
                            'Loss': f'{metrics[0].item():.4f}',
                            'CD': f'{metrics[1].item():.6f}',
                            'FM': f'{metrics[2].item():.6f}',
                            'Proto': f'{metrics[3].item():.6f}',
                        })
                        tb.add_scalar('Train/Loss_Total', metrics[0].item(), global_step)
                        tb.add_scalar('Train/CD_Global', metrics[1].item(), global_step)
                        tb.add_scalar('Train/Loss_FM_Ctrl', metrics[2].item(), global_step)
                        tb.add_scalar('Train/Loss_ProtoConsistency', metrics[3].item(), global_step)
                        tb.add_scalar('Train/CD_Total', metrics[4].item(), global_step)
                        tb.add_scalar('Train/Local_Loss', metrics[5].item(), global_step)

                if is_main_process:
                    global_step += 1

                if max_train_steps > 0 and (i + 1) >= max_train_steps:
                    if is_main_process:
                        pbar.write(f'Reached max_train_steps={max_train_steps}, stopping epoch early for debug.')
                    break

            scheduler.step()

            epoch_stats = reduce_metrics_sums(
                [
                    total_loss,
                    total_cd,
                    total_cd_total,
                    total_cd_global,
                    total_local_loss,
                    total_fm_ctrl,
                    total_proto,
                    steps_this_epoch,
                ],
                device=device,
                world_size=args.world_size,
            )
            denom = max(epoch_stats[7].item(), 1.0)
            avg_loss = epoch_stats[0].item() / denom
            avg_cd = epoch_stats[1].item() / denom
            avg_cd_total = epoch_stats[2].item() / denom
            avg_cd_global = epoch_stats[3].item() / denom
            avg_local_loss = epoch_stats[4].item() / denom
            avg_fm_ctrl = epoch_stats[5].item() / denom
            avg_proto = epoch_stats[6].item() / denom

            log_string(
                f'Epoch {epoch + 1} | Loss: {avg_loss:.4f}  CD(Global,L2): {avg_cd:.6f} '
                f'(Total:{avg_cd_total:.6f}, G:{avg_cd_global:.6f}, Local:{avg_local_loss:.6f}) '
                f'FM: {avg_fm_ctrl:.6f} Proto: {avg_proto:.6f}',
                summary=f'[train] Epoch {epoch + 1}/{args.epochs} | Loss={avg_loss:.4f} | CD={avg_cd:.6f} | FM={avg_fm_ctrl:.6f} | Proto={avg_proto:.6f}'
            )
            if is_main_process:
                tb.add_scalar('Train/Epoch_Loss_Total', avg_loss, epoch)
                tb.add_scalar('Train/Epoch_CD_Global', avg_cd, epoch)
                tb.add_scalar('Train/Epoch_CD_Total', avg_cd_total, epoch)
                tb.add_scalar('Train/Epoch_Local_Loss', avg_local_loss, epoch)
                tb.add_scalar('Train/Epoch_Loss_FM', avg_fm_ctrl, epoch)
                tb.add_scalar('Train/Epoch_Loss_ProtoConsistency', avg_proto, epoch)

            val_cd = None
            val_local_loss = None
            current_epoch = epoch + 1
            if current_epoch >= late_stage_start_epoch:
                effective_val_freq = late_stage_val_freq
            elif current_epoch >= mid_stage_start_epoch:
                effective_val_freq = mid_stage_val_freq
            else:
                effective_val_freq = early_stage_val_freq
            if should_validate(current_epoch, args.epochs, effective_val_freq):
                val_cd, val_local_loss = run_validation()

                shutdown_dataloader_workers(val_dataloader)
                gc.collect()
                if empty_cache_after_val and device.type == 'cuda':
                    torch.cuda.synchronize(device)
                    torch.cuda.empty_cache()
                if reload_train_loader_after_val and num_workers > 0:
                    shutdown_dataloader_workers(dataloader)
                    del dataloader
                    dataloader = create_train_dataloader()
                    gc.collect()
                    if is_main_process:
                        log_string('Validation finished; rebuilt train DataLoader to avoid post-val worker stalls.',
                            detail=True
                        )
                if args.world_size > 1:
                    dist.barrier()

                val_tag = 'no-image'
                log_string(
                    f'Val CD(Global,L2) | {val_tag}: {val_cd:.6f} (Local:{val_local_loss:.6f}) '
                    f'[freq={effective_val_freq}] (best so far: {best_cd:.6f})',
                    summary=f'[val] Epoch {current_epoch} | CD={val_cd:.6f} | Local={val_local_loss:.6f}'
                )

                if is_main_process:
                    tb.add_scalar('Val/CD_Global', val_cd, epoch)
                    tb.add_scalar('Val/Local_Loss', val_local_loss, epoch)

            new_best_val = val_cd is not None and val_cd < best_cd
            if new_best_val:
                best_cd = val_cd

            if is_main_process:
                save_dict = {
                    'epoch': epoch,
                    'model': unwrap_model(model).state_dict(),
                    'model_kwargs': model_kwargs,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'loss': avg_loss,
                    'loss_cd': avg_cd,
                    'loss_fm': avg_fm_ctrl,
                    'loss_proto_consistency': avg_proto,
                    'val_cd': val_cd,
                    'best_cd': best_cd,
                    'coord_normalization': 'sparse_input_centroid_furthest_distance',
                }
                if val_local_loss is not None:
                    save_dict['val_local_loss'] = val_local_loss
                    save_dict['val_cd_part'] = val_local_loss

                checkpoint_events = []
                log_string(f'Checkpoint save begin | epoch={current_epoch} target=last',
                    detail=True
                )
                save_t0 = time.perf_counter()
                save_checkpoint_atomic(
                    os.path.join(run_dir, 'last_checkpoint.pth'),
                    save_dict,
                    device=device,
                )
                checkpoint_events.append(f'last={time.perf_counter() - save_t0:.2f}s')
                log_string(f'Checkpoint save done | epoch={current_epoch} target=last',
                    detail=True
                )

                if new_best_val:
                    log_string(f'Checkpoint save begin | epoch={current_epoch} target=best_val',
                        detail=True
                    )
                    save_t0 = time.perf_counter()
                    save_checkpoint_atomic(
                        os.path.join(run_dir, 'best_val_checkpoint.pth'),
                        save_dict,
                        device=device,
                    )
                    checkpoint_events.append(f'best_val={time.perf_counter() - save_t0:.2f}s')
                    log_string(f'Checkpoint save done | epoch={current_epoch} target=best_val',
                        detail=True
                    )
                    log_string(f'[checkpoint] best_val_checkpoint.pth | CD: {best_cd:.6f}')

                if checkpoint_events:
                    log_string(f'Checkpoint IO | epoch={current_epoch} ' + ' '.join(checkpoint_events),
                        detail=True
                    )

            if args.world_size > 1:
                sync = torch.tensor([best_cd], dtype=torch.float64, device=device)
                dist.broadcast(sync, src=0)
                best_cd = float(sync[0].item())

        log_string(f'[train] Finished | output: {run_dir}')
    finally:
        shutdown_dataloader_workers(locals().get('dataloader'))
        shutdown_dataloader_workers(locals().get('val_dataloader'))
        if tb is not None:
            tb.close()
        if log_fout is not None:
            log_fout.close()
        maybe_destroy_process_group()


if __name__ == '__main__':
    main()
