#!/usr/bin/env python3
"""Export point-aligned float32 [N, 128] Utonia features.

Sparse features are saved to input/sparse_local_feat.npy. The --target gt
or --target both options also export dense features to gt/gt_feat.npy.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.runtime_utils import preset_cuda_visible_devices_from_argv
from utils.dataset_feature_layout import collect_dataset_samples
from utils.dataset_feature_layout import DEFAULT_UTONIA_CKPT, require_local_checkpoint
from utils.feature_io import atomic_save_npy, read_aligned_dataset_points


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_root', default='./data/ShapeNetPU')
    parser.add_argument('--split', choices=['train', 'test', 'all'], default='train')
    parser.add_argument('--target', choices=['sparse', 'gt', 'both'], default='sparse')
    parser.add_argument('--checkpoint', default=str(DEFAULT_UTONIA_CKPT))
    parser.add_argument('--projector_ckpt', default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu_ids', default='0')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--normalize_coord', action='store_true')
    parser.add_argument('--no_apply_z_positive', action='store_true')
    parser.add_argument('--disable_flash', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--max_samples', type=int, default=0)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.max_samples < 0 or args.scale <= 0:
        parser.error('batch_size/scale must be positive; num_workers/max_samples must be nonnegative')
    return args


def build_jobs(samples, target='sparse'):
    targets = ('sparse', 'gt') if target == 'both' else (target,)
    if any(t not in ('sparse', 'gt') for t in targets):
        raise ValueError(f'Unknown target: {target!r}')
    jobs = []
    for selected in targets:
        for sample in samples:
            # Export dense target features only for training.
            if selected == 'gt' and sample.split != 'train':
                continue
            output = sample.sparse_feature_path if selected == 'sparse' else sample.gt_feature_path
            jobs.append((sample, selected, sample.point_path(selected), output))
    return jobs


def export_encoded_rows(rows, jobs, allowed_indices=None):
    """Validate and save encoded features in source point order."""
    seen = set()
    for index, features in rows:
        index = int(index)
        if index < 0 or index >= len(jobs) or index in seen or (allowed_indices is not None and index not in allowed_indices):
            raise RuntimeError(f'Encoder returned invalid or duplicate sample index: {index}')
        sample, target, source, output = jobs[index]
        count = len(read_aligned_dataset_points(source))
        atomic_save_npy(output, features, count)
        seen.add(index)
        print(f'[done] {sample.identifier} {target}: [{count},128] float32 -> {output}', flush=True)
    return seen


def main():
    args = parse_args()
    samples = collect_dataset_samples(args.dataset_root, args.split, args.max_samples)
    jobs = build_jobs(samples, args.target)
    pending = [i for i, (_, _, _, out) in enumerate(jobs) if args.overwrite or not out.is_file()]
    if args.dry_run:
        for index in pending:
            _, target, source, output = jobs[index]
            print(f'[plan] {target}: {source} -> {output}')
        print(f'[plan] {len(pending)} pending exports, {len(jobs) - len(pending)} existing files')
        return
    # Validate existing features before reusing them.
    pending_set = set(pending)
    if len(pending) < len(jobs):
        from utils.feature_io import read_sparse_features
        for i, (_, _, source, output) in enumerate(jobs):
            if i not in pending_set:
                read_sparse_features(output, len(read_aligned_dataset_points(source)))
                if jobs[i][1] == 'sparse':
                    from utils.check_dataset_features import validate_input_artifacts
                    validate_input_artifacts(output.parent, encoders='utonia', num_points=0, point_path=source)
    if not pending:
        print('[done] all requested Utonia feature files already exist and are valid', flush=True)
        return
    checkpoint = require_local_checkpoint(args.checkpoint, encoder='Utonia')
    preset_cuda_visible_devices_from_argv(default=args.gpu_ids)
    from torch.utils.data import Subset
    from utils.utonia_feature_encoder import PointArrayDataset, UtoniaSparseFeatureEncoder
    # Preserve original sample indices so resumed exports reuse the same seeds.
    dataset = PointArrayDataset(
        length=len(jobs), load_points=lambda index: read_aligned_dataset_points(jobs[int(index)][2]),
        seed=args.seed, scale=args.scale, normalize_coord=args.normalize_coord,
        apply_z_positive=not args.no_apply_z_positive,
    )
    encoder = UtoniaSparseFeatureEncoder(
        device=args.device, checkpoint=checkpoint, projector_ckpt=args.projector_ckpt,
        disable_flash=args.disable_flash, seed=args.seed,
    )
    rows = encoder.iter_encode_dataset(Subset(dataset, pending), batch_size=args.batch_size,
                                       num_workers=args.num_workers, amp=args.amp)
    written = export_encoded_rows(rows, jobs, pending_set)
    if written != set(pending):
        raise RuntimeError(f'Encoder did not complete all pending feature files: missing={set(pending) - written}')
    print(f'[done] encoded {len(written)} point clouds; existing valid files were preserved', flush=True)


if __name__ == '__main__':
    main()
