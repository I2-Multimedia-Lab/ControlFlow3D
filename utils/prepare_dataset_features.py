#!/usr/bin/env python3
"""Run Utonia/VGGT feature export and validation for dataset samples.

The --utonia_python and --vggt_python options select separate encoder
interpreters.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.dataset_feature_layout import collect_dataset_samples
from utils.dataset_feature_layout import DEFAULT_UTONIA_CKPT, DEFAULT_VGGT_CKPT, require_local_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_root', default='./data/ShapeNetPU')
    parser.add_argument('--split', choices=['train', 'test', 'all'], default='train')
    parser.add_argument('--encoders', choices=['all', 'utonia', 'vggt'], default='all')
    parser.add_argument('--utonia_python', default=sys.executable, help='Python executable with Utonia installed')
    parser.add_argument('--vggt_python', default=sys.executable, help='Python executable with VGGT installed')
    parser.add_argument('--utonia_checkpoint', default=str(DEFAULT_UTONIA_CKPT))
    parser.add_argument('--vggt_checkpoint', default=str(DEFAULT_VGGT_CKPT))
    parser.add_argument('--utonia_projector_ckpt', default='')
    parser.add_argument('--utonia_seed', type=int, default=42)
    parser.add_argument('--utonia_scale', type=float, default=1.0)
    parser.add_argument('--utonia_normalize_coord', action='store_true')
    parser.add_argument('--no_utonia_apply_z_positive', action='store_true')
    parser.add_argument('--disable_flash', action='store_true')
    parser.add_argument('--render_root', default='', help='Optional override; dataset-bundled views are auto-discovered by default')
    parser.add_argument('--vggt_feature_reduction', choices=['mean-pool', 'projection'], default='mean-pool',
                        help='mean-pool explicitly defines a new deterministic export; use the original projection for historical features')
    parser.add_argument('--vggt_projection_file', default='')
    parser.add_argument('--vggt_token_layer', type=int, default=-1)
    parser.add_argument('--view_indices', default='')
    parser.add_argument('--preprocess_mode', choices=['crop', 'pad'], default='crop')
    parser.add_argument('--num_views', type=int, default=8)
    parser.add_argument('--max_tokens_per_view', type=int, default=32)
    parser.add_argument('--feature_dtype', choices=['float16', 'float32'], default='float16')
    parser.add_argument('--include_gt_features', action='store_true', help='Also generate training-only gt/gt_feat.npy; not one of the three input artifacts')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu_ids', default='0')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--max_samples', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    if args.max_samples < 0 or args.num_workers < 0 or args.batch_size <= 0 or args.utonia_scale <= 0:
        parser.error('Invalid max_samples/num_workers/batch_size/utonia_scale')
    if args.num_views <= 0 or args.max_tokens_per_view < 0:
        parser.error('Invalid view/token counts')
    if args.include_gt_features and args.encoders == 'vggt':
        parser.error('--include_gt_features requires the Utonia encoder')
    if args.vggt_feature_reduction == 'projection' and not args.vggt_projection_file:
        parser.error('projection requires --vggt_projection_file')
    return args


def _optional_path(path):
    return str(Path(path).expanduser().resolve()) if path else ''


def build_commands(args):
    root = str(Path(args.dataset_root).expanduser().resolve())
    commands = []
    common = ['--dataset_root', root, '--split', args.split,
              '--device', args.device, '--gpu_ids', args.gpu_ids,
              '--max_samples', str(args.max_samples)]
    if args.amp:
        common += ['--amp']
    if args.overwrite:
        common += ['--overwrite']
    if args.encoders in ('all', 'utonia'):
        command = [args.utonia_python, str(ROOT_DIR / 'utils/generate_utonia_dataset_features.py'),
                   *common, '--checkpoint', _optional_path(args.utonia_checkpoint),
                   '--target', 'both' if args.include_gt_features else 'sparse',
                   '--batch_size', str(args.batch_size), '--num_workers', str(args.num_workers),
                   '--seed', str(args.utonia_seed), '--scale', str(args.utonia_scale)]
        if args.utonia_projector_ckpt:
            command += ['--projector_ckpt', _optional_path(args.utonia_projector_ckpt)]
        if args.utonia_normalize_coord:
            command += ['--normalize_coord']
        if args.no_utonia_apply_z_positive:
            command += ['--no_apply_z_positive']
        if args.disable_flash:
            command += ['--disable_flash']
        commands.append(('utonia', command))
    if args.encoders in ('all', 'vggt'):
        command = [args.vggt_python, str(ROOT_DIR / 'utils/generate_vggt_features.py'),
                   *common, '--checkpoint', _optional_path(args.vggt_checkpoint),
                   '--feature_reduction', args.vggt_feature_reduction,
                   '--token_layer', str(args.vggt_token_layer), '--num_views', str(args.num_views),
                   '--max_tokens_per_view', str(args.max_tokens_per_view),
                   '--feature_dtype', args.feature_dtype, '--preprocess_mode', args.preprocess_mode]
        if args.render_root:
            command += ['--render_root', _optional_path(args.render_root)]
        if args.vggt_projection_file:
            command += ['--projection_file', _optional_path(args.vggt_projection_file)]
        if args.view_indices:
            command += ['--view_indices', args.view_indices]
        commands.append(('vggt', command))
    for encoder, _ in list(commands):
        python = args.utonia_python if encoder == 'utonia' else args.vggt_python
        check = [python, str(ROOT_DIR / 'utils/check_dataset_features.py'),
                 '--dataset_root', root, '--split', args.split, '--encoders', encoder,
                 '--num_points', '0', '--max_samples', str(args.max_samples)]
        if encoder == 'vggt':
            check += ['--num_views', str(args.num_views), '--max_tokens_per_view', str(args.max_tokens_per_view),
                      '--feature_dtype', args.feature_dtype]
        elif args.include_gt_features:
            check += ['--include_gt_features']
        commands.append((encoder + '-check', check))
    return commands


def run_commands(commands):
    for stage, command in commands:
        print(f'[{stage}] {shlex.join(command)}', flush=True)
        subprocess.run(command, cwd=ROOT_DIR, check=True)


def main():
    args = parse_args()
    samples = collect_dataset_samples(args.dataset_root, args.split, args.max_samples)
    if args.encoders in ('all', 'utonia'):
        for sample in samples:
            sample.point_path('sparse')
            if args.include_gt_features and sample.split == 'train':
                sample.point_path('gt')
    if args.encoders in ('all', 'vggt'):
        from utils.generate_vggt_features import ordered_image_paths
        for sample in samples:
            from utils.dataset_feature_layout import DEFAULT_IMAGE_COND_FILENAME
            if args.overwrite or not (sample.input_dir / DEFAULT_IMAGE_COND_FILENAME).is_file():
                ordered_image_paths(sample.rendering_dir(args.render_root), args.num_views, args.view_indices)
        if args.vggt_feature_reduction == 'mean-pool':
            print('[config] For NEW VGGT exports, mean-pool is a deterministic 512-channel reduction, not a claim of historical numerical equivalence.', flush=True)
    commands = build_commands(args)
    print(f'[dataset] {len(samples)} samples; default training artifacts: input/sparse_local_feat.npy, input/vggt_multiview_features.npz, input/vggt_feature_meta.json', flush=True)
    if args.dry_run:
        for stage, command in commands:
            print(f'[plan:{stage}] {shlex.join(command)}')
        return
    # Validate required weights before launching encoder processes.
    if args.encoders in ('all', 'utonia'):
        needed = any(args.overwrite or not sample.sparse_feature_path.is_file()
                     or (args.include_gt_features and sample.split == 'train' and not sample.gt_feature_path.is_file())
                     for sample in samples)
        if needed:
            require_local_checkpoint(args.utonia_checkpoint, encoder='Utonia')
            if args.utonia_projector_ckpt and not Path(args.utonia_projector_ckpt).expanduser().is_file():
                raise FileNotFoundError(args.utonia_projector_ckpt)
    if args.encoders in ('all', 'vggt'):
        from utils.dataset_feature_layout import DEFAULT_IMAGE_COND_FILENAME
        needed = any(args.overwrite or not (sample.input_dir / DEFAULT_IMAGE_COND_FILENAME).is_file() for sample in samples)
        if needed:
            require_local_checkpoint(args.vggt_checkpoint, encoder='VGGT')
            if args.vggt_feature_reduction == 'projection' and not Path(args.vggt_projection_file).expanduser().is_file():
                raise FileNotFoundError(args.vggt_projection_file)
    if any((Path(args.dataset_root).expanduser().resolve() / '.controlflow3d_memmap' / split / 'manifest.json').is_file()
           for split in {sample.split for sample in samples}):
        print('[cache] Existing memmaps are not rewritten here. Rebuild them after changing source features to avoid stale tensors.', flush=True)
    run_commands(commands)
    print('[done] Requested feature artifacts were generated and validated; existing valid files were preserved.', flush=True)


if __name__ == '__main__':
    main()
