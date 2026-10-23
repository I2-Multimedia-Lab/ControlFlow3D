#!/usr/bin/env python3
"""Export VGGT image patch tokens and per-sample metadata.

Channel reduction uses mean pooling or a supplied projection matrix.
The encoder, checkpoint and rendered views are provided separately.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.runtime_utils import preset_cuda_visible_devices_from_argv
from utils.dataset_feature_layout import (
    DEFAULT_VGGT_CKPT, DEFAULT_IMAGE_COND_FILENAME, DEFAULT_IMAGE_COND_KEY,
    DEFAULT_IMAGE_COND_DIM, DEFAULT_IMAGE_COND_MAX_TOKENS, DEFAULT_IMAGE_META_FILENAME, require_local_checkpoint,
)

from utils.feature_io import sync_vggt_metadata, write_vggt_export
from utils.dataset_feature_layout import collect_dataset_samples


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_root', default='./data/ShapeNetPU')
    parser.add_argument('--split', choices=['train', 'test', 'all'], default='train')
    parser.add_argument('--render_root', default='', help='Optional override for the bundled image root; otherwise auto-discover the dataset renderings')
    parser.add_argument('--checkpoint', default=str(DEFAULT_VGGT_CKPT), help='Manually downloaded local VGGT model.pt')
    parser.add_argument('--feature_reduction', choices=['projection', 'mean-pool'], required=True,
                        help='projection uses the original training projection; mean-pool is an explicit new export configuration')
    parser.add_argument('--projection_file', default='', help='NPY matrix [D,512], or NPZ matrix [D,512]/weight [512,D] with optional bias [512]')
    parser.add_argument('--token_layer', type=int, default=-1, help='Aggregator output layer index; -1 selects the final layer')
    parser.add_argument('--num_views', type=int, default=8)
    parser.add_argument('--view_indices', default='', help='Optional comma-separated indices into the ordered rendering list')
    parser.add_argument('--max_tokens_per_view', type=int, default=DEFAULT_IMAGE_COND_MAX_TOKENS)
    parser.add_argument('--preprocess_mode', choices=['crop', 'pad'], default='crop')
    parser.add_argument('--output_name', default=DEFAULT_IMAGE_COND_FILENAME)
    parser.add_argument('--feature_dtype', choices=['float16', 'float32'], default='float16', help='Match released features: float16 patch_tokens')
    parser.add_argument('--meta_name', default=DEFAULT_IMAGE_META_FILENAME)
    parser.add_argument('--dry_run', action='store_true', help='Show planned outputs without loading weights or writing files')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu_ids', default='0')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--max_samples', type=int, default=0, help='0 means all samples')
    args = parser.parse_args()
    if args.num_views <= 0 or args.max_tokens_per_view < 0 or args.max_samples < 0:
        parser.error('num_views must be positive; max_tokens_per_view and max_samples must be nonnegative')
    if args.feature_reduction == 'projection' and not args.projection_file:
        parser.error('--feature_reduction projection requires --projection_file from the original training export')
    if Path(args.output_name).name != args.output_name or not args.output_name.endswith('.npz'):
        parser.error('output_name must be a .npz basename')
    if Path(args.meta_name).name != args.meta_name or not args.meta_name.endswith('.json'):
        parser.error('meta_name must be a .json basename')
    return args


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_image_paths(render_dir, num_views=8, view_indices=''):
    render_dir = Path(render_dir).expanduser().resolve()
    listing = render_dir / 'renderings.txt'
    if listing.is_file():
        names = [line.strip() for line in listing.read_text(encoding='utf-8').splitlines() if line.strip()]
        paths = [(render_dir / name).resolve() for name in names]
    else:
        paths = sorted(p for p in render_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    if any(render_dir not in path.parents for path in paths):
        raise ValueError(f'Rendering list must contain files within {render_dir}')
    if view_indices:
        indices = [int(value.strip()) for value in view_indices.split(',')]
        if len(indices) != num_views or len(set(indices)) != num_views or any(i < 0 or i >= len(paths) for i in indices):
            raise ValueError(f'view_indices must select {num_views} distinct valid rendering indices')
        paths = [paths[i] for i in indices]
    else:
        if len(paths) < num_views:
            raise ValueError(f'Need {num_views} views, found {len(paths)} in {render_dir}')
        paths = paths[:num_views]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing rendering files: {missing}')
    return paths


def reduce_feature_dim(tokens, mode, projection_file=''):
    import numpy as np
    tokens = np.asarray(tokens, dtype=np.float32)
    if tokens.ndim != 3 or any(size <= 0 for size in tokens.shape) or not np.isfinite(tokens).all():
        raise ValueError('Expected finite [views,tokens,channels] VGGT features')
    source_dim = tokens.shape[-1]
    if mode == 'mean-pool':
        if source_dim % DEFAULT_IMAGE_COND_DIM:
            raise ValueError(f'{source_dim} channels cannot be grouped evenly into {DEFAULT_IMAGE_COND_DIM}')
        out = tokens.reshape(*tokens.shape[:-1], DEFAULT_IMAGE_COND_DIM, source_dim // DEFAULT_IMAGE_COND_DIM).mean(axis=-1)
    elif mode == 'projection':
        path = Path(projection_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'Original VGGT feature projection not found: {path}')
        bias = None
        if path.suffix.lower() == '.npy':
            matrix = np.load(path, allow_pickle=False)
        elif path.suffix.lower() == '.npz':
            with np.load(path, allow_pickle=False) as package:
                if 'matrix' in package:
                    matrix = np.asarray(package['matrix'])
                elif 'weight' in package:
                    matrix = np.asarray(package['weight']).T
                else:
                    raise KeyError('Projection NPZ must contain matrix [D,512] or weight [512,D]')
                if 'bias' in package:
                    bias = np.asarray(package['bias'], dtype=np.float32)
        else:
            raise ValueError('Projection file must be .npy or .npz')
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.shape != (source_dim, DEFAULT_IMAGE_COND_DIM) or not np.isfinite(matrix).all():
            raise ValueError(f'Projection must be finite [{source_dim},512], got {matrix.shape}')
        out = tokens @ matrix
        if bias is not None:
            if bias.shape != (DEFAULT_IMAGE_COND_DIM,) or not np.isfinite(bias).all():
                raise ValueError('Projection bias must be finite [512]')
            out = out + bias
    else:
        raise ValueError(f'Unknown feature reduction: {mode}')
    if not np.isfinite(out).all():
        raise ValueError('Reduced VGGT features contain NaN/Inf')
    return np.ascontiguousarray(out, dtype=np.float32)


def pool_patch_tokens(tokens, max_tokens=32):
    import numpy as np
    tokens = np.asarray(tokens, dtype=np.float32)
    if tokens.ndim != 3 or tokens.shape[1] == 0 or not np.isfinite(tokens).all():
        raise ValueError('Expected finite nonempty [views,tokens,channels] features')
    if max_tokens <= 0 or tokens.shape[1] <= max_tokens:
        return np.ascontiguousarray(tokens)
    boundaries = np.linspace(0, tokens.shape[1], num=max_tokens + 1, dtype=np.int64)
    pooled = np.stack([tokens[:, boundaries[i]:boundaries[i + 1]].mean(axis=1)
                       for i in range(max_tokens)], axis=1)
    return np.ascontiguousarray(pooled, dtype=np.float32)


def collect_samples(dataset_root, splits, render_root='', max_samples=0):
    records = []
    for split in splits:
        for sample in collect_dataset_samples(dataset_root, split):
            records.append((sample.split, sample.category, sample.sample_id,
                            sample.input_dir, sample.rendering_dir(render_root)))
    return records[:max_samples] if max_samples else records


def main():
    args = parse_args()
    splits = ['train', 'test'] if args.split == 'all' else [args.split]
    records = collect_samples(args.dataset_root, splits, args.render_root, args.max_samples)
    pending = []
    for row in records:
        output, meta_path = row[3] / args.output_name, row[3] / args.meta_name
        if args.overwrite or not output.is_file():
            pending.append(row)
        elif not args.dry_run:
            _, mismatches = sync_vggt_metadata(
                output, meta_path, repair=True, num_views=args.num_views,
                max_tokens=args.max_tokens_per_view, feature_dtype=args.feature_dtype,
            )
            if mismatches:
                print(f'[meta] corrected observed fields {mismatches}: {meta_path}', flush=True)
    if args.dry_run:
        for split, category, sample, input_dir, render_dir in pending:
            paths = ordered_image_paths(render_dir, args.num_views, args.view_indices)
            print(f'[plan] {split}/{category}/{sample}: {len(paths)} views -> {input_dir / args.output_name} + {args.meta_name}')
        print(f'[plan] {len(pending)} pending VGGT exports; existing NPZs are preserved')
        return
    if not pending:
        print('[done] all VGGT feature files already exist; shapes, dtype and metadata checked', flush=True)
        return
    checkpoint = require_local_checkpoint(args.checkpoint, encoder='VGGT')
    if args.feature_reduction == 'projection' and not Path(args.projection_file).expanduser().is_file():
        raise FileNotFoundError(args.projection_file)
    preset_cuda_visible_devices_from_argv(default=args.gpu_ids)
    import inspect
    import numpy as np
    import torch
    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
    except ImportError as exc:
        raise ImportError('Install the external facebookresearch/vggt package before feature export; see README.md') from exc
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA for VGGT, but CUDA is unavailable')
    model = VGGT()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    elif isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    del state
    checkpoint_hash = sha256_file(checkpoint)
    aggregator_source_hash = sha256_file(inspect.getfile(type(model.aggregator)))
    projection_hash = sha256_file(Path(args.projection_file).expanduser()) if args.projection_file else None
    amp_dtype = (torch.bfloat16 if device.type == 'cuda' and torch.cuda.get_device_capability(device)[0] >= 8
                 else torch.float16)
    with torch.inference_mode():
        for split, category, sample, input_dir, render_dir in pending:
            paths = ordered_image_paths(render_dir, args.num_views, args.view_indices)
            images = load_and_preprocess_images([str(p) for p in paths], mode=args.preprocess_mode).to(device)
            if images.shape[0] != args.num_views:
                raise RuntimeError('VGGT preprocessing changed the view count')
            amp_context = (torch.autocast(device_type='cuda', dtype=amp_dtype)
                           if args.amp and device.type == 'cuda' else contextlib.nullcontext())
            with amp_context:
                layers, patch_start_idx = model.aggregator(images.unsqueeze(0))
            try:
                layer = layers[args.token_layer]
            except IndexError as exc:
                raise ValueError(f'token_layer {args.token_layer} is outside the aggregator outputs') from exc
            if layer is None or layer.ndim != 4 or layer.shape[0] != 1 or layer.shape[1] != args.num_views:
                raise RuntimeError('Selected aggregator layer is unavailable or not [1,views,tokens,channels]')
            raw_tokens = layer[0, :, int(patch_start_idx):, :].float().cpu().numpy()
            patch_tokens = pool_patch_tokens(
                reduce_feature_dim(raw_tokens, args.feature_reduction, args.projection_file),
                args.max_tokens_per_view,
            )
            if patch_tokens.shape[-1] != DEFAULT_IMAGE_COND_DIM:
                raise RuntimeError('Exported image feature dimension must be 512')
            metadata = {
                'schema': 'controlflow3d.vggt.v1', 'npz_key': DEFAULT_IMAGE_COND_KEY,
                'weights': Path(checkpoint).name,
                'image_preprocess_mode': args.preprocess_mode,
                'patch_shape': list(patch_tokens.shape), 'raw_patch_shape': list(raw_tokens.shape),
                'token_layer': args.token_layer, 'patch_start_idx': int(patch_start_idx),
                'feature_reduction': args.feature_reduction, 'projection_sha256': projection_hash,
                'checkpoint_sha256': checkpoint_hash, 'aggregator_source_sha256': aggregator_source_hash,
                'preprocess_mode': args.preprocess_mode, 'max_tokens_per_view': args.max_tokens_per_view,
                'view_names': [p.name for p in paths], 'image_sha256': [sha256_file(p) for p in paths],
            }
            output = input_dir / args.output_name
            metadata = write_vggt_export(output, patch_tokens, metadata,
                                         feature_dtype=args.feature_dtype, meta_name=args.meta_name)
            print(f'[done] {split}/{category}/{sample}: {tuple(patch_tokens.shape)} -> {output}', flush=True)
            del layers, layer, images, raw_tokens


if __name__ == '__main__':
    main()
