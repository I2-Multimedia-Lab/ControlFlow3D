#!/usr/bin/env python3
"""Validate point/image features and optionally update VGGT metadata.

Dense target features are checked with --include_gt_features. Metadata repair
updates JSON fields without changing feature arrays.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.dataset_feature_layout import collect_dataset_samples
from utils.dataset_feature_layout import (
    DEFAULT_SPARSE_FEAT_FILENAME, DEFAULT_GT_FEAT_FILENAME,
    DEFAULT_IMAGE_COND_FILENAME, DEFAULT_IMAGE_META_FILENAME,
)
from utils.feature_io import sync_vggt_metadata, atomic_write_json, read_aligned_dataset_points


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument('--dataset_root')
    location.add_argument('--input_dir', help='Inspect one sample input/ directory without scanning the dataset')
    parser.add_argument('--split', choices=['train', 'test', 'all'], default='train')
    parser.add_argument('--encoders', choices=['all', 'utonia', 'vggt'], default='all')
    parser.add_argument('--include_gt_features', action='store_true')
    parser.add_argument('--num_points', type=int, default=2048, help='Expected sparse rows; 0 uses source point count if present')
    parser.add_argument('--num_views', type=int, default=8)
    parser.add_argument('--max_tokens_per_view', type=int, default=32)
    parser.add_argument('--feature_dtype', choices=['float16', 'float32'], default='float16')
    parser.add_argument('--repair_metadata', action='store_true', help='Correct/create observable JSON fields only, not encoder provenance')
    parser.add_argument('--max_samples', type=int, default=0)
    parser.add_argument('--report', default='')
    args = parser.parse_args()
    if min(args.num_points, args.max_samples, args.max_tokens_per_view) < 0 or args.num_views <= 0:
        parser.error('Counts must be nonnegative and num_views must be positive')
    if args.include_gt_features and not args.dataset_root:
        parser.error('--include_gt_features requires --dataset_root')
    return args


def validate_input_artifacts(input_dir, *, encoders='all', num_points=2048,
                             point_path=None, num_views=8, max_tokens=32,
                             feature_dtype='float16', repair_metadata=False):
    import numpy as np
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    results = {}
    if encoders in ('all', 'utonia'):
        features_path = input_dir / DEFAULT_SPARSE_FEAT_FILENAME
        features = np.load(features_path, allow_pickle=False)
        if features.ndim != 2 or features.shape[1] != 128 or features.shape[0] == 0:
            raise ValueError(f'sparse features must be nonempty [N,128], got {features.shape}: {features_path}')
        if features.dtype != np.dtype('float32') or not np.isfinite(features).all():
            raise ValueError(f'Sparse features must be finite float32: {features_path}')
        if num_points and features.shape[0] != num_points:
            raise ValueError(f'Expected {num_points} sparse rows, got {features.shape[0]}')
        if point_path is not None:
            if len(read_aligned_dataset_points(point_path)) != features.shape[0]:
                raise ValueError(f'Sparse feature rows do not match source points: {features_path}')
        results['sparse_features'] = {'shape': list(features.shape), 'dtype': features.dtype.name}
    if encoders in ('all', 'vggt'):
        observed, mismatches = sync_vggt_metadata(
            input_dir / DEFAULT_IMAGE_COND_FILENAME, input_dir / DEFAULT_IMAGE_META_FILENAME,
            repair=repair_metadata, num_views=num_views, max_tokens=max_tokens,
            feature_dtype=feature_dtype,
        )
        if mismatches and not repair_metadata:
            raise ValueError(f'VGGT JSON does not match the NPZ: {mismatches}; pass --repair_metadata to correct observable fields')
        results['vggt'] = observed
        results['metadata_corrected_fields'] = mismatches if repair_metadata else []
    return results


def main():
    args = parse_args()
    if args.input_dir:
        records = [(str(Path(args.input_dir)), Path(args.input_dir), None)]
    else:
        samples = collect_dataset_samples(args.dataset_root, args.split, args.max_samples)
        records = [(sample.identifier, sample.input_dir, sample) for sample in samples]
    valid, failures = [], []
    for identifier, input_dir, sample in records:
        try:
            point_path = sample.point_path('sparse') if sample and args.encoders in ('all', 'utonia') else None
            result = validate_input_artifacts(
                input_dir, encoders=args.encoders, num_points=args.num_points,
                point_path=point_path, num_views=args.num_views,
                max_tokens=args.max_tokens_per_view, feature_dtype=args.feature_dtype,
                repair_metadata=args.repair_metadata,
            )
            if args.include_gt_features and sample.split == 'train':
                from utils.feature_io import read_sparse_features
                count = len(read_aligned_dataset_points(sample.point_path('gt')))
                read_sparse_features(sample.gt_feature_path, count)
                result['gt_features'] = {'shape': [count, 128], 'path': 'gt/' + DEFAULT_GT_FEAT_FILENAME}
            valid.append({'sample': identifier, **result})
        except (ValueError, KeyError, OSError) as exc:
            failures.append({'sample': identifier, 'error': str(exc)})
            print(f'[invalid] {identifier}: {exc}', file=sys.stderr)
    summary = {'samples': len(records), 'valid': len(valid), 'invalid': len(failures), 'failures': failures}
    print(json.dumps(summary, indent=2))
    if args.report:
        atomic_write_json(args.report, {**summary, 'validated_samples': valid})
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
