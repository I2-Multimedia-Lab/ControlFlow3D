"""Pack dataset coordinates and features into memmap arrays."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from datasetsource.memmap_sample_cache import DEFAULT_MEMMAP_CACHE_DIRNAME, get_split_cache_dir
from datasetsource.ply_io import read_patch_ply_xyz
from utils.dataset_feature_layout import (
    DEFAULT_IMAGE_COND_FILENAME, DEFAULT_IMAGE_COND_KEY, DEFAULT_IMAGE_COND_ARRAY_KEY,
)

POINTCLOUD_BASENAMES = {
    "sparse_path": ("sparse_xyz", "sparse_point_cloud"),
    "gt_path": ("gt_xyz", "upsampled_point_cloud"),
}
SPARSE_FEAT_FILENAME = "sparse_local_feat.npy"
GT_FEAT_FILENAME = "gt_feat.npy"
DEFAULT_REUSE_ARRAY_KEYS = ("sparse_feat", DEFAULT_IMAGE_COND_ARRAY_KEY)
TRAIN_BASE_ARRAY_KEYS = (
    "sparse_xyz",
    "sparse_feat",
    "gt_feat",
    "gt_xyz",
    "sparse_patch_idx",
    "gt_patch_idx",
)
# Dense target features are cached only for training.
TEST_BASE_ARRAY_KEYS = (
    "sparse_xyz",
    "sparse_feat",
    "gt_xyz",
    "sparse_patch_idx",
    "gt_patch_idx",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build memmap cache for ControlFlow3D dataset")
    parser.add_argument("--dataset_root", type=str, default="./data/ShapeNetPU")
    parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"])
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "image-only"],
        help="full: rebuild selected split cache; image-only: only refresh packed image_cond arrays/manifests.",
    )
    parser.add_argument("--feature_dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--index_mode", choices=["part", "patch"], default="part",
                        help="Match train.py: part uses contiguous blocks; patch requires PLY patch-index metadata.")
    parser.add_argument(
        "--include_image_cond",
        action="store_true",
        help="Also pack image conditions for the test split. Train always packs them; test defaults to point-only.",
    )
    parser.add_argument("--image_cond_file", type=str, default=DEFAULT_IMAGE_COND_FILENAME, help="Image condition NPZ filename in each sample input folder.")
    parser.add_argument("--image_cond_key", type=str, default=DEFAULT_IMAGE_COND_KEY, help="Key to read from image condition NPZ.")
    parser.add_argument(
        "--image_cond_spec",
        action="append",
        default=[],
        help=(
            "Pack an image condition array as ARRAY_KEY:NPZ_FILENAME:NPZ_KEY. "
            "May be passed multiple times. When omitted, vggt_image_cond is packed from the patch_tokens NPZ key."
        ),
    )
    parser.add_argument("--image_cond_dtype", type=str, default="float16", choices=["float16", "float32"], help="Storage dtype for packed image condition arrays.")
    parser.add_argument(
        "--image_cond_max_tokens_per_view",
        type=int,
        default=32,
        help="If > 0 and image_cond is [V,T,C], average-pool each view to at most this many tokens before packing.",
    )
    parser.add_argument("--shard_size", type=int, default=2048, help="Samples per shard. 0 disables sharding.")
    parser.add_argument("--shuffle_seed", type=int, default=0, help="Seed used to shuffle samples before sharding. Negative disables reshuffle.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reuse_cache_from",
        type=str,
        default="",
        help=(
            "Reuse selected memmap arrays from an existing cache and skip repacking them. "
            "Provide a dataset root or a .controlflow3d_memmap directory."
        ),
    )
    parser.add_argument(
        "--reuse_array_keys",
        type=str,
        default=",".join(DEFAULT_REUSE_ARRAY_KEYS),
        help=(
            "Comma-separated array keys to reuse from --reuse_cache_from. "
            f"Default: {','.join(DEFAULT_REUSE_ARRAY_KEYS)}"
        ),
    )
    parser.add_argument(
        "--reuse_materialize_mode",
        type=str,
        default="link",
        choices=["link", "copy", "move"],
        help="How to materialize reused arrays into the new cache. link tries hardlink/symlink, copy copies, move moves source files.",
    )
    parser.add_argument("--worker_shards", type=int, default=1, help="How many parallel workers cooperate on the selected split(s).")
    parser.add_argument("--worker_id", type=int, default=0, help="Current worker id in [0, worker_shards).")
    parser.add_argument(
        "--finalize_only",
        action="store_true",
        help="Do not pack arrays; only merge per-shard metadata into manifest.json after all workers finish.",
    )
    return parser.parse_args()


def parse_image_cond_specs(spec_values, default_filename=DEFAULT_IMAGE_COND_FILENAME, default_key=DEFAULT_IMAGE_COND_KEY):
    specs = []
    for raw in spec_values or []:
        parts = str(raw).split(":")
        if len(parts) != 3 or not all(part.strip() for part in parts):
            raise ValueError(
                "--image_cond_spec must be ARRAY_KEY:NPZ_FILENAME:NPZ_KEY, "
                f"got {raw!r}"
            )
        array_key, filename, key = [part.strip() for part in parts]
        specs.append({"array_key": array_key, "filename": filename, "key": key})
    if specs:
        seen = set()
        for spec in specs:
            array_key = spec["array_key"]
            if array_key in seen:
                raise ValueError(f"Duplicate image_cond array key: {array_key}")
            seen.add(array_key)
        return specs
    return [{"array_key": DEFAULT_IMAGE_COND_ARRAY_KEY, "filename": default_filename, "key": default_key}]


def image_cond_array_keys(image_cond_specs):
    return [spec["array_key"] for spec in image_cond_specs]


def resolve_split_array_config(split, include_image_cond, image_cond_specs=None, index_mode='part'):
    image_cond_specs = image_cond_specs or parse_image_cond_specs(())
    if split == 'train':
        array_keys = list(TRAIN_BASE_ARRAY_KEYS)
        pack_image_cond = True
    elif split == 'test':
        array_keys = list(TEST_BASE_ARRAY_KEYS)
        pack_image_cond = bool(include_image_cond)
    else:
        raise ValueError(f"Unsupported split: {split!r}")
    if index_mode == 'part':
        # Part indices are constructed at runtime from contiguous point blocks.
        array_keys = [key for key in array_keys if key not in ('sparse_patch_idx', 'gt_patch_idx')]
    elif index_mode != 'patch':
        raise ValueError(f"Unsupported index mode: {index_mode!r}")
    if pack_image_cond:
        array_keys.extend(image_cond_array_keys(image_cond_specs))
    return {'array_keys': array_keys, 'pack_image_cond': pack_image_cond}


def resolve_pointcloud_path(directory, key, prefer_ply=False):
    directory = Path(directory)
    for base in POINTCLOUD_BASENAMES[key]:
        exts = (".ply", ".npy") if prefer_ply else (".npy", ".ply")
        for ext in exts:
            candidate = directory / f"{base}{ext}"
            if candidate.exists():
                return str(candidate)
    return str(directory / f"{POINTCLOUD_BASENAMES[key][0]}.ply")


def _maybe_prefer_patch_ply(path):
    path = str(path)
    if not path.endswith(".npy"):
        return path
    ply_path = path[:-4] + ".ply"
    return ply_path if os.path.isfile(ply_path) else path


def _image_cond_sample_path_key(array_key):
    return f"{array_key}_path"


def _attach_image_cond_paths(dataset_root, split, sample, image_cond_specs):
    input_dir = Path(dataset_root) / split / str(sample['category']) / str(sample['sample_id']) / 'input'
    for spec in image_cond_specs:
        path = str(input_dir / spec["filename"])
        sample[_image_cond_sample_path_key(spec["array_key"])] = path
        if spec["array_key"] == "image_cond":
            sample["image_cond_path"] = path


def load_index_or_scan(dataset_root, split, include_image_cond=False, image_cond_filename=DEFAULT_IMAGE_COND_FILENAME, image_cond_specs=None):
    image_cond_specs = image_cond_specs or parse_image_cond_specs((), image_cond_filename, DEFAULT_IMAGE_COND_KEY)
    cache_path = Path(dataset_root) / f'.sample_index_external_cond_{split}.json'
    if cache_path.exists():
        with open(cache_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        samples = obj.get('samples', [])
        if samples:
            normalized = []
            for sample in samples:
                item = dict(sample)
                for key in ("sparse_path", "gt_path"):
                    if key in item:
                        item[key] = _maybe_prefer_patch_ply(item[key])
                if include_image_cond:
                    _attach_image_cond_paths(dataset_root, split, item, image_cond_specs)
                normalized.append(item)
            return normalized

    split_dir = Path(dataset_root) / split
    samples = []
    for cat_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
        for sample_dir in sorted([p for p in cat_dir.iterdir() if p.is_dir()]):
            input_dir = sample_dir / 'input'
            gt_dir = sample_dir / 'gt'
            sample = {
                'sample_id': sample_dir.name,
                'category': cat_dir.name,
                'sparse_path': resolve_pointcloud_path(input_dir, 'sparse_path', prefer_ply=True),
                'x0_feat_path': str(input_dir / SPARSE_FEAT_FILENAME),
                'gt_feat_path': str(gt_dir / GT_FEAT_FILENAME),
                'gt_path': resolve_pointcloud_path(gt_dir, 'gt_path', prefer_ply=True),
            }
            if include_image_cond:
                for spec in image_cond_specs:
                    path = str(input_dir / spec["filename"])
                    sample[_image_cond_sample_path_key(spec["array_key"])] = path
                    if spec["array_key"] == "image_cond":
                        sample["image_cond_path"] = path
            samples.append(sample)
    return samples


def parse_reuse_array_keys(value):
    keys = []
    for item in str(value or "").split(","):
        key = item.strip()
        if key:
            keys.append(key)
    return tuple(keys)


def load_xyz_and_patch(path):
    path = str(path)
    if path.endswith('.npy'):
        return np.load(path, allow_pickle=False).astype(np.float32, copy=False), None
    if path.endswith('.ply'):
        return read_patch_ply_xyz(path)
    raise ValueError(f'Unsupported xyz file: {path}')


def load_image_cond(path, key, dtype):
    with np.load(path, allow_pickle=False) as package:
        if key not in package:
            raise KeyError(f"Image condition key='{key}' not found in {path}")
        image_cond = np.asarray(package[key])

    if image_cond.ndim == 2:
        normalized = image_cond
    elif image_cond.ndim == 3:
        normalized = image_cond
    elif image_cond.ndim == 4:
        normalized = image_cond.reshape(image_cond.shape[0], -1, image_cond.shape[-1])
    else:
        raise ValueError(
            f'Image condition must be [V,C] / [V,T,C] / [V,H,W,C], but got {tuple(image_cond.shape)} from {path}'
        )

    if not np.isfinite(normalized).all():
        raise ValueError(f'Image condition has NaN/Inf: {path}')
    return np.ascontiguousarray(normalized, dtype=dtype)


def maybe_pool_image_cond_tokens(image_cond, max_tokens_per_view, dtype):
    arr = np.asarray(image_cond)
    max_tokens = max(int(max_tokens_per_view), 0)
    if max_tokens <= 0 or arr.ndim != 3 or arr.shape[1] <= max_tokens:
        return np.ascontiguousarray(arr, dtype=dtype)

    num_views, token_count, feat_dim = arr.shape
    boundaries = np.linspace(0, token_count, num=max_tokens + 1, dtype=np.int64)
    pooled = np.empty((num_views, max_tokens, feat_dim), dtype=np.float32)
    for idx in range(max_tokens):
        start = int(boundaries[idx])
        end = int(boundaries[idx + 1])
        if end <= start:
            end = min(start + 1, token_count)
        pooled[:, idx, :] = arr[:, start:end, :].mean(axis=1, dtype=np.float32)
    return np.ascontiguousarray(pooled, dtype=dtype)


def _sample_image_cond_path(dataset_root, split, sample, image_cond_filename):
    return str(
        Path(dataset_root)
        / split
        / str(sample['category'])
        / str(sample['sample_id'])
        / 'input'
        / image_cond_filename
    )


def refresh_split_image_only(
    dataset_root,
    split,
    image_cond_filename,
    image_cond_key,
    image_cond_dtype_name,
    image_cond_max_tokens_per_view,
    image_cond_specs=None,
):
    """Refresh named image arrays without changing geometry or sample order."""
    cache_dir = Path(get_split_cache_dir(dataset_root, split))
    manifest_path = cache_dir / 'manifest.json'
    if not manifest_path.is_file():
        raise RuntimeError(f'Missing manifest for split={split}: {manifest_path}')
    with manifest_path.open('r', encoding='utf-8') as handle:
        manifest = json.load(handle)
    if str(manifest.get('split')) != str(split):
        raise ValueError('Image refresh split does not match the cache manifest')
    samples = manifest.get('samples', [])
    if not samples:
        raise RuntimeError(f'No samples in manifest: {manifest_path}')
    specs = image_cond_specs or parse_image_cond_specs((), image_cond_filename, image_cond_key)
    dtype = np.float16 if image_cond_dtype_name == 'float16' else np.float32
    shards = manifest.get('shards') or [{
        'id': 0, 'start_index': 0, 'num_samples': len(samples),
        'arrays': manifest.setdefault('arrays', {}),
    }]
    covered = []
    for shard in shards:
        covered.extend(range(int(shard['start_index']), int(shard['start_index']) + int(shard['num_samples'])))
    if sorted(covered) != list(range(len(samples))):
        raise RuntimeError('Cache shards do not cover every sample exactly once')
    for spec in specs:
        key = spec['array_key']
        first_path = _sample_image_cond_path(dataset_root, split, samples[0], spec['filename'])
        first = maybe_pool_image_cond_tokens(
            load_image_cond(first_path, spec['key'], dtype), image_cond_max_tokens_per_view, dtype)
        shape = tuple(first.shape)
        for shard in shards:
            shard_id = int(shard.get('id', 0))
            start = int(shard['start_index'])
            count = int(shard['num_samples'])
            default_path = (f'shard_{shard_id:05d}/{key}.npy' if manifest.get('shards') else f'{key}.npy')
            relative = shard.setdefault('arrays', {}).get(key, {}).get('path', default_path)
            path = (cache_dir / relative).resolve()
            if cache_dir.resolve() not in path.parents:
                raise ValueError(f'Image array path escapes the cache: {relative}')
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + '.tmp.npy')
            arr = None
            try:
                arr = np.lib.format.open_memmap(temporary, mode='w+', dtype=dtype, shape=(count,) + shape)
                for local_idx in range(count):
                    sample = samples[start + local_idx]
                    image_path = _sample_image_cond_path(dataset_root, split, sample, spec['filename'])
                    value = maybe_pool_image_cond_tokens(
                        load_image_cond(image_path, spec['key'], dtype), image_cond_max_tokens_per_view, dtype)
                    if tuple(value.shape) != shape:
                        raise ValueError(f'Image shapes must be constant within the cache: {image_path}')
                    arr[local_idx] = value
                arr.flush()
                arr._mmap.close()
                arr = None
                os.replace(temporary, path)
            finally:
                if arr is not None:
                    arr._mmap.close()
                if temporary.exists():
                    temporary.unlink()
            shard['arrays'][key] = {
                'path': str(path.relative_to(cache_dir.resolve())),
                'shape': [count] + list(shape), 'dtype': np.dtype(dtype).name,
            }
        if manifest.get('shards'):
            manifest.setdefault('arrays', {})[key] = {
                'shape': [len(samples)] + list(shape), 'dtype': np.dtype(dtype).name,
            }
        manifest.setdefault('image_conds', {})[key] = {
            'enabled': True, 'filename': spec['filename'], 'key': spec['key'],
            'dtype': np.dtype(dtype).name,
            'max_tokens_per_view': max(int(image_cond_max_tokens_per_view), 0),
        }
    primary = next((spec for spec in specs if spec['array_key'] == DEFAULT_IMAGE_COND_ARRAY_KEY), specs[0])
    manifest['image_cond'] = {
        'array_key': primary['array_key'], **manifest['image_conds'][primary['array_key']],
    }
    temporary_manifest = manifest_path.with_name('manifest.json.tmp')
    try:
        temporary_manifest.write_text(json.dumps(manifest), encoding='utf-8')
        os.replace(temporary_manifest, manifest_path)
    finally:
        if temporary_manifest.exists():
            temporary_manifest.unlink()
    print(f'[{split}] image-only refresh done -> {manifest_path}')


def prepare_sample_arrays(
    sample,
    feature_dtype,
    array_keys,
    include_image_cond=False,
    image_cond_key=DEFAULT_IMAGE_COND_KEY,
    image_cond_dtype=np.float16,
    image_cond_max_tokens_per_view=32,
    image_cond_specs=None,
    skip_keys=None,
):
    image_cond_specs = image_cond_specs or parse_image_cond_specs((), DEFAULT_IMAGE_COND_FILENAME, image_cond_key)
    image_cond_keys = set(image_cond_array_keys(image_cond_specs))
    skip_keys = set(skip_keys or [])
    wanted_keys = {key for key in array_keys if key not in skip_keys}

    need_sparse_xyz = 'sparse_xyz' in wanted_keys
    need_sparse_feat = 'sparse_feat' in wanted_keys
    need_sparse_patch_idx = 'sparse_patch_idx' in wanted_keys
    need_gt_xyz = 'gt_xyz' in wanted_keys
    need_gt_feat = 'gt_feat' in wanted_keys
    need_gt_patch_idx = 'gt_patch_idx' in wanted_keys
    need_image_cond_keys = image_cond_keys.intersection(wanted_keys) if include_image_cond else set()

    sparse_xyz = None
    sparse_patch_idx = None
    if need_sparse_xyz or need_sparse_patch_idx:
        sparse_xyz, sparse_patch_idx = load_xyz_and_patch(sample['sparse_path'])
        sparse_xyz = sparse_xyz.astype(np.float32, copy=False)

    sparse_feat = None
    if need_sparse_feat:
        sparse_feat = np.load(sample['x0_feat_path'], allow_pickle=False).astype(feature_dtype, copy=False)

    gt_xyz = None
    gt_patch_idx = None
    if need_gt_xyz or need_gt_patch_idx:
        gt_xyz, gt_patch_idx = load_xyz_and_patch(sample['gt_path'])
        gt_xyz = gt_xyz.astype(np.float32, copy=False)

    gt_feat = None
    if need_gt_feat:
        gt_feat_path = str(sample.get('gt_feat_path', ''))
        if not os.path.isfile(gt_feat_path):
            raise RuntimeError(
                f"Missing required gt_feat file: {gt_feat_path} "
                f"(sample_id={sample.get('sample_id')}). "
                "Generate it with utils/generate_utonia_dataset_features.py --target gt first, "
                "then rebuild the cache."
            )
        gt_feat = np.load(gt_feat_path, allow_pickle=False).astype(feature_dtype, copy=False)

    if need_sparse_patch_idx and sparse_patch_idx is None:
        raise RuntimeError(
            f"Missing sparse patch indices in sample: sparse={sample.get('sparse_path')}"
        )
    if need_gt_patch_idx and gt_patch_idx is None:
        raise RuntimeError(
            f"Missing gt patch indices in sample: gt={sample.get('gt_path')}"
        )

    packed = {}
    if need_sparse_xyz:
        packed['sparse_xyz'] = np.ascontiguousarray(sparse_xyz, dtype=np.float32)
    if need_sparse_feat:
        packed['sparse_feat'] = np.ascontiguousarray(sparse_feat, dtype=feature_dtype)
    if need_gt_xyz:
        packed['gt_xyz'] = np.ascontiguousarray(gt_xyz, dtype=np.float32)
    if need_gt_feat:
        packed['gt_feat'] = np.ascontiguousarray(gt_feat, dtype=feature_dtype)
    if need_sparse_patch_idx:
        packed['sparse_patch_idx'] = np.ascontiguousarray(sparse_patch_idx, dtype=np.int32)
    if need_gt_patch_idx:
        packed['gt_patch_idx'] = np.ascontiguousarray(gt_patch_idx, dtype=np.int32)

    for spec in image_cond_specs:
        array_key = spec["array_key"]
        if array_key not in need_image_cond_keys:
            continue
        image_cond_path = sample.get(_image_cond_sample_path_key(array_key))
        if not image_cond_path and array_key == "image_cond":
            image_cond_path = sample.get("image_cond_path")
        if not image_cond_path:
            raise KeyError(
                f"Sample missing {array_key} path: sample_id={sample.get('sample_id', '<unknown>')}"
            )
        packed[array_key] = maybe_pool_image_cond_tokens(
            load_image_cond(image_cond_path, spec["key"], image_cond_dtype),
            image_cond_max_tokens_per_view,
            image_cond_dtype,
        )
    return packed


def _prepare_cache_dir(cache_dir, overwrite):
    if cache_dir.exists():
        if not overwrite:
            raise RuntimeError(f'Manifest already exists: {cache_dir / "manifest.json"} (use --overwrite)')
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)


def _partial_meta_dir(cache_dir: Path) -> Path:
    return cache_dir / "_partial_meta"


def _write_partial_shard_meta(cache_dir: Path, shard_id: int, payload: dict):
    meta_dir = _partial_meta_dir(cache_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    path = meta_dir / f"shard_{int(shard_id):05d}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _load_partial_shard_meta(cache_dir: Path):
    meta_dir = _partial_meta_dir(cache_dir)
    if not meta_dir.is_dir():
        return {}
    result = {}
    for path in sorted(meta_dir.glob("shard_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        result[int(data["id"])] = data
    return result


def _maybe_shuffle_samples(samples, enable_shuffle, shuffle_seed):
    if not enable_shuffle or shuffle_seed < 0 or len(samples) <= 1:
        return samples, None
    rng = np.random.default_rng(int(shuffle_seed))
    order = rng.permutation(len(samples)).tolist()
    shuffled = [samples[idx] for idx in order]
    return shuffled, int(shuffle_seed)


def _build_shard(
    cache_dir,
    split,
    shard_id,
    shard_samples,
    feature_dtype,
    array_keys,
    include_image_cond=False,
    image_cond_key=DEFAULT_IMAGE_COND_KEY,
    image_cond_dtype=np.float16,
    image_cond_max_tokens_per_view=32,
    image_cond_specs=None,
    skip_keys=None,
    reuse_arrays_meta=None,
    reuse_cache_dir=None,
    reuse_materialize_mode="link",
):
    image_cond_specs = image_cond_specs or parse_image_cond_specs((), DEFAULT_IMAGE_COND_FILENAME, image_cond_key)
    skip_keys = set(skip_keys or [])
    reuse_arrays_meta = reuse_arrays_meta or {}
    shard_dir = cache_dir / f'shard_{shard_id:05d}'
    shard_dir.mkdir(parents=True, exist_ok=True)

    first = prepare_sample_arrays(
        shard_samples[0],
        feature_dtype,
        array_keys=array_keys,
        include_image_cond=include_image_cond,
        image_cond_key=image_cond_key,
        image_cond_dtype=image_cond_dtype,
        image_cond_max_tokens_per_view=image_cond_max_tokens_per_view,
        image_cond_specs=image_cond_specs,
        skip_keys=skip_keys,
    )
    shapes = {key: (len(shard_samples),) + tuple(first[key].shape) for key in array_keys if key not in skip_keys}
    dtypes = {
        'sparse_xyz': np.float32,
        'sparse_feat': feature_dtype,
        'gt_feat': feature_dtype,
        'gt_xyz': np.float32,
        'sparse_patch_idx': np.int32,
        'gt_patch_idx': np.int32,
    }
    if include_image_cond:
        for spec in image_cond_specs:
            dtypes[spec["array_key"]] = image_cond_dtype
    arrays = {}
    meta = {}
    for key in array_keys:
        if key in skip_keys:
            if key not in reuse_arrays_meta:
                raise RuntimeError(f'Missing reuse metadata for key={key} in shard={shard_id}')
            if reuse_cache_dir is None:
                raise RuntimeError(f'Missing reuse_cache_dir for key={key} in shard={shard_id}')
            _materialize_reused_array(
                cache_dir,
                reuse_cache_dir,
                reuse_arrays_meta[key],
                mode=reuse_materialize_mode,
            )
            meta[key] = dict(reuse_arrays_meta[key])
            continue
        path = shard_dir / f'{key}.npy'
        arrays[key] = np.lib.format.open_memmap(path, mode='w+', dtype=dtypes[key], shape=shapes[key])
        meta[key] = {'path': str(path.relative_to(cache_dir)), 'shape': list(shapes[key]), 'dtype': np.dtype(dtypes[key]).name}

    for local_idx, sample in enumerate(shard_samples):
        packed = first if local_idx == 0 else prepare_sample_arrays(
            sample,
            feature_dtype,
            array_keys=array_keys,
            include_image_cond=include_image_cond,
            image_cond_key=image_cond_key,
            image_cond_dtype=image_cond_dtype,
            image_cond_max_tokens_per_view=image_cond_max_tokens_per_view,
            image_cond_specs=image_cond_specs,
            skip_keys=skip_keys,
        )
        for key in array_keys:
            if key in skip_keys:
                continue
            arrays[key][local_idx] = packed[key]
        if (local_idx + 1) % 100 == 0 or local_idx == len(shard_samples) - 1:
            print(f'[{split}][shard {shard_id:05d}] packed {local_idx + 1}/{len(shard_samples)}')

    for key in array_keys:
        if key in skip_keys:
            continue
        arrays[key].flush()
    return meta, shapes, dtypes


def _resolve_memmap_root(path: str) -> Path:
    root = Path(path)
    if (root / DEFAULT_MEMMAP_CACHE_DIRNAME).is_dir():
        return root / DEFAULT_MEMMAP_CACHE_DIRNAME
    if (root / 'train' / 'manifest.json').is_file() or (root / 'test' / 'manifest.json').is_file():
        return root
    return root


def _load_reuse_manifest(reuse_root: str, split: str):
    if not reuse_root:
        return None
    memmap_root = _resolve_memmap_root(reuse_root)
    manifest_path = memmap_root / split / 'manifest.json'
    if not manifest_path.exists():
        raise RuntimeError(f'Reuse manifest not found: {manifest_path}')
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _materialize_reused_array(cache_dir: Path, reuse_cache_dir: Path, array_meta: dict, mode="link"):
    rel_path = str(array_meta.get('path', '')).strip()
    if not rel_path:
        raise RuntimeError(f"Reuse array metadata missing path: {array_meta}")

    dest_path = cache_dir / rel_path
    if dest_path.exists() or dest_path.is_symlink():
        return

    source_path = reuse_cache_dir / rel_path
    if not source_path.is_file():
        raise RuntimeError(f"Reuse array file not found: {source_path}")

    mode = str(mode or "link").strip().lower()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "move":
        shutil.move(str(source_path), str(dest_path))
        return
    if mode == "copy":
        shutil.copy2(source_path, dest_path)
        return

    try:
        os.link(source_path, dest_path)
        return
    except OSError:
        pass

    try:
        os.symlink(source_path, dest_path)
        return
    except OSError:
        pass

    shutil.copy2(source_path, dest_path)


def build_split_cache(
    dataset_root,
    split,
    feature_dtype_name,
    overwrite,
    shard_size,
    shuffle_seed,
    include_image_cond,
    image_cond_filename,
    image_cond_key,
    image_cond_dtype_name,
    image_cond_max_tokens_per_view,
    image_cond_specs=None,
    reuse_cache_from="",
    reuse_array_keys=(),
    reuse_materialize_mode="link",
    worker_shards=1,
    worker_id=0,
    finalize_only=False,
    index_mode='part',
):
    image_cond_specs = image_cond_specs or parse_image_cond_specs((), image_cond_filename, image_cond_key)
    split_config = resolve_split_array_config(split, include_image_cond, image_cond_specs=image_cond_specs, index_mode=index_mode)
    array_keys = split_config['array_keys']
    pack_image_cond = split_config['pack_image_cond']

    samples = load_index_or_scan(
        dataset_root,
        split,
        include_image_cond=pack_image_cond,
        image_cond_filename=image_cond_filename,
        image_cond_specs=image_cond_specs,
    )
    if not samples:
        raise RuntimeError(f'No samples found for split={split}')

    reuse_manifest = _load_reuse_manifest(reuse_cache_from, split) if reuse_cache_from else None
    reuse_cache_dir = (_resolve_memmap_root(reuse_cache_from) / split) if reuse_cache_from else None
    requested_reuse_keys = {key for key in reuse_array_keys if key}
    reuse_keys = set()
    if reuse_manifest:
        for key in reuse_manifest.get('arrays', {}):
            if key in requested_reuse_keys and key not in array_keys:
                array_keys.append(key)
        available_reuse_keys = set(reuse_manifest.get('arrays', {}).keys())
        unknown_keys = sorted(requested_reuse_keys - available_reuse_keys)
        if unknown_keys:
            raise RuntimeError(
                f"Reuse manifest missing requested array keys for split={split}: {unknown_keys}"
            )
        reuse_keys = {
            key for key in requested_reuse_keys
            if key in array_keys and key in available_reuse_keys
        }

    feature_dtype = np.float16 if feature_dtype_name == 'float16' else np.float32
    image_cond_dtype = np.float16 if image_cond_dtype_name == 'float16' else np.float32

    cache_dir = Path(get_split_cache_dir(dataset_root, split))
    worker_shards = max(int(worker_shards), 1)
    worker_id = int(worker_id)
    if worker_id < 0 or worker_id >= worker_shards:
        raise ValueError(f"worker_id must be in [0, {worker_shards}), got {worker_id}")
    if overwrite and not finalize_only and worker_id == 0:
        _prepare_cache_dir(cache_dir, overwrite)
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / 'manifest.json'

    requested_shard_size = int(shard_size)
    if reuse_manifest:
        reuse_samples = reuse_manifest.get('samples', [])
        if len(reuse_samples) != len(samples):
            raise RuntimeError(
                f'Reuse manifest sample count mismatch: reuse={len(reuse_samples)} current={len(samples)}'
            )
        lookup = {(str(s['category']), str(s['sample_id'])): s for s in samples}
        ordered = []
        missing = []
        for item in reuse_samples:
            key = (str(item['category']), str(item['sample_id']))
            sample = lookup.get(key)
            if sample is None:
                missing.append(key)
                continue
            ordered.append(sample)
        if missing:
            preview = ", ".join([f"{c}:{sid}" for c, sid in missing[:5]])
            raise RuntimeError(f'Reuse manifest samples missing from dataset (first 5): {preview}')
        samples = ordered

        reuse_shard_size = int(reuse_manifest.get('shard_size', 0))
        if reuse_shard_size <= 0:
            raise RuntimeError(f'Reuse manifest has invalid shard_size={reuse_shard_size}')
        if requested_shard_size > 0 and requested_shard_size != reuse_shard_size:
            raise RuntimeError(
                f'shard_size mismatch: requested={requested_shard_size} reuse_manifest={reuse_shard_size}'
            )
        effective_shard_size = reuse_shard_size

        reuse_shuffle_seed = reuse_manifest.get('shuffle_seed', None)
        if shuffle_seed >= 0 and reuse_shuffle_seed is not None and int(shuffle_seed) != int(reuse_shuffle_seed):
            raise RuntimeError(
                f'shuffle_seed mismatch: requested={shuffle_seed} reuse_manifest={reuse_shuffle_seed}'
            )
        effective_shuffle_seed = reuse_shuffle_seed
    else:
        effective_shard_size = len(samples) if requested_shard_size <= 0 else max(1, requested_shard_size)
        samples, effective_shuffle_seed = _maybe_shuffle_samples(samples, requested_shard_size > 0, shuffle_seed)

    reuse_shard_meta = {}
    reuse_array_meta = {}
    if reuse_manifest:
        if not reuse_manifest.get('arrays'):
            raise RuntimeError('Reuse manifest missing arrays metadata.')
        for key in reuse_keys:
            if key not in reuse_manifest['arrays']:
                raise RuntimeError(f'Reuse manifest missing array key: {key}')
        reuse_array_meta = {key: reuse_manifest['arrays'][key] for key in reuse_keys}
        if reuse_manifest.get('num_samples') is not None and int(reuse_manifest['num_samples']) != len(samples):
            raise RuntimeError(
                f'Reuse manifest num_samples mismatch: reuse={reuse_manifest["num_samples"]} current={len(samples)}'
            )
        if reuse_manifest.get('shards'):
            shard_key_sets = []
            for shard in reuse_manifest['shards']:
                shard_id = int(shard.get('id', len(reuse_shard_meta)))
                arrays = shard.get('arrays', {})
                shard_key_sets.append(set(arrays.keys()))
                reuse_shard_meta[shard_id] = {key: arrays[key] for key in reuse_keys if key in arrays}
            if shard_key_sets:
                shard_available_reuse_keys = set.intersection(*shard_key_sets)
                dropped_keys = sorted(reuse_keys - shard_available_reuse_keys)
                if dropped_keys:
                    print(
                        f"[{split}] reuse keys missing from shard arrays and skipped: {dropped_keys}",
                        flush=True,
                    )
                    array_keys = [key for key in array_keys if key not in dropped_keys]
                    for key in dropped_keys:
                        reuse_array_meta.pop(key, None)
                reuse_keys = {key for key in reuse_keys if key in shard_available_reuse_keys}
        else:
            reuse_shard_meta[0] = {key: reuse_manifest['arrays'][key] for key in reuse_keys}

        expected_shards = int(reuse_manifest.get('num_shards', 0))
        actual_shards = (len(samples) + effective_shard_size - 1) // effective_shard_size
        if expected_shards and expected_shards != actual_shards:
            raise RuntimeError(
                f'num_shards mismatch: reuse={expected_shards} current={actual_shards} (shard_size={effective_shard_size})'
            )

    shard_jobs = []
    start_index = 0
    for shard_id, shard_start in enumerate(range(0, len(samples), effective_shard_size)):
        shard_samples = samples[shard_start:shard_start + effective_shard_size]
        shard_jobs.append({
            'id': int(shard_id),
            'start_index': int(start_index),
            'num_samples': int(len(shard_samples)),
            'samples': shard_samples,
        })
        start_index += len(shard_samples)

    if not finalize_only:
        for shard_job in shard_jobs:
            shard_id = int(shard_job['id'])
            if shard_id % worker_shards != worker_id:
                continue
            shard_meta, shard_shapes, shard_dtypes = _build_shard(
                cache_dir,
                split,
                shard_id,
                shard_job['samples'],
                feature_dtype,
                array_keys,
                include_image_cond=pack_image_cond,
                image_cond_key=image_cond_key,
                image_cond_dtype=image_cond_dtype,
                image_cond_max_tokens_per_view=image_cond_max_tokens_per_view,
                image_cond_specs=image_cond_specs,
                skip_keys=reuse_keys,
                reuse_arrays_meta=reuse_shard_meta.get(shard_id, {}),
                reuse_cache_dir=reuse_cache_dir,
                reuse_materialize_mode=reuse_materialize_mode,
            )
            aggregate_meta = dict(reuse_array_meta)
            for key in array_keys:
                if key in aggregate_meta:
                    continue
                aggregate_meta[key] = {
                    'shape': [len(samples)] + list(shard_shapes[key][1:]),
                    'dtype': np.dtype(shard_dtypes[key]).name,
                }
            _write_partial_shard_meta(
                cache_dir,
                shard_id,
                {
                    'id': int(shard_id),
                    'start_index': int(shard_job['start_index']),
                    'num_samples': int(shard_job['num_samples']),
                    'arrays': shard_meta,
                    'aggregate_arrays': aggregate_meta,
                },
            )
        print(f'[{split}] worker {worker_id}/{worker_shards} finished assigned shards.', flush=True)
        # Finalize single-worker caches immediately.
        # Multi-worker caches are finalized after all shards finish.
        if worker_shards > 1:
            return

    partial_meta = _load_partial_shard_meta(cache_dir)
    missing_shards = [job['id'] for job in shard_jobs if int(job['id']) not in partial_meta]
    if missing_shards:
        preview = ",".join(str(v) for v in missing_shards[:10])
        raise RuntimeError(f'Cannot finalize split={split}; missing partial shard metadata: {preview}')

    shards = []
    aggregate_meta = dict(reuse_array_meta)
    for shard_job in shard_jobs:
        shard_id = int(shard_job['id'])
        shard_payload = partial_meta[shard_id]
        shard_meta = dict(shard_payload['arrays'])
        shards.append({
            'id': int(shard_id),
            'start_index': int(shard_job['start_index']),
            'num_samples': int(shard_job['num_samples']),
            'arrays': shard_meta,
        })
        if not aggregate_meta:
            aggregate_meta = {}
        for key, value in dict(shard_payload.get('aggregate_arrays', {})).items():
            aggregate_meta.setdefault(key, value)

    legacy_image_cond = next((spec for spec in image_cond_specs if spec["array_key"] == "image_cond"), None)
    if legacy_image_cond is None:
        legacy_image_cond = image_cond_specs[0] if image_cond_specs else None

    manifest = {
        'version': 2,
        'split': split,
        'num_samples': len(samples),
        'point_index_mode': index_mode,
        'feature_dtype': np.dtype(feature_dtype).name,
        'arrays': aggregate_meta,
        'shard_size': int(effective_shard_size),
        'num_shards': int(len(shards)),
        'shuffle_seed': effective_shuffle_seed,
        'shards': shards,
        'image_cond': {
            'enabled': bool(pack_image_cond),
            'array_key': legacy_image_cond["array_key"] if legacy_image_cond else None,
            'filename': legacy_image_cond["filename"] if legacy_image_cond else image_cond_filename,
            'key': legacy_image_cond["key"] if legacy_image_cond else image_cond_key,
            'dtype': np.dtype(image_cond_dtype).name if pack_image_cond else None,
            'max_tokens_per_view': int(max(image_cond_max_tokens_per_view, 0)) if pack_image_cond else 0,
        },
        'image_conds': {
            spec["array_key"]: {
                'enabled': bool(pack_image_cond),
                'filename': spec["filename"],
                'key': spec["key"],
                'dtype': np.dtype(image_cond_dtype).name if pack_image_cond else None,
                'max_tokens_per_view': int(max(image_cond_max_tokens_per_view, 0)) if pack_image_cond else 0,
            }
            for spec in image_cond_specs
        },
        'samples': [{'sample_id': s['sample_id'], 'category': s['category']} for s in samples],
    }
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f)
    print(f'Saved manifest to {manifest_path}')


def main():
    args = parse_args()
    reuse_cache_from = args.reuse_cache_from
    reuse_array_keys = parse_reuse_array_keys(args.reuse_array_keys)
    image_cond_specs = parse_image_cond_specs(
        args.image_cond_spec,
        default_filename=args.image_cond_file,
        default_key=args.image_cond_key,
    )
    splits = ['train', 'test'] if args.split == 'all' else [args.split]
    if args.mode == 'image-only':
        for split in splits:
            refresh_split_image_only(
                dataset_root=args.dataset_root,
                split=split,
                image_cond_filename=args.image_cond_file,
                image_cond_key=args.image_cond_key,
                image_cond_dtype_name=args.image_cond_dtype,
                image_cond_max_tokens_per_view=args.image_cond_max_tokens_per_view,
                image_cond_specs=image_cond_specs,
            )
        return

    for split in splits:
        build_split_cache(
            args.dataset_root,
            split,
            args.feature_dtype,
            args.overwrite,
            args.shard_size,
            args.shuffle_seed,
            args.include_image_cond,
            args.image_cond_file,
            args.image_cond_key,
            args.image_cond_dtype,
            args.image_cond_max_tokens_per_view,
            image_cond_specs=image_cond_specs,
            reuse_cache_from=reuse_cache_from,
            reuse_array_keys=reuse_array_keys,
            reuse_materialize_mode=args.reuse_materialize_mode,
            worker_shards=args.worker_shards,
            worker_id=args.worker_id,
            finalize_only=args.finalize_only,
            index_mode=args.index_mode,
        )


if __name__ == '__main__':
    main()
