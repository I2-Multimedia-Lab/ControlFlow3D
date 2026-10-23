import os
import time
import json
import torch
import numpy as np
from torch.utils.data import Dataset
try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False

from datasetsource.memmap_sample_cache import SplitMemmapBackend
from utils.dataset_feature_layout import (
    DEFAULT_IMAGE_COND_FILENAME, DEFAULT_IMAGE_COND_KEY, DEFAULT_IMAGE_COND_ARRAY_KEY,
)
from datasetsource.ply_io import (
    read_generic_ply_xyz,
    read_part_ply_indices,
    read_patch_ply_xyz,
)


def _should_log_dataset_info():
    return os.environ.get('CONTROLFLOW3D_VERBOSE_DATASET', '0') == '1' and int(os.environ.get('RANK', '0')) == 0


def _dataset_info(msg):
    if _should_log_dataset_info():
        print(msg)


class MultimodalPointCloudDataset(Dataset):
    """Load aligned point clouds and features from source files or memmap arrays.

    Samples are stored under <root>/<split>/<category>/<sample>/input and gt.
    Sparse and dense point features follow their coordinate order. Image
    conditions and point-group indices are included when requested.
    """
    SPARSE_FEAT_FILENAME = 'sparse_local_feat.npy'
    _POINTCLOUD_BASENAMES = {
        'sparse_path': ('sparse_xyz', 'sparse_point_cloud'),
        'gt_path': ('gt_xyz', 'upsampled_point_cloud'),
    }

    def __init__(
        self,
        root_dir,
        split='train',
        num_points=2048,
        transform=None,
        return_category=False,
        return_sample_id=True,
        normalize_by_input=True,
        return_normalization_meta=True,
        load_gt_feat=False,
        load_image_cond=False,
        image_cond_filename=DEFAULT_IMAGE_COND_FILENAME,
        image_cond_key=DEFAULT_IMAGE_COND_KEY,
        image_cond_array_key=DEFAULT_IMAGE_COND_ARRAY_KEY,
        image_cond_dim=512,
        image_cond_max_tokens_per_view=32,
        point_index_mode='part',
        num_parts=8,
        skip_bad_samples=None,
    ):
        self.root_dir = root_dir
        self.split = split
        # Training may retry; validation and testing must preserve sample identity.
        self.skip_bad_samples = (split == 'train') if skip_bad_samples is None else bool(skip_bad_samples)
        self.num_points = num_points
        self.transform = transform
        self.return_category = return_category
        self.return_sample_id = return_sample_id
        self.normalize_by_input = bool(normalize_by_input)
        self.return_normalization_meta = bool(return_normalization_meta)
        self.load_gt_feat = bool(load_gt_feat)
        self.load_image_cond = bool(load_image_cond)
        self.image_cond_filename = str(image_cond_filename)
        self.image_cond_key = str(image_cond_key)
        self.image_cond_array_key = self._resolve_image_cond_array_key(image_cond_array_key)
        self.image_cond_dim = int(image_cond_dim)
        self.image_cond_max_tokens_per_view = max(int(image_cond_max_tokens_per_view), 0)
        self.point_index_mode = str(point_index_mode or 'part').strip().lower()
        if self.point_index_mode not in ('part', 'patch'):
            raise ValueError(f"point_index_mode must be 'part' or 'patch', got {point_index_mode!r}")
        self.num_parts = max(int(num_parts), 1)
        self.text_prompt_meta = {}
        self._reported_bad_samples = set()
        self._reported_patch_idx_fixes = set()
        self.cache_path = os.path.join(self.root_dir, f'.sample_index_external_cond_{self.split}.json')
        self.memmap_backend = self._try_init_memmap_backend()
        self._memmap_owner_pid = os.getpid() if self.memmap_backend is not None else None
        
        # Collect sample paths.
        self.samples = self._collect_samples()
        if len(self.samples) == 0:
            missing = getattr(self, '_last_scan_missing', {}) or {}
            detail = ''
            if missing:
                worst = sorted(missing.items(), key=lambda kv: -kv[1])[:4]
                detail = (' No sample had a complete set of required files. Missing counts: '
                          + ', '.join(f'{name} ({n}x)' for name, n in worst) + '.')
            raise RuntimeError(
                f'No {split} data samples found under {root_dir}.{detail} '
                f'Expected <split>/<category_id>/<sample_id>/ with input/ and gt/; '
                f'see data/README.md for the full layout and how to generate the derived files.'
            )

        self.sample_payload_bytes = self._estimate_sample_payload_bytes()
        self._log_storage_summary()
            
        _dataset_info(f"[{split}] Loaded {len(self.samples)} samples from {root_dir}")

    def _loaded_array_keys(self):
        keys = ['sparse_xyz', 'sparse_feat', 'gt_xyz']
        if self.load_gt_feat:
            keys.append('gt_feat')
        if self.memmap_backend is not None:
            if self.memmap_backend.has_array_key('sparse_patch_idx'):
                keys.append('sparse_patch_idx')
            if self.memmap_backend.has_array_key('gt_patch_idx'):
                keys.append('gt_patch_idx')
        if (
            self.load_image_cond
            and self.memmap_backend is not None
            and self.memmap_backend.image_cond_matches(
                key=self.image_cond_key,
                filename=self.image_cond_filename,
                array_key=self.image_cond_array_key,
            )
        ):
            keys.append(self.image_cond_array_key)
        return tuple(keys)

    def _resolve_image_cond_array_key(self, image_cond_array_key):
        if image_cond_array_key is not None and str(image_cond_array_key).strip():
            return str(image_cond_array_key).strip()
        filename = str(self.image_cond_filename).lower()
        if 'vggt' in filename:
            return DEFAULT_IMAGE_COND_ARRAY_KEY
        return 'image_cond'

    def _try_init_memmap_backend(self):
        if os.environ.get('CONTROLFLOW3D_DISABLE_MEMMAP_CACHE', '0') == '1':
            return None
        if not SplitMemmapBackend.exists(self.root_dir, self.split):
            return None
        backend = SplitMemmapBackend(self.root_dir, self.split)
        _dataset_info(f"[{self.split}] Using memmap cache: {backend.manifest_path} ({backend.num_samples} samples)")
        return backend

    def _ensure_process_local_memmap_backend(self):
        if self.memmap_backend is None:
            return None
        current_pid = os.getpid()
        if self._memmap_owner_pid != current_pid:
            self.memmap_backend = self.memmap_backend.clone_for_current_process()
            self._memmap_owner_pid = current_pid
        return self.memmap_backend

    def _collect_samples(self):
        """Collect sample records from a memmap manifest, cached index or directory scan."""
        if self.memmap_backend is not None:
            return list(self.memmap_backend.samples)

        split_dir = os.path.join(self.root_dir, self.split)
        if not os.path.isdir(split_dir):
            raise RuntimeError(f"split directory does not exist: {split_dir}")

        # Rescan source files during strict evaluation to include incomplete samples.
        if self.skip_bad_samples and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    cache_obj = json.load(f)
                cached_samples = cache_obj.get('samples', [])
                if len(cached_samples) > 0:
                    cached_samples, updated_paths = self._refresh_cached_pointcloud_paths(cached_samples)
                    first = cached_samples[0]
                    required_keys = ('sparse_path', 'x0_feat_path', 'gt_path')
                    if self.load_gt_feat:
                        required_keys = required_keys + ('gt_feat_path',)
                    if self.load_image_cond:
                        required_keys = required_keys + ('image_cond_path',)
                    cache_uses_expected_sparse_feat = (
                        os.path.basename(first.get('x0_feat_path', '')) == self.SPARSE_FEAT_FILENAME
                    )
                    if (
                        all(k in first for k in required_keys)
                        and all(os.path.exists(first.get(k, '')) for k in required_keys)
                        and cache_uses_expected_sparse_feat
                    ):
                        if updated_paths > 0:
                            self._save_cache_samples(cached_samples, note=f'updated {updated_paths} cached point-cloud paths')
                        _dataset_info(f"[{self.split}] Using sample index cache: {self.cache_path} ({len(cached_samples)} samples)")
                        return cached_samples
            except Exception as e:
                _dataset_info(f"[{self.split}] Sample cache load failed, fallback to rescan: {e}")

        # Scan sample directories and validate required files.
        t0 = time.time()
        samples = []
        missing_counts = {}
        self._last_scan_missing = missing_counts
        cat_entries = [e for e in os.scandir(split_dir) if e.is_dir()]
        cat_entries = sorted(cat_entries, key=lambda x: x.name)
        _dataset_info(f"[{self.split}] Scanning dataset index under {split_dir} ...")
        _dataset_info(f"[{self.split}] Categories found: {len(cat_entries)}")

        for ci, cat_entry in enumerate(cat_entries):
            if ci % 20 == 0 or ci == len(cat_entries) - 1:
                _dataset_info(f"[{self.split}] Index progress: {ci + 1}/{len(cat_entries)} categories, samples={len(samples)}")

            for sample_entry in os.scandir(cat_entry.path):
                if not sample_entry.is_dir():
                    continue

                sample_dir = sample_entry.path
                input_dir = os.path.join(sample_dir, 'input')
                gt_dir = os.path.join(sample_dir, 'gt')

                sparse_path = self._resolve_pointcloud_path(input_dir, 'sparse_path')
                x0_feat_path = os.path.join(input_dir, self.SPARSE_FEAT_FILENAME)
                gt_path = self._resolve_pointcloud_path(gt_dir, 'gt_path')
                gt_feat_path = os.path.join(gt_dir, 'gt_feat.npy')
                image_cond_path = os.path.join(input_dir, self.image_cond_filename)

                required = [
                    (sparse_path, 'input/sparse_xyz.(ply|npy)'),
                    (x0_feat_path, self.SPARSE_FEAT_FILENAME),
                    (gt_path, 'gt/gt_xyz.(ply|npy)'),
                ]
                if self.load_gt_feat:
                    required.append((gt_feat_path, 'gt/gt_feat.npy'))
                if self.load_image_cond:
                    required.append((image_cond_path, self.image_cond_filename))
                missing = [label for path, label in required if not os.path.exists(path)]
                if missing:
                    if not self.skip_bad_samples:
                        raise FileNotFoundError(
                            f"Incomplete {self.split} sample {cat_entry.name}/{sample_entry.name}: "
                            f"missing {', '.join(missing)}. Evaluation does not skip samples."
                        )
                    for label in missing:
                        missing_counts[label] = missing_counts.get(label, 0) + 1
                else:
                    samples.append({
                        'sample_id': sample_entry.name,
                        'category': cat_entry.name,
                        'sparse_path': sparse_path,
                        'x0_feat_path': x0_feat_path,
                        'gt_path': gt_path,
                        'gt_feat_path': gt_feat_path,
                        'image_cond_path': image_cond_path,
                    })

        dt = time.time() - t0
        _dataset_info(f"[{self.split}] Index scan done in {dt:.2f}s, valid samples={len(samples)}")

        # Save the sample index when the output directory is writable.
        self._save_cache_samples(samples)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if not self.skip_bad_samples:
            sample = self.samples[idx]
            try:
                return self._load_one_sample(sample)
            except Exception as exc:
                sid = sample.get('sample_id', f'idx_{idx}')
                category = sample.get('category', '<unknown>')
                raise RuntimeError(
                    f"Failed to load {self.split} sample {category}/{sid} (index={idx}): {exc}. "
                    "Evaluation does not substitute another sample."
                ) from exc

        total = len(self.samples)
        max_retry = min(64, total)
        for off in range(max_retry):
            sample = self.samples[(idx + off) % total]
            try:
                return self._load_one_sample(sample)
            except Exception as e:
                sid = sample.get('sample_id', f'idx_{(idx + off) % total}')
                if sid not in self._reported_bad_samples:
                    self._reported_bad_samples.add(sid)
                    print(f"[Dataset Warning] Skip bad sample {sid}: {e}")
                continue
        raise RuntimeError(f"Failed to load {max_retry} consecutive samples; please check dataset integrity.")

    @staticmethod
    def _normalize_feature_dtype(array):
        arr = np.asarray(array)
        if arr.dtype in (np.float16, np.float32):
            return arr
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _to_writable_tensor(array, dtype=None):
        arr = np.asarray(array, dtype=dtype)
        return torch.from_numpy(np.array(arr, copy=True))

    def _resolve_image_cond_path(self, sample):
        path = sample.get('image_cond_path', '')
        if path:
            return path
        return os.path.join(
            self.root_dir,
            self.split,
            str(sample['category']),
            str(sample['sample_id']),
            'input',
            self.image_cond_filename,
        )

    def _load_image_cond(self, sample):
        image_cond_path = self._resolve_image_cond_path(sample)
        if not os.path.isfile(image_cond_path):
            raise FileNotFoundError(
                f"Missing image condition file: {image_cond_path} (sample_id={sample['sample_id']})"
            )

        with np.load(image_cond_path, allow_pickle=False) as package:
            if self.image_cond_key not in package:
                raise KeyError(
                    f"Image condition file is missing key='{self.image_cond_key}': {image_cond_path}"
                )
            image_cond = np.asarray(package[self.image_cond_key])

        return self._normalize_image_cond_array(image_cond, sample, image_cond_path)

    def _normalize_image_cond_array(self, image_cond, sample, source):
        image_cond = np.asarray(image_cond)

        # Supported image feature layouts:
        # [V, C]: per-view features.
        # [V, T, C]: per-view tokens.
        # [V, H, W, C]: patch grids flattened to [V, T, C].
        if image_cond.ndim == 2:
            normalized = image_cond
        elif image_cond.ndim == 3:
            normalized = image_cond
        elif image_cond.ndim == 4:
            normalized = image_cond.reshape(image_cond.shape[0], -1, image_cond.shape[-1])
        else:
            raise ValueError(
                f"image_cond only supports [V,C] / [V,T,C] / [V,H,W,C], but got {tuple(image_cond.shape)} "
                f"(sample_id={sample['sample_id']})"
            )

        if normalized.shape[-1] != self.image_cond_dim:
            raise ValueError(
                f"image_cond dimension mismatch: got {normalized.shape[-1]}, expected {self.image_cond_dim} "
                f"(sample_id={sample['sample_id']})"
            )
        if not np.isfinite(normalized).all():
            raise ValueError(
                f"image_cond contains NaN/Inf: {source} (sample_id={sample['sample_id']})"
            )
        return normalized.astype(np.float32, copy=False)

    @staticmethod
    def _normalize_by_sparse_input(sparse_xyz, *aligned_xyz):
        centroid = np.mean(sparse_xyz, axis=0, keepdims=True).astype(np.float32, copy=False)
        centered_sparse = sparse_xyz.astype(np.float32, copy=False) - centroid
        furthest_distance = np.max(np.linalg.norm(centered_sparse, axis=1, keepdims=True))
        furthest_distance = max(float(furthest_distance), 1e-8)

        normalized_sparse = (centered_sparse / furthest_distance).astype(np.float32, copy=False)
        normalized_aligned = [
            ((points.astype(np.float32, copy=False) - centroid) / furthest_distance).astype(np.float32, copy=False)
            for points in aligned_xyz
        ]
        furthest_meta = np.array([[furthest_distance]], dtype=np.float32)
        return normalized_sparse, normalized_aligned, centroid, furthest_meta

    def _load_one_sample(self, sample):
        t0 = time.time()
        packed_sample = None
        backend = self._ensure_process_local_memmap_backend()
        memmap_mode = backend is not None and 'packed_index' in sample
        if memmap_mode:
            requested_keys = ['sparse_xyz', 'sparse_feat', 'gt_xyz']
            has_memmap_gt_feat = bool(
                self.load_gt_feat
                and backend.has_array_key('gt_feat')
            )
            if has_memmap_gt_feat:
                requested_keys.append('gt_feat')
            if backend.has_array_key('sparse_patch_idx'):
                requested_keys.append('sparse_patch_idx')
            if backend.has_array_key('gt_patch_idx'):
                requested_keys.append('gt_patch_idx')
            if (
                self.load_image_cond
                and backend.image_cond_matches(
                    key=self.image_cond_key,
                    filename=self.image_cond_filename,
                    array_key=self.image_cond_array_key,
                )
            ):
                requested_keys.append(self.image_cond_array_key)
            packed_sample = backend.load_sample(sample['packed_index'], keys=requested_keys)

        if self.load_image_cond:
            use_memmap_image_cond = (
                packed_sample is not None
                and self.image_cond_array_key in packed_sample
            )
            if use_memmap_image_cond:
                image_cond = self._normalize_image_cond_array(
                    packed_sample[self.image_cond_array_key],
                    sample,
                    f'<memmap:{self.image_cond_array_key}>',
                )
            else:
                # Read the source NPZ when cached image metadata does not match the request.
                image_cond = self._load_image_cond(sample)
        else:
            image_cond = None

        if packed_sample is not None:
            sparse_xyz = np.asarray(packed_sample['sparse_xyz'], dtype=np.float32)
            sparse_feat = self._normalize_feature_dtype(packed_sample['sparse_feat'])
            gt_xyz = np.asarray(packed_sample['gt_xyz'], dtype=np.float32)
            sparse_patch_idx = packed_sample.get('sparse_patch_idx')
            gt_patch_idx = packed_sample.get('gt_patch_idx')
            if self.point_index_mode == 'part':
                sparse_patch_idx = self._build_contiguous_part_indices(
                    sparse_xyz.shape[0],
                    self.num_parts,
                    f"{sample['sample_id']}:sparse",
                )
                gt_patch_idx = self._build_contiguous_part_indices(
                    gt_xyz.shape[0],
                    self.num_parts,
                    f"{sample['sample_id']}:gt",
                )
            gt_path = '<memmap>'
            if self.load_gt_feat:
                if 'gt_feat' in packed_sample:
                    gt_feat = self._normalize_feature_dtype(packed_sample['gt_feat'])
                    gt_feat_path = '<memmap>'
                elif self.split == 'train':
                    raise RuntimeError(
                        'train memmap is missing gt_feat; '
                        'please generate gt/gt_feat.npy first, then use the new gt_feat memmap packing script to fill in the cache.'
                    )
                else:
                    gt_feat = np.empty((0, sparse_feat.shape[1]), dtype=sparse_feat.dtype)
                    gt_feat_path = '<memmap:missing>'
            else:
                gt_feat = np.empty((0, sparse_feat.shape[1]), dtype=sparse_feat.dtype)
                gt_feat_path = '<skipped>'
        else:
            # Load sparse coordinates and grouping indices.
            sparse_path = sample['sparse_path']
            if sparse_path.endswith('.ply'):
                sparse_xyz, sparse_patch_idx = read_patch_ply_xyz(sparse_path)
                if self.point_index_mode == 'part':
                    sparse_patch_idx = read_part_ply_indices(sparse_path)
                    if sparse_patch_idx is None:
                        sparse_patch_idx = self._build_contiguous_part_indices(
                            sparse_xyz.shape[0],
                            self.num_parts,
                            sparse_path,
                        )
            else:
                sparse_xyz = self._load_point_cloud(sparse_path)
                sparse_patch_idx = (
                    self._build_contiguous_part_indices(sparse_xyz.shape[0], self.num_parts, sparse_path)
                    if self.point_index_mode == 'part'
                    else None
                )

            # Load point-aligned features: [N_sparse, C_flow].
            sparse_feat = self._normalize_feature_dtype(np.load(sample['x0_feat_path'], allow_pickle=False))

            if self.load_gt_feat:
                gt_feat_path = sample['gt_feat_path']
                if not os.path.exists(gt_feat_path):
                    raise FileNotFoundError(
                        f"Missing required gt_feat file: {gt_feat_path} (sample_id={sample['sample_id']})"
                    )
                gt_feat = self._normalize_feature_dtype(np.load(gt_feat_path, allow_pickle=False))
            else:
                gt_feat = np.empty((0, sparse_feat.shape[1]), dtype=sparse_feat.dtype)
                gt_feat_path = '<skipped>'

            # Load dense target coordinates and grouping indices.
            gt_path = sample['gt_path']
            if gt_path.endswith('.ply'):
                gt_xyz, gt_patch_idx = read_patch_ply_xyz(gt_path)
                if self.point_index_mode == 'part':
                    gt_patch_idx = read_part_ply_indices(gt_path)
                    if gt_patch_idx is None:
                        gt_patch_idx = self._build_contiguous_part_indices(
                            gt_xyz.shape[0],
                            self.num_parts,
                            gt_path,
                        )
            else:
                gt_xyz = self._load_point_cloud(gt_path)
                gt_patch_idx = (
                    self._build_contiguous_part_indices(gt_xyz.shape[0], self.num_parts, gt_path)
                    if self.point_index_mode == 'part'
                    else None
                )

        sparse_xyz = self._ensure_exact_points(sparse_xyz, self.num_points)
        if sparse_patch_idx is not None:
            sparse_patch_idx = self._normalize_patch_index_array(
                sparse_patch_idx,
                point_count=sparse_xyz.shape[0],
                sample_id=sample['sample_id'],
                field_name='sparse_patch_idx',
                points_xyz=sparse_xyz,
            )
            max_idx = int(sparse_patch_idx.max()) if sparse_patch_idx.size else -1
            if max_idx >= sparse_xyz.shape[0]:
                raise ValueError(
                    f"sparse_patch_idx index out of range: max={max_idx}, "
                    f"n={sparse_xyz.shape[0]} (sample_id={sample['sample_id']})"
                )
        if sparse_feat.ndim != 2:
            raise ValueError(
                f"sparse_feat shape must be 2D [N_sparse, C], but got {tuple(sparse_feat.shape)} "
                f"(sample_id={sample['sample_id']})"
            )
        if sparse_feat.shape[0] != sparse_xyz.shape[0]:
            raise ValueError(
                f"sparse_feat point count must strictly align with sparse_xyz: feat={sparse_feat.shape[0]}, "
                f"xyz={sparse_xyz.shape[0]} (sample_id={sample['sample_id']})"
            )
        if not np.isfinite(sparse_feat).all():
            raise ValueError(
                f"sparse_feat contains NaN/Inf: {sample.get('x0_feat_path', '<memmap>')} (sample_id={sample['sample_id']})"
            )

        if self.load_gt_feat and gt_feat.shape[0] != gt_xyz.shape[0]:
            min_n = min(gt_feat.shape[0], gt_xyz.shape[0])
            gt_feat = gt_feat[:min_n]
            gt_xyz = gt_xyz[:min_n]

        if self.load_gt_feat and gt_feat.ndim != 2:
            raise ValueError(
                f"gt_feat shape must be 2D [N_gt, C], but got {tuple(gt_feat.shape)} "
                f"(sample_id={sample['sample_id']})"
            )
        if self.load_gt_feat and not np.isfinite(gt_feat).all():
            raise ValueError(
                f"gt_feat contains NaN/Inf: {gt_feat_path} (sample_id={sample['sample_id']})"
            )

        if gt_patch_idx is not None:
            gt_patch_idx = self._normalize_patch_index_array(
                gt_patch_idx,
                point_count=gt_xyz.shape[0],
                sample_id=sample['sample_id'],
                field_name='gt_patch_idx',
                points_xyz=gt_xyz,
            )
            max_idx = int(gt_patch_idx.max()) if gt_patch_idx.size else -1
            if max_idx >= gt_xyz.shape[0]:
                raise ValueError(
                    f"gt_patch_idx index out of range: max={max_idx}, "
                    f"n={gt_xyz.shape[0]} (sample_id={sample['sample_id']})"
                )

        if self.normalize_by_input:
            sparse_xyz, normalized_xyz, centroid, furthest_distance = self._normalize_by_sparse_input(
                sparse_xyz,
                gt_xyz,
            )
            gt_xyz = normalized_xyz[0]
        else:
            centroid = np.zeros((1, 3), dtype=np.float32)
            furthest_distance = np.ones((1, 1), dtype=np.float32)

        # Copy read-only memmap slices into writable arrays before tensor conversion.
        sparse_xyz = self._to_writable_tensor(sparse_xyz, dtype=np.float32)
        sparse_feat = self._to_writable_tensor(sparse_feat)
        gt_feat = self._to_writable_tensor(gt_feat)
        gt_xyz = self._to_writable_tensor(gt_xyz, dtype=np.float32)
        if sparse_patch_idx is not None:
            sparse_patch_idx = self._to_writable_tensor(sparse_patch_idx, dtype=np.int64)
        if gt_patch_idx is not None:
            gt_patch_idx = self._to_writable_tensor(gt_patch_idx, dtype=np.int64)
        centroid = self._to_writable_tensor(centroid, dtype=np.float32)
        furthest_distance = self._to_writable_tensor(furthest_distance, dtype=np.float32)
        if image_cond is not None:
            image_cond = self._to_writable_tensor(image_cond, dtype=np.float32)

        payload = [sparse_xyz, sparse_feat, gt_feat, gt_xyz]
        if sparse_patch_idx is not None:
            payload.append(sparse_patch_idx)
        if gt_patch_idx is not None:
            payload.append(gt_patch_idx)
        if self.return_normalization_meta:
            payload.extend([centroid, furthest_distance])
        if self.return_category:
            if self.return_sample_id:
                payload.extend([sample['sample_id'], sample['category']])
            else:
                payload.append(sample['category'])
        elif self.return_sample_id:
            payload.append(sample['sample_id'])

        if image_cond is not None:
            payload.append(image_cond)
        return tuple(payload)

    def _load_point_cloud(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == '.npy':
            # Map NPY coordinates without reading the entire file eagerly.
            points = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
        elif ext == '.ply':
            points = read_generic_ply_xyz(path)
        elif ext in ('.pcd', '.xyz'):
            if not HAS_O3D:
                raise ImportError('open3d is required to load PCD/XYZ files. For PLY-only datasets, run utils/convert_pointcloud_ply_to_npy.py first.')
            pc = o3d.io.read_point_cloud(path)
            points = np.asarray(pc.points, dtype=np.float32)
        else:
            points = np.loadtxt(path).astype(np.float32)
        return np.ascontiguousarray(points[:, :3], dtype=np.float32)

    def _sample_or_pad(self, points, num_points):
        total = points.shape[0]
        if total == num_points:
            return points
        if total > num_points:
            choice = np.random.choice(total, num_points, replace=False)
            return points[choice]
        choice = np.random.choice(total, num_points, replace=True)
        return points[choice]

    def _ensure_exact_points(self, points, num_points):
        total = points.shape[0]
        if num_points is None:
            return points
        if total != num_points:
            raise ValueError(
                f"sparse point count does not match configuration: got={total}, expected={num_points}. "
                f"The current data pipeline disables downsampling/padding; ensure the input point count matches exactly."
            )
        return points

    @staticmethod
    def _build_contiguous_part_indices(num_points, num_parts, source):
        num_points = int(num_points)
        num_parts = int(num_parts)
        if num_points % num_parts != 0:
            raise ValueError(
                f"Cannot build contiguous part indices for {source}: "
                f"num_points={num_points}, num_parts={num_parts}"
            )
        return np.arange(num_points, dtype=np.int64).reshape(num_parts, num_points // num_parts)

    @staticmethod
    def _resample_indices_to_length(indices, target_len):
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        target_len = int(target_len)
        if target_len <= 0:
            raise ValueError(f'target_len must be positive, got {target_len}')
        if idx.size == 0:
            return np.zeros((target_len,), dtype=np.int64)
        if idx.size == target_len:
            return idx
        if idx.size > target_len:
            picks = np.linspace(0, idx.size - 1, num=target_len, dtype=np.int64)
            return idx[picks]
        reps = (target_len + idx.size - 1) // idx.size
        return np.tile(idx, reps)[:target_len]

    def _emit_patch_idx_fix_once(self, sample_id, field_name, old_shape, new_shape):
        key = (str(sample_id), str(field_name))
        if key in self._reported_patch_idx_fixes:
            return
        self._reported_patch_idx_fixes.add(key)
        print(
            f"[Dataset Warning] Auto-fix {field_name} for sample {sample_id}: "
            f"{old_shape} -> {new_shape} (num_parts={self.num_parts})"
        )

    def _normalize_patch_index_array(self, patch_idx, point_count, sample_id, field_name, points_xyz=None):
        arr = np.asarray(patch_idx, dtype=np.int64)
        point_count = int(point_count)
        if arr.ndim != 2:
            raise ValueError(
                f"{field_name} shape must be 2D [P, K]; 1D label arrays are no longer supported, "
                f"but got {tuple(arr.shape)} "
                f"(sample_id={sample_id})"
            )

        if arr.shape[0] != int(self.num_parts):
            target_parts = int(self.num_parts)
            rows = [arr[i] for i in range(arr.shape[0])]
            if rows:
                if len(rows) >= target_parts:
                    rows = rows[:target_parts]
                else:
                    base = list(rows)
                    while len(rows) < target_parts:
                        rows.append(base[len(rows) % len(base)])
                converted = np.stack(rows, axis=0)
                self._emit_patch_idx_fix_once(sample_id, field_name, tuple(arr.shape), tuple(converted.shape))
                arr = converted
        return arr

    def _refresh_cached_pointcloud_paths(self, cached_samples):
        updated = 0
        normalized = []
        for sample in cached_samples:
            sample = dict(sample)
            for key in self._POINTCLOUD_BASENAMES:
                path = sample.get(key, '')
                if not path:
                    continue
                directory = os.path.dirname(path)
                preferred_path = self._resolve_pointcloud_path(directory, key, fallback_path=path)
                if preferred_path != path:
                    sample[key] = preferred_path
                    updated += 1
            normalized.append(sample)
        return normalized, updated

    @classmethod
    def _resolve_pointcloud_path(cls, directory, key, fallback_path=None):
        for base_name in cls._POINTCLOUD_BASENAMES[key]:
            npy_path = os.path.join(directory, f'{base_name}.npy')
            if os.path.exists(npy_path):
                return npy_path
            ply_path = os.path.join(directory, f'{base_name}.ply')
            if os.path.exists(ply_path):
                return ply_path
        if fallback_path is not None:
            return fallback_path
        primary = cls._POINTCLOUD_BASENAMES[key][0]
        return os.path.join(directory, f'{primary}.ply')

    def _save_cache_samples(self, samples, note=None):
        try:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump({'split': self.split, 'samples': samples}, f)
            suffix = f" ({note})" if note else ''
            _dataset_info(f"[{self.split}] Sample index cache saved: {self.cache_path}{suffix}")
        except Exception as e:
            _dataset_info(f"[{self.split}] Sample cache save failed: {e}")

    def _estimate_sample_payload_bytes(self):
        if not self.samples:
            return 0
        if self.memmap_backend is not None:
            try:
                total = int(self.memmap_backend.total_bytes())
                if not self.load_gt_feat:
                    loaded_total = 0
                    for key in self._loaded_array_keys():
                        meta = self.memmap_backend.array_meta.get(key)
                        if meta is None:
                            continue
                        shape = meta.get('shape')
                        dtype = np.dtype(meta.get('dtype'))
                        loaded_total += int(np.prod(shape) * dtype.itemsize)
                    total = loaded_total
                return int(total / max(len(self.samples), 1))
            except Exception:
                return 0
        try:
            sample = self.samples[0]
            keys = ["sparse_path", "x0_feat_path", "gt_path"]
            if self.load_gt_feat:
                keys.append("gt_feat_path")
            if self.load_image_cond:
                keys.append("image_cond_path")
            return int(sum(os.path.getsize(sample[k]) for k in keys if os.path.exists(sample[k])))
        except Exception:
            return 0

    def _log_storage_summary(self):
        if not self.samples:
            return
        if self.memmap_backend is not None:
            parts = []
            for key in self._loaded_array_keys():
                meta = self.memmap_backend.array_meta.get(key)
                if meta is None:
                    continue
                parts.append(f"{key}={meta.get('dtype')}@{meta.get('shape')}")
            storage_mode = 'memmap-sharded' if getattr(self.memmap_backend, 'is_sharded', False) else 'memmap'
            msg = f"[{self.split}] Point-cloud storage: {storage_mode} | " + ", ".join(parts)
            if getattr(self.memmap_backend, 'is_sharded', False):
                msg += f" | shards={self.memmap_backend.num_shards}"
            if self.sample_payload_bytes > 0:
                msg += f" | sample_payload≈{self.sample_payload_bytes / (1024 ** 2):.2f} MB"
            if not self.load_gt_feat:
                msg += " | gt_feat=disabled"
            if self.load_image_cond:
                if self.memmap_backend.image_cond_matches(
                    key=self.image_cond_key,
                    filename=self.image_cond_filename,
                    array_key=self.image_cond_array_key,
                ):
                    msg += f" | image_cond={self.image_cond_array_key}:{self.image_cond_key} via-memmap"
                else:
                    cached_meta = getattr(self.memmap_backend, 'image_cond_meta', {}) or {}
                    cached_desc = (
                        f"{cached_meta.get('array_key', '<none>')}:"
                        f"{cached_meta.get('filename', '<none>')}:"
                        f"{cached_meta.get('key', '<none>')}"
                    )
                    msg += (
                        f" | image_cond=requested {self.image_cond_array_key}:"
                        f"{self.image_cond_filename}:{self.image_cond_key} "
                        f"cache={cached_desc} fallback=npz"
                    )
                if self.image_cond_max_tokens_per_view > 0:
                    msg += f" | image_tokens_per_view<={self.image_cond_max_tokens_per_view}"
            _dataset_info(msg)
            return
        parts = []
        for key in ("sparse_path", "gt_path"):
            counts = {}
            for sample in self.samples:
                ext = os.path.splitext(sample[key])[1].lower()
                counts[ext] = counts.get(ext, 0) + 1
            parts.append(f"{key}={counts}")
        msg = f"[{self.split}] Point-cloud storage: " + ", ".join(parts)
        if self.sample_payload_bytes > 0:
            msg += f" | sample_payload≈{self.sample_payload_bytes / (1024 ** 2):.2f} MB"
        if not self.load_gt_feat:
            msg += " | gt_feat=disabled"
        if self.load_image_cond:
            msg += f" | image_cond={self.image_cond_filename}:{self.image_cond_key}"
            if self.image_cond_max_tokens_per_view > 0:
                msg += f" | image_tokens_per_view<={self.image_cond_max_tokens_per_view}"
        _dataset_info(msg)
