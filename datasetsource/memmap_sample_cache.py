"""Read split-level or sharded point-cloud feature caches."""

import json
import os
import bisect

import numpy as np

from utils.dataset_feature_layout import (
    DEFAULT_IMAGE_COND_FILENAME, DEFAULT_IMAGE_COND_KEY, DEFAULT_IMAGE_COND_ARRAY_KEY,
)

DEFAULT_MEMMAP_CACHE_DIRNAME = ".controlflow3d_memmap"


def get_split_cache_dir(root_dir, split):
    return os.path.join(root_dir, DEFAULT_MEMMAP_CACHE_DIRNAME, split)


def get_manifest_path(root_dir, split):
    return os.path.join(get_split_cache_dir(root_dir, split), "manifest.json")


class SplitMemmapBackend:
    def __init__(self, root_dir, split):
        self.root_dir = root_dir
        self.split = split
        self.owner_pid = os.getpid()
        self.cache_dir = get_split_cache_dir(root_dir, split)
        self.manifest_path = get_manifest_path(root_dir, split)
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if str(manifest.get("split")) != str(split):
            raise RuntimeError(f"Memmap split mismatch: {manifest.get('split')} vs {split}")
        self.version = int(manifest.get("version", 1))
        self.array_meta = dict(manifest.get("arrays", {}))
        self.image_cond_meta = dict(manifest.get("image_cond", {}))
        self.image_conds_meta = dict(manifest.get("image_conds", {}))
        self.is_sharded = bool(manifest.get("shards"))
        self.shards = []
        self._shard_starts = []
        self.samples = []
        for index, item in enumerate(manifest["samples"]):
            sample = {
                "sample_id": item["sample_id"],
                "category": str(item["category"]),
                "packed_index": int(index),
            }
            self.samples.append(sample)
        self.num_samples = len(self.samples)
        self._array_cache = {}
        if self.is_sharded:
            for shard in manifest["shards"]:
                start_index = int(shard["start_index"])
                num_samples = int(shard["num_samples"])
                self.shards.append(
                    {
                        "id": int(shard.get("id", len(self.shards))),
                        "start_index": start_index,
                        "num_samples": num_samples,
                        "end_index": start_index + num_samples,
                        "arrays": dict(shard["arrays"]),
                    }
                )
                self._shard_starts.append(start_index)
            self.num_shards = len(self.shards)
            self.shard_ranges = [(item["start_index"], item["end_index"], item["id"]) for item in self.shards]
        else:
            self.num_shards = 1
            self.shard_ranges = [(0, self.num_samples, 0)]
        self._validate_manifest_files()

    def _iter_manifest_array_paths(self):
        if not self.is_sharded:
            for key, meta in self.array_meta.items():
                yield key, os.path.join(self.cache_dir, meta["path"])
            return

        for shard in self.shards:
            shard_id = int(shard.get("id", -1))
            for key, meta in shard["arrays"].items():
                yield f"shard={shard_id}:{key}", os.path.join(self.cache_dir, meta["path"])

    def _validate_manifest_files(self):
        missing = []
        for key, path in self._iter_manifest_array_paths():
            if not os.path.isfile(path):
                missing.append((key, path))
                if len(missing) >= 8:
                    break
        if not missing:
            return

        missing_preview = "; ".join([f"{k} -> {p}" for k, p in missing])
        raise RuntimeError(
            "Memmap cache is incomplete/corrupted; some files referenced by manifest are missing. "
            f"Examples: {missing_preview}. Please rebuild cache with utils/build_memmap_dataset_cache.py --overwrite."
        )

    def clone_for_current_process(self):
        return SplitMemmapBackend(self.root_dir, self.split)

    @classmethod
    def exists(cls, root_dir, split):
        return os.path.isfile(get_manifest_path(root_dir, split))

    def _physical_array_keys(self):
        if not self.is_sharded:
            return self.array_meta.keys()
        return self.shards[0]["arrays"].keys() if self.shards else ()

    def _resolve_array_key(self, key):
        key = str(key)
        available = self._physical_array_keys()
        if key in available:
            return key
        # Resolve the legacy VGGT key only when its feature metadata matches.
        if key == DEFAULT_IMAGE_COND_ARRAY_KEY and "image_cond" in available:
            meta = self.image_conds_meta.get("image_cond", self.image_cond_meta)
            if (meta.get("enabled", False)
                    and meta.get("filename") == DEFAULT_IMAGE_COND_FILENAME
                    and meta.get("key") == DEFAULT_IMAGE_COND_KEY):
                return "image_cond"
        return key

    def has_array_key(self, key):
        return self._resolve_array_key(key) in self._physical_array_keys()

    def image_cond_matches(self, *, key=None, filename=None, array_key=None):
        requested = str(array_key or self.image_cond_meta.get("array_key") or "image_cond")
        physical = self._resolve_array_key(requested)
        meta = self.image_conds_meta.get(physical)
        if meta is None:
            legacy_key = self.image_cond_meta.get("array_key") or "image_cond"
            if physical != legacy_key:
                return False
            meta = self.image_cond_meta
        if not meta.get("enabled", False) or not self.has_array_key(requested):
            return False
        if key is not None and str(meta.get("key")) != str(key):
            return False
        if filename is not None and str(meta.get("filename", "")) != str(filename):
            return False
        return True

    def _default_load_keys(self):
        if not self.is_sharded:
            return tuple(self.array_meta.keys())
        if not self.shards:
            return tuple()
        return tuple(self.shards[0]["arrays"].keys())

    def _normalize_load_keys(self, keys):
        if keys is None:
            keys = self._default_load_keys()
        normalized = tuple(str(key) for key in keys)
        if not normalized:
            return tuple()
        missing = [key for key in normalized if not self.has_array_key(key)]
        if missing:
            raise KeyError(f"Memmap key(s) missing from manifest: {missing}")
        return normalized

    def _get_array(self, key, shard_index=None):
        key = self._resolve_array_key(key)
        if not self.is_sharded:
            array = self._array_cache.get(key)
            if array is not None:
                return array
            meta = self.array_meta[key]
            path = os.path.join(self.cache_dir, meta["path"])
            array = np.load(path, mmap_mode="r")
            self._array_cache[key] = array
            return array

        shard_cache = self._array_cache.setdefault(int(shard_index), {})
        array = shard_cache.get(key)
        if array is not None:
            return array
        meta = self.shards[int(shard_index)]["arrays"][key]
        path = os.path.join(self.cache_dir, meta["path"])
        array = np.load(path, mmap_mode="r")
        shard_cache[key] = array
        return array

    def total_bytes(self):
        total = 0
        if not self.is_sharded:
            for meta in self.array_meta.values():
                path = os.path.join(self.cache_dir, meta["path"])
                total += os.path.getsize(path)
            return int(total)

        for shard in self.shards:
            for meta in shard["arrays"].values():
                path = os.path.join(self.cache_dir, meta["path"])
                total += os.path.getsize(path)
        return int(total)

    def _resolve_shard_index(self, index):
        shard_pos = bisect.bisect_right(self._shard_starts, int(index)) - 1
        if shard_pos < 0 or shard_pos >= len(self.shards):
            raise IndexError(f"packed sample index out of range: {index}")
        shard = self.shards[shard_pos]
        if int(index) >= shard["end_index"]:
            raise IndexError(f"packed sample index out of range: {index}")
        return shard_pos

    def load_sample(self, index, keys=None):
        index = int(index)
        if index < 0 or index >= self.num_samples:
            raise IndexError(f"packed sample index out of range: {index}")
        load_keys = self._normalize_load_keys(keys)
        sample = {}
        if not load_keys:
            return sample
        if not self.is_sharded:
            for key in load_keys:
                sample[key] = np.asarray(self._get_array(key)[index])
            return sample

        shard_index = self._resolve_shard_index(index)
        shard = self.shards[shard_index]
        local_index = index - shard["start_index"]
        for key in load_keys:
            sample[key] = np.asarray(self._get_array(key, shard_index=shard_index)[local_index])
        return sample
