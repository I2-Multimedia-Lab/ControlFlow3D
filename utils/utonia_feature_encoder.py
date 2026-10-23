"""Load Utonia and export point-aligned latent features."""

import contextlib
import hashlib
import os
import random
import sys
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    import flash_attn  # noqa: F401
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


from utils.dataset_feature_layout import DEFAULT_UTONIA_CKPT, require_local_checkpoint
LOCAL_UTONIA_DEFAULT_CONFIG = {
    'in_channels': 9,
    'enc_channels': (54, 108, 216, 432, 576),
    'enc_depths': (3, 3, 3, 12, 3),
    'enc_num_head': (3, 6, 12, 24, 32),
}
_UTONIA_MODULE = None


def _maybe_append_extra_utonia_site_packages() -> None:
    """Append encoder package directories from CONTROLFLOW3D_UTONIA_SITE_PACKAGES.

    Multiple directories are separated by os.pathsep.
    """
    env_path = os.environ.get('CONTROLFLOW3D_UTONIA_SITE_PACKAGES', '').strip()
    if not env_path:
        return
    for path in env_path.split(os.pathsep):
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)


def get_utonia_module():
    global _UTONIA_MODULE
    if _UTONIA_MODULE is None:
        _maybe_append_extra_utonia_site_packages()
        try:
            import utonia as _utonia
        except ImportError as exc:
            raise ImportError(
                'Utonia is required for 3D feature encoding, but it is not installed.\n'
                'It is an external dependency and is not bundled with this repository. Install it with:\n'
                '    git clone https://github.com/Pointcept/Utonia.git\n'
                '    cd Utonia && pip install -e .\n'
                'Alternatively, set CONTROLFLOW3D_UTONIA_SITE_PACKAGES to the site-packages\n'
                'directory of an environment that already has Utonia installed.\n'
                'See the "3D feature encoding" section of README.md.'
            ) from exc
        _UTONIA_MODULE = _utonia
    return _UTONIA_MODULE


def merge_utonia_config_with_local_fallback(config: Optional[Dict[str, object]]) -> Dict[str, object]:
    config = dict(config or {})
    merged = dict(LOCAL_UTONIA_DEFAULT_CONFIG)
    merged.update({key: value for key, value in config.items() if value is not None})
    return merged


class ProjectorTo128(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def init_projector(projector: ProjectorTo128, seed: int) -> None:
    torch.manual_seed(int(seed))
    for module in projector.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def _extract_projector_input_dim_from_state(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    for key in ('net.0.weight', '0.weight'):
        weight = state_dict.get(key)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            return int(weight.shape[1])
    return None


def load_projector(
    projector_ckpt: Optional[str],
    device: torch.device,
    seed: int,
    in_dim: int,
) -> nn.Module:
    projector = ProjectorTo128(in_dim=in_dim)
    if projector_ckpt:
        state = torch.load(projector_ckpt, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        clean_state = {}
        for key, value in state.items():
            if key.startswith('module.'):
                key = key[7:]
            if key.startswith('projector.'):
                key = key[10:]
            clean_state[key] = value
        ckpt_in_dim = _extract_projector_input_dim_from_state(clean_state)
        if ckpt_in_dim is not None and ckpt_in_dim != in_dim:
            raise RuntimeError(
                f'Projector checkpoint input dim mismatch: ckpt={ckpt_in_dim}, required={in_dim}'
            )
        projector.load_state_dict(clean_state, strict=True)
    else:
        init_projector(projector, seed=seed)
    projector.eval()
    projector.requires_grad_(False)
    return projector.to(device)


def infer_upcast_feat_dim(enc_channels: Optional[Sequence[int]], upcast_levels: int = 2) -> Optional[int]:
    if not enc_channels:
        return None
    channels = [int(channel) for channel in enc_channels]
    required_scales = upcast_levels + 1
    if len(channels) < required_scales:
        return None
    return int(sum(channels[-required_scales:]))


def inspect_utonia_checkpoint(
    checkpoint: str,
    disable_flash: bool,
) -> Dict[str, object]:
    checkpoint = require_local_checkpoint(checkpoint, encoder='Utonia')
    utonia = get_utonia_module()
    custom_config = None
    if disable_flash or not FLASH_ATTN_AVAILABLE:
        custom_config = dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False)
    ckpt = utonia.model.load(
        checkpoint,
        custom_config=custom_config,
        ckpt_only=True,
    )
    return merge_utonia_config_with_local_fallback(ckpt.get('config', {}))


def load_utonia_model(
    checkpoint: str,
    disable_flash: bool,
    device: torch.device,
) -> nn.Module:
    checkpoint = require_local_checkpoint(checkpoint, encoder='Utonia')
    utonia = get_utonia_module()
    custom_config = None
    if disable_flash or not FLASH_ATTN_AVAILABLE:
        custom_config = dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False)
    model = utonia.load(
        checkpoint,
        custom_config=custom_config,
    ).to(device)
    model.eval()
    return model


def upcast_to_feature(point, expected_dim: Optional[int] = None) -> torch.Tensor:
    for _ in range(2):
        if 'pooling_parent' not in point.keys():
            raise RuntimeError('Expected at least 2 pooling levels for 2-level feature upcast.')
        parent = point.pop('pooling_parent')
        inverse = point.pop('pooling_inverse')
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    while 'pooling_parent' in point.keys():
        parent = point.pop('pooling_parent')
        inverse = point.pop('pooling_inverse')
        parent.feat = point.feat[inverse]
        point = parent
    feat = point.feat[point.inverse]
    if expected_dim is not None and feat.shape[-1] != expected_dim:
        raise RuntimeError(
            f'Expected {expected_dim}-d feature after 2-level upcast, got {tuple(feat.shape)}'
        )
    return feat


def split_by_counts(feat: torch.Tensor, counts: Sequence[int]) -> List[torch.Tensor]:
    splits: List[torch.Tensor] = []
    start = 0
    total = int(feat.shape[0])
    for count in counts:
        count = int(count)
        end = start + count
        if end > total:
            raise RuntimeError(
                f'Split-by-counts overflow: requested end={end}, total_points={total}, counts={list(counts)}'
            )
        splits.append(feat[start:end])
        start = end
    if start != total:
        raise RuntimeError(
            f'Split-by-counts mismatch: consumed={start}, total_points={total}, counts={list(counts)}'
        )
    return splits


class PointArrayDataset(Dataset):
    def __init__(
        self,
        length: int,
        load_points: Callable[[int], np.ndarray],
        seed: int,
        scale: float,
        normalize_coord: bool,
        apply_z_positive: bool,
    ) -> None:
        self.length = int(length)
        self.load_points = load_points
        self.seed = int(seed)
        utonia = get_utonia_module()
        self.transform = utonia.transform.default(
            scale=float(scale),
            apply_z_positive=bool(apply_z_positive),
            normalize_coord=bool(normalize_coord),
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, object]:
        coord = np.asarray(self.load_points(int(idx)), dtype=np.float32)
        if coord.ndim != 2 or coord.shape[1] != 3:
            raise RuntimeError(f'Point array must be [N,3], got {tuple(coord.shape)} at idx={idx}')
        point = {
            'coord': coord,
            'color': np.zeros_like(coord, dtype=np.float32),
            'normal': np.zeros_like(coord, dtype=np.float32),
        }
        point = self._transform_with_stable_rng(point, idx)
        point['meta_index'] = int(idx)
        point['meta_num_points'] = int(coord.shape[0])
        return point

    def _transform_with_stable_rng(self, point: Dict[str, np.ndarray], idx: int) -> Dict[str, object]:
        digest = hashlib.blake2b(f'{self.seed}:{idx}'.encode('utf-8'), digest_size=8).digest()
        sample_seed = (self.seed + int.from_bytes(digest, byteorder='little')) % (2 ** 32)
        np_state = np.random.get_state()
        py_state = random.getstate()
        try:
            np.random.seed(sample_seed)
            random.seed(sample_seed)
            return self.transform(point)
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)


def collate_with_meta(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    utonia = get_utonia_module()
    meta_indices = [sample.pop('meta_index') for sample in batch]
    meta_num_points = [sample.pop('meta_num_points') for sample in batch]
    inverse_chunks: List[torch.Tensor] = []
    inverse_offset = 0
    for sample in batch:
        inverse = sample.get('inverse')
        coord = sample.get('coord')
        if not isinstance(inverse, torch.Tensor) or not isinstance(coord, torch.Tensor):
            raise TypeError('Expected transformed samples to contain tensor inverse and coord fields.')
        inverse_chunks.append(inverse + inverse_offset)
        inverse_offset += int(coord.shape[0])
    point = utonia.data.collate_fn(list(batch))
    point['inverse'] = torch.cat(inverse_chunks, dim=0)
    point['meta_index'] = meta_indices
    point['meta_num_points'] = meta_num_points
    return point


class UtoniaSparseFeatureEncoder:
    def __init__(
        self,
        device: str = 'cuda:0',
        checkpoint: str = '',
        projector_ckpt: Optional[str] = None,
        disable_flash: bool = False,
        seed: int = 42,
    ) -> None:
        self.checkpoint = require_local_checkpoint(checkpoint, encoder='Utonia')
        self.device = torch.device(device)
        self.seed = int(seed)
        self.projector_ckpt = projector_ckpt
        self.disable_flash = bool(disable_flash)
        utonia = get_utonia_module()
        utonia.utils.set_seed(self.seed)
        config = inspect_utonia_checkpoint(
            checkpoint=self.checkpoint,
            disable_flash=self.disable_flash,
        )
        self.projector_in_dim = infer_upcast_feat_dim(config.get('enc_channels'))
        if self.projector_in_dim is None:
            raise RuntimeError('Failed to infer Utonia upcast feature dimension from checkpoint config.')
        self.model = load_utonia_model(
            checkpoint=self.checkpoint,
            disable_flash=self.disable_flash,
            device=self.device,
        )
        self.projector = load_projector(
            projector_ckpt=self.projector_ckpt,
            device=self.device,
            seed=self.seed,
            in_dim=self.projector_in_dim,
        )

    def iter_encode_dataset(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
        amp: bool = False,
    ) -> Iterator[Tuple[int, np.ndarray]]:
        loader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=(self.device.type == 'cuda'),
            collate_fn=collate_with_meta,
        )
        amp_enabled = bool(amp and self.device.type == 'cuda')

        with torch.inference_mode():
            for batch in loader:
                meta_indices = [int(idx) for idx in batch.pop('meta_index')]
                meta_num_points = [int(num) for num in batch.pop('meta_num_points')]
                for key, value in batch.items():
                    if torch.is_tensor(value):
                        batch[key] = value.to(self.device, non_blocking=True)

                with (
                    torch.autocast(device_type='cuda', dtype=torch.float16)
                    if amp_enabled
                    else contextlib.nullcontext()
                ):
                    point = self.model(batch)
                    feat = upcast_to_feature(point, expected_dim=self.projector_in_dim)
                    local_feat = self.projector(feat.float()).detach().cpu()

                for sample_idx, sample_feat in zip(meta_indices, split_by_counts(local_feat, meta_num_points)):
                    yield sample_idx, sample_feat.numpy().astype(np.float32, copy=False)
