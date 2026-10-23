"""Dataset paths, feature conventions and local encoder checkpoint checks."""

from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_UTONIA_CKPT = ROOT_DIR / "weights" / "utonia" / "utonia.pth"
DEFAULT_VGGT_CKPT = ROOT_DIR / "weights" / "vggt" / "model.pt"
DEFAULT_SPARSE_FEAT_FILENAME = "sparse_local_feat.npy"
DEFAULT_GT_FEAT_FILENAME = "gt_feat.npy"
DEFAULT_IMAGE_META_FILENAME = "vggt_feature_meta.json"
DEFAULT_IMAGE_COND_FILENAME = "vggt_multiview_features.npz"
DEFAULT_IMAGE_COND_KEY = "patch_tokens"  # Key in each sample NPZ.
DEFAULT_IMAGE_COND_ARRAY_KEY = "vggt_image_cond"  # Array name in a memmap cache.
DEFAULT_IMAGE_COND_DIM = 512
DEFAULT_IMAGE_COND_MAX_TOKENS = 32

UTONIA_WEIGHTS_URL = "https://huggingface.co/Pointcept/Utonia/resolve/main/utonia.pth"
VGGT_WEIGHTS_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


def require_local_checkpoint(checkpoint=None, *, encoder="Utonia"):
    """Resolve an existing local checkpoint path."""
    locations = {
        "Utonia": (DEFAULT_UTONIA_CKPT, UTONIA_WEIGHTS_URL),
        "VGGT": (DEFAULT_VGGT_CKPT, VGGT_WEIGHTS_URL),
    }
    if encoder not in locations:
        raise ValueError(f"Unknown checkpoint component: {encoder}")
    default, url = locations[encoder]
    path = Path(checkpoint or default).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"{encoder} checkpoint not found: {path}\n"
            f"Download the weights manually from {url}\n"
            f"Save them at {default}, or pass an existing checkpoint file explicitly.\n"
            "Automatic checkpoint downloads are disabled."
        )
    return str(path)


@dataclass(frozen=True)
class SamplePaths:
    split: str
    category: str
    sample_id: str
    sample_dir: Path

    @property
    def input_dir(self):
        return self.sample_dir / 'input'

    @property
    def gt_dir(self):
        return self.sample_dir / 'gt'

    @property
    def sparse_feature_path(self):
        return self.input_dir / DEFAULT_SPARSE_FEAT_FILENAME

    @property
    def gt_feature_path(self):
        return self.gt_dir / DEFAULT_GT_FEAT_FILENAME

    @property
    def identifier(self):
        return f'{self.split}/{self.category}/{self.sample_id}'

    def point_path(self, target='sparse'):
        if target == 'sparse':
            directory, bases = self.input_dir, ('sparse_xyz', 'sparse_point_cloud')
        elif target == 'gt':
            directory, bases = self.gt_dir, ('gt_xyz', 'upsampled_point_cloud')
        else:
            raise ValueError(f'Unknown point target: {target!r}')
        # Prefer NPY coordinates, matching the source-file dataset loader.
        # The export reader checks that sibling NPY/PLY files retain the same point order.
        for base in bases:
            for suffix in ('.npy', '.ply'):
                path = directory / (base + suffix)
                if path.is_file():
                    return path
        raise FileNotFoundError(f'No {target} point cloud in {directory}; expected {bases} as PLY or NPY')

    def rendering_dir(self, render_root=''):
        if render_root:
            return Path(render_root).expanduser().resolve() / self.category / self.sample_id / 'rendering'
        # Search per-sample images and dataset-level rendering trees.
        dataset_root = self.sample_dir.parents[2]
        candidates = [self.input_dir / 'images', self.input_dir / 'rendering',
                      self.sample_dir / 'rendering']
        for base in (dataset_root, dataset_root.parent):
            for name in ('ShapeNetRendering', 'ShapeNetRendering_selected'):
                candidates.append(base / name / self.category / self.sample_id / 'rendering')
        for directory in candidates:
            if directory.is_dir() and (
                    (directory / 'renderings.txt').is_file()
                    or any(p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg')
                           for p in directory.iterdir())):
                return directory
        return self.input_dir / 'images'


def collect_dataset_samples(dataset_root, split='train', max_samples=0):
    root = Path(dataset_root).expanduser().resolve()
    if split not in ('train', 'test', 'all') or int(max_samples) < 0:
        raise ValueError('split must be train/test/all and max_samples must be nonnegative')
    splits = ('train', 'test') if split == 'all' else (split,)
    samples = []
    for selected in splits:
        split_dir = root / selected
        if not split_dir.is_dir():
            raise NotADirectoryError(split_dir)
        for category in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for sample in sorted(p for p in category.iterdir() if p.is_dir()):
                if (sample / 'input').is_dir():
                    samples.append(SamplePaths(selected, category.name, sample.name, sample))
    if not samples:
        raise RuntimeError(f'No <split>/<category>/<sample>/input directories under {root}')
    return samples[:int(max_samples)] if max_samples else samples
