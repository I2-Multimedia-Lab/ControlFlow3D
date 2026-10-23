"""Dataset point-cloud reading, feature export and metadata validation."""

from pathlib import Path
import json
import os
import tempfile

from utils.dataset_feature_layout import (
    DEFAULT_IMAGE_COND_KEY, DEFAULT_IMAGE_COND_FILENAME, DEFAULT_IMAGE_META_FILENAME,
)


def _read_ascii_ply_xyz(path):
    import numpy as np
    with Path(path).open('r', encoding='ascii') as handle:
        if handle.readline().strip() != 'ply':
            raise ValueError(f'Invalid PLY signature: {path}')
        vertex_count = None
        properties = []
        in_vertex = False
        first_element = None
        ascii_format = False
        for raw in handle:
            line = raw.strip()
            if line == 'end_header':
                break
            if line.startswith('format '):
                ascii_format = line.split()[1] == 'ascii'
            elif line.startswith('element '):
                _, element, count = line.split()
                if first_element is None:
                    first_element = element
                in_vertex = element == 'vertex'
                if in_vertex:
                    vertex_count = int(count)
            elif in_vertex and line.startswith('property '):
                fields = line.split()
                if fields[1] == 'list':
                    raise ValueError(f'List-valued vertex properties are not supported: {path}')
                properties.append(fields[-1])
        else:
            raise ValueError(f'Truncated PLY header: {path}')
        if not ascii_format or first_element != 'vertex' or vertex_count is None:
            raise ValueError(f'Expected an ASCII PLY with vertices as its first element: {path}')
        if not all(name in properties for name in ('x', 'y', 'z')):
            raise ValueError(f'PLY vertex properties must contain x/y/z: {path}')
        indices = [properties.index(name) for name in ('x', 'y', 'z')]
        points = np.empty((vertex_count, 3), dtype=np.float32)
        for row in range(vertex_count):
            fields = handle.readline().split()
            if len(fields) != len(properties):
                raise ValueError(f'Invalid or truncated PLY vertex row {row}: {path}')
            points[row] = [float(fields[col]) for col in indices]
        return points


def read_point_cloud_xyz(path):
    """Read NPY, XYZ/TXT or common ASCII/binary-little-endian PLY in file order."""
    import numpy as np
    from datasetsource.ply_io import read_generic_ply_xyz
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == '.npy':
        points = np.load(path, allow_pickle=False)
    elif suffix in ('.xyz', '.txt'):
        points = np.loadtxt(path, dtype=np.float32, ndmin=2)
    elif suffix == '.ply':
        with path.open('rb') as handle:
            if handle.readline().strip() != b'ply':
                raise ValueError(f'Invalid PLY signature: {path}')
            fmt = None
            first_element = None
            for line in handle:
                if line.startswith(b'format '):
                    fmt = line.split()[1]
                if line.startswith(b'element ') and first_element is None:
                    first_element = line.split()[1]
                if line.strip() == b'end_header':
                    break
        if first_element != b'vertex':
            raise ValueError(f'PLY vertices must be the first element: {path}')
        if fmt == b'ascii':
            points = _read_ascii_ply_xyz(path)
        elif fmt == b'binary_little_endian':
            points = read_generic_ply_xyz(str(path))
        else:
            raise ValueError(f'Unsupported PLY format: {fmt!r} in {path}')
    else:
        raise ValueError(f'Unsupported point-cloud extension {suffix!r}; use .xyz, .txt, .ply or .npy')
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3 or points.shape[0] == 0:
        raise ValueError(f'Expected nonempty [N, >=3] points, got {points.shape}: {path}')
    points = np.ascontiguousarray(points[:, :3], dtype=np.float32)
    if not np.isfinite(points).all():
        raise ValueError(f'Point cloud contains NaN/Inf; remove invalid rows before encoding: {path}')
    return points


def read_sparse_features(path, num_points, feature_dim=128):
    import numpy as np
    features = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if features.shape != (int(num_points), int(feature_dim)):
        raise ValueError(f'Features must be [{num_points},{feature_dim}], got {features.shape}: {path}')
    if not np.isfinite(features).all():
        raise ValueError(f'Sparse features contain NaN/Inf: {path}')
    return np.ascontiguousarray(features)


def read_aligned_dataset_points(path):
    """Reject stale/reordered sibling NPY/PLY copies before positional encoding."""
    import numpy as np
    path = Path(path)
    points = read_point_cloud_xyz(path)
    if path.suffix.lower() in ('.npy', '.ply'):
        sibling = path.with_suffix('.ply' if path.suffix.lower() == '.npy' else '.npy')
        if sibling.is_file():
            alternate = read_point_cloud_xyz(sibling)
            if not np.array_equal(points, alternate):
                raise ValueError(
                    f'NPY/PLY coordinate copies differ in values/order: {path} and {sibling}. '
                    'Regenerate the stale copy from the same source ordering before feature export.'
                )
    return points


def atomic_write_json(path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(metadata, handle, indent=2, allow_nan=False)
            handle.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_save_npy(path, features, num_points, feature_dim=128):
    import numpy as np
    features = np.asarray(features, dtype=np.float32)
    if features.shape != (num_points, feature_dim) or not np.isfinite(features).all():
        raise ValueError(f'Expected finite [{num_points},{feature_dim}] features, got {features.shape}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, 'wb') as handle:
            np.save(handle, np.ascontiguousarray(features), allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def inspect_vggt_npz(path, *, num_views=8, max_tokens=32, feature_dim=512, feature_dtype=None):
    import numpy as np
    with np.load(path, allow_pickle=False) as package:
        if DEFAULT_IMAGE_COND_KEY not in package:
            raise KeyError(f'Missing {DEFAULT_IMAGE_COND_KEY!r} key: {path}')
        tokens = np.asarray(package[DEFAULT_IMAGE_COND_KEY])
        if tokens.ndim != 3 or tokens.shape[0] != num_views or tokens.shape[2] != feature_dim:
            raise ValueError(f'Expected [{num_views},T,{feature_dim}] patch_tokens, got {tokens.shape}: {path}')
        if tokens.shape[1] <= 0 or (max_tokens > 0 and tokens.shape[1] != max_tokens):
            raise ValueError(f'Expected {max_tokens} tokens/view, got {tokens.shape[1]}: {path}')
        if tokens.dtype not in (np.dtype('float16'), np.dtype('float32')) or not np.isfinite(tokens).all():
            raise ValueError(f'patch_tokens must be finite float16/float32 values: {path}')
        if feature_dtype and tokens.dtype != np.dtype(feature_dtype):
            raise ValueError(f'Expected {feature_dtype}, got {tokens.dtype}: {path}; use --overwrite to re-export')
        return {'patch_shape': list(tokens.shape), 'feature_dtype': tokens.dtype.name,
                'view_count': int(tokens.shape[0]), 'feature_file': Path(path).name,
                'feature_key': DEFAULT_IMAGE_COND_KEY}


def metadata_mismatches(metadata, observed):
    return [key for key, value in observed.items() if metadata.get(key) != value]


def sync_vggt_metadata(npz_path, meta_path, *, repair=False, **expected):
    """Validate VGGT arrays and optionally update observable JSON metadata."""
    npz_path, meta_path = Path(npz_path), Path(meta_path)
    observed = inspect_vggt_npz(npz_path, **expected)
    if meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding='utf-8'))
        if not isinstance(metadata, dict):
            raise ValueError(f'VGGT metadata must be a JSON object: {meta_path}')
    else:
        metadata = {'schema': 'controlflow3d.vggt.observed',
                    'metadata_source': 'existing_npz_inspection',
                    'export_settings_known': False}
    mismatches = metadata_mismatches(metadata, observed)
    if repair and (mismatches or not meta_path.is_file()):
        metadata.update(observed)
        metadata['meta_file'] = meta_path.name
        atomic_write_json(meta_path, metadata)
    return observed, mismatches


def write_vggt_export(output, patch_tokens, metadata, *, feature_dtype='float16', meta_name=DEFAULT_IMAGE_META_FILENAME):
    """Save validated image tokens and their matching metadata."""
    import numpy as np
    output = Path(output)
    if Path(meta_name).name != meta_name or not meta_name.endswith('.json'):
        raise ValueError('meta_name must be a JSON basename')
    with np.errstate(over='ignore', invalid='ignore'):
        tokens = np.asarray(patch_tokens, dtype=np.dtype(feature_dtype))
    if tokens.ndim != 3 or any(size <= 0 for size in tokens.shape) or not np.isfinite(tokens).all():
        raise ValueError('Export patch_tokens must be nonempty and finite after dtype conversion')
    if tokens.dtype not in (np.dtype('float16'), np.dtype('float32')):
        raise ValueError('Export dtype must be float16 or float32')
    metadata = dict(metadata)
    metadata.update({'model': 'vggt', 'model_id': 'facebook/VGGT-1B',
                     'patch_shape': list(tokens.shape), 'feature_dtype': tokens.dtype.name,
                     'view_count': int(tokens.shape[0]), 'feature_file': output.name,
                     'meta_file': meta_name, 'feature_key': DEFAULT_IMAGE_COND_KEY,
                     'npz_key': DEFAULT_IMAGE_COND_KEY})
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + output.name + '.', suffix='.tmp', dir=output.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, 'wb') as handle:
            # Store image tokens under the shared patch_tokens key.
            np.savez_compressed(handle, patch_tokens=np.ascontiguousarray(tokens))
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    atomic_write_json(output.parent / meta_name, metadata)
    return metadata
