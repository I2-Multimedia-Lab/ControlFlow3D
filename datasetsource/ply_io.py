"""Read binary little-endian PLY coordinates and point-group indices.

The dataset loader and cache builder share these readers. XYZ arrays retain
file vertex order and use float32 values. Part and patch lists contain point
indices.
"""

import struct

import numpy as np

PLY_DTYPE_MAP = {
    'float': np.float32,
    'float32': np.float32,
    'double': np.float64,
    'float64': np.float64,
    'char': np.int8,
    'uchar': np.uint8,
    'short': np.int16,
    'ushort': np.uint16,
    'int': np.int32,
    'int32': np.int32,
    'uint': np.uint32,
    'uint32': np.uint32,
}

def read_generic_ply_xyz(path):
        """Read XYZ coordinates from a binary little-endian PLY file.

        Args:
            path: PLY file with scalar vertex properties, including x, y and z.

        Returns:
            Coordinates [N, 3] with dtype float32, in file vertex order.
        """
        with open(path, 'rb') as f:
            vertex_count = None
            vertex_props = []
            in_vertex = False
            data_start = 0

            while True:
                raw_line = f.readline()
                if not raw_line:
                    raise ValueError(f"Invalid PLY header (unexpected EOF): {path}")
                line = raw_line.decode('ascii').strip()
                if line == 'end_header':
                    data_start = f.tell()
                    break
                if line.startswith('format '):
                    if 'binary_little_endian' not in line:
                        raise ValueError(f"Only binary_little_endian PLY is supported: {path}")
                elif line.startswith('element '):
                    tok = line.split()
                    elem_name = tok[1]
                    elem_count = int(tok[2])
                    in_vertex = (elem_name == 'vertex')
                    if in_vertex:
                        vertex_count = elem_count
                        vertex_props = []
                elif in_vertex and line.startswith('property '):
                    tok = line.split()
                    if tok[1] == 'list':
                        raise ValueError(f"Vertex list properties are not supported in fast PLY reader: {path}")
                    prop_type = tok[1].lower()
                    prop_name = tok[2]
                    if prop_type not in PLY_DTYPE_MAP:
                        raise ValueError(f"Unsupported vertex property type {prop_type} in {path}")
                    vertex_props.append((prop_name, PLY_DTYPE_MAP[prop_type]))

            if vertex_count is None or not vertex_props:
                raise ValueError(f"PLY vertex header missing in {path}")

            xyz_names = ('x', 'y', 'z')
            if tuple(name for name, _ in vertex_props[:3]) != xyz_names:
                prop_names = [name for name, _ in vertex_props]
                if not all(name in prop_names for name in xyz_names):
                    raise ValueError(f"PLY vertex properties missing x/y/z in {path}")

            vertex_dtype = np.dtype(vertex_props)
            f.seek(data_start)
            vertex_data = np.frombuffer(f.read(vertex_count * vertex_dtype.itemsize), dtype=vertex_dtype, count=vertex_count)
            coords = np.stack(
                [vertex_data['x'], vertex_data['y'], vertex_data['z']],
                axis=1,
            ).astype(np.float32, copy=False)
        return coords.copy()

def read_part_ply_indices(path):
        with open(path, 'rb') as f:
            vertex_count = None
            part_count = 0
            vertex_props = []
            part_props = []
            in_vertex = False
            in_part = False

            while True:
                raw_line = f.readline()
                if not raw_line:
                    raise ValueError(f"Invalid PLY header (unexpected EOF): {path}")
                line = raw_line.decode('ascii').strip()
                if line == 'end_header':
                    break
                if line.startswith('element '):
                    tok = line.split()
                    elem_name = tok[1]
                    elem_count = int(tok[2])
                    in_vertex = (elem_name == 'vertex')
                    in_part = (elem_name == 'part')
                    if in_vertex:
                        vertex_count = elem_count
                        vertex_props = []
                    elif in_part:
                        part_count = elem_count
                        part_props = []
                elif line.startswith('property '):
                    if in_vertex:
                        vertex_props.append(line)
                    elif in_part:
                        part_props.append(line)

            if vertex_count is None:
                raise ValueError(f"PLY vertex header missing in {path}")
            if part_count <= 0 or not part_props:
                return None

            vertex_fields = []
            for prop in vertex_props:
                tok = prop.split()
                if tok[1] == 'list':
                    raise ValueError(f"Vertex list properties are not supported in part PLY reader: {path}")
                prop_type = tok[1].lower()
                prop_name = tok[2]
                if prop_type not in PLY_DTYPE_MAP:
                    raise ValueError(f"Unsupported vertex property type {prop_type} in {path}")
                vertex_fields.append((prop_name, PLY_DTYPE_MAP[prop_type]))
            vertex_dtype = np.dtype(vertex_fields)
            f.read(vertex_count * vertex_dtype.itemsize)

            type_size = {
                'char': 1, 'uchar': 1,
                'short': 2, 'ushort': 2,
                'int': 4, 'uint': 4,
                'float': 4, 'double': 8,
            }
            type_struct = {
                'char': 'b', 'uchar': 'B',
                'short': 'h', 'ushort': 'H',
                'int': 'i', 'uint': 'I',
                'float': 'f', 'double': 'd',
            }
            list_prop = None
            scalar_props = []
            for prop in part_props:
                tok = prop.split()
                if tok[1] == 'list':
                    list_prop = {'count_type': tok[2].lower(), 'item_type': tok[3].lower()}
                else:
                    scalar_props.append(tok[1].lower())
            if list_prop is None:
                return None

            count_type = list_prop['count_type']
            item_type = list_prop['item_type']
            if count_type not in type_struct or item_type not in type_struct:
                raise ValueError(f"Unsupported part list types count={count_type} item={item_type} in {path}")
            count_fmt = '<' + type_struct[count_type]
            item_fmt = type_struct[item_type]
            count_size = type_size[count_type]
            item_size = type_size[item_type]

            part_indices = []
            for _ in range(part_count):
                for prop_type in scalar_props:
                    if prop_type not in type_size:
                        raise ValueError(f"Unsupported part property type {prop_type} in {path}")
                    f.read(type_size[prop_type])
                count_raw = f.read(count_size)
                if not count_raw:
                    break
                (count,) = struct.unpack(count_fmt, count_raw)
                data = f.read(item_size * count)
                items = struct.unpack('<' + item_fmt * count, data) if count > 0 else ()
                part_indices.append(items)

            if not part_indices:
                return None
            lengths = {len(items) for items in part_indices}
            if len(lengths) != 1:
                raise ValueError(f"Part list lengths not equal in {path}: {sorted(lengths)}")
            part_indices = np.asarray(part_indices, dtype=np.int64)
            max_idx = int(part_indices.max()) if part_indices.size else -1
            if max_idx >= vertex_count:
                raise ValueError(f"Part indices out of range in {path}: max={max_idx}, n={vertex_count}")
            return part_indices

def remap_patch_indices_to_local(path, patch_indices, dense_index, patch_index_space):
    patch_indices = np.asarray(patch_indices, dtype=np.int64)
    patch_index_space = str(patch_index_space or "local_vertex").strip().lower()
    if patch_index_space == "local_vertex":
        return patch_indices
    if patch_index_space != "dense_index":
        raise ValueError(f"Unsupported patch_index_space={patch_index_space!r} in {path}")
    if dense_index is None:
        raise ValueError(f"patch_index_space=dense_index but vertex.dense_index is missing in {path}")

    dense_index = np.asarray(dense_index, dtype=np.int64)
    dense_to_local = {int(value): idx for idx, value in enumerate(dense_index.tolist())}
    remapped = np.empty_like(patch_indices, dtype=np.int64)
    missing = []
    flat_src = patch_indices.reshape(-1)
    flat_dst = remapped.reshape(-1)
    for idx, value in enumerate(flat_src):
        local_idx = dense_to_local.get(int(value))
        if local_idx is None:
            if len(missing) < 16:
                missing.append(int(value))
            continue
        flat_dst[idx] = local_idx
    if missing:
        raise ValueError(
            f"Patch dense indices cannot be resolved back to local vertices in {path}: "
            f"first_missing={missing}"
        )
    return remapped

def _skip_ply_element_payload(handle, element_count, prop_lines):
    type_size = {
        'char': 1, 'uchar': 1,
        'short': 2, 'ushort': 2,
        'int': 4, 'int32': 4, 'uint': 4,
        'float': 4, 'float32': 4, 'double': 8,
    }
    type_struct = {
        'char': 'b', 'uchar': 'B',
        'short': 'h', 'ushort': 'H',
        'int': 'i', 'int32': 'i', 'uint': 'I',
        'float': 'f', 'float32': 'f', 'double': 'd',
    }
    props = []
    for line in prop_lines:
        tok = line.split()
        if tok[1] == 'list':
            props.append(('list', tok[2].lower(), tok[3].lower()))
        else:
            props.append(('scalar', tok[1].lower(), None))

    for _ in range(int(element_count)):
        for kind, first_type, second_type in props:
            if kind == 'scalar':
                handle.read(type_size[first_type])
                continue
            count_struct = struct.Struct('<' + type_struct[first_type])
            raw_count = handle.read(count_struct.size)
            if len(raw_count) != count_struct.size:
                raise ValueError('Truncated PLY list count while skipping element payload')
            (count,) = count_struct.unpack(raw_count)
            handle.read(type_size[second_type] * int(count))

def read_patch_ply_xyz(path):
    with open(path, 'rb') as f:
        vertex_count = None
        part_count = 0
        patch_count = 0
        in_vertex = False
        in_part = False
        in_patch = False
        vertex_props = []
        part_props = []
        patch_props = []
        element_order = []
        patch_index_space = "local_vertex"

        while True:
            raw_line = f.readline()
            if not raw_line:
                raise ValueError(f'Invalid PLY header: {path}')
            line = raw_line.decode('ascii').strip()
            if line.startswith('comment patch_index_space '):
                patch_index_space = line.split(' ', 2)[2].strip().lower()
            if line == 'end_header':
                break
            if line.startswith('format ') and 'binary_little_endian' not in line:
                raise ValueError(f'Only binary_little_endian PLY is supported: {path}')
            elif line.startswith('element '):
                tok = line.split()
                elem_name = tok[1]
                elem_count = int(tok[2])
                element_order.append(elem_name)
                in_vertex = elem_name == 'vertex'
                in_part = elem_name == 'part'
                in_patch = elem_name == 'patch'
                if in_vertex:
                    vertex_count = elem_count
                    vertex_props = []
                elif in_part:
                    part_count = elem_count
                    part_props = []
                elif in_patch:
                    patch_count = elem_count
                    patch_props = []
            elif line.startswith('property '):
                if in_vertex:
                    tok = line.split()
                    if tok[1] == 'list':
                        raise ValueError(f'Vertex list property is unsupported: {path}')
                    prop_type = tok[1].lower()
                    prop_name = tok[2]
                    vertex_props.append((prop_name, PLY_DTYPE_MAP[prop_type]))
                elif in_part:
                    part_props.append(line)
                elif in_patch:
                    patch_props.append(line)

        if vertex_count is None:
            raise ValueError(f'PLY vertex header missing in {path}')

        vertex_dtype = np.dtype(vertex_props)
        vertex_data = np.frombuffer(
            f.read(vertex_count * vertex_dtype.itemsize),
            dtype=vertex_dtype,
            count=vertex_count,
        )
        xyz_names = ('x', 'y', 'z')
        if not all(name in vertex_data.dtype.names for name in xyz_names):
            raise ValueError(f'PLY vertex properties missing x/y/z in {path}')
        coords = np.stack([vertex_data['x'], vertex_data['y'], vertex_data['z']], axis=1).astype(np.float32, copy=False)
        dense_index = (
            np.asarray(vertex_data['dense_index'], dtype=np.int64)
            if 'dense_index' in vertex_data.dtype.names
            else None
        )

        if patch_count <= 0 or not patch_props:
            return coords.copy(), None

        if part_count > 0 and part_props and 'part' in element_order and 'patch' in element_order:
            if element_order.index('part') < element_order.index('patch'):
                _skip_ply_element_payload(f, part_count, part_props)

        list_prop = None
        patch_fields = []
        for prop in patch_props:
            tok = prop.split()
            if tok[1] == 'list':
                list_prop = {
                    'count_type': tok[2].lower(),
                    'item_type': tok[3].lower(),
                    'name': tok[4],
                }
            else:
                patch_fields.append(tok[1].lower())

        if list_prop is None:
            return coords.copy(), None

        type_size = {
            'char': 1, 'uchar': 1,
            'short': 2, 'ushort': 2,
            'int': 4, 'uint': 4,
            'float': 4, 'double': 8,
        }
        type_struct = {
            'char': 'b', 'uchar': 'B',
            'short': 'h', 'ushort': 'H',
            'int': 'i', 'uint': 'I',
            'float': 'f', 'double': 'd',
        }

        list_count_type = list_prop['count_type']
        list_item_type = list_prop['item_type']
        if list_count_type not in type_struct or list_item_type not in type_struct:
            raise ValueError(
                f'Unsupported patch list types count={list_count_type} item={list_item_type} in {path}'
            )
        count_fmt = '<' + type_struct[list_count_type]
        item_fmt = type_struct[list_item_type]
        count_size = type_size[list_count_type]
        item_size = type_size[list_item_type]

        patch_indices = []
        for _ in range(patch_count):
            for prop_type in patch_fields:
                if prop_type not in type_size:
                    raise ValueError(f'Unsupported patch property type {prop_type} in {path}')
                f.read(type_size[prop_type])

            count_raw = f.read(count_size)
            if not count_raw:
                break
            (count,) = struct.unpack(count_fmt, count_raw)
            if count > 0:
                data = f.read(item_size * count)
                items = struct.unpack('<' + item_fmt * count, data)
                patch_indices.append(items)
            else:
                patch_indices.append(())

        if not patch_indices:
            return coords.copy(), None
        lengths = {len(items) for items in patch_indices}
        if len(lengths) != 1:
            raise ValueError(f'Patch list lengths not equal in {path}: {sorted(lengths)}')
        patch_indices = remap_patch_indices_to_local(
            path,
            np.asarray(patch_indices, dtype=np.int64),
            dense_index=dense_index,
            patch_index_space=patch_index_space,
        )
        max_idx = int(patch_indices.max()) if patch_indices.size else -1
        if max_idx >= coords.shape[0]:
            raise ValueError(f'Patch indices out of range in {path}: max={max_idx}, n={coords.shape[0]}')
    return coords.copy(), patch_indices
