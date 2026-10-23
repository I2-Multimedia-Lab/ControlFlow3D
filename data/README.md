# ShapeNetPU Data

Download [ShapeNetPU from Hugging Face](https://huggingface.co/datasets/BaldFatTiger/ShapeNetPU)
to `data/ShapeNetPU`. Commands are in [Data preparation](../README.md#data-preparation).

<a id="raw-samples"></a>

## Raw Samples

```text
ShapeNetPU/
|-- train/<category>/<sample>/
|   |-- input/sparse_xyz.ply
|   |-- input/rendering/               # PNG views and renderings.txt
|   `-- gt/gt_xyz.ply
`-- test/<category>/<sample>/
    |-- input/sparse_xyz.ply
    |-- gt/gt_xyz.ply
    `-- gt/gt_mesh.obj
```

Each pair contains 2,048 sparse points and 8,192 GT points. Coordinates support
`binary_little_endian` PLY and NPY. The basenames `sparse_point_cloud` and
`upsampled_point_cloud` are also recognized.

<a id="prepared-features"></a>

## Prepared Features

| Path within a sample | Shape / dtype | Usage |
| --- | --- | --- |
| `input/sparse_local_feat.npy` | `[2048,128]`, float32 | Training and inference |
| `input/vggt_multiview_features.npz` | `patch_tokens=[8,32,512]`, float16 | Training |
| `input/vggt_feature_meta.json` | View order, export settings and hashes | Feature metadata |
| `gt/gt_feat.npy` | `[8192,128]`, float32 | Training |

Feature rows follow point order. The VGGT NPZ key is `patch_tokens`; its memmap
array name is `vggt_image_cond`. Export commands are in
[Feature preparation](../README.md#feature-preparation).

Images are read from `input/rendering/`, `input/images/`, or a
`ShapeNetRendering/<category>/<sample>/rendering/` tree inside or alongside the
dataset. View order follows `renderings.txt` when present, otherwise sorted
filenames. Use `--render_root` for an external rendering tree.

<a id="memmap-cache"></a>

## Data Loading

| Mode | Data source |
| --- | --- |
| Source files | Per-sample point clouds and prepared NPY/NPZ features |
| Memmap (recommended) | Prepacked, memory-mapped NPY arrays |

Memmap reduces small-file I/O and point-cloud parsing. The loader prefers
`.controlflow3d_memmap/<split>/manifest.json` when present and reads source files
otherwise. Set `CONTROLFLOW3D_DISABLE_MEMMAP_CACHE=1` to force source-file loading.

Build the optional cache after feature preparation:

```bash
python utils/build_memmap_dataset_cache.py \
  --dataset_root data/ShapeNetPU --split all \
  --index_mode part --feature_dtype float16
```

Training caches include coordinates, sparse/dense features and image conditions;
test caches include coordinates and sparse features. Add `--include_image_cond`
for image-conditioned testing. Rebuild with `--overwrite` after changing features.

- `part`: contiguous point blocks, without requiring PLY patch indices.
- `patch`: uses PLY patch indices; select `--index_mode patch` for both packing and training.

Additional options: `python utils/build_memmap_dataset_cache.py --help`.
