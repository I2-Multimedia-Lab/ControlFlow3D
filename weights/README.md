# Encoder Weights

Configure [Utonia](https://github.com/Pointcept/Utonia) and
[VGGT](https://github.com/facebookresearch/vggt) in separate environments using
their official instructions.

<a id="checkpoints"></a>

## Checkpoints

| Encoder | Download | Local path |
| --- | --- | --- |
| [Utonia](https://huggingface.co/Pointcept/Utonia) | [utonia.pth](https://huggingface.co/Pointcept/Utonia/resolve/main/utonia.pth) | `weights/utonia/utonia.pth` |
| [VGGT](https://huggingface.co/facebook/VGGT-1B) | [model.pt](https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt) | `weights/vggt/model.pt` |

Export commands are in [Feature preparation](../README.md#feature-preparation).

<a id="vggt-projection"></a>
<a id="feature-projections"></a>

## Feature Projections

The unified export command supports:

| Option | Purpose |
| --- | --- |
| `--utonia_projector_ckpt path/to/projector.pth` | Load a saved Utonia projector |
| `--vggt_feature_reduction mean-pool` | Reduce VGGT channels by mean pooling |
| `--vggt_feature_reduction projection --vggt_projection_file path/to/projection.npz` | Load a saved VGGT projection |

VGGT projections accept NPY `[D,512]`, or NPZ `matrix [D,512]` /
`weight [512,D]` with optional `bias [512]`.

For pretrained models, use the corresponding projectors, preprocessing and view
order. Model-weight terms are provided in the linked model cards.
