# Model Checkpoints

<a id="released-model"></a>

## Pretrained Model

Download the **ControlFlow3D ShapeNetPU** checkpoint from
[Google Drive](https://drive.google.com/drive/folders/1Q2PHsJjLeunaD4E-fMe1kjfxKZ6iYM31?usp=sharing) and save it as:

```text
checkpoint/ControlFlow3D_ShapeNetPU.pth
```

Set the checkpoint path, then run the [evaluation command](../README.md#evaluation):

```bash
CHECKPOINT=checkpoint/ControlFlow3D_ShapeNetPU.pth
```

<a id="training-checkpoints"></a>

## Training Checkpoints

Training saves checkpoints under `logs/<run_dir>/`:

| File | Purpose |
| --- | --- |
| `last_checkpoint.pth` | Resume training; includes model, optimizer and scheduler states |
| `best_val_checkpoint.pth` | Model with the best validation CD |

See [Training](../README.md#training) for training and resume commands.

<a id="format"></a>

## File Format

| Field | Contents |
| --- | --- |
| `model` | Model parameter state dictionary |
| `model_kwargs` | Model architecture settings |

The evaluation loader restores the model with strict parameter matching.
