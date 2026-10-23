"""GPU visibility and training validation cadence.

Configure visibility before importing PyTorch. This module uses only the
standard library.
"""

import os
import sys

__all__ = ['preset_cuda_visible_devices_from_argv', 'validation_intervals', 'should_validate']


def _gpu_ids_from_argv():
    """Read --gpu_ids from the command line, or return None."""
    argv = sys.argv[1:]
    for idx, arg in enumerate(argv):
        if arg == '--gpu_ids' and idx + 1 < len(argv):
            return argv[idx + 1]
        if arg.startswith('--gpu_ids='):
            return arg.split('=', 1)[1]
    return None


def preset_cuda_visible_devices_from_argv(default=None, set_unconditionally=False):
    """Set GPU visibility before importing PyTorch.

    An existing CUDA_VISIBLE_DEVICES value is preserved.

    Args:
        default: GPU selection when --gpu_ids is absent; None leaves it unset.
        set_unconditionally: Whether to assign an empty GPU selection.
    """
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        return
    gpu_ids = _gpu_ids_from_argv()
    if gpu_ids is None:
        gpu_ids = default
    if gpu_ids is None:
        return
    if set_unconditionally or gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids


def validation_intervals(requested_frequency):
    """Use the requested interval initially, then cap it at 10 and 5 epochs."""
    early = max(int(requested_frequency), 1)
    return early, min(early, 10), min(early, 5)


def should_validate(current_epoch, total_epochs, frequency):
    """Check the validation interval, including the final training epoch."""
    if frequency < 1:
        raise ValueError('Validation frequency must be positive')
    return current_epoch % frequency == 0 or current_epoch == total_epochs
