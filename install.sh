#!/usr/bin/env bash
# Install the main environment and build its CUDA extensions.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-python}"
export MAX_JOBS="${MAX_JOBS:-4}"

# Read bootstrap versions from the same dependency file used for the full install.
"$PYTHON" - <<'PY'
from pathlib import Path
import re
import subprocess
import sys

requirements = {}
for line in Path('requirements.txt').read_text(encoding='utf-8').splitlines():
    line = line.strip()
    if line and not line.startswith('#'):
        name = re.match(r'[A-Za-z0-9][A-Za-z0-9_.-]*', line).group(0)
        requirements[name.lower().replace('_', '-')] = line

for names, options in (
    (('setuptools', 'wheel', 'packaging', 'ninja', 'numpy'), []),
    (('torch', 'torchvision', 'torchaudio'),
     ['--index-url', 'https://download.pytorch.org/whl/cu121']),
):
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', *options,
        *(requirements[name] for name in names),
    ])
PY

"$PYTHON" - <<'PY'
import os
import re
import subprocess
import torch
from torch.utils.cpp_extension import CUDA_HOME
if CUDA_HOME is None:
    raise SystemExit('Set CUDA_HOME to the CUDA 12.x toolkit directory before running install.sh.')
nvcc = os.path.join(CUDA_HOME, 'bin', 'nvcc')
version = subprocess.check_output([nvcc, '--version'], text=True)
match = re.search(r'release (\d+)\.(\d+)', version)
if match is None or torch.version.cuda is None:
    raise SystemExit('Use a CUDA-enabled PyTorch installation and CUDA nvcc.')
if int(match.group(1)) != int(torch.version.cuda.split('.')[0]):
    raise SystemExit(f'Select a CUDA {torch.version.cuda.split(".")[0]}.x toolkit: '
                     f'PyTorch uses {torch.version.cuda}, CUDA_HOME={CUDA_HOME}.')
print(f'PyTorch: {torch.__version__}; CUDA toolkit: {match.group(0)}; CUDA_HOME: {CUDA_HOME}', flush=True)
PY

# CUDA package builds use the installed PyTorch, NumPy and build tools.
CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE \
    "$PYTHON" -m pip install --no-build-isolation -r requirements.txt

for extension in extensions/chamfer_dist extensions/Pointnet2/pointnet2 models/pointops; do
    echo "Building $extension"
    "$PYTHON" -m pip install --no-build-isolation --no-deps "./$extension"
done
"$PYTHON" -m pip check
"$PYTHON" - <<'PY'
import torch
import chamfer
import pointnet2_cuda
import pointops_cuda
from models.controlflow3d import ControlFlow3D
print('ControlFlow3D and CUDA extensions import successfully.')
PY
