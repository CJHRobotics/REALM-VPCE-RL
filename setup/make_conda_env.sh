#!/bin/bash
#
# Build this repo's conda environment on GAIVI, reproducibly.
#
# There was no script for this. The environment was made by hand, the
# place-field repo's was made by hand on an earlier day, both installed
# plain `torch`, and the two got different wheels -- which is why one of
# them could use the cluster's 1080 Tis and the other could not. Nobody
# chose that difference and nobody could see it: `pip list` shows a version,
# not the day it was resolved. The fix for that class of problem is a file
# that builds the environment the same way every time.
#
# ORDER MATTERS HERE. torch is installed first, pinned, from PyTorch's own
# cu121 index. PyPI's `torch==2.5.1` is a different build with a different
# CUDA runtime, and on this cluster's 530.30.02 driver that build cannot
# allocate on a 1080 Ti at all -- it raises cudaErrorDevicesUnavailable on
# an idle card. With torch already satisfied, pip resolves the rest around
# it and backtracks stable-baselines3 to a release that accepts it (2.9.2
# requires torch>=2.8, so it will not be that one).
#
# torchvision is deliberately absent. e560b2d dropped the CNN feature path
# and torchvision with it; it stayed installed in the old environment and
# was one of the two packages blocking the downgrade.
#
# Usage:
#   bash setup/make_conda_env.sh                    # realm-vpce-rl
#   bash setup/make_conda_env.sh realm-vpce-rl2     # side by side, to test
#   FORCE=1 bash setup/make_conda_env.sh            # replace an existing one
#
# Building under a second name costs nothing and keeps the old environment
# until the new one is proven: every job script takes CONDA_ENV_NAME, and
# REALM_PY overrides it outright.
#
#   CONDA_ENV_NAME=realm-vpce-rl2 sbatch slurm/fit_place_fields.sh <arena>
#
set -euo pipefail

ENV_NAME="${1:-realm-vpce-rl}"
PY_VERSION="${PY_VERSION:-3.11}"        # matches the container's interpreter
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

BASE="$(conda info --base 2>/dev/null || echo /apps/anaconda3)"
# shellcheck disable=SC1091
source "$BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    if [[ "${FORCE:-0}" != "1" ]]; then
        echo "ERROR: conda environment '$ENV_NAME' already exists." >&2
        echo "       FORCE=1 to remove and rebuild it, or pass another name:" >&2
        echo "         bash setup/make_conda_env.sh ${ENV_NAME}2" >&2
        exit 1
    fi
    echo "removing existing '$ENV_NAME'"
    conda env remove -y -n "$ENV_NAME"
fi

echo "===================================================================="
echo "env     : $ENV_NAME"
echo "python  : $PY_VERSION   (the container ships 3.11; the Webots bindings"
echo "          are built against it, so this is not a free choice)"
echo "torch   : $TORCH_VERSION from $TORCH_INDEX"
echo "===================================================================="

conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
conda activate "$ENV_NAME"

python -m pip install --upgrade pip

# First, and from the right index. Everything else resolves around it.
python -m pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"

python -m pip install -r setup/requirements.txt

echo
echo "--- what came out ---"
python - <<'PY'
import importlib.metadata as md
import torch
print(f'  torch              {torch.__version__}  (CUDA {torch.version.cuda})')
print(f'  arch list          {torch.cuda.get_arch_list()}')
for pkg in ('stable-baselines3', 'gymnasium', 'numpy', 'opencv-python',
            'h5py', 'matplotlib'):
    try:
        print(f'  {pkg:<18} {md.version(pkg)}')
    except md.PackageNotFoundError:
        print(f'  {pkg:<18} MISSING')
try:
    print(f'  torchvision        {md.version("torchvision")}'
          '   <-- should not be here; e560b2d dropped it')
except md.PackageNotFoundError:
    pass
PY

cat <<EOF

Built. It is NOT verified: this is a login node with no GPU, and the whole
point of the pin is what happens on a card. Run that next, on a node with
one:

  srun --gres=gpu:1 --time=00:05:00 bash slurm/gpu_check.sh

Look for "allocation: OK" in the torch section. Until it says that, nothing
here has been tested against the hardware it was pinned for.

If this was built under a second name, point a job at it with:

  CONDA_ENV_NAME=$ENV_NAME sbatch slurm/fit_place_fields.sh <arena>
EOF
