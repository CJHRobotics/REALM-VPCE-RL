#!/bin/bash
#
# Why will this node's GPU not give us a CUDA context?
#
# Job 635251 was refused one on a card that nvidia-smi reported as idle:
# compute mode Default, 2 MiB used, no processes. A CUDA context costs
# 250-300 MiB, so 2 MiB means there was no context on that card at all --
# nothing was holding it, and the usual explanations (another job, a
# leftover Webots, exclusive-process mode) are all ruled out by that one
# number. An idle card that refuses a context is a different problem, and
# torch's AcceleratorError is too far from the driver to say which.
#
# So ask the driver directly. cuInit() returns a CUresult, and its value
# distinguishes causes that torch reports identically:
#
#     0    success              -- the driver is fine, look higher up
#     100  NO_DEVICE            -- nothing visible: CUDA_VISIBLE_DEVICES, cgroup
#     101  INVALID_DEVICE       -- visible but not usable as asked
#     201  INVALID_CONTEXT
#     802  SYSTEM_NOT_READY     -- nvidia-uvm not loaded, or the GPU is resetting
#     803  SYSTEM_DRIVER_MISMATCH   -- kernel module and libcuda disagree
#     804  COMPAT_NOT_SUPPORTED_ON_DEVICE -- forward-compat libs on a card
#                                           that cannot take them
#     999  UNKNOWN
#
# 803 and 804 are the ones to expect from a conda environment whose libcuda
# is not the node's: a stub libcuda.so from a toolkit install, or a
# cuda-compat package ahead of the driver on LD_LIBRARY_PATH, both answer
# every call with an error while looking perfectly installed. That is why
# this prints which libcuda is actually loaded, rather than assuming.
#
# Usage, on the node you want to ask about:
#   srun --nodelist=GPU3 --gres=gpu:1 --time=00:05:00 bash slurm/gpu_check.sh
#   sbatch --gres=gpu:1 slurm/gpu_check.sh        # whatever the queue gives
#
#SBATCH --job-name=gpucheck
#SBATCH --partition=general
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR" || exit 1

PYTHON_BIN="${REALM_PY:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    base="$(conda info --base 2>/dev/null || echo /apps/anaconda3)"
    # shellcheck disable=SC1091
    source "$base/etc/profile.d/conda.sh" 2>/dev/null
    conda activate "${CONDA_ENV_NAME:-realm-vpce-rl}" 2>/dev/null
    PYTHON_BIN="${CONDA_PREFIX:-}/bin/python3"
fi

echo "===================================================================="
echo "node    : $(hostname)"
echo "job     : ${SLURM_JOB_ID:-interactive}"
echo "python  : $PYTHON_BIN"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "===================================================================="

echo
echo "--- device nodes (nvidia-uvm must be there for any context) ---"
ls -l /dev/nvidia* 2>&1 | head -20

echo
echo "--- what the driver says about the card ---"
nvidia-smi --query-gpu=index,name,driver_version,compute_mode,persistence_mode,memory.used,ecc.errors.uncorrected.volatile.total \
           --format=csv 2>&1 | head -10
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 | head -10

echo
echo "--- which libcuda is on the path, and is it the real one ---"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"
ldconfig -p 2>/dev/null | grep -i 'libcuda\.so' | head -5
find "${CONDA_PREFIX:-/nonexistent}" -name 'libcuda.so*' 2>/dev/null | head -5 \
    || true
echo "(a libcuda.so under the conda env, or anything named .../stubs/, is a"
echo " stub: it links and then fails every call. The real one belongs to the"
echo " driver, in /usr/lib64 or /usr/lib/x86_64-linux-gnu.)"

echo
echo "--- the driver's own answer, before torch gets involved ---"
"$PYTHON_BIN" - <<'PY'
import ctypes, os, sys

NAMES = {0: 'SUCCESS', 100: 'NO_DEVICE', 101: 'INVALID_DEVICE',
         201: 'INVALID_CONTEXT', 802: 'SYSTEM_NOT_READY',
         803: 'SYSTEM_DRIVER_MISMATCH',
         804: 'COMPAT_NOT_SUPPORTED_ON_DEVICE', 999: 'UNKNOWN'}
try:
    lib = ctypes.CDLL('libcuda.so.1')
except OSError as exc:
    print(f'  libcuda.so.1 will not load: {exc}')
    sys.exit(0)

path = ctypes.c_char_p()
try:                                    # where did we actually load it from
    import ctypes.util
    print(f'  loaded libcuda: {ctypes.util.find_library("cuda")}')
except Exception:
    pass

rc = lib.cuInit(0)
print(f'  cuInit(0) -> {rc}  ({NAMES.get(rc, "see cuda.h")})')

n = ctypes.c_int()
rc2 = lib.cuDeviceGetCount(ctypes.byref(n))
print(f'  cuDeviceGetCount -> {rc2} ({NAMES.get(rc2, "?")}), devices={n.value}')

if rc == 0 and n.value:
    dev = ctypes.c_int()
    print(f'  cuDeviceGet(0) -> {lib.cuDeviceGet(ctypes.byref(dev), 0)}')
    ctx = ctypes.c_void_p()
    rc3 = lib.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
    print(f'  cuCtxCreate -> {rc3} ({NAMES.get(rc3, "see cuda.h")})'
          + ('   <-- a context CAN be created; the problem is above the driver'
             if rc3 == 0 else
             '   <-- the DRIVER refuses a context; torch is not the problem'))
    if rc3 == 0:
        lib.cuCtxDestroy_v2(ctx)
PY

echo
echo "--- and what torch makes of the same card ---"
"$PYTHON_BIN" - <<'PY'
import traceback
import torch
print(f'  torch {torch.__version__}, built for CUDA {torch.version.cuda}')
print(f'  arch list: {torch.cuda.get_arch_list()}')
print(f'  is_available: {torch.cuda.is_available()}  '
      f'device_count: {torch.cuda.device_count()}')
try:
    cap = torch.cuda.get_device_capability(0)
    print(f'  capability: sm_{cap[0]}{cap[1]}'
          + ('' if f'sm_{cap[0]}{cap[1]}' in torch.cuda.get_arch_list()
             else '   <-- not listed, which is NOT by itself the problem:'
                  ' cubins load upward'))
    if f'sm_{cap[0]}{cap[1]}' not in torch.cuda.get_arch_list():
        print('      (2.5.1+cu121 and 2.14.1+cu126 have the SAME list and '
              'only one works here;')
        print('       compare torch.version.cuda against the driver before '
              'blaming the arch)')
except Exception as exc:
    print(f'  capability query failed: {exc.__class__.__name__}: {exc}')
try:
    torch.zeros(1, device='cuda')
    print('  allocation: OK -- the card works for torch right now')
except Exception:
    print('  allocation FAILED:')
    traceback.print_exc()
PY
echo
echo "done."
