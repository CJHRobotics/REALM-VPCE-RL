# Shared setup for the Webots jobs. Source, do not execute.
#
# Resolves the Python that Webots will launch the controller with, makes it
# visible inside the Singularity image, and checks it actually works there
# before a world is loaded.
#
# The container supplies Webots, Xvfb and Mesa. It does NOT supply the Python
# packages -- those come from a conda environment on the host, built with
# Python 3.11 to match the image's own interpreter. This repo's environment is
# `realm-vpce-rl`; the place-field paper's `realm-vpce` is deliberately left
# alone, so an upgrade here cannot change the environment its results came
# from. An earlier version of
# these scripts built a second venv inside the container on the theory that
# wheels should link against container libraries; in practice that duplicated
# several GB (torch alone is ~3 GB), filled the home quota, and bought
# nothing that the preflight below does not verify directly.
#
# DEFINE EACH FUNCTION EXACTLY ONCE. This file carried three stacked copies of
# write_runtime_ini, gpu_args and gl_info, newest first, because two successive
# fixes were prepended instead of replacing the body they fixed. Bash keeps the
# LAST definition, so both fixes were dead code and the original -- the one
# they were written to correct -- was the one that actually ran, for weeks.

CONDA_ENV_NAME="${CONDA_ENV_NAME:-realm-vpce-rl}"

resolve_python() {
    if [[ -n "${REALM_PY:-}" ]]; then
        PYTHON_BIN="$REALM_PY"
        return
    fi
    local base
    base="$(conda info --base 2>/dev/null || echo /apps/anaconda3)"
    # shellcheck disable=SC1091
    source "$base/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
    PYTHON_BIN="${CONDA_PREFIX:?conda activate $CONDA_ENV_NAME failed}/bin/python3"
}

# Singularity auto-mounts $HOME, $PWD and /tmp but nothing else, so a conda
# environment living under, say, /apps is invisible inside the image unless
# bound. Only ever bind a NON-system top-level directory: binding /usr or
# /lib mounts the host's system tree over the container's own and destroys
# it -- the visible symptom is singularity failing to open /bin/sh.
SYSTEM_DIRS=" /usr /bin /sbin /lib /lib64 /etc /var /proc /sys /dev / "

container_binds() {
    BINDS=()
    local top
    top="/$(printf '%s' "${PYTHON_BIN#/}" | cut -d/ -f1)"
    case "$PYTHON_BIN" in
        "$HOME"/*|/tmp/*|"$REPO_DIR"/*)
            return 0 ;;                     # auto-mounted already
    esac
    if [[ " $SYSTEM_DIRS " == *" $top "* ]]; then
        echo "ERROR: $PYTHON_BIN lives under the system directory $top." >&2
        echo "       Binding that into the image would overwrite the" >&2
        echo "       container's own system tree. Point REALM_PY at a conda" >&2
        echo "       environment instead." >&2
        return 1
    fi
    BINDS=(--bind "$top")
    return 0
}

# Fail here, clearly, rather than inside Webots. A missing interpreter or an
# unimportable binding surfaces from Webots as a controller that exits
# immediately with no useful message.
preflight() {
    echo "--- preflight: $PYTHON_BIN inside the image ---"
    # GPU_ARGS, same as the run. Without them the container has no GPU at
    # all, so the preflight reported "cuda False" and "nvidia-smi
    # unavailable" while the job that followed printed "Using cuda device"
    # -- a flat contradiction that sent a real debugging session the wrong
    # way. Every caller runs gpu_args before preflight, so they are set.
    "$SINGULARITY" exec "${BINDS[@]+"${BINDS[@]}"}" "${GPU_ARGS[@]+"${GPU_ARGS[@]}"}" \
        "$SIF" "$PYTHON_BIN" - <<'PYCHK'
import sys
print('  python', sys.version.split()[0], 'at', sys.executable)
missing = []
# Collection needs the first group; RL training needs the second. A missing
# package surfaces from Webots as a controller that exits with no message, so
# check here where the failure can say what is wrong.
for m in ('numpy', 'pandas', 'h5py', 'cv2', 'matplotlib', 'PIL', 'tqdm',
          'torch', 'gymnasium', 'stable_baselines3'):
    try:
        __import__(m)
    except Exception as e:
        missing.append(f'{m} ({e.__class__.__name__})')
try:
    import torch
    print('  torch', torch.__version__, 'cuda', torch.cuda.is_available())
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f'  gpu   {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}')
    else:
        # "cuda False" on a node that clearly has a GPU is the single most
        # confusing state this setup produces, and the bare flag never says
        # why. The two causes look identical from the flag and need
        # completely different fixes, so name which one it is.
        built = getattr(torch.version, 'cuda', None)
        if built is None:
            print('  WHY NO CUDA: this is a CPU-only torch build. Install a '
                  'CUDA build.')
        else:
            import subprocess
            try:
                drv = subprocess.run(
                    ['nvidia-smi', '--query-gpu=driver_version,name',
                     '--format=csv,noheader'],
                    capture_output=True, text=True, timeout=10).stdout.strip()
            except Exception:
                drv = '(nvidia-smi unavailable)'
            print(f'  WHY NO CUDA: torch built for CUDA {built}; '
                  f'driver reports {drv}')
            major = int(str(built).split('.')[0])
            if major >= 13:
                print('    CUDA 13 needs a driver from the 580 series. A '
                      'CUDA 12.x build runs on any 525+ driver (minor '
                      'version compatibility), so install one of those.')
    # Which GPU architectures this wheel actually contains. Compatibility is
    # forward-only: a wheel can JIT onto a NEWER card than it was built for,
    # never onto an older one. On a cluster spanning sm_61 (1080 Ti) to
    # sm_90 (H100), a wheel missing the low end fails hard on those nodes
    # with "no kernel image is available", which reads like a driver fault.
    try:
        archs = torch.cuda.get_arch_list()
        print('  torch built for:', ' '.join(archs) or '(none - CPU only)')
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            # An exact sm_XY match is NOT required. CUDA guarantees binary
            # compatibility forward across minor revisions within a major,
            # so sm_60 code runs on an sm_61 card -- which is exactly the
            # 1080 Ti case here, and an exact-match check wrongly condemned
            # a wheel that works. PTX entries (compute_XY) JIT forward too.
            def _split(a):
                d = a.split('_')[1]
                return int(d[:-1]), int(d[-1])
            ok = any(a.startswith('sm_') and _split(a)[0] == cap[0]
                     and _split(a)[1] <= cap[1] for a in archs)
            ok = ok or any(a.startswith('compute_') and _split(a) <= cap
                           for a in archs)
            if not ok:
                print(f'  WARNING: this GPU is sm_{cap[0]}{cap[1]} and the '
                      f'wheel has no compatible code for it. Expect "no '
                      f'kernel image is available". Pin a supported node.')
    except Exception:
        pass

# is_available() only counts devices; it creates no context, so it can say
# True and the first real allocation still fail. That is what happened
# here: SB3 chose cuda, then .to(device) raised
# cudaErrorDevicesUnavailable and killed the run. Probe properly.
try:
    import torch
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device='cuda')
            print('  cuda allocation OK')
        except Exception as e:
            print(f'  WARNING: cuda reports available but allocation FAILED '
                  f'({e.__class__.__name__}: {e}).')
            print('    Usually the GPU is in exclusive-process compute mode '
                  'with a context already held, or another job is on the '
                  'same card. Training falls back to CPU rather than dying.')
except Exception:
    pass
except Exception:
    pass
try:
    import controller                      # Webots bindings, from PYTHONPATH
    print('  controller module OK')
except Exception as e:
    missing.append(f'controller ({e})')
if missing:
    print('  MISSING:', ', '.join(missing))
    sys.exit(1)
print('  all imports OK')
PYCHK
}

write_runtime_ini() {
    # Webots launches the controller itself and takes the interpreter from
    # here, so a stale path fails looking like a Webots fault. Rewritten every
    # run; the file is gitignored and machine specific.
    local d
    for d in simulation/controllers/*/; do
        printf '[python]\nCOMMAND = %s\n' "$PYTHON_BIN" > "$d/runtime.ini"
    done
    echo "runtime.ini -> $PYTHON_BIN"
}


# --- optional GPU passthrough --------------------------------------------
# `--nv` binds the host NVIDIA driver into the image. On its own that is not
# enough here for two reasons, and both are worth knowing before reading the
# result:
#
#   1. The image pins LIBGL_ALWAYS_SOFTWARE=1 and GALLIUM_DRIVER=llvmpipe at
#      build time, so GL stays software unless those are overridden.
#   2. Webots renders through GLX, and GLX acceleration comes from the X
#      server. Xvfb is a software X server with no NVIDIA GLX extension, so
#      the vendor dispatch is expected to land back on Mesa regardless.
#
# The honest expectation is therefore "no change, or a GL error". gl_info()
# prints the renderer actually in use, which settles it in one line rather
# than by inference from a timing.
gpu_args() {
    GPU_ARGS=()
    [[ "${USE_GPU:-0}" == "1" ]] || return 0
    GPU_ARGS=(--nv
              --env LIBGL_ALWAYS_SOFTWARE=0
              --env GALLIUM_DRIVER=
              --env __GLX_VENDOR_LIBRARY_NAME=nvidia)
    return 0
}

gl_info() {
    # Reports the renderer actually in use and, when hardware GL was asked for
    # but is not available, FALLS BACK to software rather than failing the job.
    #
    # This used to abort, on the reasoning that a silent fallback turns a
    # 12-minute arena into as much as ten hours. That reasoning is sound and
    # the guard was still wrong: it blocked collection entirely on a cluster
    # where software rendering demonstrably works, twice, for a GPU path that
    # is an optimisation rather than a requirement. A loud warning in the log
    # and in the mailed report carries the same information without costing a
    # day. Set STRICT_GPU=1 to abort instead.
    local out r
    echo "--- GL renderer actually in use ---"
    echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    if [[ "${USE_GPU:-0}" == "1" ]]; then
        "$SINGULARITY" exec "${GPU_ARGS[@]+"${GPU_ARGS[@]}"}" "$SIF" \
            nvidia-smi -L 2>&1 | head -5 || echo "  (no nvidia-smi in image)"
    fi

    # One call, not two, and `-a` rather than `-n`. Two calls on the same
    # display number is the bug this function kept regressing to: `xvfb-run -n`
    # fails outright when the number is still held, so the second call returned
    # nothing and the guard reported a broken GL stack on a machine whose GL
    # was fine. `-a` probes for a free number instead, which also survives a
    # stale /tmp/.X<n>-lock left behind by a killed job. Webots itself keeps
    # its deterministic -n display, assigned in the job script.
    out="$("$SINGULARITY" exec "${BINDS[@]+"${BINDS[@]}"}" "${GPU_ARGS[@]+"${GPU_ARGS[@]}"}" \
        "$SIF" xvfb-run -a \
        -s "-screen 0 ${XVFB_SCREEN:-1280x1024x24}" glxinfo -B 2>&1 || true)"
    echo "$out" | grep -Ei 'vendor|renderer|version|error' | head -8 \
        || echo "  (no glxinfo output)"

    [[ "${USE_GPU:-0}" == "1" ]] || return 0

    r="$(echo "$out" | grep -i 'OpenGL renderer' || true)"
    if [[ -n "$r" && "$r" != *llvmpipe* && "$r" != *softpipe* && "$r" != *swrast* ]]; then
        echo "  hardware GL confirmed:$r"
        return 0
    fi

    echo "WARNING: hardware GL unavailable (${r:-glxinfo returned no renderer})." >&2
    echo "         Falling back to software rendering: expect roughly 2.5-10 h" >&2
    echo "         per arena instead of 0.2 h, depending on the node." >&2
    if [[ "${STRICT_GPU:-0}" == "1" ]]; then
        echo "         STRICT_GPU=1 set -- aborting instead." >&2
        return 1
    fi
    GPU_ARGS=()
    USE_GPU=0
    # The small screen is worth ~2.4x under llvmpipe on the slow nodes and
    # nothing on fast ones, so it is only applied on this path.
    XVFB_SCREEN="320x240x24"
    echo "         XVFB_SCREEN set to $XVFB_SCREEN for the software path." >&2
    return 0
}
