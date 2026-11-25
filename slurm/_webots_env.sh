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
# write_runtime_ini and gpu_args, newest first, because two successive
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
    # GPU_ARGS so "cuda" below reports what the RUN sees; without
    # them the container has no GPU and the line is misleading.
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
# That expectation turned out to be wrong -- --nv plus those overrides is
# 75x faster on these cards, measured. See container/README.md. Webots
# reports the renderer it ended up with in the job log, which is the only
# place the question has ever been answered reliably.
# --- which renderer the job asks for -------------------------------------
#
#   REALM_RENDER=software   (default) Mesa llvmpipe inside the image. No GPU
#                           is bound. This is what every dataset in this repo
#                           was collected with, and it starts every time.
#
#   REALM_RENDER=vgl        VirtualGL with the EGL back end, on the GPU.
#                           Binds --nv, overrides the image's software-GL
#                           pins, and wraps Webots in `vglrun -d egl0`.
#
# Changing this changes the IMAGES, and therefore the descriptors and the
# place fields built from them. A dataset collected under one renderer is
# not comparable with one collected under the other -- the same rule
# container/README.md states for moving collection between machines. Switch
# deliberately, between experiments, and re-collect.
#
# Sets USE_GPU, calls gpu_args, and leaves the wrapper in VGL for the job
# script to put in front of webots.
render_mode() {
    VGL=()
    case "${REALM_RENDER:-software}" in
        software)
            USE_GPU=0
            gpu_args
            ;;
        vgl)
            USE_GPU=1
            gpu_args
            VGL=(vglrun -d "${VGL_DEVICE:-egl0}")
            # GLVND finds its EGL vendors through JSON files in /usr/share,
            # which --nv does not bind -- it handles libraries, not this.
            # Without the host's 10_nvidia.json, libEGL enumerates no NVIDIA
            # device and vglrun falls back or fails, which would look like a
            # VirtualGL problem and would not be one.
            if [[ -d /usr/share/glvnd/egl_vendor.d ]]; then
                GPU_ARGS+=(--bind /usr/share/glvnd/egl_vendor.d)
            else
                echo "WARNING: no /usr/share/glvnd/egl_vendor.d on this host;" >&2
                echo "         EGL may not find the NVIDIA driver inside the image." >&2
            fi
            ;;
        *)
            echo "ERROR: REALM_RENDER must be 'software' or 'vgl', not " \
                 "'${REALM_RENDER}'" >&2
            return 1
            ;;
    esac
    return 0
}


gpu_args() {
    GPU_ARGS=()
    [[ "${USE_GPU:-0}" == "1" ]] || return 0
    GPU_ARGS=(--nv
              --env LIBGL_ALWAYS_SOFTWARE=0
              --env GALLIUM_DRIVER=
              --env __GLX_VENDOR_LIBRARY_NAME=nvidia)
    return 0
}

# gl_info() WAS HERE, AND IS GONE.
#
# It ran glxinfo under its own Xvfb before Webots started, to report which
# renderer was in use and, under STRICT_GPU, to refuse the job if that
# renderer was software. It cost four jobs and most of two days, and every
# one of those failures was the probe, not the GL stack.
#
# The mechanism is worse than a false alarm. When the probe came back
# without a renderer line -- a lost race for a display number, which is what
# `xvfb-run -a` makes likely when two jobs start on one node -- gl_info
# CLEARED GPU_ARGS and set XVFB_SCREEN to 320x240 before returning. So a
# failed probe did not report software rendering, it CAUSED it: Webots was
# then launched with no --nv at all, having been told the hardware was
# unavailable by a test that never asked Webots anything.
#
# Deriving the display from the job id and retrying fixed neither the
# BadValue nor the aborts. At that point the probe had cost more than it
# could ever save, and the collection jobs -- which use the GPU perfectly
# well -- had been doing so while this function was predicting they could
# not.
#
# What replaces it is nothing. gpu_args() still binds --nv and overrides the
# image's software-GL pins, Webots reports its own renderer in the job log
# where it can be read after the fact, and the jobs that render are left to
# find out from the thing that actually does the rendering. A pre-flight
# guess about what Webots will do is not worth the authority to stop a job
# from running.
