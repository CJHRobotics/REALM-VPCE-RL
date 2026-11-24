#!/bin/bash
#
# Train a navigation agent on the VPCE representation, in Webots, on GAIVI.
#
# Proof of concept: does an agent converge when all it can see is the place
# cells? Collect, then fit the fields, then this.
#
#   sbatch slurm/collect_data.sh    circ_lm8_r10
#   sbatch slurm/fit_place_fields.sh circ_lm8_r10
#   sbatch slurm/train_rl.sh         circ_lm8_r10
#
# Usage:
#   sbatch slurm/train_rl.sh                        # circ_lm8_r10, DQN
#   sbatch slurm/train_rl.sh circ_lm8_r10
#   REALM_TIMESTEPS=200000 sbatch slurm/train_rl.sh circ_lm8_r10
#   REALM_ALGO=PPO REALM_SEED=1 sbatch slurm/train_rl.sh circ_lm8_r10
#   REALM_MAX_EPISODES=3000 sbatch slurm/train_rl.sh circ_lm8_r10
#   bash slurm/train_rl.sh --submit circ_lm8_r10   # DQN and PPO x 3 seeds
#
# Training stops early on convergence (REALM_STOP_AT, default 0.8 success
# over the last 100 episodes) or gives up after REALM_MAX_EPISODES. Either
# way it then runs one deterministic episode from each of the arena's start
# positions and draws final_paths.png -- the routes the policy actually
# takes. A failure has a shape, so that figure is drawn whether the run
# converged or not.
#
# Everything is configured through the environment; the controller reads it
# and echoes it. See simulation/controllers/rl_agent/rl_agent.py.
#
# THE SIMULATOR IS IN THE LOOP. Every environment step captures eight camera
# frames at the robot's position, extracts the descriptors and encodes them
# through the ensemble. Nothing is read from the collected grid -- that
# dataset fitted the fields and has no part in training. This is the
# expensive way round and it is the intended one: the agent is being tested
# on what it can see, not on a lookup table.
#
# ------------------------------------------------------------- SLURM header
#SBATCH --job-name=trainrl
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=GPU1,GPU2,GPU45,GPU46,GPU47,GPU48,GPU49,GPU53
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chamilton4@usf.edu
#
# The exclusions are BACK, unlike fit_place_fields.sh. That job did matrix
# arithmetic and wanted the compute-only cards; this one renders through
# GLX, and a card with no display engine cannot provide it -- an A100, H100
# or L40S drops to llvmpipe at 15-158 ms per render against 2.1 ms. At eight
# renders per environment step that is the difference between a run and a
# timeout. gl_info aborts rather than letting it happen silently.
#
# Time. Per step: 8 renders, 8 HOG extractions, a teleport and the sensor
# steps around it. Call it 50-100 ms with hardware GL, so 100k steps is
# roughly 1.5-3 h. 24 h leaves room for a longer budget or a slower node. A
# run killed by the walltime still leaves everything: the Monitor CSV is
# flushed per episode, so the learning curve up to the kill survives, and
# that is the measurement.
# --------------------------------------------------------------------------

set -euo pipefail

# --------------------------------------------------------------- submit mode
# Run with bash on the login node, not sbatch: it only calls sbatch, once per
# (algorithm, seed). One job per combination rather than a loop inside one
# job, because they are independent and queue in parallel -- and because a
# single job that dies takes every condition with it.
#
#   bash slurm/train_rl.sh --submit                       # DQN and PPO x 3 seeds
#   bash slurm/train_rl.sh --submit circ_lm8_r10
#   ALGOS=DQN SEEDS=0,1,2,3,4 bash slurm/train_rl.sh --submit circ_lm8_r10
#
# Seeds are not optional for a comparison. Learning time varies run to run
# by more than many of the effects being measured, so a single seed per
# algorithm cannot distinguish "PPO is slower" from "that PPO run was
# unlucky". Three is a floor, five is better.
if [[ "${1:-}" == "--submit" ]]; then
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
    mkdir -p slurm/logs
    SUB_ARENA="${2:-circ_lm8_r10}"
    ALGO_LIST="${ALGOS:-DQN,PPO}"; ALGO_LIST="${ALGO_LIST//,/ }"
    SEED_LIST="${SEEDS:-0,1,2}";   SEED_LIST="${SEED_LIST//,/ }"
    n=0
    for algo in $ALGO_LIST; do
        # tr rather than ${algo,,}: lowercase expansion is bash 4, and macOS
        # ships 3.2, so the dry run on a laptop would silently differ.
        lower="$(echo "$algo" | tr 'A-Z' 'a-z')"
        for seed in $SEED_LIST; do
            tag="${SUB_ARENA}_${lower}_s${seed}"
            sbatch --job-name="trainrl-${lower}-s${seed}" \
                --export=ALL,REALM_ALGO="$algo",REALM_SEED="$seed",REALM_RUN_TAG="$tag" \
                slurm/train_rl.sh "$SUB_ARENA"
            n=$((n + 1))
        done
    done
    echo
    echo "$n job(s) submitted for $SUB_ARENA: [$ALGO_LIST] x [$SEED_LIST]"
    echo "When they finish, compare them with:"
    echo "  python3 analysis/compare_runs.py --by algo"
    exit 0
fi

ARENA="${1:-circ_lm8_r10}"
shift || true
WEBOTS_EXTRA=("$@")

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"
mkdir -p slurm/logs data/vpce/rl_runs

SIF="${WEBOTS_SIF:-$REPO_DIR/webots_r2025a.sif}"
SINGULARITY=/apps/singularity/bin/singularity

# shellcheck disable=SC1091
source "$REPO_DIR/slurm/_webots_env.sh"
resolve_python
container_binds || exit 1

export REALM_MAZE="$ARENA"
ENSEMBLE="${REALM_ENSEMBLE:-data/vpce/place_fields/${ARENA}_visual.h5}"
export REALM_ENSEMBLE="$ENSEMBLE"
TAG="${REALM_RUN_TAG:-${ARENA}_$(echo "${REALM_ALGO:-DQN}" | tr 'A-Z' 'a-z')_s${REALM_SEED:-0}}"
export REALM_RUN_TAG="$TAG"
OUT_DIR="data/vpce/rl_runs/$TAG"

# Same rule as the other jobs: the figures are how you find out what
# happened, so refuse to start with nowhere to send them.
if [[ -z "${EMAIL_TO:-}" && "${ALLOW_NO_EMAIL:-0}" != "1" ]]; then
    {
        echo "ERROR: EMAIL_TO is not set, so the learning curve would be written"
        echo "       to $OUT_DIR/ and never sent."
        echo
        echo "  For this job:  EMAIL_TO=you@usf.edu sbatch slurm/train_rl.sh $ARENA"
        echo "  Files only:    ALLOW_NO_EMAIL=1 sbatch slurm/train_rl.sh $ARENA"
    } >&2
    exit 1
fi

WORLD="$REPO_DIR/simulation/worlds/rl_agent.wbt"
[[ -f "$SIF"   ]] || { echo "ERROR: image not found: $SIF" >&2; exit 1; }
[[ -f "$WORLD" ]] || { echo "ERROR: world not found: $WORLD" >&2; exit 1; }
if [[ ! -f "$ENSEMBLE" ]]; then
    {
        echo "ERROR: no place-field ensemble at $ENSEMBLE"
        echo "       Build it first:  sbatch slurm/fit_place_fields.sh $ARENA"
    } >&2
    exit 1
fi

XVFB_SCREEN="${XVFB_SCREEN:-1280x1024x24}"
# Derived from the job id rather than probed: `xvfb-run -a` races when
# several jobs start on one node, and two jobs on the same display fail with
# a GLX BadValue that reads like a driver fault.
WEBOTS_PORT=$(( 10000 + (${SLURM_JOB_ID:-0} % 20000) ))
XVFB_DISPLAY=$(( 100 + (${SLURM_JOB_ID:-0} % 400) ))

USE_GPU="${USE_GPU:-1}"
# STRICT_GPU defaults ON here, and only here. gl_info's own default is to warn
# and fall back to software rendering, which is right for collection -- it
# finishes, slowly -- and wrong for this job: a step is 9 renders, so llvmpipe
# turns a 42-minute run into roughly ten days, and the job spends its whole
# walltime producing a partial curve. Worse, the fallback is INTERMITTENT. The
# probe runs under `xvfb-run -a` and races, so the same node returned no
# renderer on jobs 632261 and 632489 and the 1080 Ti on 632586. A coin flip
# between 42 minutes and ten days is not something to discover from a timing.
# Aborting costs twenty seconds and a resubmit. Set STRICT_GPU=0 to accept the
# software path deliberately.
export STRICT_GPU="${STRICT_GPU:-1}"
gpu_args
gl_info || exit 1

echo "===================================================================="
echo "Job     : ${SLURM_JOB_ID:-local}   node $(hostname)"
echo "Arena   : $ARENA"
echo "Ensemble: $ENSEMBLE"
echo "Run tag : $TAG   ->  $OUT_DIR"
echo "Algo    : ${REALM_ALGO:-DQN}   steps=${REALM_TIMESTEPS:-100000}  seed=${REALM_SEED:-0}"
echo "Stop at : ${REALM_STOP_AT:-0.8} success   abandon below ${REALM_GIVE_UP_BELOW:-0.10} after ${REALM_GIVE_UP_AFTER:-0.5} of the budget"
echo "Teleport: ${REALM_TELEPORT:-1}   shaping=${REALM_SHAPING:-0}"
echo "Image   : $SIF"
echo "Python  : $PYTHON_BIN"
echo "Binds   : ${BINDS[*]+${BINDS[*]}}${BINDS[0]:-(none needed - auto-mounted)}"
echo "Display : :$XVFB_DISPLAY  $XVFB_SCREEN   webots port $WEBOTS_PORT"
echo "Started : $(date -Is)"
echo "Git     : $(git rev-parse --short HEAD 2>/dev/null || echo 'no git')"
echo "===================================================================="

preflight || { echo 'PREFLIGHT FAILED - not launching Webots' >&2; exit 1; }
write_runtime_ini

export LP_NUM_THREADS="${LP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export PYTHONUNBUFFERED=1

T_MARK="$(mktemp "${TMPDIR:-/tmp}/trainrl.XXXXXX")"
trap 'rm -f "$T_MARK"' EXIT
T_START=$(date +%s)

# No `timeout` wrapper here, unlike render_check. That job had a known short
# runtime and a hang was unambiguous; this one is supposed to run for hours,
# and the SLURM walltime is the right backstop. The controller calls
# simulationQuit when it finishes.
set +e
"$SINGULARITY" exec "${BINDS[@]+"${BINDS[@]}"}" "${GPU_ARGS[@]+"${GPU_ARGS[@]}"}" \
    --env REALM_MAZE="$ARENA" \
    --env REALM_ARENA_DIR="${REALM_ARENA_DIR:-vpce}" \
    --env REALM_ENSEMBLE="$ENSEMBLE" \
    --env REALM_RUN_TAG="$TAG" \
    --env REALM_TIMESTEPS="${REALM_TIMESTEPS:-100000}" \
    --env REALM_MAX_STEPS="${REALM_MAX_STEPS:-200}" \
    --env REALM_ALGO="${REALM_ALGO:-DQN}" \
    --env REALM_TELEPORT="${REALM_TELEPORT:-1}" \
    --env REALM_SHAPING="${REALM_SHAPING:-0}" \
    --env REALM_SEED="${REALM_SEED:-0}" \
    --env REALM_EVAL_ONLY="${REALM_EVAL_ONLY:-0}" \
    --env REALM_STOP_AT="${REALM_STOP_AT:-0.8}" \
    --env REALM_MAX_EPISODES="${REALM_MAX_EPISODES:-0}" \
    --env REALM_GIVE_UP_BELOW="${REALM_GIVE_UP_BELOW:-0.10}" \
    --env REALM_GIVE_UP_AFTER="${REALM_GIVE_UP_AFTER:-0.5}" \
    --env REALM_DEVICE="${REALM_DEVICE:-cpu}" \
    --env LP_NUM_THREADS="$LP_NUM_THREADS" \
    "$SIF" \
    xvfb-run -n "$XVFB_DISPLAY" -s "-screen 0 $XVFB_SCREEN" \
    webots --batch --stdout --stderr --mode=fast --minimize --port="$WEBOTS_PORT" \
        "${WEBOTS_EXTRA[@]+"${WEBOTS_EXTRA[@]}"}" "$WORLD"
STATUS=$?
set -e

ELAPSED=$(( $(date +%s) - T_START ))
echo
echo "Finished : $(date -Is)  (exit ${STATUS}, ${ELAPSED}s)"

echo "--- produced ---"
ls -lh "$OUT_DIR/" 2>/dev/null || echo "(nothing in $OUT_DIR)"

# If the walltime or a crash took the run before it could plot, the Monitor
# CSV is still there and still holds everything up to that point -- so build
# the curve here rather than lose the only record of what happened.
if [[ -f "$OUT_DIR/monitor.csv" && ! -f "$OUT_DIR/learning_curve.png" ]]; then
    echo "no curve from the run; building one from the partial monitor.csv"
    "$PYTHON_BIN" -c "
import sys; sys.path.insert(0, '.')
from realm_tools.rl_lib import learning_curve
p, s = learning_curve.plot('$OUT_DIR/monitor.csv',
                           '$OUT_DIR/learning_curve.png',
                           title='$TAG (partial)')
print('\n'.join(learning_curve.summarise(s)))
" || echo "  (could not build a curve from the partial log)"
fi

ATTACH=()
while IFS= read -r f; do
    [[ -n "$f" ]] && ATTACH+=("$f")
done < <(
    find "$OUT_DIR" -name '*.png' 2>/dev/null
    find "$OUT_DIR" -name 'result.json' -o -name 'config.json' 2>/dev/null
)

"$PYTHON_BIN" - "$ARENA" "$TAG" "$STATUS" "$ELAPSED" "$OUT_DIR" \
    "${ATTACH[@]+"${ATTACH[@]}"}" <<'PYMAIL'
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from realm_tools.experiment_lib.reporting import send_email

arena, tag, status, elapsed, out_dir = sys.argv[1:6]
attachments = sys.argv[6:]
status, elapsed = int(status), int(elapsed)
ok = status == 0

lines = [f'RL training | arena = {arena} | run = {tag}',
         f'  exit status : {status} ({"ok" if ok else "FAILED"})',
         f'  elapsed     : {elapsed // 3600}h {(elapsed % 3600) // 60}m',
         f'  job         : {os.environ.get("SLURM_JOB_ID", "local")} on '
         f'{os.environ.get("SLURMD_NODENAME", "?")}',
         '']

for name in ('config.json', 'result.json'):
    path = os.path.join(out_dir, name)
    if not os.path.exists(path):
        continue
    try:
        with open(path) as f:
            d = json.load(f)
        lines.append(name)
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f'  {k}:')
                lines += [f'    {kk}: {vv}' for kk, vv in v.items()]
            else:
                lines.append(f'  {k}: {v}')
        lines.append('')
    except Exception as exc:                                # noqa: BLE001
        lines.append(f'({name} unreadable: {exc})')

# The verdict, stated rather than left to be inferred from the attachment.
try:
    with open(os.path.join(out_dir, 'result.json')) as f:
        r = json.load(f)
    before, after = r.get('success_before'), r.get('success_after')
    if before is not None and after is not None:
        lines.append(f'VERDICT: success {before:.0%} -> {after:.0%}')
        lines.append('  the agent learned' if after > before + 0.15
                     else '  NO CLEAR LEARNING -- check the curve and whether '
                          'the goal was ever reached during exploration')
except Exception:                                           # noqa: BLE001
    lines.append('VERDICT: no result.json -- the run did not finish')

lines.append('')
figs = [a for a in attachments if a.endswith('.png')]
lines.append(f'{len(figs)} figure(s) attached' if figs
             else 'NO FIGURES -- check the log')

body = '\n'.join(lines)
print(body)
sent = send_email(f'[REALM-VPCE-RL] training — {tag}' + ('' if ok else ' [FAILED]'),
                  body, attachments=attachments)

wanted = bool(os.environ.get('EMAIL_TO'))
if wanted and not sent:
    print('ERROR: the report was not delivered -- see the [report] line above.')
    sys.exit(1)
PYMAIL

exit ${STATUS}
