#!/bin/bash
#
# Walk a trained policy to the goal and record, at every state, the smallest
# set of place cells whose removal changes the action it takes.
#
# Online, in Webots, through the same camera and encoder the policy trained
# on. Nothing cached, nothing replayed -- which is the claim the experiment
# has to be able to make, and the reason this is a job rather than a script
# over a saved array.
#
#   sbatch slurm/mask_probe.sh circ_lm10_r10_o20
#   REALM_RUN_TAG=circ_lm8_r10_ppo_s0 sbatch slurm/mask_probe.sh circ_lm8_r10
#   REALM_MASK_L1=1e-2 sbatch slurm/mask_probe.sh circ_lm10_r10_o20
#
# Needs a trained run and its fitted fields:
#   data/vpce/rl_runs/<tag>/model.zip
#   data/vpce/place_fields/<arena>_visual.h5  and  _bank.csv
#
# The bank is not optional. It is what turns a cell index into a scale and a
# place, and without it this produces a list of integers.
#
#SBATCH --job-name=maskprobe
#SBATCH --partition=general
#SBATCH --time=04:00:00
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
# Same exclusions as training: this renders 9 frames per step through GLX,
# and the compute-only cards cannot provide it. A step is about 0.25 s of rendering plus 0.3 s of mask optimisation,
# so a converged four-start run is minutes. Four hours is for the case where
# the policy wanders to the step cap.
# --------------------------------------------------------------------------

set -euo pipefail

ARENA="${1:-circ_lm10_r10_o20}"
shift || true
WEBOTS_EXTRA=("$@")

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"
mkdir -p slurm/logs data/vpce/mask_probe

SIF="${WEBOTS_SIF:-$REPO_DIR/webots_r2025a.sif}"
SINGULARITY=/apps/singularity/bin/singularity

# shellcheck disable=SC1091
source "$REPO_DIR/slurm/_webots_env.sh"
resolve_python
container_binds || exit 1

ALGO="${REALM_ALGO:-PPO}"
LOWER="$(echo "$ALGO" | tr 'A-Z' 'a-z')"
TAG="${REALM_RUN_TAG:-${ARENA}_${LOWER}_s0}"
OUT_DIR="data/vpce/mask_probe/$TAG"
WORLD="$REPO_DIR/simulation/worlds/mask_probe.wbt"

if [[ -z "${EMAIL_TO:-}" && "${ALLOW_NO_EMAIL:-0}" != "1" ]]; then
    {
        echo "ERROR: EMAIL_TO is not set, so the CSVs would be written to"
        echo "       $OUT_DIR/ and never sent."
        echo "  For this job:  EMAIL_TO=you@usf.edu sbatch slurm/mask_probe.sh $ARENA"
        echo "  Files only:    ALLOW_NO_EMAIL=1 sbatch slurm/mask_probe.sh $ARENA"
    } >&2
    exit 1
fi

[[ -f "$SIF"   ]] || { echo "ERROR: image not found: $SIF" >&2; exit 1; }
[[ -f "$WORLD" ]] || { echo "ERROR: world not found: $WORLD" >&2; exit 1; }
if [[ ! -f "data/vpce/rl_runs/$TAG/model.zip" ]]; then
    {
        echo "ERROR: no trained model at data/vpce/rl_runs/$TAG/model.zip"
        echo "       Train one first:  sbatch slurm/train_rl.sh $ARENA"
    } >&2
    exit 1
fi

XVFB_SCREEN="${XVFB_SCREEN:-1280x1024x24}"
WEBOTS_PORT=$(( 10000 + (${SLURM_JOB_ID:-0} % 20000) ))
XVFB_DISPLAY=$(( 100 + (${SLURM_JOB_ID:-0} % 400) ))

USE_GPU="${USE_GPU:-1}"
gpu_args
# Which card this job was given, for the record. This cannot fail the job
# and cannot change a setting: it is one line in a log, not a check.
echo "GPU     : $(nvidia-smi -L 2>/dev/null | head -1 || echo 'none reported')"

echo "===================================================================="
echo "Job     : ${SLURM_JOB_ID:-local}   node $(hostname)"
echo "Arena   : $ARENA"
echo "Run tag : $TAG   ->  $OUT_DIR"
echo "Mask    : l1=${REALM_MASK_L1:-3e-3} steps=${REALM_MASK_STEPS:-300} ref=${REALM_MASK_REF:-uniform}"
echo "Motion  : drive=${REALM_DRIVE:-0}  (0 = teleport, as trained)"
echo "Webots  : extra flags [${WEBOTS_EXTRA[*]:-none}]"
echo "Started : $(date -Is)"
echo "Git     : $(git rev-parse --short HEAD 2>/dev/null || echo 'no git')"
echo "===================================================================="

preflight || { echo 'PREFLIGHT FAILED - not launching Webots' >&2; exit 1; }
write_runtime_ini

export LP_NUM_THREADS="${LP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export PYTHONUNBUFFERED=1
T_START=$(date +%s)

WEBOTS_TIMEOUT="${WEBOTS_TIMEOUT:-14000}"
set +e
timeout --signal=TERM --kill-after=60 "$WEBOTS_TIMEOUT" \
"$SINGULARITY" exec "${BINDS[@]+"${BINDS[@]}"}" "${GPU_ARGS[@]+"${GPU_ARGS[@]}"}" \
    --env REALM_MAZE="$ARENA" \
    --env REALM_ARENA_DIR="${REALM_ARENA_DIR:-}" \
    --env REALM_RUN_TAG="$TAG" \
    --env REALM_ALGO="$ALGO" \
    --env REALM_ENSEMBLE="${REALM_ENSEMBLE:-}" \
    --env REALM_DRIVE="${REALM_DRIVE:-0}" \
    --env REALM_STARTS="${REALM_STARTS:-}" \
    --env REALM_MAX_STEPS="${REALM_MAX_STEPS:-200}" \
    --env REALM_MASK_L1="${REALM_MASK_L1:-3e-3}" \
    --env REALM_MASK_STEPS="${REALM_MASK_STEPS:-300}" \
    --env REALM_MASK_REF="${REALM_MASK_REF:-uniform}" \
    --env REALM_MASK_CAP="${REALM_MASK_CAP:-400}" \
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
[[ ${STATUS} -eq 124 ]] && echo "TIMED OUT after ${WEBOTS_TIMEOUT}s -- Webots does not quit on a controller crash; check for a traceback above."

echo "--- produced ---"
ls -lh "$OUT_DIR/" 2>/dev/null || echo "(nothing in $OUT_DIR)"

ATTACH=()
for f in "$OUT_DIR/mask_steps.csv" "$OUT_DIR/mask_cells.csv" "$OUT_DIR/config.json"; do
    [[ -f "$f" ]] && ATTACH+=("$f")
done

"$PYTHON_BIN" - "$ARENA" "$TAG" "$STATUS" "$ELAPSED" "$OUT_DIR" \
    "${ATTACH[@]+"${ATTACH[@]}"}" <<'PYMAIL'
import os
import sys

sys.path.insert(0, os.getcwd())
from realm_tools.experiment_lib.reporting import send_email

arena, tag, status, elapsed, out_dir = sys.argv[1:6]
attachments = sys.argv[6:]
status, elapsed = int(status), int(elapsed)
ok = status == 0

lines = [f'mask probe | arena = {arena} | run = {tag}',
         f'  exit status : {status} ({"ok" if ok else "FAILED"})',
         f'  elapsed     : {elapsed // 60}m {elapsed % 60}s',
         f'  job         : {os.environ.get("SLURM_JOB_ID", "local")} on '
         f'{os.environ.get("SLURMD_NODENAME", "?")}',
         '']

# The summary worth reading in the mail itself: how big the sets are and
# which scales they draw on, against what the bank holds. The enrichment
# column is the only one that means anything on its own -- a big field is
# active at more positions than a small one, so a raw count over-selects
# them whatever the policy is doing.
try:
    import pandas as pd
    steps = pd.read_csv(f'{out_dir}/mask_steps.csv')
    cells = pd.read_csv(f'{out_dir}/mask_cells.csv')
    lines.append(f'{len(steps)} states, {len(cells)} selected-cell rows')
    lines.append(f'  set size   : median {steps.set_size.median():.0f}, '
                 f'range {steps.set_size.min()}-{steps.set_size.max()}')
    lines.append(f'  mask time  : {steps.mask_seconds.mean():.2f}s mean')
    lines.append('')
    import json
    with open(f'{out_dir}/config.json') as fh:
        cfg = json.load(fh)
    bank = pd.read_csv(cfg['bank'])
    sel = cells.scale.value_counts(normalize=True).sort_index()
    pop = bank.scale.value_counts(normalize=True).sort_index()
    lines.append('scale   in bank   selected   enrichment')
    for s in sorted(pop.index):
        lines.append(f'  {s}      {pop.get(s, 0)*100:5.1f}%     '
                     f'{sel.get(s, 0)*100:5.1f}%      '
                     f'{(sel.get(s, 0)/pop[s]) if pop[s] else float("nan"):.2f}x')
    lines.append('')
    lines.append('Enrichment above 1 means that scale appears in the chosen')
    lines.append('sets more than its share of the bank. It is NOT yet evidence')
    lines.append('about the policy: a larger field is active at more positions,')
    lines.append('so it has more chances to be chosen. The comparison that')
    lines.append('would settle it is against the cells ACTIVE at each state,')
    lines.append('not against the bank. This is a proof of concept.')
except Exception as exc:                                    # noqa: BLE001
    lines.append(f'(could not summarise: {exc.__class__.__name__}: {exc})')

lines.append('')
lines.append(f'{len(attachments)} file(s) attached')
body = '\n'.join(lines)
print(body)
sent = send_email(f'[REALM-VPCE-RL] mask probe — {tag}'
                  + ('' if ok else ' [FAILED]'), body, attachments=attachments)
if os.environ.get('EMAIL_TO') and not sent:
    print('ERROR: the report was not delivered -- see the [report] line above.')
    sys.exit(1)
PYMAIL

exit ${STATUS}
