#!/bin/bash
#
# Build the place-field ensemble for one arena on GAIVI, and mail the figures.
#
# Reads the collected feature dataset, runs the agglomeration, writes the
# ensemble and its bank, renders the fields projected into the arena, and
# emails the pictures. The ensemble is the artefact the RL work loads as its
# state space; the figures are how you check it is worth loading.
#
# Usage:
#   sbatch slurm/fit_place_fields.sh                          # circ_lm8_r10, visual
#   sbatch slurm/fit_place_fields.sh circ_lm8_r10 --scale 3   # single scale
#   sbatch slurm/fit_place_fields.sh circ_lm8_r10 --scale 2-5 # multiscale
#   sbatch slurm/fit_place_fields.sh circ_lm8_r10 --channels hog
#   sbatch slurm/fit_place_fields.sh circ_lm8_r10 --rules paper
#
# Everything after the arena name goes straight to fit_place_fields.py, so a
# configuration can be swept without editing this script.
#
# NO CONTAINER. Unlike every other job here, this one runs directly in the
# conda environment: it is numpy, scipy and torch over an already-collected
# dataset, with no Webots, no rendering, no Xvfb and no display. channels.py
# deliberately avoids importing image_feature_lib (and so cv2, and so libGL)
# except for the legacy dataset layout, which this is not. Singularity would
# add a bind-mount surface and a startup cost for nothing.
#
# ------------------------------------------------------------- SLURM header
#SBATCH --job-name=fitfields
#SBATCH --partition=general
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chamilton4@usf.edu
#
# NOTE the absent --exclude. Every other job here excludes GPU1, GPU2, GPU45
# -GPU49 and GPU53 because Webots renders through GLX and a compute-only card
# has no display engine to provide it. This job never opens a GL context: the
# GPU is used for one large matrix product. So the compute-only A100s, H100
# and L40S are not merely acceptable here, they are the best nodes for it,
# and excluding them would queue behind the cards this job has least use for.
#
# Memory, at N = 30,129 locations and the widest channel set (~59,600
# features):
#
#   X, the feature matrix            30,129 x 59,632 x 4 B   ~7.2 GB
#   D2_feat, the N x N distances     30,129^2 x 4 B          ~3.6 GB
#   the condensed vector, float64    N(N-1)/2 x 8 B          ~3.6 GB
#   scipy's linkage working copy     about the same again    ~3.6 GB
#                                                            -------
#                                                            ~18 GB peak
#
# 64G leaves room for the response arrays and the interpreter. Drop it for a
# single narrow channel (--channels hog is a third of the width) if the queue
# is long. If this is ever OOM-killed, --subsample is the lever: the cost is
# quadratic in locations, so halving N quarters both the matrix and the time.
#
# Time: the Gram matrix goes on the GPU, but Ward linkage is sequential and
# stays in scipy on the CPU, which is what the 8 h is for. A single arena and
# channel set is far quicker than that; the headroom is for the widest
# configuration on a busy node.
# --------------------------------------------------------------------------

set -euo pipefail

ARENA="${1:-circ_lm8_r10}"
shift || true
FIT_ARGS=("$@")

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"
mkdir -p slurm/logs analysis/figures/place_fields data/vpce/place_fields

# Only resolve_python is wanted from here -- the file is function definitions
# plus two defaults, so sourcing it pulls in no container behaviour.
# shellcheck disable=SC1091
source "$REPO_DIR/slurm/_webots_env.sh"
resolve_python

# The figures are the product, so refuse to start without somewhere to send
# them, exactly as render_check.sh does. reporting.send_email never raises,
# so without this check an unset EMAIL_TO would surface as one line at the
# end of a log, after the whole fit had run.
if [[ -z "${EMAIL_TO:-}" && "${ALLOW_NO_EMAIL:-0}" != "1" ]]; then
    {
        echo "ERROR: EMAIL_TO is not set, so the figures would be written to"
        echo "       analysis/figures/place_fields/ and never sent."
        echo
        echo "  For this job:  EMAIL_TO=you@usf.edu sbatch slurm/fit_place_fields.sh $ARENA"
        echo "  Or in ~/.bashrc: export EMAIL_TO=you@usf.edu EMAIL_FROM=you@usf.edu"
        echo "                   export EMAIL_SMTP=smtp.usf.edu EMAIL_SMTP_PORT=587"
        echo
        echo "  Files only:    ALLOW_NO_EMAIL=1 sbatch slurm/fit_place_fields.sh $ARENA"
    } >&2
    exit 1
fi

DATA="$REPO_DIR/data/vpce/collect_data/$ARENA.h5"
if [[ ! -f "$DATA" ]]; then
    {
        echo "ERROR: no dataset at $DATA"
        echo "       Collect it first:  sbatch slurm/collect_data.sh $ARENA"
    } >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
T_START=$(date +%s)

# A sentinel file, compared with `find -newer`, rather than `-newermt
# "@$T_START"`. collect_data.sh uses the latter and is correct on GAIVI, but
# it depends on GNU find's epoch parsing: BSD find accepts the same syntax
# and then fails to match files written seconds afterwards, so the same
# script silently mails zero attachments when dry-run on a Mac. `-newer
# <file>` is POSIX and asks no question about timestamp formats.
T_MARK="$(mktemp "${TMPDIR:-/tmp}/fitfields.XXXXXX")"
trap 'rm -f "$T_MARK"' EXIT

echo "===================================================================="
echo "Job     : ${SLURM_JOB_ID:-local}   node $(hostname)"
echo "Arena   : $ARENA"
echo "Fit args: ${FIT_ARGS[*]:-(defaults)}"
echo "Dataset : $DATA ($(du -h "$DATA" | cut -f1))"
echo "Python  : $PYTHON_BIN"
echo "Email   : ${EMAIL_TO:-(unset - ALLOW_NO_EMAIL, files written only)}"
echo "Started : $(date -Is)"
echo "Git     : $(git rev-parse --short HEAD 2>/dev/null || echo 'no git')"
echo "===================================================================="

"$PYTHON_BIN" -c 'import torch; print("torch", torch.__version__,
      "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")' || true
echo

# --- fit ------------------------------------------------------------------
# Not under `set -e`: a non-zero exit still has figures worth mailing in some
# cases, and the status is reported and re-raised at the end.
set +e
"$PYTHON_BIN" analysis/fit_place_fields.py "$ARENA" "${FIT_ARGS[@]+"${FIT_ARGS[@]}"}"
STATUS=$?
set -e

echo
echo "Finished : $(date -Is)  (exit ${STATUS})"

# --- mail the results -----------------------------------------------------
# Found by mtime rather than by reconstructing the output names, because
# FIT_ARGS can change the channel set and the stem with it. Same approach as
# collect_data.sh.
#
# A read loop rather than `mapfile`: mapfile is a bash 4 builtin and macOS
# still ships bash 3.2, where it fails into an EMPTY array -- so the job
# would mail a cheerful report with no attachments. GAIVI has bash 5 and
# would have been fine, but a silent empty attachment list is the exact
# failure this job exists to avoid, and the loop also lets the script be
# dry-run on a laptop before it is queued.
ATTACH=()
while IFS= read -r f; do
    [[ -n "$f" ]] && ATTACH+=("$f")
done < <(
    find analysis/figures/place_fields -name '*.png' -newer "$T_MARK" 2>/dev/null
    find data/vpce/place_fields -name '*_bank.csv' -newer "$T_MARK" 2>/dev/null
)

echo "--- produced ---"
if ((${#ATTACH[@]})); then
    ls -lh "${ATTACH[@]}"
else
    echo "(nothing new - see the log above)"
fi
ls -lh data/vpce/place_fields/ 2>/dev/null || true

ELAPSED=$(( $(date +%s) - T_START ))
"$PYTHON_BIN" - "$ARENA" "$STATUS" "$ELAPSED" "${ATTACH[@]+"${ATTACH[@]}"}" <<'PYMAIL'
import os
import sys

sys.path.insert(0, os.getcwd())
from realm_tools.experiment_lib.reporting import send_email

arena, status, elapsed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
attachments = sys.argv[4:]

ok = status == 0
lines = [f'place-field fit | arena = {arena}',
         f'  exit status : {status} ({"ok" if ok else "FAILED"})',
         f'  elapsed     : {elapsed // 60}m {elapsed % 60}s',
         f'  job         : {os.environ.get("SLURM_JOB_ID", "local")} on '
         f'{os.environ.get("SLURMD_NODENAME", "?")}',
         '']

# The bank is small enough to summarise in the body, which makes the mail
# useful on a phone without opening the attachment.
banks = [a for a in attachments if a.endswith('_bank.csv')]
try:
    import pandas as pd
    for b in banks:
        df = pd.read_csv(b)
        lines.append(f'{os.path.basename(b)}: {len(df)} fields')
        if len(df):
            counts = df['scale'].value_counts().sort_index()
            lines.append('  cells per scale: ' +
                         '  '.join(f'S{k}={v}' for k, v in counts.items()))
            for col, label in (('radius_members_m', 'r from members'),
                               ('radius_env_m', 'r projected'),
                               ('area_env_m2', 'area m^2')):
                if col in df:
                    lines.append(f'  {label:15s}: {df[col].min():.3g} .. '
                                 f'{df[col].max():.3g}  '
                                 f'(median {df[col].median():.3g})')
            if 'cc_frac' in df:
                lines.append(f'  fragmented     : '
                             f'{100 * (df["cc_frac"] < 0.8).mean():.0f}% of fields')
        lines.append('')
except Exception as exc:                                    # noqa: BLE001
    lines.append(f'(bank summary unavailable: {exc})')

figs = [a for a in attachments if a.endswith('.png')]
lines.append(f'{len(figs)} figure(s) attached'
             if figs else 'NO FIGURES PRODUCED - check the log')

body = '\n'.join(lines)
print(body)
sent = send_email(f'[REALM-VPCE-RL] place fields — {arena}'
                  + ('' if ok else ' [FAILED]'),
                  body, attachments=attachments)

# Same rule as render_check: the pictures are the product, so an intended
# send that did not happen is a failed run. EMAIL_TO unset means the job was
# launched with ALLOW_NO_EMAIL, so there was nothing to deliver.
wanted = bool(os.environ.get('EMAIL_TO'))
if wanted and not sent:
    print('ERROR: the report was not delivered -- see the [report] line above.')
    sys.exit(1)
PYMAIL

exit ${STATUS}
