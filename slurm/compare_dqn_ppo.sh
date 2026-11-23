#!/bin/bash
#
# One DQN run, one PPO run, and the comparison between them -- submitted
# together, with the comparison waiting on both.
#
#   bash slurm/compare_dqn_ppo.sh                    # circ_lm8_r10, seed 0
#   bash slurm/compare_dqn_ppo.sh circ_lm8_r10
#   REALM_SEED=3 bash slurm/compare_dqn_ppo.sh circ_lm8_r10
#   REALM_TIMESTEPS=200000 bash slurm/compare_dqn_ppo.sh circ_lm8_r10
#
# Run it with BASH on the login node, not with sbatch. It submits three
# jobs: the two training runs, which queue in parallel, and a third that
# waits for both and mails the comparison figure. The third is tiny -- one
# CPU, no GPU, a few minutes -- because all it does is read two CSVs and
# draw them.
#
# The dependency is `afterany`, not `afterok`: if one training run fails or
# times out the comparison still runs, and comparing one finished run
# against one partial one is more useful than getting nothing. The Monitor
# CSV is flushed per episode, so even a killed run has a curve worth seeing.
#
# ONE SEED EACH IS NOT A COMPARISON. It is a smoke test -- enough to see
# that both algorithms learn on this representation and roughly how long
# they take. Measured on the stand-in robot, two DQN runs differing only in
# seed took 8,901 and 22,147 steps to the same threshold, a 2.5x spread,
# while two PPO runs took 11,147 and 11,543. Read against that, a single
# seed cannot distinguish "DQN is faster" from "this DQN seed was lucky".
# For anything that goes in the paper use the sweep instead:
#
#   bash slurm/train_rl.sh --submit <arena>          # 3 seeds each
#   SEEDS=0,1,2,3,4 bash slurm/train_rl.sh --submit <arena>
#
# ------------------------------------------------------------- SLURM header
# These apply only to the comparison step, which is what gets sbatched from
# this file. The training jobs carry train_rl.sh's own header.
#SBATCH --job-name=cmp-dqn-ppo
#SBATCH --partition=general
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=chamilton4@usf.edu
#
# No --gres and no --exclude. This step draws two line charts; it neither
# renders nor multiplies matrices, so a GPU would sit idle and asking for
# one would only lengthen the queue.
# --------------------------------------------------------------------------

set -euo pipefail

ARENA="${1:-circ_lm8_r10}"
SEED="${REALM_SEED:-0}"
DQN_TAG="${ARENA}_dqn_s${SEED}"
PPO_TAG="${ARENA}_ppo_s${SEED}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"
mkdir -p slurm/logs

# ===========================================================================
# Login-node mode: submit the three jobs and exit.
# ===========================================================================
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if [[ -z "${EMAIL_TO:-}" && "${ALLOW_NO_EMAIL:-0}" != "1" ]]; then
        {
            echo "ERROR: EMAIL_TO is not set, so nothing would be sent."
            echo "  EMAIL_TO=you@usf.edu bash slurm/compare_dqn_ppo.sh $ARENA"
            echo "  Files only: ALLOW_NO_EMAIL=1 bash slurm/compare_dqn_ppo.sh $ARENA"
        } >&2
        exit 1
    fi

    ENSEMBLE="${REALM_ENSEMBLE:-data/vpce/place_fields/${ARENA}_visual.h5}"
    if [[ ! -f "$ENSEMBLE" ]]; then
        {
            echo "ERROR: no place-field ensemble at $ENSEMBLE"
            echo "       Build it first:  sbatch slurm/fit_place_fields.sh $ARENA"
        } >&2
        exit 1
    fi

    # --parsable gives just the job id, so it can be fed to --dependency.
    DQN_ID=$(sbatch --parsable --job-name="trainrl-dqn-s${SEED}" \
        --export=ALL,REALM_ALGO=DQN,REALM_SEED="$SEED",REALM_RUN_TAG="$DQN_TAG" \
        slurm/train_rl.sh "$ARENA")
    PPO_ID=$(sbatch --parsable --job-name="trainrl-ppo-s${SEED}" \
        --export=ALL,REALM_ALGO=PPO,REALM_SEED="$SEED",REALM_RUN_TAG="$PPO_TAG" \
        slurm/train_rl.sh "$ARENA")

    # afterany: run the comparison whether or not both succeeded.
    CMP_ID=$(sbatch --parsable --dependency=afterany:"${DQN_ID}":"${PPO_ID}" \
        --export=ALL,REALM_SEED="$SEED" \
        slurm/compare_dqn_ppo.sh "$ARENA")

    cat <<EOF

submitted for $ARENA (seed $SEED):
  $DQN_ID  DQN  -> data/vpce/rl_runs/$DQN_TAG
  $PPO_ID  PPO  -> data/vpce/rl_runs/$PPO_TAG
  $CMP_ID  comparison, waiting on both

Watch:    squeue -u \$USER
Logs:     tail -f slurm/logs/trainrl-*-s${SEED}-*.out
Compare by hand at any point, even part-way through:
          python3 analysis/compare_runs.py --runs $DQN_TAG $PPO_TAG --by algo
EOF
    exit 0
fi

# ===========================================================================
# Under slurm: both training jobs have finished (or failed). Compare them.
# ===========================================================================
# shellcheck disable=SC1091
source "$REPO_DIR/slurm/_webots_env.sh"
resolve_python

echo "===================================================================="
echo "Job     : ${SLURM_JOB_ID}   node $(hostname)"
echo "Arena   : $ARENA   seed $SEED"
echo "Runs    : $DQN_TAG   $PPO_TAG"
echo "Python  : $PYTHON_BIN"
echo "Started : $(date -Is)"
echo "===================================================================="

for tag in "$DQN_TAG" "$PPO_TAG"; do
    if [[ -f "data/vpce/rl_runs/$tag/monitor.csv" ]]; then
        echo "  found $tag"
    else
        # Not fatal: compare_runs skips what it cannot read, and one curve
        # is still worth looking at.
        echo "  MISSING $tag (its training job produced no monitor.csv)"
    fi
done

OUT_NAME="${ARENA}_dqn_vs_ppo_s${SEED}"
FIG_DIR="analysis/figures/rl_comparison"
mkdir -p "$FIG_DIR"

set +e
"$PYTHON_BIN" analysis/compare_runs.py \
    --runs "$DQN_TAG" "$PPO_TAG" --by algo --out-name "$OUT_NAME"
STATUS=$?
set -e

ATTACH=()
while IFS= read -r f; do
    [[ -n "$f" ]] && ATTACH+=("$f")
done < <(
    find "$FIG_DIR" -name "${OUT_NAME}.png" 2>/dev/null
    find "$FIG_DIR" -name "${OUT_NAME}.csv" 2>/dev/null
    # The individual runs' own figures, so the mail carries the learning
    # curves and final paths as well as the comparison.
    find "data/vpce/rl_runs/$DQN_TAG" "data/vpce/rl_runs/$PPO_TAG" \
        -name '*.png' 2>/dev/null
)

echo "--- attaching ---"
if ((${#ATTACH[@]})); then ls -lh "${ATTACH[@]}"; else echo "(nothing)"; fi

"$PYTHON_BIN" - "$ARENA" "$SEED" "$DQN_TAG" "$PPO_TAG" "$STATUS" \
    "${ATTACH[@]+"${ATTACH[@]}"}" <<'PYMAIL'
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from realm_tools.experiment_lib.reporting import send_email

arena, seed, dqn_tag, ppo_tag, status = sys.argv[1:6]
attachments = sys.argv[6:]

lines = [f'DQN vs PPO | arena = {arena} | seed = {seed}', '']
for tag in (dqn_tag, ppo_tag):
    path = f'data/vpce/rl_runs/{tag}/result.json'
    if not os.path.exists(path):
        lines += [f'{tag}: no result.json -- the run did not finish', '']
        continue
    try:
        with open(path) as f:
            r = json.load(f)
        paths = r.get('paths') or {}
        lines.append(tag)
        lines.append(f'  stopped       : {r.get("stop_reason", "?")} after '
                     f'{r.get("episodes", "?")} episodes')
        if r.get('success_after') is not None:
            lines.append(f'  deterministic : {r["success_after"]:.0%} '
                         f'({paths.get("reached", "?")}/'
                         f'{paths.get("starts", "?")} starts)')
        curve = r.get('curve') or {}
        if curve.get('steps_to_threshold') is not None:
            lines.append(f'  steps to 80%  : '
                         f'{curve["steps_to_threshold"]:,} (sustained)')
        elif curve:
            lines.append('  steps to 80%  : never held to the end')
        if r.get('wall_seconds'):
            lines.append(f'  wall clock    : {r["wall_seconds"]/3600:.2f} h')
        lines.append('')
    except Exception as exc:                                # noqa: BLE001
        lines += [f'{tag}: unreadable ({exc})', '']

lines += [
    'ONE SEED EACH. Two DQN runs differing only in seed took 8,901 and',
    '22,147 steps to the same threshold on the stand-in task -- a 2.5x',
    'spread. A difference between these two runs smaller than that says',
    'nothing about the algorithms. For a real comparison:',
    '  bash slurm/train_rl.sh --submit ' + arena,
    '']

figs = [a for a in attachments if a.endswith('.png')]
lines.append(f'{len(figs)} figure(s) attached' if figs
             else 'NO FIGURES -- check the log')

body = '\n'.join(lines)
print(body)
sent = send_email(f'[REALM-VPCE-RL] DQN vs PPO — {arena} seed {seed}',
                  body, attachments=attachments)

if os.environ.get('EMAIL_TO') and not sent:
    print('ERROR: the report was not delivered -- see the [report] line above.')
    sys.exit(1)
PYMAIL

exit ${STATUS}
