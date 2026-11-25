"""Walk a trained policy to the goal and ask, at every step, which place
cells it was using.

The agent drives the route it learned, in the simulator, through the same
camera and the same encoder it trained on. At each state the observation is
captured, a minimal deletion mask is optimised over the 1200-odd place
cells, and the smallest set whose removal changes the chosen action is
written out with each cell's field scale and geometry. Nothing is cached
and nothing is replayed: every observation here came off a camera a
millisecond earlier, which is the claim the experiment needs to be able to
make.

WHY THIS TELEPORTS BY DEFAULT, unlike eval_agent. That controller drives,
because driving is what looks right in a recording. This one produces
numbers, so it moves exactly as training did -- teleporting 0.5 m along the
action bearing -- and a wheel that slips is not allowed to put the robot
somewhere the trained agent would never have been. REALM_DRIVE=1 is there
for the video version of the same run.

WHAT COMES OUT

  <out>/mask_cells.csv    one row per selected cell per step: the cell, its
                          deletion weight, its activation, and its field --
                          scale band, radius, centroid, distance to wall.
  <out>/mask_steps.csv    one row per step: where the robot was, what it
                          did, how many cells the set needed, and the scale
                          composition of that set.
  <out>/observations.npz  every observation vector, pose and action.

The .npz is there so the mask can be recomputed at other settings without
re-running the simulator -- the lambda, the reference, the cap are all
arguments, and nobody should burn a render to change one. The CSVs still
come from this run, online, which is the point.

A CAVEAT TO CARRY INTO ANY PLOT. Some cells are multifield: one cell, more
than one blob in the arena. The bank gives such a cell a single centroid,
which for a multifield cell is the centroid of its parts and may sit where
the cell has no field at all. The `multifield` column is carried through so
those rows can be excluded from anything that treats a centroid as a place.

    REALM_MAZE        arena                       (circ_lm10_r10_o20)
    REALM_RUN_TAG     run to load                 (<maze>_ppo_s0)
    REALM_ALGO        PPO or DQN                  (PPO)
    REALM_ENSEMBLE    place fields                (from the run's config)
    REALM_DRIVE       1 drives, 0 teleports       (0)
    REALM_STARTS      e.g. 0,2                    (all)
    REALM_MAX_STEPS   per episode                 (200)
    REALM_MASK_L1     deletion-set size penalty   (3e-3)
    REALM_MASK_STEPS  optimiser iterations        (300)
    REALM_MASK_REF    uniform or zero             (uniform)
    REALM_MASK_CAP    give up past this many      (400)
"""

import json
import os
import sys
import time

if os.environ.get('REALM_DEVICE', 'cpu').lower() == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

REPO = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np
import pandas as pd

from realm_tools.robot_lib.my_robot import MyRobot
from realm_tools.rl_lib import VpceObserver, WebotsEnv, RewardConfig
from realm_tools.rl_lib.mask_saliency import minimal_cells
from realm_tools.simulation_lib.arena_paths import arena_xml
from realm_tools.simulation_lib.webots_exit import quit_on_crash

MAZE = os.environ.get('REALM_MAZE', 'circ_lm10_r10_o20')
ALGO = os.environ.get('REALM_ALGO', 'PPO').upper()
TAG = os.environ.get('REALM_RUN_TAG', f'{MAZE}_{ALGO.lower()}_s0')
RUN_DIR = f'data/vpce/rl_runs/{TAG}'
MODEL_PATH = f'{RUN_DIR}/model.zip'
OUT_DIR = f'data/vpce/mask_probe/{TAG}'
DRIVE = os.environ.get('REALM_DRIVE', '0') == '1'
MAX_STEPS = int(os.environ.get('REALM_MAX_STEPS', 200))
L1 = float(os.environ.get('REALM_MASK_L1', 3e-3))
OPT_STEPS = int(os.environ.get('REALM_MASK_STEPS', 300))
REF = os.environ.get('REALM_MASK_REF', 'uniform')
CAP = int(os.environ.get('REALM_MASK_CAP', 400))

print(f'repo root: {REPO}', flush=True)
os.makedirs(OUT_DIR, exist_ok=True)

robot = MyRobot()
quit_webots = quit_on_crash(robot)


def die(*lines):
    for line in lines:
        print(line, file=sys.stderr, flush=True)
    quit_webots(1)
    sys.exit(1)


if not os.path.exists(MODEL_PATH):
    die(f'ERROR: no model at {MODEL_PATH}',
        f'  scp -r gaivi:REALM-VPCE-RL/{RUN_DIR} data/vpce/rl_runs/')

cfg = {}
if os.path.exists(f'{RUN_DIR}/config.json'):
    with open(f'{RUN_DIR}/config.json') as f:
        cfg = json.load(f)

ENSEMBLE = os.environ.get('REALM_ENSEMBLE') or cfg.get(
    'ensemble', f'data/vpce/place_fields/{MAZE}_visual.h5')
BANK = ENSEMBLE[:-3] + '_bank.csv' if ENSEMBLE.endswith('.h5') else ''

if not os.path.exists(ENSEMBLE):
    die(f'ERROR: no ensemble at {ENSEMBLE}')
if not os.path.exists(BANK):
    die(f'ERROR: no field bank at {BANK}',
        '  It is what turns a cell index into a scale and a place, which is',
        '  the entire point of this experiment. It is written beside the',
        '  ensemble by analysis/fit_place_fields.py.')

bank = pd.read_csv(BANK)
# cell_index is written by the fit and is the row order; checked rather than
# assumed, because every number in the output hangs off this one join.
if not (bank['cell_index'].to_numpy() == np.arange(len(bank))).all():
    die(f'ERROR: {BANK} is not in cell_index order, so the join to the '
        f'observation vector would be wrong.')

print(f'run      : {TAG} ({ALGO})   arena {MAZE}', flush=True)
print(f'ensemble : {ENSEMBLE}', flush=True)
print(f'bank     : {BANK}  ({len(bank)} cells, scales '
      f'{sorted(bank["scale"].unique())})', flush=True)
print(f'mask     : l1={L1:g} steps={OPT_STEPS} ref={REF} cap={CAP}', flush=True)
print(f'motion   : {"driving" if DRIVE else "teleporting (as trained)"}',
      flush=True)

observer = VpceObserver(ENSEMBLE)
if observer.n_cells != len(bank):
    die(f'ERROR: the ensemble has {observer.n_cells} cells and the bank has '
        f'{len(bank)}. They are not from the same fit.')

robot.load_environment(arena_xml(MAZE), floor_texture='carpet')
env = WebotsEnv(robot, observer, max_steps=MAX_STEPS, teleport=not DRIVE,
                start_mode='training', start_index=0, reward=RewardConfig())

from stable_baselines3 import DQN, PPO                        # noqa: E402

model = {'DQN': DQN, 'PPO': PPO}[ALGO].load(
    MODEL_PATH, device='cpu',
    custom_objects={'lr_schedule': lambda _: 0.0,
                    'clip_range': lambda _: 0.0})
policy = model.policy
policy.set_training_mode(False)

GOAL = robot.maze.get_goal_location()
BEARING = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
SCALES = sorted(bank['scale'].unique())

n_starts = env.n_starts()
starts = ([int(s) for s in os.environ['REALM_STARTS'].split(',')]
          if os.environ.get('REALM_STARTS')
          else (list(range(n_starts)) if n_starts > 0 else [-1]))

cell_rows, step_rows, obs_log = [], [], []
t_start = time.time()

for start_index in starts:
    obs, info = env.reset(options={'start_index': start_index}
                          if start_index >= 0 else None)
    print(f'\nstart {start_index}: {info["dist_to_goal"]:.2f} m to the goal',
          flush=True)

    for step in range(1, MAX_STEPS + 1):
        x, y, _heading = info['pose']
        t0 = time.time()
        got = minimal_cells(policy, obs, reference=REF, l1=L1,
                            steps=OPT_STEPS, cap=CAP)
        mask_s = time.time() - t0
        action = got['action']
        cells = got['cells']

        obs_log.append(np.asarray(obs, dtype=np.float32))

        sub = bank.iloc[cells] if len(cells) else bank.iloc[[]]
        d_goal = np.hypot(sub['centroid_x'] - GOAL[0], sub['centroid_y'] - GOAL[1])
        d_robot = np.hypot(sub['centroid_x'] - x, sub['centroid_y'] - y)

        for rank, (c, w, a) in enumerate(zip(cells, got['weights'],
                                             got['activations']), start=1):
            row = bank.iloc[int(c)]
            cell_rows.append(dict(
                run=TAG, arena=MAZE, start=start_index, step=step,
                robot_x=round(float(x), 4), robot_y=round(float(y), 4),
                dist_to_goal=round(float(info['dist_to_goal']), 4),
                action=action, action_bearing=BEARING[action],
                flipped_to=got['flipped_to'], set_size=got['n_cells'],
                rank=rank, cell_index=int(c),
                deletion_weight=round(float(w), 6),
                activation=round(float(a), 8),
                scale=int(row['scale']),
                radius_env_m=round(float(row['radius_env_m']), 4),
                centroid_x=round(float(row['centroid_x']), 4),
                centroid_y=round(float(row['centroid_y']), 4),
                dist_to_wall_m=round(float(row['dist_to_wall_m']), 4),
                multifield=int(row['multifield']),
                cell_to_goal_m=round(float(np.hypot(row['centroid_x'] - GOAL[0],
                                                    row['centroid_y'] - GOAL[1])), 4),
                cell_to_robot_m=round(float(np.hypot(row['centroid_x'] - x,
                                                     row['centroid_y'] - y)), 4)))

        srow = dict(run=TAG, arena=MAZE, start=start_index, step=step,
                    robot_x=round(float(x), 4), robot_y=round(float(y), 4),
                    dist_to_goal=round(float(info['dist_to_goal']), 4),
                    action=action, action_bearing=BEARING[action],
                    flipped_to=got['flipped_to'], set_size=got['n_cells'],
                    mask_seconds=round(mask_s, 3),
                    set_activation_mass=round(float(got['activations'].sum()), 6),
                    mean_scale=round(float(sub['scale'].mean()), 4) if len(sub) else np.nan,
                    mean_radius_m=round(float(sub['radius_env_m'].mean()), 4) if len(sub) else np.nan,
                    mean_cell_to_goal_m=round(float(d_goal.mean()), 4) if len(sub) else np.nan,
                    mean_cell_to_robot_m=round(float(d_robot.mean()), 4) if len(sub) else np.nan)
        for s in SCALES:
            srow[f'n_scale{s}'] = int((sub['scale'] == s).sum()) if len(sub) else 0
        step_rows.append(srow)

        print(f'  {step:3d} {BEARING[action]:>2}  {info["dist_to_goal"]:5.2f} m'
              f'   {got["n_cells"]:4d} cells flip it -> '
              f'{BEARING[got["flipped_to"]] if got["flipped_to"] is not None else "none"}'
              f'   mean scale {srow["mean_scale"]}   ({mask_s:.2f}s)', flush=True)

        obs, _r, terminated, truncated, info = env.step(action)
        if terminated:
            print(f'  reached in {step} steps', flush=True)
            break
        if truncated:
            print(f'  timed out at {MAX_STEPS}', flush=True)
            break

cells_csv = f'{OUT_DIR}/mask_cells.csv'
steps_csv = f'{OUT_DIR}/mask_steps.csv'
pd.DataFrame(cell_rows).to_csv(cells_csv, index=False)
pd.DataFrame(step_rows).to_csv(steps_csv, index=False)
np.savez_compressed(
    f'{OUT_DIR}/observations.npz',
    observations=np.asarray(obs_log, dtype=np.float32),
    start=np.array([r['start'] for r in step_rows]),
    step=np.array([r['step'] for r in step_rows]),
    action=np.array([r['action'] for r in step_rows]),
    robot_x=np.array([r['robot_x'] for r in step_rows]),
    robot_y=np.array([r['robot_y'] for r in step_rows]))

with open(f'{OUT_DIR}/config.json', 'w') as f:
    json.dump(dict(run=TAG, arena=MAZE, algo=ALGO, ensemble=ENSEMBLE,
                   bank=BANK, drive=DRIVE, max_steps=MAX_STEPS,
                   mask_l1=L1, mask_steps=OPT_STEPS, mask_reference=REF,
                   mask_cap=CAP, n_states=len(step_rows),
                   wall_seconds=round(time.time() - t_start, 1)), f, indent=2)

print(f'\n{len(step_rows)} states over {len(starts)} start(s), '
      f'{time.time() - t_start:.0f}s', flush=True)
print(f'  -> {cells_csv}  ({len(cell_rows)} rows)', flush=True)
print(f'  -> {steps_csv}', flush=True)
print(f'  -> {OUT_DIR}/observations.npz', flush=True)
quit_webots(0)
