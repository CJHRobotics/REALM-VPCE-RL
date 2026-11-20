"""Train (or evaluate) a navigation agent on the VPCE representation.

Webots launches this as the robot's controller, so the controller process IS
the training process. That is the simpler of the two ways to drive Webots
from Gymnasium and it is enough for one environment at a time, which is what
a first set of learning curves needs. Vectorised training over several
simulators needs the extern-controller route instead -- the environment
itself is unchanged, only the launching differs. See the module docstring of
realm_tools/simulation_lib/webots_torch_environment.py.

Configured by environment variables, so a SLURM job can sweep conditions
without editing anything. They are read once, at the top, and echoed:

    REALM_MAZE        arena to load                  (circ_lm8_r10)
    REALM_ENSEMBLE    place-field .h5 to observe through
                      (data/vpce/place_fields/<maze>_visual.h5)
    REALM_TIMESTEPS   training steps                 (100000)
    REALM_MAX_STEPS   steps per episode              (200)
    REALM_ALGO        PPO or DQN                     (PPO)
    REALM_TELEPORT    1 teleports actions, 0 drives  (1)
    REALM_SHAPING     potential-based distance shaping weight (0 = sparse)
    REALM_SEED        random seed                    (0)
    REALM_RUN_TAG     names the output directory     (derived)
    REALM_EVAL_ONLY   1 to load the saved model and only evaluate

Output goes to data/vpce/rl_runs/<tag>/: the SB3 model, the Monitor CSV of
per-episode reward and length -- which is the learning curve, and so the
measurement the whole experiment exists to produce -- and a config.json
recording what produced them.
"""

import json
import os
import sys
import time

# Webots launches this from the controller's own directory, so the repo is
# neither the working directory nor on sys.path. Resolve from __file__ and
# add the root explicitly -- same reasoning as collect_data.py, and the
# reason no .pth is needed on the cluster.
REPO = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
print(f"repo root: {REPO}", flush=True)

import numpy as np

from realm_tools.robot_lib.my_robot import MyRobot
from realm_tools.simulation_lib.vpce_observer import VpceObserver
from realm_tools.simulation_lib.webots_torch_environment import (
    WebotsEnv, RewardConfig)

MAZE = os.environ.get('REALM_MAZE', 'circ_lm8_r10')
ENSEMBLE = os.environ.get(
    'REALM_ENSEMBLE', f'data/vpce/place_fields/{MAZE}_visual.h5')
TIMESTEPS = int(os.environ.get('REALM_TIMESTEPS', 100_000))
MAX_STEPS = int(os.environ.get('REALM_MAX_STEPS', 200))
ALGO = os.environ.get('REALM_ALGO', 'PPO').upper()
TELEPORT = os.environ.get('REALM_TELEPORT', '1') == '1'
SHAPING = float(os.environ.get('REALM_SHAPING', 0.0))
SEED = int(os.environ.get('REALM_SEED', 0))
EVAL_ONLY = os.environ.get('REALM_EVAL_ONLY', '0') == '1'
TAG = os.environ.get(
    'REALM_RUN_TAG',
    f'{MAZE}_{ALGO.lower()}_s{SEED}' + ('_shaped' if SHAPING else ''))

OUT_DIR = f'data/vpce/rl_runs/{TAG}'
os.makedirs(OUT_DIR, exist_ok=True)

print(f'maze={MAZE}  algo={ALGO}  steps={TIMESTEPS}  teleport={TELEPORT}  '
      f'shaping={SHAPING}  seed={SEED}', flush=True)
print(f'ensemble: {ENSEMBLE}', flush=True)
print(f'output  : {OUT_DIR}', flush=True)

if not os.path.exists(ENSEMBLE):
    # Named with the command that produces it: this is the first thing that
    # goes wrong when the fit job has not run, or ran for another arena.
    print(f'ERROR: no ensemble at {ENSEMBLE}\n'
          f'  Build it first:  sbatch slurm/fit_place_fields.sh {MAZE}',
          file=sys.stderr, flush=True)
    sys.exit(1)

# --- the pieces -----------------------------------------------------------
observer = VpceObserver(ENSEMBLE)
print(f'observation: {observer.n_cells} place cells over '
      f'{", ".join(observer.channels)}', flush=True)

# Checked before the robot is built, because it is the cheap failure. A
# mismatch here means the ensemble and this arena's dataset do not belong
# together, which produces a perfectly shaped and entirely meaningless
# observation.
dataset = f'data/vpce/collect_data/{MAZE}.h5'
if os.path.exists(dataset):
    observer.check_against_dataset(dataset)
    print('ensemble matches the arena dataset', flush=True)

robot = MyRobot(enable_cnn_features=False)
robot.load_environment(f'simulation/worlds/environments/vpce/{MAZE}.xml',
                       floor_texture='carpet')

env = WebotsEnv(
    robot, observer,
    max_steps=MAX_STEPS,
    teleport=TELEPORT,
    start_mode='training',
    start_index=-1,          # random starts: a fixed start can be memorised
    reward=RewardConfig(shaping=SHAPING),
)

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor

# Monitor writes the per-episode reward and length CSV. That file IS the
# learning curve, so it is written incrementally rather than at the end --
# a run killed by the walltime still leaves everything up to that point.
env = Monitor(env, filename=f'{OUT_DIR}/monitor.csv',
              info_keywords=('is_success',))

MODEL_PATH = f'{OUT_DIR}/model.zip'
Algo = {'PPO': PPO, 'DQN': DQN}.get(ALGO)
if Algo is None:
    print(f'ERROR: REALM_ALGO must be PPO or DQN, not {ALGO!r}',
          file=sys.stderr, flush=True)
    sys.exit(1)

with open(f'{OUT_DIR}/config.json', 'w') as f:
    json.dump(dict(maze=MAZE, ensemble=ENSEMBLE, n_cells=int(observer.n_cells),
                   channels=list(observer.channels), algo=ALGO,
                   timesteps=TIMESTEPS, max_steps=MAX_STEPS,
                   teleport=TELEPORT, shaping=SHAPING, seed=SEED,
                   tag=TAG), f, indent=2)


def evaluate(model, n_episodes=20):
    """Deterministic rollouts: success rate and steps to goal."""
    wins, lengths = 0, []
    for i in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _r, term, trunc, _info = env.step(int(action))
            steps += 1
            done = term or trunc
        if term:
            wins += 1
            lengths.append(steps)
    return wins / n_episodes, (float(np.mean(lengths)) if lengths else float('nan'))


t0 = time.time()
if EVAL_ONLY:
    if not os.path.exists(MODEL_PATH):
        print(f'ERROR: no model at {MODEL_PATH}', file=sys.stderr, flush=True)
        sys.exit(1)
    model = Algo.load(MODEL_PATH, env=env)
    rate, mean_len = evaluate(model)
    print(f'eval: success {rate:.0%}  mean steps {mean_len:.1f}', flush=True)
else:
    model = Algo('MlpPolicy', env, verbose=1, seed=SEED)

    # Before training, so the learning curve has a baseline. An untrained
    # policy that already solves the task would mean the task is trivial,
    # and no comparison between representations could show anything.
    rate0, len0 = evaluate(model, n_episodes=10)
    print(f'before training: success {rate0:.0%}  mean steps {len0:.1f}',
          flush=True)

    model.learn(total_timesteps=TIMESTEPS, progress_bar=False)
    model.save(MODEL_PATH)

    rate1, len1 = evaluate(model)
    print(f'after training : success {rate1:.0%}  mean steps {len1:.1f}',
          flush=True)
    with open(f'{OUT_DIR}/result.json', 'w') as f:
        json.dump(dict(success_before=rate0, success_after=rate1,
                       steps_before=len0, steps_after=len1,
                       wall_seconds=time.time() - t0), f, indent=2)

print(f'done in {time.time() - t0:.0f}s  ->  {OUT_DIR}', flush=True)

# Quit, not reset: reset restarts the controller and this would run forever.
robot.experiment_supervisor.simulationQuit(0)
