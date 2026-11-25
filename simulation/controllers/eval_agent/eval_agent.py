"""Watch a trained agent solve the arena. For demos, videos and sanity.

rl_agent.py already evaluates under REALM_EVAL_ONLY=1, and on the cluster
that is the right thing: headless, fast mode, twenty episodes, a number at
the end. This controller is for the other case -- a person watching, in a
window, at a speed an eye can follow, with the terminal saying what the
agent is doing while it does it.

It is deliberately a separate controller rather than another flag on the
training one. The training path produces the measurements the whole project
rests on, and it should not grow branches that exist to make a recording
look good.

WHAT IS DIFFERENT FROM TRAINING, AND WHY IT MATTERS

By default this DRIVES rather than teleports. Training teleports: each
action moves the robot 0.5 m along a world bearing instantly, because 100k
steps of driving would cost weeks and the agent's task is navigation, not
motor control. perform_training_action covers the same displacement by
rotating and driving under PID, so the ACTION SEMANTICS are identical and
the policy is unchanged -- but the execution is not, and a wheel that slips
leaves the robot somewhere the teleporting agent would never have been.

That is a fine thing to show in a video and a bad thing to measure. Set
REALM_DRIVE=0 to teleport exactly as training did; use that for anything
that produces a number.

    REALM_MAZE        arena                        (circ_lm10_r10_o20)
    REALM_RUN_TAG     which run to load            (<maze>_ppo_s0)
    REALM_ALGO        PPO or DQN                   (PPO)
    REALM_ENSEMBLE    place-field .h5              (from the run's config)
    REALM_DRIVE       1 drives, 0 teleports        (1)
    REALM_MAX_STEPS   per episode                  (200)
    REALM_STARTS      which starts, e.g. 0,2,3     (all of them, in order)
    REALM_LOOP        1 repeats until you stop it  (1)
    REALM_DETERMINISTIC  1 greedy, 0 samples       (1)
    REALM_PAUSE       seconds between episodes     (1.5)

Run it from a terminal so the environment reaches Webots:

    REALM_MAZE=circ_lm10_r10_o20 webots simulation/worlds/eval_agent.wbt

In the GUI, use Real-Time rather than Fast, or there will be nothing to
watch. runtime.ini comes from setup/add_runtime_ini.py, same as every other
controller here.
"""

import json
import os
import sys
import time

# Same reasoning as rl_agent: before torch is imported by anything. There is
# no GPU worth having here -- the network is microseconds and the renderer
# wants the card -- and hiding it removes a class of surprise rather than
# relying on device='cpu' being honoured everywhere.
if os.environ.get('REALM_DEVICE', 'cpu').lower() == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

REPO = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np

from realm_tools.robot_lib.my_robot import MyRobot
from realm_tools.rl_lib import VpceObserver, WebotsEnv, RewardConfig
from realm_tools.simulation_lib.arena_paths import arena_xml
from realm_tools.simulation_lib.webots_exit import quit_on_crash

MAZE = os.environ.get('REALM_MAZE', 'circ_lm10_r10_o20')
ALGO = os.environ.get('REALM_ALGO', 'PPO').upper()
TAG = os.environ.get('REALM_RUN_TAG', f'{MAZE}_{ALGO.lower()}_s0')
RUN_DIR = f'data/vpce/rl_runs/{TAG}'
MODEL_PATH = f'{RUN_DIR}/model.zip'
DRIVE = os.environ.get('REALM_DRIVE', '1') == '1'
MAX_STEPS = int(os.environ.get('REALM_MAX_STEPS', 200))
LOOP = os.environ.get('REALM_LOOP', '1') == '1'
GREEDY = os.environ.get('REALM_DETERMINISTIC', '1') == '1'
PAUSE = float(os.environ.get('REALM_PAUSE', 1.5))

print(f'repo root: {REPO}', flush=True)

# The robot comes FIRST, before the checks below, and that order is the
# point. Webots does not exit when its controller does, so a sys.exit over a
# missing file would leave the simulator sitting in front of you with
# nothing running in it. quit_on_crash registers an atexit hook, which fires
# on sys.exit as well as on a traceback -- but only once there is a
# supervisor to call. Paying for MyRobot() before finding out the model is
# missing costs a second and makes every failure close its own window.
robot = MyRobot()
quit_webots = quit_on_crash(robot)


def die(*lines):
    for line in lines:
        print(line, file=sys.stderr, flush=True)
    quit_webots(1)
    sys.exit(1)


if not os.path.exists(MODEL_PATH):
    die(f'ERROR: no model at {MODEL_PATH}',
        '  Train one, or copy a run down:',
        f'    scp -r gaivi:REALM-VPCE-RL/{RUN_DIR} data/vpce/rl_runs/')

# The run's own record of what it was trained through. A policy is only
# meaningful against the representation it learned on, and nothing in a
# model.zip knows which ensemble that was -- so an ensemble refitted since
# would load without complaint and quietly mean nothing.
cfg = {}
if os.path.exists(f'{RUN_DIR}/config.json'):
    with open(f'{RUN_DIR}/config.json') as f:
        cfg = json.load(f)

ENSEMBLE = os.environ.get('REALM_ENSEMBLE') or cfg.get(
    'ensemble', f'data/vpce/place_fields/{MAZE}_visual.h5')

if not os.path.exists(ENSEMBLE):
    die(f'ERROR: no ensemble at {ENSEMBLE}',
        '  It is the representation this policy sees; without it the model '
        'is just weights.',
        f"    scp 'gaivi:REALM-VPCE-RL/data/vpce/place_fields/{MAZE}_visual*' "
        f"data/vpce/place_fields/")

if cfg.get('maze') and cfg['maze'] != MAZE:
    print(f'WARNING: this run was trained on {cfg["maze"]}, not {MAZE}. '
          f'The policy will act, and it will not mean anything.',
          file=sys.stderr, flush=True)

print(f'run     : {TAG}   ({ALGO})', flush=True)
print(f'arena   : {MAZE}', flush=True)
print(f'ensemble: {ENSEMBLE}', flush=True)
print(f'mode    : {"driving (video)" if DRIVE else "teleporting (as trained)"}',
      flush=True)
if cfg:
    print(f'trained : {cfg.get("timesteps", "?")} steps, seed '
          f'{cfg.get("seed", "?")}, teleport={cfg.get("teleport", "?")}',
          flush=True)

observer = VpceObserver(ENSEMBLE)
print(f'observation: {observer.n_cells} place cells over '
      f'{", ".join(observer.channels)}', flush=True)

robot.load_environment(arena_xml(MAZE), floor_texture='carpet')

env = WebotsEnv(
    robot, observer,
    max_steps=MAX_STEPS,
    teleport=not DRIVE,
    start_mode='training',
    start_index=0,
    reward=RewardConfig(),
)

from stable_baselines3 import DQN, PPO                        # noqa: E402

# A model.zip carries the training SCHEDULES as pickled callables, and this
# SB3 is not the one that wrote them: the cluster's saved a FloatSchedule
# this version has never heard of, so unpickling them warns twice and
# returns nothing useful. They are the learning rate and the PPO clip range
# -- used only by learn(), never by predict() -- so replacing them with
# constants costs an evaluation nothing and keeps a scary-looking warning
# out of a recording.
#
# It is still a version skew, and still worth knowing about, so it is said
# once in plain words rather than suppressed.
SCHEDULES = {'lr_schedule': lambda _: 0.0, 'clip_range': lambda _: 0.0}
model = {'DQN': DQN, 'PPO': PPO}[ALGO].load(
    MODEL_PATH, device='cpu', custom_objects=SCHEDULES)
print(f'loaded  : {MODEL_PATH}', flush=True)

# The zip stamps the version that wrote it, so the skew can be stated
# rather than inferred from a warning. SB3's own `load` does not hand this
# back on the model object; the file does.
import zipfile                                               # noqa: E402

import stable_baselines3 as sb3                              # noqa: E402

try:
    with zipfile.ZipFile(MODEL_PATH) as z:
        saved_by = z.read('_stable_baselines3_version').decode().strip()
except Exception:                                            # noqa: BLE001
    saved_by = None
if saved_by and saved_by != sb3.__version__:
    print(f'  note: written by stable-baselines3 {saved_by}, read by '
          f'{sb3.__version__}. The training schedules were replaced with '
          f'constants -- predict() never touches them, but do not call '
          f'learn() on this model here.', flush=True)

n_starts = env.n_starts()
if os.environ.get('REALM_STARTS'):
    starts = [int(s) for s in os.environ['REALM_STARTS'].split(',')]
else:
    starts = list(range(n_starts)) if n_starts > 0 else [-1]

# Compass bearings for the terminal, so a reader can follow the route
# without converting action indices in their head.
BEARING = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']


def run_episode(start_index):
    obs, info = env.reset(options={'start_index': start_index}
                          if start_index >= 0 else None)
    x, y, _ = info['pose']
    print(f'\nstart {start_index}  ({x:6.2f}, {y:6.2f})  '
          f'{info["dist_to_goal"]:.2f} m to the goal', flush=True)

    t0 = time.time()
    for step in range(1, MAX_STEPS + 1):
        action, _ = model.predict(obs, deterministic=GREEDY)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        x, y, _ = info['pose']
        print(f'  {step:3d}  {BEARING[action]:>2}  '
              f'({x:6.2f}, {y:6.2f})  {info["dist_to_goal"]:5.2f} m'
              + ('  [blocked]' if info['blocked'] else ''), flush=True)
        if terminated:
            print(f'  REACHED in {step} steps, {time.time() - t0:.1f}s',
                  flush=True)
            return True, step
        if truncated:
            print(f'  timed out at {MAX_STEPS} steps, '
                  f'{info["dist_to_goal"]:.2f} m short', flush=True)
            return False, step
    return False, MAX_STEPS


def pause():
    """Idle in the simulator, so the window keeps rendering while it waits."""
    for _ in range(max(1, int(PAUSE * 1000 / robot.timestep))):
        if robot.experiment_supervisor.step(robot.timestep) == -1:
            return


wins = total = 0
try:
    while True:
        for start_index in starts:
            reached, steps = run_episode(start_index)
            wins += bool(reached)
            total += 1
            pause()
        print(f'\n--- {wins}/{total} reached ---', flush=True)
        if not LOOP:
            break
except KeyboardInterrupt:
    print('\nstopped', flush=True)

print(f'\n{wins}/{total} episodes reached the goal', flush=True)
quit_webots(0)
