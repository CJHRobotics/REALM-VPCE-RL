"""Train (or evaluate) a navigation agent on the VPCE representation.

Webots launches this as the robot's controller, so the controller process IS
the training process. That is the simpler of the two ways to drive Webots
from Gymnasium and it is enough for one environment at a time, which is what
a first set of learning curves needs. Vectorised training over several
simulators needs the extern-controller route instead -- the environment
itself is unchanged, only the launching differs. See the module docstring of
realm_tools/rl_lib/webots_env.py.

Configured by environment variables, so a SLURM job can sweep conditions
without editing anything. They are read once, at the top, and echoed:

    REALM_MAZE        arena to load                  (circ_lm8_r10)
    REALM_ENSEMBLE    place-field .h5 to observe through
                      (data/vpce/place_fields/<maze>_visual.h5)
    REALM_TIMESTEPS   training steps                 (100000)
    REALM_MAX_STEPS   steps per episode              (200)
    REALM_ALGO        DQN or PPO                     (DQN)
    REALM_TELEPORT    1 teleports actions, 0 drives  (1)
    REALM_SHAPING     potential-based distance shaping weight (0 = sparse)
    REALM_SEED        random seed                    (0)
    REALM_RUN_TAG     names the output directory     (derived)
    REALM_EVAL_ONLY   1 to load the saved model and only evaluate
    REALM_STOP_AT     success rate counting as converged       (0.8)
    REALM_MAX_EPISODES give up after this many episodes        (unlimited)

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
from realm_tools.rl_lib import VpceObserver, WebotsEnv, RewardConfig
from realm_tools.rl_lib.agents import make_agent, algo_kwargs
from realm_tools.rl_lib.callbacks import ConvergenceStopper

MAZE = os.environ.get('REALM_MAZE', 'circ_lm8_r10')
ENSEMBLE = os.environ.get(
    'REALM_ENSEMBLE', f'data/vpce/place_fields/{MAZE}_visual.h5')
TIMESTEPS = int(os.environ.get('REALM_TIMESTEPS', 100_000))
MAX_STEPS = int(os.environ.get('REALM_MAX_STEPS', 200))
ALGO = os.environ.get('REALM_ALGO', 'DQN').upper()
TELEPORT = os.environ.get('REALM_TELEPORT', '1') == '1'
SHAPING = float(os.environ.get('REALM_SHAPING', 0.0))
SEED = int(os.environ.get('REALM_SEED', 0))
EVAL_ONLY = os.environ.get('REALM_EVAL_ONLY', '0') == '1'
STOP_AT = float(os.environ.get('REALM_STOP_AT', 0.8))
MAX_EPISODES = int(os.environ.get('REALM_MAX_EPISODES', 0)) or None
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

from stable_baselines3.common.monitor import Monitor

# Monitor writes the per-episode reward and length CSV. That file IS the
# learning curve, so it is written incrementally rather than at the end --
# a run killed by the walltime still leaves everything up to that point.
env = Monitor(env, filename=f'{OUT_DIR}/monitor.csv',
              info_keywords=('is_success',))

MODEL_PATH = f'{OUT_DIR}/model.zip'
if ALGO not in ('DQN', 'PPO'):
    print(f'ERROR: REALM_ALGO must be DQN or PPO, not {ALGO!r}',
          file=sys.stderr, flush=True)
    sys.exit(1)

with open(f'{OUT_DIR}/config.json', 'w') as f:
    json.dump(dict(maze=MAZE, ensemble=ENSEMBLE, n_cells=int(observer.n_cells),
                   channels=list(observer.channels), algo=ALGO,
                   timesteps=TIMESTEPS, max_steps=MAX_STEPS,
                   teleport=TELEPORT, shaping=SHAPING, seed=SEED,
                   tag=TAG,
                   # Recorded so a run can be reproduced, and so a sweep
                   # can be checked for having held them constant.
                   hyperparameters=algo_kwargs(ALGO)), f, indent=2)


def rollout(model, start_index=None):
    """One deterministic episode. Returns (reached, steps)."""
    options = None if start_index is None else {'start_index': start_index}
    obs, _ = env.reset(options=options)
    done = term = False
    steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, _info = env.step(int(action))
        steps += 1
        done = term or trunc
    return bool(term), steps


def evaluate(model, n_episodes=20):
    """Deterministic rollouts from random starts: success rate and length."""
    wins, lengths = 0, []
    for _ in range(n_episodes):
        reached, steps = rollout(model)
        if reached:
            wins += 1
            lengths.append(steps)
    return wins / n_episodes, (float(np.mean(lengths)) if lengths else float('nan'))


def rollout_every_start(model):
    """One deterministic episode from each start, for the path figure.

    Deliberately deterministic and one per start, rather than reusing
    whatever the last training episodes happened to be. Training episodes
    are epsilon-greedy, so their paths contain random actions and show the
    exploration rather than the policy; and with starts sampled at random
    the last episode from a given start can be thousands of steps stale.
    Six fresh rollouts cost six episodes of rendering and show what the
    trained policy actually does from each place.
    """
    n = env.unwrapped.n_starts()
    if n <= 0:
        print('  arena defines no start positions; skipping the path figure',
              flush=True)
        return {}
    env.unwrapped.last_paths.clear()
    for i in range(n):
        reached, steps = rollout(model, start_index=i)
        print(f'  start {i}: {"reached" if reached else "timed out"} '
              f'in {steps} steps', flush=True)
    return dict(env.unwrapped.last_paths)


t0 = time.time()
if EVAL_ONLY:
    if not os.path.exists(MODEL_PATH):
        print(f'ERROR: no model at {MODEL_PATH}', file=sys.stderr, flush=True)
        sys.exit(1)
    from stable_baselines3 import PPO, DQN
    model = {'DQN': DQN, 'PPO': PPO}[ALGO].load(MODEL_PATH, env=env)
    rate, mean_len = evaluate(model)
    print(f'eval: success {rate:.0%}  mean steps {mean_len:.1f}', flush=True)
else:
    model = make_agent(ALGO, env, seed=SEED, verbose=1)

    # Before training, so the learning curve has a baseline. An untrained
    # policy that already solves the task would mean the task is trivial,
    # and no comparison between representations could show anything.
    rate0, len0 = evaluate(model, n_episodes=10)
    print(f'before training: success {rate0:.0%}  mean steps {len0:.1f}',
          flush=True)

    def verify_deterministic():
        """Fraction of starts the greedy policy actually solves.

        Costs one episode per start, and is only called when training
        already looks converged -- cheap against the alternative of
        stopping on a policy that cycles in place under argmax.
        """
        n = env.unwrapped.n_starts()
        if n <= 0:
            return 1.0
        return sum(rollout(model, start_index=i)[0] for i in range(n)) / n

    stopper = ConvergenceStopper(threshold=STOP_AT, max_episodes=MAX_EPISODES,
                                 verify=verify_deterministic)
    model.learn(total_timesteps=TIMESTEPS, progress_bar=False,
                callback=stopper)
    model.save(MODEL_PATH)
    stop = stopper.summary()
    print(f'\ntraining ended: {stop["stop_reason"]} after {stop["episodes"]} '
          f'episodes ({stop["final_success_rate"]:.0%} success over the last '
          f'{stop["window"]})', flush=True)

    rate1, len1 = evaluate(model)
    print(f'after training : success {rate1:.0%}  mean steps {len1:.1f}',
          flush=True)
    result = dict(success_before=rate0, success_after=rate1,
                  steps_before=len0, steps_after=len1,
                  wall_seconds=time.time() - t0, **stop)

    # The path figure, drawn whether the run converged or gave up -- a
    # failure has a shape, and the shape is the diagnostic.
    print('\nfinal paths, one deterministic episode per start:', flush=True)
    try:
        import xml.etree.ElementTree as ET
        from realm_tools.rl_lib import path_plots
        paths = rollout_every_start(model)
        if paths:
            xml_root = ET.parse(
                f'simulation/worlds/environments/vpce/{MAZE}.xml').getroot()
            gx, gy = robot.maze.get_goal_location()
            png, psum = path_plots.plot_paths(
                paths, xml_root, f'{OUT_DIR}/final_paths.png',
                goal=(gx, gy),
                title=f'{TAG} | {stop["stop_reason"]} after '
                      f'{stop["episodes"]} episodes')
            result['paths'] = psum
            for line in path_plots.summarise(psum):
                print(line, flush=True)
            if png:
                print(f'  figure -> {png}', flush=True)
    except Exception as exc:                                # noqa: BLE001
        print(f'  WARNING: path figure failed '
              f'({exc.__class__.__name__}: {exc})', flush=True)

    # The learning curve. Wrapped, because the model and the Monitor CSV are
    # already on disk at this point and a matplotlib failure must not lose a
    # run that cost hours of rendering.
    try:
        from realm_tools.rl_lib import learning_curve
        png, summary = learning_curve.plot(
            f'{OUT_DIR}/monitor.csv', f'{OUT_DIR}/learning_curve.png',
            title=f'{TAG} | {MAZE} | {ALGO} | {observer.n_cells} cells')
        result['curve'] = summary
        print('\nlearning curve:', flush=True)
        for line in learning_curve.summarise(summary):
            print(line, flush=True)
        if png:
            print(f'  figure -> {png}', flush=True)
    except Exception as exc:                                # noqa: BLE001
        print(f'  WARNING: learning curve failed '
              f'({exc.__class__.__name__}: {exc})', flush=True)

    with open(f'{OUT_DIR}/result.json', 'w') as f:
        json.dump(result, f, indent=2)

print(f'done in {time.time() - t0:.0f}s  ->  {OUT_DIR}', flush=True)

# Quit, not reset: reset restarts the controller and this would run forever.
robot.experiment_supervisor.simulationQuit(0)
