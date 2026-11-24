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
    REALM_ARENA_DIR   environments/<dir> it lives in (vpce)
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
    REALM_GIVE_UP_BELOW  abandon under this success rate       (0.10)
    REALM_GIVE_UP_AFTER  fraction of the budget before checking (0.5)
    REALM_DEVICE      cpu or cuda for the network             (cpu)

Output goes to data/vpce/rl_runs/<tag>/: the SB3 model, the Monitor CSV of
per-episode reward and length -- which is the learning curve, and so the
measurement the whole experiment exists to produce -- and a config.json
recording what produced them.
"""

import json
import os
import sys
import time

# Hide the GPU from torch BEFORE anything imports it. This must stay above
# every other import: stable_baselines3 pulls torch in, and once torch has
# seen a device the setting no longer has any effect.
#
# Two reasons, and the second is the one that forced this.
#
# Webots holds the GPU for rendering, and the controller is a separate
# process. On a card in exclusive-process compute mode the controller
# cannot also create a CUDA context, so it gets cudaErrorDevicesUnavailable
# however politely it asks.
#
# A visible-but-unusable device is NOT neutral. Selecting device='cpu' is
# not enough: torch 2.14's optimizer calls torch.accelerator.current_stream()
# unconditionally in its graph-capture health check, and torch.accelerator
# resolves from AVAILABILITY rather than from where the tensors live. So a
# run that had correctly fallen back to CPU still died 2048 steps in, inside
# optimizer.step(). Hiding the device removes the whole class of problem.
#
# This costs essentially nothing: measured on this task the network is
# 0.15 ms of a 118 ms environment step. Set REALM_DEVICE=cuda to opt back
# in, which is only sensible where Webots is not holding the card.
if os.environ.get('REALM_DEVICE', 'cpu').lower() == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

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
# Which environments directory the arena lives in. `vpce` is the place-field
# paper's sweep of empty enclosures; `vpce-rl` holds the obstacle arenas.
ARENA_DIR = os.environ.get('REALM_ARENA_DIR', 'vpce').strip('/')
ARENA_XML = f'simulation/worlds/environments/{ARENA_DIR}/{MAZE}.xml'
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
# Abandon a run that is clearly not learning, rather than spending the rest
# of the budget rendering for nothing. Checked only after half the budget,
# well past DQN's 30% exploration anneal, so it reads the policy and not
# epsilon. Set REALM_GIVE_UP_BELOW=0 to disable.
GIVE_UP_BELOW = float(os.environ.get('REALM_GIVE_UP_BELOW', 0.10)) or None
GIVE_UP_AFTER = int(float(os.environ.get('REALM_GIVE_UP_AFTER', 0.5))
                    * TIMESTEPS)
TAG = os.environ.get(
    'REALM_RUN_TAG',
    f'{MAZE}_{ALGO.lower()}_s{SEED}' + ('_shaped' if SHAPING else ''))

OUT_DIR = f'data/vpce/rl_runs/{TAG}'
os.makedirs(OUT_DIR, exist_ok=True)

print(f'maze={MAZE}  algo={ALGO}  steps={TIMESTEPS}  teleport={TELEPORT}  '
      f'shaping={SHAPING}  seed={SEED}', flush=True)
print(f'stop at {STOP_AT:.0%} success; give up below '
      f'{GIVE_UP_BELOW if GIVE_UP_BELOW else 0:.0%} after {GIVE_UP_AFTER:,} '
      f'steps', flush=True)
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

robot = MyRobot()
robot.load_environment(ARENA_XML, floor_texture='carpet')

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
                   # can be checked for having held them constant. The
                   # give-up bound especially: a run it stopped is not
                   # comparable with one that had the full budget.
                   give_up_below=GIVE_UP_BELOW,
                   give_up_after_steps=GIVE_UP_AFTER,
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


def evaluate(model, n_episodes=20, label='eval'):
    """Deterministic rollouts from random starts: success rate and length.

    Reports each episode as it finishes. Ten episodes of a policy that mostly
    truncates at 200 steps is several minutes of wall clock, and without a
    line per episode that stretch is indistinguishable from a hang -- which
    is exactly how it was read. collect_data.py carries the same lesson.
    """
    wins, lengths = 0, []
    for i in range(n_episodes):
        reached, steps = rollout(model)
        if reached:
            wins += 1
            lengths.append(steps)
        print(f'  {label} {i + 1}/{n_episodes}: '
              f'{"reached in" if reached else "timed out after"} {steps} steps',
              flush=True)
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
    model = make_agent(ALGO, env, seed=SEED, verbose=1,
                       device=os.environ.get('REALM_DEVICE', 'cpu'))

    # Before training, so the learning curve has a baseline. An untrained
    # policy that already solves the task would mean the task is trivial,
    # and no comparison between representations could show anything.
    print('\nevaluating the untrained policy over 10 episodes for a '
          'baseline; expect a few minutes, one line each', flush=True)
    rate0, len0 = evaluate(model, n_episodes=10, label='baseline')
    print(f'before training: success {rate0:.0%}  mean steps {len0:.1f}',
          flush=True)

    def verify_deterministic():
        """Fraction of starts the greedy policy actually solves.

        Only ever called BETWEEN learn() calls. Calling it from inside a
        callback resets and steps the training environment in the middle of
        SB3's rollout collection, which raises "Tried to step environment
        that needs reset" -- or, worse, would silently resume training from
        an observation SB3 never took.
        """
        n = env.unwrapped.n_starts()
        if n <= 0:
            return 1.0
        return sum(rollout(model, start_index=i)[0] for i in range(n)) / n

    # The give-up floor has to sit ABOVE what an untrained policy already
    # scores, or it can never fire. Measured here: pre-training success ran
    # 20-50% across runs, because a random walk finds a 0.8 m goal in a 10 m
    # arena often enough. A fixed 10% floor was therefore inert on exactly
    # the runs it existed to cut short.
    #
    # So it is relative to the baseline this run just measured: abandon only
    # if, halfway through the budget, the agent is no better than it was
    # before training. Capped at 0.5 because rate0 comes from 10 episodes
    # and is noisy, and a high draw must not set a floor that kills a run
    # which is genuinely learning.
    give_up_floor = min(0.5, max(GIVE_UP_BELOW, rate0 + 0.10))
    print(f'give up below {give_up_floor:.0%} after {GIVE_UP_AFTER:,} steps '
          f'(untrained baseline was {rate0:.0%})', flush=True)

    stopper = ConvergenceStopper(threshold=STOP_AT, max_episodes=MAX_EPISODES,
                                 give_up_below=give_up_floor,
                                 give_up_after_steps=GIVE_UP_AFTER)

    # Training in stretches, so the greedy policy can be checked where the
    # environment is idle. The stopper ends a stretch when the trailing
    # training window clears the threshold; if the deterministic rollouts
    # agree, the run is done, and if they do not, training resumes with the
    # bar raised so it does not stop again on the same window.
    #
    # A stop for any other reason -- gave up, or the timestep budget -- is
    # final and breaks out.
    while model.num_timesteps < TIMESTEPS:
        remaining = TIMESTEPS - model.num_timesteps
        model.learn(total_timesteps=remaining, progress_bar=False,
                    callback=stopper, reset_num_timesteps=False)
        if stopper.stop_reason != 'converged':
            break
        checked = verify_deterministic()
        stopper.verifications.append(
            dict(episode=stopper.episodes, steps=int(model.num_timesteps),
                 training=stopper.success_rate, deterministic=checked))
        if checked >= 1.0:
            print(f'\nconverged: {checked:.0%} of starts solved '
                  f'deterministically after {stopper.episodes} episodes / '
                  f'{model.num_timesteps} steps', flush=True)
            break
        print(f'\n[check] {stopper.success_rate:.0%} in training but '
              f'{checked:.0%} deterministic at episode {stopper.episodes} '
              f'-- not converged, continuing', flush=True)
        stopper.stop_reason = None
        stopper.min_episodes_next = stopper.episodes + 100
        # The rollouts above ended their episodes, so Monitor now needs a
        # reset -- and learn(reset_num_timesteps=False) resumes by STEPPING,
        # which raises "Tried to step environment that needs reset".
        # Discarding _last_obs is SB3's own idiom for this (base_class.py
        # does it in set_env): _setup_learn resets the env when _last_obs is
        # None, so training restarts from a clean episode instead of from a
        # state the verification left behind.
        model._last_obs = None

    model.save(MODEL_PATH)
    stop = stopper.summary()
    if stop['stop_reason'] == 'budget_exhausted' and stopper.verifications:
        # Distinguish "never looked converged" from "looked converged
        # repeatedly and never held up" -- very different findings.
        stop['stop_reason'] = 'budget_exhausted_after_failed_checks'
    print(f'\ntraining ended: {stop["stop_reason"]} after {stop["episodes"]} '
          f'episodes ({stop["final_success_rate"]:.0%} success over the last '
          f'{stop["window"]})', flush=True)

    rate1, len1 = evaluate(model, label='final eval')
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
            xml_root = ET.parse(ARENA_XML).getroot()
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
