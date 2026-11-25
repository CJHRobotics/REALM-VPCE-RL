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
    REALM_ARENA_DIR   force the environments/<dir> to look in,
                      instead of finding the arena by name
    REALM_ENSEMBLE    place-field .h5 to observe through
                      (data/vpce/place_fields/<maze>_visual.h5)
    REALM_TIMESTEPS   step ceiling, a backstop only  (250000)
    REALM_MAX_STEPS   steps per episode              (200)
    REALM_ALGO        DQN or PPO                     (DQN)
    REALM_TELEPORT    1 teleports actions, 0 drives  (1)
    REALM_SHAPING     potential-based distance shaping weight (0 = sparse)
    REALM_SEED        random seed                    (0)
    REALM_RUN_TAG     names the output directory     (derived)
    REALM_EVAL_ONLY   1 to load the saved model and only evaluate
    REALM_MAX_EPISODES episodes the agent gets                (1000)
    REALM_ESR_THRESHOLD mean extra-step ratio to stop under    (1.5)
    REALM_ESR_WINDOW   trailing episodes the means run over    (100)
    REALM_MIN_PER_START episodes a start needs to be judged    (10)
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
# The card works now. This is a choice, not a workaround -- which is new,
# and the history is worth keeping because two earlier versions of this
# comment gave reasons that turned out to be false.
#
# It said the card was in exclusive-process compute mode and therefore could
# not give the controller a context. That was a guess; the mode is Default.
# It then said this torch had no kernels for sm_61. That was measured and
# still wrong: 2.5.1+cu121 and 2.14.1+cu126 list exactly the same
# architectures, neither lists sm_61, and on the same card only the first
# allocates. The difference was the CUDA build against a 12.1 driver, and
# setup/requirements.txt pins the one that works. gpu_check.sh confirmed it
# on a 1080 Ti: allocation OK.
#
# What remains is the real reason, and it was always the better one. The
# network is 0.15 ms of a 118 ms environment step, so the GPU buys nothing
# here, while Webots is using that card to render -- which is the entire
# cost of the step and the only thing worth giving it to.
#
# One historical hazard, now retired with the pin: torch 2.14's optimizer
# called torch.accelerator.current_stream() in its graph-capture health
# check, resolving from AVAILABILITY rather than from where the tensors
# lived, so a run that had correctly chosen CPU still died 2048 steps in
# inside optimizer.step(). Hiding the device made that impossible. On 2.5.1
# it does not arise; if this is ever unpinned, it is waiting again.
#
# REALM_DEVICE=cuda opts back in, which is sensible where Webots is not
# holding the card.
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
from realm_tools.simulation_lib.arena_paths import arena_xml
from realm_tools.simulation_lib.webots_exit import quit_on_crash
from realm_tools.rl_lib.agents import make_agent, algo_kwargs
from realm_tools.rl_lib.callbacks import ExtraStepStopper, extra_step_ratio
from realm_tools.rl_lib.optimal_paths import optimal_steps

MAZE = os.environ.get('REALM_MAZE', 'circ_lm8_r10')
# The arena's name says which environments directory it is in, so nothing
# has to be told which one. See realm_tools/simulation_lib/arena_paths.py.
ARENA_XML = arena_xml(MAZE)
ENSEMBLE = os.environ.get(
    'REALM_ENSEMBLE', f'data/vpce/place_fields/{MAZE}_visual.h5')
# A BACKSTOP, not the budget. Episodes decide when the run ends; this
# only stops a pathological run from deciding its own walltime. 1000
# episodes at the 200-step cap is 200k steps, so this sits above the
# worst case and normally never binds.
TIMESTEPS = int(os.environ.get('REALM_TIMESTEPS', 250_000))
MAX_STEPS = int(os.environ.get('REALM_MAX_STEPS', 200))
ALGO = os.environ.get('REALM_ALGO', 'DQN').upper()
TELEPORT = os.environ.get('REALM_TELEPORT', '1') == '1'
SHAPING = float(os.environ.get('REALM_SHAPING', 0.0))
SEED = int(os.environ.get('REALM_SEED', 0))
EVAL_ONLY = os.environ.get('REALM_EVAL_ONLY', '0') == '1'
MAX_EPISODES = int(os.environ.get('REALM_MAX_EPISODES', 1000))
ESR_THRESHOLD = float(os.environ.get('REALM_ESR_THRESHOLD', 1.5))
ESR_WINDOW = int(os.environ.get('REALM_ESR_WINDOW', 100))
MIN_PER_START = int(os.environ.get('REALM_MIN_PER_START', 10))
TAG = os.environ.get(
    'REALM_RUN_TAG',
    f'{MAZE}_{ALGO.lower()}_s{SEED}' + ('_shaped' if SHAPING else ''))

OUT_DIR = f'data/vpce/rl_runs/{TAG}'
os.makedirs(OUT_DIR, exist_ok=True)

print(f'maze={MAZE}  algo={ALGO}  teleport={TELEPORT}  '
      f'shaping={SHAPING}  seed={SEED}', flush=True)
print(f'stop when every start averages under {ESR_THRESHOLD:g} extra steps '
      f'per optimal step\n  over the last {ESR_WINDOW} episodes; '
      f'{MAX_EPISODES} episodes at most, {TIMESTEPS:,} steps as a backstop',
      flush=True)
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
# Before anything that can fail: a crash from here on quits Webots
# rather than leaving it holding this node's GPU for the walltime.
quit_webots = quit_on_crash(robot)
robot.load_environment(ARENA_XML, floor_texture='carpet')

# How few actions the task needs from each start, by A* over the agent's own
# move set. This is the denominator of every extra-step ratio below: the
# stopping rule, the greedy checks between stretches, and the final report
# all measure the policy against it.
#
# Computed here rather than read from the arena file. These numbers depend
# on the action length, the eight bearings, the mask range, the body buffer
# and the goal radius; a stored value would go stale the moment any of them
# moved, and silently. Two or three seconds, once.
import xml.etree.ElementTree as _ET                          # noqa: E402

_arena_root = _ET.parse(ARENA_XML).getroot()
_start_xy = [(float(p.get('x')), float(p.get('y')))
             for p in _arena_root.find('train_start_positions').findall('pos')]
GOAL_XY = robot.maze.get_goal_location()
print('\noptimal routes, A* over the agent\'s own action set:', flush=True)
OPTIMAL = optimal_steps(_arena_root, _start_xy, goal=GOAL_XY)
if not any(OPTIMAL.values()):
    print('ERROR: A* could not reach the goal from any start. The stopping '
          'rule has no\n  denominator, so this run would have no way to '
          'tell a good route from a bad one.', file=sys.stderr, flush=True)
    sys.exit(1)

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
              info_keywords=('is_success', 'start_index'))

MODEL_PATH = f'{OUT_DIR}/model.zip'
if ALGO not in ('DQN', 'PPO'):
    print(f'ERROR: REALM_ALGO must be DQN or PPO, not {ALGO!r}',
          file=sys.stderr, flush=True)
    sys.exit(1)

with open(f'{OUT_DIR}/config.json', 'w') as f:
    json.dump(dict(maze=MAZE, ensemble=ENSEMBLE, n_cells=int(observer.n_cells),
                   channels=list(observer.channels), algo=ALGO,
                   step_ceiling=TIMESTEPS, max_steps=MAX_STEPS,
                   teleport=TELEPORT, shaping=SHAPING, seed=SEED,
                   tag=TAG,
                   # Recorded so a run can be reproduced, and so a sweep
                   # can be checked for having held them constant. The
                   # give-up bound especially: a run it stopped is not
                   # comparable with one that had the full budget.
                   max_episodes=MAX_EPISODES,
                   optimal_steps={str(k): v for k, v in OPTIMAL.items()},
                   esr_threshold=ESR_THRESHOLD, esr_window=ESR_WINDOW,
                   min_per_start=MIN_PER_START,
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

    def greedy_extra_steps():
        """One deterministic rollout per start, scored as extra-step ratios.

        Only ever called BETWEEN learn() calls. From inside a callback this
        resets and steps the training environment in the middle of SB3's
        rollout collection, which raises "Tried to step environment that
        needs reset" -- or worse, would silently resume training from an
        observation SB3 never took.

        A start the greedy policy fails is charged at the episode cap, the
        same way a timeout is charged during training, so the number means
        the same thing in both places.
        """
        out = {}
        for i in range(max(env.unwrapped.n_starts(), 0)):
            if not OPTIMAL.get(i):
                continue
            reached, steps = rollout(model, start_index=i)
            out[i] = (extra_step_ratio(steps if reached else MAX_STEPS,
                                       OPTIMAL[i]), reached, steps)
        return out

    stopper = ExtraStepStopper(OPTIMAL, threshold=ESR_THRESHOLD,
                               window=ESR_WINDOW,
                               min_per_start=MIN_PER_START,
                               max_episodes=MAX_EPISODES)

    # Training in stretches, so the greedy policy can be checked where the
    # environment is idle. The stopper ends a stretch when every start's
    # trailing mean ESR is under the bar; if the greedy rollouts agree the
    # run is done, and if they do not, training resumes with the bar held
    # for another window so it cannot stop again on the same episodes.
    #
    # Any other stop -- the episode cap, the step backstop -- is final.
    while (model.num_timesteps < TIMESTEPS
           and stopper.episodes < MAX_EPISODES):
        model.learn(total_timesteps=TIMESTEPS - model.num_timesteps,
                    progress_bar=False, callback=stopper,
                    reset_num_timesteps=False)
        if stopper.stop_reason != 'converged':
            break

        checked = greedy_extra_steps()
        worst = max((v[0] for v in checked.values()), default=float('inf'))
        stopper.verifications.append(dict(
            episode=stopper.episodes, steps=int(model.num_timesteps),
            training_worst_esr=stopper.worst, greedy_worst_esr=worst,
            greedy={str(k): dict(esr=round(v[0], 4), reached=bool(v[1]),
                                 steps=int(v[2]))
                    for k, v in checked.items()}))

        print(f'\n[check] greedy routes at episode {stopper.episodes}:',
              flush=True)
        for i, (esr, reached, steps) in sorted(checked.items()):
            print(f'    start {i}: {steps} steps against {OPTIMAL[i]} '
                  f'optimal  ESR {esr:+.2f}'
                  + ('' if reached else '   TIMED OUT'), flush=True)

        if worst < ESR_THRESHOLD:
            print(f'\nconverged: worst start {worst:+.2f} extra steps per '
                  f'optimal step, under the {ESR_THRESHOLD:g} bar, after '
                  f'{stopper.episodes} episodes / {model.num_timesteps} '
                  f'steps', flush=True)
            break

        print(f'  worst greedy start is {worst:+.2f}, over the '
              f'{ESR_THRESHOLD:g} bar -- training continues', flush=True)
        stopper.stop_reason = None
        stopper.min_episodes_next = stopper.episodes + ESR_WINDOW

    model.save(MODEL_PATH)
    stop = stopper.summary()
    if (stop['stop_reason'] in ('episode_budget_exhausted',
                                'step_ceiling_reached')
            and stopper.verifications):
        # Distinguish "never looked converged" from "looked converged
        # repeatedly and never held up" -- very different findings.
        stop['stop_reason'] += '_after_failed_checks'
    worst = stop['worst_mean_esr']
    print(f'\ntraining ended: {stop["stop_reason"]} after '
          f'{stop["episodes"]} episodes; worst start averaged '
          + ('n/a' if worst is None else f'{worst:+.2f}')
          + f' extra steps per optimal step over the last {stop["window"]}',
          flush=True)

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
quit_webots(0)
