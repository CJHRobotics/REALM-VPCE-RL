"""Exercise the RL environment without Webots.

HamBot subclasses Webots' Supervisor, so it cannot be imported, let alone
constructed, outside a running simulation -- which would normally mean the
task logic could only ever be tested on the cluster, inside a job, after a
render. That is a slow and discouraging place to find out that termination
fires a step late.

WebotsEnv takes an already-connected robot instead of building one, so a
stand-in that implements the handful of methods the env actually uses is
enough to drive the whole task. FakeRobot below is a point robot in a disc
with a goal, and it is deliberately a real little simulator rather than a
mock returning canned values: a mock would agree with whatever the env did,
including the wrong thing.

Run with no arguments (no pytest required):

    python3 tests/test_webots_env.py
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realm_tools.rl_lib import (
    WebotsEnv, RewardConfig, N_ACTIONS, action_masks)

ACTION_LENGTH = 0.5
ARENA_R = 3.0
GOAL = (1.5, 1.5)
GOAL_RADIUS = 0.8          # matches HamBot.check_at_goal


class FakeRobot:
    """A point robot in a disc, with the subset of MyRobot the env uses."""

    def __init__(self, start=(-2.0, -2.0), seed=0):
        self.start = start
        self.x, self.y = start
        self.theta = 0.0
        self.rng = np.random.default_rng(seed)
        self.moves = 0

    # --- the methods WebotsEnv calls -----------------------------------
    def move_to_start(self, mode='training', index=0):
        if index == -1:
            while True:
                p = self.rng.uniform(-ARENA_R, ARENA_R, 2)
                # Reject starts already inside the goal: an episode that is
                # over before it begins would make success rates nonsense.
                if (np.hypot(*p) < ARENA_R - 0.3
                        and math.dist(p, GOAL) > GOAL_RADIUS + 0.5):
                    self.x, self.y = float(p[0]), float(p[1])
                    break
        else:
            self.x, self.y = self.start
        self.theta = 0.0
        return self.get_robot_pose()

    def get_robot_pose(self):
        # Degrees, rounded -- matching HamBot, including its lossiness, so
        # anything that wrongly round-trips heading through here shows up.
        return self.x, self.y, round(math.degrees(self.theta) % 360)

    def get_dist_to_goal(self):
        return math.dist((self.x, self.y), GOAL)

    def check_at_goal(self):
        return self.get_dist_to_goal() < GOAL_RADIUS

    def get_possible_training_action_mask(self):
        mask = np.zeros(N_ACTIONS, dtype=np.float32)
        for a in range(N_ACTIONS):
            nx, ny = self._target(a)
            mask[a] = 1.0 if math.hypot(nx, ny) <= ARENA_R else 0.0
        return mask

    def perform_training_action_teleport(self, a, settle=True):
        self.x, self.y = self._target(a)
        self.theta = math.radians(a * 45)
        self.moves += 1
        # Returns the destination directly, as MyRobot now does: calling
        # get_robot_pose() here would advance the simulation a step -- a
        # whole render -- to learn a position just computed.
        return self.x, self.y, self.theta

    perform_training_action = perform_training_action_teleport

    def _target(self, a):
        th = math.radians(a * 45)
        return self.x + ACTION_LENGTH * math.cos(th), \
               self.y + ACTION_LENGTH * math.sin(th)


class FakeObserver:
    """Place-cell-like activations on a grid of centres over the arena."""

    def __init__(self, n=24, sigma=0.9, seed=0):
        rng = np.random.default_rng(seed)
        self.centers = rng.uniform(-ARENA_R, ARENA_R, size=(n, 2))
        self.sigma = sigma
        self.n_cells = n

    def observe(self, robot):
        d2 = ((self.centers - np.array([robot.x, robot.y])) ** 2).sum(1)
        raw = np.exp(-d2 / (2 * self.sigma ** 2))
        total = raw.sum()
        return (raw / total if total > 0 else raw).astype(np.float32)


# ----------------------------------------------------------------- tests

def _env(**kw):
    kw.setdefault('start_index', 0)
    return WebotsEnv(FakeRobot(), FakeObserver(), **kw)


def test_gymnasium_api():
    """The official checker: spaces, dtypes, reset/step signatures, seeding."""
    from gymnasium.utils.env_checker import check_env
    check_env(_env(), skip_render_check=True)
    return 'gymnasium env_checker passed'


def test_observation_in_space():
    env = _env()
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs), 'reset obs outside space'
    for a in range(N_ACTIONS):
        obs, r, term, trunc, info = env.step(a)
        assert env.observation_space.contains(obs), f'step obs outside space'
        assert np.isfinite(r), 'non-finite reward'
        if term or trunc:
            break
    assert abs(obs.sum() - 1.0) < 1e-5, 'activations should sum to 1'
    return 'observations stay inside the declared space'


def test_truncation_not_termination():
    """Running out of steps must truncate, never terminate.

    An agent that bootstraps will learn a wrong value function if the two
    are swapped: termination says "no future reward exists from here", and
    a timeout does not mean that.
    """
    env = _env(max_steps=3)
    env.reset(seed=2)
    out = [env.step(0) for _ in range(3)]
    term, trunc = out[-1][2], out[-1][3]
    assert trunc and not term, f'expected truncation, got term={term} trunc={trunc}'
    assert not any(t for _, _, t, _, _ in out[:-1]), 'early termination'
    return 'step budget truncates, does not terminate'


def test_goal_terminates_and_pays():
    env = _env(max_steps=200, start_index=0)
    env.reset(seed=3)
    # Walk towards the goal with the greedy action each step.
    for _ in range(100):
        rx, ry = env.robot.x, env.robot.y
        best = min(range(N_ACTIONS),
                   key=lambda a: math.dist(env.robot._target(a), GOAL))
        obs, r, term, trunc, info = env.step(best)
        if term:
            assert info['is_success'], 'terminated without is_success'
            assert r > 0, f'reaching the goal paid {r}'
            assert env.robot.check_at_goal()
            return f'goal reached in {info["steps"]} steps, reward {r:+.3f}'
        assert not trunc, 'truncated before reaching the goal'
    raise AssertionError('never reached the goal')


def test_blocked_action_costs_and_does_not_move():
    env = _env()
    env.reset(seed=4)
    env.robot.x, env.robot.y = ARENA_R - 0.1, 0.0      # against the wall
    mask = env.robot.get_possible_training_action_mask()
    blocked = [a for a in range(N_ACTIONS) if mask[a] == 0]
    assert blocked, 'no action blocked at the wall; test is not testing anything'
    before = (env.robot.x, env.robot.y)
    obs, r, term, trunc, info = env.step(blocked[0])
    assert info['blocked'], 'blocked action not reported'
    assert (env.robot.x, env.robot.y) == before, 'robot moved on a blocked action'
    cfg = RewardConfig()
    assert abs(r - (cfg.step + cfg.blocked)) < 1e-9, f'unexpected reward {r}'
    return 'blocked actions are refused and penalised'


def test_action_mask_helper():
    env = _env()
    env.reset(seed=5)
    m = action_masks(env)
    assert m.dtype == bool and m.shape == (N_ACTIONS,)
    assert m.any(), 'every action masked off'
    return 'action_masks() returns a usable boolean mask'


def test_shaping_is_potential_based():
    """Shaping must be a potential difference, not a distance bonus.

    The check that matters: a closed loop back to the same place should earn
    about nothing from shaping. A raw -distance bonus would pay for sitting
    still near the goal; the potential-based form cannot.
    """
    env = _env(reward=RewardConfig(step=0.0, blocked=0.0, goal=0.0,
                                   shaping=1.0, shaping_gamma=1.0))
    env.reset(seed=6)
    total = 0.0
    for a in (0, 2, 4, 6):                 # right, up, left, down: a square
        _, r, term, trunc, _ = env.step(a)
        total += r
        if term or trunc:
            raise AssertionError('loop ended early; adjust the test start')
    assert abs(total) < 1e-6, f'closed loop earned {total}, should be ~0'
    return 'potential-based shaping sums to zero round a closed loop'


def test_reset_is_reproducible():
    e1, e2 = _env(start_index=-1), _env(start_index=-1)
    o1, _ = e1.reset(seed=7)
    o2, _ = e2.reset(seed=7)
    # FakeRobot seeds its own RNG, so this checks the env does not introduce
    # hidden state between constructions rather than testing gym's seeding.
    assert np.allclose(o1, o2), 'same seed gave different observations'
    return 'reset is reproducible for a fixed seed'


class StartsRobot(FakeRobot):
    """FakeRobot that declares start positions, as a real arena does."""

    STARTS = [(-2.0, -2.0), (2.0, -2.0), (-2.0, 2.0), (0.0, -2.5)]

    def __init__(self, **kw):
        super().__init__(**kw)
        import types
        self.maze = types.SimpleNamespace(
            training_starting_location=[
                types.SimpleNamespace(x=x, y=y, theta=0.0)
                for x, y in self.STARTS],
            testing_start_location=[])

    def move_to_start(self, mode='training', index=0):
        if index is None or not (0 <= index < len(self.STARTS)):
            index = int(self.rng.integers(len(self.STARTS)))
        self.x, self.y = self.STARTS[index]
        self.theta = 0.0
        return self.get_robot_pose()


def test_env_sees_the_arena_starts():
    env = WebotsEnv(StartsRobot(), FakeObserver(), start_index=-1)
    assert env.n_starts() == len(StartsRobot.STARTS)
    env.reset(seed=11)
    # The env must RESOLVE -1 itself, not pass it through: otherwise it does
    # not know which start the episode came from and paths cannot be grouped.
    assert 0 <= env.start_idx < env.n_starts(), f'start_idx={env.start_idx}'
    return f'resolves a concrete start index out of {env.n_starts()}'


def test_path_is_recorded():
    env = WebotsEnv(StartsRobot(), FakeObserver(), max_steps=5, start_index=0)
    env.reset(seed=12)
    for _ in range(5):
        _o, _r, term, trunc, _i = env.step(0)
        if term or trunc:
            break
    assert env.start_idx in env.last_paths, 'no path stored on episode end'
    rec = env.last_paths[env.start_idx]
    xy = rec['xy']
    # One point for the start, one per step taken.
    assert len(xy) == rec['steps'] + 1, f'{len(xy)} points for {rec["steps"]} steps'
    assert abs(xy[0][0] - StartsRobot.STARTS[0][0]) < 1e-9, 'path misses the start'
    assert rec['terminated'] or rec['truncated']
    return f'{len(xy)} points recorded for a {rec["steps"]}-step episode'


def test_last_path_per_start_is_the_latest():
    env = WebotsEnv(StartsRobot(), FakeObserver(), max_steps=3, start_index=0)
    for _ in range(2):
        env.reset(seed=13)
        for _ in range(3):
            if env.step(0)[3] or env.step(0)[2]:
                break
    # Two episodes from start 0 leave exactly one record, the newer one.
    assert list(env.last_paths) == [0], f'keys {list(env.last_paths)}'
    return 'a repeated start keeps only its most recent episode'


def test_path_plot_renders():
    import tempfile
    import xml.etree.ElementTree as ET
    from realm_tools.rl_lib import path_plots

    env = WebotsEnv(StartsRobot(), FakeObserver(), max_steps=40, start_index=0)
    for i in range(len(StartsRobot.STARTS)):
        obs, _ = env.reset(seed=20 + i, options={'start_index': i})
        done = False
        while not done:
            best = min(range(N_ACTIONS),
                       key=lambda a: math.dist(env.robot._target(a), GOAL))
            _o, _r, term, trunc, _in = env.step(best)
            done = term or trunc
    assert len(env.last_paths) == len(StartsRobot.STARTS)

    xml_root = ET.fromstring(
        f'<world><circular_wall radius="{ARENA_R}"/></world>')
    with tempfile.TemporaryDirectory() as d:
        png, summary = path_plots.plot_paths(
            dict(env.last_paths), xml_root, os.path.join(d, 'p.png'),
            goal=GOAL, goal_radius=GOAL_RADIUS)
        assert png and os.path.exists(png), 'no path figure written'
        assert summary['starts'] == len(StartsRobot.STARTS)
        assert summary['reached'] >= 1, 'greedy walk reached nothing'
        assert path_plots.summarise(summary)
    return (f'{summary["reached"]}/{summary["starts"]} panels drawn '
            f'from recorded paths')


class CountingRobot(StartsRobot):
    """Counts the operations that cost a Webots simulation step.

    In Webots the camera and lidar are enabled at the basic timestep, so
    EVERY simulation step is a full render. Anything that steps merely to
    read a value the supervisor already holds is a render thrown away, and
    a training step that does it four extra times is 30% slower for
    nothing. These are easy to reintroduce by accident, hence the test.
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        from collections import Counter
        self.counts = Counter()

    def get_robot_pose(self):          # STEPS in the real robot
        self.counts['stepping_pose'] += 1
        return super().get_robot_pose()

    def read_pose(self):               # free: supervisor fields only
        self.counts['free_pose'] += 1
        return self.x, self.y, self.theta

    def get_possible_training_action_mask(self):   # STEPS (reads lidar)
        self.counts['mask'] += 1
        return super().get_possible_training_action_mask()

    def perform_training_action_teleport(self, a, settle=True):
        self.counts['action'] += 1
        self.counts['settle'] += int(settle)
        return super().perform_training_action_teleport(a)


class CountingObserver(FakeObserver):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.n_observe = 0

    def observe(self, robot):
        self.n_observe += 1
        return super().observe(robot)


def test_step_does_no_redundant_simulation_steps():
    robot = CountingRobot()
    env = WebotsEnv(robot, CountingObserver(), max_steps=50, start_index=0)
    env.reset(seed=30)
    robot.counts.clear()
    env.step(0)

    c = robot.counts
    assert c['mask'] == 1, (
        f"action mask computed {c['mask']} times in one step; it reads the "
        f"lidar and so costs a render each time")
    assert c['stepping_pose'] == 0, (
        f"{c['stepping_pose']} stepping pose reads; read_pose is free and "
        f"should be used wherever the value is only recorded")
    assert c['settle'] == 0, (
        'the action teleport settled, but the observer steps before reading '
        'the camera, so that render is wasted')
    assert c['action'] == 1
    return (f"1 mask, {c['free_pose']} free pose reads, 0 stepping reads, "
            f"0 redundant settles")


def test_learn_then_verify_loop_survives(_budget=1200):
    """The real training structure: learn, verify, resume.

    This is a regression test for a crash that only appears with SB3 in the
    loop. Running deterministic rollouts from inside a callback resets and
    steps the training environment in the middle of SB3's rollout
    collection; Monitor then raises "Tried to step environment that needs
    reset" when training resumes. Verification between learn() calls, where
    the environment is idle, is safe -- and this drives exactly that, so a
    future move of the check back into the callback fails here instead of
    four hours into a cluster job.
    """
    from stable_baselines3.common.monitor import Monitor
    from realm_tools.rl_lib.agents import make_agent
    from realm_tools.rl_lib.callbacks import ConvergenceStopper

    env = Monitor(WebotsEnv(StartsRobot(), FakeObserver(n=16), max_steps=20,
                            start_index=-1),
                  info_keywords=('is_success',))
    model = make_agent('DQN', env, seed=0, verbose=0,
                       learning_starts=10, buffer_size=1000)

    def rollout(i):
        obs, _ = env.reset(options={'start_index': i})
        done = term = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _r, term, trunc, _in = env.step(int(a))
            done = term or trunc
        return bool(term)

    stopper = ConvergenceStopper(window=5, threshold=0.8, min_episodes=5,
                                 verbose=0)
    checks = 0
    while model.num_timesteps < _budget:
        model.learn(total_timesteps=_budget - model.num_timesteps,
                    callback=stopper, reset_num_timesteps=False)
        if stopper.stop_reason != 'converged':
            break
        # The call that used to corrupt the env when made from inside the
        # callback. Training must survive it and carry on.
        n = env.unwrapped.n_starts()
        _rate = sum(rollout(i) for i in range(n)) / n
        checks += 1
        stopper.stop_reason = None
        stopper.min_episodes_next = stopper.episodes + 5
        # SB3's own idiom for "the env moved under you": discarding
        # _last_obs makes _setup_learn reset it on the next learn().
        model._last_obs = None
        if checks > 4:
            break

    assert model.num_timesteps > 0, 'no training happened'
    return (f'{model.num_timesteps} steps over {checks} verification '
            f'round-trip(s), no reset error')


def test_controller_hides_gpu_before_importing_torch():
    """The CUDA_VISIBLE_DEVICES line must stay above every torch import.

    Once torch has seen a device, setting the variable does nothing. The
    line is therefore order-dependent in a way nothing else in the file is,
    and an import added above it would silently restore a bug that killed a
    run 2048 steps in: torch 2.14's optimizer calls
    torch.accelerator.current_stream() unconditionally, which raises on a
    visible-but-unusable GPU even when every tensor is on the CPU.
    """
    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'simulation', 'controllers',
        'rl_agent', 'rl_agent.py')
    lines = open(path, encoding='utf-8').read().splitlines()

    guard = next((i for i, l in enumerate(lines)
                  if "os.environ['CUDA_VISIBLE_DEVICES']" in l), None)
    assert guard is not None, 'the CUDA_VISIBLE_DEVICES guard is gone'

    # Anything that reaches torch: SB3 and the rl_lib modules that import it.
    risky = [(i, l) for i, l in enumerate(lines)
             if i < guard
             and (l.startswith('import ') or l.startswith('from '))
             and any(m in l for m in ('torch', 'stable_baselines3',
                                      'realm_tools', 'gymnasium'))]
    assert not risky, (
        f'these imports precede the GPU guard at line {guard + 1}, so it '
        f'has no effect: {[(i + 1, l) for i, l in risky]}')
    return f'guard at line {guard + 1}, above every torch-reaching import'


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    width = max(len(t.__name__) for t in tests)
    failed = 0
    for t in tests:
        try:
            msg = t()
            print(f'  PASS  {t.__name__:<{width}}  {msg}')
        except Exception as exc:                            # noqa: BLE001
            failed += 1
            print(f'  FAIL  {t.__name__:<{width}}  '
                  f'{exc.__class__.__name__}: {exc}')
    print(f'\n{len(tests) - failed}/{len(tests)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
