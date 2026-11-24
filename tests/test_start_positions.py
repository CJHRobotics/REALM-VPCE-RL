"""The arena's start positions must not be solvable without a policy.

The action set is allocentric: action k always moves one action length
along world bearing 45k degrees. So if a start's bearing to the goal is a
multiple of 45, an agent reaches the goal by repeating one action, and a
policy that has learned nothing scores on it.

This is not hypothetical. The first version of circ_lm8_r10's starts was
evenly spaced every 60 degrees, which put two of six exactly on action
rays. Two real runs reported 20 percent and 50 percent success BEFORE
training, against an expected 2/6 = 33 percent. A floor like that sits
under every condition in the sweep and compresses the range that a
comparison between representations has to resolve within.

    python3 tests/test_start_positions.py
"""

import glob
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realm_tools.simulation_lib.environment_parser import parse_environment

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Both environment directories. `vpce` is the place-field paper's sweep of
# empty enclosures; `vpce-rl` holds the obstacle arenas the RL work uses. The
# rule these tests enforce is a property of the ACTION SET, which both share,
# so neither directory is exempt from it.
ARENA_DIRS = [f'{REPO}/simulation/worlds/environments/vpce',
              f'{REPO}/simulation/worlds/environments/vpce-rl']


def arena_files():
    out = []
    for d in ARENA_DIRS:
        out += sorted(glob.glob(f'{d}/*.xml'))
    return out

N_ACTIONS = 8
ACTION_LENGTH = 0.5        # MyRobot default
GOAL_RADIUS = 0.8          # HamBot.check_at_goal
# get_possible_actions clears an action only when the 45-degree sector ahead
# is free to action_length + 0.25, so this is the width of a gap the agent
# will actually enter.
CLEARANCE = 0.75


def straight_line_miss(start, goal):
    """How far a constant-action straight run passes from the goal."""
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    r = math.hypot(dx, dy)
    bearing = math.degrees(math.atan2(dy, dx)) % 360
    step = 360 / N_ACTIONS
    off = min(abs(((bearing - step * k + 180) % 360) - 180)
              for k in range(N_ACTIONS))
    return r, off, r * math.sin(math.radians(off))


def arenas_with_starts():
    for path in arena_files():
        try:
            walls, _cw, _lm, goals, train, _test = parse_environment(path)
        except Exception as exc:                            # noqa: BLE001
            raise AssertionError(f'{os.path.basename(path)} does not parse: '
                                 f'{exc}')
        # Only arenas configured for RL: a single default start at the
        # origin is the unconfigured state, not a claim about the task.
        if goals and len(train) > 1:
            yield os.path.basename(path), goals[0], train, walls


def test_every_arena_xml_parses():
    """A stray double hyphen in a comment invalidates the whole file."""
    n = 0
    for path in arena_files():
        parse_environment(path)
        n += 1
    assert n > 0, 'no arena XML found'
    return f'{n} arena files parse'


def test_no_start_is_solvable_by_a_constant_action():
    checked = 0
    for name, goal, starts, _walls in arenas_with_starts():
        g = (goal['x'], goal['y'])
        for i, p in enumerate(starts):
            r, off, miss = straight_line_miss((p['x'], p['y']), g)
            assert miss >= GOAL_RADIUS, (
                f'{name} start {i} at r={r:.1f} is {off:.1f} deg off an '
                f'action ray, so a straight run passes {miss:.2f} m from a '
                f'goal of radius {GOAL_RADIUS} -- reachable with no policy')
            checked += 1
    assert checked, 'no configured arena to check'
    return f'{checked} starts, none reachable by repeating one action'


def test_starts_are_reachable_at_all():
    """The opposite failure: a start the agent cannot get to the goal from."""
    checked = 0
    for name, goal, starts, _walls in arenas_with_starts():
        g = (goal['x'], goal['y'])
        for i, p in enumerate(starts):
            r = math.hypot(p['x'] - g[0], p['y'] - g[1])
            assert r > GOAL_RADIUS, (
                f'{name} start {i} is {r:.2f} m from the goal, inside the '
                f'{GOAL_RADIUS} m radius -- the episode ends on its first '
                f'check, before the agent acts')
            # Octagonal moves cost at most sqrt(2) times the direct distance.
            steps = math.ceil((r - GOAL_RADIUS) / ACTION_LENGTH)
            assert steps * math.sqrt(2) < 200, (
                f'{name} start {i} needs ~{steps} steps, too close to the '
                f'200-step episode cap')
            checked += 1
    return f'{checked} starts, all outside the goal and inside the budget'


def test_starts_spread_around_the_arena():
    """Starts clustered on one side teach only one approach direction."""
    for name, goal, starts, _walls in arenas_with_starts():
        g = (goal['x'], goal['y'])
        bearings = sorted(math.degrees(math.atan2(p['y'] - g[1],
                                                  p['x'] - g[0])) % 360
                          for p in starts)
        gaps = [(b - a) % 360 for a, b in zip(bearings, bearings[1:] + bearings[:1])]
        assert max(gaps) <= 135, (
            f'{name} leaves a {max(gaps):.0f} degree gap with no start in '
            f'it; the agent never approaches from that side')
    return f'largest bearing gap {max(gaps):.0f} deg'


def point_seg_dist(px, py, w):
    """Distance from a point to one <wall> segment."""
    ax, ay, bx, by = w['x1'], w['y1'], w['x2'], w['y2']
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy
    t = 0.0 if den < 1e-12 else max(0.0, min(1.0, ((px - ax) * dx +
                                                   (py - ay) * dy) / den))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def test_starts_and_goal_are_clear_of_obstacles():
    """An arena with furniture must not park a start inside it.

    A start wedged against an obstacle has most of its action mask closed on
    the first step, and a goal with an obstacle across it can be unreachable
    from one side without anything in the file looking wrong.
    """
    checked = 0
    for name, goal, starts, walls in arenas_with_starts():
        obstacles = [w for w in walls if w.get('wall_type') == 'obstacle']
        if not obstacles:
            continue
        g = (goal['x'], goal['y'])
        places = [('goal', g)] + [(f'start {i}', (p['x'], p['y']))
                                  for i, p in enumerate(starts)]
        for label, (x, y) in places:
            d = min(point_seg_dist(x, y, w) for w in obstacles)
            assert d >= CLEARANCE, (
                f'{name} {label} is {d:.2f} m from an obstacle, inside the '
                f'{CLEARANCE} m the action mask needs to move')
            checked += 1
    return (f'{checked} places clear of obstacles' if checked
            else 'no arena with obstacles')


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    width = max(len(t.__name__) for t in tests)
    failed = 0
    for t in tests:
        try:
            print(f'  PASS  {t.__name__:<{width}}  {t()}')
        except Exception as exc:                            # noqa: BLE001
            failed += 1
            print(f'  FAIL  {t.__name__:<{width}}  '
                  f'{exc.__class__.__name__}: {exc}')
    print(f'\n{len(tests) - failed}/{len(tests)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
