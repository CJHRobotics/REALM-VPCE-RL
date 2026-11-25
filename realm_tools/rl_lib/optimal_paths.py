"""How few actions the task actually needs, from each start.

The agent's route is only meaningful against the best route available TO
IT, so this searches the same 8 allocentric bearings at the same 0.5 m
step, under the same lidar action mask and the same body buffer. A* over
the agent's own move set, not over free space -- the two differ, and the
difference changes the answer.

The number this returns is the denominator of the extra-step ratio:

    ESR = (steps taken - optimal steps) / optimal steps

0 means the agent matched the best route its action set allows. Negative is
possible: A* respects a body buffer the policy knows nothing about, so a
policy that cuts a corner can beat it. Positive is the interesting case and
the one worth a threshold.

COMPUTED, NOT STORED. An earlier plan was to write these into the arena XML
when the arena is generated. That is wrong twice over. The numbers depend on
the action length, the bearings, the mask range, the body buffer and the
goal radius, so a stored value goes stale the moment any of those move, and
silently -- nothing would catch it. And the arenas under environments/vpce
were generated before any of this existed and carry no such attribute, so
a stored scheme could not be applied to the runs already collected. Five
seconds at startup buys correctness on every arena.
"""

import heapq

import numpy as np

# The action set, as MyRobot defines it: action k moves one action length
# along world bearing 45k. The defaults here are that robot's; they are
# arguments so that a change to the robot is a change to a call site rather
# than a silent disagreement.
N_ACTIONS = 8
ACTION_LENGTH = 0.5
MASK_RANGE = ACTION_LENGTH + 0.25      # get_possible_actions' sector reach
ROBOT_BUFFER = 0.2                     # centre to any surface, hambot is 0.132
GOAL_RADIUS = 0.8                      # HamBot.check_at_goal
RES = 0.05                             # mask field resolution
KEY_RES = 0.05                         # two positions this close are one node
SWEEP_STEP = 0.025                     # how finely a move is collision-checked


def arena_geometry(xml_root):
    """Obstacle segments, half thickness and the enclosing radius."""
    walls = xml_root.findall('wall')
    segs = np.array([[float(w.get(k)) for k in ('x1', 'y1', 'x2', 'y2')]
                     for w in walls]) if walls else np.zeros((0, 4))
    half = float(walls[0].get('width', 0.012)) / 2 if walls else 0.0
    cw = xml_root.find('circular_wall')
    radius = float(cw.get('radius')) if cw is not None else None
    return segs, half, radius


def point_seg_dist(px, py, segs):
    """Distance from each point to each segment: (n_points, n_segs)."""
    if len(segs) == 0:
        return np.full((np.size(px), 1), np.inf)
    p = np.stack([np.ravel(px), np.ravel(py)], axis=1)[:, None, :]
    a, b = segs[None, :, 0:2], segs[None, :, 2:4]
    ab = b - a
    denom = np.maximum(np.sum(ab * ab, axis=2), 1e-12)
    t = np.clip(np.sum((p - a) * ab, axis=2) / denom, 0.0, 1.0)[:, :, None]
    return np.linalg.norm(p - (a + t * ab), axis=2)


def blocked_field(segs, half, radius, res=RES, mask_range=MASK_RANGE):
    """blocked[k, i, j]: action k refused by the lidar mask at that cell.

    A sector test, not a clearance test: action k is refused when anything
    lies within mask_range inside the 45-degree wedge on bearing 45k. A
    clearance test would call a cell blocked outright when an obstacle sits
    0.3 m to one side, and find routes the agent cannot take while missing
    routes it can.
    """
    n = int(np.ceil(2 * radius / res)) + 1
    xs = np.linspace(-radius, radius, n)
    ys = np.linspace(-radius, radius, n)
    blocked = np.zeros((N_ACTIONS, n, n), dtype=bool)

    groups = []
    if len(segs):
        pts = []
        for sg in segs:
            length = float(np.hypot(sg[2] - sg[0], sg[3] - sg[1]))
            t = np.linspace(0.0, 1.0, max(2, int(length / 0.02)))
            pts.append(np.stack([sg[0] + t * (sg[2] - sg[0]),
                                 sg[1] + t * (sg[3] - sg[1])], axis=1))
        # Obstacle samples sit on the centre line but the beam stops at the
        # surface, so they reach half a thickness further than the
        # perimeter, whose samples are already on its inner face.
        groups.append((np.concatenate(pts), mask_range + half))
    wt = np.linspace(0.0, 2 * np.pi, max(8, int(2 * np.pi * radius / 0.02)),
                     endpoint=False)
    groups.append((np.stack([radius * np.cos(wt), radius * np.sin(wt)],
                            axis=1), mask_range))

    for pts, reach in groups:
        half_cells = int(np.ceil(reach / res))
        for px, py in pts:
            ci = int(round((px + radius) / res))
            cj = int(round((py + radius) / res))
            i0, i1 = max(0, ci - half_cells), min(n, ci + half_cells + 1)
            j0, j1 = max(0, cj - half_cells), min(n, cj + half_cells + 1)
            if i0 >= i1 or j0 >= j1:
                continue
            dx = px - xs[i0:i1][:, None]
            dy = py - ys[j0:j1][None, :]
            within = (dx * dx + dy * dy) <= reach ** 2
            if not within.any():
                continue
            k = np.rint(np.arctan2(dy, dx) / (2 * np.pi / N_ACTIONS))
            k = (k.astype(int) % N_ACTIONS) * np.ones_like(within, dtype=int)
            for a in range(N_ACTIONS):
                sel = within & (k == a)
                if sel.any():
                    blocked[a, i0:i1, j0:j1] |= sel
    return blocked, xs, ys


def move_clear(x, y, nx, ny, segs, half, radius, buffer=ROBOT_BUFFER):
    """Does the robot's centre keep `buffer` along the whole swept move?"""
    n = max(2, int(np.ceil(ACTION_LENGTH / SWEEP_STEP)) + 1)
    t = np.linspace(0.0, 1.0, n)
    px, py = x + t * (nx - x), y + t * (ny - y)
    if np.hypot(px, py).max() > radius - buffer:
        return False
    if len(segs) == 0:
        return True
    return bool(point_seg_dist(px, py, segs).min() >= buffer + half)


def astar(start, blocked, segs, half, radius, goal=(0.0, 0.0),
          goal_radius=GOAL_RADIUS, res=RES, cap=400_000):
    """Fewest of the agent's own actions from `start` into the goal radius.

    Nodes are exact positions reached by summing whole actions, so every
    point on the route is one the agent could stand on. Only the visited set
    is discretised, at KEY_RES: the eight bearings generate a set dense in
    the plane and the search would otherwise never close. That makes this a
    small sufficient route rather than a certified minimum, which is the
    honest claim and is quite enough for a denominator.
    """
    bearings = np.arange(N_ACTIONS) * (2 * np.pi / N_ACTIONS)
    n = blocked.shape[1]

    def ij(x, y):
        return (min(max(int(round((x + radius) / res)), 0), n - 1),
                min(max(int(round((y + radius) / res)), 0), n - 1))

    def key(x, y):
        return (int(round(x / KEY_RES)), int(round(y / KEY_RES)))

    def h(x, y):
        return max(0.0, float(np.hypot(x - goal[0], y - goal[1])) - goal_radius)

    sx, sy = float(start[0]), float(start[1])
    k0 = key(sx, sy)
    pos, came, best = {k0: (sx, sy)}, {k0: None}, {k0: 0.0}
    pq, seen = [(h(sx, sy), 0.0, k0)], 0
    while pq:
        _f, g, k = heapq.heappop(pq)
        if g > best.get(k, np.inf):
            continue
        x, y = pos[k]
        if np.hypot(x - goal[0], y - goal[1]) < goal_radius:
            out = []
            while k is not None:
                out.append(pos[k])
                k = came[k]
            return out[::-1]
        seen += 1
        if seen > cap:
            return None
        i, j = ij(x, y)
        for a in range(N_ACTIONS):
            if blocked[a, i, j]:
                continue
            nx = x + ACTION_LENGTH * np.cos(bearings[a])
            ny = y + ACTION_LENGTH * np.sin(bearings[a])
            if not move_clear(x, y, nx, ny, segs, half, radius):
                continue
            nk, ng = key(nx, ny), g + ACTION_LENGTH
            if ng < best.get(nk, np.inf) - 1e-9:
                best[nk], pos[nk], came[nk] = ng, (float(nx), float(ny)), k
                heapq.heappush(pq, (ng + h(nx, ny), ng, nk))
    return None


def optimal_steps(xml_root, starts, goal=(0.0, 0.0),
                  goal_radius=GOAL_RADIUS, verbose=True):
    """{start index: fewest actions}, or None for a start A* cannot solve.

    One blocked field for the arena, then one search per start.
    """
    segs, half, radius = arena_geometry(xml_root)
    if radius is None:
        raise ValueError('optimal_steps needs a circular_wall to bound the '
                         'search; this arena has none')
    blocked, _xs, _ys = blocked_field(segs, half, radius)
    out = {}
    for i, (sx, sy) in enumerate(starts):
        path = astar((sx, sy), blocked, segs, half, radius, goal, goal_radius)
        out[i] = (len(path) - 1) if path else None
        if verbose:
            d = float(np.hypot(sx - goal[0], sy - goal[1]))
            print(f'  start {i}: {d:.2f} m direct, optimal '
                  + (f'{out[i]} actions' if path else 'UNREACHABLE'),
                  flush=True)
    return out
