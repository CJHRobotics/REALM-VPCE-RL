"""Generate the obstacle arena the RL agent is trained and tested in.

    circ_lm10_r10_o20    disc, r = 10, 10 landmarks, 20 internal obstacles

This directory is deliberately separate from `environments/vpce/`. Those nine
arenas are the place-field paper's design -- a controlled sweep over size and
geometry with nothing inside the enclosure -- and their datasets are the
measurement that paper rests on. An arena with obstacles in it answers a
different question and must not be mistaken for one of those, so it lives
under its own directory and its own name.

WHAT THE OBSTACLES ARE FOR. A disc with a goal at its centre is solvable by
distance to the goal alone: the policy "move whichever way reduces it" works
from everywhere, and a representation that encodes nothing but a bearing
would score as well as one encoding place. Scattered obstacles break that.
Stretches of the arena now require moving ACROSS or AWAY from the goal to get
around something, which only a policy that knows where it is can do. That is
the property the VPCE comparison needs the arena to have.

The geometry follows the maze series in the figure this was specified from
(M0, M1, M2, M108, M300, M600): short straight segments at random
orientations scattered through the interior, with the count as the difficulty
knob. Twenty segments puts this between M2 and M108 -- sparse enough that
most of the arena is open, dense enough that few start-goal pairs have a
clear straight line.

WHAT IS CHECKED BEFORE THE FILE IS WRITTEN.

Reachability is not assumed. The agent's action mask (`get_possible_actions`)
clears an action only when the whole 45-degree sector ahead of it is free to
0.75 m, so a gap the robot would physically fit through is not necessarily a
gap it will ever choose to enter. Obstacles are therefore sampled under a
minimum separation, and the arena is then flood-filled at 5 cm resolution with
every obstacle and the wall inflated by CLEAR = 0.8 m. An arena is accepted
only if the goal and all four starts are in one connected component of that
inflated free space. A layout that fails is discarded whole and resampled --
not patched, which would bias the obstacle distribution toward whatever the
patch does.

START BEARINGS ARE OFF THE ACTION RAYS, for the reason recorded in
`vpce/circ_lm8_r10.xml`: the action set is allocentric, so a start whose
bearing to the goal is a multiple of 45 degrees is solvable by repeating one
action with no policy at all, and that puts a success floor under every
condition. Four starts at 22.5 + 90k degrees sit on the midpoints between
action rays, as far from one as a start can be.

    python simulation/worlds/environments/vpce-rl/make_rl_env.py
    REALM_ARENA_SEED=12 python .../make_rl_env.py     # another layout
"""

import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '..', '..', '..', '..')))
from realm_tools.rl_lib import optimal_paths as OP          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

NAME        = 'circ_lm10_r10_o20'
RADIUS      = 10.0
N_LANDMARKS = 10
N_OBSTACLES = 20
SEED        = int(os.environ.get('REALM_ARENA_SEED', 7))

# Matching vpce/make_envs.py, so an arena from either directory is built to
# the same physical conventions and the two are comparable.
PANEL       = 0.75      # fixed physical cue size, m
WALL_H      = 0.5
WALL_THICK  = 0.5       # circular wall
SUBDIV      = 128
HALF_THICK  = 0.015     # half a panel's own thickness
MARGIN      = 0.2       # collection keep-out from the perimeter wall, m
N_TARGET    = 30147     # positions per arena, held constant

# Obstacles. Thin and wall-height, so they occlude and read on the lidar
# exactly as the perimeter does -- the lidar sits 0.185 m off the floor and
# the camera at 0.188 m, both well under 0.5 m.
OBS_H       = 0.5
# Thick enough to be a barrier rather than a sheet. At the PROTO's 0.012 m
# default an obstacle is a razor blade: seen edge-on from across the arena it
# covers a fifth of a pixel of a 224 px camera at 45 degrees, so the camera
# that the whole representation is built on could not see it at all from half
# the bearings. At 0.1 m it covers about 3 px at 10 m and is visible from
# every side, which is the point of putting it there.
OBS_W       = 0.1
# Every distance below is measured to an obstacle's CENTRE LINE, because that
# is what the XML stores and what point_seg_dist returns. Physical clearance
# is to its SURFACE, so half the thickness is added wherever the two meet.
# At 0.012 m that difference was 6 mm and ignoring it was harmless; at 0.1 m
# it is 50 mm, which is a quarter of the robot's buffer.
OBS_HALF    = OBS_W / 2
# Obstacles are textured, the perimeter is flat black. They are different
# kinds of thing -- one bounds the arena, the other furnishes it -- and the
# agent sees through a camera, so telling them apart is something the
# representation can actually learn from. The wall PROTO takes a texture URL
# relative to the world file, as the landmark flags do.
OBS_TEXTURE = '../protos/world_objects/textures/wall_textures/caution_tape.jpg'
OBS_LEN     = (1.5, 3.0)    # segment length range, m
OBS_MARGIN  = 0.3           # collection keep-out from an obstacle, m

# Free space the agent needs. get_possible_actions refuses an action unless
# the 45-degree sector ahead is clear to action_length + 0.25 = 0.75 m, so
# 0.8 m is the honest width of a corridor it will actually use.
CLEAR       = 0.8
GAP         = 2 * CLEAR     # minimum separation between two obstacles
WALL_GAP    = 1.0           # minimum gap between an obstacle tip and the wall

GOAL        = (0.0, 0.0)
GOAL_R      = 0.8           # hambot.check_at_goal: distance_to_goal < 0.8
START_BEAR  = np.radians([22.5, 112.5, 202.5, 292.5])
START_RAD   = [8.0, 5.0, 8.0, 5.0]
START_KEEP  = 1.2           # obstacle keep-out around a start, m
GOAL_KEEP   = 2.0           # obstacle keep-out around the goal, m

RES         = 0.05          # flood-fill resolution, m

COLORS = [(1.00, 0.00, 0.00), (0.00, 1.00, 0.00), (0.00, 0.00, 1.00),
          (1.00, 1.00, 0.00), (0.00, 1.00, 1.00), (1.00, 0.50, 0.00),
          (0.50, 0.00, 0.50), (0.00, 0.50, 0.50), (1.00, 0.00, 1.00),
          (0.50, 1.00, 0.00)]


def fmt(v, prec):
    """Leading-space alignment, matching the existing world files."""
    s = f'{v:.{prec}f}'
    if s.startswith('-') and float(s) == 0.0:
        s = s[1:]
    return s if s.startswith('-') else ' ' + s


def landmark_angles(n):
    """Bearings anchored at 90 degrees, so no panel sits on a camera seam.

    The robot captures at headings 0, 45 ... 315 with a 45-degree camera, so
    the views partition the circle at 22.5 + 45k. Ten panels at 36-degree
    spacing cannot all sit mid-view the way eight can: four of them land 4.5
    degrees from a seam. That is still clear of it -- a 0.75 m panel seen
    from across a 10 m disc subtends 4.3 degrees, so 2.15 either side of
    centre -- but the margin is 2.35 degrees rather than the 22.5 that eight
    panels enjoy, and a panel is therefore wholly inside one view only
    because it is small and the arena is large.
    """
    return (np.pi / 2.0) + np.arange(n) * (2.0 * np.pi / n)


def wall_mount_radius(radius, width=PANEL):
    """Centre radius that puts a panel's outer corners on the wall's face."""
    return float(np.sqrt(radius ** 2 - (width / 2) ** 2) - HALF_THICK)


def starts():
    """The four episode starts, as (x, y, theta) facing the centre."""
    out = []
    for bearing, r in zip(START_BEAR, START_RAD):
        x, y = r * np.cos(bearing), r * np.sin(bearing)
        out.append((float(x), float(y), float(bearing - np.pi)))
    return out


# ------------------------------------------------------------- geometry

def point_seg_dist(px, py, segs):
    """Distance from each point to each segment: (n_points, n_segs)."""
    p = np.stack([np.ravel(px), np.ravel(py)], axis=1)[:, None, :]   # (P,1,2)
    a = segs[None, :, 0:2]                                           # (1,S,2)
    b = segs[None, :, 2:4]
    ab = b - a
    denom = np.sum(ab * ab, axis=2)
    denom = np.where(denom < 1e-12, 1e-12, denom)
    t = np.sum((p - a) * ab, axis=2) / denom
    t = np.clip(t, 0.0, 1.0)[:, :, None]
    closest = a + t * ab
    return np.linalg.norm(p - closest, axis=2)


def seg_seg_dist(s1, s2, n=41):
    """Distance between two segments, by sampling one against the other."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    p1 = s1[0:2] + t * (s1[2:4] - s1[0:2])
    p2 = s2[0:2] + t * (s2[2:4] - s2[0:2])
    d1 = point_seg_dist(p1[:, 0], p1[:, 1], s2[None, :]).min()
    d2 = point_seg_dist(p2[:, 0], p2[:, 1], s1[None, :]).min()
    return float(min(d1, d2))


def sample_obstacles(rng, start_xy):
    """One candidate layout. Returns (N_OBSTACLES, 4) of x1,y1,x2,y2."""
    segs = []
    guard = 0
    while len(segs) < N_OBSTACLES:
        guard += 1
        if guard > 20000:
            return None
        length = rng.uniform(*OBS_LEN)
        ang = rng.uniform(0.0, np.pi)
        # Centre sampled uniformly over the disc, not uniformly in (r, theta),
        # which would pile obstacles into the middle.
        rad = (RADIUS - WALL_GAP - length / 2) * np.sqrt(rng.uniform(0.0, 1.0))
        phi = rng.uniform(0.0, 2 * np.pi)
        cx, cy = rad * np.cos(phi), rad * np.sin(phi)
        dx, dy = (length / 2) * np.cos(ang), (length / 2) * np.sin(ang)
        cand = np.array([cx - dx, cy - dy, cx + dx, cy + dy])

        # Inside the wall, with a passable gap left at each tip.
        if max(np.hypot(cand[0], cand[1]),
               np.hypot(cand[2], cand[3])) > RADIUS - WALL_GAP - OBS_HALF:
            continue
        # Clear of the goal, so it stays approachable from several bearings.
        if point_seg_dist(np.array([GOAL[0]]), np.array([GOAL[1]]),
                          cand[None, :])[0, 0] < GOAL_KEEP + OBS_HALF:
            continue
        # Clear of every start, so no episode begins wedged against one.
        sx = np.array([s[0] for s in start_xy])
        sy = np.array([s[1] for s in start_xy])
        if point_seg_dist(sx, sy, cand[None, :]).min() < START_KEEP + OBS_HALF:
            continue
        # Clear of each other, so no pair forms a gap too narrow to use.
        if any(seg_seg_dist(cand, s) < GAP + OBS_W for s in segs):
            continue
        segs.append(cand)
    return np.array(segs)


def occupancy(segs):
    """Free-space mask at RES, with obstacles and the wall inflated by CLEAR.

    Returns (free, xs, ys). `free` is True where the robot's centre can be
    with CLEAR metres to every obstacle and to the wall.
    """
    n = int(np.ceil(2 * RADIUS / RES)) + 1
    xs = np.linspace(-RADIUS, RADIUS, n)
    ys = np.linspace(-RADIUS, RADIUS, n)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    inside = np.hypot(X, Y) <= RADIUS - CLEAR
    d = point_seg_dist(X, Y, segs).min(axis=1).reshape(X.shape)
    return inside & (d >= CLEAR + OBS_HALF), xs, ys


def flood(free, seed_ij):
    """8-connected component of `free` containing seed_ij."""
    seen = np.zeros_like(free, dtype=bool)
    if not free[seed_ij]:
        return seen
    q = deque([seed_ij])
    seen[seed_ij] = True
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
            (0, 1), (1, -1), (1, 0), (1, 1)]
    ni, nj = free.shape
    while q:
        i, j = q.popleft()
        for di, dj in nbrs:
            a, b = i + di, j + dj
            if 0 <= a < ni and 0 <= b < nj and free[a, b] and not seen[a, b]:
                seen[a, b] = True
                q.append((a, b))
    return seen


def geodesic(free, xs, ys, src):
    """Shortest path length through the inflated free space, from src.

    Dijkstra on the 8-connected grid, so the number it returns is the route
    an agent with CLEAR metres of elbow room would actually have to walk.
    Against the straight-line distance it says how much the obstacles cost,
    which is the whole reason they are here: a detour factor of 1.0 means
    the layout is decorative.
    """
    import heapq
    ni, nj = free.shape
    dist = np.full(free.shape, np.inf)
    si = cell(xs, ys, *src)
    if not free[si]:
        return dist
    dist[si] = 0.0
    pq = [(0.0, si)]
    step = {(-1, 0): RES, (1, 0): RES, (0, -1): RES, (0, 1): RES,
            (-1, -1): RES * 1.41421356, (-1, 1): RES * 1.41421356,
            (1, -1): RES * 1.41421356, (1, 1): RES * 1.41421356}
    while pq:
        d, (i, j) = heapq.heappop(pq)
        if d > dist[i, j]:
            continue
        for (di, dj), w in step.items():
            a, b = i + di, j + dj
            if 0 <= a < ni and 0 <= b < nj and free[a, b] and d + w < dist[a, b]:
                dist[a, b] = d + w
                heapq.heappush(pq, (d + w, (a, b)))
    return dist


def straight_blocked(x, y, segs):
    """Does the straight line from (x, y) to the goal pass an obstacle?"""
    t = np.linspace(0.0, 1.0, 400)
    px = x + t * (GOAL[0] - x)
    py = y + t * (GOAL[1] - y)
    return bool(point_seg_dist(px, py, segs).min() < CLEAR + OBS_HALF)


def cell(xs, ys, x, y):
    return int(np.argmin(np.abs(xs - x))), int(np.argmin(np.abs(ys - y)))


def check_layout(segs, start_xy):
    """Goal and every start in one component of the inflated free space."""
    free, xs, ys = occupancy(segs)
    gi = cell(xs, ys, *GOAL)
    if not free[gi]:
        return None
    comp = flood(free, gi)
    for (x, y, _th) in start_xy:
        si = cell(xs, ys, x, y)
        if not comp[si]:
            return None
    return free, comp, xs, ys


# ----------------------------------------------------------------- output

def write_xml(segs, start_xy, path):
    a_deg = np.round(np.degrees(landmark_angles(N_LANDMARKS)), 1).tolist()
    cover = 100 * N_LANDMARKS * PANEL / (2 * np.pi * RADIUS)
    lines = [
        '<?xml version="1.0" encoding="us-ascii"?>\n\n',
        f'<!-- Circular arena, radius {RADIUS:g} m, area {np.pi*RADIUS**2:.2f} m^2, '
        f'with {N_LANDMARKS} landmarks of {PANEL} m\n'
        f'     and {N_OBSTACLES} internal obstacles.\n\n'
        f'     Generated by make_rl_env.py at seed {SEED}. DO NOT EDIT BY HAND:\n'
        f'     the obstacle layout was accepted only after a flood fill proved\n'
        f'     the goal reachable from all four starts through free space\n'
        f'     {CLEAR} m wide, and moving a segment by hand voids that proof.\n\n'
        f'     XML `radius` is the inner walkable radius; the wall material\n'
        f'     extends outward by `thickness`. Panels sit flush on the inner\n'
        f'     face. Interlandmark spacing along the wall is '
        f'{2*np.pi*RADIUS/N_LANDMARKS:.2f} m; the panels\n'
        f'     cover {cover:.1f}% of the circumference. Landmark bearings are\n'
        f'     anchored at 90 deg: {a_deg} -->\n\n',
        '<world>\n',
        f'\t<circular_wall radius="{RADIUS:g}" height="{WALL_H}" '
        f'thickness="{WALL_THICK}" subdivision="{SUBDIV}"/>\n\n',
    ]
    r_mount = wall_mount_radius(RADIUS)
    for k, a in enumerate(landmark_angles(N_LANDMARKS)):
        x, y, th = r_mount * np.cos(a), r_mount * np.sin(a), a - np.pi
        r, g, b = COLORS[k % len(COLORS)]
        lines.append(
            f'    <landmark type="panel" x="{fmt(x,4)}" y="{fmt(y,4)}" '
            f'theta="{fmt(th,4)}" height="{PANEL}" width="{PANEL}"\n'
            f'              texture="../protos/world_objects/textures/flags/flag_{k}.png"\n'
            f'              red="{r:.2f}" green="{g:.2f}" blue="{b:.2f}"/>\n')

    lines.append(
        f'\n    <!-- {N_OBSTACLES} obstacles, {OBS_LEN[0]}-{OBS_LEN[1]} m segments at random\n'
        f'         orientations, centres uniform over the disc. Every pair is at\n'
        f'         least {GAP} m apart and every tip at least {WALL_GAP} m from the wall,\n'
        f'         so no two of them form a gap narrower than the {CLEAR} m the action\n'
        f'         mask needs to enter it. They are textured rather than flat black,\n'
        f'         so the camera can tell furniture from the perimeter. -->\n')
    for s in segs:
        lines.append(
            f'    <wall x1="{fmt(s[0],4)}" y1="{fmt(s[1],4)}" '
            f'x2="{fmt(s[2],4)}" y2="{fmt(s[3],4)}"\n'
            f'          height="{OBS_H}" width="{OBS_W}"\n'
            f'          texture="{OBS_TEXTURE}"/>\n')

    lines.append(
        '\n    <!-- Four starts at 22.5 + 90k degrees, alternating r = 8 and r = 5 so\n'
        '         they differ in distance to the goal as well as in bearing. The\n'
        '         bearings are midway between action rays: the action set is\n'
        '         allocentric, so a start at a multiple of 45 degrees is solvable\n'
        '         by repeating one action and puts a success floor under every\n'
        '         condition. theta is cosmetic, since the VPCE observation spans\n'
        '         all eight headings; it faces the centre to read better in the\n'
        '         path plots. -->\n'
        '    <train_start_positions>\n')
    for (x, y, th), bear, rad in zip(start_xy, np.degrees(START_BEAR), START_RAD):
        lines.append(f'        <pos x="{fmt(x,4)}" y="{fmt(y,4)}" '
                     f'theta="{fmt(th,4)}"/>   <!-- {bear:5.1f} deg, r = {rad:g} m -->\n')
    lines.append('    </train_start_positions>\n\n'
                 '    <test_start_positions>\n'
                 '        <pos x="0.0" y="0.0" theta="0.0"/>\n'
                 '    </test_start_positions>\n\n'
                 f'    <goal id="0" x="{GOAL[0]:.1f}" y="{GOAL[1]:.1f}"/>\n'
                 '</world>\n')
    text = ''.join(lines)
    # `--` is illegal inside an XML comment, and every one of these files is
    # mostly comment. ElementTree rejects the whole file for it, pointing at a
    # line number rather than at the dashes, so check it here where the
    # message can say what is actually wrong.
    for n, line in enumerate(text.splitlines(), start=1):
        body = line.split('-->')[0]
        if '--' in body.replace('<!--', ''):
            raise ValueError(f'line {n} has a double hyphen inside an XML '
                             f'comment, which no parser will accept:\n  {line}')
    with open(path, 'w') as f:
        f.write(text)


def write_positions(segs, path):
    """Collection grid: ~N_TARGET positions over the free floor.

    Spacing is solved against the area actually walkable -- the disc less the
    keep-out around each obstacle -- rather than the disc, so this arena
    carries the same number of samples as the vpce arenas do and field count
    stays uncoupled from how much furniture is in the room.
    """
    usable = RADIUS - MARGIN
    probe = np.linspace(-usable, usable, 601)
    PX, PY = np.meshgrid(probe, probe, indexing='ij')
    ok = np.hypot(PX, PY) <= usable
    ok &= (point_seg_dist(PX, PY, segs).min(axis=1).reshape(PX.shape)
           >= OBS_MARGIN + OBS_HALF)
    cell_area = (probe[1] - probe[0]) ** 2
    free_area = float(ok.sum() * cell_area)

    spacing = float(np.sqrt(free_area / N_TARGET))
    n = int(np.floor(usable / spacing + 1e-9))
    ax = np.round(np.arange(-n, n + 1) * spacing, 5)
    X, Y = np.meshgrid(ax, ax, indexing='ij')
    keep = np.hypot(X, Y) <= usable + 1e-9
    keep &= (point_seg_dist(X, Y, segs).min(axis=1).reshape(X.shape)
             >= OBS_MARGIN + OBS_HALF)
    xs, ys = X[keep], Y[keep]
    with open(path, 'w') as f:
        f.write('x,y,theta\n')
        for x, y in zip(xs, ys):
            f.write(f'{x:g},{y:g},0.0\n')
    return (xs, ys), spacing, free_area



# --------------------------------------------------------------------- A*
#
# The search itself lives in realm_tools/rl_lib/optimal_paths.py, because
# the training job needs the same numbers: the extra-step ratio it stops on
# is measured against exactly this optimum, over exactly this action set.
# Two copies would drift, and the drift would be invisible -- a generator
# that accepts a layout on one A* while the stopper judges the policy on
# another produces a threshold that means nothing.
#
# The constants below are re-exported so this file still reads as the
# record of what the arena was built under.
ACTION_LENGTH  = OP.ACTION_LENGTH
N_ACTIONS      = OP.N_ACTIONS
MASK_RANGE     = OP.MASK_RANGE
ROBOT_BUFFER   = OP.ROBOT_BUFFER
SWEEP_STEP     = OP.SWEEP_STEP


def route_clearance(path, segs):
    """Smallest distance from the robot's centre to a surface, on a route."""
    worst = np.inf
    for a, b in zip(path, path[1:]):
        n = max(2, int(np.ceil(ACTION_LENGTH / SWEEP_STEP)) + 1)
        t = np.linspace(0.0, 1.0, n)
        px = a[0] + t * (b[0] - a[0])
        py = a[1] + t * (b[1] - a[1])
        worst = min(worst,
                    float(point_seg_dist(px, py, segs).min()) - OBS_HALF,
                    float(RADIUS - np.hypot(px, py).max()))
    return worst


def straight_blocked(x, y, segs):
    """Does the straight line from (x, y) to the goal pass an obstacle?"""
    t = np.linspace(0.0, 1.0, 400)
    px = x + t * (GOAL[0] - x)
    py = y + t * (GOAL[1] - y)
    return bool(point_seg_dist(px, py, segs).min() < CLEAR + OBS_HALF)


def cell(xs, ys, x, y):
    return int(np.argmin(np.abs(xs - x))), int(np.argmin(np.abs(ys - y)))


def _draw_arena(ax, segs, start_xy, faint=False):
    """Walls, obstacles, landmarks, goal, starts. Nothing else."""
    from matplotlib.patches import Circle
    alpha = 0.35 if faint else 1.0

    ax.add_patch(Circle((0, 0), RADIUS, fill=False, lw=2.5, color='0.15',
                        alpha=alpha, zorder=2))

    # Obstacles, drawn as the caution tape they are textured with, so they
    # read as a different kind of surface from the perimeter at a glance.
    for sg in segs:
        ax.plot([sg[0], sg[2]], [sg[1], sg[3]], lw=4.5, color='#e8b100',
                solid_capstyle='round', alpha=alpha, zorder=3)
        ax.plot([sg[0], sg[2]], [sg[1], sg[3]], lw=4.5, color='0.1',
                solid_capstyle='butt', alpha=alpha, zorder=4,
                dashes=(0.9, 1.1))

    r_mount = wall_mount_radius(RADIUS)
    for k, a in enumerate(landmark_angles(N_LANDMARKS)):
        x, y = r_mount * np.cos(a), r_mount * np.sin(a)
        tx, ty = -np.sin(a), np.cos(a)
        ax.plot([x - PANEL / 2 * tx, x + PANEL / 2 * tx],
                [y - PANEL / 2 * ty, y + PANEL / 2 * ty],
                lw=6, color=COLORS[k], solid_capstyle='butt',
                alpha=alpha, zorder=5)
        ax.text(10.95 * np.cos(a), 10.95 * np.sin(a), str(k), fontsize=8,
                ha='center', va='center', color='0.4', alpha=alpha)

    ax.add_patch(Circle(GOAL, GOAL_R, fill=False, lw=1.0, ls='--',
                        color='crimson', alpha=alpha, zorder=6))
    ax.plot(*GOAL, 'o', ms=11, color='crimson', alpha=alpha, zorder=7)
    for i, (x, y, _th) in enumerate(start_xy, start=1):
        ax.plot(x, y, 'o', ms=11, color='darkgreen', alpha=alpha, zorder=7)
        ax.annotate(str(i), (x, y), textcoords='offset points',
                    xytext=(11, 8), fontsize=12, fontweight='bold',
                    color='darkgreen', alpha=alpha)

    ax.set_aspect('equal')
    ax.set_xlim(-RADIUS - 1.4, RADIUS + 1.4)
    ax.set_ylim(-RADIUS - 1.4, RADIUS + 1.4)
    ax.set_xticks(range(-10, 11, 5))
    ax.set_yticks(range(-10, 11, 5))
    ax.set_xlabel('x (m)')


def plot(segs, start_xy, paths, path_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 7))

    _draw_arena(ax1, segs, start_xy)
    ax1.set_title('Environment', fontsize=12)
    ax1.set_ylabel('y (m)')

    _draw_arena(ax2, segs, start_xy, faint=True)
    route = ['#1f77b4', '#d62728', '#7b3fa0', '#0d7d5a']
    for i, path in enumerate(paths):
        if path is None:
            continue
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        ax2.plot(px, py, '-', lw=2.0, color=route[i % len(route)], zorder=9,
                 label=f'start {i + 1}: {len(path) - 1} actions, '
                       f'{(len(path) - 1) * ACTION_LENGTH:.1f} m')
        ax2.plot(px, py, 'o', ms=2.6, color=route[i % len(route)], zorder=10)
    ax2.legend(loc='lower left', fontsize=8.5, framealpha=0.9)
    ax2.set_title('Shortest path estimated by A*', fontsize=12)

    fig.tight_layout()
    fig.savefig(path_png, dpi=150)
    plt.close(fig)


def main():
    start_xy = starts()
    rng = np.random.default_rng(SEED)
    for attempt in range(1, 501):
        segs = sample_obstacles(rng, start_xy)
        if segs is None:
            continue
        got = check_layout(segs, start_xy)
        if got is None:
            continue
        free, comp, xs, ys = got
        # The flood fill says a corridor exists; A* over the real action set
        # says the agent can actually take it. The mask refuses a direction
        # on a whole 45-degree sector, so a passage wide enough to stand in
        # is not always one the agent will enter. A layout that survives the
        # fill but not the search is rejected like any other.
        blocked, _bxs, _bys = OP.blocked_field(segs, OBS_HALF, RADIUS)
        paths = [OP.astar(p[:2], blocked, segs, OBS_HALF, RADIUS,
                          GOAL, GOAL_R) for p in start_xy]
        if all(p is not None for p in paths):
            break
        print(f'  attempt {attempt}: flood fill passed but A* found no route '
              f'from {sum(p is None for p in paths)} start(s); resampling')
    else:
        raise SystemExit('no layout passed the reachability check; '
                         'lower N_OBSTACLES or CLEAR')

    xml_path = os.path.join(HERE, NAME + '.xml')
    csv_path = os.path.join(HERE, 'positions', NAME + '_positions.csv')
    png_path = os.path.join(HERE, NAME + '.png')

    write_xml(segs, start_xy, xml_path)
    grid_xy, spacing, free_area = write_positions(segs, csv_path)
    npts = len(grid_xy[0])
    plot(segs, start_xy, paths, png_path)

    gap = min(seg_seg_dist(segs[i], segs[j])
              for i in range(len(segs)) for j in range(i + 1, len(segs)))
    reach = 100.0 * comp.sum() / max(1, free.sum())
    mean_len = float(np.mean(np.hypot(segs[:, 2] - segs[:, 0],
                                      segs[:, 3] - segs[:, 1])))
    print(f'{NAME}  seed {SEED}, accepted on attempt {attempt}')
    print(f'  obstacles      : {len(segs)}, {mean_len:.2f} m mean length')
    print(f'  closest pair   : {gap:.2f} m  (need {GAP:.2f})')
    print(f'  free space     : {free_area:.1f} m^2 of {np.pi*RADIUS**2:.1f}')
    print(f'  reachable      : {reach:.1f}% of the {CLEAR} m free space '
          f'is connected to the goal')
    print(f'  positions      : {npts:,} at {spacing:.4f} m spacing '
          f'(target {N_TARGET:,})')
    print(f'  {"start":<22s} {"direct":>7s} {"A* actions":>11s} '
          f'{"A* walked":>10s} {"detour":>7s}  straight line')
    for i, ((x, y, _t), path) in enumerate(zip(start_xy, paths), start=1):
        d = float(np.hypot(x - GOAL[0], y - GOAL[1]))
        steps = len(path) - 1
        walked = steps * ACTION_LENGTH
        blocked_line = straight_blocked(x, y, segs)
        print(f'  {i} ({x:7.4f},{y:7.4f})     {d:7.2f}m {steps:11d} '
              f'{walked:9.1f}m {walked/d:7.2f}  '
              f'{"BLOCKED" if blocked_line else "clear"}')
    worst = max(len(p) - 1 for p in paths)
    tight = min(route_clearance(p, segs) for p in paths)
    print(f'  longest route is {worst} actions against the '
          f'{int(os.environ.get("REALM_MAX_STEPS", 200))}-step episode cap')
    print(f'  tightest squeeze on any route: {tight:.2f} m from the robot\'s '
          f'centre to a surface (buffer {ROBOT_BUFFER})')
    print(f'  -> {xml_path}')
    print(f'  -> {csv_path}')
    print(f'  -> {png_path}')


if __name__ == '__main__':
    main()
