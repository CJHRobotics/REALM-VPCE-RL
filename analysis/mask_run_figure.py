"""The run-level figure: does what the policy leans on change with where it is?

Four panels, two questions crossed with two measures of where the robot is.

                      near an obstacle        near the goal
  how many cells          A                        B
  which scales            C                        D

A and B ask whether the decision gets narrower or broader -- a small set
means a few cells carry it, a large set means the choice is spread thin.
C and D ask whether the SIZE of the fields it leans on changes, which is
the multiscale claim in its most direct form.

TWO THINGS THE PANELS ARE BUILT TO STOP YOU CONCLUDING.

Exposure. A 6 m field is active almost everywhere and a 0.4 m field almost
nowhere, so the big one gets far more chances to be chosen whatever the
policy is doing. C and D therefore carry an extra bar on the right: the
composition you would get by picking cells in proportion to how much of the
arena each covers. A bin that looks like that bar says nothing. Reading the
coloured bars without it is how this analysis produces a result it has not
earned.

One trajectory. 148 of these 207 states are start 1, which took 148 steps
where the other three took 13 to 26. An aggregate over states is therefore
mostly an aggregate over one episode that went badly, so A and B colour by
start and say the counts out loud rather than averaging them away.

    python analysis/mask_run_figure.py --run circ_lm10_r10_o20_ppo_s0_gaivi
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import matplotlib                                            # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                              # noqa: E402

from realm_tools.simulation_lib.arena_paths import arena_xml  # noqa: E402

SCALE_COLOURS = ['#08306b', '#2171b5', '#4292c6', '#6baed6', '#9ecae1',
                 '#c6dbef']
START_COLOURS = ['#1f77b4', '#d62728', '#7b3fa0', '#0d7d5a']


def barrier_distance(xs, ys, xml_root):
    """Distance from the robot to the nearest obstacle surface.

    The perimeter is included for completeness and never wins in this arena
    -- the agent's routes keep it 2 m away at the closest -- but an arena
    with fewer obstacles would need it.
    """
    walls = xml_root.findall('wall')
    half = float(walls[0].get('width', 0.012)) / 2 if walls else 0.0
    p = np.stack([xs, ys], axis=1)
    best = np.full(len(p), np.inf)
    if walls:
        segs = np.array([[float(w.get(k)) for k in ('x1', 'y1', 'x2', 'y2')]
                         for w in walls])
        a, b = segs[:, :2], segs[:, 2:]
        ab = b - a
        t = np.clip(((p[:, None] - a) * ab).sum(-1)
                    / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)[..., None]
        best = np.linalg.norm(p[:, None] - (a + t * ab), axis=-1).min(1) - half
    cw = xml_root.find('circular_wall')
    if cw is not None:
        best = np.minimum(best, float(cw.get('radius')) - np.hypot(xs, ys))
    return best


def scatter_panel(ax, steps, xcol, xlabel, nbins=5):
    """Set size against a distance, with the per-start detail kept visible."""
    for st, grp in steps.groupby('start'):
        ax.scatter(grp[xcol], grp['set_size'], s=16, alpha=0.55,
                   color=START_COLOURS[int(st) % len(START_COLOURS)],
                   edgecolors='none', label=f'start {int(st)} ({len(grp)})')
    edges = np.quantile(steps[xcol], np.linspace(0, 1, nbins + 1))
    edges[-1] += 1e-9
    mids, med = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = steps[(steps[xcol] >= lo) & (steps[xcol] < hi)]
        if len(sel):
            mids.append(sel[xcol].median())
            med.append(sel['set_size'].median())
    ax.plot(mids, med, '-o', color='black', lw=1.8, ms=5, zorder=6,
            label='median per bin')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('cells needed to change the action')
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.9)


def composition_panel(ax, steps, cells, bank, xcol, xlabel, scales, nbins=5):
    """Scale makeup of the chosen sets per distance bin, against exposure."""
    # mask_cells already carries the robot's distance to the goal, so a
    # blind merge would collide and produce dist_to_goal_x / _y. Only bring
    # across what is not already there.
    merged = (cells if xcol in cells.columns
              else cells.merge(steps[['start', 'step', xcol]],
                               on=['start', 'step']))
    edges = np.quantile(steps[xcol], np.linspace(0, 1, nbins + 1))
    edges[-1] += 1e-9

    labels, fracs, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = merged[(merged[xcol] >= lo) & (merged[xcol] < hi)]
        if not len(sel):
            continue
        labels.append(f'{lo:.1f}-{hi:.1f}')
        counts.append(sel[['start', 'step']].drop_duplicates().shape[0])
        v = sel['scale'].value_counts(normalize=True)
        fracs.append([float(v.get(s, 0.0)) for s in scales])

    # What picking by opportunity alone would give: each cell weighted by
    # how many of the collection positions fall inside its field.
    expo = bank.groupby('scale')['n_members'].sum()
    expo = (expo / expo.sum()).reindex(scales).fillna(0.0).to_numpy()
    labels.append('if by\nexposure')
    fracs.append(list(expo))
    counts.append(None)

    fracs = np.asarray(fracs)
    xs = np.arange(len(labels), dtype=float)
    xs[-1] += 0.45                                   # set the reference apart
    bottom = np.zeros(len(labels))
    for j, s in enumerate(scales):
        ax.bar(xs, fracs[:, j], 0.72, bottom=bottom,
               color=SCALE_COLOURS[int(s) % len(SCALE_COLOURS)],
               edgecolor='white', lw=0.5,
               label=f'scale {int(s)} ({bank[bank.scale == s].radius_env_m.mean():.1f} m)')
        bottom += fracs[:, j]
    for xi, n in zip(xs, counts):
        if n is not None:
            ax.text(xi, 1.015, f'n={n}', ha='center', fontsize=7, color='0.35')
    ax.axvline(xs[-1] - 0.33, color='0.4', ls=':', lw=1.0)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('share of the chosen cells')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    probe = f'{REPO}/data/vpce/mask_probe/{args.run}'
    with open(f'{probe}/config.json') as f:
        cfg = json.load(f)
    steps = pd.read_csv(f'{probe}/mask_steps.csv')
    cells = pd.read_csv(f'{probe}/mask_cells.csv')
    bank = pd.read_csv(os.path.join(REPO, cfg['bank']))
    xml_root = ET.parse(arena_xml(cfg['arena'], REPO)).getroot()

    steps['obstacle_m'] = barrier_distance(steps.robot_x.to_numpy(),
                                           steps.robot_y.to_numpy(), xml_root)
    scales = sorted(bank['scale'].unique())

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))
    (axA, axB), (axC, axD) = axes

    scatter_panel(axA, steps, 'obstacle_m',
                  'robot distance to the nearest obstacle (m)')
    axA.set_title('A   how concentrated the decision is, near obstacles',
                  fontsize=10.5)
    scatter_panel(axB, steps, 'dist_to_goal',
                  'robot distance to the goal (m)')
    axB.set_title('B   how concentrated the decision is, near the goal',
                  fontsize=10.5)

    composition_panel(axC, steps, cells, bank, 'obstacle_m',
                      'robot distance to the nearest obstacle (m)', scales)
    axC.set_title('C   which field sizes it leans on, near obstacles',
                  fontsize=10.5)
    composition_panel(axD, steps, cells, bank, 'dist_to_goal',
                      'robot distance to the goal (m)', scales)
    axD.set_title('D   which field sizes it leans on, near the goal',
                  fontsize=10.5)
    axD.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0),
               frameon=False, title='field size', title_fontsize=8)

    fig.suptitle(f'Minimal-mask saliency across the run — {cfg["run"]}, '
                 f'{len(steps)} states over {steps.start.nunique()} starts',
                 fontsize=12.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = args.out or f'{probe}/mask_run.png'
    fig.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')

    print(f'obstacle distance {steps.obstacle_m.min():.2f}-'
          f'{steps.obstacle_m.max():.2f} m, goal {steps.dist_to_goal.min():.2f}-'
          f'{steps.dist_to_goal.max():.2f} m')
    for col in ('obstacle_m', 'dist_to_goal'):
        print(f'  set size vs {col:<13}: r = {steps.set_size.corr(steps[col]):+.3f}'
              f'   mean scale vs {col}: '
              f'r = {steps.mean_scale.corr(steps[col]):+.3f}')
    print(f'  -> {out}')


if __name__ == '__main__':
    main()
