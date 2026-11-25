"""The methods figure: what the mask analysis does, on one real state.

Four panels, read left to right, top to bottom. Nothing here is a schematic
-- every number is from a recorded run of the trained policy in the
simulator, and the state is one the agent actually passed through.

  A  where the agent is, and what it did there
  B  what it saw: 1228 place-cell activations, and where the chosen cells
     sit in that ranking
  C  that the choice is real: the action distribution before and after
     those cells are deleted
  D  which places and which field scales the chosen cells correspond to

Panel B is the one worth looking at twice. If importance were just
activation, the selected cells would cluster at the left edge of that curve.
Where they actually fall is a property of the policy, not of the encoder,
and it is visible before any statistics are done.

    python analysis/mask_figure.py --run circ_lm10_r10_o20_ppo_s0
    python analysis/mask_figure.py --run <tag> --start 0 --step 11
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
from matplotlib.lines import Line2D                          # noqa: E402
from matplotlib.patches import Ellipse                       # noqa: E402

from realm_tools.place_cell_lib.field_plots import draw_env   # noqa: E402
from realm_tools.rl_lib.mask_saliency import (                # noqa: E402
    action_logits, perturb, reference_vector)
from realm_tools.rl_lib.path_plots import env_from_xml        # noqa: E402
from realm_tools.simulation_lib.arena_paths import arena_xml  # noqa: E402

BEARING = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
# One colour per scale band, dark to light with increasing field size, so a
# reader can tell a 0.4 m field from a 6 m one without the legend.
SCALE_COLOURS = ['#08306b', '#2171b5', '#4292c6', '#6baed6', '#9ecae1',
                 '#c6dbef']


def pick_state(steps, start=None, step=None):
    """A representative state, or the one asked for.

    Default is the median set size rather than the smallest: a figure built
    on the easiest state in the run flatters the method.
    """
    if start is not None and step is not None:
        hit = steps[(steps['start'] == start) & (steps['step'] == step)]
        if hit.empty:
            raise SystemExit(f'no state at start {start} step {step}')
        return int(hit.index[0])
    order = (steps['set_size'] - steps['set_size'].median()).abs()
    return int(order.idxmin())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, help='mask_probe run tag')
    ap.add_argument('--start', type=int, default=None)
    ap.add_argument('--step', type=int, default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    probe = f'{REPO}/data/vpce/mask_probe/{args.run}'
    if not os.path.isdir(probe):
        raise SystemExit(f'no mask-probe output at {probe}\n'
                         f'  sbatch slurm/mask_probe.sh <arena>')
    with open(f'{probe}/config.json') as f:
        cfg = json.load(f)
    steps = pd.read_csv(f'{probe}/mask_steps.csv')
    cells = pd.read_csv(f'{probe}/mask_cells.csv')
    bank = pd.read_csv(os.path.join(REPO, cfg['bank']))
    obs_all = np.load(f'{probe}/observations.npz')['observations']

    i = pick_state(steps, args.start, args.step)
    st = steps.iloc[i]
    obs = obs_all[i]
    chosen = cells[(cells['start'] == st['start']) &
                   (cells['step'] == st['step'])].sort_values('rank')
    idx = chosen['cell_index'].to_numpy()

    xml_root = ET.parse(os.path.join(REPO, arena_xml(cfg['arena'], REPO))
                        if not os.path.isabs(arena_xml(cfg['arena'], REPO))
                        else arena_xml(cfg['arena'], REPO)).getroot()
    env = env_from_xml(xml_root)

    # Panel C needs the policy itself: the figure asserts that deleting
    # these cells changes the action, so it had better recompute it rather
    # than quote the CSV back.
    from stable_baselines3 import DQN, PPO
    import torch
    algo = {'DQN': DQN, 'PPO': PPO}[cfg['algo']]
    model = algo.load(f"{REPO}/data/vpce/rl_runs/{cfg['run']}/model.zip",
                      device='cpu',
                      custom_objects={'lr_schedule': lambda _: 0.0,
                                      'clip_range': lambda _: 0.0})
    x = torch.as_tensor(np.array(obs, dtype=np.float32)).unsqueeze(0)
    ref = reference_vector(x, cfg['mask_reference'])
    delete = torch.zeros_like(x)
    delete[0, idx] = 1.0
    with torch.no_grad():
        p_before = torch.softmax(action_logits(model.policy, x), -1)[0].numpy()
        p_after = torch.softmax(
            action_logits(model.policy, perturb(x, delete, ref)), -1)[0].numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5))
    (axA, axB), (axC, axD) = axes

    # ---------------------------------------------------------------- A
    draw_env(axA, env, xml_root)
    path = steps[(steps['start'] == st['start']) &
                 (steps['step'] <= st['step'])]
    axA.plot(path['robot_x'], path['robot_y'], '-', color='0.35', lw=1.6,
             zorder=8)
    axA.scatter(path['robot_x'], path['robot_y'], s=10, color='0.35', zorder=8)
    axA.add_patch(plt.Circle((0, 0), 0.8, facecolor='gold', edgecolor='black',
                             lw=1.0, alpha=0.7, zorder=5))
    axA.plot(st['robot_x'], st['robot_y'], 'o', ms=11, color='darkgreen',
             mec='black', mew=1.0, zorder=11)
    th = np.radians(45 * st['action'])
    axA.annotate('', xytext=(st['robot_x'], st['robot_y']),
                 xy=(st['robot_x'] + 1.6 * np.cos(th),
                     st['robot_y'] + 1.6 * np.sin(th)), zorder=12,
                 arrowprops=dict(arrowstyle='-|>', color='crimson', lw=2.2))
    axA.set_title(f'A   the state: start {st["start"]}, step {st["step"]}\n'
                  f'{st["dist_to_goal"]:.2f} m from the goal, '
                  f'policy chooses {st["action_bearing"]}', fontsize=10)
    axA.set_xticks([]); axA.set_yticks([])

    # ---------------------------------------------------------------- B
    order = np.argsort(-obs)
    rank_of = np.empty_like(order)
    rank_of[order] = np.arange(len(order))
    axB.plot(np.arange(1, len(obs) + 1), obs[order], color='0.6', lw=1.2,
             label=f'all {len(obs)} place cells')
    axB.scatter(rank_of[idx] + 1, obs[idx], s=26, color='crimson', zorder=5,
                label=f'the {len(idx)} the action needs')
    axB.set_yscale('log')
    axB.set_xlabel('place cell, ranked by activation at this state')
    axB.set_ylabel('activation (share of the population code)')
    lo, hi = int(rank_of[idx].min()) + 1, int(rank_of[idx].max()) + 1
    axB.set_title('B   what it saw, and which of it mattered\n'
                  f'the set is drawn from the active tail (ranks {lo}-{hi}) '
                  f'but is not simply the top {len(idx)}', fontsize=10)
    axB.legend(fontsize=8.5, loc='upper right')
    axB.grid(alpha=0.25, lw=0.6)

    # ---------------------------------------------------------------- C
    w = 0.38
    pos = np.arange(8)
    axC.bar(pos - w / 2, p_before, w, color='0.45', label='as observed')
    axC.bar(pos + w / 2, p_after, w, color='crimson', alpha=0.85,
            label=f'with those {len(idx)} cells deleted')
    axC.set_xticks(pos); axC.set_xticklabels(BEARING)
    axC.set_xlabel('action (world bearing)')
    axC.set_ylabel('policy probability')
    axC.axvline(int(st['action']), color='0.45', ls=':', lw=1.2)
    axC.axvline(int(st['flipped_to']), color='crimson', ls=':', lw=1.2)
    a0, a1 = int(st['action']), int(st['flipped_to'])
    axC.annotate(f'margin before {p_before[a0] - p_before[a1]:+.3f}\n'
                 f'margin after  {p_after[a0] - p_after[a1]:+.3f}',
                 xy=(0.98, 0.72), xycoords='axes fraction', ha='right',
                 fontsize=8.5, family='monospace',
                 bbox=dict(fc='white', ec='0.7', lw=0.6, pad=0.4))
    axC.set_title(f'C   the set is verified, not thresholded\n'
                  f'{BEARING[a0]} -> {BEARING[a1]} when they are replaced '
                  f'by the uniform code', fontsize=10)
    axC.legend(fontsize=8.5)
    axC.grid(axis='y', alpha=0.25, lw=0.6)

    # ---------------------------------------------------------------- D
    draw_env(axD, env, xml_root)
    sub = bank.iloc[idx]
    for _, r in sub.iterrows():
        c = SCALE_COLOURS[int(r['scale']) % len(SCALE_COLOURS)]
        axD.add_patch(Ellipse((r['centroid_x'], r['centroid_y']),
                              width=2 * r['semi_major_m'],
                              height=2 * r['semi_minor_m'],
                              angle=np.degrees(r['orientation_rad']),
                              facecolor=c, edgecolor=c, alpha=0.25, lw=0.9,
                              zorder=3))
    axD.scatter(sub['centroid_x'], sub['centroid_y'],
                s=14, c=[SCALE_COLOURS[int(s) % len(SCALE_COLOURS)]
                         for s in sub['scale']], zorder=9)
    axD.add_patch(plt.Circle((0, 0), 0.8, facecolor='gold', edgecolor='black',
                             lw=1.0, alpha=0.7, zorder=5))
    axD.plot(st['robot_x'], st['robot_y'], 'o', ms=11, color='darkgreen',
             mec='black', mew=1.0, zorder=11)
    present = sorted(sub['scale'].unique())
    axD.legend(handles=[Line2D([], [], marker='o', ls='', color=SCALE_COLOURS[s],
                               label=f'scale {s}  '
                                     f'({bank[bank.scale == s].radius_env_m.mean():.1f} m)')
                        for s in present], fontsize=8, loc='upper left',
               bbox_to_anchor=(1.005, 1.0), borderaxespad=0.0,
               title='field size', title_fontsize=8, frameon=False)
    axD.set_title('D   where those cells have their fields\n'
                  'ellipse is the fitted field; green is the robot, '
                  'gold the goal', fontsize=10)
    axD.set_xticks([]); axD.set_yticks([])

    out = args.out or f'{probe}/mask_process.png'
    fig.suptitle(f'Minimal-mask saliency over place cells — {cfg["run"]}, '
                 f'{cfg["arena"]}', fontsize=12.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')
    print(f'state: start {st["start"]} step {st["step"]}, '
          f'{len(idx)} cells, {st["action_bearing"]} -> '
          f'{BEARING[int(st["flipped_to"])]}')
    print(f'  -> {out}')


if __name__ == '__main__':
    main()
