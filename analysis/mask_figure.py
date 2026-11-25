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
    # Importance over SPACE, as a field rather than a pile of ellipses.
    #
    # Two things forced this. The optimiser's deletion weights are nearly
    # all saturated at ~0.995 -- it drives the chosen cells to one and
    # leaves the rest at zero, which makes it a selector and not a ranking,
    # so shading by it coloured noise. And drawing 62 fitted ellipses puts a
    # 6 m scale-5 blob on top of everything else: the largest fields own the
    # ink regardless of what they contributed.
    #
    # So importance here is measured, one cell at a time: how far the
    # policy's probability of its chosen action falls when that cell alone
    # is deleted. 62 forward passes, and it means something a caption can
    # state. Each cell then contributes its own fitted Gaussian, weighted by
    # that drop, and the sum is the map -- so a large field spreads its
    # contribution over the area it actually covers instead of hiding the
    # small ones underneath it.
    with torch.no_grad():
        p0 = float(torch.softmax(action_logits(model.policy, x), -1)[0, int(st['action'])])
    solo = []
    for cell in idx:
        d = torch.zeros_like(x)
        d[0, int(cell)] = 1.0
        with torch.no_grad():
            solo.append(p0 - float(torch.softmax(
                action_logits(model.policy, perturb(x, d, ref)),
                -1)[0, int(st['action'])]))
    solo = np.asarray(solo)

    R = env['env_R']
    gx = np.linspace(-R, R, 260)
    GX, GY = np.meshgrid(gx, gx)
    heat = np.zeros_like(GX)
    for (_, r), wi in zip(bank.iloc[idx].iterrows(), solo):
        if wi <= 0:
            continue                      # deleting it HELPED; not support
        th = float(r['orientation_rad'])
        dx, dy = GX - r['centroid_x'], GY - r['centroid_y']
        u = dx * np.cos(th) + dy * np.sin(th)
        v = -dx * np.sin(th) + dy * np.cos(th)
        a = max(float(r['semi_major_m']), 0.15)
        b = max(float(r['semi_minor_m']), 0.15)
        heat += wi * np.exp(-0.5 * ((u / a) ** 2 + (v / b) ** 2))
    heat[GX ** 2 + GY ** 2 > R ** 2] = np.nan

    im = axB.imshow(heat, origin='lower', extent=[-R, R, -R, R],
                    cmap='magma_r', interpolation='bilinear', zorder=1)
    draw_env(axB, env, xml_root)
    axB.add_patch(plt.Circle((0, 0), 0.8, facecolor='none', edgecolor='gold',
                             lw=2.0, zorder=6))
    axB.plot(st['robot_x'], st['robot_y'], 'o', ms=11, color='limegreen',
             mec='black', mew=1.2, zorder=11)
    cb = fig.colorbar(im, ax=axB, fraction=0.046, pad=0.02)
    cb.set_label('drop in P(action) if that cell alone is removed',
                 fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    axB.set_title(f'B   where the decision came from\n'
                  f'{len(idx)} cells of the {int((obs > obs.max() * 0.02).sum())} '
                  f'active here, each weighted by what it alone contributes',
                  fontsize=10)
    axB.set_xticks([]); axB.set_yticks([])

    # ---------------------------------------------------------------- C
    # The question this answers: is the ORDER meaningful, or would deleting
    # any 62 cells do? Delete the ranked cells one at a time and watch the
    # chosen action's probability fall; delete the same number at random and
    # watch it not. Bars over eight actions were here before and nobody
    # could tell what they were for, which is fair -- they showed the end
    # state and not the thing being claimed.
    ks = np.unique(np.clip(
        np.r_[0, np.round(np.geomspace(1, max(len(idx) * 4, 8), 24))],
        0, len(obs)).astype(int))
    a0 = int(st['action'])

    def p_after_deleting(cols):
        d = torch.zeros_like(x)
        d[0, list(cols)] = 1.0
        with torch.no_grad():
            return float(torch.softmax(
                action_logits(model.policy, perturb(x, d, ref)), -1)[0, a0])

    full_order = np.argsort(-np.asarray(
        [chosen.set_index('cell_index')['deletion_weight'].get(i, 0.0)
         for i in range(len(obs))]))
    ranked = [p_after_deleting(full_order[:k]) for k in ks]

    rng = np.random.default_rng(0)
    draws = np.array([[p_after_deleting(rng.choice(len(obs), k, replace=False))
                       for k in ks] for _ in range(20)])

    axC.plot(ks, ranked, '-o', ms=3.5, color='crimson', lw=1.8,
             label='deleted in importance order', zorder=5)
    axC.plot(ks, draws.mean(0), '-', color='0.45', lw=1.6,
             label='the same number, chosen at random')
    axC.fill_between(ks, np.percentile(draws, 5, axis=0),
                     np.percentile(draws, 95, axis=0), color='0.45',
                     alpha=0.18, lw=0, label='5-95% over 20 random draws')
    axC.axvline(len(idx), color='crimson', ls=':', lw=1.2)
    axC.annotate(f'{len(idx)} cells:\nthe action changes',
                 xy=(len(idx), ranked[int(np.searchsorted(ks, len(idx)))
                                      if len(idx) in ks else -1]),
                 xytext=(len(idx) * 1.5, max(ranked) * 0.92), fontsize=8.5,
                 arrowprops=dict(arrowstyle='->', color='crimson', lw=1.0))
    axC.set_xscale('log')
    axC.set_xlim(1, ks.max())
    axC.set_xlabel('place cells deleted')
    axC.set_ylabel(f'policy probability of {BEARING[a0]}')
    axC.set_title('C   the ordering is doing the work\n'
                  'removing these cells collapses the choice; removing as '
                  'many at random does not', fontsize=10)
    axC.legend(fontsize=8.5, loc='lower left')
    axC.grid(alpha=0.25, lw=0.6)

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
