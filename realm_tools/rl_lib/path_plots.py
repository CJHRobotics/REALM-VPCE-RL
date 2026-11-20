"""The routes the agent actually took, drawn in the arena it took them in.

A learning curve says whether reward went up. It cannot say what the policy
does, and the difference matters here: an agent that reaches the goal from
every start by a direct route and one that spirals in from the wall have the
same success rate and very different representations behind them. These are
the figures to look at when a curve converges and you want to know what it
converged ON -- and the figures to look at when it does not, because a
failure has a shape. Circling at a fixed radius, oscillating between two
cells, running into the wall and stopping: each says something different
about where the representation is thin.

The arena is drawn with the same primitives as the place-field maps
(`place_cell_lib.field_plots`), so a path figure and a field figure for the
same arena line up and can be read side by side -- which is the comparison
worth making, since a path that goes wrong in a region of sparse field
coverage is a different finding from one that goes wrong where coverage is
dense.
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from realm_tools.place_cell_lib.field_plots import draw_env

OK_COLOUR = '#2ca02c'
FAIL_COLOUR = '#d62728'


def env_from_xml(xml_root, fallback_xy=None):
    """Arena geometry for drawing, from the world XML."""
    from realm_tools.place_cell_lib.rules import build_env
    xy = np.asarray(fallback_xy if fallback_xy is not None else [[0.0, 0.0]],
                    dtype=float).reshape(-1, 2)
    return build_env(xy, xml_root)


def _draw_one(ax, rec, env, xml_root, goal, goal_radius, show_start_label=True):
    draw_env(ax, env, xml_root)
    xy = np.asarray(rec['xy'], dtype=float).reshape(-1, 2)
    ok = rec['terminated']
    colour = OK_COLOUR if ok else FAIL_COLOUR

    if goal is not None:
        ax.add_patch(plt.Circle(goal, goal_radius, facecolor='gold',
                                edgecolor='black', lw=1.0, alpha=0.65,
                                zorder=5))

    # The route, then the direction of travel. Points are drawn under the
    # line so a path that doubles back still reads as one stroke.
    ax.plot(xy[:, 0], xy[:, 1], '-', color=colour, lw=1.6, alpha=0.9, zorder=8)
    ax.scatter(xy[:, 0], xy[:, 1], s=6, color=colour, alpha=0.5, zorder=7)
    if len(xy) > 1:
        # One arrow at the midpoint: enough to show which way it ran without
        # cluttering a 200-step path.
        m = len(xy) // 2
        d = xy[min(m + 1, len(xy) - 1)] - xy[m]
        if np.hypot(*d) > 1e-9:
            ax.annotate('', xy=xy[m] + d, xytext=xy[m], zorder=9,
                        arrowprops=dict(arrowstyle='-|>', color=colour, lw=1.4))

    ax.plot(*xy[0], marker='o', ms=8, color='white', mec='black', mew=1.4,
            zorder=10)
    ax.plot(*xy[-1], marker='*' if ok else 'X', ms=14 if ok else 10,
            color=colour, mec='black', mew=0.8, zorder=10)

    verdict = 'reached' if ok else f'{rec["final_dist"]:.1f} m short'
    title = f'start {rec["start_idx"]} | {rec["steps"]} steps | {verdict}'
    if show_start_label:
        ax.set_title(title, fontsize=9,
                     color='black' if ok else FAIL_COLOUR)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_paths(paths, xml_root, out_path, goal=(0.0, 0.0), goal_radius=0.8,
               title='', max_cols=3):
    """One panel per start, plus a combined panel. Returns (path, summary).

    `paths` is the env's `last_paths`: start index -> episode record.
    """
    recs = [paths[k] for k in sorted(paths)]
    if not recs:
        return None, {'starts': 0}

    all_xy = np.concatenate([np.asarray(r['xy']).reshape(-1, 2) for r in recs])
    env = env_from_xml(xml_root, fallback_xy=all_xy)

    n = len(recs) + 1
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.3 * rows),
                             squeeze=False)
    flat = [a for row in axes for a in row]

    # Combined first: the whole policy at a glance.
    ax = flat[0]
    draw_env(ax, env, xml_root)
    if goal is not None:
        ax.add_patch(plt.Circle(goal, goal_radius, facecolor='gold',
                                edgecolor='black', lw=1.0, alpha=0.65, zorder=5))
    for rec in recs:
        xy = np.asarray(rec['xy'], dtype=float).reshape(-1, 2)
        colour = OK_COLOUR if rec['terminated'] else FAIL_COLOUR
        ax.plot(xy[:, 0], xy[:, 1], '-', color=colour, lw=1.3, alpha=0.75,
                zorder=8)
        ax.plot(*xy[0], marker='o', ms=6, color='white', mec='black', mew=1.2,
                zorder=10)
        ax.plot(*xy[-1], marker='*' if rec['terminated'] else 'X',
                ms=11 if rec['terminated'] else 8, color=colour, mec='black',
                mew=0.7, zorder=10)
    n_ok = sum(r['terminated'] for r in recs)
    ax.set_title(f'all starts — {n_ok}/{len(recs)} reached the goal',
                 fontsize=10, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    for ax, rec in zip(flat[1:], recs):
        _draw_one(ax, rec, env, xml_root, goal, goal_radius)
    for ax in flat[n:]:
        ax.axis('off')

    handles = [
        plt.Line2D([], [], color=OK_COLOUR, lw=2, label='reached the goal'),
        plt.Line2D([], [], color=FAIL_COLOUR, lw=2, label='timed out'),
        plt.Line2D([], [], marker='o', ls='', color='white', mec='black',
                   label='start'),
        plt.Line2D([], [], marker='*', ls='', color=OK_COLOUR, mec='black',
                   ms=12, label='goal reached'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    if title:
        fig.suptitle(title, y=1.0, fontsize=12)
    fig.tight_layout()

    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    lengths = [r['steps'] for r in recs if r['terminated']]
    summary = {
        'starts': len(recs),
        'reached': int(n_ok),
        'mean_steps_when_reached': float(np.mean(lengths)) if lengths else None,
        'per_start': {int(r['start_idx']): dict(
            reached=bool(r['terminated']), steps=int(r['steps']),
            final_dist=float(r['final_dist'])) for r in recs},
    }
    return out_path, summary


def summarise(summary):
    """Lines for a log or an email body."""
    if not summary.get('starts'):
        return ['no paths recorded']
    lines = [f'  starts        : {summary["reached"]}/{summary["starts"]} '
             f'reached the goal']
    if summary.get('mean_steps_when_reached') is not None:
        lines.append(f'  steps to goal : '
                     f'{summary["mean_steps_when_reached"]:.1f} on average '
                     f'(successful runs)')
    for k, v in sorted(summary.get('per_start', {}).items()):
        state = 'reached' if v['reached'] else f'{v["final_dist"]:.1f} m short'
        lines.append(f'    start {k}: {v["steps"]:3d} steps, {state}')
    return lines
