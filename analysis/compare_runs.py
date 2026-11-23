#!/usr/bin/env python3
"""Compare training runs: algorithms, representations, seeds.

Each run under data/vpce/rl_runs/<tag>/ leaves a config.json saying what it
was, a monitor.csv holding every episode, and a result.json with the final
deterministic evaluation. This reads a set of them and puts the learning
curves on one pair of axes, which is the comparison the paper is built from.

    python3 analysis/compare_runs.py                       # every run
    python3 analysis/compare_runs.py --by algo             # DQN against PPO
    python3 analysis/compare_runs.py --by n_cells          # representation size
    python3 analysis/compare_runs.py --runs tagA tagB
    python3 analysis/compare_runs.py --by algo --out-name dqn_vs_ppo

Grouping is the whole point. Runs that share a group value are treated as
repeats and drawn as a mean with a spread band; different groups get
different colours. So `--by algo` answers "which algorithm learns faster",
`--by n_cells` answers "does a bigger representation help", and `--by seed`
shows the run-to-run variation inside one condition, which is the number
that says whether any other difference is real.

Two cautions this script tries to make hard to ignore.

Learning time is reported as steps to a SUSTAINED success threshold, not to
the first crossing. Early in training epsilon is high and a random walk
reaches the goal often enough to cross a threshold on luck; see
rl_lib.learning_curve for the measurement and why the distinction changed a
real result by 4x.

Comparing across groups that differ in more than the grouping variable is
not a comparison. The script checks for that and says so: if two runs tagged
different algorithms also used different arenas or different ensembles, the
difference between their curves is not attributable to the algorithm.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from realm_tools.rl_lib import learning_curve

RUNS_DIR = f'{REPO}/data/vpce/rl_runs'
PALETTE = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#17becf']

# Fields that must match across groups for a comparison to mean anything.
# Anything differing here is a confound with whatever is being compared.
CONTROLLED = ('maze', 'ensemble', 'n_cells', 'max_steps', 'teleport', 'shaping')


def load_run(run_dir):
    """One run's config, episodes and final result. None if unusable."""
    cfg_path = os.path.join(run_dir, 'config.json')
    mon_path = os.path.join(run_dir, 'monitor.csv')
    if not (os.path.exists(cfg_path) and os.path.exists(mon_path)):
        return None
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        df = learning_curve.load_monitor(mon_path)
    except Exception as exc:                                # noqa: BLE001
        print(f'  skipping {os.path.basename(run_dir)}: {exc}')
        return None
    if not len(df):
        return None

    result = {}
    res_path = os.path.join(run_dir, 'result.json')
    if os.path.exists(res_path):
        try:
            with open(res_path) as f:
                result = json.load(f)
        except Exception:                                   # noqa: BLE001
            pass

    return dict(tag=cfg.get('tag', os.path.basename(run_dir)),
                dir=run_dir, config=cfg, df=df, result=result)


def run_metrics(run, threshold=0.8, window=100):
    df, res = run['df'], run['result']
    sustained = learning_curve.steps_to_threshold(df, threshold, window,
                                                  sustained=True)
    first = learning_curve.steps_to_threshold(df, threshold, window,
                                              sustained=False)
    tail = max(1, len(df) // 10)
    paths = res.get('paths') or {}
    return dict(
        tag=run['tag'],
        algo=run['config'].get('algo', '?'),
        seed=run['config'].get('seed', '?'),
        n_cells=run['config'].get('n_cells', '?'),
        episodes=len(df),
        total_steps=int(df['cum_steps'].iloc[-1]),
        steps_to_threshold=sustained,
        first_crossing=first,
        train_success=float(df['is_success'].iloc[-tail:].mean())
        if 'is_success' in df else float('nan'),
        eval_success=res.get('success_after'),
        starts_reached=(f"{paths.get('reached')}/{paths.get('starts')}"
                        if paths else '-'),
        stop_reason=res.get('stop_reason', '-'),
        wall_hours=(res.get('wall_seconds', 0) or 0) / 3600.0,
    )


def check_controlled(runs, by):
    """Warn when groups differ in anything other than the grouping field."""
    issues = []
    for field in CONTROLLED:
        if field == by:
            continue
        values = {str(r['config'].get(field)) for r in runs}
        if len(values) > 1:
            issues.append(f'{field}: {sorted(values)}')
    return issues


def _rolling(df, col, window):
    y = df[col].astype(float).to_numpy()
    w = max(1, min(int(window), max(1, len(y) // 5)))
    roll = np.convolve(y, np.ones(w) / w, mode='valid')
    return df['cum_steps'].to_numpy()[w - 1:], roll


def _band(ax, runs, col, window, colour, label):
    """Per-run lines plus their mean on a common step grid."""
    curves = [_rolling(r['df'], col, window) for r in runs]
    for x, y in curves:
        ax.plot(x, y, color=colour, lw=0.8, alpha=0.35)
    # Resample onto shared steps so repeats of different length can be
    # averaged; stopping early is informative, so the mean is drawn only as
    # far as the shortest run and the individual lines carry the rest.
    end = min(x[-1] for x, _ in curves) if curves else 0
    if end > 0 and len(curves) > 1:
        grid = np.linspace(0, end, 200)
        stacked = np.vstack([np.interp(grid, x, y) for x, y in curves])
        ax.plot(grid, stacked.mean(0), color=colour, lw=2.4, label=label)
        ax.fill_between(grid, stacked.min(0), stacked.max(0), color=colour,
                        alpha=0.15)
    elif curves:
        ax.plot(*curves[0], color=colour, lw=2.4, label=label)


def compare(runs, by='algo', threshold=0.8, window=100, out_path=None,
            title=''):
    groups = {}
    for r in runs:
        groups.setdefault(str(r['config'].get(by, '?')), []).append(r)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for i, (name, members) in enumerate(sorted(groups.items())):
        colour = PALETTE[i % len(PALETTE)]
        label = f'{by}={name}  (n={len(members)})'
        if 'is_success' in members[0]['df']:
            _band(axes[0], members, 'is_success', window, colour, label)
        _band(axes[1], members, 'l', window, colour, label)

    axes[0].axhline(threshold, ls=':', lw=1, color='grey')
    axes[0].set_ylabel('success rate')
    axes[0].set_title(f'success (rolling {window} episodes)', fontsize=10)
    axes[1].set_ylabel('episode length (steps)')
    axes[1].set_title('episode length', fontsize=10)
    for ax in axes[:2]:
        ax.set_xlabel('environment steps')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    # Learning time per run, grouped. The bar is the headline number; a run
    # that never held the threshold gets a hatched bar at the full budget so
    # it is visible as a failure rather than absent from the chart.
    ax = axes[2]
    labels, values, colours, hatches = [], [], [], []
    for i, (name, members) in enumerate(sorted(groups.items())):
        for r in members:
            m = run_metrics(r, threshold, window)
            labels.append(f'{name}/s{m["seed"]}')
            failed = m['steps_to_threshold'] is None
            values.append(m['total_steps'] if failed else m['steps_to_threshold'])
            colours.append(PALETTE[i % len(PALETTE)])
            hatches.append('///' if failed else '')
    bars = ax.bar(range(len(values)), values, color=colours)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
        if h:
            b.set_alpha(0.45)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('environment steps')
    ax.set_title(f'steps to {threshold:.0%} sustained\n'
                 f'(hatched = never reached)', fontsize=10)
    ax.grid(alpha=0.25, axis='y')

    if title:
        fig.suptitle(title, y=1.03, fontsize=12)
    fig.tight_layout()
    if out_path:
        d = os.path.dirname(out_path)
        if d:
            os.makedirs(d, exist_ok=True)
        fig.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


def table(rows):
    cols = [('tag', 28), ('algo', 5), ('seed', 5), ('n_cells', 8),
            ('episodes', 9), ('total_steps', 12), ('steps_to_threshold', 19),
            ('eval_success', 13), ('starts_reached', 15), ('stop_reason', 18),
            ('wall_hours', 11)]
    head = '  '.join(f'{c:<{w}}' for c, w in cols)
    lines = [head, '-' * len(head)]
    for r in rows:
        cells = []
        for c, w in cols:
            v = r.get(c)
            if v is None:
                v = 'never'
            elif c == 'wall_hours':
                v = f'{v:.2f}'
            elif c == 'eval_success' and isinstance(v, float):
                v = f'{v:.0%}'
            elif isinstance(v, int):
                v = f'{v:,}'
            cells.append(f'{str(v):<{w}}')
        lines.append('  '.join(cells))
    return lines


def main():
    ap = argparse.ArgumentParser(description='Compare RL training runs.')
    ap.add_argument('--runs', nargs='*', default=None,
                    help='run tags or directories (default: all)')
    ap.add_argument('--by', default='algo',
                    help='config field to group by (default: algo)')
    ap.add_argument('--threshold', type=float, default=0.8)
    ap.add_argument('--window', type=int, default=100)
    ap.add_argument('--runs-dir', default=RUNS_DIR)
    ap.add_argument('--out-name', default=None)
    args = ap.parse_args()

    if args.runs:
        dirs = [r if os.path.isdir(r) else os.path.join(args.runs_dir, r)
                for r in args.runs]
    else:
        dirs = sorted(glob.glob(os.path.join(args.runs_dir, '*')))

    runs = [r for r in (load_run(d) for d in dirs if os.path.isdir(d)) if r]
    if not runs:
        print(f'no usable runs under {args.runs_dir}\n'
              f'  A run needs config.json and monitor.csv. Train one with:\n'
              f'    sbatch slurm/train_rl.sh circ_lm8_r10', file=sys.stderr)
        return 1

    print(f'{len(runs)} run(s), grouped by {args.by}\n')
    rows = [run_metrics(r, args.threshold, args.window) for r in runs]
    for line in table(rows):
        print(line)

    issues = check_controlled(runs, args.by)
    if issues:
        # Not fatal: sometimes the differences are the point. But a
        # comparison across them is not attributable to `--by`, and that
        # should be said out loud rather than discovered in review.
        print(f'\nWARNING: these runs differ in more than {args.by!r}, so a '
              f'difference between the curves\n         cannot be attributed '
              f'to {args.by!r} alone:')
        for i in issues:
            print(f'           {i}')

    name = args.out_name or f'by_{args.by}'
    out = f'{REPO}/analysis/figures/rl_comparison/{name}.png'
    compare(runs, by=args.by, threshold=args.threshold, window=args.window,
            out_path=out, title=f'RL comparison by {args.by}')
    print(f'\nfigure -> {out}')

    csv_path = f'{REPO}/analysis/figures/rl_comparison/{name}.csv'
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f'table  -> {csv_path}')
    except Exception as exc:                                # noqa: BLE001
        print(f'(table not written: {exc})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
