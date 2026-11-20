"""Turn a run's Monitor CSV into the figure that answers "did it learn?".

SB3's Monitor writes one row per episode: reward, length, elapsed time, and
whatever info keys were requested -- here `is_success`. That file is the
measurement this whole pipeline exists to produce, so this module is
deliberately independent of it being complete: a run killed by the walltime
leaves a truncated CSV, and plotting it should work anyway.

What to read off the figure:

  reward rising and flattening       learning, then converged
  success rate rising to near 1      the task is actually being solved,
                                     which reward alone can hide when a
                                     step penalty dominates
  episode length falling             paths getting shorter, the clearest
                                     sign of a policy rather than luck
  flat everything                    with sparse reward, usually means the
                                     goal was never found -- look at whether
                                     exploration ran long enough before
                                     concluding the representation is at fault

Steps-to-threshold is printed rather than only drawn, because it is the
quantity the experiment compares across representations and eyeballing a
curve is not a measurement.
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_monitor(path):
    """Read an SB3 Monitor CSV, tolerating a run that was cut short."""
    import pandas as pd
    # Monitor's first line is a '#' JSON header; the last line of a killed
    # run can be half-written, which skipping bad lines handles.
    df = pd.read_csv(path, skiprows=1, on_bad_lines='skip')
    if 'r' not in df.columns:
        raise ValueError(f'{path} has no reward column; is it a Monitor CSV?')
    df = df.dropna(subset=['r', 'l']).reset_index(drop=True)
    df['episode'] = np.arange(len(df))
    df['cum_steps'] = df['l'].cumsum()
    return df


def _roll(x, window):
    s = max(1, min(int(window), max(1, len(x) // 5)))
    return np.convolve(np.asarray(x, dtype=float), np.ones(s) / s, mode='valid'), s


def steps_to_threshold(df, threshold=0.8, window=100, sustained=True):
    """Environment steps before the rolling success rate reaches `threshold`.

    Reported in STEPS rather than episodes, because episodes vary in length
    and a run that solves the task has shorter ones -- counting episodes
    would flatter the very runs that learned.

    `sustained` is the important argument, and it defaults on. With it, the
    answer is the first point after which success NEVER AGAIN drops below
    the threshold. Without it, the answer is the first crossing, and that
    number is not a measure of learning.

    The reason is epsilon. Early in training epsilon is near 1, and in an
    arena where a random walk stumbles into the goal reasonably often, the
    rolling success rate can cross a high threshold on luck alone -- then
    collapse as epsilon anneals and the half-trained policy takes over, and
    recover only once the policy is genuinely good. A first-crossing measure
    reads the luck and ignores the learning. It is also biased
    inconsistently across conditions, because how often a random walk
    succeeds depends on the arena, not on the representation being tested,
    so it would corrupt the comparison this experiment exists to make.

    Observed directly on a proof-of-concept run: first crossing at 2,805
    steps, collapse to 17% success at 9,000, sustained 80% only at ~12,500.

    Returns None when the threshold is never reached, or never held.
    """
    if 'is_success' not in df.columns or not len(df):
        return None
    roll, w = _roll(df['is_success'].astype(float), window)
    if not len(roll):
        return None

    def at(i):
        """Cumulative steps at the episode that completed rolling index i."""
        return int(df['cum_steps'].iloc[min(i + w - 1, len(df) - 1)])

    if not sustained:
        hit = np.flatnonzero(roll >= threshold)
        return at(hit[0]) if len(hit) else None

    # The first index from which every later window also clears the bar.
    # Walking backwards finds it in one pass: the answer is one past the
    # last point that failed.
    below = np.flatnonzero(roll < threshold)
    if not len(below):
        return at(0)                       # cleared from the very first window
    first_ok = below[-1] + 1
    if first_ok >= len(roll):
        return None                        # never held to the end of the run
    return at(first_ok)


def plot(monitor_csv, out_path, title='', window=100, threshold=0.8):
    """Three panels against environment steps. Returns (path, summary dict)."""
    df = load_monitor(monitor_csv)
    if not len(df):
        return None, {'episodes': 0}

    has_success = 'is_success' in df.columns
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))

    series = [('r', 'episode reward', '#1f77b4'),
              ('l', 'episode length (steps)', '#ff7f0e')]
    if has_success:
        series.append(('is_success', 'success rate', '#2ca02c'))

    for ax, (col, label, colour) in zip(axes, series):
        y = df[col].astype(float)
        ax.plot(df['cum_steps'], y, lw=0.6, alpha=0.25, color=colour)
        roll, w = _roll(y, window)
        ax.plot(df['cum_steps'].iloc[w - 1:], roll, lw=1.8, color=colour,
                label=f'rolling mean ({w})')
        ax.set_xlabel('environment steps')
        ax.set_ylabel(label)
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.25)

    summary = {
        'episodes': int(len(df)),
        'total_steps': int(df['cum_steps'].iloc[-1]),
        'reward_first_10pct': float(df['r'].iloc[:max(1, len(df) // 10)].mean()),
        'reward_last_10pct': float(df['r'].iloc[-max(1, len(df) // 10):].mean()),
        'length_last_10pct': float(df['l'].iloc[-max(1, len(df) // 10):].mean()),
    }
    if has_success:
        tail = df['is_success'].astype(float).iloc[-max(1, len(df) // 10):]
        summary['success_last_10pct'] = float(tail.mean())
        sct = steps_to_threshold(df, threshold=threshold, window=window)
        first = steps_to_threshold(df, threshold=threshold, window=window,
                                   sustained=False)
        summary['steps_to_threshold'] = sct
        summary['steps_first_crossing'] = first
        summary['threshold'] = threshold
        axes[2].axhline(threshold, ls=':', lw=1, color='grey')
        # Both are drawn: a wide gap between them is itself the diagnostic,
        # saying the early success was exploration rather than policy.
        if first is not None:
            axes[2].axvline(first, ls=':', lw=1, color='tab:red')
        if sct is not None:
            axes[2].axvline(sct, ls='--', lw=1.4, color='black')
            axes[2].set_title(f'{threshold:.0%} sustained at {sct:,} steps'
                              + (f' (first touched {first:,})'
                                 if first is not None and first < sct else ''),
                              fontsize=9)
        else:
            axes[2].set_title(f'never held {threshold:.0%} to the end',
                              fontsize=10)

    if title:
        fig.suptitle(title, y=1.04, fontsize=12)
    fig.tight_layout()
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path, summary


def summarise(summary):
    """The summary as lines for a log or an email body."""
    if not summary.get('episodes'):
        return ['no episodes recorded -- the run produced nothing to plot']
    lines = [f'  episodes      : {summary["episodes"]}',
             f'  total steps   : {summary["total_steps"]:,}',
             f'  reward        : {summary["reward_first_10pct"]:+.3f} '
             f'-> {summary["reward_last_10pct"]:+.3f}  (first/last 10%)',
             f'  episode length: {summary["length_last_10pct"]:.1f} '
             f'steps at the end']
    if 'success_last_10pct' in summary:
        lines.append(f'  success rate  : {summary["success_last_10pct"]:.0%} '
                     f'over the last 10% of episodes')
        sct = summary.get('steps_to_threshold')
        first = summary.get('steps_first_crossing')
        lines.append(f'  steps to {summary["threshold"]:.0%}  : '
                     + (f'{sct:,} (sustained)' if sct is not None
                        else 'never held to the end'))
        if first is not None and sct is not None and first < sct:
            lines.append(f'  first touched : {first:,} -- the gap is '
                         f'exploration luck, not learning')
    return lines
