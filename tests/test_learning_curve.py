"""Tests for the learning-time measure.

`steps_to_threshold` is the number the whole experiment compares across
representations, so it is worth more scrutiny than the plotting around it.
The case that matters is the one found on a real run: an agent whose success
rate crosses the threshold early on epsilon-greedy luck, collapses when the
half-trained policy takes over, and only later earns the threshold back. A
first-crossing measure reports the luck.

    python3 tests/test_learning_curve.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realm_tools.rl_lib import learning_curve


def _df(successes, length=10):
    """A Monitor-shaped frame from a sequence of 0/1 episode outcomes."""
    import pandas as pd
    s = np.asarray(successes, dtype=float)
    df = pd.DataFrame({'r': s, 'l': np.full(len(s), length),
                       'is_success': s})
    df['episode'] = np.arange(len(df))
    df['cum_steps'] = df['l'].cumsum()
    return df


def test_sustained_ignores_a_lucky_spike():
    """The real-run shape: lucky early crossing, collapse, genuine recovery."""
    series = ([1] * 120          # lucky: random walk finds the goal
              + [0] * 300        # collapse: epsilon anneals, policy is bad
              + [1] * 300)       # learned
    df = _df(series)
    first = learning_curve.steps_to_threshold(df, 0.8, window=100,
                                              sustained=False)
    held = learning_curve.steps_to_threshold(df, 0.8, window=100,
                                             sustained=True)
    assert first is not None and held is not None
    assert held > first, f'sustained ({held}) should be later than first ({first})'
    # The sustained answer must fall in the final, genuinely-good stretch.
    assert held >= 420 * 10 * 0.8, f'sustained answer {held} is too early'
    return f'first {first:,} vs sustained {held:,}'


def test_never_reached_is_none():
    df = _df([0] * 400)
    assert learning_curve.steps_to_threshold(df, 0.8, window=100) is None
    return 'a run that never succeeds reports None, not 0'


def test_reached_but_not_held_is_none():
    """Ending below the threshold means it was not learned, however good the
    middle looked. Reporting the mid-run peak would credit a run that fell
    apart."""
    df = _df([1] * 300 + [0] * 300)
    assert learning_curve.steps_to_threshold(df, 0.8, window=100) is None
    assert learning_curve.steps_to_threshold(df, 0.8, window=100,
                                             sustained=False) is not None
    return 'crossing then collapsing reports None when sustained'


def test_clean_run_agrees_both_ways():
    df = _df([0] * 200 + [1] * 400)
    first = learning_curve.steps_to_threshold(df, 0.8, window=100,
                                              sustained=False)
    held = learning_curve.steps_to_threshold(df, 0.8, window=100)
    assert first == held, f'{first} != {held} on a monotone run'
    return 'a clean monotone run gives the same answer either way'


def test_plot_writes_a_figure(tmp='/tmp'):
    import tempfile
    df = _df([0] * 200 + [1] * 400)
    with tempfile.TemporaryDirectory() as d:
        csv = os.path.join(d, 'monitor.csv')
        # Monitor's own format: a '#' JSON header line, then the table.
        with open(csv, 'w') as f:
            f.write('#{"t_start": 0, "env_id": "test"}\n')
            df[['r', 'l', 'is_success']].assign(t=0.0).to_csv(f, index=False)
        png, summary = learning_curve.plot(csv, os.path.join(d, 'c.png'))
        assert png and os.path.exists(png), 'no figure written'
        assert summary['episodes'] == len(df)
        assert summary['steps_to_threshold'] is not None
        lines = learning_curve.summarise(summary)
        assert any('steps to' in l for l in lines)
    return 'plot() parses a Monitor CSV and writes a figure'


def test_truncated_csv_still_parses():
    """A walltime kill leaves a half-written final line; it must not throw."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        csv = os.path.join(d, 'monitor.csv')
        with open(csv, 'w') as f:
            f.write('#{"t_start": 0}\n')
            f.write('r,l,t,is_success\n')
            for i in range(50):
                f.write(f'1.0,10,{i * 0.1:.1f},1.0\n')
            f.write('0.5,7')                    # cut off mid-row
        df = learning_curve.load_monitor(csv)
        assert len(df) >= 50, f'lost rows: {len(df)}'
    return 'a truncated Monitor CSV loads without losing the run'


def _drive(stop, success, max_calls=400):
    """Feed episodes until the callback asks to stop, as SB3 would.

    SB3 ends training the moment _on_step returns False, so a test that
    keeps calling past it is testing something that cannot happen -- and
    will miscount episodes.
    """
    ep = {'episode': {'r': float(success), 'l': 5, 'is_success': float(success)}}
    stop.locals = {'infos': [ep]}
    for i in range(max_calls):
        if stop._on_step() is False:
            return i + 1
    return None


def test_stopper_can_be_resumed_after_a_failed_check():
    """The caller raises the bar and training continues.

    Verification cannot happen inside the callback -- resetting the
    training env mid-rollout raises "Tried to step environment that needs
    reset". So the stopper ends a stretch, the caller checks the greedy
    policy, and on a failure sets min_episodes_next and resumes. This is
    that contract.
    """
    from realm_tools.rl_lib.callbacks import ConvergenceStopper
    stop = ConvergenceStopper(window=10, threshold=0.8, min_episodes=10,
                              verbose=0)
    stop.num_timesteps = 0

    first = _drive(stop, success=1)
    assert first == 10, f'first stretch ended at {first}'
    assert stop.stop_reason == 'converged'

    # The caller checked the greedy policy, it failed, so it resumes with
    # the bar raised.
    stop.stop_reason = None
    stop.min_episodes_next = stop.episodes + 25
    second = _drive(stop, success=1)
    assert second is not None, 'never stopped again'
    assert stop.episodes == 35, (
        f'resumed to episode {stop.episodes}; should have run 25 more '
        f'before stopping again')
    return 'a failed check resumes training with the bar raised'


def test_stopper_without_verify_stops_on_training_alone():
    """The old behaviour is still available when no check is supplied."""
    from realm_tools.rl_lib.callbacks import ConvergenceStopper
    stop = ConvergenceStopper(window=10, threshold=0.8, min_episodes=10,
                              verbose=0)
    stop.num_timesteps = 0
    stopped_at = _drive(stop, success=1)
    assert stopped_at == 10, f'stopped after {stopped_at}'
    assert stop.stop_reason == 'converged'
    return 'stops on the training window when no verifier is given'


def test_stopper_gives_up():
    from realm_tools.rl_lib.callbacks import ConvergenceStopper
    stop = ConvergenceStopper(window=10, threshold=0.9, min_episodes=5,
                              max_episodes=20, verbose=0)
    stop.num_timesteps = 0
    stopped_at = _drive(stop, success=0)
    assert stopped_at == 20, f'gave up after {stopped_at}, expected 20'
    assert stop.stop_reason == 'gave_up'
    assert stop.summary()['episodes'] == 20
    return 'gives up after the episode limit with no convergence'


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
