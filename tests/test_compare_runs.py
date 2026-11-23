"""Tests for the cross-run comparison.

The figure is the easy part. What is worth testing is the bookkeeping around
it: that a run directory is read the way the training job writes it, that
learning time is taken as the sustained measure, and above all that a
comparison across runs differing in more than the grouping variable is
flagged. That last one is the difference between a result and an artefact,
and it is silent if nobody checks.

    python3 tests/test_compare_runs.py
"""

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'analysis'))

import compare_runs


def make_run(root, tag, successes, algo='DQN', seed=0, ep_len=10, **cfg):
    """Write a run directory shaped exactly as the training job leaves one."""
    d = os.path.join(root, tag)
    os.makedirs(d, exist_ok=True)
    base = dict(maze='circ_lm8_r10', ensemble='e.h5', n_cells=160,
                max_steps=120, teleport=True, shaping=0.0,
                algo=algo, seed=seed, tag=tag)
    base.update(cfg)
    with open(os.path.join(d, 'config.json'), 'w') as f:
        json.dump(base, f)
    with open(os.path.join(d, 'monitor.csv'), 'w') as f:
        f.write('#{"t_start": 0, "env_id": "WebotsEnv"}\n')
        f.write('r,l,t,is_success\n')
        for i, s in enumerate(successes):
            f.write(f'{float(s):.1f},{ep_len},{i * 0.1:.1f},{float(s):.1f}\n')
    with open(os.path.join(d, 'result.json'), 'w') as f:
        json.dump(dict(success_after=float(np.mean(successes[-50:])),
                       wall_seconds=3600.0, stop_reason='converged',
                       paths=dict(starts=6, reached=6)), f)
    return d


def test_loads_a_run_directory():
    with tempfile.TemporaryDirectory() as root:
        make_run(root, 'a', [0] * 100 + [1] * 300)
        run = compare_runs.load_run(os.path.join(root, 'a'))
        assert run is not None, 'did not load'
        assert run['tag'] == 'a'
        assert len(run['df']) == 400
        assert run['config']['algo'] == 'DQN'
        assert run['result']['stop_reason'] == 'converged'
    return 'reads config, monitor and result as the job writes them'


def test_incomplete_run_is_skipped_not_fatal():
    """A job killed before writing result.json still has a learning curve."""
    with tempfile.TemporaryDirectory() as root:
        d = make_run(root, 'partial', [1] * 200)
        os.remove(os.path.join(d, 'result.json'))
        run = compare_runs.load_run(d)
        assert run is not None, 'a run without result.json should still load'
        assert run['result'] == {}
        m = compare_runs.run_metrics(run)
        assert m['eval_success'] is None
        assert m['steps_to_threshold'] is not None
        # And a directory with nothing in it is skipped rather than crashing.
        empty = os.path.join(root, 'empty')
        os.makedirs(empty)
        assert compare_runs.load_run(empty) is None
    return 'a half-finished run still contributes its curve'


def test_metrics_use_the_sustained_measure():
    """A lucky early crossing must not be reported as the learning time."""
    with tempfile.TemporaryDirectory() as root:
        series = [1] * 120 + [0] * 300 + [1] * 300
        run = compare_runs.load_run(make_run(root, 'spiky', series))
        m = compare_runs.run_metrics(run)
        assert m['first_crossing'] is not None
        assert m['steps_to_threshold'] > m['first_crossing'], (
            f'sustained {m["steps_to_threshold"]} should be later than '
            f'first {m["first_crossing"]}')
    return (f'sustained {m["steps_to_threshold"]:,} vs first '
            f'{m["first_crossing"]:,}')


def test_confounded_comparison_is_flagged():
    """Two algorithms on different arenas is not an algorithm comparison."""
    with tempfile.TemporaryDirectory() as root:
        make_run(root, 'dqn', [1] * 300, algo='DQN', maze='circ_lm8_r10')
        make_run(root, 'ppo', [1] * 300, algo='PPO', maze='circ_lm8_r3')
        runs = [compare_runs.load_run(os.path.join(root, t))
                for t in ('dqn', 'ppo')]
        issues = compare_runs.check_controlled(runs, by='algo')
        assert any('maze' in i for i in issues), f'maze not flagged: {issues}'

        # And a clean comparison raises nothing.
        make_run(root, 'ppo2', [1] * 300, algo='PPO', maze='circ_lm8_r10')
        clean = [compare_runs.load_run(os.path.join(root, t))
                 for t in ('dqn', 'ppo2')]
        assert not compare_runs.check_controlled(clean, by='algo'), \
            'flagged a comparison that is actually controlled'
    return 'differing arenas flagged, matched runs not'


def test_grouping_field_itself_is_not_flagged():
    """Comparing by n_cells must not warn that n_cells differs."""
    with tempfile.TemporaryDirectory() as root:
        make_run(root, 'small', [1] * 300, n_cells=29)
        make_run(root, 'big', [1] * 300, n_cells=222)
        runs = [compare_runs.load_run(os.path.join(root, t))
                for t in ('small', 'big')]
        assert not compare_runs.check_controlled(runs, by='n_cells')
        # But it IS a confound when comparing by something else.
        assert compare_runs.check_controlled(runs, by='algo')
    return 'the grouping variable is exempt, and only that one'


def test_figure_renders_with_a_failed_run():
    """A run that never converged belongs on the chart, not missing from it."""
    with tempfile.TemporaryDirectory() as root:
        make_run(root, 'good', [0] * 100 + [1] * 400, algo='DQN', seed=0)
        make_run(root, 'good2', [0] * 150 + [1] * 350, algo='DQN', seed=1)
        make_run(root, 'bad', [0] * 500, algo='PPO', seed=0)
        runs = [compare_runs.load_run(os.path.join(root, t))
                for t in ('good', 'good2', 'bad')]
        out = os.path.join(root, 'fig.png')
        assert compare_runs.compare(runs, by='algo', out_path=out) == out
        assert os.path.exists(out) and os.path.getsize(out) > 5000
        rows = [compare_runs.run_metrics(r) for r in runs]
        assert compare_runs.table(rows)
        failed = [r for r in rows if r['steps_to_threshold'] is None]
        assert len(failed) == 1, 'the failing run should report None'
    return 'renders two groups including a run that never converged'


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
