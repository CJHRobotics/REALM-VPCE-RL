"""The stopping rule, exercised on the failure that prompted it.

A run was declared converged having reached all four starts, while one of
them took 148 actions against an optimum of 10. These tests hold the rule
that would have caught it, and the three properties it rests on:

  * a start is judged on ITS OWN optimum, so starts needing 10 and 18
    actions are comparable;
  * every start must clear the bar, because a pooled mean lets three good
    starts carry a bad one;
  * a timeout costs the full episode cap, so failure is inside the metric
    and needs no separate success threshold.

    python3 tests/test_stopping_rule.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realm_tools.rl_lib.callbacks import ExtraStepStopper, extra_step_ratio

OPTIMAL = {0: 16, 1: 10, 2: 18, 3: 10}      # circ_lm10_r10_o20, measured
CAP = 200


class Harness(ExtraStepStopper):
    """The callback without SB3: feed it episodes, see when it stops."""

    num_timesteps = 0

    def feed(self, start, length):
        self.locals = {'infos': [{'episode': {'l': length},
                                  'start_index': start}]}
        return self._on_step()

    def run(self, lengths, episodes=600):
        for i in range(episodes):
            s = i % len(lengths)
            if not self.feed(s, lengths[s]):
                return self.stop_reason
        return None


def make(**kw):
    kw.setdefault('threshold', 1.0)
    kw.setdefault('window', 100)
    kw.setdefault('min_per_start', 10)
    kw.setdefault('max_episodes', 1000)
    return Harness(OPTIMAL, verbose=0, **kw)


def test_the_ratio_is_relative_to_each_start():
    """20 steps is excellent from a start needing 18 and poor from one
    needing 10. A rule on raw step counts cannot say that."""
    assert extra_step_ratio(18, 18) == 0.0
    assert extra_step_ratio(20, 18) < 0.2
    assert extra_step_ratio(20, 10) == 1.0
    assert extra_step_ratio(9, 10) < 0                # better than A*
    assert extra_step_ratio(50, None) is None         # unscorable start
    return 'ESR is per-start, and negative when the policy beats A*'


def test_one_bad_start_blocks_convergence():
    """The actual run: starts 0, 2, 3 near optimal and start 1 at 148."""
    st = make()
    assert st.run({0: 20, 1: 148, 2: 20, 3: 12}) is None, (
        f'converged anyway, at episode {st.stop_episode}')
    means = {s: round(m, 2) for s, (_n, m) in st.per_start().items()}
    assert means[1] > 10, means
    return f'three good starts cannot carry one at ESR {means[1]}'


def test_it_converges_when_every_start_is_good():
    st = make()
    assert st.run({0: 20, 1: 14, 2: 20, 3: 12}) == 'converged'
    assert st.worst < 1.0
    return f'converged at episode {st.stop_episode}, worst {st.worst:.2f}'


def test_a_timeout_is_charged_at_the_cap():
    """No separate success threshold: a window of failures scores ~19 and
    can never pass, because a failed episode IS a full-length episode."""
    st = make(max_episodes=250)
    assert st.run({s: CAP for s in OPTIMAL}) == 'episode_budget_exhausted'
    assert st.worst > 10, st.worst
    return f'a window of timeouts averages ESR {st.worst:.1f}'


def test_a_start_too_rarely_seen_is_not_judged():
    """A start seen twice cannot pass on two lucky episodes."""
    st = make(min_per_start=10)
    for i in range(400):
        # start 1 appears once every fifty episodes
        s = 1 if i % 50 == 0 else (i % 3)
        if not st.feed(s, {0: 20, 1: 11, 2: 20, 3: 12}[s]):
            break
    assert st.stop_reason is None, 'judged a start it had barely seen'
    return 'a start under min_per_start holds the decision open'


def test_it_can_be_resumed_after_a_failed_greedy_check():
    """The caller raises the bar so the next stretch cannot stop on the
    same window. Without it, a policy whose greedy routes are worse than
    its sampled ones would stop again immediately, every time."""
    st = make()
    assert st.run({0: 20, 1: 14, 2: 20, 3: 12}) == 'converged'
    first = st.stop_episode
    st.stop_reason = None
    st.min_episodes_next = st.episodes + st.window
    assert st.run({0: 20, 1: 14, 2: 20, 3: 12}, episodes=50) is None, (
        'stopped again inside the window it was told to train through')
    assert st.run({0: 20, 1: 14, 2: 20, 3: 12}, episodes=200) == 'converged'
    assert st.stop_episode >= first + st.window
    return f'resumed at {first}, could not stop again before {first + st.window}'


def test_the_episode_cap_is_the_budget():
    st = make(max_episodes=120)
    assert st.run({s: CAP for s in OPTIMAL}, episodes=600) == \
        'episode_budget_exhausted'
    assert st.episodes == 120, st.episodes
    return 'the run ends on episodes, not steps'


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
