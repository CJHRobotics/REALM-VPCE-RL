"""Stopping training when there is nothing left to learn, or no sign of it.

Two reasons to stop early, and both save renders, which is the expensive
thing here. An agent that has converged learns nothing more from another
20,000 steps. An agent that has shown no sign of succeeding after a long
while is unlikely to start, and a sweep across representations will contain
conditions that simply do not work -- discovering that in an hour instead of
a day is the difference between a sweep that finishes and one that does not.

Either way the run ends with a reason recorded, so the figures and the
result file say which happened rather than leaving it to be inferred from a
step count.
"""

from collections import deque

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:                                         # pragma: no cover
    BaseCallback = object


class ConvergenceStopper(BaseCallback):
    """Stop on sustained success, or give up after `max_episodes`.

    Parameters
    ----------
    window : int
        Episodes in the trailing window the success rate is measured over.
    threshold : float
        Success rate the window must reach to count as converged.
    min_episodes : int
        Never declare convergence before this many episodes. This is not
        caution for its own sake -- early in training epsilon is near 1, and
        in an arena where a random walk reaches the goal reasonably often
        the trailing success rate can clear a high bar on luck alone. Left
        unguarded, a run would "converge" during exploration and stop before
        learning anything.
    max_episodes : int or None
        Give up after this many episodes regardless. None leaves the
        timestep budget as the only limit.
    NOT verifying here, deliberately
        An earlier version took a `verify` callable and ran deterministic
        rollouts from inside this callback. That is wrong, and it raised
        "Tried to step environment that needs reset": a callback runs in the
        middle of SB3's rollout collection, and resetting or stepping the
        training environment from there destroys the episode state the
        VecEnv and Monitor are holding. Had it not raised, it would have
        been worse -- training would have resumed from an observation SB3
        did not take, corrupting the run silently.

        Verification belongs BETWEEN calls to learn(), where the environment
        is idle. This callback stops on the training criterion and records
        why; the caller then checks the greedy policy and resumes training
        if it is not good enough. See the loop in the rl_agent controller.
    """

    def __init__(self, window=100, threshold=0.8, min_episodes=300,
                 max_episodes=None, verbose=1):
        super().__init__(verbose)
        self.window = int(window)
        self.threshold = float(threshold)
        self.min_episodes = int(min_episodes)
        self.max_episodes = max_episodes
        self.successes = deque(maxlen=self.window)
        self.episodes = 0
        # Set by the caller after a failed deterministic check, so the next
        # stretch of training does not stop again on the same window.
        self.min_episodes_next = 0
        self.verifications = []
        # Set when training stops early: 'converged' or 'gave_up'. Left None
        # when the timestep budget ran out first.
        self.stop_reason = None
        self.stop_episode = None

    @property
    def success_rate(self):
        if not self.successes:
            return 0.0
        return sum(self.successes) / len(self.successes)

    def _on_step(self):
        for info in self.locals.get('infos', []) or []:
            # Monitor attaches this on the step an episode ends, and copies
            # the requested info_keywords into it.
            ep = info.get('episode')
            if ep is None:
                continue
            self.episodes += 1
            self.successes.append(float(ep.get('is_success', 0.0)))

            if (self.episodes >= max(self.window, self.min_episodes)
                    and len(self.successes) == self.window
                    and self.success_rate >= self.threshold
                    and self.episodes >= self.min_episodes_next):
                rate = self.success_rate
                self.stop_reason = 'converged'
                self.stop_episode = self.episodes
                if self.verbose:
                    print(f'\n[stop] converged: {rate:.0%} success after '
                          f'{self.episodes} episodes / {self.num_timesteps} '
                          f'steps', flush=True)
                return False

            if self.max_episodes and self.episodes >= self.max_episodes:
                self.stop_reason = 'gave_up'
                self.stop_episode = self.episodes
                if self.verbose:
                    print(f'\n[stop] no convergence after {self.episodes} '
                          f'episodes: {self.success_rate:.0%} success over the '
                          f'last {len(self.successes)}', flush=True)
                return False
        return True

    def summary(self):
        return dict(
            stop_reason=self.stop_reason or 'budget_exhausted',
            episodes=self.episodes,
            final_success_rate=self.success_rate,
            window=self.window,
            threshold=self.threshold,
            # Every time training looked converged and the caller checked
            # the greedy policy. A long list means the policy kept
            # oscillating, which is worth knowing when reading the curve.
            verifications=self.verifications,
        )
