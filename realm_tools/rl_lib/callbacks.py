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
    verify : callable or None
        Called when the trailing window first clears the threshold, and
        expected to return the fraction of DETERMINISTIC rollouts that
        succeed. Convergence is only declared if that fraction also clears
        `verify_threshold`.

        This is not belt-and-braces, it is the difference between a correct
        and an incorrect answer. Training is epsilon-greedy, and even at a
        final epsilon of 0.05 the occasional random action is enough to jolt
        an agent out of a limit cycle. A policy that oscillates between two
        states forever under argmax can therefore show a high success rate
        during training and fail completely when run deterministically.
        Observed directly: a run reporting 98% success over its last 100
        training episodes reached the goal from only 4 of 6 starts when
        rolled out deterministically, the other two cycling in place until
        the step budget ran out.
    verify_threshold : float
        Fraction of deterministic rollouts that must succeed. Defaults to
        1.0: every start, because a policy that cannot solve one of six
        starting positions has not converged on the task.
    verify_cooldown : int
        Episodes to wait before verifying again after a failed check, so a
        near-threshold run does not spend all its budget on verification.
    """

    def __init__(self, window=100, threshold=0.8, min_episodes=300,
                 max_episodes=None, verify=None, verify_threshold=1.0,
                 verify_cooldown=100, verbose=1):
        super().__init__(verbose)
        self.window = int(window)
        self.threshold = float(threshold)
        self.min_episodes = int(min_episodes)
        self.max_episodes = max_episodes
        self.verify = verify
        self.verify_threshold = float(verify_threshold)
        self.verify_cooldown = int(verify_cooldown)
        self.successes = deque(maxlen=self.window)
        self.episodes = 0
        self._next_verify_at = 0
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
                    and self.episodes >= self._next_verify_at):
                rate = self.success_rate
                if self.verify is not None:
                    checked = float(self.verify())
                    self.verifications.append(
                        dict(episode=self.episodes, training=rate,
                             deterministic=checked))
                    if checked < self.verify_threshold:
                        # Training looks converged, the greedy policy is not.
                        # Keep going; check again after the cooldown.
                        self._next_verify_at = self.episodes + self.verify_cooldown
                        if self.verbose:
                            print(f'\n[check] {rate:.0%} in training but '
                                  f'{checked:.0%} deterministic at episode '
                                  f'{self.episodes} -- not converged, '
                                  f'continuing', flush=True)
                        continue
                    rate = checked
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
            # Every time training looked converged and the greedy policy was
            # checked. A long list means the policy kept oscillating, which
            # is worth knowing when reading the learning curve.
            verifications=self.verifications,
        )
