"""Stopping training when the routes are good enough, not merely finished.

THE RULE, in full:

    An episode scores an EXTRA-STEP RATIO

        ESR = (steps taken - optimal steps) / optimal steps

    where the optimum is A* over the agent's own action set from that
    episode's own start (realm_tools/rl_lib/optimal_paths.py). 0 means the
    agent matched the best route available to it. Negative is possible --
    A* respects a body buffer the policy does not know about. Positive is
    how much further it went than it had to.

    Training stops as CONVERGED when, over the trailing `window` episodes:

      * every start has at least `min_per_start` episodes in the window, and
      * every start's MEAN ESR is below `threshold`.

    Training stops as EPISODE BUDGET EXHAUSTED at `max_episodes`.

    The caller then verifies greedily -- one deterministic rollout per
    start -- and only accepts convergence if every start's ESR clears the
    same threshold there too. Otherwise the bar is raised and training
    resumes.

WHY PER START AND NOT POOLED. A pooled mean over randomly drawn starts is
exactly the statistic that hides the failure this rule exists to catch. The
run that prompted it reached all four starts and was declared converged,
while one of them took 148 actions against an optimum of 10 -- ESR 13.8,
against 0.6, 0.1 and 0.3 for the others. Pooled, that is a mean of 3.7 and
a quarter of the window; one bad start can be carried by three good ones
and, as episodes are drawn at random, by luck in how often it comes up.
Per start, it cannot.

WHY THERE IS NO SUCCESS THRESHOLD ANY MORE. There used to be one -- 80% of
the last 100 episodes reaching the goal -- and it is now redundant and
worse than redundant. A timeout runs the full episode cap, so its length IS
the cap and its ESR is (max_steps - optimal)/optimal: at a 200-step cap and
a 10-step optimum, 19.0. Failure is already inside the metric, charged at a
rate no window of failures can average away. Keeping a separate success
threshold would mean two knobs that fight: a policy could satisfy one and
not the other, and which stopped the run would depend on their relative
settings rather than on the agent.

WHAT WENT WITH IT. The old give-up rule abandoned a run whose success rate
stayed under a floor past half the step budget. It was a success-rate
device and it goes with the success rate. The episode cap now bounds the
cost instead: a run that never learns spends `max_episodes` episodes of at
most `max_steps` each and stops, with its reason recorded. That is a
blunter bound and a far more predictable one.

NOT VERIFYING HERE, deliberately. An earlier version took a `verify`
callable and ran deterministic rollouts from inside the callback. That
raised "Tried to step environment that needs reset", because a callback
runs in the middle of SB3's rollout collection and resetting the training
environment there destroys the episode state the VecEnv and Monitor hold.
Had it not raised it would have been worse: training would have resumed
from an observation SB3 never took. Verification belongs between calls to
learn(), where the environment is idle. See the loop in the rl_agent
controller.
"""

from collections import defaultdict, deque

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:                                         # pragma: no cover
    BaseCallback = object


def extra_step_ratio(steps, optimal):
    """(steps - optimal) / optimal, or None when the start has no optimum."""
    if not optimal:
        return None
    return (float(steps) - float(optimal)) / float(optimal)


class ExtraStepStopper(BaseCallback):
    """Stop when every start's trailing mean ESR is under `threshold`.

    Parameters
    ----------
    optimal : dict
        {start index: fewest actions}, from optimal_paths.optimal_steps. A
        start mapped to None or missing is skipped: it cannot be scored, and
        scoring it against a guess would be worse than not scoring it.
    threshold : float
        Mean extra-step ratio a start must be under. 1.0 means routes
        averaging twice the optimum: one extra action for every action
        the task needs.
    window : int
        Trailing episodes the means are taken over, across all starts.
    min_per_start : int
        Episodes a start needs inside that window before it can be judged.
        Without it a start seen twice could pass on two lucky episodes.
    max_episodes : int
        Hard cap. Episodes, not steps: it is the number of ATTEMPTS the
        agent gets, which is the quantity a learning-speed comparison is
        about, and it does not shrink as the agent improves the way a step
        budget effectively does.
    """

    def __init__(self, optimal, threshold=1.0, window=100, min_per_start=10,
                 max_episodes=1000, verbose=1):
        super().__init__(verbose)
        self.optimal = {int(k): v for k, v in (optimal or {}).items()}
        self.threshold = float(threshold)
        self.window = int(window)
        self.min_per_start = int(min_per_start)
        self.max_episodes = int(max_episodes)

        self.recent = deque(maxlen=self.window)     # (start_index, esr)
        self.episodes = 0
        self.scored = 0
        self.unscored = 0
        # Raised by the caller after a failed greedy check, so the next
        # stretch cannot stop again on the same window.
        self.min_episodes_next = 0
        self.verifications = []
        self.stop_reason = None
        self.stop_episode = None

    # ------------------------------------------------------------- state

    def per_start(self):
        """{start index: (n in window, mean ESR)}."""
        acc = defaultdict(list)
        for start, esr in self.recent:
            acc[start].append(esr)
        return {s: (len(v), sum(v) / len(v)) for s, v in acc.items()}

    @property
    def worst(self):
        """The mean ESR of the start doing worst, or None if unjudgeable."""
        stats = self.per_start()
        judged = [m for _s, (n, m) in stats.items() if n >= self.min_per_start]
        return max(judged) if judged else None

    def ready(self):
        """Is there enough in the window to judge every scorable start?"""
        wanted = {s for s, o in self.optimal.items() if o}
        if not wanted:
            return False
        stats = self.per_start()
        return all(stats.get(s, (0, 0))[0] >= self.min_per_start
                   for s in wanted)

    # -------------------------------------------------------------- hook

    def _on_step(self):
        for info in self.locals.get('infos', []) or []:
            ep = info.get('episode')
            if ep is None:
                continue
            self.episodes += 1

            start = info.get('start_index')
            opt = self.optimal.get(int(start)) if start is not None else None
            # A timeout has already run the full cap, so ep['l'] IS the cap
            # and no special case is needed to charge it: the penalty is
            # built into the length of a failed episode.
            esr = extra_step_ratio(ep.get('l', 0), opt)
            if esr is None:
                self.unscored += 1
            else:
                self.scored += 1
                self.recent.append((int(start), esr))

            if (len(self.recent) == self.window
                    and self.episodes >= self.min_episodes_next
                    and self.ready()
                    and self.worst is not None
                    and self.worst < self.threshold):
                self.stop_reason = 'converged'
                self.stop_episode = self.episodes
                if self.verbose:
                    print(f'\n[stop] every start under {self.threshold:g} '
                          f'extra steps per optimal step, worst '
                          f'{self.worst:.2f}, after {self.episodes} episodes '
                          f'/ {self.num_timesteps} steps', flush=True)
                    self.report()
                return False

            if self.episodes >= self.max_episodes:
                self.stop_reason = 'episode_budget_exhausted'
                self.stop_episode = self.episodes
                if self.verbose:
                    w = self.worst
                    print(f'\n[stop] {self.episodes} episodes used. Worst '
                          f'start averages '
                          f'{"n/a" if w is None else f"{w:.2f}"} extra steps '
                          f'per optimal step, against a {self.threshold:g} '
                          f'bar.', flush=True)
                    self.report()
                return False
        return True

    def report(self):
        for s, (n, mean) in sorted(self.per_start().items()):
            mark = '' if n >= self.min_per_start else '   (too few to judge)'
            print(f'    start {s}: mean ESR {mean:+.2f} over {n} episodes'
                  f'{mark}', flush=True)

    def summary(self):
        stats = {str(s): dict(episodes=n, mean_esr=round(m, 4))
                 for s, (n, m) in sorted(self.per_start().items())}
        return dict(
            stop_reason=self.stop_reason or 'step_ceiling_reached',
            episodes=self.episodes,
            scored_episodes=self.scored,
            unscored_episodes=self.unscored,
            window=self.window,
            threshold=self.threshold,
            min_per_start=self.min_per_start,
            max_episodes=self.max_episodes,
            worst_mean_esr=self.worst,
            per_start=stats,
            optimal_steps={str(k): v for k, v in self.optimal.items()},
            # Every time training looked converged and the caller checked
            # the greedy policy. A long list means the policy kept
            # oscillating, which is worth knowing when reading the curve.
            verifications=self.verifications,
        )
