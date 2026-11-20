"""Agent construction and hyperparameters.

Kept here rather than in the Webots controller for two reasons. The
controller runs top-level code that needs a live simulation, so anything
defined in it cannot be imported, inspected or tested from anywhere else --
including by the test suite that runs off-cluster. And the hyperparameters
are a condition of the experiment: every run in a sweep must use the same
ones, or tuning effort gets confounded with the representation being
measured, so they belong in one place that every entry point reads.

Vanilla DQN is the default. The reasoning, briefly, because it is the kind
of choice that gets silently reversed later: environment steps here cost
eight camera renders and eight HOG extractions, so the scarce resource is
transitions, not compute. Off-policy learning reuses each transition many
times from the replay buffer; PPO discards its rollouts. That is worth
roughly an order of magnitude in environment steps, and it matters more
than PPO's better-behaved optimisation.
"""

# Environment steps are expensive and gradient steps are nearly free. Every
# departure from SB3's defaults below follows from that one asymmetry, or
# from the reward being sparse.
DQN_KWARGS = dict(
    # Replay ratio. SB3 defaults to train_freq=4, gradient_steps=1 -- one
    # update per four steps, tuned for Atari where stepping is cheap. Here
    # it is backwards: update every step, twice.
    train_freq=1,
    gradient_steps=2,
    # Larger than any run we intend, so nothing is evicted. An observation
    # is a few hundred float32s, so this costs megabytes, not gigabytes.
    buffer_size=200_000,
    batch_size=64,
    learning_starts=1_000,
    # SB3 defaults to 10,000. Over a short run that means the target network
    # barely moves and the agent spends most of training chasing a stale
    # estimate.
    target_update_interval=500,
    # Sparse reward: the goal has to be stumbled into before anything can be
    # learned at all, so explore over a third of the budget rather than a
    # tenth, and keep a floor under it.
    exploration_fraction=0.3,
    exploration_final_eps=0.05,
    learning_rate=1e-3,
    gamma=0.99,
)

# Left at SB3's defaults deliberately. PPO is the on-policy control here,
# not a tuned competitor, and tuning one arm of a comparison and not the
# other is worse than tuning neither.
PPO_KWARGS = dict()


def algo_kwargs(algo):
    """Hyperparameters for an algorithm name, or {} if it has none set."""
    return dict({'DQN': DQN_KWARGS, 'PPO': PPO_KWARGS}.get(algo.upper(), {}))


def make_agent(algo, env, seed=0, verbose=1, **overrides):
    """Build an SB3 agent with this project's hyperparameters.

    `overrides` wins over the defaults, so a sweep can vary one setting
    without copying the rest -- but anything overridden is a difference
    between runs, and belongs in the run's config.json.
    """
    algo = algo.upper()
    # Imported here so that importing this module does not pull in torch:
    # the test suite and the analysis code have no use for it.
    from stable_baselines3 import PPO, DQN

    cls = {'PPO': PPO, 'DQN': DQN}.get(algo)
    if cls is None:
        raise ValueError(f'unknown algorithm {algo!r}; expected DQN or PPO')

    kwargs = algo_kwargs(algo)
    kwargs.update(overrides)
    return cls('MlpPolicy', env, verbose=verbose, seed=seed, **kwargs)
