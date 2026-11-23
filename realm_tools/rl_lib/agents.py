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

# PPO was originally left at SB3's stock defaults, on the grounds that it
# was a control rather than a competitor. Once it became an arm of a
# published comparison that stopped being defensible: DQN carries eight
# task-motivated departures from its defaults and PPO carried none, which is
# the exact asymmetry the DQN comment warns against.
#
# The fix is to give PPO the SAME KIND of adjustment, not more. Of DQN's
# departures, most are algorithm-specific and have no counterpart -- buffer
# size, target update interval, learning starts. Two are about the task
# rather than the algorithm:
#
#   replay ratio    "steps are expensive, so learn more from each one".
#                   PPO's counterpart is n_epochs, passes over each rollout,
#                   and SB3 already defaults it to 10. Nothing to change.
#   exploration     "the reward is sparse, so the goal has to be found
#                   before anything can be learned". PPO's counterpart is
#                   the entropy bonus, and SB3 defaults ent_coef to 0.0 --
#                   no exploration pressure at all beyond the policy's own
#                   stochasticity. On a sparse-reward task that is a real
#                   handicap, and leaving it at 0 while DQN explores for 30%
#                   of its budget would not be a fair comparison.
#
# Everything else stays at SB3's tuned defaults. In particular the learning
# rate is NOT matched to DQN's: 3e-4 is PPO's own tuned value and 1e-3 is
# DQN's, and forcing them equal would be a false symmetry rather than a fair
# one -- the two optimise different objectives.
PPO_KWARGS = dict(
    ent_coef=0.01,
    # Named explicitly rather than left implicit, so the comparison records
    # that they were matched where they are comparable.
    gamma=0.99,
    n_epochs=10,
    n_steps=2048,
)


def pick_device(prefer=None):
    """'cuda' only if a tensor can actually be put on it, else 'cpu'.

    torch.cuda.is_available() is not enough. It reports a device count,
    which succeeds without ever creating a CUDA context -- so on a cluster
    it can return True and the first real allocation still fail with
    cudaErrorDevicesUnavailable (the GPU in exclusive-process compute mode
    with a context already held, another job occupying it, or a driver too
    old for the build in a way that only surfaces on context creation).

    SB3's device='auto' trusts is_available(), so that failure arrives
    inside Model.__init__ as an unhandled AcceleratorError and takes the
    whole run with it. Probing here converts a crash into a slower run,
    which is the right trade: measured on this task, the network is 0.15 ms
    of a 118 ms environment step, so CPU costs almost nothing while a dead
    job costs everything.

    `prefer` of 'cpu' or 'cuda' overrides the probe; None auto-detects.
    """
    import torch

    if prefer in ('cpu', 'cuda'):
        return prefer
    if not torch.cuda.is_available():
        return 'cpu'
    try:
        # Forces context creation, which is what actually fails.
        torch.zeros(1, device='cuda')
        return 'cuda'
    except Exception as exc:                                # noqa: BLE001
        print(f'  CUDA reported available but unusable, falling back to CPU: '
              f'{exc.__class__.__name__}: {exc}', flush=True)
        return 'cpu'


def algo_kwargs(algo):
    """Hyperparameters for an algorithm name, or {} if it has none set."""
    return dict({'DQN': DQN_KWARGS, 'PPO': PPO_KWARGS}.get(algo.upper(), {}))


def make_agent(algo, env, seed=0, verbose=1, device=None, **overrides):
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
    kwargs.setdefault('device', pick_device(device))
    return cls('MlpPolicy', env, verbose=verbose, seed=seed, **kwargs)
