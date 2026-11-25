"""Which place cells is the policy actually using, here, right now?

A minimal-mask saliency in the sense of Fong and Vedaldi (ICCV 2017): rather
than reading gradients, SEARCH for the smallest perturbation of the input
that changes the model's answer, and call that the explanation. Their
version masks pixels; this one masks place cells, because that is what the
agent sees -- 1200-odd activations, never an image.

THE PERTURBATION IS THE WHOLE DESIGN, and for a population code it is not
obvious. Their argument is that the perturbed input must stay in the space
the model was trained on, or the "explanation" is partly a measurement of
off-manifold behaviour. Here the observation is ROW-NORMALISED: it sums to
one, and every cell is a share of a total. So:

  - Zeroing a cell is not deletion, it is a claim that the total dropped.
  - The uninformative value for a share is the uniform share, 1/K. That is
    the maximum-entropy population code: "something is here, I cannot say
    what". It is the default reference below.
  - After blending, the vector is renormalised back to sum one, because
    everything downstream of the encoder assumes that and the Box space
    declares it.

Renormalising means deleting one cell moves every other cell a little. That
is uncomfortable and it is also just true of a population code: there is no
way to remove a share without the remaining shares growing. Stating it is
better than pretending a sparse code was being masked.

TWO PASSES, and the second is the one to trust.

The optimiser gives every cell a continuous deletion weight, which is a
ranking, not a set. A ranking cannot be reported as "the cells that
mattered" -- a threshold would be arbitrary and a soft mask is not something
the policy ever sees. So the ranking is then VERIFIED: delete the top-k and
ask whether the greedy action actually changed, and search for the smallest
k that does. What comes back is a set with a property you can state in a
caption -- remove these k cells and the agent chooses differently -- rather
than a heatmap to squint at.

The search is a bracket and bisect over k, not a scan, because the forward
passes are the cost. Flipping is not strictly monotone in k, so this finds a
small sufficient set rather than a provably minimal one; the docstring of
`smallest_flipping_set` says so again where it matters.
"""

import numpy as np
import torch


def action_logits(policy, x):
    """Logits over the discrete action set, differentiable in x."""
    if hasattr(policy, 'get_distribution'):                 # PPO, A2C
        return policy.get_distribution(x).distribution.logits
    if hasattr(policy, 'q_net'):                            # DQN
        return policy.q_net(x)
    raise TypeError(f'no way to get action scores from {type(policy).__name__}')


def reference_vector(x, kind='uniform'):
    """The value a deleted cell is replaced BY.

    'uniform' is 1/K everywhere: the population code that carries no
    information while remaining a valid distribution. 'zero' is the literal
    deletion, kept because it is what most saliency code does and the
    difference is worth being able to show.
    """
    if kind == 'uniform':
        return torch.full_like(x, 1.0 / x.shape[-1])
    if kind == 'zero':
        return torch.zeros_like(x)
    raise ValueError(f'unknown reference {kind!r}')


def perturb(x, delete, ref, renormalize=True):
    """x with `delete` (0..1 per cell) blended toward the reference."""
    out = x * (1.0 - delete) + ref * delete
    if renormalize:
        out = out / out.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return out


def deletion_weights(policy, obs, reference='uniform', l1=3e-3, steps=300,
                     lr=0.05, renormalize=True, seed=0):
    """Per-cell deletion weight in [0, 1]; higher means 'the action needs it'.

    Optimises Fong and Vedaldi's deletion objective: push down the logit of
    the action the policy currently takes, while keeping the deleted set
    small. l1 is the trade-off -- larger means a smaller, blunter set.
    """
    torch.manual_seed(seed)
    x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
    ref = reference_vector(x, reference)

    with torch.no_grad():
        action = int(action_logits(policy, x).argmax(dim=-1))

    # Start near zero deletion, so the search grows a set rather than
    # carving one out of everything.
    theta = torch.full_like(x, -3.0).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        delete = torch.sigmoid(theta)
        logits = action_logits(policy, perturb(x, delete, ref, renormalize))
        # log_softmax, not the raw logit: a logit can be driven down by
        # dragging every action down together, which explains nothing. The
        # normalised score only falls when this action loses to another.
        score = torch.log_softmax(logits, dim=-1)[0, action]
        (score + l1 * delete.mean()).backward()
        opt.step()

    return action, torch.sigmoid(theta).detach().squeeze(0).numpy()


def smallest_flipping_set(policy, obs, order, reference='uniform',
                          renormalize=True, cap=None):
    """Fewest top-ranked cells whose deletion changes the greedy action.

    Bracket then bisect over k. Flipping is not guaranteed monotone in k --
    deleting more cells can in principle restore the original action -- so
    this returns a small SUFFICIENT set, not a certified minimum. Every
    returned set has been checked by a forward pass, which is the property
    worth reporting.

    Returns (k, new_action) or (None, None) if nothing up to `cap` flips it.
    """
    x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
    ref = reference_vector(x, reference)
    cap = int(cap or len(order))

    def flipped(k):
        delete = torch.zeros_like(x)
        delete[0, order[:k]] = 1.0
        with torch.no_grad():
            a = int(action_logits(
                policy, perturb(x, delete, ref, renormalize)).argmax(dim=-1))
        return a

    with torch.no_grad():
        base = int(action_logits(policy, x).argmax(dim=-1))

    hi, found = 1, None
    while hi <= cap:                                        # bracket upward
        a = flipped(hi)
        if a != base:
            found = (hi, a)
            break
        hi *= 2
    if found is None:
        return None, None

    lo = hi // 2                                            # known not to flip
    best_k, best_a = found
    while lo + 1 < best_k:                                  # bisect down
        mid = (lo + best_k) // 2
        a = flipped(mid)
        if a != base:
            best_k, best_a = mid, a
        else:
            lo = mid
    return best_k, best_a


def minimal_cells(policy, obs, reference='uniform', l1=3e-3, steps=300,
                  lr=0.05, renormalize=True, cap=None, seed=0):
    """The whole thing: rank by optimisation, then verify by deletion.

    Returns a dict with the action taken, the verified set and its weights.
    """
    action, weights = deletion_weights(policy, obs, reference=reference,
                                       l1=l1, steps=steps, lr=lr,
                                       renormalize=renormalize, seed=seed)
    order = np.argsort(-weights)
    k, new_action = smallest_flipping_set(policy, obs, order,
                                          reference=reference,
                                          renormalize=renormalize, cap=cap)
    cells = order[:k] if k else np.array([], dtype=int)
    return dict(action=action, n_cells=(k or 0), cells=cells,
                flipped_to=new_action, weights=weights[cells],
                activations=np.asarray(obs, dtype=np.float32)[cells],
                all_weights=weights)
