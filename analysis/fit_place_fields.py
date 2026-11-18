#!/usr/bin/env python3
"""Fit a VPCE place-field ensemble from a collected dataset, and save it.

This is the step between data collection and the RL work: it turns an arena's
POV feature dataset into a set of place fields, and writes them in a form that
a later session -- a training run, a Webots controller -- can load and use to
turn a new observation into an activation vector.

    collect_data.sh  ->  data/vpce/collect_data/<arena>.h5     (features)
    fit_place_fields ->  data/vpce/place_fields/<name>.h5      (the ensemble)
    VisualPlaceCellEnsemble.load(...).encode(blocks)  ->  (N, K) activations

Fields are built by the rule-governed agglomeration in
`realm_tools/place_cell_lib/rules.py`: Ward linkage over the feature-space
distance matrix, then admission by three rules -- the field must fall inside
the measured size range (Rules 8, 9), be one connected patch of floor
(Rule 1), and not be ground already claimed by a larger field at the same
scale (Rule 11). The surviving nodes' feature-space means become the
ensemble's centers and their feature sigmas become its radii.

Why not k-means. `clustering.fit_kmeans` is still there and is a reasonable
baseline, but it needs K chosen in advance and every centroid it produces is
a cell whether or not it corresponds to anything spatially coherent. The
agglomeration instead admits cells on evidence, so K is an outcome.

A caveat worth being explicit about, because it bears on what the RL paper
can claim: admission consults ground-truth position. Rules 1, 8, 9 and 11 all
ask where on the floor a candidate's members lie, so fitting is spatially
supervised even though the resulting encoder runs on images alone. LAMBDA
(Rule 4) goes further and puts position directly into the merge cost, which
is why it defaults to 0 here and did not in the paper's sweep. Keep it at 0
unless you mean to make the representation partly positional.

Usage
-----
    python3 analysis/fit_place_fields.py circ_lm8_r10
    python3 analysis/fit_place_fields.py circ_lm8_r10 --channels visual
    python3 analysis/fit_place_fields.py circ_lm8_r10 --subsample 8000   # quick
    python3 analysis/fit_place_fields.py --inspect data/vpce/place_fields/...h5

Compute: the dominant cost is the N x N feature distance matrix -- about
3.6 GB and ~5e16 MACs at N = 30,129 for the full configuration. It uses the
GPU when one is visible. On a login node or a laptop, pass --subsample to
work on a random subset of locations.
"""

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from realm_tools.place_cell_lib import channels as ch
from realm_tools.place_cell_lib import rules as R
from realm_tools.place_cell_lib.place_cells import VisualPlaceCellEnsemble

# Matches the collection job's lidar handling, so features assembled here are
# the same ones the paper's analysis saw.
LIDAR_MAX_RANGE = 8.0
LIDAR_SENTINEL = -1.0


def inspect(path):
    """Print what a saved ensemble contains, and prove it round-trips."""
    ens = VisualPlaceCellEnsemble.load(path)
    print(f'{path}')
    print(f'  cells        : {ens.n_cells}')
    print(f'  feature dim  : {ens.feature_dim}')
    print(f'  channels     : {", ".join(ens.channels) if ens.channels else "(not recorded)"}')
    if ens.scales is not None:
        print(f'  scales       : {np.array2string(ens.scales, precision=4)}')
    print(f'  radii        : min {ens.radii.min():.4g}  '
          f'median {np.median(ens.radii):.4g}  max {ens.radii.max():.4g}')

    # Activating the centroids themselves is the cheapest end-to-end check:
    # every cell must respond most strongly to its own centre, so the argmax
    # of the activation matrix has to be the diagonal. A centers/radii pair
    # that got transposed, truncated or misaligned on save fails this.
    act = ens.activate(ens.centers, normalize=None)
    own = np.diag(act)
    # Maximal, not uniquely maximal: two cells at the same point in feature
    # space both respond 1.0 there, and argmax would blame the later one.
    diag_wins = int((own >= act.max(axis=1) - 1e-12).sum())
    print(f'  self-response: {diag_wins}/{ens.n_cells} cells peak at their own centre')

    dup = ens.n_cells - len({c.tobytes() for c in ens.centers})
    if dup:
        print(f'  WARNING: {dup} cells share a centre with another cell; '
              f'their activation columns are not independent')
    return 0 if diag_wins == ens.n_cells else 1


def main():
    ap = argparse.ArgumentParser(
        description='Fit and save a VPCE place-field ensemble.')
    ap.add_argument('arena', nargs='?', default='circ_lm8_r10',
                    help='arena name, e.g. circ_lm8_r10')
    ap.add_argument('--channels', default='visual',
                    help=f'channel set: {", ".join(ch.CHANNEL_SETS)} '
                         '(default: visual = hog + color_hist + spatial)')
    ap.add_argument('--lam', type=float, default=0.0,
                    help='Rule 4 spatial weight; 0 keeps merging purely '
                         'feature-driven (default: 0)')
    ap.add_argument('--subsample', type=int, default=0,
                    help='fit on this many random locations instead of all')
    ap.add_argument('--bin-m', type=float, default=R.DEFAULT_CFG['BIN_M'],
                    help='analysis bin size in metres (default: the '
                         'collection lattice spacing)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-gpu', action='store_true')
    ap.add_argument('--out', default=None,
                    help='output .h5 (default: data/vpce/place_fields/'
                         '<arena>_<channels>.h5)')
    ap.add_argument('--inspect', metavar='ENSEMBLE_H5', default=None,
                    help='describe a saved ensemble and exit')
    args = ap.parse_args()

    if args.inspect:
        return inspect(args.inspect)

    if args.channels not in ch.CHANNEL_SETS:
        ap.error(f'unknown channel set {args.channels!r}; '
                 f'choose from {", ".join(ch.CHANNEL_SETS)}')
    keys = ch.CHANNEL_SETS[args.channels]

    data_path = f'{REPO}/data/vpce/collect_data/{args.arena}.h5'
    xml_path = f'{REPO}/simulation/worlds/environments/vpce/{args.arena}.xml'
    out_path = args.out or (f'{REPO}/data/vpce/place_fields/'
                            f'{args.arena}_{args.channels}.h5')

    if not os.path.exists(data_path):
        # Naming the collection command here rather than just the path: this
        # is the first thing that goes wrong when the dataset is still on the
        # cluster, or the arena name was mistyped.
        print(f'ERROR: no dataset at {data_path}\n'
              f'  Collect it first:  sbatch slurm/collect_data.sh {args.arena}',
              file=sys.stderr)
        return 1

    # ---------------------------------------------------------------- load
    t0 = time.time()
    print(f'loading {data_path}')
    blocks, xy = ch.load_channel_blocks(
        data_path,
        lidar_max_range=LIDAR_MAX_RANGE,
        lidar_sentinel=LIDAR_SENTINEL,
        lidar_mask_channel=True,
    )
    missing = [k for k in keys if k not in blocks]
    if missing:
        print(f'ERROR: dataset has no {", ".join(missing)} block(s); '
              f'it carries {", ".join(sorted(blocks))}', file=sys.stderr)
        return 1

    xml_root = ET.parse(xml_path).getroot() if os.path.exists(xml_path) else None
    if xml_root is None:
        # The arena outline comes from the XML; without it build_env falls
        # back to the extent of the sampled points, which understates the
        # floor by the collection keep-out and shifts every area measure.
        print(f'  WARNING: {xml_path} not found -- arena geometry will be '
              f'inferred from the sampled positions')
    env = R.build_env(xy, xml_root)

    if args.subsample and args.subsample < len(xy):
        sel = np.random.default_rng(args.seed).choice(
            len(xy), args.subsample, replace=False)
        sel.sort()
        blocks = {k: v[sel] for k, v in blocks.items()}
        xy = xy[sel]
        print(f'  subsampled to {len(xy)} locations')

    N = len(xy)
    print(f'  {"disc" if env["is_circular"] else "box"}  '
          f'area={env["env_area"]:.1f} m^2  {N} locations  '
          f'({N/env["env_area"]:.1f} per m^2)   {time.time()-t0:.1f}s')

    # ----------------------------------------------------------------- fit
    device = R.pick_device(use_gpu=not args.no_gpu)
    rng = np.random.default_rng(args.seed)

    # Positional normaliser for Rule 4. Computed even at lam=0 because
    # build_bank takes it either way; at lam=0 it has no effect.
    ij = rng.integers(0, N, size=(2, min(2_000_000, N * N)))
    keep = ij[0] != ij[1]
    xy_med = float(np.median(((xy[ij[0][keep]] - xy[ij[1][keep]]) ** 2).sum(1)))
    xy_med = xy_med if xy_med > 0 else 1.0

    # return_scales: these divisors are measured on THIS dataset and must be
    # saved, or nothing encoded later lands in the same space. See
    # channels.assemble.
    X, scales = ch.assemble(blocks, keys, normalize=True, return_scales=True)
    print(f'  feature matrix: {X.shape}  ({X.nbytes/1e9:.2f} GB)')
    for name, s in zip(keys, scales):
        print(f'    scale {name:11s} {s:.6g}')

    D2_feat = R.feature_sq_distances(X, device=device)
    feat_med = R._median_offdiag(D2_feat, rng)
    print(f'  median feature d^2: {feat_med:.4g}')

    cfg = dict(LAMBDA=args.lam, BIN_M=args.bin_m, RANDOM_SEED=args.seed,
               USE_GPU=not args.no_gpu)
    bank_df, kept_mu, report = R.build_bank(
        X, xy, env, D2_feat, feat_med, xy_med, cfg=cfg, device=device,
        tag=f'{args.arena}/{args.channels}')

    if len(bank_df) == 0:
        print('\nERROR: the rules admitted no fields, so there is no ensemble '
              'to save.\n  The funnel above says which rule rejected what. A '
              'channel with no\n  spatial locality (check median_sigma_ratio '
              'near 1) is the usual cause.', file=sys.stderr)
        return 1

    # --------------------------------------------------------------- save
    # The admitted nodes' feature-space means are the cell centres; their
    # feature sigmas are the RBF widths. This is the whole translation from
    # "measured place fields" to "an encoder that answers for new input".
    #
    # One wrinkle, and it matters. The bank holds one row per SUBFIELD, but a
    # multifield unit is a single node whose response happens to cover two or
    # more disconnected patches of floor -- and kept_mu is indexed per node.
    # Taking the bank's rows directly would therefore emit the same centre and
    # the same radius two or three times over. As a measurement that is right:
    # each subfield is a separately measured field and the paper counts them
    # that way. As an encoder it is wrong twice -- the duplicate columns are
    # identical for every input, so they carry no information, and because
    # `normalize='row'` makes each activation row sum to 1, a cell duplicated
    # k times takes k times the share of the population code. A silent
    # reweighting of the observation space by how fragmented a field happened
    # to be is not something to hand an RL agent.
    #
    # So the ensemble gets one cell per distinct node, while the bank keeps
    # every subfield row and gains a cell_index column pointing back.
    node_ids = bank_df['node_id'].to_numpy()
    _, first, inverse = np.unique(node_ids, return_index=True,
                                  return_inverse=True)
    order = np.sort(first)                       # keep bank order, not sorted ids
    remap = {int(node_ids[i]): c for c, i in enumerate(order)}
    bank_df['cell_index'] = [remap[int(n)] for n in node_ids]

    n_sub, n_cells = len(bank_df), len(order)
    if n_cells != n_sub:
        print(f'\n  {n_sub} subfields over {n_cells} distinct units -- the '
              f'ensemble gets {n_cells} cells,\n  one per unit; the bank keeps '
              f'all {n_sub} rows with a cell_index column.')

    ensemble = VisualPlaceCellEnsemble(
        centers=kept_mu[order],
        radii=bank_df['sigma_feature'].to_numpy()[order],
        channels=keys,
        scales=scales,
    )
    ensemble.save(out_path, method=f'agglomerative(lam={args.lam:g})',
                  maze=args.arena)

    # The bank alongside it: one row per cell, in the same order as the
    # centres, carrying what the field looks like on the floor -- area,
    # elongation, centroid, distance to wall. The ensemble does not need it,
    # but every question about WHY a cell exists is answered from it.
    bank_path = out_path.replace('.h5', '_bank.csv')
    bank_df.to_csv(bank_path, index=False)

    area = bank_df['area_env_m2']
    print(f'\n  {n_sub} fields admitted over {n_cells} cells  '
          f'(from {report["n_candidates"]} candidates)')
    print(f'  field area  : min {area.min():.3g}  median {area.median():.3g}  '
          f'max {area.max():.3g} m^2')
    # coverage is per scale band, not one number: a band covering little of
    # the floor is informative on its own (Rule 12 used to delete such bands
    # and no longer does), so report them rather than collapsing them.
    cov = report['coverage']
    if isinstance(cov, dict):
        print('  coverage    : ' + '  '.join(
            f'B{b}={v:.0%}' for b, v in sorted(cov.items())))
    else:
        print(f'  coverage    : {cov:.1%} of the floor')
    print(f'  multifield  : {int(bank_df["multifield"].sum())} cells with '
          f'two or more lobes')
    print(f'\n  ensemble -> {out_path}')
    print(f'  bank     -> {bank_path}')
    print(f'  total: {time.time()-t0:.1f}s')

    # Round-trip before claiming success: reload from disk and check the
    # cells still peak at their own centres. A save that silently wrote the
    # wrong shape would otherwise only surface during training.
    print()
    return inspect(out_path)


if __name__ == '__main__':
    sys.exit(main())
