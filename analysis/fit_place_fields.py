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
distance matrix, then a single admission rule -- size. A node becomes a cell
if it holds neither too few nor too many of the sampled locations (Rules 8
and 9, measured on member count). The surviving nodes' feature-space means
become the ensemble's centers and their feature sigmas become its radii.

Nothing else filters. Contiguity (Rule 1), competition (Rule 11), tiling
(Rule 12) and reliability (Rule 2) are all still COMPUTED, and every one of
them is written to the bank, but none of them rejects anything. That is the
`size-only` preset and it is the default; `--rules paper` restores the
paper's funnel for a comparison run.

Why this matters more than it looks. Measuring size on member count instead
of on the field's footprint is not a change of precision -- members are
roughly density times area and the two agree closely in practice -- it is a
change in what the admission decision is allowed to know. The footprint is
measured by projecting a node's members onto the floor, so selecting on it
means the library was chosen using ground-truth position; the encoder would
run on images while the choice of which cells exist did not. Member count
uses only the global sampling density, so admission is position-free, and
with the spatial rules off no admission decision anywhere consults a
position. LAMBDA stays at 0 for the same reason: Rule 4 would put position
into the merge cost itself.

Scale survives all of this, since later analysis asks how scale affects
learning. The bank carries `n_members`, `radius_members_m` (the position-free
scale, and the one admission used), `radius_env_m` and `area_env_m2` (the
projected measures), plus `scale`. `--scale` selects scales, which is the
single-scale / multiscale knob.

Why not k-means. `clustering.fit_kmeans` is still there and is a reasonable
baseline, but it needs K chosen in advance and every centroid it produces is
a cell whether or not anything justifies it. The agglomeration admits cells
on evidence, so K is an outcome.

Usage
-----
    python3 analysis/fit_place_fields.py circ_lm8_r10
    python3 analysis/fit_place_fields.py circ_lm8_r10 --scale 3        # one scale
    python3 analysis/fit_place_fields.py circ_lm8_r10 --scale 2-5      # multiscale
    python3 analysis/fit_place_fields.py circ_lm8_r10 --scale-m 0.4 1.5
    python3 analysis/fit_place_fields.py circ_lm8_r10 --channels hog   # one channel
    python3 analysis/fit_place_fields.py circ_lm8_r10 --subsample 8000 # quick
    python3 analysis/fit_place_fields.py --inspect data/vpce/place_fields/...h5

The three knobs the learning experiment varies -- scale (`--scale`,
`--scale-m`), multiscale (one scale against several) and feature space
(`--channels`) -- each produce a differently sized observation space from the
same dataset, which is the point.

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

# Size is the only rule that rejects anything, and it is measured on how many
# sampled locations a node holds rather than on the field's footprint, so no
# admission decision consults a position. Everything else the paper's funnel
# did is turned off here -- not deleted, so a paper-comparable library is one
# flag away, and every stage still reports what it WOULD have rejected.
RULE_PRESETS = {
    'size-only': dict(
        SIZE_MEASURE='members',      # Rules 8, 9 on member count
        FIELD_UNIT='cluster',        # one cell per node, no subfield split
        CC_FRAC_MIN=0.0,             # Rule 1  off (measured, reported)
        SAME_SCALE_COMPETITION=False,  # Rule 11 off -- see the note below
        TILING_FRAC_MIN=0.0,         # Rule 12 off (already the default)
        SPLIT_HALF_IOU_MIN=None,     # Rule 2  off (already the default)
    ),
    # The paper's configuration, for a comparison run.
    'paper': dict(SIZE_MEASURE='area', FIELD_UNIT='subfield'),
}

# Dropping Rule 11 has one consequence worth stating. Competition was what
# stopped a node and its own ancestors from both being admitted; without it,
# every node in the member-count window is a cell, nested ones included, and
# parent's members are a superset of its child's. Those cells are correlated
# by construction. For a multiscale representation that is the point -- it is
# how the same place gets described at several scales at once. For a
# single-scale run it is not, and the lever is how many scales are kept:
# --scale selects geometric scales, and a narrow selection excludes a node's
# direct ancestors because a parent sits at least BAND_RATIO times its
# child's radius. The fit reports the redundancy either way.


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
    ap.add_argument('--rules', default='size-only', choices=sorted(RULE_PRESETS),
                    help='admission rules (default: size-only -- size is the '
                         'only rule that rejects, measured on member count)')
    ap.add_argument('--scale-m', nargs=2, type=float, metavar=('LO', 'HI'),
                    default=None,
                    help='admit only fields whose implied radius is in this '
                         'range, in metres; converted to a member count via '
                         'the arena sampling density')
    ap.add_argument('--scale', default=None, metavar='SPEC',
                    help='keep only these geometric scales, e.g. "3" for a '
                         'single-scale ensemble or "1-4" for multiscale '
                         '(default: every admitted scale)')
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
    ap.add_argument('--no-figure', action='store_true',
                    help='skip the field-map and scale figures')
    ap.add_argument('--fig-dir', default=None,
                    help='where figures go (default: analysis/figures/'
                         'place_fields/)')
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
    # REALM_ARENA_DIR picks the environments directory, so an obstacle
    # arena under vpce-rl fits with the same command as a vpce one.
    arena_dir = os.environ.get('REALM_ARENA_DIR', 'vpce').strip('/')
    xml_path = (f'{REPO}/simulation/worlds/environments/'
                f'{arena_dir}/{args.arena}.xml')
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

    cfg = dict(RULE_PRESETS[args.rules])
    cfg.update(LAMBDA=args.lam, BIN_M=args.bin_m, RANDOM_SEED=args.seed,
               USE_GPU=not args.no_gpu)
    if args.scale_m:
        # Stated in metres because that is the unit the scale question is
        # asked in, converted to the area fractions the rules hold, and from
        # there to a member count by the arena's mean density. The decision
        # still only ever compares counts.
        lo, hi = sorted(args.scale_m)
        cfg['RULE8_AREA_FRAC'] = float(np.pi * lo ** 2 / env['env_area'])
        cfg['RULE9_AREA_FRAC'] = float(np.pi * hi ** 2 / env['env_area'])
        print(f'  size window: r in [{lo:g}, {hi:g}] m')
    print(f'  rules: {args.rules}  (size measured on '
          f'{cfg.get("SIZE_MEASURE", "area")})')

    bank_df, kept_mu, report = R.build_bank(
        X, xy, env, D2_feat, feat_med, xy_med, cfg=cfg, device=device,
        tag=f'{args.arena}/{args.channels}')

    if len(bank_df) == 0:
        print('\nERROR: the rules admitted no fields, so there is no ensemble '
              'to save.\n  The funnel above says which rule rejected what. A '
              'channel with no\n  spatial locality (check median_sigma_ratio '
              'near 1) is the usual cause.', file=sys.stderr)
        return 1

    # Scale selection: the single-scale / multiscale knob. Applied after
    # admission rather than inside it, so the funnel still reports what the
    # rules did and only the ensemble is narrowed.
    if args.scale:
        want = set()
        for part in args.scale.split(','):
            if '-' in part.strip():
                a, b = part.split('-')
                want.update(range(int(a), int(b) + 1))
            else:
                want.add(int(part))
        mask = bank_df['scale'].isin(want).to_numpy()
        if not mask.any():
            present = sorted(bank_df['scale'].unique())
            print(f'ERROR: no admitted field is at scale(s) {sorted(want)}; '
                  f'this fit produced scales {present}', file=sys.stderr)
            return 1
        print(f'  scales {sorted(want)}: keeping {int(mask.sum())} of '
              f'{len(bank_df)} fields')
        bank_df = bank_df[mask].reset_index(drop=True)
        kept_mu = kept_mu[mask]

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
    # coverage is per scale, not one number: a scale covering little of the
    # floor is informative on its own (Rule 12 used to delete such scales and
    # no longer does), so report them rather than collapsing them.
    cov = report['coverage']
    if isinstance(cov, dict):
        print('  coverage    : ' + '  '.join(
            f'B{b}={v:.0%}' for b, v in sorted(cov.items())))
    else:
        print(f'  coverage    : {cov:.1%} of the floor')
    print(f'  multifield  : {int(bank_df["multifield"].sum())} cells with '
          f'two or more lobes')

    # Scale, reported on both measures, because later analysis asks how scale
    # affects learning and the two are not interchangeable: radius_members_m
    # is what admission used and is position-free, radius_env_m is measured
    # through the floor projection. Both are in the bank.
    if 'radius_members_m' in bank_df:
        rm = bank_df['radius_members_m']
        print(f'  scale (members) : r = {rm.min():.3g} .. {rm.max():.3g} m  '
              f'(median {rm.median():.3g})')
    print(f'  scale (projected): r = {bank_df["radius_env_m"].min():.3g} .. '
          f'{bank_df["radius_env_m"].max():.3g} m')
    counts = bank_df['scale'].value_counts().sort_index()
    print('  cells per scale : ' + '  '.join(
        f'S{b}={n}' for b, n in counts.items()))

    # How much of the observation space is redundant. With Rule 11 off a node
    # and its ancestors can both be cells, and a parent's members include its
    # child's, so some columns move together by construction. This is the
    # number that matters for an RL state space -- correlated inputs are
    # survivable where exact duplicates are not -- so report it rather than
    # filtering on it.
    act_fit = ensemble.activate(X, normalize=None)
    keepcol = act_fit.std(axis=0) > 0
    if keepcol.sum() > 1:
        C_abs = np.abs(np.corrcoef(act_fit[:, keepcol], rowvar=False))
        np.fill_diagonal(C_abs, 0.0)
        print(f'  cell correlation: median {np.median(C_abs):.3f}  '
              f'max {C_abs.max():.3f}  '
              f'pairs > 0.99: {int((C_abs > 0.99).sum() // 2)}')
    if int((~keepcol).sum()):
        print(f'  WARNING: {int((~keepcol).sum())} cells never vary across '
              f'the fitted data -- dead inputs for an agent')

    print(f'\n  ensemble -> {out_path}')
    print(f'  bank     -> {bank_path}')

    # --------------------------------------------------------------- figures
    figures = []
    if not args.no_figure:
        # Imported here rather than at module scope so that a fit on a host
        # with no working matplotlib still produces the ensemble, which is
        # the part nothing else can regenerate.
        from realm_tools.place_cell_lib import field_plots
        fig_dir = args.fig_dir or f'{REPO}/analysis/figures/place_fields'
        stem = os.path.splitext(os.path.basename(out_path))[0]
        label = (f'{args.arena} | {args.channels} | {args.rules}'
                 f'{"" if not args.scale else " | scale " + args.scale}')
        try:
            f1 = field_plots.field_map(
                bank_df, env, xml_root, f'{fig_dir}/{stem}_fields.png',
                title=f'{label}\n{n_cells} place fields projected into the arena')
            f2 = field_plots.scale_summary(
                bank_df, f'{fig_dir}/{stem}_scale.png', title=label)
            figures = [f for f in (f1, f2) if f]
            for f in figures:
                print(f'  figure   -> {f}')
        except Exception as exc:                            # noqa: BLE001
            # A figure failure must not lose the fit: the ensemble and bank
            # are already on disk, and the paper's first GAIVI run died in
            # matplotlib after the expensive part had succeeded.
            print(f'  WARNING: figures failed ({exc.__class__.__name__}: {exc})')

    print(f'  total: {time.time()-t0:.1f}s')

    # Round-trip before claiming success: reload from disk and check the
    # cells still peak at their own centres. A save that silently wrote the
    # wrong shape would otherwise only surface during training.
    print()
    return inspect(out_path)


if __name__ == '__main__':
    sys.exit(main())
