"""Place fields drawn projected into the arena they were measured in.

The arena outline, the landmark panels and the field ellipses are the paper's
F1 field map (analysis/experiment_channel_isolation/plots.py at c947bf0),
kept deliberately recognisable so a figure from this repo can be put beside
one from the place-field paper and compared.

What is new is the panel breakdown. The paper's grid was channel x lambda,
because those were its variables; here the variable is scale, so the panels
are scales -- the whole library first, then one panel per scale. A
single-scale ensemble (`--scale 3`) and a multiscale one (`--scale 2-5`) are
then the same figure read at different widths, which is the comparison the
learning experiment rests on.

Fields are ellipses rather than discs because a field has two axes that can
differ severalfold, and they are coloured by the floor area they cover: area
is what the field occupies, and it is the unit the recording literature
reports (Harland et al. 2021).

Note that the ellipses are a PROJECTION. Under the size-only rules a field is
admitted on how many locations its node holds, with no reference to where
those locations are; the footprint drawn here is measured afterwards, for
looking at. A field that appears as a tidy ellipse and one that is scattered
across the arena are both legitimate cells, and the fragmented ones are worth
spotting -- a cell that responds in several distant places aliases those
places in the agent's state, which is a reason a run might learn slowly. The
`cc_frac` column carries that per field, and the panel titles report it.
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')          # set before pyplot: these run headless on GAIVI
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse


def draw_env(ax, env, xml_root):
    """Arena outline, walls and landmark panels, in their real colours."""
    if env['is_circular']:
        th = np.linspace(0, 2 * np.pi, 240)
        ax.plot(env['env_cx'] + env['env_R'] * np.cos(th),
                env['env_cy'] + env['env_R'] * np.sin(th),
                color='black', lw=1.6, zorder=6)
    if xml_root is not None:
        for w in xml_root.findall('wall'):
            ax.plot([float(w.get('x1')), float(w.get('x2'))],
                    [float(w.get('y1')), float(w.get('y2'))],
                    color='black', lw=1.6, zorder=6)
        for lm in xml_root.findall('landmark'):
            x, y = float(lm.get('x')), float(lm.get('y'))
            t = float(lm.get('theta', 0))
            hw = float(lm.get('width', 0.75)) / 2
            tx, ty = -np.sin(t), np.cos(t)
            c = (float(lm.get('red', 0)), float(lm.get('green', 0)),
                 float(lm.get('blue', 0)))
            ax.plot([x - hw * tx, x + hw * tx], [y - hw * ty, y + hw * ty],
                    color=c, lw=4, solid_capstyle='butt', zorder=7)
    ax.set_xlim(env['x_min'] - 0.5, env['x_max'] + 0.5)
    ax.set_ylim(env['y_min'] - 0.5, env['y_max'] + 0.5)
    ax.set_aspect('equal')


def draw_fields(ax, bank, cmap, norm):
    """Fields as ellipses: a filled pass for mass, an outline for edges."""
    for _, r in bank.iterrows():
        c = cmap(norm(r['area_env_m2']))
        for face, alpha, z in ((c, 0.18, 3), ('none', 0.8, 4)):
            ax.add_patch(Ellipse(
                (r['centroid_x'], r['centroid_y']),
                width=2 * r['semi_major_m'], height=2 * r['semi_minor_m'],
                angle=np.degrees(r['orientation_rad']),
                facecolor=face, edgecolor=c, alpha=alpha, lw=0.9, zorder=z))


def field_map(bank, env, xml_root, out_path, title='', max_panels=6):
    """One field map for the library, plus one per scale.

    Returns the path written, or None when the bank is empty.
    """
    if not len(bank):
        return None

    scales = sorted(bank['scale'].unique())
    # With a wide --scale-m the number of scales can run past what fits on a
    # row; the per-scale panels are the detail view, so drop the sparsest
    # than shrink every panel to illegibility. The combined panel always
    # shows the whole library regardless.
    if len(scales) > max_panels - 1:
        by_size = bank['scale'].value_counts()
        scales = sorted(by_size.index[:max_panels - 1])

    # One colour scale across every panel, so a colour means the same area in
    # all of them. Log, because field area spans orders of magnitude.
    areas = bank['area_env_m2'].to_numpy()
    lo = max(float(areas[areas > 0].min()) if (areas > 0).any() else 1e-3, 1e-3)
    cmap = plt.get_cmap('plasma')
    norm = mcolors.LogNorm(lo, max(float(areas.max()), lo * 1.01))

    n = 1 + len(scales)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.0), squeeze=False)
    axes = axes[0]

    draw_env(axes[0], env, xml_root)
    draw_fields(axes[0], bank, cmap, norm)
    frag = float((bank['cc_frac'] < 0.8).mean()) if 'cc_frac' in bank else 0.0
    axes[0].set_title(f'all scales\n{len(bank)} fields | '
                      f'{100*frag:.0f}% fragmented', fontsize=9)

    for ax, b in zip(axes[1:], scales):
        sub = bank[bank['scale'] == b]
        draw_env(ax, env, xml_root)
        draw_fields(ax, sub, cmap, norm)
        r = sub['radius_members_m'] if 'radius_members_m' in sub else sub['radius_env_m']
        ax.set_title(f'scale {b}\n{len(sub)} fields | '
                     f'r ~ {r.median():.2f} m', fontsize=9)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.02).set_label('field area (m$^2$)')
    if title:
        fig.suptitle(title, y=1.02, fontsize=12)

    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


def scale_summary(bank, out_path, title=''):
    """Scale distribution, and projected area against the position-free scale.

    The right panel is the one to look at before trusting a run: admission
    used `radius_members_m`, everything spatial is a later measurement, and
    the two agreeing is what says the member-count substitution is sound for
    this dataset. A cloud with no trend there means the member count is not
    tracking anything spatial, and the scale axis of the experiment would be
    measuring something other than scale.
    """
    if not len(bank):
        return None

    has_mem = 'radius_members_m' in bank
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    axes[0].hist(bank['area_env_m2'], bins=30, color='#ff7f0e',
                 edgecolor='white')
    axes[0].set_xlabel('field area (m$^2$)')
    axes[0].set_ylabel('fields')
    axes[0].set_title('projected area', fontsize=10)

    counts = bank['scale'].value_counts().sort_index()
    axes[1].bar([f'S{b}' for b in counts.index], counts.to_numpy(),
                color='#1f77b4', edgecolor='white')
    axes[1].set_ylabel('fields')
    axes[1].set_title('cells per scale', fontsize=10)

    if has_mem:
        axes[2].scatter(bank['radius_members_m'], bank['radius_env_m'],
                        s=14, alpha=0.6, color='#2ca02c')
        lim = [0, max(bank['radius_members_m'].max(),
                      bank['radius_env_m'].max()) * 1.05]
        axes[2].plot(lim, lim, ls='--', lw=1, color='grey')
        axes[2].set_xlim(lim)
        axes[2].set_ylim(lim)
        axes[2].set_xlabel('radius from member count (m)')
        axes[2].set_ylabel('radius from projection (m)')
        r = float(np.corrcoef(bank['radius_members_m'],
                              bank['radius_env_m'])[0, 1]) if len(bank) > 2 else float('nan')
        axes[2].set_title(f'the two scale measures  (r = {r:.2f})', fontsize=10)
    else:
        axes[2].axis('off')

    if title:
        fig.suptitle(title, y=1.04, fontsize=12)
    fig.tight_layout()
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path
