# VPCE paper job scripts

Eleven GAIVI job scripts from the REALM-VPCE place-field paper.

**None of these runs as-is.** Their Python entry points lived in
`analysis/experiment_channel_isolation/`, which left with the paper analysis
when this repo was forked. They are kept because the `#SBATCH` headers and the
job bodies encode things that took real debugging to find, and rewriting them
from a blank file would mean learning the same lessons twice:

- which GPU nodes can render Webots at all (compute-only cards have no display
  engine, so GLX falls back to llvmpipe — `collect_data.sh` carries the full
  finding)
- how to make a conda environment visible inside the Singularity image without
  mounting the host's system tree over the container's
- `--submit` fan-out: one job per arena plus a dependent report job that mails
  once they all finish
- chaining stages with `--dependency` so a rerun does not redo what it does
  not have to
- separating "redraw the figures" from "recompute the results", so a label
  change is a 1-hour CPU job rather than a day on a GPU

See the table in `../README.md` for which script demonstrates which pattern.

## Using one

Copy it, point it at a new entry point under `analysis/`, and change
`--mail-user` to your own address — these carry the VPCE paper author's.
Re-check `--time`, `--mem` and `--gres` against what your job actually needs
rather than inheriting the paper's requests; `../README.md` explains why
`--mem=128G` silently excludes GPU6.

The analysis code these called is at `c947bf0` in this repo's history, and
current on the `vpce` remote.
