# Analysis

One directory per experiment, each holding its own entry points. Figures and
cached outputs land in `analysis/<experiment>/figures/` and are git-ignored —
every figure in this repo is reproducible by rerunning its script.

This directory is empty on purpose. The RL experiments have not been written
yet; see "Where the RL work starts" in the root `README.md`.

## Convention

```
analysis/experiment_<name>/
├── README.md          # the question, and what each entry point answers
├── run_<thing>.py     # entry points, invoked by a slurm/ job script
├── plots.py           # figure drawing, importable so replot needs no rerun
└── figures/           # git-ignored output
```

Two conventions from the VPCE paper that were worth having:

- **Separate computing from drawing.** Entry points write their results to
  `data_cache/`, and plotting reads from there. A change of label or layout
  then costs a minute instead of a 24-hour rerun.
- **Keep a `RETIRED.md`.** When an experiment stops making sense, write down
  why before deleting it. The VPCE paper's is worth reading as an example —
  it recorded a sweep that could not produce a result because the parameter
  cancelled out of the equation, which is the kind of thing that gets
  rediscovered expensively.

## The VPCE paper's analysis

Removed from this fork. `analysis/experiment_channel_isolation/` held the
rule-governed place-field pipeline — 22 entry points covering field geometry,
scale distribution, wall proximity, pruning audits and the paper's figures and
numbers.

It is not gone, it is just not here:

```bash
git show c947bf0:analysis/experiment_channel_isolation/rules.py
```

```bash
git checkout c947bf0 -- analysis/experiment_channel_isolation
```

The same code is on the `vpce` remote, where it is still the active line of
work. Prefer that remote over this history if you need the current version —
this fork's copy stopped at `c947bf0`.
