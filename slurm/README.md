# SLURM job scripts for GAIVI

Job scripts here submit this repo's work to GAIVI's scheduler. Each script is
a thin wrapper: resolve the interpreter, `cd` into the repo, and run one entry
point. The data-collection and render-check jobs additionally run Webots
headless inside the Singularity image, which `_webots_env.sh` sets up.

The RL training and evaluation jobs are not written yet. Write them against
the patterns in `vpce_paper_jobs/` — see **Job templates** below — rather than
from a blank file: the `#SBATCH` headers there encode node choices that took
real debugging to find.

## Quick reference

Every script carries a complete `#SBATCH` header, so `sbatch slurm/<name>.sh`
is all that is needed. Arguments are forwarded to the Python entry point.

```bash
sbatch --exclusive --export=ALL,STRICT_GPU=1,REALM_FORCE=1 slurm/collect_data.sh circ_lm8_r3
```

`collect_data.sh --submit` is the one exception, run with `bash`: it submits
one collection job per arena with the flags above (`--exclusive`,
`STRICT_GPU=1`, `REALM_FORCE=1`), defaulting to the newest arenas in its
`SUBMIT_MAZES` list.

```bash
bash slurm/collect_data.sh --submit
```

| script | job name | time | mem | GPU | what it runs |
|---|---|---|---|---|---|
| `collect_data.sh` | `collect` | 24 h | 32 G | `gpu:1` | renders one arena headless in Webots and writes its feature dataset, via `simulation/controllers/collect_data`. Needs the Singularity image; `_webots_env.sh` sets it up |
| `render_check.sh` | `render-chk` | 1 h | 16 G | `gpu:1` | is the headless renderer actually producing images? Run this before a long collection |

`check_dataset.py` and `report_collection.py` are helpers called by
`collect_data.sh`, not job scripts.

## Job templates (`vpce_paper_jobs/`)

Eleven job scripts from the VPCE place-field paper are kept there. **None of
them runs as-is** — their Python entry points lived in
`analysis/experiment_channel_isolation/`, which left with the paper analysis
when this repo was forked. They are here as worked examples:

| script | the pattern worth copying |
|---|---|
| `scale_distribution.sh` | a long analysis job that mails an `ExperimentReport` with its own CSVs and figures attached |
| `extent_validation.sh` | `--submit` fan-out: one job per arena plus a dependent report job that mails once they all finish |
| `wall_proximity.sh` | reusing another job's cached libraries instead of recomputing them |
| `prune_audit.sh` | a diagnostic job that rebuilds what no cache holds |
| `multifield_examples.sh` | a short single-arena figure job (2 h, 64 G) |
| `field_geometry.sh`, `kappa_rerun.sh`, `paper_rerun.sh`, `paper_finish.sh`, `paper_figures.sh` | multi-stage reruns that chain stages with `--dependency` |
| `replot.sh` | redrawing figures from the data cache with no GPU and no dataset load |

To restore one as a working job, point it at a new entry point under
`analysis/`. The full VPCE analysis code is on the `vpce` remote and in this
repo's history at `c947bf0`.

All run on the **`general`** partition with 16 CPUs. Logs land in
`slurm/logs/<job-name>-<jobid>.out` and `.err`, git-ignored.

## GAIVI partitions

Snapshot from `sinfo`; re-run it rather than trusting this table for
availability. `general` is the default and the only one every job script
targets.

| partition | time limit | nodes |
|---|---|---|
| `general`* | 7 days | GPU6, GPU42, GPU43, GPU44 |
| `nopreempt` | 7 days | GPU2, GPU3, GPU4 |
| `Quick` | 1 day | GPU1, GPU45, GPU46, GPU47 |
| `Extended` | unlimited | GPU8, GPU9, GPU22 |
| `Contributors` | unlimited | GPU1–2, GPU42–55 |
| `YES` | unlimited | GPU56 |

```bash
sinfo -p general -N -o "%N %G %t"
```

## GPU devices on `general`

Four nodes, three card types. GRES strings are **case-sensitive** — `A40`, not
`a40`; a wrong string does not error, it just never matches.

| node | GRES string | cards | VRAM per card | node RAM | cores | TF32 |
|---|---|---|---|---|---|---|
| GPU6 | `gpu:1080Ti:4` | 4 × 1080 Ti | 11 GB | 128 GB | 32 | no |
| GPU42 | `gpu:TitanRTX:4` | 4 × Titan RTX | 24 GB | 192 GB | 64 | no |
| GPU43 | `gpu:A40:4` | 4 × A40 | 48 GB | 512 GB | 128 | **yes** |
| GPU44 | `gpu:A40:4` | 4 × A40 | 48 GB | 512 GB | 64 | **yes** |

### Why the scripts ask for `gpu:1` rather than a named type

Pinning the GRES to `gpu:A40:1` queues the job behind everything else waiting
on GPU43 and GPU44, which in practice costs far more than the A40 saves.
Asking for any GPU schedules more or less immediately.

Nothing breaks on a smaller card, provided the code asks. The VPCE paper's
widest feature matrices (`visual` 7.2 GB, `all` 7.3 GB) did not fit an 11 GB
1080 Ti alongside their working set; the pipeline tested free VRAM and fell
back to the CPU for those stages, so a small card cost time rather than
correctness. Worth keeping for the RL work: an agent that holds a replay
buffer and a VPCE encoder on the same card has a real memory ceiling, and
`gpu:1` can land it on 11 GB.

Take an A40 for a run that needs the speed, without editing anything:

```bash
sbatch --gres=gpu:A40:1 slurm/<job>.sh
```

### Two things the node table implies

**`--mem=128G` excludes GPU6.** That is exactly GPU6's total RAM, and SLURM
only offers a node slightly less than its physical memory, so the request can
never fit there. "Any GPU" therefore means GPU42/43/44 in practice. The paper
jobs asked for 128 G and did not need it — their largest working set was a
7.3 GB feature matrix plus its 3.6 GB pairwise block, peaking around 12 GB
locally. Ask for what a job actually uses; `--mem=64G` widens where it lands.

**`Quick` holds the A100 nodes and allows 1-day jobs.** If `general` is
congested it is worth trying — but note the A100 caveat in `collect_data.sh`:
compute-only cards have no display engine, so anything that renders through
Webots drops to software GL there. Analysis and training jobs are unaffected.

```bash
sbatch --partition=Quick slurm/<job>.sh
```

## Full node inventory

| node | cores | processor | RAM | cards | GPU RAM |
|---|---|---|---|---|---|
| GPU1 | 96 | AMD EPYC 7352 24-Core | 256 GB | 3 × A100 | 240 GB |
| GPU2 | 96 | AMD EPYC 7352 24-Core | 256 GB | 3 × A100 | 240 GB |
| GPU3 | 32 | Dual Xeon E5-2620 v4 @ 2.10 GHz | 128 GB | 4 × 1080 Ti | 44 GB |
| GPU4 | 32 | Dual Xeon E5-2620 v4 @ 2.10 GHz | 128 GB | 4 × 1080 Ti | 44 GB |
| GPU6 | 32 | Dual Xeon E5-2620 v4 @ 2.10 GHz | 128 GB | 4 × 1080 Ti | 44 GB |
| GPU8 | 32 | Dual Xeon E5-2620 v4 @ 2.10 GHz | 128 GB | 4 × 1080 Ti | 44 GB |
| GPU9 | 32 | Dual Xeon E5-2620 v4 @ 2.10 GHz | 128 GB | 4 × 1080 Ti | 44 GB |
| GPU22 | 40 | Dual Xeon E5-2630 v4 @ 2.20 GHz | 1024 GB | 8 × 1080 Ti | 88 GB |
| GPU42 | 64 | Dual Xeon Gold 6226R @ 2.90 GHz | 192 GB | 4 × Titan RTX | 96 GB |
| GPU43 | 128 | AMD EPYC 7662 64-Core | 512 GB | 4 × A40 | 192 GB |
| GPU44 | 64 | AMD EPYC 7532 32-Core | 512 GB | 4 × A40 | 192 GB |
| GPU45 | 48 | AMD EPYC 7413 24-Core | 256 GB | 2 × A100 | 160 GB |
| GPU46 | 96 | AMD EPYC 7352 24-Core | 2 TB | 3 × A100 | 240 GB |
| GPU47 | 64 | AMD EPYC 7513 32-Core | 512 GB | 2 × A100 | 160 GB |
| GPU48 | 256 | AMD EPYC 9554 64-Core | 768 GB | 6 × H100 | 480 GB |
| GPU49 | 64 | Xeon Silver 4314 @ 2.40 GHz | 128 GB | 2 × A100 + 2 × L40S | 256 GB |
| GPU50 | 64 | AMD EPYC Genoa 9354 @ 3.3 GHz | 384 GB | 1 × RTX A6000 | 48 GB |
| GPU51 | 64 | AMD EPYC Genoa 9374F @ 3.85 GHz | 384 GB | 2 × RTX 6000 Ada | 96 GB |
| GPU52 | 64 | AMD EPYC 7543 32-Core @ 2.8 GHz | 524 GB | 4 × RTX A6000 | 192 GB |
| GPU53 | 192 | AMD EPYC Milan 7643 @ 2.3 GHz | 1024 GB | 8 × L40S | 384 GB |
| GPU54 | 256 | AMD EPYC Genoa 9554 64-Core @ 3.1 GHz | 768 GB | 8 × RTX 6000 Ada | 384 GB |
| GPU55 | 128 | AMD EPYC 9354 32-Core @ 3.25 GHz | 384 GB | 2 × RTX Pro 6000 BW | 192 GB |
| GPU56 | 128 | AMD EPYC 9354 32-Core | 384 GB | 3 × RTX Pro 6000 BW | 288 GB |
| GPU57 | 128 | AMD EPYC 9354 32-Core | 1128 GB | 8 × RTX Pro 6000 BW | 768 GB |

## Useful scheduler commands

```bash
squeue -u "$USER" -o "%.10i %.12j %.8T %.10M %.10l %.20R"
```

```bash
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS,ReqTRES%40,ExitCode
```

```bash
scancel <jobid>
```

## Emailed reports

New experiments build their own report by subclassing `ExperimentReport` in
`realm_tools/experiment_lib/reporting.py` — the body is composed from the
experiment's results rather than scraped from the job log, and attaches its
own CSVs and figures. `collect_data.sh` mails through `report_collection.py`,
which uses the same SMTP transport; `vpce_paper_jobs/scale_distribution.sh` is
the worked example of the `ExperimentReport` path. Every send is a silent
no-op without `EMAIL_TO`.

Set `--mail-user` in a new job's `#SBATCH` header to your own address. The
parked scripts carry the VPCE paper author's.
