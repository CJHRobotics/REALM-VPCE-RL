# REALM-VPCE-RL

**REALM** (Robotic Environment for Autonomous Learning and Mapping) with the
**Visual Place Cell Encoding (VPCE)** model, used here as a *state
representation* for reinforcement learning agents.

This repository is the working codebase for the RL paper. It is forked from
[REALM-VPCE](https://github.com/CJHRobotics/REALM-VPCE), which is itself a
fork of [REALM](https://github.com/CJHRobotics/REALM), both developed at the
USF BioRobotics Lab.

---

## The question this repo is for

VPCE turns a robot's point-of-view images into a population of place-cell-like
activations, with no odometry, path integration, ground-truth coordinates or
task feedback. The REALM-VPCE paper established that those activations behave
like place fields: spatially compact, boundary-respecting, remapping under
structural change.

This repo asks what they are *good for*. A VPCE population is a candidate
observation space for an RL agent — low-dimensional, learned without
supervision, and grounded in what the robot can actually see. The work here
trains agents on that representation and compares it against the obvious
alternatives (raw pixels, CNN embeddings, ground-truth pose).

---

## What carries over from the VPCE paper

The model and the simulation infrastructure, not the analysis.

| | where | what it does |
|---|---|---|
| **Feature extraction** | `realm_tools/image_lib/feature_extractor.py` | ResNet50 plus HOG, colour and spatial histograms, per POV image |
| **Ensemble formation** | `realm_tools/place_cell_lib/clustering.py` | k-means (`fit_kmeans`) or GMM (`fit_gmm`) over feature vectors; each cluster becomes a place cell |
| **Activation encoding** | `realm_tools/place_cell_lib/place_cells.py` | `rbf_activations`, `VisualPlaceCellEnsemble`, `SpatialPlaceCellEnsemble` |
| **Robot and arenas** | `realm_tools/robot_lib/`, `simulation/` | HamBot, the Webots protos, the VPCE arena set and its position grids |
| **Data collection** | `simulation/controllers/collect_data/` | renders POV images across an arena's position grid and writes the feature dataset |
| **GAIVI jobs** | `slurm/` | headless Webots in Singularity, the GPU node findings, emailed reports |

The place-field analysis that produced the VPCE paper's figures and numbers is
**not** here. It is on the `vpce` remote and in this repo's history at
`c947bf0`. See `analysis/README.md`.

---

## Where the RL work starts

The pipeline runs end to end. Four stages, each a SLURM job:

| stage | job | produces |
|---|---|---|
| collect | `slurm/collect_data.sh <arena>` | `data/vpce/collect_data/<arena>.h5` |
| fit fields | `slurm/fit_place_fields.sh <arena>` | an ensemble + the field maps |
| train | `slurm/train_rl.sh <arena>` | a policy + the learning curve |
| compare | *still to write* | learning time against representation |

The ensemble is **fitted once and frozen** as an encoder. Refitting it online
as the agent explores is the more interesting claim and remains open, but the
frozen path is what every current measurement assumes.

The agent observes the simulator and nothing else: every step captures eight
camera frames at the robot's position, builds the same descriptors the
dataset was built from, and encodes them through the ensemble. No pose, no
goal bearing, no lookup from the collected grid.

Still to write: the sweep that varies scale, multiscale and feature space
across seeds, and the analysis that turns the Monitor CSVs into
learning-time-against-representation.

---

## Requirements

### 1. Python 3.11

**macOS:**
```bash
brew install python@3.11
```

**Linux:**
```bash
sudo apt-get install python3.11
```

**Windows:** Install Python 3.11 from the 
[Microsoft Store](https://apps.microsoft.com/store/detail/python-311/9NRWMJP3717K) 
to avoid PATH issues.

### 2. Webots R2025a

Download and install from the [Cyberbotics website](https://cyberbotics.com/#download).

> **Linux users:** Do not install Webots via Snap. Use the `.deb` package or 
> tarball instead.

### 3. Git

**Windows:** [git-scm.com](https://git-scm.com/download/win)  
**Linux:** `sudo apt-get install git`  
**macOS:** `brew install git`

---

## Setup

### 1. Clone the repository

```bash
git clone git@github.com:CJHRobotics/REALM-VPCE-RL.git
cd REALM-VPCE-RL
```

### 2. Run the install script

```shell
python3.11 setup/realm_install.py
```

This will:
- Find Python 3.11 on your system
- Create `venv-vpce-rl` with all dependencies
- Add the project root to the venv's Python path
- Generate `runtime.ini` files in all Webots controller directories

To remove the environment:
```shell
python3.11 setup/realm_install.py --uninstall
```

### 3. Activate the environment

**macOS/Linux:**
```shell
source venv-vpce-rl/bin/activate
```

**Windows:**
```shell
venv-vpce-rl\Scripts\activate
```

> If you add a new controller under `simulation/controllers/`, re-run 
> `python3.11 setup/add_runtime_ini.py` to generate its `runtime.ini`.

---

## Project Structure

```
REALM-VPCE-RL/
├── setup/
│   ├── realm_install.py          # Install / uninstall script
│   ├── add_runtime_ini.py        # Generates Webots runtime.ini files
│   └── requirements.txt          # Venv dependencies
│
├── realm_tools/
│   ├── robot_lib/
│   │   ├── hambot.py             # Base robot class (sensors, motors, supervisor)
│   │   ├── my_robot.py           # Experiment extension point (inherits HamBot)
│   │   └── navigation_tools.py   # Shared navigation utilities
│   ├── simulation_lib/
│   │   ├── environment.py        # Environment class and environment objects
│   │   ├── environment_parser.py # XML arena file parser
│   │   ├── start_position_generator.py  # Position grids for collection
│   │   └── (robot and arena plumbing)
│   ├── image_lib/
│   │   ├── feature_extractor.py  # ResNet50 + HOG feature extraction
│   │   ├── image_feature_lib.py  # Image processing utilities
│   │   └── analysis_plots.py     # Shared plotting helpers
│   ├── place_cell_lib/
│   │   ├── clustering.py         # k-means / GMM ensemble formation
│   │   └── place_cells.py        # RBF activation, place-cell ensembles
│   └── experiment_lib/
│       ├── loggers/              # Experiment logger, dataset and model I/O
│       ├── place_cell_analysis.py  # Place-field scoring
│       └── reporting.py          # ExperimentReport — emailed job reports
│
├── simulation/
│   ├── controllers/
│   │   ├── collect_data/         # Renders an arena's POV dataset (GAIVI job)
│   │   ├── camera_capture/       # Grabs sample POV images for inspection
│   │   ├── render_check/         # Confirms the headless renderer works
│   │   ├── remote_control/       # Keyboard driving, for looking around
│   │   ├── perf_probe/           # Per-step timing
│   │   └── example/              # Minimal controller template
│   ├── protos/                   # HamBot and world object Webots protos
│   └── worlds/                   # Webots worlds, arena XMLs, position grids
│
├── analysis/                     # Experiment entry points (see its README)
├── slurm/                        # GAIVI job scripts + vpce_paper_jobs/ templates
├── container/                    # Dockerfile for the Webots Singularity image
├── data/                         # Datasets and caches (git-ignored)
└── docs/                         # Figures and documentation assets
```

---

## Extending for Your Project

1. **Robot logic** — subclass `HamBot` in `my_robot.py` and add
   experiment-specific methods (action sets, observation processing, etc.)
2. **Environment** — `rl_lib/webots_env.py` holds `WebotsEnv`; reward lives
   in `RewardConfig` and the observation in `rl_lib/vpce_observer.py`
3. **Controllers** — add new Webots controllers under
   `simulation/controllers/` then re-run `add_runtime_ini.py`
4. **Personal files** — your personal robot subclass (e.g.
   `yourname_robot.py`) can be gitignored so the template stays clean
   for others

---

## Contributing back to REALM

Three remotes are configured:

```bash
git remote -v
```

| remote | repo | use |
|---|---|---|
| `origin` | `CJHRobotics/REALM-VPCE-RL` | this work |
| `vpce` | `CJHRobotics/REALM-VPCE` | the place-field paper |
| `upstream` | `CJHRobotics/REALM` | the base simulation framework |

Infrastructure fixes that are not specific to this paper — robot classes,
arena parsing, the Webots container, the GAIVI setup — belong upstream. Put
them on their own branch off `upstream/main` so the PR carries only that
change:

```bash
git fetch upstream && git checkout -b fix/<name> upstream/main
```

---

## Calibration Controller

A keyboard-driven controller is provided for manually testing robot 
behaviour in Webots:

| Key           | Action      |
|---------------|-------------|
| Arrow Up      | Forward     |
| Arrow Down    | Backward    |
| Arrow Left    | Turn left   |
| Arrow Right   | Turn right  |
| Any other key | Stop        |

Open `simulation/worlds/calibration.wbt` in Webots to use it.

---

## Additional Documentation

- [HamBot Reference](realm_tools/README.md)
- [GAIVI Job Scripts](slurm/README.md)
- [Simulation & Webots Guide](simulation/README.md)
- [Controller Setup Guide](simulation/controllers/README.md)
- [Webots Controller Guide](https://cyberbotics.com/doc/guide/controller-programming)
- [Gymnasium API Reference](https://gymnasium.farama.org/api/env/)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)

---

## Funding

This work was supported in part by NSF IIS Robust Intelligence grant 
#1703225 — *Experimental and Robotics Investigations of Multiscale Spatial 
Memory Consolidation in Complex Environments*, University of South Florida.