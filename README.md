# bio_sim

Simulation of a biology-lab scene in **Isaac Sim 5.x / Isaac Lab 2.3** with a
mobile humanoid (**Dexmate Vega-1**) performing pick-and-place on lab consumables
(tubes, racks, plates).

> Status: scaffolding stage — only `README.md` and `PLAN.md` exist. No code yet.
> See [`PLAN.md`](./PLAN.md) for the build plan, milestones, and open questions.

## Goal

A reproducible scene + scripted pick-and-place demo that we can later extend into:

1. **Demo / visualization** (MVP — current target)
2. **Imitation-learning data collection** (teleop → trajectories → policy training)

## Stack

| Layer | Choice | Notes |
| --- | --- | --- |
| Physics / renderer | Isaac Sim 5.1 (pip) | Installed via `uv pip`, not workstation |
| Framework | Isaac Lab 2.3 | Manager-based env workflow |
| Python env | **`uv venv` + `uv pip`** | Isaac Lab has native `uv` support (PR #3172) |
| Python | 3.11 | Required by Isaac Sim 5.1 pip wheels |
| Robot | Dexmate Vega-1 (`dexmate-urdf` v0.8.3+) | Apache-2.0, URDF in repo + USD in releases |
| Lab assets | AutoBio meshes (converted) + NVIDIA Sim-Ready + primitives | Hybrid approach |
| Task style (MVP) | Scripted motion planning | Not RL; demo first |

## Hardware

- **Local dev machine** — RTX 3050 Laptop, 4 GB VRAM. **Not enough to run Sim 5.1 viewport smoothly.** Used for code authoring and headless smoke tests only.
- **Target run machine** — TBD. To be specified before Milestone M1.

## Reference repos

We do **not** start from scratch. The plan adapts patterns from:

| Repo | Used for | License |
| --- | --- | --- |
| [`isaac-sim/IsaacLab`](https://github.com/isaac-sim/IsaacLab) | Project template generator (`./isaaclab.sh -n`) | BSD-3 |
| [`userguide-galaxea/Galaxea_Lab`](https://github.com/userguide-galaxea/Galaxea_Lab) | `pick_fruit_env.py` — wheeled humanoid pick-place env structure | check repo |
| [`dexmate-ai/dexmate-urdf`](https://github.com/dexmate-ai/dexmate-urdf) | Dexmate Vega-1 URDF + USD release artifacts | Apache-2.0 |
| [`unitreerobotics/unitree_sim_isaaclab`](https://github.com/unitreerobotics/unitree_sim_isaaclab) | Secondary reference for humanoid Isaac Lab integration | check repo |
| [`autobio-bench/AutoBio`](https://github.com/autobio-bench/AutoBio) | Bio-lab asset meshes (centrifuge / PCR / pipette / tube / rack) | check repo |

See [`PLAN.md`](./PLAN.md) §"Reference usage" for **what we take from each** and what we rebuild.

## Quick start

Not implemented yet. Will be filled in by milestone **M0** (see `PLAN.md`). Sketch:

```bash
# (M0) — placeholder, do NOT run yet
uv venv --python 3.11 --seed .venv
source .venv/bin/activate
uv pip install --upgrade pip
# Isaac Sim 5.1 + Isaac Lab 2.3 install — exact commands tracked in PLAN.md M0
```

## Layout (planned)

```
bio_sim/
├── pyproject.toml
├── source/bio_sim/           # IsaacLab external project (from template)
│   └── bio_sim/
│       ├── tasks/pick_place/
│       ├── assets/{robots,lab}/
│       └── scene/
├── scripts/{play.py, teleop_record.py}
└── third_party/autobio_assets/   # submodule, meshes only
```
