# bio_sim

Simulation of a biology-lab scene in **Isaac Sim 5.x / Isaac Lab 2.3** with a
mobile humanoid (**Dexmate Vega-1**) performing pick-and-place on lab consumables
(tubes, racks) using **cuRobo motion planning** — no RL, no imitation learning.

> Status: scaffolding stage — only `README.md` and `PLAN.md` exist. No code yet.
> See [`PLAN.md`](./PLAN.md) for the build plan, milestones, and open questions.

## Goal

A reproducible scene + cuRobo-driven pick-and-place demo. RL and imitation
learning are explicitly out of scope for now (see PLAN §1.2). The MVP is one
end-to-end script: drive base, pick a tube from a rack, drive to a target bench,
place the tube.

## Stack

| Layer | Choice | Notes |
| --- | --- | --- |
| Physics / renderer | Isaac Sim 5.1 (pip) | Installed via `uv pip`, not workstation |
| Framework | Isaac Lab 2.3 | `InteractiveScene` directly — no manager-based env (no MDP terms to manage) |
| Motion planning | **cuRobo v0.8.0** (CUDA-accelerated) | Whole-body planning over holonomic base + dual arms |
| Base planner fallback | PythonRobotics `a_star.py` | Copy-in only if cuRobo whole-body doesn't pan out |
| Python env | **`uv venv` + `uv pip`** | Isaac Lab has native `uv` support (PR #3172) |
| Python | 3.11 | Required by Isaac Sim 5.1 pip wheels |
| Robot | Dexmate Vega-1 (`dexmate-urdf` v0.8.3+) | Apache-2.0, URDF in repo + USD in releases |
| Lab assets | AutoBio meshes (converted) + NVIDIA Sim-Ready + primitives | Hybrid approach |

## Hardware

- **Current dev machine** — 2× NVIDIA RTX 2080 Ti (11 GB each), CUDA 12.8. Capable of running Sim 5.1 viewport + cuRobo locally.
- **Target run machine** — same as dev unless we move; reassess before M2 if multi-machine work is needed.

## Reference repos

We do **not** start from scratch. Sources are split into **code-pattern** references
(env-cfg structure, project layout) and **asset** references (USD/URDF/mesh files).
We prefer first-party (Isaac Sim / NVIDIA) and high-star community repos; low-star
third-party IsaacLab forks are intentionally excluded.

| Repo | Used for | License |
| --- | --- | --- |
| [`NVlabs/curobo`](https://github.com/NVlabs/curobo) | Core motion planner — dual-arm + holonomic-base whole-body planning (~1.5k★) | Apache-2.0 |
| [`isaac-sim/IsaacLab`](https://github.com/isaac-sim/IsaacLab) | First-party `CuroboPlanner` wrapper (`isaaclab_mimic/motion_planners/curobo/`, SkillGen PR #3303), `InteractiveScene`, `ridgeback_franka.py` (wheeled base template), `openarm/bimanual/...` (dual-arm joint conventions) | BSD-3 |
| [`AtsushiSakai/PythonRobotics`](https://github.com/AtsushiSakai/PythonRobotics) | Backup 2D base planner — copy-in `PathPlanning/AStar/a_star.py` if cuRobo whole-body falls short (29.5k★) | MIT |
| [`dexmate-ai/dexmate-urdf`](https://github.com/dexmate-ai/dexmate-urdf) | Dexmate Vega-1 URDF + USD release artifacts (asset only) | Apache-2.0 |
| [`autobio-bench/AutoBio`](https://github.com/autobio-bench/AutoBio) | Bio-lab asset meshes (centrifuge / PCR / pipette / tube / rack) — meshes only, MJCF code ignored | check repo |
| NVIDIA SO-101 sim-to-real tutorial | Only first-party lab-adjacent USDs (vials + rack) | NVIDIA tutorial |

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
├── src/bio_sim/
│   ├── robot/                # ArticulationCfg + cuRobo planning URDF/yml
│   ├── scene/                # InteractiveScene for the bio lab
│   ├── assets/               # bench, consumables, instruments
│   ├── motion/               # cuRobo wrapper, base controller, fallback A*
│   └── pipeline.py           # scripted pick-place state machine
├── scripts/{play.py, inspect_robot.py, curobo_smoke.py, convert_assets.py, download_assets.py}
└── third_party/autobio_assets/   # submodule, meshes only
```
