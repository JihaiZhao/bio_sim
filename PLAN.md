# bio_sim — Build Plan

Detailed plan for building the bio-lab pick-and-place simulation. This document
is the source of truth for design decisions; `README.md` is a summary.

---

## 1. Scope

### 1.1 In-scope (MVP, Milestone M2)

- A **bio-lab scene** rendered in Isaac Sim 5.1 / Isaac Lab 2.3.
- **Dexmate Vega-1** loaded with working articulation (wheels + torso + dual arms + grippers).
- One **scripted pick-and-place task**: robot navigates to a lab bench, picks a tube from a rack, places it in a target rack.
- A `play.py` script that launches the scene and runs the demo end-to-end.

### 1.2 Out-of-scope (deferred)

- RL training (no reward design, no parallel envs — but **leave Manager-based env structure compatible** so we can add later).
- Teleoperation / IL data collection (M3+).
- Photorealistic rendering / domain randomization.
- Cross-machine deployment automation.
- Sim-to-real transfer.

### 1.3 Explicitly punted

- **Liquid simulation** (AutoBio's quasi-static liquid plugin is MuJoCo-only, no PhysX equivalent in scope).
- **Articulated lab instruments** (centrifuge lid, pipette plunger) — start with rigid bodies, articulate later if needed.

---

## 2. Stack & versions

| Item | Version | Why |
| --- | --- | --- |
| Isaac Sim | 5.1 (pip) | Latest stable, supports `uv pip` install path |
| Isaac Lab | 2.3.x | Built on Sim 5.1, includes SkillGen + Mimic |
| Python | 3.11 | Required by Sim 5.1 pip wheels |
| uv | ≥ 0.9.18 | Already installed |
| CUDA driver | 570 / CUDA 12.8 | Already installed, compatible |
| dexmate-urdf | ≥ 0.8.3 | Latest as of 2026-05-08 |

**Avoid**: conda, workstation Isaac Sim binary, Isaac Lab v3.0-beta (Newton branch — not ready).

---

## 3. Reference usage — what we take from each

### `isaac-sim/IsaacLab`
- **Take**: Run `./isaaclab.sh -n` (template generator) to scaffold `source/bio_sim/` as an external project. This is the official path; the standalone `IsaacLabExtensionTemplate` repo is deprecated.
- **Don't take**: Don't fork IsaacLab core; install it as a pip dependency.

### `userguide-galaxea/Galaxea_Lab`
- **Take**: Read `source/extensions/.../galaxea/direct/lift/pick_fruit_env.py` and **port the env-config pattern** (scene config, action terms, observation terms, reward terms) to `bio_sim/tasks/pick_place/`. Galaxea R1 is also a wheeled dual-arm humanoid, so the actuator grouping and base-controller pattern map almost 1:1 to Dexmate.
- **Don't take**: Their fruit assets, their R1 robot config (we use Dexmate).
- **Risk**: They fork IsaacLab itself. We must extract the env logic without inheriting their fork structure.

### `dexmate-ai/dexmate-urdf`
- **Take**: `pip install dexmate_urdf` to get URDF; download USD from GitHub release v0.8.3 for use as `ArticulationCfg.spawn.usd_path`. Use the `vega_1` variant (full body, dual gripper hand — not `vega_1u` upper-only).
- **Build ourselves**: `ArticulationCfg` with actuator groups (`wheels`, `torso`, `left_arm`, `right_arm`, `left_gripper`, `right_gripper`), joint position limits, and an action-space mapping. No upstream IsaacLab example exists yet — this is original work.

### `unitreerobotics/unitree_sim_isaaclab`
- **Take**: Reference for **how to expose a humanoid as an IsaacLab env** outside the IsaacLab core repo. Useful patterns: how they handle dual-arm action splitting, how they manage the base controller vs the arm controller.
- **Don't take**: G1 assets.

### `autobio-bench/AutoBio`
- **Take**: Lab consumable meshes (`tube`, `pipette`, `rack`, `centrifuge` body, `PCR` body). Pull as a git submodule under `third_party/autobio_assets/`. Convert OBJ/STL → USD via Isaac Sim's mesh importer or a one-shot `omni.kit.tool.mesh_to_usd` script.
- **Don't take**: Their MuJoCo plugins (thread/detent/eccentric/liquid mechanisms) — re-implement only what we need in PhysX, and only when needed. Their MJCF env configs are MuJoCo-format and not directly portable.

---

## 4. Repository layout

```
bio_sim/
├── pyproject.toml                  # uv workspace root
├── .python-version                 # "3.11"
├── README.md
├── PLAN.md                         # this file
├── .gitignore
├── source/
│   └── bio_sim/                    # IsaacLab external project (template-generated)
│       ├── pyproject.toml          # extension package
│       └── bio_sim/
│           ├── __init__.py
│           ├── tasks/
│           │   ├── __init__.py
│           │   └── pick_place/
│           │       ├── pick_place_env_cfg.py
│           │       ├── mdp/                # actions, observations, rewards, terminations
│           │       └── config/
│           │           └── dexmate/
│           │               └── joint_pos_env_cfg.py
│           ├── assets/
│           │   ├── robots/
│           │   │   └── dexmate.py          # ArticulationCfg
│           │   └── lab/
│           │       ├── bench.py            # tables/shelves (primitive or Sim-Ready)
│           │       ├── consumables.py      # tubes / racks (from AutoBio)
│           │       └── instruments.py      # centrifuge / PCR (rigid for now)
│           └── scene/
│               └── bio_lab_cfg.py
├── scripts/
│   ├── play.py                     # MVP: launch scene + scripted pick-place
│   ├── inspect_robot.py            # load Dexmate alone, no scene
│   └── convert_autobio_meshes.py   # one-shot OBJ → USD
└── third_party/
    └── autobio_assets/             # git submodule, meshes only
```

---

## 5. Milestones

### M0 — Environment & scaffolding *(target: 1 day, deferrable to next session)*
- `uv venv --python 3.11 --seed .venv`
- `uv pip install isaacsim[all,extscache]==5.1.* --extra-index-url <nvidia>`
- Clone IsaacLab to a sibling dir, run template generator → produces `source/bio_sim/`
- Wire up `pyproject.toml` workspace, `.gitignore`, basic `play.py` that launches an empty stage
- **Exit criteria**: `python scripts/play.py` opens an empty Isaac Sim stage and exits cleanly

### M1 — Robot loading
- Pull `dexmate-urdf` v0.8.3 release tarball, extract USD
- Write `assets/robots/dexmate.py` with `ArticulationCfg`, actuator groups
- `inspect_robot.py`: spawn Dexmate on a ground plane, run for 5 s, robot stays standing
- **Exit criteria**: Dexmate visible, arms reach a hand-coded joint target without flipping over

### M2 — Bio-lab scene + scripted pick-place *(MVP)*
- Convert 4 AutoBio meshes: `tube`, `tube_rack`, `bench` (or use Sim-Ready), `pipette` (visual-only)
- `scene/bio_lab_cfg.py`: floor + bench + rack with 6 tubes + target rack
- Write scripted motion-planning routine in `play.py`: drive base → arm IK to tube → close gripper → lift → drive to target → place
- Use `isaaclab.controllers.DifferentialIKController` for arm; base controller TBD (likely position holonomic for MVP)
- **Exit criteria**: full pick-and-place runs end-to-end with no manual intervention. Record a screen-capture.

### M3 — Optional follow-ons (separate plan when reached)
- Teleop with `IsaacLab Mimic` for demo collection
- Manager-based env for RL (port from `pick_fruit_env.py`)
- Domain randomization
- Move to target machine, headless training

---

## 6. Open questions / decisions deferred

| # | Question | Blocks | Decide by |
| --- | --- | --- | --- |
| Q1 | What is the target run machine (GPU, OS, network access)? | M2 (likely runs slow on local 4GB) | Before M2 |
| Q2 | Do we want Dexmate's `F5D6` dexterous hand or the simpler 2-finger gripper for MVP? | M1 actuator design | M1 start |
| Q3 | Base controller model for MVP — holonomic position cmd, diff-drive, or scripted spline? | M2 motion code | M2 start |
| Q4 | Should we set up a Linear/GitHub for issue tracking? | — | When team > 1 |

---

## 7. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| 4 GB VRAM cannot open Sim 5.1 viewport at all | Medium | Blocks local dev | Headless mode for M0/M1; budget time to move to target machine before M2 |
| Dexmate USD release doesn't include physics properties / inertia | Medium | M1 stability issues | Fall back to URDF importer in Isaac Sim, generate USD ourselves |
| AutoBio meshes unusable (non-watertight, no UVs, license unclear) | Medium | Slows M2 | Fall back to primitive cylinders for tubes; verify license before submoduling |
| Galaxea_Lab patterns don't port cleanly (their IsaacLab fork has private hooks) | Low-Med | M2 design churn | Use `unitree_sim_isaaclab` as backup reference pattern |
| Isaac Lab 2.3 → 2.4 breaks our env config | Low | Refactor cost | Pin Isaac Lab version in `pyproject.toml` |

---

## 8. Definition of done (MVP)

- `uv sync && python scripts/play.py` (on target machine) shows Dexmate completing one full pick-and-place cycle.
- Repo includes a 10-second screen capture.
- README install steps are reproducible from a fresh checkout.
- This `PLAN.md` is updated with the actual versions / commit SHAs that worked.
