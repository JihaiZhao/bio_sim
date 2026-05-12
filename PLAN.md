# bio_sim — Build Plan

Detailed plan for building the bio-lab pick-and-place simulation. This document
is the source of truth for design decisions; `README.md` is a summary.

---

## 1. Scope

### 1.1 In-scope (MVP, Milestone M2)

- A **bio-lab scene** rendered in Isaac Sim 5.1 / Isaac Lab 2.3.
- **Dexmate Vega-1** loaded with working articulation (wheels/holonomic base + torso + dual arms + grippers).
- **Traditional motion planning only** — cuRobo for the arms (and ideally the base too, via whole-body planning over a holonomic dummy-joint chain).
- One pick-and-place pipeline: drive base to source bench → dual-arm pick a tube from a rack → drive to target bench → place in target rack.
- A `play.py` script that launches the scene and runs the demo end-to-end.

### 1.2 Out-of-scope (explicitly punted, may revisit later)

- **RL training** — no reward design, no parallel envs, no manager-based MDP terms. If we ever want RL, the pipeline is a separate plan.
- **Imitation learning / teleop / data collection** — no SpaceMouse, no trajectory recording, no robomimic/LeRobot integration. Pure scripted planner.
- Photorealistic rendering / domain randomization.
- Cross-machine deployment automation.
- Sim-to-real transfer.
- **Liquid simulation** (AutoBio's quasi-static liquid plugin is MuJoCo-only, no PhysX equivalent in scope).
- **Articulated lab instruments** (centrifuge lid, pipette plunger) — start with rigid bodies, articulate later if needed.

---

## 2. Stack & versions

| Item | Version | Why |
| --- | --- | --- |
| Isaac Sim | 5.1 (pip) | Latest stable, supports `uv pip` install path |
| Isaac Lab | 2.3.x | Built on Sim 5.1, ships first-party `CuroboPlanner` (via SkillGen, PR #3303) |
| Python | 3.11 | Required by Sim 5.1 pip wheels (cuRobo also supports up to 3.10, but Isaac Sim's `omni_python` resolves this) |
| uv | ≥ 0.9.18 | Already installed |
| CUDA driver | 570 / CUDA 12.8 | Already installed, compatible |
| dexmate-urdf | ≥ 0.8.3 | Latest as of 2026-05-08 |
| cuRobo | v0.8.0 (cuRoboV2, Apr 2026) | Motion planning — arms + holonomic base (whole-body) |
| PythonRobotics | latest, copy-in | Fallback 2D `a_star.py` for base, only if cuRobo whole-body doesn't pan out |

**Avoid**: conda, workstation Isaac Sim binary, Isaac Lab v3.0-beta (Newton branch — not ready).

---

## 3. Reference usage — what we take from each

Selection rule: prefer first-party (NVIDIA / Isaac Sim / Isaac Lab) and high-star
community repos. Avoid low-star third-party IsaacLab forks — their patterns drift
fast and we can't trust them not to disappear or break against IsaacLab 2.3.

### `NVlabs/curobo` *(motion planner — core dependency)*
- **What it does**: CUDA-accelerated motion planning. IK, collision-free trajectory optimization, MPC. Apache-2.0, ~1.5k stars, v0.8.0 / cuRoboV2 (Apr 2026).
- **Branch choice**: `main` (cuRoboV2 / v0.8.0) is the target. The repo restructured between `isaac-*` branches (had `src/curobo/...` + `examples/...`) and `main` (has `curobo/...`, no top-level `examples/`).
- **Built-in robot configs (`curobo/content/configs/robot/`)** — what cuRobo ships:
  - V2 / `main`: `franka.yml`, `ur10e.yml`, `dual_ur10e.yml`, `unitree_g1.yml` (+ retarget variant), `simple_mimic_robot.yml`.
  - Older `isaac-4.0` branch (worth pulling for **template files**): all of the above plus `franka_mobile.yml`, `iiwa.yml`, `iiwa_allegro.yml`, `jaco7.yml`, `kinova_gen3.yml`, `quad_ur10e.yml`, `tri_ur10e.yml`, `tm12.yml`, `ur5e.yml`, `ur5e_robotiq_2f_140.yml`, plus `template.yml`.
  - **No Dexmate Vega config exists.** We author `dexmate_vega.yml` from scratch (M1 task) using two templates: `dual_ur10e.yml` (dual-arm primary `ee_link` + `link_poses` + sphere structure) and `franka_mobile.yml` (3-dummy-joint holonomic base — pulled from `isaac-4.0` branch as reference; the pattern still works in V2 even though V2 stopped shipping the file).
- **Dual-arm**: Supported. One robot config holds both arms; primary `ee_link` + extra targets via `link_poses`. **Caveat**: collision constraints currently honor primary ee_link cleanly; secondary ee_link constraints are experimental.
- **Holonomic base (whole-body)**: Supported via 3 dummy joints (x_prismatic, y_prismatic, yaw_revolute) prepended to the kinematic chain. **Non-holonomic NOT supported** (cuRobo discussion #425). Vega-1's wheeled base must be modeled as holonomic in the planning URDF.
- **Take — first-party IsaacLab wrapper**: `source/isaaclab_mimic/isaaclab_mimic/motion_planners/curobo/curobo_planner.py` (`CuroboPlanner`, merged via IsaacLab PR #3303 as part of SkillGen). Use this directly instead of writing our own glue.
- **Install**: source-only, ~20 min build, via `omni_python -m pip install -e .[isaacsim] --no-build-isolation`. See Risk §7 for known install pain (CUDA 11.8 / GLIBCXX ABI).

### Mobile-base planning — primary plan: **cuRobo whole-body** (no second library)
- Model Vega-1 as `[x_prismatic, y_prismatic, yaw_revolute] + left_arm + right_arm` in a planning URDF. cuRobo's `MotionGen` plans the entire trajectory in one call, sharing one collision world. Eliminates the integration glue between two planners.
- Wiring on the sim side: feed the planned (x, y, yaw) trajectory into IsaacLab via `JointPositionActionCfg` on the same dummy joints. (No first-party `HolonomicActionCfg`, but `NonHolonomicActionCfg`'s dummy-joint convention is the IsaacLab norm — symmetric with cuRobo.)
- **Backup if this doesn't work** (e.g., cuRobo whole-body too slow over a 10 m bay): copy `PathPlanning/AStar/a_star.py` from `AtsushiSakai/PythonRobotics` (29.5k stars, MIT, copy-in style — not pip-installable). Plan the base path classically, freeze the base, then cuRobo plans the arms.

### `isaac-sim/IsaacLab` *(primary code-pattern source)*
- **Take — project scaffolding**: official extension template, run `./isaaclab.sh -n` to scaffold an external project. We use a **lighter** version (see §4) — no manager-based env wrapper since we have no MDP terms to manage.
- **Take — `InteractiveScene`** (`isaaclab.scene.InteractiveScene`): build the bio-lab scene directly with `ArticulationCfg` / `RigidObjectCfg` / `AssetBaseCfg`. This is the right granularity for a scripted planner — we don't need `ManagerBasedEnv` overhead.
- **Take — cuRobo wrapper**: `source/isaaclab_mimic/.../curobo_planner.py` (see above). Tests at `test_curobo_planner_franka.py` and `test_curobo_planner_cube_stack.py` show the usage idiom.
- **Take — bimanual joint cfg pattern**: `manager_based/manipulation/reach/config/openarm/bimanual/joint_pos_env_cfg.py` (PR #4089) — the only first-party bimanual setup. Read for actuator-group / joint-name conventions, not for the MDP terms.
- **Take — wheeled base ArticulationCfg**: `isaaclab_assets/robots/ridgeback_franka.py` (`RIDGEBACK_FRANKA_PANDA_CFG`). Use as a structural template for Dexmate's holonomic base modeling. See Risk §7 (issue #2254).
- **Don't take**: Don't fork IsaacLab; pip-install it.

### `dexmate-ai/dexmate-urdf` *(robot assets — only Vega source)*
- **Take**: `pip install dexmate_urdf` to get URDF; download USD from GitHub release v0.8.3 for `ArticulationCfg.spawn.usd_path`. Use the `vega_1` variant (full body, dual gripper).
- **Build ourselves**: `ArticulationCfg` with actuator groups (`base`, `torso`, `left_arm`, `right_arm`, `left_gripper`, `right_gripper`). Plus a **planning URDF** that replaces the wheeled base with the 3 dummy x/y/yaw joints cuRobo expects. The sim USD and the planning URDF stay in sync via the same robot description except for the base section.
- **Trust note**: Repo has ~5 stars but it is the official `dexmate-ai` org and the only published Vega source. Acceptable as an asset dependency.

### `autobio-bench/AutoBio` *(lab asset meshes)*
- **Take**: Tube, pipette, rack, centrifuge body, PCR body meshes. Git submodule under `third_party/autobio_assets/`. Convert OBJ/STL → USD via `isaaclab.sim.converters.MeshConverter` / `UrdfConverter` in `scripts/convert_assets.py`.
- **Don't take**: Their MJCF env configs or MuJoCo plugins.

### NVIDIA SO-101 sim-to-real tutorial *(only first-party lab-adjacent USDs)*
- **Take**: SO-101 "lightbox" tutorial scene ships **vials and a rack** USDs — currently the only NVIDIA-official lab-adjacent assets. Lift them as a sanity-check scene for M1. URL: `docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/`.
- **Don't take**: The SO-101 arm or the full tutorial scene.

### Assets we'll have to author or import
NVIDIA's built-in asset library has **no laboratory equipment** (kitchen/warehouse/hospital/office only). For non-AutoBio items (lab bench, centrifuge, PCR machine, pipettes): GrabCAD / Thingiverse / NIH 3D, then `MeshConverter` / `UrdfConverter`.

---

## 4. Repository layout

Pure scripted-planner layout — no manager-based env wrapper, no MDP terms, no
teleop infra. The scene is built with `InteractiveScene` and stepped from
`play.py`, which delegates planning to a thin cuRobo wrapper.

```
bio_sim/
├── pyproject.toml                  # uv workspace root
├── .python-version                 # "3.11"
├── README.md
├── PLAN.md                         # this file
├── .gitignore
├── src/
│   └── bio_sim/
│       ├── __init__.py
│       ├── robot/
│       │   ├── dexmate_cfg.py         # ArticulationCfg (sim USD with real wheels)
│       │   ├── dexmate_planning.urdf  # planning model: dummy x/y/yaw + torso + dual arms
│       │   └── curobo_robot.yml       # cuRobo robot config (collision spheres, ee_links, link_poses)
│       ├── scene/
│       │   └── bio_lab.py             # InteractiveScene: floor, bench, racks, tubes
│       ├── assets/
│       │   ├── bench.py               # tables/shelves (primitive or Sim-Ready)
│       │   ├── consumables.py         # tubes / racks (from AutoBio)
│       │   └── instruments.py         # centrifuge / PCR bodies (rigid)
│       ├── motion/
│       │   ├── planner.py             # cuRobo MotionGen wrapper (whole-body or arms-only)
│       │   ├── world.py               # cuRobo WorldConfig built from the scene
│       │   ├── base_controller.py     # maps planned (x,y,yaw) → wheel velocities, if needed
│       │   └── base_planner_fallback.py  # PythonRobotics A* (only if cuRobo whole-body fails)
│       └── pipeline.py                # scripted pick-place state machine
├── scripts/
│   ├── play.py                     # MVP entry point: scene + pipeline
│   ├── inspect_robot.py            # load Dexmate alone, verify articulation
│   ├── curobo_smoke.py             # run cuRobo motion_gen_reacher on Dexmate, no scene
│   ├── convert_assets.py           # one-shot mesh/URDF → USD via MeshConverter/UrdfConverter
│   └── download_assets.py          # fetch Dexmate USD release, init AutoBio submodule, lift SO-101 USDs
└── third_party/
    ├── autobio_assets/             # git submodule, meshes only
    └── PythonRobotics/             # (optional) only if base fallback is needed
```

**Why no `source/bio_sim/` extension layout**: dropped because we have no
manager-based MDP terms to register. If/when RL or IL is reconsidered, the
refactor to add an external IsaacLab extension is mechanical (~1-2 days).

---

## 5. Milestones

### M0 — Environment & scaffolding *(target: 1 day, deferrable to next session)*
- `uv venv --python 3.11 --seed .venv`
- `uv pip install isaacsim[all,extscache]==5.1.* --extra-index-url <nvidia>` + IsaacLab 2.3
- Install cuRobo from source: `omni_python -m pip install -e .[isaacsim] --no-build-isolation`. Verify with cuRobo's `examples/isaac_sim/motion_gen_reacher.py` (Franka) before touching Dexmate.
- Wire up `pyproject.toml` workspace, `.gitignore`, basic `play.py` that launches an empty stage
- **Smoke test for wheeled-base risk**: spawn `RIDGEBACK_FRANKA_PANDA_CFG` from `isaaclab_assets/robots/ridgeback_franka.py` and try a 1 m base translation. If broken (see issue #2254), this informs the M1 base-modeling decision.
- **Exit criteria**: empty stage launches; cuRobo reacher example runs on Franka; Ridgeback base behavior is known.

### M1 — Robot loading + planning model
- Pull `dexmate-urdf` v0.8.3 release tarball, extract sim USD.
- Build **two robot descriptions** that stay in sync:
  - `dexmate_cfg.py`: IsaacLab `ArticulationCfg` against the sim USD (real wheels). Reference patterns: `isaaclab_assets/robots/ridgeback_franka.py` (wheeled base) + `manager_based/manipulation/reach/config/openarm/bimanual/joint_pos_env_cfg.py` (dual-arm actuator/joint conventions).
  - `dexmate_planning.urdf`: planning URDF with the wheel chain replaced by 3 dummy joints `[x_prismatic, y_prismatic, yaw_revolute]`. This is what cuRobo plans against.
- Author `curobo_robot.yml`: collision spheres, primary `ee_link` (one arm), `link_poses` for the second arm, joint limits matching the planning URDF.
- Build incrementally: dummy-base only → +torso → +single arm → full bimanual + base.
- `inspect_robot.py`: spawn Dexmate sim USD, verify articulation. `curobo_smoke.py`: run a cuRobo plan from one config to another, no scene.
- **Exit criteria**: Dexmate visible in sim; cuRobo plans a collision-free trajectory between two arm poses with the dummy base fixed.

### M2 — Bio-lab scene + cuRobo pick-place *(MVP)*
- Convert 4 AutoBio meshes via `scripts/convert_assets.py`: `tube`, `tube_rack`, `bench` (or use Sim-Ready), `pipette` (visual-only). Optional: lift SO-101 vial+rack USDs.
- `src/bio_sim/scene/bio_lab.py`: `InteractiveScene` with floor + source bench + rack of 6 tubes + target bench + target rack.
- `src/bio_sim/motion/world.py`: build a cuRobo `WorldConfig` from the scene (meshes or cuboid approximations for non-target obstacles).
- `src/bio_sim/motion/planner.py`: thin wrapper over IsaacLab's first-party `CuroboPlanner` (or `MotionGen` directly), parameterized for whole-body planning.
- `src/bio_sim/pipeline.py` — scripted state machine:
  1. Plan whole-body trajectory: home → grasp pose above source rack (base near source bench, right arm above tube).
  2. Execute trajectory.
  3. Plan arm-only descent to grasp pose; execute; close gripper.
  4. Plan whole-body lift + transit to place pose above target rack.
  5. Plan arm-only descent to place pose; execute; open gripper.
- Base execution: feed planned (x, y, yaw) to dummy joints (planning model) and reconcile to wheel velocities for the sim USD via `motion/base_controller.py`. If Ridgeback smoke test in M0 flagged movement bugs, fall back to teleporting the base between waypoints for the MVP.
- **Exit criteria**: `python scripts/play.py` runs the full pick-and-place end-to-end with no manual intervention. Record a screen-capture.

### M3 — Robustness / scale-up (separate plan when reached)
- Multi-tube cycle (pick 6 tubes in sequence)
- Dynamic obstacles (other moving objects in the scene)
- Articulated lab instruments (centrifuge lid open/close)
- Move to target run machine for performance / replay

---

## 6. Open questions / decisions deferred

| # | Question | Blocks | Decide by |
| --- | --- | --- | --- |
| Q1 | ~~What is the target run machine (GPU, OS, network access)?~~ **Resolved 2026-05-12**: dev machine is 2× RTX 2080 Ti (11 GB each), CUDA 12.8 — adequate for local dev. Reassess only if multi-machine work appears. | — | — |
| Q2 | Do we want Dexmate's `F5D6` dexterous hand or the simpler 2-finger gripper for MVP? | M1 actuator design + cuRobo robot config | M1 start |
| Q3 | Whole-body cuRobo planning vs decoupled (cuRobo arms + classical 2D base)? | M2 planner wiring | After M1 cuRobo smoke |
| Q4 | Is the Dexmate USD already authored as a holonomic base, or only with real wheels? Drives whether we maintain two robot descriptions (sim USD + planning URDF) or one. | M1 robot loading | M1 start |
| Q5 | Should we set up a Linear/GitHub for issue tracking? | — | When team > 1 |

---

## 7. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| ~~4 GB VRAM cannot open Sim 5.1 viewport at all~~ **Resolved** — dev machine is 2× RTX 2080 Ti (11 GB each) | — | — | — |
| cuRobo install fails: CUDA 11.8 vs Isaac Sim's bundled CUDA; `_GLIBCXX_USE_CXX11_ABI` PyTorch/Isaac-Sim ABI mismatch (cuRobo issues #163, #411) | High | M0 blocked | Always install through `omni_python`; pin cuRobo v0.8.0; if persistent, fall back to running `motion_gen_reacher` in cuRobo's standalone path before integrating |
| Dexmate USD release doesn't include physics properties / inertia | Medium | M1 stability issues | Fall back to URDF importer in Isaac Sim, generate USD ourselves |
| Sim USD has real wheels but cuRobo needs a holonomic planar joint chain — two robot descriptions can drift out of sync | High | M1/M2 base motion incorrect | Keep both files (`dexmate_cfg.py` against sim USD, `dexmate_planning.urdf` for cuRobo) in `src/bio_sim/robot/`; smoke test that planned (x,y,yaw) and executed base pose agree within tolerance |
| cuRobo dual-arm secondary-ee_link constraints are experimental (discussion #209) — collision behavior on the off-side arm is not fully honored | Medium | Bimanual planning brittle | Plan one arm at a time when both arms are critical; use whole-arm collision spheres rather than tight ee_link constraints |
| AutoBio meshes unusable (non-watertight, no UVs, license unclear) | Medium | Slows M2 | Fall back to primitive cylinders for tubes; verify license before submoduling |
| Wheeled-base movement broken in IsaacLab 2.3 (issue #2254 against `RIDGEBACK_FRANKA_PANDA_CFG`) — Dexmate base may hit same bugs | Medium | M2 mobility blocked | M0 smoke test on Ridgeback; if broken, teleport the base between waypoints for MVP and plan arms with cuRobo |
| Isaac Lab 2.3 → 2.4 breaks `CuroboPlanner` wrapper API | Low | Refactor cost | Pin Isaac Lab version in `pyproject.toml`; the `CuroboPlanner` shipped in 2.3 is stable for SkillGen |
| Dexmate-urdf repo (5 stars, official org) goes stale or disappears | Low | Robot asset loss | Vendor the v0.8.3 release tarball into `third_party/dexmate_vega_usd/` once it works |

---

## 8. Definition of done (MVP)

- `uv sync && python scripts/play.py` (on target machine) shows Dexmate completing one full pick-and-place cycle, with **cuRobo** producing all trajectories (no hand-tuned joint splines).
- Repo includes a 10-second screen capture.
- README install steps are reproducible from a fresh checkout (Isaac Sim 5.1 + Isaac Lab 2.3 + cuRobo v0.8.0).
- This `PLAN.md` is updated with the actual versions / commit SHAs that worked, and the resolution of Q3 (whole-body vs decoupled) and Q4 (sim USD base modeling).
