# bio_sim

Bio-lab pick-and-place simulation on **Isaac Sim 5.1** with two robots
(**G2** dual-arm omnipicker, **R1 Pro** BEHAVIOR-1K holonomic base) and a
cuRobo-planned skill pipeline. Runs windowed, native-headless, or **headless
with WebRTC livestream + an HTTP control plane** so a remote frontend can
drive it.

## Quick start

```bash
uv sync                          # installs Isaac Sim 5.1 + cuRobo (editable)

uv run python -m bio_sim list    # show robots / scenes / tasks
uv run python -m bio_sim run     # default: g2 + ot_one + pick_place, windowed
```

The first sync builds cuRobo CUDA extensions — slow once, cached after.
Press **Play** in the viewport to start; **R** to reset the env and replay.

## CLI

Three subcommands, all driven off the registry in `bio_sim/specs.py`:

```bash
bio_sim list                                       # registry tables
bio_sim run    --robot {g2,r1pro} --scene {bio,ot_one,demo} --task {pick_place,demo}
bio_sim serve  --port 8000                         # headless + WebRTC + HTTP
```

Shared knobs: `--headless-mode {native,websocket}`, `--robot-yml PATH`
(override cuRobo planner yml), `--use-urdf-kinematics`, `--reactive`
(replanning instead of one-shot).

CLI is a **selector, not a config-override surface**: a new variation is a
new registry entry, not a new flag.

## What ships

| Layer | Pieces |
| --- | --- |
| **Robots**   | `g2` (kinematic base), `r1pro` (holonomic — pos drive on x/y/rz) |
| **Scenes**   | `bio` (A/B tables), `ot_one` (plate on OT-One deck), `demo` (single-table) |
| **Tasks**    | `pick_place` (scripted navigate + pick + place), `demo` (pure-arm, no base motion) |
| **Skills**   | `grasp`, `move_arm` (cuRobo MotionGen), `navigate` (scripted FaceYaw / DriveStraight) |
| **Planning** | cuRobo MotionGen via `nvidia-curobo` (editable `third_party/curobo`) |
| **Multi-env**| `Cloner.clone /World/env_0 -> env_1..N-1` for visual mirrors; per-env grasp welds |
| **Frontend bridge** | WebRTC livestream on `:49100` + FastAPI on `:8000` (see below) |

## Frontend bridge (`bio_sim serve`)

Two independent planes to a running headless sim:

- **Video** — WebRTC on `:49100` via `omni.kit.livestream.webrtc`. Any
  WebRTC client works; NVIDIA's
  [`web-viewer-sample`](https://github.com/NVIDIA-Omniverse/web-viewer-sample)
  (branch `1.5.2`) is the verification client.
- **Control** — FastAPI on `--port` (default 8000):
  - `GET  /status`  — runner progress snapshot
  - `POST /reset`   — env reset (same code path as the windowed R-key)
  - `GET  /tasks`   — task registry metadata
  - `GET  /healthz` — uvicorn liveness
  - `GET  /docs`    — Swagger UI

Threading: uvicorn runs in a daemon thread, handlers enqueue
`Command(op, args, Future)`, the sim main thread drains the queue at the
top of every tick. No USD / PhysX touched off-thread.

Smoke test the stream alone: `python scripts/livestream_smoke.py`.

## Layout

```
bio_sim/
├── bio_sim/
│   ├── cli.py           # typer-style CLI (run / serve / list)
│   ├── commands.py      # shared action bus (reset_env, status)
│   ├── server.py        # FastAPI control plane + queue drain
│   ├── runner.py        # SkillRunner state machine
│   ├── specs.py         # Robot / Scene / Task registry
│   ├── robot/           # g2.py, r1pro.py, robot_base.py, arm/gripper/base/holonomic
│   ├── scene/           # bio_scene, ot_one_scene, demo_scene, lighting
│   ├── tasks/           # pick_place.py, demo.py (skill-list builders)
│   ├── skills/          # grasp, move_arm, navigate
│   ├── sim/             # SimApp: SimulationApp + World + main step loop
│   └── config/          # task & robot yaml overlays, cuRobo configs
├── assets/              # USDs + meshes (robot/G2, robot/r1pro, objects, lighting)
├── scripts/             # livestream_smoke + diag_*.py debug tools
└── third_party/curobo/  # editable cuRobo install
```

## Requirements

- **Python 3.11** exactly (Isaac Sim 5.1 wheels are CPython 3.11 only).
- **uv** for dependency management — `uv sync` is the only supported path.
  Don't `pip install` anything: cuRobo must stay declared in `pyproject.toml`
  or `uv sync` will wipe it.
- **NVIDIA GPU + CUDA 12.8** drivers (torch wheels are pinned to cu128).
- cuRobo builds CUDA extensions on first sync; sm_120 (RTX 5090) needs the
  LBFGS kernel flag set — `use_cuda_kernel: False` is already in the planner ymls.

## Notes

- Per-robot task overlays live in `bio_sim/config/robots/<robot>.yaml`;
  adding a new robot = new spec + new overlay file, no CLI changes.
- Authoritative state of what runs is the registry in `bio_sim/specs.py`;
  this README is a summary.
