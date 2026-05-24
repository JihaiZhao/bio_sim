# bio_sim

Bio-lab pick-and-place simulation on **Isaac Sim 5.1** with two robots
(**G2** dual-arm omnipicker, **R1 Pro** BEHAVIOR-1K holonomic base) and a
cuRobo-planned skill pipeline. Runs windowed, native-headless, or **headless
with WebRTC livestream + an HTTP control plane** so a remote frontend can
drive it.

---

## Prerequisites

- **Linux** + **NVIDIA GPU** with CUDA 12.8 drivers (`nvidia-smi` reports
  `CUDA Version: 12.8+`). torch is pinned to cu128 wheels; older drivers
  will load torch but cuRobo's CUDA kernels will refuse to build.
- **Python 3.11 exactly**. Isaac Sim 5.1 wheels are CPython 3.11 only;
  3.10 and 3.12 will fail at install.
- **uv** ≥ 0.5 — single-source dep manager:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **git** + **disk** — fresh clone+sync grows to ~10 GB (Isaac Sim wheels
  + Kit extension cache + cuRobo build artifacts).
- **Node ≥ 18 / npm ≥ 10** — *only* if you want to run the verification
  frontend locally. `nvm install 20` works on Ubuntu 24.04 (apt's npm is
  9.x, too old).

---

## Setup

```bash
git clone <this-repo> && cd bio_sim

# cuRobo is referenced as an editable install at third_party/curobo/, but
# the directory is .gitignored (not vendored, not a submodule). Clone it
# yourself before syncing:
mkdir -p third_party && \
  git clone https://github.com/NVlabs/curobo.git third_party/curobo
# Known-working pin (latest tested in this project):
git -C third_party/curobo checkout d64c4b0   # past tag v0.7.8

uv sync
```

The first `uv sync` does three slow things back to back:

1. Pulls Isaac Sim 5.1 wheels from `pypi.nvidia.com` (~2 GB).
2. Pulls torch cu128 from PyTorch's index.
3. Builds cuRobo's CUDA extensions from `third_party/curobo/` (the
   editable install in `pyproject.toml`'s `[tool.uv.sources]`).
   This is the longest step (~5–15 min depending on the GPU arch). If
   `third_party/curobo/` doesn't exist, `uv sync` fails with
   `path does not exist` — that means you skipped the clone above.

After it finishes, verify nothing is broken **without booting Isaac**:

```bash
uv run python -m bio_sim list
```

This prints the robot / scene / task registry tables. It doesn't import
`isaacsim.core`, so it's the cheap way to confirm the install resolved.

> ⚠️ **Don't `pip install` into `.venv`.** cuRobo must stay declared in
> `pyproject.toml`; a bare `uv sync` wipes anything pip-installed that
> isn't declared, including cuRobo itself, which silently breaks every
> robot import. Always go through `uv add` / edit `pyproject.toml`.

> ⚠️ **Assets gap.** Several scene/robot USDs (`assets/lighting/`,
> `assets/objects/`, `assets/Collected_World0/`, parts of `assets/robot/r1pro/`)
> aren't checked in yet. A fresh clone has enough to import code and run
> `list`, but `run` / `serve` will fail until the asset pack is added.
> This is being tracked separately.

---

## Running a task

The default — `g2` + `ot_one` + `pick_place`, windowed:

```bash
uv run python -m bio_sim run
```

What you'll see, in order:

1. Kit window opens. Stage builds (table, OT-One, plate, G2). Console
   prints `Ready. Press the PLAY button in the viewport to start.`
2. Click the **Play** ▶ button in the toolbar. The first cuRobo plan
   takes a few seconds; afterwards the arm starts moving.
3. `[runner] -> NavigateTo` / `MoveArmTo` / `Grasp` / ... log lines
   advance through the skill list.
4. On `task COMPLETE` the sim keeps running (timeline still on). Press
   **R** in the viewport to reset the env + replay the same task without
   relaunching. Press **P** to flip RealTime ↔ PathTracing for screenshots.

### Variations

```bash
# Different robot / scene / task — pick any compatible combo from `list`
uv run python -m bio_sim run --robot g2 --scene ot_one --task pick_place
uv run python -m bio_sim run --robot g2 --scene demo --task demo

# Headless (no window, exits when task finishes)
uv run python -m bio_sim run --headless-mode native

# Override cuRobo planner yml (paths under bio_sim/config/curobo/)
uv run python -m bio_sim run --robot-yml my_tuned_planner.yml

# cuRobo reactive replanning (vs default one-shot plan-then-execute)
uv run python -m bio_sim run --reactive
```

CLI is a **selector, not a config-override surface**: a new variation is
a new registry entry in `bio_sim/specs.py`, not a new flag. Use a YAML
overlay if you need to tweak task parameters per robot — see
`bio_sim/config/robots/`.

> ⚠️ **Stick with `--robot g2` for now.** `r1pro` is registered and
> partially wired (BEHAVIOR-1K holonomic base + cuRobo arm), but the
> end-to-end pick-and-place isn't validated on it yet — grasp poses,
> base trajectories, and some scene combinations still misbehave.
> Use it only if you're working on R1 Pro bring-up.

---

## What ships

| Layer | Pieces |
| --- | --- |
| **Robots**   | `g2` ✅ fully working — recommended default. `r1pro` ⚠️ WIP (see note below) |
| **Scenes**   | `bio` (A/B tables), `ot_one` (plate on OT-One deck), `demo` (single-table) |
| **Tasks**    | `pick_place` (scripted navigate + pick + place), `demo` (pure-arm, no base motion) |
| **Skills**   | `grasp`, `move_arm` (cuRobo MotionGen), `navigate` (scripted FaceYaw / DriveStraight) |
| **Planning** | cuRobo MotionGen via `nvidia-curobo` (editable `third_party/curobo/`) |
| **Multi-env**| `Cloner.clone /World/env_0 -> env_1..N-1` for visual mirrors; per-env grasp welds |
| **Frontend bridge** | WebRTC livestream on `:49100` + FastAPI on `:8000` (see below) |

---

## Frontend integration (`bio_sim serve`)

`serve` is the headless mode meant for an external frontend. Two
independent planes come up on the same machine:

| Plane | Port | Protocol | Source |
| --- | --- | --- | --- |
| **Video**   | `49100` | WebRTC signaling (websocket) | `omni.kit.livestream.webrtc` |
| **Control** | `--port` (default `8000`) | HTTP / JSON | FastAPI (`bio_sim/server.py`) |

### Step-by-step: localhost

**Terminal 1 — sim:**
```bash
uv run python -m bio_sim serve --scene demo --task demo
# wait for:
#   [server] uvicorn listening on http://0.0.0.0:8000
#   [reset] press  R  to reset...   <-- (R key won't work in serve, use POST /reset)
```

**Terminal 2 — verification viewer** (NVIDIA's
[`web-viewer-sample`](https://github.com/NVIDIA-Omniverse/web-viewer-sample)
branch `1.5.2`):
```bash
git clone --branch 1.5.2 --depth 1 \
  https://github.com/NVIDIA-Omniverse/web-viewer-sample.git
cd web-viewer-sample
npm install
npm run dev      # serves on http://localhost:5173/
```
The default `stream.config.json` already targets `127.0.0.1:49100`, no
edit needed.

**Browser:**
1. Open `http://localhost:5173/`.
2. Pick **"UI for any streaming app"** (NOT "USD Viewer" — that template
   expects the USD Viewer Kit app, not bio_sim).
3. The Isaac stage renders in the page. To trigger a reset:
   ```bash
   curl -X POST http://localhost:8000/reset
   ```
   or open `http://localhost:8000/docs` for Swagger UI.

### HTTP endpoints

| Route | Returns / Effect |
| --- | --- |
| `GET  /status`  | runner progress: `step_index`, current `skill`, `done`, `failed`, `held` |
| `POST /reset`   | gripper release → reset arm/base → randomize + reset plate → runner.restart() |
| `GET  /tasks`   | task registry metadata (no sim round-trip) |
| `GET  /healthz` | uvicorn liveness only; use `/status` for sim liveness |
| `GET  /docs`    | FastAPI's auto-generated Swagger UI |

Threading model: uvicorn runs in a daemon thread; handlers enqueue
`Command(op, args, Future)` onto a queue; the sim main thread drains the
queue at the top of every tick (between physics steps) and fulfils each
`Future`. No USD / PhysX is ever touched off-thread.

### Cross-machine deployment

Sim on machine A (with the GPU), frontend on machine B:

- `bio_sim serve --host 0.0.0.0` (default) binds both ports on all
  interfaces. From the frontend machine, point its WebRTC client at
  `ws://<A's IP>:49100` and its HTTP client at `http://<A's IP>:8000`.
- Open both ports in any firewall between them.
- CORS on `/` is fully open (`allow_origins=["*"]`) for dev convenience.
  Tighten before exposing this publicly.

### Smoke test the stream alone

If `serve` won't connect, isolate the bug:
```bash
uv run python scripts/livestream_smoke.py
```
This boots a minimal headless stage (ground + magenta cube) with the
WebRTC ext enabled and nothing else. If the cube renders in the viewer
but `serve` doesn't, the bug is in the bio_sim sim setup, not the
streaming layer.

---

## Layout

```
bio_sim/
├── bio_sim/
│   ├── cli.py           # CLI (run / serve / list)
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
├── assets/              # USDs + meshes (G2, R1 Pro, objects, lighting)
├── scripts/             # livestream_smoke + diag_*.py debug tools
└── third_party/curobo/  # editable cuRobo install (vendored, not a submodule)
```

---

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `ModuleNotFoundError: curobo` after a fresh `uv sync` | Someone `pip install`'d into the venv and cuRobo got reaped. Re-`uv sync` to reinstall. |
| `LBFGScu` / register-count crash on first plan | sm_120 (RTX 5090) kernel issue. Confirm the planner yml has `use_cuda_kernel: False` (default in shipped ymls). |
| `serve` boots but viewer shows no video | The WebRTC ext didn't enable. Check console for `omni.kit.livestream.webrtc` lines. Try `scripts/livestream_smoke.py` to isolate. |
| `POST /reset` returns 504 | The sim main loop isn't draining the queue. Are you still on the "press Play" screen? `serve` mode auto-plays — if you don't see `[sim] task ...` lines, the sim never settled. |
| `npm install` ECONNRESET (from CN) | npmjs.org is unreliable from CN. In `web-viewer-sample/.npmrc` keep `@nvidia:registry=...edge.urm.nvidia.com...` and switch the default `registry=` to `https://registry.npmmirror.com`. |
| First plan takes >30 s | Normal on the first launch — cuRobo builds collision world from the USD stage. Subsequent resets are fast. |

---

## Notes

- Authoritative state of what runs is the registry in `bio_sim/specs.py`;
  this README is the human view.
- Per-robot task overlays live in `bio_sim/config/robots/<robot>.yaml`;
  adding a new robot = new spec + new overlay file, no CLI changes.
- cuRobo lives at `third_party/curobo/` as an **editable install
  declared in `[tool.uv.sources]`**, but the directory itself is
  `.gitignore`d — *not* a submodule, *not* vendored. Each developer
  clones `NVlabs/curobo` into that path themselves (see Setup). If we
  later promote it to a submodule, this Note goes away.
