#
# bio_sim CLI. Two subcommands: `list` (no sim boot, pure metadata read) and
# `run` (boots SimApp, lazy-imports the picked specs, runs the task).
#
#   python -m bio_sim list
#   python -m bio_sim list robots
#   python -m bio_sim run
#   python -m bio_sim run --robot r1pro --scene ot_one --task pick_place
#   python -m bio_sim run --headless-mode native
#
# DESIGN:
#   * tyro over @dataclass commands joined by Union[Annotated[..., subcommand]].
#   * --robot/--scene/--task Enums are BUILT FROM bio_sim.specs.* dicts at
#     import time -> adding a spec auto-extends CLI choices + tab completion.
#   * The CLI is a SELECTOR, not a config-override surface. We deliberately
#     do NOT expose yaml fields (--cube-xyz, --riser-size, ...). New behaviour
#     comes from a new spec + new yaml, not from new flags.
#   * Boot ordering: this module imports ONLY bio_sim.specs + tyro/rich, no
#     heavy deps. `list` returns in <1 s. `run` constructs SimApp FIRST and
#     only then resolves spec.cls_ref / builder_ref via load_ref().
#

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Optional, Union

import numpy as np
import tyro
from rich.console import Console
from rich.table import Table

from bio_sim.specs import (
    DEFAULTS,
    ROBOTS,
    SCENES,
    TASKS,
    SceneSpec,
    TaskSpec,
    load_ref,
)


# --------------------------------------------------------------------------- #
# Enums dynamically derived from the registries. Adding a spec to specs.py
# auto-extends the CLI's --robot/--scene/--task choices (and --help output);
# cli.py itself does not change.
# --------------------------------------------------------------------------- #
RobotName = Enum("RobotName", {n: n for n in ROBOTS})
SceneName = Enum("SceneName", {n: n for n in SCENES})
TaskName = Enum("TaskName", {n: n for n in TASKS})
HeadlessMode = Enum("HeadlessMode", {"native": "native", "websocket": "websocket"})


# --------------------------------------------------------------------------- #
# Command dataclasses. Each one is a tyro subcommand; field defaults become
# CLI defaults; type annotations drive parsing + --help.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class List_:
    """List available robots, scenes, and tasks (no sim boot).

    Positional `kind` filters to one section: robots | scenes | tasks.
    Omit to show all three.
    """

    kind: tyro.conf.Positional[Optional[str]] = None


@dataclass(frozen=True)
class Run:
    """Boot Isaac Sim and run a robot+scene+task combination."""

    robot: RobotName = RobotName[DEFAULTS["robot"]]
    """Which robot to load."""

    scene: SceneName = SceneName[DEFAULTS["scene"]]
    """Which scene layout to build."""

    task: TaskName = TaskName[DEFAULTS["task"]]
    """Which task (skill list) to run."""

    headless_mode: Optional[HeadlessMode] = None
    """Omit for a window; native = no GUI; websocket = livestreamed."""

    robot_yml: Optional[str] = None
    """Override the cuRobo planner yml for the chosen robot.
    Omit -> the spec's default_curobo_yml."""

    use_urdf_kinematics: bool = False
    """Use URDF kinematics instead of the USD ArticulationView."""

    reactive: bool = False
    """Enable cuRobo reactive replanning (vs one-shot plan-then-execute)."""


@dataclass(frozen=True)
class Serve:
    """Boot Isaac Sim headless with WebRTC livestream + an HTTP control plane.

    Two ports come up:
      * 49100  -- WebRTC signaling (video; consumed by a browser client)
      * --port -- HTTP/JSON commands (POST /reset, GET /status, GET /tasks)

    Pair with NVIDIA's web-viewer-sample (or any WebRTC client) for video,
    and any HTTP client (curl, the frontend) for commands. The task list
    is loaded at boot and not switchable at runtime in this iteration --
    relaunch `serve` with a different --task to change it.
    """

    robot: RobotName = RobotName[DEFAULTS["robot"]]
    """Which robot to load."""

    scene: SceneName = SceneName[DEFAULTS["scene"]]
    """Which scene layout to build."""

    task: TaskName = TaskName[DEFAULTS["task"]]
    """Which task (skill list) to run."""

    port: int = 8000
    """HTTP control-plane port (FastAPI / uvicorn)."""

    host: str = "0.0.0.0"
    """Bind address. 0.0.0.0 = all interfaces; 127.0.0.1 = local only."""

    no_livestream: bool = False
    """Skip the WebRTC ext (debug only -- no video will reach the frontend)."""

    robot_yml: Optional[str] = None
    """Override the cuRobo planner yml for the chosen robot."""

    use_urdf_kinematics: bool = False
    """Use URDF kinematics instead of the USD ArticulationView."""

    reactive: bool = False
    """Enable cuRobo reactive replanning."""


# tyro command-tree pattern: Union of dataclasses; each Annotated with its
# subcommand name. Names become `python -m bio_sim list` / `run` / `serve`.
Cmd = Union[
    Annotated[List_, tyro.conf.subcommand(name="list")],
    Annotated[Run, tyro.conf.subcommand(name="run")],
    Annotated[Serve, tyro.conf.subcommand(name="serve")],
]


# --------------------------------------------------------------------------- #
# `list` rendering. Pure metadata read from specs.py -- never touches SimApp.
# --------------------------------------------------------------------------- #
_console = Console()


def _mark_default(name: str, kind: str) -> str:
    return f"[bold]{name}[/bold] [cyan]*[/cyan]" if name == DEFAULTS[kind] else name


def _table_robots() -> Table:
    t = Table(title="ROBOTS", title_style="bold", show_lines=False)
    t.add_column("name", style="green")
    t.add_column("description")
    for name, spec in ROBOTS.items():
        t.add_row(_mark_default(name, "robot"), spec.description)
    return t


def _table_scenes() -> Table:
    t = Table(title="SCENES", title_style="bold", show_lines=False)
    t.add_column("name", style="green")
    t.add_column("description")
    for name, spec in SCENES.items():
        t.add_row(_mark_default(name, "scene"), spec.description)
    return t


def _table_tasks() -> Table:
    t = Table(title="TASKS", title_style="bold", show_lines=False)
    t.add_column("name", style="green")
    t.add_column("description")
    t.add_column("scenes", style="magenta")
    for name, spec in TASKS.items():
        t.add_row(
            _mark_default(name, "task"),
            spec.description,
            ", ".join(spec.compatible_scenes),
        )
    return t


def _render_tables(kind: Optional[str]) -> None:
    sections = {
        None: (_table_robots, _table_scenes, _table_tasks),
        "robots": (_table_robots,),
        "scenes": (_table_scenes,),
        "tasks": (_table_tasks,),
    }
    if kind not in sections:
        _console.print(
            f"[red]unknown kind {kind!r}; expected one of: robots, scenes, tasks[/red]"
        )
        raise SystemExit(2)
    for make in sections[kind]:
        _console.print(make())
    if kind is None:
        _console.print("[dim]  [cyan]*[/cyan] = default[/dim]")


# --------------------------------------------------------------------------- #
# `run` -- equivalent to the old play.py:main, modulo the spec lookup.
# Booting SimApp() is the FIRST thing that touches the sim runtime; only
# after that is it safe to load_ref() the robot/scene/task implementations.
# --------------------------------------------------------------------------- #


def _validate_compat(task: str, scene: str) -> TaskSpec:
    spec = TASKS[task]
    if scene not in spec.compatible_scenes:
        _console.print(
            f"[red]task '{task}' is not compatible with scene '{scene}'.[/red]\n"
            f"  valid scenes for this task: "
            f"{', '.join(spec.compatible_scenes)}"
        )
        raise SystemExit(2)
    return spec


def _build(sim, robot_spec, scene_spec, task_spec,
           robot_yml, use_urdf_kinematics, reactive):
    """After SimApp is up, resolve specs and assemble scene/robot/runner.

    Shared by `_run` and `_serve`. Returns (ctx, runner, on_world_sync)
    ready to be handed to sim.run(). Side effect: prints config banner.
    """
    SceneCls = load_ref(scene_spec.cls_ref)
    RobotCls = load_ref(robot_spec.cls_ref)
    build_task = load_ref(task_spec.builder_ref)

    from bio_sim.runner import SkillRunner
    from bio_sim.skills import SkillContext
    from bio_sim.tasks.pick_place import load_full_cfg

    cfg = load_full_cfg(robot_spec.cfg_overlay, task_spec.config_file)
    print(
        f"[cli] robot={robot_spec.name} scene={scene_spec.name} task={task_spec.name}"
        f"  cfg = {task_spec.config_file} + robots/{robot_spec.cfg_overlay}.yaml"
    )

    scene = SceneCls.from_cfg(cfg)
    scene.build(sim)

    yml = robot_yml if robot_yml is not None else robot_spec.default_curobo_yml
    robot = RobotCls(
        robot_yml=yml,
        use_urdf_kinematics=use_urdf_kinematics,
        reactive=reactive,
    )
    # Non-invasive per-task init pose (genie_sim-style): overlay cfg's
    # init_arm_pose onto retract_config IN MEMORY before load_into rebuilds
    # the cuRobo planner, so the committed robot yml is untouched.
    robot.apply_init_pose(cfg)
    # Grasp mechanism (physics friction vs. assist FixedJoint weld).
    robot.gripper.set_mode(cfg.get("grasp_mode", "physics"))
    robot.load_into(sim, scene)
    scene.place_for_validation(robot, cfg)
    # Phase 2: clone env_0 -> env_1..N-1 BEFORE physics initializes so
    # PhysX picks up every articulation root in one initialize_physics()
    # call. replicate_physics=True duplicates every UsdPhysics.*API on
    # the source subtree (robot articulation, RigidBody fixtures, etc.).
    num_envs = int(cfg.get("num_envs", 1))
    env_spacing = float(cfg.get("env_spacing", 0.0))
    if num_envs > 1:
        from isaacsim.core.cloner import Cloner

        cloner = Cloner(stage=sim.world.stage)
        env_paths = [f"/World/env_{i}" for i in range(num_envs)]
        positions = np.array(
            [[i * env_spacing, 0.0, 0.0] for i in range(num_envs)],
            dtype=np.float32,
        )
        cloner.clone(
            source_prim_path="/World/env_0",
            prim_paths=env_paths,
            positions=positions,
            replicate_physics=True,
            base_env_path="/World",
            root_path="/World/env_",
        )
        print(f"[cli] cloned env_0 -> {num_envs - 1} replica(s) "
              f"at +X spacing={env_spacing:.2f}")
    # Phase 3: stash num_envs / env_spacing on the robot. The broadcast
    # Articulation view itself is built lazily in ensure_initialized
    # (after world.play()) so its constructor sees a live physics sim
    # view and eagerly inits without a race against STOP-event teardown.
    robot.set_multi_env(num_envs, env_spacing=env_spacing)
    robot.finalize_physics(sim)
    scene.attach_to_stage(sim)
    sim.add_extensions()

    ctx = SkillContext(
        world=sim, robot=robot, scene=scene,
        num_envs=num_envs, env_spacing=env_spacing,
    )
    runner = SkillRunner(build_task(cfg))

    def on_world_sync(step_index):
        scene.maybe_sync(step_index, robot.arm, robot.robot_prim_path)

    return ctx, runner, on_world_sync


def _run(cmd: Run) -> None:
    task_spec = _validate_compat(cmd.task.value, cmd.scene.value)
    robot_spec = ROBOTS[cmd.robot.value]
    scene_spec: SceneSpec = SCENES[cmd.scene.value]
    headless = cmd.headless_mode.value if cmd.headless_mode is not None else None

    # SimApp must be constructed before any curobo/isaacsim.core import.
    from bio_sim.sim import SimApp

    sim = SimApp(headless=headless)

    ctx, runner, on_world_sync = _build(
        sim, robot_spec, scene_spec, task_spec,
        cmd.robot_yml, cmd.use_urdf_kinematics, cmd.reactive,
    )

    sim.run(ctx, runner, on_world_sync=on_world_sync)


def _serve(cmd: Serve) -> None:
    """Headless boot with WebRTC livestream + FastAPI control plane."""
    task_spec = _validate_compat(cmd.task.value, cmd.scene.value)
    robot_spec = ROBOTS[cmd.robot.value]
    scene_spec: SceneSpec = SCENES[cmd.scene.value]

    from bio_sim.sim import SimApp

    # `headless="websocket"` is the existing tag for "no native window";
    # the actual livestream wiring is the SimApp(livestream=True) extra_args.
    sim = SimApp(headless="websocket", livestream=not cmd.no_livestream)

    ctx, runner, on_world_sync = _build(
        sim, robot_spec, scene_spec, task_spec,
        cmd.robot_yml, cmd.use_urdf_kinematics, cmd.reactive,
    )

    # Start the HTTP control plane AFTER scene/robot are built but BEFORE
    # sim.run, so the server is ready to enqueue the first reset the
    # instant the sim loop starts draining the queue.
    from bio_sim import server

    server.start_in_thread(port=cmd.port, host=cmd.host)

    sim.run(
        ctx, runner,
        on_world_sync=on_world_sync,
        on_pre_tick=server.drain_queue,
        keep_alive=True,
    )


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #


def main() -> None:
    # `tyro.cli(Union[...])` dispatches on the chosen subcommand name and
    # returns the populated dataclass instance.
    cmd = tyro.cli(Cmd, prog="bio_sim")
    if isinstance(cmd, List_):
        _render_tables(cmd.kind)
        return
    if isinstance(cmd, Run):
        _run(cmd)
        return
    if isinstance(cmd, Serve):
        _serve(cmd)
        return
    # Unreachable: tyro raises before this on an unknown subcommand.
    _console.print(f"[red]unhandled command: {cmd!r}[/red]")
    sys.exit(2)


if __name__ == "__main__":
    main()
