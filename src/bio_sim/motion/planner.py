"""cuRobo V2 motion-planner wrapper for bio_sim.

Vendored Agibot G1 / hand-authored Dexmate Vega cuRobo YAMLs reference relative
``urdf_path`` / ``asset_root_path`` (genie_sim convention: resolved against
cuRobo's content/assets root). bio_sim keeps URDFs in ``src/bio_sim/robot/``,
so this loader substitutes absolute paths before handing the dict to
``MotionPlannerCfg.create``.

Wrapper shape is informed by genie_sim's ``CuroboMotion`` class
(third_party/genie_sim/source/geniesim/app/utils/motion_gen_reacher.py)
but uses cuRobo V2 APIs directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

# --- Isaac Sim 5.1 ↔ pip warp 1.13 compat shim --------------------------------
# Isaac Sim 5.1 ships omni.warp.core-1.8.2 in its extscache and many of its
# exts/replicator code does `from warp.X import Y` against the 1.8 public
# namespace. cuRobo V2 needs pip warp 1.13's `wp.func(module=)`. Importing
# pip warp first pins 1.13 in sys.modules; then we expose warp 1.13's private
# `_src` submodules back at the old public paths so Isaac Sim's imports
# resolve. Anything 1.13 dropped entirely (e.g. `warp.sim`, `warp.constants`)
# is handled by attaching empty placeholder modules — losing only those
# specific features (Isaac Sim 5.1 still loads core robot code).
import sys as _sys
import types as _types

import warp as _wp  # noqa: E402

def _expose_src(name: str) -> None:
    src_name = f"warp._src.{name}"
    pub_name = f"warp.{name}"
    try:
        src_mod = __import__(src_name, fromlist=["*"])
    except ImportError:
        return
    pub_mod = _sys.modules.get(pub_name)
    if pub_mod is None:
        _sys.modules[pub_name] = src_mod
        setattr(_wp, name, src_mod)
    else:
        for attr in dir(src_mod):
            if not attr.startswith("_") and not hasattr(pub_mod, attr):
                try:
                    setattr(pub_mod, attr, getattr(src_mod, attr))
                except (AttributeError, TypeError):
                    pass


for _name in ("utils", "types", "context", "codegen", "tape", "fabric", "builtins"):
    _expose_src(_name)

for _missing_module in ("sim", "constants"):
    _full = f"warp.{_missing_module}"
    if _full not in _sys.modules:
        _placeholder = _types.ModuleType(_full)
        _sys.modules[_full] = _placeholder
        setattr(_wp, _missing_module, _placeholder)
# ------------------------------------------------------------------------------

from curobo._src.robot.loader.kinematics_loader_cfg import KinematicsLoaderCfg  # noqa: E402
from curobo._src.robot.types.cspace_params import CSpaceParams  # noqa: E402
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg  # noqa: E402
from curobo.types import GoalToolPose, JointState, Pose  # noqa: E402

# V1 cuRobo YAMLs (e.g. those vendored from genie_sim) include USD-kinematics
# and a few other fields that V2 removed. Strip them silently.
_V2_KINEMATICS_FIELDS = set(KinematicsLoaderCfg.__dataclass_fields__.keys())
_V2_CSPACE_FIELDS = set(CSpaceParams.__dataclass_fields__.keys())
# V1 cspace field → V2 cspace field
_CSPACE_RENAMES = {"retract_config": "default_joint_position"}

REPO_ROOT = Path(__file__).resolve().parents[3]
ROBOT_DIR = REPO_ROOT / "src" / "bio_sim" / "robot"


@dataclass(frozen=True)
class PlannerHandle:
    planner: MotionPlanner
    joint_names: list[str]
    tool_frames: list[str]
    default_joint_state: JointState


def load_cfg(
    yaml_path: Path | str,
    urdf_override: Path | str | None = None,
    tool_frames: list[str] | None = None,
) -> dict[str, Any]:
    """Load a cuRobo robot YAML and inject absolute paths.

    The on-disk YAML keeps genie_sim-style relative paths so it travels well
    across forks; this function resolves them against ``ROBOT_DIR`` (or an
    explicit ``urdf_override``) at load time.
    """
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    kin = cfg["robot_cfg"]["kinematics"]
    if urdf_override is not None:
        urdf_abs = Path(urdf_override).resolve()
    else:
        # YAML's urdf_path is resolved relative to the YAML's own directory.
        urdf_abs = (yaml_path.parent / Path(kin["urdf_path"]).name).resolve()
        if not urdf_abs.exists():
            # Fall back: look up the URDF in ROBOT_DIR by basename.
            urdf_abs = ROBOT_DIR / Path(kin["urdf_path"]).name

    if not urdf_abs.exists():
        raise FileNotFoundError(
            f"URDF not found for cuRobo: {urdf_abs}. "
            f"Original urdf_path in YAML was '{kin['urdf_path']}'."
        )

    kin["urdf_path"] = str(urdf_abs)
    kin["asset_root_path"] = str(urdf_abs.parent)
    if tool_frames is not None:
        kin["tool_frames"] = list(tool_frames)

    if isinstance(kin.get("collision_spheres"), str):
        spheres_rel = Path(kin["collision_spheres"])
        spheres_abs = (yaml_path.parent / spheres_rel).resolve()
        if spheres_abs.exists():
            kin["collision_spheres"] = str(spheres_abs)

    if isinstance(kin.get("cspace"), dict):
        cspace = kin["cspace"]
        for old, new in _CSPACE_RENAMES.items():
            if old in cspace and new not in cspace:
                cspace[new] = cspace.pop(old)
        dropped_cspace = sorted(set(cspace) - _V2_CSPACE_FIELDS)
        for key in dropped_cspace:
            cspace.pop(key)
        if dropped_cspace:
            print(f"[planner] dropped V1-only cspace fields: {dropped_cspace}")

    dropped = sorted(set(kin) - _V2_KINEMATICS_FIELDS)
    for key in dropped:
        kin.pop(key)
    if dropped:
        print(f"[planner] dropped V1-only kinematics fields: {dropped}")

    return cfg


def build(
    yaml_path: Path | str,
    *,
    urdf_override: Path | str | None = None,
    tool_frames: list[str] | None = None,
    scene_model: Path | str | dict[str, Any] | None = None,
    num_ik_seeds: int = 32,
    num_trajopt_seeds: int = 4,
    use_cuda_graph: bool = True,
    warmup_iterations: int = 5,
    enable_graph_warmup: bool = True,
    position_tolerance: float = 0.01,
    orientation_tolerance: float = 0.1,
) -> PlannerHandle:
    """Build a warmed-up ``MotionPlanner`` from a bio_sim robot YAML.

    Pass ``tool_frames`` to override the YAML's IK targets (e.g.
    ``["gripper_r_center_link"]`` for single-arm planning).

    ``scene_model`` may be a path to a cuRobo scene YAML, or a dict with
    ``cuboid`` / ``mesh`` / etc. keys (see cuRobo's ``collision_test.yml``).
    """
    cfg = load_cfg(yaml_path, urdf_override=urdf_override, tool_frames=tool_frames)
    create_kwargs: dict[str, Any] = dict(
        robot=cfg["robot_cfg"],
        num_ik_seeds=num_ik_seeds,
        num_trajopt_seeds=num_trajopt_seeds,
        use_cuda_graph=use_cuda_graph,
        position_tolerance=position_tolerance,
        orientation_tolerance=orientation_tolerance,
    )
    if scene_model is not None:
        create_kwargs["scene_model"] = (
            str(scene_model) if isinstance(scene_model, Path) else scene_model
        )
    planner_cfg = MotionPlannerCfg.create(**create_kwargs)
    planner = MotionPlanner(planner_cfg)
    planner.warmup(
        enable_graph=enable_graph_warmup, num_warmup_iterations=warmup_iterations
    )
    default = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0),
        joint_names=planner.joint_names,
    )
    return PlannerHandle(
        planner=planner,
        joint_names=list(planner.joint_names),
        tool_frames=list(planner.tool_frames),
        default_joint_state=default,
    )


def make_goal(
    tool_frames: list[str],
    position: tuple[float, float, float] | list[tuple[float, float, float]],
    quaternion: tuple[float, float, float, float] | list[tuple[float, float, float, float]] = (1.0, 0.0, 0.0, 0.0),
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> GoalToolPose:
    """Build a ``GoalToolPose`` for one or more tool frames.

    ``position``/``quaternion`` may be a single tuple (applied to the primary
    frame only, others left unconstrained — cuRobo will treat secondary frames
    via the YAML retract config) or a list with one entry per tool frame.
    """
    if isinstance(position, (tuple, list)) and len(position) == 3 and not isinstance(position[0], (tuple, list)):
        positions = [position]
        quaternions = [quaternion]
    else:
        positions = list(position)
        quaternions = list(quaternion) if isinstance(quaternion, list) else [quaternion] * len(positions)

    pos_t = torch.tensor([[[[p] for p in positions]]], device=device, dtype=dtype)
    quat_t = torch.tensor([[[[q] for q in quaternions]]], device=device, dtype=dtype)
    return GoalToolPose(
        tool_frames=tool_frames[: len(positions)],
        position=pos_t,
        quaternion=quat_t,
    )


def plan_arm_pose(
    handle: PlannerHandle,
    target_position: tuple[float, float, float],
    target_quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    q_start: JointState | None = None,
):
    """Plan a single-frame pose target for the planner's primary tool frame.

    Returns the cuRobo ``MotionPlannerResult``.
    """
    goal = make_goal(handle.tool_frames, target_position, target_quaternion)
    start = q_start if q_start is not None else handle.default_joint_state
    return handle.planner.plan_pose(goal, start)


def trajectory_to_numpy(result, planner: MotionPlanner):
    """Extract (positions, joint_names, dt) from a plan result.

    Returns positions of shape ``(n_waypoints, n_joints)``. ``joint_names`` is
    the interpolated plan's *own* joint-name list — it includes locked
    joints too, so length may exceed ``planner.joint_names`` (which is the
    set of unlocked / planning-active joints).
    """
    interp = result.get_interpolated_plan()
    pos = interp.position.detach().cpu().numpy()
    # cuRobo's interpolated plan tensors are typically (batch, n_tools,
    # n_waypoints, n_joints); strip leading dims to get a 2D trajectory.
    while pos.ndim > 2:
        pos = pos[0]
    dt = planner.trajopt_solver.config.interpolation_dt
    joint_names = list(interp.joint_names) if hasattr(interp, "joint_names") else list(planner.joint_names)
    return pos, joint_names, dt
