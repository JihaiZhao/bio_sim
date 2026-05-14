"""Isaac Sim + cuRobo tabletop pick demo for Agibot G1.

Scene: ground + table (in front of G1) + a small cube on the table.
Plan: right-arm pre-grasp -> grasp -> close gripper -> lift.
Collision world: the table is registered as a cuboid obstacle so cuRobo
plans around it (the cube is intentionally NOT in the collision world during
the grasp segment).

Prerequisites: same as g1_smoke.py — needs the G1 USD at
``bio_sim.robot.agibot_g1_cfg.AGIBOT_G1_USD_PATH``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# planner.py imports pin pip warp 1.13 and install the compat shims for
# Isaac Sim 5.1's old warp namespace paths.
from bio_sim.motion import planner as p  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402


# Table / cube placement (all in robot base frame, robot at origin facing +X).
TABLE_DIMS = (0.8, 1.0, 0.05)
TABLE_TOP_Z = 0.70
TABLE_CENTER = (0.60, 0.0, TABLE_TOP_Z - TABLE_DIMS[2] / 2)

CUBE_SIZE = 0.05
CUBE_XY = (0.45, -0.25)
CUBE_CENTER = (*CUBE_XY, TABLE_TOP_Z + CUBE_SIZE / 2)

PREGRASP_OFFSET = 0.15
LIFT_OFFSET = 0.20
# Gripper pointing straight down: 180-deg flip around X axis (wxyz).
GRIPPER_DOWN_QUAT = (0.0, 1.0, 0.0, 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G1 tabletop pick demo")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _scene_model() -> dict:
    """cuRobo collision-world description (cuboid table only)."""
    tx, ty, tz = TABLE_CENTER
    return {
        "cuboid": {
            "table": {
                "dims": list(TABLE_DIMS),
                "pose": [tx, ty, tz, 1.0, 0.0, 0.0, 0.0],
            }
        }
    }


def main() -> int:
    args = parse_args()
    launcher = AppLauncher(args)
    sim_app = launcher.app

    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

    from bio_sim.robot.agibot_g1_cfg import (
        AGIBOT_G1_CFG,
        AGIBOT_G1_USD_PATH,
        RIGHT_EE_LINK,
        LEFT_GRIPPER_DRIVE_JOINT,
        RIGHT_GRIPPER_DRIVE_JOINT,
    )

    if not AGIBOT_G1_USD_PATH.exists():
        print(
            f"\nG1 USD not found at {AGIBOT_G1_USD_PATH}\n"
            f"Drop the GenieSimAssets-converted USD there and re-run.\n",
            file=sys.stderr,
        )
        sim_app.close()
        return 1

    print("[1/6] building scene")
    # No GroundPlane — robot is pinned via fix_root_link, table is kinematic.
    # Matches genie_sim's pattern (floor would come from a scene USD).
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )
    scene_cfg.robot = AGIBOT_G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene_cfg.table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_DIMS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.45, 0.3)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_CENTER),
    )
    scene_cfg.cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE,) * 3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.15, 0.15)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_CENTER),
    )

    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    robot = scene["robot"]

    print(f"[2/6] robot loaded: {len(robot.data.joint_names)} joints")

    print("[3/6] building cuRobo planner (right-arm-only, table obstacle)")
    handle = p.build(
        REPO_ROOT / "src/bio_sim/robot/agibot_g1_curobo.yml",
        tool_frames=[RIGHT_EE_LINK],
        scene_model=_scene_model(),
        warmup_iterations=2,
    )

    sim_joint_names = list(robot.data.joint_names)
    planner_to_sim = [sim_joint_names.index(j) for j in handle.joint_names]
    full_target = robot.data.default_joint_pos.clone()

    def execute(positions, steps_per_waypoint: int = 1, label: str = "") -> None:
        print(f"   executing {label} ({positions.shape[0]} waypoints)")
        for step, q in enumerate(positions):
            full_target[0, planner_to_sim] = torch.from_numpy(q).to(
                full_target.device
            ).to(full_target.dtype)
            robot.set_joint_position_target(full_target)
            for _ in range(steps_per_waypoint):
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim.get_physics_dt())

    def plan_to(pos, label: str):
        result = p.plan_arm_pose(
            handle,
            target_position=pos,
            target_quaternion=GRIPPER_DOWN_QUAT,
            q_start=p.JointState.from_position(
                robot.data.joint_pos[0, planner_to_sim].unsqueeze(0),
                joint_names=handle.joint_names,
            ),
        )
        if result is None or not result.success.any().item():
            raise RuntimeError(f"plan failed at stage '{label}'")
        positions, _, _ = p.trajectory_to_numpy(result, handle.planner)
        execute(positions, label=label)

    cx, cy, cz = CUBE_CENTER
    pregrasp = (cx, cy, cz + PREGRASP_OFFSET)
    grasp = (cx, cy, cz)
    lift = (cx, cy, cz + LIFT_OFFSET)

    try:
        print("[4/6] plan pregrasp -> grasp -> lift")
        plan_to(pregrasp, "pregrasp")
        plan_to(grasp, "descend")

        # Close right gripper. The omnipicker is a 4-bar linkage — driving
        # idx81_gripper_r_outer_joint1 toward its upper limit (0.785 rad)
        # closes the jaws; the mimic chain follows.
        print("[5/6] closing right gripper")
        grip_idx = sim_joint_names.index(RIGHT_GRIPPER_DRIVE_JOINT)
        for ratio in (0.25, 0.5, 0.75, 0.95):
            full_target[0, grip_idx] = ratio * 0.7854
            robot.set_joint_position_target(full_target)
            for _ in range(20):
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim.get_physics_dt())

        print("[6/6] lift")
        plan_to(lift, "lift")

        print("done. Holding scene — close the window to exit.")
        while sim_app.is_running():
            sim.step()
            scene.update(sim.get_physics_dt())
    finally:
        sim_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
