"""Isaac Sim + cuRobo smoke test for Agibot G1 (no scene objects).

Spawns the G1 robot on a ground plane, plans a single right-arm pose target
via cuRobo, and steps the simulation through the resulting trajectory.

Prerequisites:
  - bio_sim.robot.agibot_g1_cfg.AGIBOT_G1_USD_PATH must point at a valid USD.
    Run scripts/download_assets.py (and any USD fetch the user provides) first.

Headless run:
  python scripts/g1_smoke.py --headless

GUI run:
  python scripts/g1_smoke.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Order matters. Isaac Sim 5.1 bundles omni.warp.core-1.8.2 which shadows
# the pip warp-lang 1.13 at AppLauncher init time UNLESS warp is already
# imported. Importing cuRobo first pins warp 1.13. We then add the
# `wp.types.array` shim that Isaac Sim's internal warp utilities expect
# (renamed to `wp.array` upstream). Without this shim AppLauncher crashes
# inside isaacsim.core.utils.warp.rotations.
from bio_sim.motion import planner as p  # noqa: E402

import warp as _wp  # noqa: E402
if not hasattr(_wp.types, "array"):
    _wp.types.array = _wp.array

from isaaclab.app import AppLauncher  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spawn G1 + cuRobo arm motion smoke test")
    parser.add_argument("--steps-per-waypoint", type=int, default=1,
                        help="Sim sub-steps per planner waypoint (1 = sync with planner dt).")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    launcher = AppLauncher(args)
    sim_app = launcher.app

    # IsaacLab modules require AppLauncher to be up (they import carb).
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.assets import AssetBaseCfg

    from bio_sim.robot.agibot_g1_cfg import (
        AGIBOT_G1_CFG,
        AGIBOT_G1_USD_PATH,
        RIGHT_EE_LINK,
    )

    if not AGIBOT_G1_USD_PATH.exists():
        print(
            f"\nG1 USD not found at {AGIBOT_G1_USD_PATH}\n"
            f"Drop the GenieSimAssets-converted USD there and re-run.\n"
            f"(Until then, scripts/curobo_g1_plan.py is the headless cuRobo-only smoke.)\n",
            file=sys.stderr,
        )
        sim_app.close()
        return 1

    print("[1/5] building scene")
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    scene_cfg.dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )
    scene_cfg.robot = AGIBOT_G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    robot = scene["robot"]

    print(f"[2/5] robot loaded: {len(robot.data.joint_names)} joints")

    print("[3/5] building cuRobo planner (right-arm-only)")
    handle = p.build(
        REPO_ROOT / "src/bio_sim/robot/agibot_g1_curobo.yml",
        tool_frames=[RIGHT_EE_LINK],
        warmup_iterations=2,
    )

    fk = handle.planner.compute_kinematics(handle.default_joint_state)
    home_pos = fk.tool_poses.position.squeeze().detach().cpu().numpy()
    home_quat = fk.tool_poses.quaternion.squeeze().detach().cpu().numpy()
    target_pos = (0.35, -0.30, 0.85)
    target_quat = tuple(float(x) for x in home_quat)
    print(f"      home: pos={home_pos.tolist()}")
    print(f"    target: pos={target_pos}")

    print("[4/5] planning")
    result = p.plan_arm_pose(handle, target_pos, target_quat)
    if result is None or not result.success.any().item():
        print("plan FAILED", file=sys.stderr)
        sim_app.close()
        return 1
    positions, planner_joint_names, dt = p.trajectory_to_numpy(result, handle.planner)
    print(f"      trajectory: {positions.shape[0]} waypoints @ dt={dt:.3f}s")

    # Map planner joints onto the sim's articulation joint ordering.
    sim_joint_names = list(robot.data.joint_names)
    planner_to_sim = [sim_joint_names.index(j) for j in planner_joint_names]
    full_target = robot.data.default_joint_pos.clone()

    print("[5/5] executing trajectory")
    for step, q in enumerate(positions):
        full_target[0, planner_to_sim] = torch.from_numpy(q).to(full_target.device).to(full_target.dtype)
        robot.set_joint_position_target(full_target)
        for _ in range(args.steps_per_waypoint):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
        if step % 10 == 0:
            print(f"  step {step}/{positions.shape[0]}")

    print("done. Holding scene for inspection — close the window to exit.")
    try:
        while sim_app.is_running():
            sim.step()
            scene.update(sim.get_physics_dt())
    finally:
        sim_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
