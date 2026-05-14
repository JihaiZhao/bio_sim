"""Bio-lab pick demo: G1 picks a 500 mL beaker off the Thorlabs bench.

This is M2's first end-to-end demo using real lab assets instead of cuboid
placeholders. The scene is composed from ``bio_sim.scene.bio_lab``; the
planner is cuRobo with the bench registered as a cuboid obstacle.

Pipeline: pregrasp (hover) → descend → close right gripper → lift.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# planner.py installs the warp 1.13 ↔ Isaac Sim 5.1 compat shim on import,
# pins pip warp 1.13 in sys.modules, and brings in cuRobo — all of which must
# happen before AppLauncher.
from bio_sim.motion import planner as p  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G1 picks a beaker off a Thorlabs bench")
    parser.add_argument(
        "--no-obstacles",
        action="store_true",
        help="Skip registering the bench in cuRobo's collision world (debug only). "
             "Useful when planning hangs — isolates whether the issue is "
             "reachability or start-state-in-collision.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    launcher = AppLauncher(args)
    sim_app = launcher.app

    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.assets import AssetBaseCfg

    from bio_sim.robot.agibot_g1_cfg import (
        AGIBOT_G1_CFG,
        AGIBOT_G1_USD_PATH,
        RIGHT_EE_LINK,
        RIGHT_GRIPPER_DRIVE_JOINT,
    )
    from bio_sim.scene import bio_lab

    for usd in (AGIBOT_G1_USD_PATH, bio_lab.BENCH_USD, bio_lab.BEAKER_USD):
        if not usd.exists():
            print(f"missing asset: {usd}", file=sys.stderr)
            sim_app.close()
            return 1

    print("[1/6] building scene (G1 + Thorlabs bench + 500 mL beaker)")
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )
    scene_cfg.robot = AGIBOT_G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene_cfg.bench = bio_lab.make_bench_cfg()
    scene_cfg.beaker = bio_lab.make_beaker_cfg()

    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    robot = scene["robot"]
    print(f"[2/6] robot loaded: {len(robot.data.joint_names)} joints")

    scene_model = None if args.no_obstacles else bio_lab.cuRobo_obstacles()
    label = "no obstacles" if args.no_obstacles else "bench as obstacle"
    print(f"[3/6] building cuRobo planner (right-arm-only, {label})")
    handle = p.build(
        REPO_ROOT / "src/bio_sim/robot/agibot_g1_curobo.yml",
        tool_frames=[RIGHT_EE_LINK],
        scene_model=scene_model,
        warmup_iterations=2,
        # Looser tolerances so the planner doesn't grind forever on a target
        # near the reach limit.
        position_tolerance=0.02,
        orientation_tolerance=0.20,
    )

    bx, by, bz = bio_lab.BEAKER_POS
    pregrasp = (bx, by, bz + bio_lab.GRASP_APPROACH_HEIGHT)
    grasp = (bx, by, bz + bio_lab.BEAKER_HEIGHT / 2 + 0.01)
    lift = (bx, by, bz + bio_lab.GRASP_LIFT_HEIGHT)
    # Gripper pointing straight down (180° flip around X, wxyz).
    GRIPPER_DOWN = (0.0, 1.0, 0.0, 0.0)
    print(f"      beaker at {bio_lab.BEAKER_POS}, bench top z={bio_lab.BENCH_TOP_CENTER[2]}")

    sim_joint_names = list(robot.data.joint_names)
    full_target = robot.data.default_joint_pos.clone()

    def execute(positions, traj_joint_names, label: str) -> None:
        traj_to_sim = [sim_joint_names.index(j) for j in traj_joint_names]
        print(f"      {label}: {positions.shape[0]} waypoints")
        for q in positions:
            full_target[0, traj_to_sim] = torch.from_numpy(q).to(
                full_target.device
            ).to(full_target.dtype)
            robot.set_joint_position_target(full_target)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())

    def plan_to(pos, label: str) -> None:
        import time as _t
        t0 = _t.perf_counter()
        print(f"      [{label}] planning to {tuple(round(v,3) for v in pos)} ...", flush=True)
        q_start = p.JointState.from_position(
            robot.data.joint_pos[0, [sim_joint_names.index(j) for j in handle.joint_names]].unsqueeze(0),
            joint_names=handle.joint_names,
        )
        result = p.plan_arm_pose(handle, pos, GRIPPER_DOWN, q_start=q_start)
        dt = _t.perf_counter() - t0
        if result is None or not result.success.any().item():
            raise RuntimeError(
                f"plan failed at stage '{label}' targeting {pos} (after {dt:.2f}s) — "
                f"likely unreachable. Try moving the bench closer or the beaker forward."
            )
        positions, traj_names, _ = p.trajectory_to_numpy(result, handle.planner)
        print(f"      [{label}] success: {positions.shape[0]} waypoints in {dt:.2f}s", flush=True)
        execute(positions, traj_names, label)

    try:
        print("[4/6] plan pregrasp → descend")
        plan_to(pregrasp, "pregrasp")
        plan_to(grasp, "descend")

        print("[5/6] close right gripper")
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
