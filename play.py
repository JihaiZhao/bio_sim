#!/usr/bin/env python
#
# Entrypoint: wire scene + robot + task and run the pick-and-place.
#
#   python play.py                       # windowed
#   python play.py --headless_mode native
#
# ORDERING: SimApp() boots SimulationApp (RTX/Kit plugins, isaacsim.core).
# Everything that imports curobo/torch/isaacsim.core is imported AFTER it.
#

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--headless_mode", type=str, default=None,
                   help="native | websocket (omit for a window)")
    p.add_argument("--robot", type=str, default="G2_omnipicker_fixed_dual.yml",
                   help="G2 cuRobo config (…_dual.yml = right-arm ee, "
                        "…_left.yml = left-arm ee)")
    p.add_argument("--r1", action="store_true", default=False,
                   help="use the R1 Pro (DYNAMIC BEHAVIOR-1K holonomic "
                        "base) instead of G2 -- the dynamic base carries a "
                        "friction-held object without flinging it")
    p.add_argument("--use_urdf_kinematics", action="store_true", default=False)
    p.add_argument("--reactive", action="store_true", default=False)
    args, _ = p.parse_known_args()  # let --/rtx/... pass to SimulationApp
    return args


def main():
    args = parse_args()

    # 1. boot the sim runtime (must precede curobo/isaacsim.core imports)
    from bio_sim.sim import SimApp
    sim = SimApp(headless=args.headless_mode)

    # 2. now safe to import the heavy layers
    from bio_sim.runner import SkillRunner
    from bio_sim.scene import BioScene
    from bio_sim.robot import G2Robot, R1ProRobot
    from bio_sim.skills import SkillContext
    from bio_sim.tasks import build_pick_place, load_full_cfg

    # Shared task config + per-robot overlay (cube/grasp/init pose).
    # Adding a new robot = drop a new file under config/robots/<name>.yaml.
    robot_key = "r1pro" if args.r1 else "g2"
    cfg = load_full_cfg(robot_key)
    print(f"[play] cfg = task_pick_place.yaml + robots/{robot_key}.yaml")

    # 3. build the world (declarative: fixtures come from the cfg recipe,
    #    resolved through the shared asset library / SIM_ASSETS)
    scene = BioScene.from_cfg(cfg)
    scene.build(sim)

    if args.r1:
        # --robot only overrides when an R1Pro_* yml is passed; else the
        # default G2 yml is swapped for the R1 Pro config.
        yml = (args.robot if args.robot.startswith("R1Pro")
               else "R1Pro_arm_no_torso.yml")
        robot = R1ProRobot(
            robot_yml=yml,
            use_urdf_kinematics=args.use_urdf_kinematics,
            reactive=args.reactive,
        )
    else:
        robot = G2Robot(
            robot_yml=args.robot,
            use_urdf_kinematics=args.use_urdf_kinematics,
            reactive=args.reactive,
        )
    # Non-invasive per-task init pose (genie_sim-style): overlay the task
    # cfg's init_arm_pose onto retract_config IN MEMORY *before* load_into
    # rebuilds the cuRobo planner, so the committed robot yml is untouched.
    robot.apply_init_pose(cfg)
    # Grasp mechanism (physics friction vs. assist FixedJoint weld). Both
    # robot facades expose `.gripper` (a shared Gripper); set it from cfg.
    robot.gripper.set_mode(cfg.get("grasp_mode", "physics"))
    robot.load_into(sim, scene)
    # derive a deterministic, IK-reachable layout from the robot workspace
    scene.place_for_validation(robot, cfg)
    scene.attach_to_stage(sim)
    sim.add_extensions()

    # 4. build the task + run
    ctx = SkillContext(world=sim, robot=robot, scene=scene)
    runner = SkillRunner(build_pick_place(cfg))

    def on_world_sync(step_index):
        scene.maybe_sync(step_index, robot.arm, robot.robot_prim_path)

    sim.run(ctx, runner, on_world_sync=on_world_sync)


if __name__ == "__main__":
    main()
