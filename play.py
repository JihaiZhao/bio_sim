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
    from bio_sim.robot import G2Robot
    from bio_sim.skills import SkillContext
    from bio_sim.tasks import build_pick_place, load_cfg

    cfg = load_cfg()

    # 3. build the world
    scene = BioScene()
    scene.build(sim)

    robot = G2Robot(
        robot_yml=args.robot,
        use_urdf_kinematics=args.use_urdf_kinematics,
        reactive=args.reactive,
    )
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
