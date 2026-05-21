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
    p.add_argument("--robot", type=str, default=None,
                   help="cuRobo planner yml; omit to use the registry's "
                        "default for the chosen robot. Pass an explicit "
                        "yml (e.g. G2_omnipicker_fixed_dual_left.yml) to "
                        "override.")
    p.add_argument("--r1", action="store_true", default=False,
                   help="pick the R1 Pro spec from the registry (DYNAMIC "
                        "BEHAVIOR-1K holonomic base) instead of G2 -- the "
                        "dynamic base carries a friction-held object "
                        "without flinging it")
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
    from bio_sim.scene import OtOneScene
    from bio_sim.robot.registry import resolve
    from bio_sim.skills import SkillContext
    from bio_sim.tasks import build_pick_place, load_full_cfg

    # Registry resolves "robot name" -> (class, default cuRobo yml,
    # cfg overlay filename). Adding a new robot = new RobotSpec in
    # bio_sim/robot/registry.py + a config overlay under
    # bio_sim/config/robots/<name>.yaml; play.py and the rest of the
    # pipeline stay untouched.
    spec = resolve("r1pro" if args.r1 else "g2")
    cfg = load_full_cfg(spec.cfg_overlay)
    print(f"[play] robot={spec.name}; cfg = task_pick_place.yaml + "
          f"robots/{spec.cfg_overlay}.yaml")

    # 3. build the world (declarative: fixtures come from the cfg recipe,
    #    resolved through the shared asset library / SIM_ASSETS)
    # OtOneScene = BioScene + an OT-One mounted on table A; the plate
    # spawns ON the OT-One deck (between the 4 frame columns) instead of
    # directly on the table top. Layout math is the same as BioScene's so
    # the existing task_pick_place.yaml + robots/<robot>.yaml work
    # unchanged (grasp z == release z is preserved by lowering table A
    # by the OT-One deck offset). See bio_sim/scene/ot_one_scene.py.
    scene = OtOneScene.from_cfg(cfg)
    scene.build(sim)

    yml = args.robot if args.robot is not None else spec.default_curobo_yml
    robot = spec.cls(
        robot_yml=yml,
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
