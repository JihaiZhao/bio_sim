#
# G2 (genie) dual-arm motion-gen reacher.
#
# This is curobo's examples/isaac_sim/motion_gen_reacher.py, adapted for the
# G2 omnipicker robot using the genie_sim cuRobo configs/assets that now live
# in this project:
#
#   robot config : config/curobo/configs/robot/G2_omnipicker_fixed_dual.yml
#                   (ee_link = gripper_r_center_link, right arm active)
#                  config/curobo/configs/robot/G2_omnipicker_fixed_left.yml
#                   (ee_link = gripper_l_center_link, left arm active)
#   urdf/usd     : curobo_robot/assets/robot/G2/...
#
# The G2 is a 30+ DOF mobile dual-arm robot. The config freezes torso/head and
# the passive gripper joints via `lock_joints`, leaving the two 7-DOF arms free,
# and declares BOTH gripper centers as target links:
#       link_names: ["gripper_r_center_link", "gripper_l_center_link"]
#
# Like genie_sim's reacher: the *active* arm tracks its red target cube while
# the *idle* arm's center link is pinned to its current world pose (via
# link_poses) so it holds station instead of drifting.
#

try:
    import isaacsim
except ImportError:
    pass

import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
parser.add_argument(
    "--robot",
    type=str,
    default="G2_omnipicker_fixed_dual.yml",
    help="G2 cuRobo config. Use G2_omnipicker_fixed_dual.yml (right-arm ee) "
    "or G2_omnipicker_fixed_left.yml (left-arm ee).",
)
parser.add_argument(
    "--active_arm",
    type=str,
    default=None,
    choices=["right", "left"],
    help="Which arm tracks its target by default when both cubes are static. "
    "Defaults to the ee_link side of the chosen config.",
)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parser.add_argument(
    "--external_asset_path",
    type=str,
    default=os.path.join(_PROJECT_ROOT, "curobo_robot", "assets"),
    help="Asset root; config urdf/usd paths (robot/G2/...) resolve under this.",
)
parser.add_argument(
    "--external_robot_configs_path",
    type=str,
    default=os.path.join(_PROJECT_ROOT, "config", "curobo", "configs", "robot"),
    help="Directory containing the G2_omnipicker_fixed_*.yml cuRobo configs.",
)
parser.add_argument(
    "--use_urdf_kinematics",
    action="store_true",
    default=False,
    help="Build cuRobo kinematics from the URDF instead of the USD "
    "(the configs ship use_usd_kinematics: True; URDF is more robust here).",
)
parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot collision spheres",
    default=False,
)
parser.add_argument(
    "--reactive",
    action="store_true",
    help="When True, runs in reactive mode",
    default=False,
)
args, _ = parser.parse_known_args()  # unknown args (e.g. --/rtx/...) pass through to SimulationApp

############################################################

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(  # RTX plugin initializes here; import torch AFTER
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)

from typing import Dict

import carb
import numpy as np
import torch

# curobo's isaac_sim example helper (add_extensions). We do NOT use its
# add_robot_to_scene: it only URDF-imports, and the G2 URDF's meshes point at an
# unavailable package://genie_robot_description ROS package, so the robot would
# load invisible. We load the robot from its USD (baked geometry) instead.
sys.path.append(
    os.path.join(_PROJECT_ROOT, "third_party", "curobo", "examples", "isaac_sim")
)
from helper import add_extensions
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.utils.types import ArticulationAction

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import log_error, setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)

############################################################


def _other_link(link_names, ee_link):
    """The link in link_names that is NOT the primary ee_link."""
    for ln in link_names:
        if ln != ee_link:
            return ln
    return None


def _add_robot_from_usd(robot_cfg, my_world, position=np.array([0.0, 0.0, 0.0])):
    """Reference the G2 robot USD (baked geometry) into the Isaac stage.

    Replaces curobo's URDF-only add_robot_to_scene. The USD's articulation
    root is its default prim (usd_robot_root, e.g. /genie) and its joint names
    match the USD-kinematics cspace joint names, so downstream dof lookups stay
    consistent.
    """
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.utils.stage import add_reference_to_stage

    kin = robot_cfg["kinematics"]
    asset_root = kin.get("external_asset_path")
    if asset_root is None:
        from curobo.util_file import get_assets_path

        asset_root = get_assets_path()
    usd_path = join_path(asset_root, kin["usd_path"])
    robot_prim_path = "/World/" + kin["usd_robot_root"].strip("/")

    add_reference_to_stage(usd_path=usd_path, prim_path=robot_prim_path)
    robot = my_world.scene.add(
        Robot(prim_path=robot_prim_path, name="robot", position=position)
    )
    # Mirror curobo helper's initialize_world=True (Isaac 4.5): without this the
    # articulation view stays None and articulation_controller.apply_action
    # crashes once a plan succeeds.
    my_world.initialize_physics()
    robot.initialize()
    return robot, robot_prim_path


def main():
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")

    setup_curobo_logger("warn")
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 100

    usd_help = UsdHelper()
    tensor_args = TensorDeviceType()

    # ---- load the G2 cuRobo robot config -------------------------------
    robot_cfg_path = get_robot_configs_path()
    if args.external_robot_configs_path is not None:
        robot_cfg_path = args.external_robot_configs_path
    robot_cfg = load_yaml(join_path(robot_cfg_path, args.robot))["robot_cfg"]

    if args.external_asset_path is not None:
        robot_cfg["kinematics"]["external_asset_path"] = args.external_asset_path
    if args.external_robot_configs_path is not None:
        robot_cfg["kinematics"]["external_robot_configs_path"] = (
            args.external_robot_configs_path
        )
    if args.use_urdf_kinematics:
        robot_cfg["kinematics"]["use_usd_kinematics"] = False

    ee_link = robot_cfg["kinematics"]["ee_link"]
    link_names = robot_cfg["kinematics"]["link_names"]
    idle_link = _other_link(link_names, ee_link)

    # ee_link side is the "active" arm for this config; allow override.
    default_active = "right" if "_r_" in ee_link else "left"
    active_arm = args.active_arm if args.active_arm is not None else default_active
    print(
        f"[G2] config={args.robot}  ee_link={ee_link}  idle_link={idle_link}  "
        f"default active arm={active_arm}"
    )

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    robot, robot_prim_path = _add_robot_from_usd(robot_cfg, my_world)
    articulation_controller = None

    # ---- world (same table obstacle as the curobo example) -------------
    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[2] -= 0.02
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.5
    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    trajopt_dt = None
    optimize_dt = True
    trajopt_tsteps = 32
    trim_steps = None
    max_attempts = 4
    interpolation_dt = 0.05
    enable_finetune_trajopt = True
    if args.reactive:
        trajopt_tsteps = 40
        trajopt_dt = 0.04
        optimize_dt = False
        max_attempts = 1
        trim_steps = [1, None]
        interpolation_dt = trajopt_dt
        enable_finetune_trajopt = False

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        interpolation_dt=interpolation_dt,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        optimize_dt=optimize_dt,
        trajopt_dt=trajopt_dt,
        trajopt_tsteps=trajopt_tsteps,
        trim_steps=trim_steps,
    )
    motion_gen = MotionGen(motion_gen_config)
    if not args.reactive:
        print("warming up... (G2 is large; this can take a minute)")
        motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
    print("CuRobo is Ready")

    add_extensions(simulation_app, args.headless_mode)

    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=2,
        max_attempts=max_attempts,
        enable_finetune_trajopt=enable_finetune_trajopt,
        time_dilation_factor=0.5 if not args.reactive else 1.0,
    )

    usd_help.load_stage(my_world.stage)
    usd_help.add_world_to_stage(world_cfg, base_frame="/World")

    # ---- two target cubes, one per gripper center ----------------------
    # Place each cube at the corresponding link's FK pose at the retract config.
    start_js = JointState.from_position(
        tensor_args.to_device(default_config).view(1, -1),
        joint_names=j_names,
    )
    kin = motion_gen.compute_kinematics(start_js)
    init_pose = {ee_link: kin.ee_pose, idle_link: kin.link_poses[idle_link]}

    targets = {}
    for ln, color in ((ee_link, [1.0, 0, 0]), (idle_link, [1.0, 0.4, 0])):
        p = init_pose[ln].position.cpu().numpy().ravel()
        q = init_pose[ln].quaternion.cpu().numpy().ravel()
        targets[ln] = cuboid.VisualCuboid(
            "/World/target_" + ("right" if "_r_" in ln else "left"),
            position=p,
            orientation=q,
            color=np.array(color),
            size=0.05,
        )

    cmd_plan = None
    cmd_idx = 0
    idx_list = None
    my_world.scene.add_default_ground_plane()
    i = 0
    spheres = None
    past_cmd = None
    past = {ln: None for ln in link_names}  # last seen cube pose per link
    last = {ln: None for ln in link_names}  # last solved-for cube pose per link

    while simulation_app.is_running():
        my_world.step(render=True)
        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            continue

        step_index = my_world.current_time_step_index
        if articulation_controller is None:
            articulation_controller = robot.get_articulation_controller()
        if step_index < 10:
            robot._articulation_view.initialize()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for _ in range(len(idx_list))]),
                joint_indices=idx_list,
            )
        if step_index < 20:
            continue

        if step_index == 50 or step_index % 1000 == 0.0:
            print("Updating world, reading w.r.t.", robot_prim_path)
            obstacles = usd_help.get_obstacles_from_stage(
                only_paths=["/World"],
                reference_prim_path=robot_prim_path,
                ignore_substring=[
                    robot_prim_path,
                    "/World/target_right",
                    "/World/target_left",
                    "/World/defaultGroundPlane",
                    "/curobo",
                ],
            ).get_collision_check_world()
            motion_gen.update_world(obstacles)
            print("Updated World")
            carb.log_info("Synced CuRobo world from stage.")

        # current cube poses
        cube_pose = {}
        for ln, t in targets.items():
            cp, co = t.get_world_pose()
            cube_pose[ln] = (cp, co)
            if past[ln] is None:
                past[ln] = cp
            if last[ln] is None:
                last[ln] = cp

        sim_js = robot.get_joints_state()
        if sim_js is None:
            continue
        sim_js_names = robot.dof_names
        if np.any(np.isnan(sim_js.positions)):
            log_error("isaac sim has returned NAN joint position values.")
        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities),
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )
        if not args.reactive:
            cu_js.velocity *= 0.0
            cu_js.acceleration *= 0.0
        if args.reactive and past_cmd is not None:
            cu_js.position[:] = past_cmd.position
            cu_js.velocity[:] = past_cmd.velocity
            cu_js.acceleration[:] = past_cmd.acceleration
        cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

        if args.visualize_spheres and step_index % 2 == 0:
            sph_list = motion_gen.kinematics.get_robot_as_spheres(cu_js.position)
            if spheres is None:
                spheres = []
                for si, s in enumerate(sph_list[0]):
                    spheres.append(
                        sphere.VisualSphere(
                            prim_path="/curobo/robot_sphere_" + str(si),
                            position=np.ravel(s.position),
                            radius=float(s.radius),
                            color=np.array([0, 0.8, 0.2]),
                        )
                    )
            else:
                for si, s in enumerate(sph_list[0]):
                    if not np.isnan(s.position[0]):
                        spheres[si].set_world_pose(position=np.ravel(s.position))
                        spheres[si].set_radius(float(s.radius))

        robot_static = (np.max(np.abs(sim_js.velocities)) < 0.5) or args.reactive

        # Which cube moved? That arm becomes active this solve; the other arm
        # is pinned to its current world pose so it holds station.
        moved = None
        for ln in link_names:
            cp, co = cube_pose[ln]
            if (
                np.linalg.norm(cp - last[ln]) > 1e-3
                and np.linalg.norm(past[ln] - cp) == 0.0
            ):
                moved = ln
                break
        if moved is None and robot_static:
            # nothing moved: keep tracking the configured active arm so the
            # demo still drives one arm to its cube.
            active_ln = ee_link if active_arm == default_active else idle_link
            cp, _ = cube_pose[active_ln]
            if np.linalg.norm(cp - last[active_ln]) > 1e-3:
                moved = active_ln

        if moved is not None and robot_static:
            # current FK, used to pin the idle arm
            cur_kin = motion_gen.compute_kinematics(cu_js.unsqueeze(0))
            cur_pose = {
                ee_link: cur_kin.ee_pose,
                idle_link: cur_kin.link_poses[idle_link],
            }

            goal_for = {}
            for ln in link_names:
                if ln == moved:
                    cp, co = cube_pose[ln]
                    goal_for[ln] = Pose(
                        position=tensor_args.to_device(cp),
                        quaternion=tensor_args.to_device(co),
                    )
                else:
                    goal_for[ln] = cur_pose[ln].clone()

            # plan_single: goal_pose is for ee_link; link_poses holds the rest.
            goal_pose = goal_for[ee_link]
            link_poses = {idle_link: goal_for[idle_link]}

            result = motion_gen.plan_single(
                cu_js.unsqueeze(0), goal_pose, plan_config, link_poses=link_poses
            )
            succ = result.success.item()
            if succ:
                cmd_plan = result.get_interpolated_plan()
                cmd_plan = motion_gen.get_full_js(cmd_plan)
                common_js_names = []
                idx_list = []
                for x in sim_js_names:
                    if x in cmd_plan.joint_names:
                        idx_list.append(robot.get_dof_index(x))
                        common_js_names.append(x)
                cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)
                cmd_idx = 0
            else:
                carb.log_warn(
                    "Plan did not converge to a solution: " + str(result.status)
                )
            for ln in link_names:
                last[ln] = cube_pose[ln][0]

        for ln in link_names:
            past[ln] = cube_pose[ln][0]

        if cmd_plan is not None:
            cmd_state = cmd_plan[cmd_idx]
            past_cmd = cmd_state.clone()
            art_action = ArticulationAction(
                cmd_state.position.cpu().numpy(),
                cmd_state.velocity.cpu().numpy(),
                joint_indices=idx_list,
            )
            articulation_controller.apply_action(art_action)
            cmd_idx += 1
            for _ in range(2):
                my_world.step(render=False)
            if cmd_idx >= len(cmd_plan.position):
                cmd_idx = 0
                cmd_plan = None
                past_cmd = None
    simulation_app.close()


if __name__ == "__main__":
    main()
