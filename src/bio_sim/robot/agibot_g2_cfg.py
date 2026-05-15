"""Agibot G2 (omnipicker, dual-arm) ArticulationCfg for Isaac Lab.

Mirrors :mod:`agibot_g1_cfg` but for G2. Two key differences vs G1:

* G2 has 5 revolute body joints (G1 has prismatic + revolute) and 3 head
  joints (G1 has 2).
* G2's URDF includes 8 chassis wheel joints (4 steer + 4 drive). For the
  SE(2) "teleport" base controller (see :mod:`bio_sim.motion.mobile_base`)
  we leave the wheels passive (low-stiffness damping only) and move the
  whole articulation by writing the root pose each step. To allow that,
  ``fix_root_link=False`` here — otherwise PhysX would weld the base to
  the world and ``write_root_pose_to_sim`` would have no effect.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_REPO_ROOT = Path(__file__).resolve().parents[3]
AGIBOT_G2_USD_PATH = (
    _REPO_ROOT / "src" / "bio_sim" / "assets" / "robot" / "G2_omnipicker" / "robot.usd"
)


AGIBOT_G2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(AGIBOT_G2_USD_PATH),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # Gravity OFF: with fix_root_link=False the base would otherwise
            # accelerate downward between teleport writes and (because wheel
            # collision is commented out in the URDF) sink past the ground
            # until the chassis box collision catches — leaving the visual
            # mesh half-buried. The MobileBase controller pins the root
            # pose each step, so we don't need physics to support the robot.
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            # Mobile base: free root so write_root_pose_to_sim takes effect.
            # The MobileBase controller resets root velocity to zero each
            # step, so the chassis behaves kinematically (no gravity drift)
            # without needing a physical fixed joint.
            fix_root_link=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # The G2 base_link visual mesh has its origin near the top of the
        # chassis (mounting plane for the body column), so the rounded
        # chassis hangs ~15cm BELOW base_link. Spawn at z=0.18 so the
        # visual sits cleanly on the ground (gravity is disabled, so the
        # robot stays where placed). Bump higher if your floor isn't at z=0.
        pos=(0.0, 0.0, 0.18),
        joint_pos={
            # Body (5-DOF torso column) — defaults from genie_sim's
            # G2_omnipicker/config.yaml.
            "idx01_body_joint1": 0.0,
            "idx02_body_joint2": 0.0,
            "idx03_body_joint3": 0.0,
            "idx04_body_joint4": 0.0,
            "idx05_body_joint5": 0.0,
            # Head.
            "idx11_head_joint1": 0.0,
            "idx12_head_joint2": 0.3,
            "idx13_head_joint3": 0.174,
            # Arms — initial pose from genie_sim defaults.
            "idx21_arm_l_joint1": -1.57,
            "idx22_arm_l_joint2": 1.57,
            "idx23_arm_l_joint3": 1.57,
            "idx24_arm_l_joint4": -1.57,
            "idx25_arm_l_joint5": -1.57,
            "idx26_arm_l_joint6": 0.0,
            "idx27_arm_l_joint7": 0.0,
            "idx61_arm_r_joint1": 1.57,
            "idx62_arm_r_joint2": 1.57,
            "idx63_arm_r_joint3": -1.57,
            "idx64_arm_r_joint4": -1.57,
            "idx65_arm_r_joint5": -1.57,
            "idx66_arm_r_joint6": 0.0,
            "idx67_arm_r_joint7": 0.0,
        },
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=["idx0[1-5]_body_joint[1-5]"],
            effort_limit=200.0,
            velocity_limit=1.0,
            stiffness=2000.0,
            damping=200.0,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["idx1[1-3]_head_joint[1-3]"],
            effort_limit=50.0,
            velocity_limit=1.0,
            stiffness=100.0,
            damping=10.0,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["idx2[1-7]_arm_l_joint[1-7]"],
            effort_limit=60.0,
            velocity_limit=3.14,
            stiffness=400.0,
            damping=40.0,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["idx6[1-7]_arm_r_joint[1-7]"],
            effort_limit=60.0,
            velocity_limit=3.14,
            stiffness=400.0,
            damping=40.0,
        ),
        "left_gripper": ImplicitActuatorCfg(
            joint_names_expr=["idx(31|32|33|39|41|42|43|49)_gripper_l_.*"],
            effort_limit=10.0,
            velocity_limit=10.0,
            stiffness=200.0,
            damping=10.0,
        ),
        "right_gripper": ImplicitActuatorCfg(
            joint_names_expr=["idx(71|72|73|79|81|82|83|89)_gripper_r_.*"],
            effort_limit=10.0,
            velocity_limit=10.0,
            stiffness=200.0,
            damping=10.0,
        ),
        # Passive wheels for the SE(2) teleport controller — zero stiffness
        # so they don't resist when the chassis is moved by setting the root
        # pose directly. Light damping bleeds off any spurious spin.
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["idx1[1-4][12]_chassis_.wheel_.*"],
            effort_limit=0.0,
            velocity_limit=10.0,
            stiffness=0.0,
            damping=0.5,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)


# Joint name groupings, mirrored from agibot_g1_cfg for downstream use.
BODY_JOINTS = [f"idx0{i}_body_joint{i}" for i in range(1, 6)]
HEAD_JOINTS = [f"idx1{i}_head_joint{i}" for i in range(1, 4)]
LEFT_ARM_JOINTS = [f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)]
LEFT_GRIPPER_DRIVE_JOINT = "idx41_gripper_l_outer_joint1"
RIGHT_GRIPPER_DRIVE_JOINT = "idx81_gripper_r_outer_joint1"
LEFT_EE_LINK = "gripper_l_center_link"
RIGHT_EE_LINK = "gripper_r_center_link"
