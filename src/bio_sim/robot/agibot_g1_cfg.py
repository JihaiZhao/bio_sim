"""Agibot G1 (omnipicker) ArticulationCfg for Isaac Lab.

Vendored from genie_sim (https://github.com/AgibotTech/genie_sim). The cuRobo
config at ``agibot_g1_curobo.yml`` is pre-authored and known-good — see
genie_sim's ``source/geniesim/app/utils/motion_gen_reacher.py`` for the
reference wrapper around it.

The URDF abstracts the wheeled chassis as a fixed base from ``base_link`` —
base motion is NOT planned by cuRobo. In a bio_sim pipeline that uses G1, the
base must be commanded separately (e.g. teleport-style SE(2) update, mirroring
genie_sim's ``APICore._update_robot_base``).

End-effector frames for IK: ``gripper_l_center_link``, ``gripper_r_center_link``.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_REPO_ROOT = Path(__file__).resolve().parents[3]
AGIBOT_G1_USD_PATH = (
    _REPO_ROOT / "src" / "bio_sim" / "assets" / "robot" / "G1_omnipicker" / "robot.usd"
)


AGIBOT_G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(AGIBOT_G1_USD_PATH),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Self-collisions enforced by cuRobo (spheres + SRDF disable pairs);
            # PhysX-level disabled to avoid spurious base-to-torso contacts.
            enabled_self_collisions=False,
            # Pin the chassis. Mimics genie_sim's setup where the robot lives
            # in a scene USD with a baked-in floor at the right z; we don't
            # load that scene yet, so without fixing the root the robot would
            # fall through any flat ground plane spawned at z=0 (the USD's
            # base_link extends to z=-0.21).
            fix_root_link=True,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "idx01_body_joint1": 0.3,
            "idx02_body_joint2": 0.5236,
            "idx11_head_joint1": 0.0,
            "idx12_head_joint2": 0.0,
            "idx21_arm_l_joint1": 2.07,
            "idx22_arm_l_joint2": -0.61,
            "idx23_arm_l_joint3": -1.57,
            "idx24_arm_l_joint4": 1.0,
            "idx25_arm_l_joint5": -1.57,
            "idx26_arm_l_joint6": -1.57,
            "idx27_arm_l_joint7": 1.57,
            "idx61_arm_r_joint1": -2.07,
            "idx62_arm_r_joint2": 0.61,
            "idx63_arm_r_joint3": 1.57,
            "idx64_arm_r_joint4": -1.0,
            "idx65_arm_r_joint5": 1.57,
            "idx66_arm_r_joint6": 1.57,
            "idx67_arm_r_joint7": -1.57,
            # NOTE: the 4 mimic-loop joints idx{39,49,79,89} are emitted by
            # IsaacLab's UrdfConverter as "..._joint0" rather than the URDF's
            # "..._joint2" suffix. Match the USD names; cuRobo continues to
            # reference the URDF-side names internally.
            "idx31_gripper_l_inner_joint1": 0.0,
            "idx32_gripper_l_inner_joint3": 0.0,
            "idx33_gripper_l_inner_joint4": 0.349,
            "idx39_gripper_l_inner_joint0": 0.0,
            "idx41_gripper_l_outer_joint1": 0.0,
            "idx42_gripper_l_outer_joint3": 0.01,
            "idx43_gripper_l_outer_joint4": -0.35,
            "idx49_gripper_l_outer_joint0": 0.0,
            "idx71_gripper_r_inner_joint1": 0.0,
            "idx72_gripper_r_inner_joint3": 0.0,
            "idx73_gripper_r_inner_joint4": 0.349,
            "idx79_gripper_r_inner_joint0": 0.0,
            "idx81_gripper_r_outer_joint1": 0.0,
            "idx82_gripper_r_outer_joint3": 0.01,
            "idx83_gripper_r_outer_joint4": -0.35,
            "idx89_gripper_r_outer_joint0": 0.0,
        },
    ),
    actuators={
        "waist_lift": ImplicitActuatorCfg(
            joint_names_expr=["idx01_body_joint1"],
            effort_limit=100.0,
            velocity_limit=0.1,
            stiffness=10000.0,
            damping=500.0,
        ),
        "torso_yaw": ImplicitActuatorCfg(
            joint_names_expr=["idx02_body_joint2"],
            effort_limit=100.0,
            velocity_limit=0.5,
            stiffness=1000.0,
            damping=100.0,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["idx1[12]_head_joint[12]"],
            effort_limit=50.0,
            velocity_limit=1.0,
            stiffness=100.0,
            damping=10.0,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["idx2[1-7]_arm_l_joint[1-7]"],
            effort_limit={
                "idx21_arm_l_joint1": 60.0, "idx22_arm_l_joint2": 60.0,
                "idx23_arm_l_joint3": 60.0, "idx24_arm_l_joint4": 60.0,
                "idx25_arm_l_joint5": 30.0, "idx26_arm_l_joint6": 30.0,
                "idx27_arm_l_joint7": 30.0,
            },
            velocity_limit=3.14,
            stiffness=400.0,
            damping=40.0,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["idx6[1-7]_arm_r_joint[1-7]"],
            effort_limit={
                "idx61_arm_r_joint1": 60.0, "idx62_arm_r_joint2": 60.0,
                "idx63_arm_r_joint3": 60.0, "idx64_arm_r_joint4": 60.0,
                "idx65_arm_r_joint5": 30.0, "idx66_arm_r_joint6": 30.0,
                "idx67_arm_r_joint7": 30.0,
            },
            velocity_limit=3.14,
            stiffness=400.0,
            damping=40.0,
        ),
        # 4-bar linkage parallel jaws: primary drive joints are idx41/idx81
        # (outer_joint1); the rest follow via mimic constraints in the URDF.
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
    },
    soft_joint_pos_limit_factor=0.95,
)


WAIST_LIFT_JOINT = "idx01_body_joint1"
TORSO_YAW_JOINT = "idx02_body_joint2"
HEAD_JOINTS = ["idx11_head_joint1", "idx12_head_joint2"]
LEFT_ARM_JOINTS = [f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)]
LEFT_GRIPPER_DRIVE_JOINT = "idx41_gripper_l_outer_joint1"
RIGHT_GRIPPER_DRIVE_JOINT = "idx81_gripper_r_outer_joint1"

LEFT_EE_LINK = "gripper_l_center_link"
RIGHT_EE_LINK = "gripper_r_center_link"
