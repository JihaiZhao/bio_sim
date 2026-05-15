"""Agibot G2 ArticulationCfg for **physics-driven** wheel motion (method B).

Companion to :mod:`agibot_g2_cfg`, which is the SE(2) teleport variant.
Differences from that cfg:

* Points at ``assets/robot/G2_omnipicker_drive/robot.usd`` — produced by
  ``scripts/patch_g2_drive_usd.py``, which copies the working
  ``G2_omnipicker`` USD tree and adds the two things the source asset
  lacks for driving: sphere colliders on the 4 wheel link2 prims, and
  wide-open limits (±1e6) on the 4 rolling joints.
* ``fix_root_link=False`` and **gravity ON** — the wheels physically
  support the chassis via friction.
* Wheel actuators are split: steering joints are position-controlled,
  rolling joints are velocity-controlled (zero stiffness, high damping
  acts as the velocity gain).
* Spawn z is set so wheel centres land at ground + wheel-radius — see
  the geometry comment on ``init_state``.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_REPO_ROOT = Path(__file__).resolve().parents[3]
AGIBOT_G2_DRIVE_USD_PATH = (
    _REPO_ROOT
    / "src" / "bio_sim" / "assets" / "robot" / "G2_omnipicker_drive" / "robot.usd"
)


# Wheel geometry — from the source URDF (config/robot_cfg/G2/
# G2_omnipicker_fixed_dual.urdf). Used by swerve kinematics; keep
# WHEEL_RADIUS in sync with scripts/patch_g2_drive_usd.py.
WHEEL_RADIUS = 0.07          # cylinder radius in URDF
WHEEL_HUB_HEIGHT = 0.16675   # steering joint origin z above chassis_link
WHEEL_CENTER_OFFSET = -0.13675  # rolling joint z relative to steering link
# Wheel CENTER body-z = WHEEL_HUB_HEIGHT + WHEEL_CENTER_OFFSET = 0.03
# Wheel BOTTOM body-z = 0.03 - 0.07 = -0.04
# So for wheel-bottom to touch z=0 floor → base_link world z = +0.04.

WHEEL_POSITIONS_BODY = {
    # name → (x, y) in chassis frame (z is implicit at WHEEL_HUB_HEIGHT)
    "fl": (+0.23, +0.218),
    "fr": (+0.23, -0.218),
    "rl": (-0.23, +0.218),
    "rr": (-0.23, -0.218),
}

# URDF joint names per wheel. jointN1 is steering (axis Z), jointN2 is
# rolling (axis Y in the post-steering frame, now type=continuous).
WHEEL_STEER_JOINTS = {
    "fl": "idx111_chassis_lwheel_front_joint1",
    "fr": "idx131_chassis_rwheel_front_joint1",
    "rl": "idx121_chassis_lwheel_rear_joint1",
    "rr": "idx141_chassis_rwheel_rear_joint1",
}
WHEEL_ROLL_JOINTS = {
    "fl": "idx112_chassis_lwheel_front_joint2",
    "fr": "idx132_chassis_rwheel_front_joint2",
    "rl": "idx122_chassis_lwheel_rear_joint2",
    "rr": "idx142_chassis_rwheel_rear_joint2",
}


AGIBOT_G2_DRIVE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(AGIBOT_G2_DRIVE_USD_PATH),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # Gravity ON: wheels support the chassis via ground friction.
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            max_linear_velocity=5.0,
            max_angular_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # base_link world z = wheel_radius + |wheel_center_offset_in_body|
        #                   = 0.07 + 0.04 = 0.11
        # Slight headroom so PhysX can settle without interpenetration.
        pos=(0.0, 0.0, 0.12),
        joint_pos={
            "idx01_body_joint1": 0.0,
            "idx02_body_joint2": 0.0,
            "idx03_body_joint3": 0.261,
            "idx04_body_joint4": 0.0,
            "idx05_body_joint5": 0.0,
            "idx11_head_joint1": 0.0,
            "idx12_head_joint2": 0.0,
            "idx13_head_joint3": 0.174,
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
            # Wheels start straight (steering=0) and stationary.
            "idx111_chassis_lwheel_front_joint1": 0.0,
            "idx131_chassis_rwheel_front_joint1": 0.0,
            "idx121_chassis_lwheel_rear_joint1": 0.0,
            "idx141_chassis_rwheel_rear_joint1": 0.0,
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
        # Steering: position-controlled. Wide effort to react quickly.
        "wheel_steer": ImplicitActuatorCfg(
            joint_names_expr=["idx1[1-4]1_chassis_.wheel_.*_joint1"],
            effort_limit=15.0,
            velocity_limit=4.08,
            stiffness=300.0,
            damping=30.0,
        ),
        # Rolling: velocity-controlled. Stiffness=0 + damping acts as a PI on
        # joint velocity (joint_vel_target tracked by damping). 8 Nm matches
        # the URDF effort limit.
        "wheel_roll": ImplicitActuatorCfg(
            joint_names_expr=["idx1[1-4]2_chassis_.wheel_.*_joint2"],
            effort_limit=8.0,
            velocity_limit=20.9,
            stiffness=0.0,
            damping=2.0,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)


# Convenience joint groupings.
BODY_JOINTS = [f"idx0{i}_body_joint{i}" for i in range(1, 6)]
HEAD_JOINTS = [f"idx1{i}_head_joint{i}" for i in range(1, 4)]
LEFT_ARM_JOINTS = [f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)]
LEFT_EE_LINK = "gripper_l_center_link"
RIGHT_EE_LINK = "gripper_r_center_link"
