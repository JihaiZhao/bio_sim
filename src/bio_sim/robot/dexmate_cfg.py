"""Dexmate Vega-1 (gripper variant) ArticulationCfg for Isaac Lab.

Sim model has a real wheeled base (two driven steerable wheels + passive castor).
cuRobo plans against a separate URDF (``dexmate_planning.urdf``) where the base
is modeled as ``[x_prismatic, y_prismatic, yaw_revolute]``. Joint names from
``torso_j1`` onward match between the two descriptions.

End-effector frames for IK: ``L_ee`` (left), ``R_ee`` (right).
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_REPO_ROOT = Path(__file__).resolve().parents[3]
VEGA_USD_PATH = _REPO_ROOT / "third_party" / "dexmate_urdf" / "usd" / "vega_1_gripper.usd"


DEXMATE_VEGA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(VEGA_USD_PATH),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Self-collision is enforced by cuRobo (spheres + SRDF disable pairs).
            # Disable in PhysX to avoid spurious base-to-torso contact penalties.
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "B_wheel_j1": 0.0, "B_wheel_j2": 0.0,
            "R_wheel_j1": 0.0, "R_wheel_j2": 0.0,
            "L_wheel_j1": 0.0, "L_wheel_j2": 0.0,
            "torso_j1": 0.0, "torso_j2": 0.5, "torso_j3": 0.0,
            "head_j1": 0.0, "head_j2": 0.0, "head_j3": 0.0,
            "L_arm_j1": 0.0, "L_arm_j2": 0.5, "L_arm_j3": 0.0,
            "L_arm_j4": -1.5, "L_arm_j5": 0.0, "L_arm_j6": 0.5, "L_arm_j7": 0.0,
            "R_arm_j1": 0.0, "R_arm_j2": -0.5, "R_arm_j3": 0.0,
            "R_arm_j4": -1.5, "R_arm_j5": 0.0, "R_arm_j6": -0.5, "R_arm_j7": 0.0,
            "L_gripper_j1": 0.0, "L_gripper_j2": 0.0,
            "R_gripper_j1": 0.0, "R_gripper_j2": 0.0,
        },
    ),
    actuators={
        # Wheels: velocity-controlled by motion/base_controller.py.
        "base_wheels": ImplicitActuatorCfg(
            joint_names_expr=["[BLR]_wheel_j[12]"],
            effort_limit=16.0,
            velocity_limit=12.0,
            stiffness=0.0,
            damping=100.0,
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["torso_j[1-3]"],
            effort_limit={"torso_j1": 700.0, "torso_j2": 380.0, "torso_j3": 380.0},
            velocity_limit=0.9,
            stiffness=2000.0,
            damping=200.0,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_j[1-3]"],
            effort_limit=6.0,
            velocity_limit=3.2,
            stiffness=100.0,
            damping=10.0,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["L_arm_j[1-7]"],
            effort_limit={
                "L_arm_j1": 150.0, "L_arm_j2": 150.0,
                "L_arm_j3": 80.0,  "L_arm_j4": 80.0,
                "L_arm_j5": 25.0,  "L_arm_j6": 25.0, "L_arm_j7": 25.0,
            },
            velocity_limit=2.7,
            stiffness=400.0,
            damping=40.0,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["R_arm_j[1-7]"],
            effort_limit={
                "R_arm_j1": 150.0, "R_arm_j2": 150.0,
                "R_arm_j3": 80.0,  "R_arm_j4": 80.0,
                "R_arm_j5": 25.0,  "R_arm_j6": 25.0, "R_arm_j7": 25.0,
            },
            velocity_limit=2.7,
            stiffness=400.0,
            damping=40.0,
        ),
        "left_gripper": ImplicitActuatorCfg(
            joint_names_expr=["L_gripper_j[12]"],
            effort_limit=20.0,
            velocity_limit=0.5,
            stiffness=200.0,
            damping=10.0,
        ),
        "right_gripper": ImplicitActuatorCfg(
            joint_names_expr=["R_gripper_j[12]"],
            effort_limit=20.0,
            velocity_limit=0.5,
            stiffness=200.0,
            damping=10.0,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)


LEFT_ARM_JOINTS = [f"L_arm_j{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"R_arm_j{i}" for i in range(1, 8)]
TORSO_JOINTS = [f"torso_j{i}" for i in range(1, 4)]
HEAD_JOINTS = [f"head_j{i}" for i in range(1, 4)]
LEFT_GRIPPER_JOINTS = ["L_gripper_j1", "L_gripper_j2"]
RIGHT_GRIPPER_JOINTS = ["R_gripper_j1", "R_gripper_j2"]
BASE_WHEEL_JOINTS = ["B_wheel_j1", "B_wheel_j2", "R_wheel_j1", "R_wheel_j2", "L_wheel_j1", "L_wheel_j2"]

LEFT_EE_LINK = "L_ee"
RIGHT_EE_LINK = "R_ee"
