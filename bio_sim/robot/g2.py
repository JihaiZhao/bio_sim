#
# G2Robot: the facade skills talk to. Hides the Isaac articulation bridge
# (DOF index map, joint-state read, ArticulationAction streaming) and owns the
# arm planner, kinematic swerve base, and gripper. Skills never touch
# _articulation_view; they call robot.arm / robot.base / robot.gripper and
# work in WORLD coordinates (the facade handles the base-frame transform).
#
# Robot/USD/config wiring is ported from the validated reacher.
#

from __future__ import annotations

import math
import os

import numpy as np

from curobo.types.state import JointState
from curobo.util.logger import log_error
from curobo.util_file import get_robot_configs_path, join_path, load_yaml

from .arm import ArmPlanner
from .base import KeyboardTeleop, NavController, SwerveBaseController
from .gripper import Gripper

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_ASSETS = os.path.join(_PROJECT_ROOT, "curobo_robot", "assets")
_DEFAULT_CFG_DIR = os.path.join(_PROJECT_ROOT, "config", "curobo", "configs", "robot")


def _other_link(link_names, ee_link):
    for ln in link_names:
        if ln != ee_link:
            return ln
    return None


class G2Robot:
    def __init__(self, robot_yml: str = "G2_omnipicker_fixed_dual.yml",
                 use_urdf_kinematics: bool = False, reactive: bool = False):
        cfg_dir = _DEFAULT_CFG_DIR if os.path.isdir(_DEFAULT_CFG_DIR) else get_robot_configs_path()
        self.robot_cfg = load_yaml(join_path(cfg_dir, robot_yml))["robot_cfg"]
        kin = self.robot_cfg["kinematics"]
        kin["external_asset_path"] = _DEFAULT_ASSETS
        kin["external_robot_configs_path"] = cfg_dir
        if use_urdf_kinematics:
            kin["use_usd_kinematics"] = False

        self.ee_link = kin["ee_link"]
        self.link_names = kin["link_names"]
        self.idle_link = _other_link(self.link_names, self.ee_link)
        self.j_names = kin["cspace"]["joint_names"]
        self.retract_config = kin["cspace"]["retract_config"]
        self._reactive = reactive

        self.arm = ArmPlanner(self.robot_cfg, None, self.ee_link, self.idle_link,
                               reactive=reactive)
        self.base: NavController | None = None
        self.gripper = Gripper(self)

        self._robot = None
        self.robot_prim_path = None
        self._art_ctrl = None
        self._view = None
        self._arm_idx = None
        self._initialized = False

        # active arm trajectory stream
        self._cmd_plan = None
        self._cmd_idx = 0
        self._cmd_idx_list = None

    # ---- world build (called by play before sim.run) ------------------
    def load_into(self, sim, scene):
        from isaacsim.core.api.robots import Robot
        from isaacsim.core.utils.stage import add_reference_to_stage

        kin = self.robot_cfg["kinematics"]
        asset_root = kin.get("external_asset_path")
        usd_path = join_path(asset_root, kin["usd_path"])
        self.robot_prim_path = "/World/" + kin["usd_robot_root"].strip("/")
        add_reference_to_stage(usd_path=usd_path, prim_path=self.robot_prim_path)
        self._robot = sim.world.scene.add(
            Robot(prim_path=self.robot_prim_path, name="robot",
                  position=np.array([0.0, 0.0, 0.0]))
        )
        sim.world.initialize_physics()
        self._robot.initialize()
        # rebuild the arm planner now that we can give it the scene's world cfg
        self.arm = ArmPlanner(self.robot_cfg, scene.curobo_world,
                              self.ee_link, self.idle_link, reactive=self._reactive)
        self.arm.warmup()

    # ---- per-step init (settle window) --------------------------------
    @property
    def base_ready(self) -> bool:
        return self.base is not None

    def ensure_initialized(self, ctx) -> None:
        if self._initialized:
            return
        if self._art_ctrl is None:
            self._art_ctrl = self._robot.get_articulation_controller()
        self._robot._articulation_view.initialize()
        self._arm_idx = [self._robot.get_dof_index(x) for x in self.j_names]
        self._robot.set_joint_positions(self.retract_config, self._arm_idx)
        self._robot._articulation_view.set_max_efforts(
            values=np.array([5000 for _ in self._arm_idx]),
            joint_indices=self._arm_idx,
        )
        swerve = SwerveBaseController(self._robot, self._robot._articulation_view)
        swerve.configure_drive_modes()
        self.base = NavController(swerve)
        self._initialized = True

    def base_hold(self, ctx) -> None:
        sim_js = self._robot.get_joints_state()
        self.base.step(ctx.world, sim_js)
        self.gripper.hold_step(ctx)

    # ---- joint state --------------------------------------------------
    def read_cu_js(self):
        sim_js = self._robot.get_joints_state()
        if sim_js is None:
            return None
        if np.any(np.isnan(sim_js.positions)):
            log_error("isaac sim returned NAN joint positions")
        cu_js = JointState(
            position=self.arm.tensor_args.to_device(sim_js.positions),
            velocity=self.arm.tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=self.arm.tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=self.arm.tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=self._robot.dof_names,
        )
        return cu_js.get_ordered_joint_state(self.arm.joint_names)

    def robot_static(self) -> bool:
        sim_js = self._robot.get_joints_state()
        if sim_js is None:
            return False
        return np.max(np.abs(sim_js.velocities)) < 0.5

    # ---- arm trajectory streaming -------------------------------------
    def plan_arm_to(self, p_world, q_world) -> bool:
        cu_js = self.read_cu_js()
        if cu_js is None:
            return False
        plan = self.arm.plan_to_world_pose(cu_js, p_world, q_world, self.base)
        if plan is None:
            return False
        common = [n for n in self._robot.dof_names if n in plan.joint_names]
        self._cmd_idx_list = [self._robot.get_dof_index(n) for n in common]
        self._cmd_plan = plan.get_ordered_joint_state(common)
        self._cmd_idx = 0
        return True

    def advance_arm_plan(self, sim) -> bool:
        """Stream one waypoint. Returns True when the plan is exhausted."""
        from isaacsim.core.utils.types import ArticulationAction

        if self._cmd_plan is None:
            return True
        st = self._cmd_plan[self._cmd_idx]
        self._art_ctrl.apply_action(ArticulationAction(
            st.position.cpu().numpy(),
            st.velocity.cpu().numpy(),
            joint_indices=self._cmd_idx_list,
        ))
        self._cmd_idx += 1
        for _ in range(2):
            sim.step(render=False)
        if self._cmd_idx >= len(self._cmd_plan.position):
            self._cmd_plan = None
            self._cmd_idx = 0
            return True
        return False

    # ---- frames -------------------------------------------------------
    def base_to_world(self, p_base, q_base):
        x, y, z, yaw = self.base.base_pose()
        c, s = math.cos(yaw), math.sin(yaw)
        px, py, pz = float(p_base[0]), float(p_base[1]), float(p_base[2])
        p_world = np.array([c * px - s * py + x, s * px + c * py + y, pz + z],
                           dtype=np.float64)
        h = yaw / 2.0
        bq = np.array([math.cos(h), 0.0, 0.0, math.sin(h)])
        qw, qx, qy, qz = (float(q_base[0]), float(q_base[1]),
                          float(q_base[2]), float(q_base[3]))
        rw = bq[0] * qw - bq[3] * qz
        rx = bq[0] * qx - bq[3] * qy
        ry = bq[0] * qy + bq[3] * qx
        rz = bq[0] * qz + bq[3] * qw
        q_world = np.array([rw, rx, ry, rz], dtype=np.float64)
        q_world /= np.linalg.norm(q_world) + 1e-12
        return p_world, q_world

    def ee_world_pose(self, ctx):
        """Current gripper-center pose in WORLD (FK in base frame -> world)."""
        cu_js = self.read_cu_js()
        pose = self.arm.fk_link_pose(cu_js.unsqueeze(0), self.ee_link)
        p_b = pose.position.cpu().numpy().ravel()
        q_b = pose.quaternion.cpu().numpy().ravel()
        return self.base_to_world(p_b, q_b)


__all__ = ["G2Robot", "KeyboardTeleop"]
