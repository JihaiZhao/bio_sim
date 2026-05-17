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

# Arm PD gains (USD ships ~1e4 stiffness -> too soft; stiffen for tracking).
ARM_KP = 1.0e5
ARM_KD = 1.0e4
# Gripper force-control (genie_sim ParallelGripper, idx81 command joint).
GRIP_OPEN_Q = 0.8           # position-mode open target
GRIP_KP = 1.0e5             # position-mode stiffness (open hold)
GRIP_KD = 1.0e3             # damping (both modes; velocity tracking on close)
# Closing VELOCITY ejects (fast sweeping fingers punch a free object away);
# force CAP holds (clamp strength once in contact). So: slow close + firm
# cap. -0.6 didn't eject; 25 was too weak to lift -> keep -0.6, raise cap.
# Chunky box: fingers stop at a firm mid-range clamp (not over-closed on a
# thin object). Moderate force -- enough to hold, gentle enough not to tip.
GRIP_MAX_FORCE = 70.0
GRIP_CLOSE_VEL = -0.6       # slow approach (doesn't punch the object)
# Post-contact vice-grip hold (clamp_hold): squeeze a touch past the
# contacted aperture and hold it stiffly with a strong effort cap so the
# fingers can't ratchet shut when a carry disturbance shifts the object.
GRIP_HOLD_SQUEEZE = 0.03    # rad further closed than the contact pose
GRIP_HOLD_FORCE = 300.0     # effort cap for the held clamp (>> close cap)


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
        # physical link a grasped object is fixed-jointed to (a real rigid
        # articulation link with a collider, not the ee frame). cuRobo's
        # `attached_object` link is parented here too.
        self.grasp_link = (
            "gripper_r_base_link" if "_r_" in self.ee_link
            else "gripper_l_base_link"
        )
        self._grasp_link_path = None
        # Underactuated omnipicker: exactly ONE driven command joint per
        # hand (outer_joint1). robot.usda DELETES the DriveAPI on
        # inner_joint1 and gives it a PhysxMimicJointAPI:rotX slaved to
        # outer_joint1 (joint3/4 are passive linkage). So we drive ONLY
        # outer_joint1 and let PhysX propagate the inner finger via the
        # mimic constraint -- directly commanding inner_joint1 fights that
        # constraint (a tensor-API drive vs the PhysX mimic on one DOF)
        # and is non-physical. (Earlier two-finger drive was a mistake;
        # genie_sim uses this same single-drive + mimic topology.)
        # 0.0 = closed, GRIP_OPEN_Q = open.
        self.grip_cmd_joints = (
            ["idx81_gripper_r_outer_joint1"] if "_r_" in self.ee_link
            else ["idx41_gripper_l_outer_joint1"]
        )
        self.grip_cmd_joint = self.grip_cmd_joints[0]  # the single command joint
        self._grip_idxs = None
        self._grip_close = False
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
        self._nongrip_idx = None  # dof indices excluding gripper (robot_static)
        self._initialized = False

        # active arm trajectory stream
        self._cmd_plan = None
        self._cmd_idx = 0
        self._cmd_idx_list = None

        # per-phase arm execution: kinematic (accurate centering) vs PD
        # (smooth carry of a friction-held object). Default kinematic so the
        # arm is firmly held (e.g. retract during NavigateTo(A)).
        self._arm_kinematic = True
        self._arm_hold_pos = None   # last kinematic config to re-assert
        self._arm_hold_idx = None

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
        # hold retract kinematically until the first reach
        self._arm_hold_pos = np.asarray(self.retract_config, dtype=np.float32)
        self._arm_hold_idx = list(self._arm_idx)
        self._robot._articulation_view.set_max_efforts(
            values=np.array([5000 for _ in self._arm_idx]),
            joint_indices=self._arm_idx,
        )
        # Arm is PD-driven (apply_action) so a friction-held object follows
        # the hand smoothly (a hard-teleported hand would fling it). The USD
        # arm drive is soft (stiffness 1e4) -> stiffen it so PD tracking is
        # tight enough for the fingers to close on the cube.
        n = len(self._arm_idx)
        self._robot._articulation_view.set_gains(
            kps=np.full((1, n), ARM_KP, dtype=np.float32),
            kds=np.full((1, n), ARM_KD, dtype=np.float32),
            joint_indices=np.asarray(self._arm_idx, dtype=np.int32),
        )
        swerve = SwerveBaseController(self._robot, self._robot._articulation_view)
        swerve.configure_drive_modes()
        self.base = NavController(swerve)
        # Gripper: force-controlled friction grasp. Configure the command
        # dof on the ARTICULATION VIEW (set_gains / switch_dof_control_mode /
        # set_max_efforts) -- the same proven path the swerve base uses.
        # Setting USD DriveAPI attrs after articulation init does NOT take
        # effect (Isaac caches drive gains), which is why the fingers never
        # actuated and the cube was never picked up.
        self._view = self._robot._articulation_view
        self._grip_idxs = [self._robot.get_dof_index(n)
                           for n in self.grip_cmd_joints]
        self._grip_mode = None
        self._grip_close = False
        self._grip_state = "open"
        self._grip_hold_pos = None
        self.set_gripper(close=False)
        self._initialized = True

    def set_gripper(self, close: bool) -> None:
        """close = velocity/force mode (kp=0, capped effort, closing vel) to
        drive the fingers onto the object. open = position mode -> open."""
        self._grip_state = "close" if close else "open"
        self._grip_close = close          # back-compat for base_hold etc.
        self._grip_hold_pos = None
        self._sync_grip_mode("velocity" if close else "position")
        self._apply_gripper()

    def clamp_hold(self) -> None:
        """Lock the fingers as a stiff vice at the CURRENTLY-contacted
        aperture (a small squeeze past it), with a high effort cap. Unlike a
        continuous closing-velocity command, a position-hold cannot ratchet
        shut when a disturbance shifts the object -- the box keeps a stable
        pocket and friction a stable normal force. Still pure friction (no
        joint between object and gripper); just a real vice-grip clamp."""
        sjs = self._robot.get_joints_state()
        if sjs is None:
            return
        pos = np.array([float(sjs.positions[j]) for j in self._grip_idxs])
        # The command joint (idx81) CLOSES by decreasing position
        # (+0.8 open -> ~0 closed). Squeeze = a small decrement in the
        # closing direction; the mimic-driven inner finger follows.
        self._grip_hold_pos = pos - GRIP_HOLD_SQUEEZE
        self._grip_state = "hold"
        self._grip_close = True           # base_hold keeps re-applying it
        self._sync_grip_mode("position")
        self._apply_gripper()

    def _sync_grip_mode(self, mode: str) -> None:
        n = len(self._grip_idxs)
        idx = np.asarray(self._grip_idxs, dtype=np.int32)
        if mode != self._grip_mode:
            kp = 0.0 if mode == "velocity" else GRIP_KP
            self._view.set_gains(
                kps=np.full((1, n), kp, dtype=np.float32),
                kds=np.full((1, n), GRIP_KD, dtype=np.float32),
                joint_indices=idx,
            )
            for j in self._grip_idxs:
                self._view.switch_dof_control_mode(mode, j)
            self._grip_mode = mode

    def _apply_gripper(self) -> None:
        if self._grip_idxs is None:
            return
        n = len(self._grip_idxs)
        idx = np.asarray(self._grip_idxs, dtype=np.int32)
        state = getattr(self, "_grip_state", "open")
        if state == "close":
            # Effort cap scales with aperture (genie_sim: max_force +
            # 2*|pos|): wider open -> clamp harder. Use the widest finger.
            cur = 0.0
            sjs = self._robot.get_joints_state()
            if sjs is not None:
                cur = max(float(abs(sjs.positions[j]))
                          for j in self._grip_idxs)
            try:
                self._view.set_max_efforts(
                    values=np.full(n, GRIP_MAX_FORCE + 2.0 * cur,
                                   dtype=np.float32),
                    joint_indices=idx,
                )
            except Exception:  # noqa: BLE001
                pass
            # Drive only the command joint; the PhysX mimic closes the
            # opposing inner finger in lockstep (underactuated linkage).
            self._view.set_joint_velocity_targets(
                np.full((1, n), GRIP_CLOSE_VEL, dtype=np.float32),
                joint_indices=idx,
            )
        elif state == "hold":
            # Stiff vice clamp at the contacted aperture with a strong
            # effort cap -- can't ratchet shut on a disturbance.
            try:
                self._view.set_max_efforts(
                    values=np.full(n, GRIP_HOLD_FORCE, dtype=np.float32),
                    joint_indices=idx,
                )
            except Exception:  # noqa: BLE001
                pass
            self._view.set_joint_position_targets(
                self._grip_hold_pos.reshape(1, n).astype(np.float32),
                joint_indices=idx,
            )
        else:  # open
            self._view.set_joint_position_targets(
                np.full((1, n), GRIP_OPEN_Q, dtype=np.float32),
                joint_indices=idx,
            )

    def gripper_joint_state(self):
        """Returns (dof indices, positions) for the single driven command
        joint (idx81); the inner finger follows via the PhysX mimic."""
        sjs = self._robot.get_joints_state()
        if sjs is None:
            return self._grip_idxs, None
        return self._grip_idxs, [float(sjs.positions[j])
                                 for j in self._grip_idxs]

    def set_arm_mode(self, kinematic: bool) -> None:
        """Switch arm execution: kinematic (accurate centering) vs PD
        (smooth carry of a friction-held object). Switching to PD clears the
        kinematic hold so the PD drive's last targets govern instead."""
        self._arm_kinematic = kinematic
        if not kinematic:
            self._arm_hold_pos = None
            self._arm_hold_idx = None

    def base_hold(self, ctx) -> None:
        sim_js = self._robot.get_joints_state()
        self.base.step(ctx.world, sim_js)
        self._apply_gripper()
        # Carry-integrity probe: while an object is held, log |ee-obj| and
        # obj_z every ~120 steps so a slip is visible WHEN it happens (not
        # only post-hoc at Release). genie_sim's blind spot was exactly this.
        held = ctx.blackboard.get("held") if hasattr(ctx, "blackboard") else None
        if held is not None:
            self._carry_dbg = getattr(self, "_carry_dbg", 0) + 1
            if self._carry_dbg % 40 == 0:
                try:
                    ee_p, _ = self.ee_world_pose(ctx)
                    op, _ = ctx.scene.object_pose(held)
                    d = float(np.linalg.norm(np.asarray(ee_p) - np.asarray(op)))
                    gi, gp = self.gripper_joint_state()
                    gps = ("n/a" if gp is None
                           else "[" + ",".join(f"{x:.3f}" for x in gp) + "]")
                    print(f"[carry] |ee-obj|={d:.4f} obj_z={float(op[2]):.3f} "
                          f"fingers={gps}")
                except Exception:  # noqa: BLE001
                    pass
        # In kinematic mode, re-assert the last arm config every step while
        # no plan is streaming, so the arm stays put (e.g. holds the grasp
        # pose steady while the fingers close, holds retract during nav).
        if (self._arm_kinematic and self._cmd_plan is None
                and self._arm_hold_pos is not None):
            self._robot.set_joint_positions(
                self._arm_hold_pos, self._arm_hold_idx
            )

    def hand_link_paths(self, stage) -> list:
        """Prim paths of every active-hand link, for collision filtering of a
        held object (the cube must not fight the fingers/wrist meshes)."""
        side = "gripper_r_" if "_r_" in self.ee_link else "gripper_l_"
        root = self.robot_prim_path
        out = []
        for prim in stage.Traverse():
            p = prim.GetPath().pathString
            if p.startswith(root) and prim.GetName().startswith(side):
                out.append(p)
        return out

    def grasp_link_path(self, stage) -> str:
        """USD prim path of the physical grasp link under the robot."""
        if self._grasp_link_path is not None:
            return self._grasp_link_path
        root = self.robot_prim_path
        for prim in stage.Traverse():
            p = prim.GetPath().pathString
            if prim.GetName() == self.grasp_link and p.startswith(root):
                self._grasp_link_path = p
                return p
        raise RuntimeError(
            f"grasp link prim '{self.grasp_link}' not found under {root}"
        )

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
        """Arm/base quasi-static check. EXCLUDES the gripper finger joints:
        while clamping an object the gripper is intentionally force-driven
        and a thin/underactuated grip keeps the finger joints jittering
        above threshold -- including them made MoveArmTo(lift)'s pre-plan
        gate wait forever, so the arm only lifted AFTER the box was ejected
        (the gripper finally stopping). The gate means 'is the arm settled',
        not 'has the gripper stopped clamping'."""
        sim_js = self._robot.get_joints_state()
        if sim_js is None:
            return False
        if self._nongrip_idx is None:
            self._nongrip_idx = [
                i for i, n in enumerate(self._robot.dof_names)
                if "gripper_" not in n
            ]
        v = np.abs(np.asarray(sim_js.velocities)[self._nongrip_idx])
        return float(np.max(v)) < 0.5

    # ---- arm trajectory streaming -------------------------------------
    def plan_arm_to(self, p_world, q_world) -> bool:
        cu_js = self.read_cu_js()
        if cu_js is None:
            return False
        plan = self.arm.plan_to_world_pose(cu_js, p_world, q_world, self.base)
        if plan is None:
            return False
        # Exclude gripper joints: cuRobo locks them OPEN and get_full_js
        # carries that value, so streaming them would re-open the gripper
        # every tick and fight the close command. We drive the gripper
        # ourselves (set_gripper); USD mimic joints follow.
        common = [n for n in self._robot.dof_names
                  if n in plan.joint_names and "gripper_" not in n]
        self._cmd_idx_list = [self._robot.get_dof_index(n) for n in common]
        self._cmd_plan = plan.get_ordered_joint_state(common)
        self._cmd_idx = 0
        return True

    def advance_arm_plan(self, sim) -> bool:
        """Stream one waypoint. Returns True when the plan is exhausted.

        kinematic mode: hard set_joint_positions -> mm-accurate (used to
        center the grasp). PD mode: apply_action -> smooth, so a friction-
        held object follows the hand during the carry.
        """
        from isaacsim.core.utils.types import ArticulationAction

        if self._cmd_plan is None:
            return True
        st = self._cmd_plan[self._cmd_idx]
        pos = st.position.cpu().numpy()
        if self._arm_kinematic:
            self._robot.set_joint_positions(pos, self._cmd_idx_list)
            self._arm_hold_pos = pos          # keep holding the final config
            self._arm_hold_idx = list(self._cmd_idx_list)
        else:
            self._art_ctrl.apply_action(ArticulationAction(
                pos, st.velocity.cpu().numpy(),
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
