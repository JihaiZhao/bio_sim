#
# RobotBase: shared scaffold for the skills-facing robot facades. Owns:
#
#   * robot_cfg/kin loading (cuRobo yml + asset_root injection)
#   * ee/idle link + joint-name extraction, retract_config bookkeeping
#   * ArmPlanner construction + load_into (USD reference, articulation init,
#     post-scene planner rebuild + warmup)
#   * generic apply_init_pose (joint-side matching delegated to subclass)
#   * gripper PD/force machinery (set_gripper / clamp_hold / _sync_grip_mode /
#     _apply_gripper), parameterized by class-level GRIP_* constants
#   * the simple shared accessors (read_cu_js, robot_static, ee_world_pose,
#     grasp_link_path, gripper_joint_state, base_ready, set_arm_mode)
#
# Things subclasses keep, because they are structurally different across
# robots (kinematic swerve vs. holonomic PhysX base): ensure_initialized,
# reset_arm, base_hold, plan_arm_to, advance_arm_plan, base_to_world,
# hand_link_paths. See bio_sim/robot/{g2,r1pro}.py.
#
# Naming: this file is NOT `base.py` because `base.py` already exists in
# this package (NavController, G2's mobile base). The "Robot" prefix
# disambiguates and avoids confusion with IsaacSim's `Robot` symbol.
#

from __future__ import annotations

import os

import numpy as np

from curobo.types.state import JointState
from curobo.util.logger import log_error
from curobo.util_file import get_robot_configs_path, join_path, load_yaml

from ..asset_lib import asset_root
from .arm import ArmPlanner
from .gripper import Gripper

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Robot geometry is an ASSET (resolved via the shared asset library / the
# SIM_ASSETS env var); the cuRobo *.yml files are CONFIG and stay in-repo
# (genie keeps robot_cfg separate from the asset tree the same way).
_DEFAULT_CFG_DIR = os.path.join(
    _PROJECT_ROOT, "config", "curobo", "configs", "robot")


def _other_link(link_names, ee_link):
    for ln in link_names:
        if ln != ee_link:
            return ln
    return None


class RobotBase:
    # Arm PD gains (USD ships ~1e4 stiffness -> too soft; stiffen for
    # tracking so a friction-held object follows the hand). Subclasses
    # override if the USD ships different defaults.
    ARM_KP: float = 1.0e5
    ARM_KD: float = 1.0e4
    # Gripper force-control constants. GRIP_OPEN_Q / GRIP_HOLD_SQUEEZE
    # differ per hardware (G2 omnipicker: idx81 0..0.8; R1 dual finger:
    # 0..0.05); the others are shared across both validated configs.
    GRIP_OPEN_Q: float = 0.8
    GRIP_KP: float = 1.0e5
    GRIP_KD: float = 1.0e3
    GRIP_MAX_FORCE: float = 70.0
    GRIP_CLOSE_VEL: float = -0.6
    # SQUEEZE = position-target step at the close->hold transition. With
    # KP=1e5 each rad of step asks PD for 1e5 N*m, so even 0.01 saturates
    # the effort cap. Keep this SMALL so the transition impulse is gentle:
    # the steady-state contact force is set by HOLD_FORCE, not by SQUEEZE.
    # 0.03 was tuned for the heavy cube; on a 0.04 kg well plate it gave
    # PD an impulsive kick that ejected the plate sideways before
    # bilateral contact settled.
    GRIP_HOLD_SQUEEZE: float = 0.01
    # Hold-phase effort cap on the driven joint (omnipicker outer_joint1
    # for G2; finger joints for R1). With omnipicker linkage advantage,
    # 200 N*m at outer_joint1 generates several hundred N of fingertip
    # normal force -- plenty for a 0.04 kg plate (need ~0.08 N at mu=5).
    # 1500 was the diagnostic-headroom value used to confirm contact
    # was being detected; it also slams a light plate out of the grasp.
    # 200 is the goldilocks: enough to lock, mild enough to not flick.
    GRIP_HOLD_FORCE: float = 200.0

    def __init__(self, robot_yml: str,
                 use_urdf_kinematics: bool = False,
                 reactive: bool = False):
        cfg_dir = (_DEFAULT_CFG_DIR if os.path.isdir(_DEFAULT_CFG_DIR)
                   else get_robot_configs_path())
        self.robot_cfg = load_yaml(join_path(cfg_dir, robot_yml))["robot_cfg"]
        kin = self.robot_cfg["kinematics"]
        kin["external_asset_path"] = asset_root()
        kin["external_robot_configs_path"] = cfg_dir
        if use_urdf_kinematics:
            kin["use_usd_kinematics"] = False

        self.ee_link = kin["ee_link"]
        self.link_names = kin["link_names"]
        self.idle_link = _other_link(self.link_names, self.ee_link)
        self.j_names = kin["cspace"]["joint_names"]
        self.retract_config = kin["cspace"]["retract_config"]
        self._reactive = reactive

        # Subclass populates these in _init_specifics():
        #   self.grasp_link  (str)            # physical rigid link for weld
        #   self.grip_cmd_joints (list[str])  # commanded gripper DOFs
        # plus any robot-specific state (base_start, _side, ...).
        self.grasp_link: str = ""
        self.grip_cmd_joints: list[str] = []
        self._grasp_link_path = None

        # Shared bookkeeping (subclasses inherit the slots; the few that
        # only one robot uses, like G2.base_start or R1._act_arm_idx,
        # live on the subclass).
        self._robot = None
        self.robot_prim_path = None
        self._art_ctrl = None
        self._view = None
        self._initialized = False
        self._nongrip_idx = None

        # Active-arm trajectory stream.
        self._cmd_plan = None
        self._cmd_idx = 0
        self._cmd_idx_list = None

        # Per-phase arm execution: kinematic (mm-accurate centering) vs
        # PD (smooth carry of a friction-held object).
        self._arm_kinematic = True
        self._arm_hold_pos = None
        self._arm_hold_idx = None

        # Gripper state machine (open / close / hold).
        self._grip_idxs = None
        self._grip_close = False
        self._grip_mode = None
        self._grip_state = "open"
        self._grip_hold_pos = None

        # Subclass concretizes the base controller (NavController for G2,
        # HolonomicNav for R1).
        self.base = None

        # Subclass-specific knobs (grasp_link, grip_cmd_joints, _side,
        # base_start, ...) before we build the planner / Gripper(self),
        # since Gripper reads back from self.
        self._init_specifics()

        self.arm = ArmPlanner(self.robot_cfg, None, self.ee_link,
                              self.idle_link, reactive=reactive)
        self.gripper = Gripper(self)

    # ---- subclass hooks ------------------------------------------------
    def _init_specifics(self) -> None:
        """Set self.grasp_link, self.grip_cmd_joints, and any
        robot-specific bookkeeping (base_start, _side, ...)."""
        raise NotImplementedError

    def _arm_joint_indices(self, side: str) -> list[int]:
        """Return the (up to 7) dof indices in self.j_names belonging to
        the requested arm side ('left' / 'right' or shorter token).
        Subclass chooses the joint-name matcher."""
        raise NotImplementedError

    def _init_pose_sides(self, iap: dict):
        """Yield (side_label, vals) for apply_init_pose. side_label is
        both the log label and the argument passed to
        _arm_joint_indices(). Default emits 'left'/'right'; G2 overrides
        to keep its historical 'l'/'r' labels (and matcher token)."""
        return [("left", iap.get("left")), ("right", iap.get("right"))]

    def _body_joint_indices(self) -> list[int]:
        """DOF indices for the torso/body chain. Default: none -- robots
        without a body chain (or whose body is locked) return []."""
        return []

    def _head_joint_indices(self) -> list[int]:
        """DOF indices for the head/neck chain. Default: none."""
        return []

    def _apply_base_start(self, task_cfg: dict) -> None:
        """Hook for robots that read robot_start from the task cfg.
        Default: no-op (R1's holonomic base spawns at origin)."""
        pass

    # ---- per-task init pose overlay ------------------------------------
    def apply_init_pose(self, task_cfg: dict) -> None:
        """Overlay task_cfg init poses onto the IN-MEMORY retract_config
        -> drives BOTH the physical init pose (ensure_initialized /
        reset_arm) AND the cuRobo IK seed / null-space (ArmPlanner is
        rebuilt from self.robot_cfg in load_into, which runs AFTER
        this). The committed robot yml is never touched. Call before
        load_into(). Recognised keys (omit any to keep the yml value):
          init_arm_pose:  {left: [7], right: [7]}
          init_body_pose: [N]   N = len(_body_joint_indices())
          init_head_pose: [N]   N = len(_head_joint_indices())
        """
        self._apply_base_start(task_cfg)
        task_cfg = task_cfg or {}
        iap = task_cfg.get("init_arm_pose")
        body_vals = task_cfg.get("init_body_pose")
        head_vals = task_cfg.get("init_head_pose")
        if not iap and body_vals is None and head_vals is None:
            return
        rc = list(self.retract_config)
        if iap:
            for side, vals in self._init_pose_sides(iap):
                if vals is None:
                    continue
                idxs = self._arm_joint_indices(side)
                if len(idxs) != 7 or len(vals) != 7:
                    print(f"[init_pose] {side}-arm needs 7 values "
                          f"(got {len(vals)}, slots {len(idxs)}) -- skipped")
                    continue
                for i, v in zip(idxs, vals):
                    rc[i] = float(v)
                print(f"[init_pose] {side}-arm <- {[float(v) for v in vals]} "
                      f"(task override, in-memory; committed yml untouched)")
        for label, vals, idxs in (
            ("body", body_vals, self._body_joint_indices()),
            ("head", head_vals, self._head_joint_indices()),
        ):
            if vals is None:
                continue
            if not idxs:
                print(f"[init_pose] {label} pose given but this robot has "
                      f"no {label} joints -- skipped")
                continue
            if len(vals) != len(idxs):
                print(f"[init_pose] {label} needs {len(idxs)} values "
                      f"(got {len(vals)}) -- skipped")
                continue
            for i, v in zip(idxs, vals):
                rc[i] = float(v)
            print(f"[init_pose] {label} <- {[float(v) for v in vals]} "
                  f"(task override, in-memory; committed yml untouched)")
        self.retract_config = rc
        self.robot_cfg["kinematics"]["cspace"]["retract_config"] = rc

    # ---- world build (called by play before sim.run) ------------------
    def load_into(self, sim, scene) -> None:
        from isaacsim.core.api.robots import Robot
        from isaacsim.core.utils.stage import add_reference_to_stage

        kin = self.robot_cfg["kinematics"]
        usd_path = join_path(kin.get("external_asset_path"), kin["usd_path"])
        self.robot_prim_path = "/World/" + kin["usd_robot_root"].strip("/")
        add_reference_to_stage(usd_path=usd_path,
                               prim_path=self.robot_prim_path)
        self._robot = sim.world.scene.add(
            Robot(prim_path=self.robot_prim_path, name="robot",
                  position=np.array([0.0, 0.0, 0.0])))
        sim.world.initialize_physics()
        self._robot.initialize()
        # Rebuild the arm planner now that we can give it the scene's
        # world cfg.
        self.arm = ArmPlanner(self.robot_cfg, scene.curobo_world,
                              self.ee_link, self.idle_link,
                              reactive=self._reactive)
        self.arm.warmup()

    # ---- per-step init ------------------------------------------------
    @property
    def base_ready(self) -> bool:
        return self.base is not None

    # ---- gripper (shared force-control machinery) ---------------------
    def set_gripper(self, close: bool) -> None:
        """close = velocity/force mode (kp=0, capped effort, closing vel)
        to drive the fingers onto the object. open = position mode to
        the rest aperture."""
        self._grip_state = "close" if close else "open"
        self._grip_close = close
        self._grip_hold_pos = None
        self._sync_grip_mode("velocity" if close else "position")
        self._apply_gripper()

    def reset_gripper(self) -> None:
        """KINEMATICALLY snap finger joints to the open aperture and reassert
        the open PD target. Used by the R-key env reset: `set_gripper(open)`
        alone only writes a POSITION TARGET, so right after release() the
        fingers are still near the closed grasp pose and the PD has to
        converge over many ticks. With the immediate runner.restart() the
        arm reaches pre-grasp before the fingers fully open -> the lower
        finger sweeps INTO the plate and pushes it sideways (the "slide on
        2nd run" bug). Snap-open removes that race window."""
        if self._robot is None or self._grip_idxs is None:
            return
        idx = np.asarray(self._grip_idxs, dtype=np.int32)
        n = len(self._grip_idxs)
        q_open = np.full(n, self.GRIP_OPEN_Q, dtype=np.float32)
        self._robot.set_joint_positions(q_open, list(self._grip_idxs))
        # Re-establish position mode and target so PD holds it open.
        self._grip_state = "open"
        self._grip_close = False
        self._grip_hold_pos = None
        self._sync_grip_mode("position")
        self._apply_gripper()

    def clamp_hold(self) -> None:
        """Lock the fingers as a stiff vice at the CURRENTLY-contacted
        aperture (a small squeeze past it) with a high effort cap. Unlike
        a continuous closing-velocity command, a position-hold cannot
        ratchet shut when a disturbance shifts the object -- the box
        keeps a stable pocket and friction a stable normal force. Still
        pure friction (no joint between object and gripper); just a real
        vice-grip clamp."""
        sjs = self._robot.get_joints_state()
        if sjs is None:
            return
        pos = np.array([float(sjs.positions[j]) for j in self._grip_idxs])
        # The command joint CLOSES by decreasing position (open -> ~0
        # closed). Squeeze = a small decrement in the closing direction.
        self._grip_hold_pos = pos - self.GRIP_HOLD_SQUEEZE
        self._grip_state = "hold"
        self._grip_close = True
        self._sync_grip_mode("position")
        self._apply_gripper()

    def _sync_grip_mode(self, mode: str) -> None:
        n = len(self._grip_idxs)
        idx = np.asarray(self._grip_idxs, dtype=np.int32)
        if mode != self._grip_mode:
            kp = 0.0 if mode == "velocity" else self.GRIP_KP
            self._view.set_gains(
                kps=np.full((1, n), kp, dtype=np.float32),
                kds=np.full((1, n), self.GRIP_KD, dtype=np.float32),
                joint_indices=idx)
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
                    values=np.full(n, self.GRIP_MAX_FORCE + 2.0 * cur,
                                   dtype=np.float32),
                    joint_indices=idx)
            except Exception:  # noqa: BLE001
                pass
            self._view.set_joint_velocity_targets(
                np.full((1, n), self.GRIP_CLOSE_VEL, dtype=np.float32),
                joint_indices=idx)
        elif state == "hold":
            try:
                self._view.set_max_efforts(
                    values=np.full(n, self.GRIP_HOLD_FORCE, dtype=np.float32),
                    joint_indices=idx)
            except Exception:  # noqa: BLE001
                pass
            self._view.set_joint_position_targets(
                self._grip_hold_pos.reshape(1, n).astype(np.float32),
                joint_indices=idx)
        else:  # open
            self._view.set_joint_position_targets(
                np.full((1, n), self.GRIP_OPEN_Q, dtype=np.float32),
                joint_indices=idx)

    def gripper_joint_state(self):
        sjs = self._robot.get_joints_state()
        if sjs is None:
            return self._grip_idxs, None
        return self._grip_idxs, [float(sjs.positions[j])
                                 for j in self._grip_idxs]

    # ---- arm mode (kinematic vs PD) -----------------------------------
    def set_arm_mode(self, kinematic: bool) -> None:
        """Switch arm execution: kinematic (accurate centering) vs PD
        (smooth carry of a friction-held object). Switching to PD clears
        the kinematic hold so the PD drive's last targets govern."""
        self._arm_kinematic = kinematic
        if not kinematic:
            self._arm_hold_pos = None
            self._arm_hold_idx = None

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
            acceleration=self.arm.tensor_args.to_device(
                sim_js.velocities) * 0.0,
            jerk=self.arm.tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=self._robot.dof_names,
        )
        return cu_js.get_ordered_joint_state(self.arm.joint_names)

    def robot_static(self) -> bool:
        """Arm/base quasi-static check. EXCLUDES the gripper finger
        joints: while clamping an object the gripper is intentionally
        force-driven and a thin/underactuated grip keeps the finger
        joints jittering above threshold -- including them made
        MoveArmTo(lift)'s pre-plan gate wait forever, so the arm only
        lifted AFTER the box was ejected. The gate means 'is the arm
        settled', not 'has the gripper stopped clamping'."""
        sim_js = self._robot.get_joints_state()
        if sim_js is None:
            return False
        if self._nongrip_idx is None:
            self._nongrip_idx = [
                i for i, n in enumerate(self._robot.dof_names)
                if "gripper_" not in n]
        v = np.abs(np.asarray(sim_js.velocities)[self._nongrip_idx])
        return float(np.max(v)) < 0.5

    # ---- frames -------------------------------------------------------
    def base_to_world(self, p_base, q_base):
        raise NotImplementedError

    def ee_world_pose(self, ctx):
        """Current gripper-center pose in WORLD (FK in base frame
        -> world). base_to_world is robot-specific."""
        cu_js = self.read_cu_js()
        pose = self.arm.fk_link_pose(cu_js.unsqueeze(0), self.ee_link)
        p_b = pose.position.cpu().numpy().ravel()
        q_b = pose.quaternion.cpu().numpy().ravel()
        return self.base_to_world(p_b, q_b)

    # ---- collision-filter / grasp link prims --------------------------
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
            f"grasp link prim '{self.grasp_link}' not found under {root}")


__all__ = ["RobotBase"]
