#
# G2Robot: the facade skills talk to. Hides the Isaac articulation bridge
# (DOF index map, joint-state read, ArticulationAction streaming) and owns
# the arm planner, KINEMATIC swerve base, and gripper. Skills never touch
# _articulation_view; they call robot.arm / robot.base / robot.gripper and
# work in WORLD coordinates (the facade handles the base-frame transform).
#
# Most of the boilerplate (cfg load, ee/idle link extraction, planner init,
# apply_init_pose, load_into, gripper force-control, robot_static, ...)
# lives on RobotBase; this file only carries the G2-specific divergence:
# the kinematic swerve base wiring, the underactuated omnipicker gripper
# topology (single command joint + PhysX mimic), and the per-step arm
# re-assert (legal here because the kinematic base does NOT own the
# articulation -- on R1's PhysX-driven holonomic base a per-step
# set_joint_positions stomps the base drive; see r1pro.py).
#

from __future__ import annotations

import math

import numpy as np

from .base import KeyboardTeleop, NavController, SwerveBaseController
from .robot_base import RobotBase


class G2Robot(RobotBase):
    def __init__(self, robot_yml: str = "G2_omnipicker_fixed_dual.yml",
                 use_urdf_kinematics: bool = False, reactive: bool = False,
                 env_root: str = "/World/env_0"):
        super().__init__(robot_yml, use_urdf_kinematics, reactive,
                         env_root=env_root)

    def _init_specifics(self) -> None:
        # physical link a grasped object is fixed-jointed to (a real
        # rigid articulation link with a collider, not the ee frame).
        # cuRobo's `attached_object` link is parented here too.
        self.grasp_link = (
            "gripper_r_base_link" if "_r_" in self.ee_link
            else "gripper_l_base_link")
        # Underactuated omnipicker: exactly ONE driven command joint per
        # hand (outer_joint1). robot.usda DELETES the DriveAPI on
        # inner_joint1 and gives it a PhysxMimicJointAPI:rotX slaved to
        # outer_joint1 (joint3/4 are passive linkage). So we drive ONLY
        # outer_joint1 and let PhysX propagate the inner finger via the
        # mimic constraint -- directly commanding inner_joint1 fights
        # that constraint (a tensor-API drive vs the PhysX mimic on one
        # DOF) and is non-physical. (Earlier two-finger drive was a
        # mistake; genie_sim uses this same single-drive + mimic
        # topology.) 0.0 = closed, GRIP_OPEN_Q = open.
        self.grip_cmd_joints = (
            ["idx81_gripper_r_outer_joint1"] if "_r_" in self.ee_link
            else ["idx41_gripper_l_outer_joint1"])
        self.grip_cmd_joint = self.grip_cmd_joints[0]
        # Mimic follower: inner_joint1 is slaved to outer_joint1 with
        # multiplier=-1. Not commanded directly, but reset_gripper needs to
        # SNAP it to -outer + zero velocity so cross-run state matches the
        # first-run state exactly (otherwise the second run starts with the
        # inner finger still near the prior grasp pose + non-zero residual
        # velocity, which the mimic constraint then resolves over a few
        # ticks -- enough to skew the friction-close timing).
        self.grip_passive_joints = (
            ["idx71_gripper_r_inner_joint1"] if "_r_" in self.ee_link
            else ["idx31_gripper_l_inner_joint1"])
        # Base spawn pose (x, y, yaw); overlaid from cfg.robot_start in
        # _apply_base_start, applied once when the NavController is
        # created and on the R-key env reset.
        self.base_start = (0.0, 0.0, 0.0)
        self._base_start_applied = False
        self._arm_idx = None

    def _apply_base_start(self, task_cfg: dict) -> None:
        rs = (task_cfg or {}).get("robot_start")
        if rs is not None and len(rs) == 3:
            self.base_start = (float(rs[0]), float(rs[1]), float(rs[2]))
            print(f"[init_pose] base spawn <- {self.base_start} "
                  f"(task robot_start)")

    def _init_pose_sides(self, iap):
        # G2 keeps its historical "l"/"r" side labels (matcher tokens).
        return [("l", iap.get("left")), ("r", iap.get("right"))]

    def _arm_joint_indices(self, side: str) -> list[int]:
        # side is "l" or "r"; G2 USD joint names contain "arm_l_" /
        # "arm_r_" as a substring (joint21..27 for left, 61..67 for right).
        return [i for i, n in enumerate(self.j_names)
                if f"arm_{side}_" in n][:7]

    def _body_joint_indices(self) -> list[int]:
        # G2 torso = idx01..idx05_body_joint*. Five DOF, in cspace order.
        return [i for i, n in enumerate(self.j_names) if "body_joint" in n]

    def _head_joint_indices(self) -> list[int]:
        # G2 head = idx11..idx13_head_joint*. Three DOF, in cspace order.
        return [i for i, n in enumerate(self.j_names) if "head_joint" in n]

    # ---- per-step init (settle window) --------------------------------
    def ensure_initialized(self, ctx) -> None:
        if self._initialized:
            return
        if self._art_ctrl is None:
            self._art_ctrl = self._robot.get_articulation_controller()
        self._robot._articulation_view.initialize()
        # Initialize the broadcast view too (lazy no-op if already up or
        # not built for N=1). write_* helpers tile from this point on.
        self._broadcast_initialized()
        self._arm_idx = [self._robot.get_dof_index(x) for x in self.j_names]
        self.write_joint_positions(self.retract_config, self._arm_idx)
        # Also PIN the PD drive targets to retract_config -- otherwise the
        # idle arm drifts toward 0 (shoulder roll = horizontal stick) the
        # instant the active arm starts streaming a plan and the
        # _cmd_plan-gated kinematic re-assert stops covering it.
        self.write_joint_position_targets(
            np.asarray(self.retract_config, dtype=np.float32),
            self._arm_idx,
        )
        # hold retract kinematically until the first reach
        self._arm_hold_pos = np.asarray(self.retract_config, dtype=np.float32)
        self._arm_hold_idx = list(self._arm_idx)
        self.write_max_efforts(
            np.array([5000 for _ in self._arm_idx], dtype=np.float32),
            self._arm_idx,
        )
        # Arm is PD-driven (apply_action) so a friction-held object
        # follows the hand smoothly. Stiffen the USD arm drive so PD
        # tracks tight enough for fingers to close on the cube.
        n = len(self._arm_idx)
        self.write_gains(
            kps=np.full(n, self.ARM_KP, dtype=np.float32),
            kds=np.full(n, self.ARM_KD, dtype=np.float32),
            joint_indices=self._arm_idx,
        )
        swerve = SwerveBaseController(
            self._robot, self._robot._articulation_view,
            av=self._av, num_envs=self._num_envs,
            env_spacing=getattr(self, "_env_spacing", 0.0),
            robot_facade=self,
        )
        swerve.configure_drive_modes()
        self.base = NavController(swerve)
        if not self._base_start_applied:
            self.base.reset_pose(*self.base_start)
            self._base_start_applied = True
            print(f"[g2] base spawned at {self.base_start}")
        # Gripper: force-controlled friction grasp. Configure the
        # command dof on the ARTICULATION VIEW (set_gains /
        # switch_dof_control_mode / set_max_efforts) -- the same proven
        # path the swerve base uses. Setting USD DriveAPI attrs after
        # articulation init does NOT take effect (Isaac caches drive
        # gains), which is why the fingers never actuated and the cube
        # was never picked up.
        self._view = self._robot._articulation_view
        self._grip_idxs = [self._robot.get_dof_index(n)
                           for n in self.grip_cmd_joints]
        self._grip_mode = None
        self._grip_close = False
        self._grip_state = "open"
        self._grip_hold_pos = None
        self.set_gripper(close=False)
        self._initialized = True

    def reset_gripper(self) -> None:
        """G2 override: ALSO snap the mimic inner-joint follower and ZERO
        finger velocities. The base reset_gripper only writes the DRIVEN
        outer joint -- inner_joint1 (PhysxMimicJointAPI:rotX, slave =
        -outer) is left wherever the prior grasp closed it, and the mimic
        constraint then has to resolve the asymmetry over several PhysX
        ticks. That transient is exactly the cross-run state drift that
        makes the same _CLOSE_TICKS land the fingers in different places
        on the first run vs after-reset run."""
        if self._robot is None or self._grip_idxs is None:
            return
        # Active (outer) joint(s) + passive (inner mimic) joint(s).
        outer_idx = list(self._grip_idxs)
        passive_idx = [self._robot.get_dof_index(n)
                       for n in self.grip_passive_joints]
        all_idx = outer_idx + passive_idx
        # outer -> +GRIP_OPEN_Q, inner -> -GRIP_OPEN_Q (mimic mult=-1).
        positions = np.concatenate([
            np.full(len(outer_idx), self.GRIP_OPEN_Q, dtype=np.float32),
            np.full(len(passive_idx), -self.GRIP_OPEN_Q, dtype=np.float32),
        ])
        self.write_joint_positions(positions, all_idx)
        # Zero residual velocities -- otherwise PhysX integrates the
        # carry-over velocity (from the prior close) for a few ticks before
        # PD damps it out, and that shifts the effective close window.
        try:
            self.write_joint_velocities(
                np.zeros(len(all_idx), dtype=np.float32), all_idx)
        except Exception as exc:  # noqa: BLE001
            print(f"[reset_gripper] zero-velocity skipped: {exc}")
        # Re-establish position mode + open target so PD holds it open.
        self._grip_state = "open"
        self._grip_close = False
        self._grip_hold_pos = None
        # Force the mode-switch path even if _grip_mode is already
        # "position" so switch_dof_control_mode + set_gains hit the
        # articulation view -- the prior run's drive cache may otherwise
        # leak into the new run.
        self._grip_mode = None
        self._sync_grip_mode("position")
        self._apply_gripper()

    def reset_arm(self) -> None:
        """Snap BOTH arms back to the retract/init pose and hold there
        (used by the R-key env reset -- without this only the base +
        cube reset and the arm stays wherever the last task left it)."""
        if self._robot is None or self._arm_idx is None:
            return
        rc = np.asarray(self.retract_config, dtype=np.float32)
        self.write_joint_positions(rc, self._arm_idx)
        self._cmd_plan = None
        self._cmd_idx = 0
        self._arm_kinematic = True
        self._arm_hold_pos = rc
        self._arm_hold_idx = list(self._arm_idx)

    def base_hold(self, ctx) -> None:
        sim_js = self._robot.get_joints_state()
        held = ctx.blackboard.get("held") if hasattr(ctx, "blackboard") else None
        # Carrying -> base.step uses the scaled-down speed/accel caps so
        # the kinematic in-place turn can't shear the friction-held cube.
        self.base.set_carrying(held is not None)
        self.base.step(ctx.world, sim_js)
        self._apply_gripper()
        # Carry-integrity probe: while an object is held, log |ee-obj|
        # and obj_z every ~120 steps so a slip is visible WHEN it happens
        # (not only post-hoc at Release). genie_sim's blind spot was
        # exactly this.
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
                    # Outer-joint actual applied effort: if this is pinned
                    # at +/- GRIP_HOLD_FORCE the PD is saturated (force
                    # ceiling is the bottleneck); if it's well under cap,
                    # the bottleneck is contact geometry, not force.
                    eff = "n/a"
                    try:
                        applied = self._robot._articulation_view\
                            .get_applied_joint_efforts(
                                joint_indices=np.asarray(
                                    self._grip_idxs, dtype=np.int32))
                        e_val = float(applied.flatten()[0])
                        eff = f"{e_val:+.1f}"
                    except Exception:  # noqa: BLE001
                        pass
                    print(f"[carry] |ee-obj|={d:.4f} obj_z={float(op[2]):.3f} "
                          f"fingers={gps} eff(outer)={eff} (cap=±"
                          f"{self.GRIP_HOLD_FORCE:.0f})")
                except Exception:  # noqa: BLE001
                    pass
        # In kinematic mode, re-assert the last arm config every step
        # while no plan is streaming, so the arm stays put (e.g. holds
        # the grasp pose steady while the fingers close, holds retract
        # during nav). Legal on G2 because the kinematic swerve base
        # does NOT own the articulation -- on R1's PhysX-driven
        # holonomic base this re-assert stomps the base drive (see
        # r1pro.py:base_hold for the explanation).
        if (self._arm_kinematic and self._cmd_plan is None
                and self._arm_hold_pos is not None):
            self.write_joint_positions(
                self._arm_hold_pos, self._arm_hold_idx
            )

    def hand_link_paths(self, stage) -> list:
        """Prim paths of every active-hand link, for collision filtering
        of a held object (the cube must not fight the fingers/wrist
        meshes)."""
        side = "gripper_r_" if "_r_" in self.ee_link else "gripper_l_"
        root = self.robot_prim_path
        out = []
        for prim in stage.Traverse():
            p = prim.GetPath().pathString
            if p.startswith(root) and prim.GetName().startswith(side):
                out.append(p)
        return out

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
        # ALSO exclude the IDLE arm: cuRobo only pins its gripper-center
        # LINK pose, leaving the 7 idle-arm joints free in the
        # null-space, so streaming get_full_js would swing the idle
        # elbow around. Don't command it -> it just holds its retract
        # pose (PD target from init).
        idle_side = "l" if "_r_" in self.ee_link else "r"
        idle_arm_tok = f"arm_{idle_side}_"
        common = [n for n in self._robot.dof_names
                  if n in plan.joint_names
                  and "gripper_" not in n
                  and idle_arm_tok not in n]
        self._cmd_idx_list = [self._robot.get_dof_index(n) for n in common]
        self._cmd_plan = plan.get_ordered_joint_state(common)
        self._cmd_idx = 0
        return True

    def advance_arm_plan(self, sim) -> bool:
        """Stream one waypoint. Returns True when the plan is exhausted.

        kinematic mode: hard set_joint_positions -> mm-accurate (used to
        center the grasp). PD mode: apply_action -> smooth, so a
        friction-held object follows the hand during the carry.
        """
        from isaacsim.core.utils.types import ArticulationAction

        if self._cmd_plan is None:
            return True
        st = self._cmd_plan[self._cmd_idx]
        pos = st.position.cpu().numpy()
        if self._arm_kinematic:
            # Broadcast: (K,) waypoint tiles to (N, K) across all envs.
            self.write_joint_positions(pos, self._cmd_idx_list)
            self._arm_hold_pos = pos          # keep holding the final config
            self._arm_hold_idx = list(self._cmd_idx_list)
        else:
            # PD mode: position + velocity targets via the broadcast view.
            vel = st.velocity.cpu().numpy()
            self.write_joint_position_targets(pos, self._cmd_idx_list)
            self.write_joint_velocity_targets(vel, self._cmd_idx_list)
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


__all__ = ["G2Robot", "KeyboardTeleop"]
