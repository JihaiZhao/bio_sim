#
# R1ProRobot: the skills-facing facade for the R1 Pro, mirroring G2Robot's
# public surface (both inherit RobotBase) so the pick-place pipeline
# (skills/task/runner/Gripper) is robot-agnostic. Differences vs G2:
#
#   * BASE = the DYNAMIC BEHAVIOR-1K holonomic base (HolonomicBaseDriver +
#     HolonomicNav adapter), NOT the kinematic swerve. This is the whole
#     point: a kinematic-teleport base flings a friction-held object during
#     the carry; a PhysX-driven holonomic base moves it for real so the
#     cube follows the hand.
#   * The driver owns base(x/y/rz) + torso (stiff one-time hold). cuRobo
#     plans ONLY the arm (R1Pro_arm_no_torso.yml lock_joints freezes
#     base+torso+grippers). So the per-step arm re-assert here is
#     ACTIVE-ARM-ONLY: re-asserting base/torso every step via
#     set_joint_positions is exactly the articulation-stomp regression
#     that breaks the base position drive.
#   * Gripper = R1 dual independently-driven fingers (no omnipicker mimic).
#   * ee = right_eef_link, idle = left_eef_link; grasp link =
#     right_gripper_link; hand prefix "right_gripper".
#

from __future__ import annotations

import numpy as np

from .holonomic import HolonomicBaseDriver, HolonomicNav
from .robot_base import RobotBase


class R1ProRobot(RobotBase):
    # OmniGibson default_qpos has the fingers at 0.05 (open); closing
    # drives toward 0 under a capped force. Hold-squeeze is smaller than
    # G2's because the R1 finger throw is shorter (0..0.05 vs 0..0.8).
    GRIP_OPEN_Q = 0.05
    GRIP_HOLD_SQUEEZE = 0.01

    def __init__(self, robot_yml: str = "R1Pro_arm_no_torso.yml",
                 use_urdf_kinematics: bool = False, reactive: bool = False):
        super().__init__(robot_yml, use_urdf_kinematics, reactive)

    def _init_specifics(self) -> None:
        side = "right" if "right" in self.ee_link else "left"
        self._side = side
        # physical rigid link the held object's cuRobo attached_object is
        # parented to (R1: the gripper body link, not the eef frame).
        self.grasp_link = f"{side}_gripper_link"
        # R1 has TWO independently-driven finger joints per hand (no
        # omnipicker mimic) -- drive both symmetrically.
        self.grip_cmd_joints = [
            f"{side}_gripper_finger_joint1",
            f"{side}_gripper_finger_joint2",
        ]
        self._driver: HolonomicBaseDriver | None = None
        self._all_idx = None          # all 28 cspace dof idx (one-time init)
        self._act_arm_idx = None      # ACTIVE (right) arm dof idx (7)

    def _arm_joint_indices(self, side: str) -> list[int]:
        # side is "left" or "right" (from RobotBase's default
        # _init_pose_sides). R1 joint names start with "left_arm_joint" /
        # "right_arm_joint".
        return [i for i, n in enumerate(self.j_names)
                if n.startswith(f"{side}_arm_joint")][:7]

    # ---- per-step init ------------------------------------------------
    def ensure_initialized(self, ctx) -> None:
        if self._initialized:
            return
        if self._art_ctrl is None:
            self._art_ctrl = self._robot.get_articulation_controller()
        av = self._robot._articulation_view
        # initialize the articulation view ONCE, BEFORE reading joints --
        # get_joints_state() is None until the view is initialized AND a
        # physics step has run, so guarding on it before init bails forever.
        if not getattr(self, "_av_inited", False):
            av.initialize()
            self._av_inited = True
        self._view = av
        if self._robot.get_joints_state() is None:
            return  # view warm but no physics step yet -- retry (10 tries)

        # 1. put the WHOLE robot at retract ONCE (one-time
        #    set_joint_positions is fine; the regression is doing it
        #    EVERY step).
        self._all_idx = [self._robot.get_dof_index(x) for x in self.j_names]
        self._robot.set_joint_positions(self.retract_config, self._all_idx)

        # 2. ACTIVE-arm (7) + IDLE-arm (7) dof idx. cuRobo plans these;
        #    the driver owns base/torso/grippers.
        self._act_arm_idx = [
            self._robot.get_dof_index(f"{self._side}_arm_joint{i}")
            for i in range(1, 8)]
        idle_side = "left" if self._side == "right" else "right"
        idle_arm_idx = [
            self._robot.get_dof_index(f"{idle_side}_arm_joint{i}")
            for i in range(1, 8)]
        arm_idx = self._act_arm_idx + idle_arm_idx
        # PD-stiffen both arms (USD drive is soft); friction-held object
        # then follows the hand smoothly under apply_action.
        n = len(arm_idx)
        av.set_gains(
            kps=np.full((1, n), self.ARM_KP, dtype=np.float32),
            kds=np.full((1, n), self.ARM_KD, dtype=np.float32),
            joint_indices=np.asarray(arm_idx, dtype=np.int32))
        av.set_max_efforts(values=np.array([5000 for _ in arm_idx]),
                           joint_indices=arm_idx)
        # FREEZE BOTH arms at the tucked init pose via a ONE-TIME PD
        # position target (same principle as the driver's torso hold):
        # set_joint_positions only sets STATE -- the stiff ARM_KP PD
        # needs an explicit TARGET or the (idle) arm drifts under
        # gravity ("arm not frozen" -> COM shifts -> the passive base
        # tips). No per-step set_joint_positions (that teleport stomps
        # the base drive).
        arm_names = ([f"{self._side}_arm_joint{i}" for i in range(1, 8)]
                     + [f"{idle_side}_arm_joint{i}" for i in range(1, 8)])
        arm_tgt = np.asarray(
            [self.retract_config[self.j_names.index(nm)]
             for nm in arm_names], dtype=np.float32)
        av.set_joint_position_targets(
            arm_tgt.reshape(1, n),
            joint_indices=np.asarray(arm_idx, dtype=np.int32))
        # active-arm hold bookkeeping (used post-plan to re-pin the PD
        # target -- NOT a per-step teleport).
        self._arm_hold_pos = arm_tgt[:7].copy()
        self._arm_hold_idx = list(self._act_arm_idx)

        # 3. DYNAMIC holonomic base: driver owns base(x/y/rz, position
        #    drive) + torso (stiff one-time hold). Torso loaded pose is
        #    now retract(0) so it holds upright. Then wrap in the
        #    NavController-compatible adapter the skills speak.
        self._driver = HolonomicBaseDriver(
            self._robot, av, ctx.world.stage, self.robot_prim_path)
        self._driver.setup()
        self.base = HolonomicNav(self._driver)

        # 4. gripper: force-controlled friction grasp on BOTH R1 fingers.
        self._grip_idxs = [self._robot.get_dof_index(j)
                           for j in self.grip_cmd_joints]
        self._grip_mode = None
        self._grip_close = False
        self._grip_state = "open"
        self._grip_hold_pos = None
        self.set_gripper(close=False)
        self._initialized = True
        print(f"[r1pro] initialized. ee={self.ee_link} idle={self.idle_link} "
              f"active_arm_idx={self._act_arm_idx} grip={self._grip_idxs} "
              f"(dynamic holonomic base + cuRobo arm)")

    def reset_arm(self) -> None:
        if self._robot is None or self._act_arm_idx is None:
            return
        rc = np.asarray(
            [self.retract_config[self.j_names.index(
                f"{self._side}_arm_joint{i}")] for i in range(1, 8)],
            dtype=np.float32)
        self._robot.set_joint_positions(rc, self._act_arm_idx)
        self._cmd_plan = None
        self._cmd_idx = 0
        self._arm_kinematic = True
        self._arm_hold_pos = rc
        self._arm_hold_idx = list(self._act_arm_idx)

    # ---- per-step: drive base + hold arm (NEVER base/torso re-assert) -
    def base_hold(self, ctx) -> None:
        sim_js = self._robot.get_joints_state()
        held = (ctx.blackboard.get("held")
                if hasattr(ctx, "blackboard") else None)
        self.base.set_carrying(held is not None)
        self.base.step(ctx.world, sim_js)
        self._apply_gripper()
        if held is not None:
            self._carry_dbg = getattr(self, "_carry_dbg", 0) + 1
            if self._carry_dbg % 40 == 0:
                try:
                    ee_p, _ = self.ee_world_pose(ctx)
                    op, _ = ctx.scene.object_pose(held)
                    d = float(np.linalg.norm(
                        np.asarray(ee_p) - np.asarray(op)))
                    _, gp = self.gripper_joint_state()
                    gps = ("n/a" if gp is None
                           else "[" + ",".join(f"{x:.3f}" for x in gp) + "]")
                    print(f"[carry] |ee-obj|={d:.4f} obj_z={float(op[2]):.3f} "
                          f"fingers={gps}")
                except Exception:  # noqa: BLE001
                    pass
        # NO per-step arm re-assert here. The arm is frozen by the
        # ONE-TIME PD position target set in ensure_initialized (and
        # re-pinned at the end of each cuRobo plan in advance_arm_plan).
        # A per-step set_joint_positions on ANY subset of this shared
        # articulation stomps the HolonomicBaseDriver position drive --
        # the exact regression we spent the session killing.

    # ---- collision-filter prims --------------------------------------- #
    def hand_link_paths(self, stage) -> list:
        prefix = f"{self._side}_gripper"
        root = self.robot_prim_path
        out = []
        for prim in stage.Traverse():
            p = prim.GetPath().pathString
            if p.startswith(root) and prim.GetName().startswith(prefix):
                out.append(p)
        return out

    # ---- arm trajectory streaming ------------------------------------- #
    def plan_arm_to(self, p_world, q_world) -> bool:
        cu_js = self.read_cu_js()
        if cu_js is None:
            return False
        plan = self.arm.plan_to_world_pose(cu_js, p_world, q_world, self.base)
        if plan is None:
            return False
        # Stream ONLY the active (right) arm joints. cuRobo's
        # get_full_js carries the locked base/torso/gripper + free
        # idle-arm values; streaming any of those would (a) stomp the
        # base/torso driver and (b) swing the idle arm. The
        # driver/gripper are driven separately.
        act_tok = f"{self._side}_arm_joint"
        common = [n for n in self._robot.dof_names
                  if n in plan.joint_names and act_tok in n]
        self._cmd_idx_list = [self._robot.get_dof_index(n) for n in common]
        self._cmd_plan = plan.get_ordered_joint_state(common)
        self._cmd_idx = 0
        return True

    def advance_arm_plan(self, sim) -> bool:
        from isaacsim.core.utils.types import ArticulationAction

        if self._cmd_plan is None:
            return True
        st = self._cmd_plan[self._cmd_idx]
        pos = st.position.cpu().numpy()
        if self._arm_kinematic:
            self._robot.set_joint_positions(pos, self._cmd_idx_list)
            self._arm_hold_pos = pos
            self._arm_hold_idx = list(self._cmd_idx_list)
        else:
            self._art_ctrl.apply_action(ArticulationAction(
                pos, st.velocity.cpu().numpy(),
                joint_indices=self._cmd_idx_list))
        self._cmd_idx += 1
        for _ in range(2):
            sim.step(render=False)
        if self._cmd_idx >= len(self._cmd_plan.position):
            # plan done -> re-pin the PD target to the final pose so the
            # ARM_KP servo holds it (base_hold does NOT re-assert; a
            # per-step teleport would stomp the base drive).
            self._view.set_joint_position_targets(
                np.asarray(pos, dtype=np.float32).reshape(1, -1),
                joint_indices=np.asarray(self._cmd_idx_list,
                                         dtype=np.int32))
            self._arm_hold_pos = pos
            self._arm_hold_idx = list(self._cmd_idx_list)
            self._cmd_plan = None
            self._cmd_idx = 0
            return True
        return False

    # ---- frames ------------------------------------------------------- #
    def base_to_world(self, p_base, q_base):
        return self.base.base_to_world(p_base, q_base)


__all__ = ["R1ProRobot"]
