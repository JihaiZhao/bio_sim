#
# HolonomicBaseDriver -- a self-contained port of BEHAVIOR-1K's
# HolonomicBaseJointController (OmniGibson omnigibson/controllers/
# holonomic_base_joint_controller.py + holonomic_base_robot.py).
#
# WHY a separate class (not more edits in base.py): base.py's
# NavController/SwerveBaseController is the KINEMATIC swerve base (root
# teleported by integrating a twist). This is a different control law --
# a statically-stable 3-wheel mobile manipulator whose chassis is moved
# by PhysX position drives on a 6-DOF virtual joint chain. Keeping them
# apart keeps each contract clean.
#
# CONTRACT:
#   * base is the virtual chain  world -> x -> y -> z -> rx -> ry -> rz
#     -> base_link  (needs r1pro_holonomic.usda's WorldAnchor so it is
#     world-referenced, not a free-floating chain).
#   * ONLY x, y, rz are controlled, by POSITION drive via the BEHAVIOR-1K
#     q_to_action interface: action is a WORLD-absolute goal [gx,gy,grz];
#     world pos error == x/y joint error, Δyaw = wrap(grz - cur_rz_joint),
#     x/y/rz commanded SIMULTANEOUSLY (holonomic strafe -- no 90 deg
#     in-place turn, so the welded wheels are never scrubbed). Per-step
#     target is RATE-LIMITED (the lightweight trajectory; never the bare
#     endpoint -- OmniGibson only feeds q_to_action curobo waypoints).
#     Pure velocity drive (kp=0) had no position stiffness so skid-scrub
#     drifted the base ~27cm/turn; absolute-chase + per-step
#     set_joint_positions was the OTHER (earlier) regression -- this is
#     neither: position drive, incremental rate-limited, no teleport.
#   * z, rx, ry are LEFT PASSIVE -- no balance controller. A 3-wheel base
#     is statically stable; the body rests on its boundingSphere wheels
#     and finds its own level (this is why we DON'T lock them).
#   * every other DOF (trunk/arms/grippers/wheels/steers) is held at its
#     loaded pose by a ONE-TIME position joint drive (stiffness + critical
#     damping), NOT a per-step set_joint_positions teleport (that per-step
#     kinematic reset was what stomped the base drive each physics step).
#

from __future__ import annotations

import math

import numpy as np

# BEHAVIOR-1K HolonomicBaseRobot limits (omnigibson/robots/
# holonomic_base_robot.py:19-21, 207-214).
MAX_LIN_VEL = 1.5        # m/s
MAX_ANG_VEL = math.pi    # rad/s
MAX_EFFORT = 1000.0      # N / N*m  (per-DOF drive force cap)

# The 6 virtual base DOFs, in chain order. Only x/y/rz are driven.
_BASE_DOFS = ("x", "y", "z", "rx", "ry", "rz")
_DRIVEN = ("x", "y", "rz")

# Real hardware DOFs that BEHAVIOR-1K leaves PASSIVE (no drive): the
# steer/drive wheels just spin/caster freely -- locomotion is the virtual
# joints, the wheels are not the prime mover. Holding them rigid would
# make the boundingSphere contacts skid instead of roll. Matched by
# substring against the joint name.
_PASSIVE_NAME_HINTS = ("wheel_motor", "steer_motor")

# This driver's contract is the MOBILE BASE only. It must keep the
# superstructure that rides the base (the torso) rigid so the upper mass
# doesn't tip the chassis -- that IS a mobile-base concern. The ARMS /
# GRIPPERS are NOT this driver's job: they have a separate interface
# (final integration co-tunes them). So the driver holds ONLY the torso;
# anything matching this hint is held, the rest of the non-base DOFs are
# left to the arm interface (the nav_probe scaffolds them for a clean
# base-only test).
_HOLD_NAME_HINTS = ("torso",)


class HolonomicBaseDriver:
    """BEHAVIOR-1K holonomic base: POSITION-drive x/y/rz via the
    q_to_action interface with a mandatory per-step rate-limit, passive
    z/rx/ry, one-time joint-drive hold on the rest. See module docstring
    for the full contract."""

    def __init__(self, robot, articulation_view, stage, robot_prim,
                 *, base_prefix="base_footprint",
                 base_kp=1.0e5, hold_kp=1.0e7, hold_kd=1.0e5,
                 v_lin=0.5, v_ang=1.0,
                 max_lin=MAX_LIN_VEL, max_ang=MAX_ANG_VEL,
                 max_effort=MAX_EFFORT):
        self.robot = robot
        self.av = articulation_view
        self.stage = stage
        self.robot_prim = robot_prim
        self.base_prefix = base_prefix
        self.base_kp = float(base_kp)
        self.hold_kp = float(hold_kp)
        self.hold_kd = float(hold_kd)
        self.arm_idx = []        # non-base, non-torso (the arm interface's)
        # per-step rate-limit (the lightweight trajectory layer -- NEVER
        # feed q_to_action the bare endpoint, else a single-step |Δyaw|>π
        # wraps backwards / Δxy overshoots). v_ang*dt must stay << π.
        self.v_lin = float(v_lin)
        self.v_ang = float(v_ang)
        self.max_lin = float(max_lin)
        self.max_ang = float(max_ang)
        self.max_effort = float(max_effort)
        self._q_base0 = None

        self.jname = {n: f"{base_prefix}_{n}_joint" for n in _BASE_DOFS}
        self.idx = {}            # name -> dof index
        self.base_idx = []       # [x, y, rz] dof indices (velocity-driven)
        self.passive_idx = []    # z/rx/ry-equivalent + wheel/steer (free)
        self.hold_idx = []       # trunk/arms/grippers (one-time pos drive)
        self.q_hold = None       # loaded pose of the held DOFs
        self._blink = None       # base_link prim (true-pose readback)
        self._xc = None
        self._ready = False

    # ------------------------------------------------------------------ #
    # setup: call ONCE, after world.initialize_physics() + robot/av
    # .initialize() AND at least one sim step so get_joints_state() is live.
    # ------------------------------------------------------------------ #
    def setup(self):
        from pxr import UsdGeom

        n_dof = len(self.robot.dof_names)
        self.idx = {n: self.robot.get_dof_index(self.jname[n])
                    for n in _BASE_DOFS}
        missing = [self.jname[n] for n, i in self.idx.items() if i is None]
        if missing:
            raise RuntimeError(
                f"[holonomic] base joints not found in articulation: "
                f"{missing}  (dof_names={list(self.robot.dof_names)})")
        self.base_idx = [self.idx[n] for n in _DRIVEN]
        base6 = {self.idx[n] for n in _BASE_DOFS}
        names = list(self.robot.dof_names)
        # passive = the 6 virtual base DOFs (x/y/rz driven, z/rx/ry free)
        # + the real wheel/steer joints (BEHAVIOR-1K leaves them free).
        self.passive_idx = sorted(
            i for i in range(n_dof)
            if any(h in names[i] for h in _PASSIVE_NAME_HINTS))
        # driver holds ONLY the torso (mobile-base contract). arms/grippers
        # are a separate interface -> NOT held here (probe scaffolds them).
        self.hold_idx = [
            i for i in range(n_dof)
            if i not in base6 and i not in set(self.passive_idx)
            and any(h in names[i] for h in _HOLD_NAME_HINTS)]
        self.arm_idx = [
            i for i in range(n_dof)
            if i not in base6 and i not in set(self.passive_idx)
            and i not in set(self.hold_idx)]   # exposed for the probe

        # locate base_link prim for the TRUE world pose (never trust the
        # velocity-integrated virtual-joint scalars -- those drift).
        for pr in self.stage.Traverse():
            if pr.GetName() == "base_link" and \
                    pr.GetPath().pathString.startswith(self.robot_prim):
                self._blink = pr
                break
        if self._blink is None:
            raise RuntimeError(
                f"[holonomic] base_link not found under {self.robot_prim}")
        self._xc = UsdGeom.XformCache()

        # snapshot the loaded pose; hold the non-base body there.
        js = self.robot.get_joints_state()
        if js is None:
            raise RuntimeError(
                "[holonomic] setup() called before joints state is live "
                "(step the sim once after play() first)")
        q = np.asarray(js.positions, dtype=np.float32)
        hold_ids = np.asarray(self.hold_idx, dtype=np.int32)
        self.q_hold = q[hold_ids].copy()

        # --- (a) TORSO hold: one-time STIFF position drive (genie_sim /
        #     G2-style: kp~1e7, kd~1e5). The torso rides the base; soft
        #     gains let it sag/sway under base accel -> COM slosh ->
        #     excites the passive rx/ry tip. A stiff PD makes it a quasi-
        #     rigid body that just translates/rotates with base_link.
        #     Target set ONCE; NO per-step set_joint_positions (that was
        #     the regression that stomped the base drive). -------------- #
        def _kd(kp):
            return 2.0 * math.sqrt(kp)
        self.av.set_gains(
            kps=np.full((1, len(hold_ids)), self.hold_kp, dtype=np.float32),
            kds=np.full((1, len(hold_ids)), self.hold_kd, dtype=np.float32),
            joint_indices=hold_ids)
        for di in self.hold_idx:
            try:
                self.av.switch_dof_control_mode("position", di)
            except Exception as exc:  # noqa: BLE001
                print(f"[holonomic] hold mode switch dof {di}: {exc}")
        self.av.set_joint_position_targets(
            self.q_hold.reshape(1, -1), joint_indices=hold_ids)

        # --- (b) base x/y/rz: POSITION drive. Pure velocity drive (kp=0)
        #     has NO position stiffness -> when the welded (non-rolling)
        #     sphere wheels SKID-SCRUB during a turn the zero-stiffness x/y
        #     joints get shoved sideways and rz can't hold heading. Need
        #     position authority (force = kp*err, endpoint doesn't vanish).
        #     This is NOT the regressed config: the regression was
        #     absolute-setpoint-chase + per-step set_joint_positions; here
        #     targets are q_to_action incremental + RATE-LIMITED and the
        #     body is NEVER kinematically reset. z/rx/ry: NOT touched
        #     -> passive (rests on boundingSphere wheels). ------------- #
        bidx = np.asarray(self.base_idx, dtype=np.int32)
        self.av.set_gains(
            kps=np.full((1, 3), self.base_kp, dtype=np.float32),
            kds=np.full((1, 3), _kd(self.base_kp), dtype=np.float32),
            joint_indices=bidx)
        for di in self.base_idx:
            try:
                self.av.switch_dof_control_mode("position", di)
            except Exception as exc:  # noqa: BLE001
                print(f"[holonomic] base mode switch dof {di}: {exc}")
        try:
            self.av.set_max_efforts(
                np.full((1, 3), self.max_effort, dtype=np.float32),
                joint_indices=bidx)
        except Exception as exc:  # noqa: BLE001
            print(f"[holonomic] set_max_efforts: {exc}")
        # hold station at the loaded base pose until first drive_to()
        self._q_base0 = np.asarray(
            q[np.asarray(self.base_idx)], dtype=np.float32).copy()
        self.av.set_joint_position_targets(
            self._q_base0.reshape(1, -1), joint_indices=bidx)

        self._ready = True
        print(f"[holonomic] ready. POSITION base(x,y,rz)={self.base_idx}  "
              f"STIFF torso hold(kp={self.hold_kp:g},kd={self.hold_kd:g})="
              f"{self.hold_idx}  arm_idx(NOT driver's -> probe scaffolds)="
              f"{self.arm_idx}  passive(z/rx/ry+wheels)="
              f"{len(self.passive_idx)}")

    # ------------------------------------------------------------------ #
    # TRUE base pose = base_link world transform (x, y, z, yaw). Robot
    # local +X is the heading; matrix row 0 is local X in world (pxr
    # row-vector convention).
    # ------------------------------------------------------------------ #
    def base_pose(self):
        if self._blink is None:
            return float("nan"), float("nan"), float("nan"), float("nan")
        self._xc.Clear()
        m = self._xc.GetLocalToWorldTransform(self._blink)
        t = m.ExtractTranslation()
        yaw = math.atan2(float(m[0][1]), float(m[0][0]))
        return float(t[0]), float(t[1]), float(t[2]), yaw

    def base_pose_full(self):
        """(x, y, z, yaw, tip_deg) -- tip = angle between base_link local
        +Z and world +Z. z/rx/ry are passive, so a non-trivial tip means
        the 3-wheel static-stability assumption broke (it's falling over),
        which is exactly what this probe is meant to catch."""
        if self._blink is None:
            return (float("nan"),) * 5
        self._xc.Clear()
        m = self._xc.GetLocalToWorldTransform(self._blink)
        t = m.ExtractTranslation()
        yaw = math.atan2(float(m[0][1]), float(m[0][0]))
        m22 = max(-1.0, min(1.0, float(m[2][2])))
        tip = math.degrees(math.acos(m22))
        return float(t[0]), float(t[1]), float(t[2]), yaw, tip

    # ------------------------------------------------------------------ #
    # drive_to: WORLD-absolute goal [gx, gy, grz] -> rate-limited
    # q_to_action position targets on x/y/rz.
    #
    # BEHAVIOR-1K q_to_action (holonomic_base_robot.py:374-401): world
    # target -> base-LOCAL [dx, dy] + Δyaw = wrap(grz - cur_rz_joint); the
    # x/y joints are world-axis (anchored, pre-rz) so the world position
    # error IS the x/y joint-space error; rz delta is on the CURRENT rz
    # joint scalar. x/y/rz commanded SIMULTANEOUSLY (holonomic strafe --
    # no 90 deg in-place turn, so the welded wheels never get scrubbed).
    #
    # MANDATORY rate-limit: OmniGibson only ever feeds q_to_action a
    # curobo WAYPOINT sequence, never a bare endpoint. Feeding the raw
    # goal would let a single-step |Δyaw|>π wrap backwards / Δxy
    # overshoot. So we clamp the per-step target step (v_lin*dt, v_ang*dt
    # with v_ang*dt << π) -- this clamp IS our lightweight trajectory.
    # ------------------------------------------------------------------ #
    def drive_to(self, gx, gy, grz, dt):
        if not self._ready:
            raise RuntimeError("[holonomic] drive_to() before setup()")
        bx, by, _bz, _yaw = self.base_pose()
        js = self.robot.get_joints_state()
        p = js.positions
        cur_x = float(p[self.idx["x"]])
        cur_y = float(p[self.idx["y"]])
        cur_rz = float(p[self.idx["rz"]])

        # world position error == x/y joint error (world-axis joints)
        ex = float(gx) - bx
        ey = float(gy) - by
        # wrapped yaw delta on the CURRENT rz joint (q_to_action)
        eyaw = math.atan2(math.sin(float(grz) - cur_rz),
                          math.cos(float(grz) - cur_rz))

        # rate-limit -> the trajectory. step caps strictly < pi for yaw.
        dxy = self.v_lin * float(dt)
        dyaw = min(self.v_ang * float(dt), math.pi * 0.5)
        tx = cur_x + max(-dxy, min(dxy, ex))
        ty = cur_y + max(-dxy, min(dxy, ey))
        trz = cur_rz + max(-dyaw, min(dyaw, eyaw))

        self.av.set_joint_position_targets(
            np.array([[tx, ty, trz]], dtype=np.float32),
            joint_indices=np.asarray(self.base_idx, dtype=np.int32))

    def stop(self):
        """Hold station: command the CURRENT base joint pose as the
        position target (position drive then rejects skid-scrub)."""
        if not self._ready:
            return
        p = self.robot.get_joints_state().positions
        cur = np.array([[float(p[self.idx["x"]]),
                         float(p[self.idx["y"]]),
                         float(p[self.idx["rz"]])]], dtype=np.float32)
        self.av.set_joint_position_targets(
            cur, joint_indices=np.asarray(self.base_idx, dtype=np.int32))


__all__ = ["HolonomicBaseDriver", "MAX_LIN_VEL", "MAX_ANG_VEL", "MAX_EFFORT"]
