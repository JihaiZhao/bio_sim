#
# Base control. Reuses the validated swerve IK + kinematic base from
# curobo_robot/swerve_base.py (imported, NOT reimplemented) and adds a
# go-to-pose P-controller so a scripted task can navigate without teleop.
#

from __future__ import annotations

import math
import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_PROJECT_ROOT, "curobo_robot"))

from swerve_base import (  # noqa: E402  (path-dependent import)
    BASE_STAND_Z,
    MAX_ANG_SPEED,
    MAX_LIN_SPEED,
    KeyboardTeleop,
    SwerveBaseController,
    _wrap_pi,
)

# Arrival tolerances for navigate-to-pose.
POS_TOL = 0.05      # m
YAW_TOL = 0.0175    # rad (~1 deg). Loose 0.05 (~3 deg) let FaceYaw exit
#                     while the base was still rotating; tightened so the
#                     residual yaw mismatch when DriveStraight takes over
#                     is sub-degree (visually invisible).
# P gains mapping pose error -> body twist (then capped by swerve limits).
KP_LIN = 0.8
KP_ANG = 1.2
# Acceleration limits for the commanded base twist. The base is KINEMATIC
# (root teleported by integrating the twist), so an unramped P-controller
# jumps 0 -> MAX_LIN_SPEED in ONE step = an infinite-accel root jump that
# shears a friction-held payload out of the gripper. Slew the twist so the
# base accelerates like a real mobile base and the carried object follows.
MAX_LIN_ACCEL = 0.15   # m/s^2  (gentle: friction-carried payload follows)
MAX_ANG_ACCEL = 0.30   # rad/s^2
# While CARRYING a payload, scale the speed/accel caps WAY down. A kinematic
# in-place turn swings the far-extended gripper through a big arc; the cube
# only follows by fingertip friction, so even a slewed turn shears it out
# (confirmed: solid grip through lift, cube dropped the instant Navigate(B)
# started its turn). Angular is the culprit -> scaled hardest.
CARRY_LIN_SCALE = 0.70
CARRY_ANG_SCALE = 0.50


class NavController:
    """Wraps SwerveBaseController with a position controller.

    The base is kinematic: every sim step we feed a body twist to
    step_kinematic(), which integrates and teleports the root. With no nav
    goal the twist is zero (hold station). With a goal we P-control the
    world-frame error, rotated into the body frame.
    """

    # turn-drive-turn phases (a holonomic swerve base CAN strafe, but a cube
    # carried by friction shears out when the base translates sideways under
    # the extended arm -- and it just looks wrong. So we DRIVE it like a
    # differential base: rotate to face the travel bearing, drive straight
    # forward, then rotate to the final table-facing yaw).
    _TURN_BEARING, _DRIVE, _TURN_FINAL, _DONE = 0, 1, 2, 3

    def __init__(self, swerve: SwerveBaseController):
        self.swerve = swerve
        self._goal = None  # (x, y, yaw) in world
        self._v = np.zeros(3)  # last commanded (vx, vy, wz) for accel slew
        self._phase = self._DONE
        self._reverse = False   # pure back-up: drive vx<0, hold yaw, no turns
        self._carrying = False  # set by base_hold from blackboard['held']

    def set_carrying(self, flag: bool) -> None:
        self._carrying = bool(flag)

    def _caps(self):
        """Effective (lin_speed, ang_speed, lin_accel, ang_accel) -- scaled
        down while carrying so the kinematic turn doesn't shear the payload."""
        ls = CARRY_LIN_SCALE if self._carrying else 1.0
        as_ = CARRY_ANG_SCALE if self._carrying else 1.0
        return (MAX_LIN_SPEED * ls, MAX_ANG_SPEED * as_,
                MAX_LIN_ACCEL * ls, MAX_ANG_ACCEL * as_)

    def set_goal(self, x: float, y: float, yaw: float,
                 reverse: bool = False) -> None:
        """Drive to world (x, y, yaw). reverse=True => pure back-up: drive
        straight backward toward (x, y) WITHOUT turning to face it and
        WITHOUT a final yaw turn (yaw is held). Used so the robot retreats
        from the table still facing it (the cube is welded in assist mode,
        so a kinematic reverse is safe)."""
        self._goal = (float(x), float(y), float(yaw))
        self._reverse = bool(reverse)
        self._phase = self._DRIVE if reverse else self._TURN_BEARING

    def clear_goal(self) -> None:
        self._goal = None
        self._reverse = False
        self._phase = self._DONE

    def reset_pose(self, x: float = 0.0, y: float = 0.0,
                    yaw: float = 0.0) -> None:
        """Teleport the kinematic base back to a pose (used to reset the
        world to the validated start state for a repeat). The base is
        kinematic so this is just rewriting the integrated pose + root."""
        self.swerve._pose = [float(x), float(y), float(yaw)]
        self.swerve._z = BASE_STAND_Z
        quat = np.array([math.cos(yaw / 2.0), 0.0, 0.0,
                         math.sin(yaw / 2.0)], dtype=np.float32)
        pos = np.array([x, y, BASE_STAND_Z], dtype=np.float32)
        try:
            self.swerve._set_root_pose(pos, quat)
        except Exception as exc:  # noqa: BLE001
            print(f"[base] reset_pose set root pose failed: {exc}")
        self._goal = None
        self._reverse = False
        self._phase = self._DONE
        self._v[:] = 0.0

    def _twist_to_goal(self) -> tuple[float, float, float]:
        if self._goal is None:
            return 0.0, 0.0, 0.0
        gx, gy, gyaw = self._goal
        x, y, _z, yaw = self.swerve.base_pose()
        dx, dy = gx - x, gy - y
        dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)  # world heading from here to the goal
        lin_v, ang_v, _la, _aa = self._caps()

        # Pure back-up: never turn. Drive straight backward (vx < 0) until
        # within tol; yaw held (wz = 0). The goal point is directly behind
        # the nose by construction (the skill places it along -heading).
        if self._reverse:
            if dist < POS_TOL:
                self._phase = self._DONE
                return 0.0, 0.0, 0.0
            return -float(np.clip(KP_LIN * dist, 0.0, lin_v)), 0.0, 0.0

        # Phase 1: rotate IN PLACE to face the travel bearing. Skipped when
        # there's nothing to drive (e.g. the first nav to A is a pure
        # turn-to-face-the-table from the spawn pose).
        if self._phase == self._TURN_BEARING:
            if dist < POS_TOL:
                self._phase = self._TURN_FINAL
            else:
                eb = _wrap_pi(bearing - yaw)
                if abs(eb) < YAW_TOL:
                    self._phase = self._DRIVE
                else:
                    return 0.0, 0.0, float(np.clip(
                        KP_ANG * eb, -ang_v, ang_v))

        # Phase 2: drive straight forward (NO commanded vy -> no strafe),
        # gently steering wz so the nose stays on the goal as we approach.
        if self._phase == self._DRIVE:
            if dist < POS_TOL:
                self._phase = self._TURN_FINAL
            else:
                eb = _wrap_pi(bearing - yaw)
                vx = float(np.clip(KP_LIN * dist, 0.0, lin_v))
                wz = float(np.clip(KP_ANG * eb, -ang_v, ang_v))
                return vx, 0.0, wz

        # Phase 3: rotate in place to the final (table-facing) yaw.
        if self._phase == self._TURN_FINAL:
            edy = _wrap_pi(gyaw - yaw)
            if abs(edy) < YAW_TOL:
                self._phase = self._DONE
            else:
                return 0.0, 0.0, float(np.clip(
                    KP_ANG * edy, -ang_v, ang_v))

        return 0.0, 0.0, 0.0  # _DONE / settled

    def arrived(self) -> bool:
        # Phase-only gate (NO velocity-slew wait): the next skill's start()
        # may read a base_pose() whose yaw is still slewing the last 1-2
        # degrees, BUT the validated grasp / place geometry was tuned with
        # this exact behavior. Adding a `|self._v| < eps` gate changes the
        # actual base-end-pose by a few cm (the FaceYaw rotation-during-
        # drive curve goes away), which silently shifts the grasp target
        # OUT of the IK_OK region for edge poses like top-down. If that
        # premature-drive cosmetic bothers you, tighten YAW_TOL instead --
        # the visible mismatch shrinks linearly with it.
        return self._goal is None or self._phase == self._DONE

    def step(self, sim, sim_js) -> None:
        """Drive the base one sim step (P-control toward goal, else hold)."""
        cur_steer = (
            self.swerve.read_cur_steer(sim_js) if sim_js is not None else np.zeros(4)
        )
        tgt = np.array(self._twist_to_goal(), dtype=float)
        # Slew the twist toward the target under accel limits so the
        # kinematic base never steps velocity discontinuously (which would
        # fling a friction-held payload). dt-scaled per-step deltas.
        dt = float(sim.physics_dt)
        _lv, _av, la, aa = self._caps()
        dv_max = np.array([la, la, aa]) * dt
        self._v += np.clip(tgt - self._v, -dv_max, dv_max)
        vx, vy, wz = float(self._v[0]), float(self._v[1]), float(self._v[2])
        self.swerve.step_kinematic(vx, vy, wz, dt, cur_steer)

    # frame helpers used by the arm planner (delegate to swerve)
    def base_pose(self):
        return self.swerve.base_pose()

    def world_to_base(self, p_world, q_world):
        return self.swerve.world_to_base(p_world, q_world)


__all__ = [
    "NavController", "SwerveBaseController", "KeyboardTeleop", "BASE_STAND_Z",
]
