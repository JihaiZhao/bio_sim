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
YAW_TOL = 0.05      # rad
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


class NavController:
    """Wraps SwerveBaseController with a position controller.

    The base is kinematic: every sim step we feed a body twist to
    step_kinematic(), which integrates and teleports the root. With no nav
    goal the twist is zero (hold station). With a goal we P-control the
    world-frame error, rotated into the body frame.
    """

    def __init__(self, swerve: SwerveBaseController):
        self.swerve = swerve
        self._goal = None  # (x, y, yaw) in world
        self._v = np.zeros(3)  # last commanded (vx, vy, wz) for accel slew

    def set_goal(self, x: float, y: float, yaw: float) -> None:
        self._goal = (float(x), float(y), float(yaw))

    def clear_goal(self) -> None:
        self._goal = None

    def _twist_to_goal(self) -> tuple[float, float, float]:
        if self._goal is None:
            return 0.0, 0.0, 0.0
        gx, gy, gyaw = self._goal
        x, y, _z, yaw = self.swerve.base_pose()
        dx, dy = gx - x, gy - y
        dyaw = _wrap_pi(gyaw - yaw)

        # world error -> body frame (rotate by -yaw)
        c, s = math.cos(-yaw), math.sin(-yaw)
        ex = c * dx - s * dy
        ey = s * dx + c * dy

        vx = float(np.clip(KP_LIN * ex, -MAX_LIN_SPEED, MAX_LIN_SPEED))
        vy = float(np.clip(KP_LIN * ey, -MAX_LIN_SPEED, MAX_LIN_SPEED))
        wz = float(np.clip(KP_ANG * dyaw, -MAX_ANG_SPEED, MAX_ANG_SPEED))
        return vx, vy, wz

    def arrived(self) -> bool:
        if self._goal is None:
            return True
        gx, gy, gyaw = self._goal
        x, y, _z, yaw = self.swerve.base_pose()
        return (
            math.hypot(gx - x, gy - y) < POS_TOL
            and abs(_wrap_pi(gyaw - yaw)) < YAW_TOL
        )

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
        dv_max = np.array([MAX_LIN_ACCEL, MAX_LIN_ACCEL, MAX_ANG_ACCEL]) * dt
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
