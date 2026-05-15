"""4-wheel swerve-drive kinematics for Agibot G2.

Inverse kinematics: given a desired body-frame chassis velocity
``(vx, vy, omega_z)``, produce per-wheel ``(steer_angle, roll_speed)``
targets for the 4 wheels (FL, FR, RL, RR).

Each wheel at position ``(x_i, y_i)`` in chassis frame must move with
velocity ``(vx - omega*y_i, vy + omega*x_i)`` in chassis frame. The
steering angle is the direction of that velocity; the rolling speed is
its magnitude divided by wheel radius.

Companion to :class:`bio_sim.motion.mobile_base.MobileBase` (kinematic
SE(2) teleport, method A).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from bio_sim.robot.agibot_g2_drive_cfg import (
    WHEEL_POSITIONS_BODY,
    WHEEL_RADIUS,
    WHEEL_ROLL_JOINTS,
    WHEEL_STEER_JOINTS,
)

WHEEL_NAMES = ("fl", "fr", "rl", "rr")


@dataclass
class WheelCommand:
    steer_angle: float = 0.0   # rad, joint-space target
    roll_speed: float = 0.0    # rad/s, joint-space velocity target


@dataclass
class SwerveCommand:
    wheels: dict[str, WheelCommand] = field(
        default_factory=lambda: {n: WheelCommand() for n in WHEEL_NAMES}
    )


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def compute_swerve(
    vx_body: float,
    vy_body: float,
    omega: float,
    prev_steer: dict[str, float] | None = None,
    min_speed_for_resteer: float = 1e-3,
) -> SwerveCommand:
    """Inverse kinematics for the 4-wheel swerve module.

    Args:
        vx_body, vy_body: chassis-frame linear velocity (m/s).
        omega: yaw rate about chassis Z (rad/s).
        prev_steer: previous steering angles (rad), per wheel name. When the
            commanded wheel speed is below ``min_speed_for_resteer`` we
            HOLD the previous steering angle instead of letting it snap to
            atan2(0, 0) — avoids spinning steering rapidly when the chassis
            is nearly stopped.
        min_speed_for_resteer: m/s of wheel-frame velocity below which the
            previous steering angle is held.

    Wheel angles can also be flipped 180° for "shortest steer" — we pick
    whichever of (theta, theta+pi) is closer to ``prev_steer``, and negate
    roll speed accordingly. This avoids long re-steering when the desired
    direction reverses.
    """
    cmd = SwerveCommand()
    for name in WHEEL_NAMES:
        x_i, y_i = WHEEL_POSITIONS_BODY[name]
        wheel_vx = vx_body - omega * y_i
        wheel_vy = vy_body + omega * x_i
        speed = math.hypot(wheel_vx, wheel_vy)

        if speed < min_speed_for_resteer:
            # Below threshold: hold previous steering, zero roll.
            steer = (prev_steer or {}).get(name, 0.0)
            roll_speed = 0.0
        else:
            steer = math.atan2(wheel_vy, wheel_vx)
            roll_speed = speed / WHEEL_RADIUS
            # Shortest-steer: try the 180°-flipped solution and reverse roll
            # direction if it's a smaller angular delta from the previous
            # steering.
            prev = (prev_steer or {}).get(name)
            if prev is not None:
                d_direct = abs(_wrap_pi(steer - prev))
                steer_flipped = _wrap_pi(steer + math.pi)
                d_flipped = abs(_wrap_pi(steer_flipped - prev))
                if d_flipped < d_direct:
                    steer = steer_flipped
                    roll_speed = -roll_speed

        cmd.wheels[name] = WheelCommand(steer_angle=steer, roll_speed=roll_speed)
    return cmd


def waypoint_body_velocity(
    target_x: float,
    target_y: float,
    target_theta: float,
    cur_x: float,
    cur_y: float,
    cur_theta: float,
    *,
    max_lin: float = 0.4,
    max_ang: float = 0.8,
    k_lin: float = 1.5,
    k_ang: float = 1.5,
    pos_tol: float = 0.05,
    ang_tol: float = 0.05,
) -> tuple[float, float, float, bool]:
    """Proportional waypoint controller in chassis-body frame.

    Returns ``(vx_body, vy_body, omega, reached)``. Holonomic — no need to
    align heading first; the swerve module can sidle.
    """
    dx = target_x - cur_x
    dy = target_y - cur_y
    dtheta = _wrap_pi(target_theta - cur_theta)
    dist = math.hypot(dx, dy)

    if dist < pos_tol and abs(dtheta) < ang_tol:
        return 0.0, 0.0, 0.0, True

    cos_t = math.cos(cur_theta)
    sin_t = math.sin(cur_theta)
    err_x_body = cos_t * dx + sin_t * dy
    err_y_body = -sin_t * dx + cos_t * dy

    vx_body = k_lin * err_x_body
    vy_body = k_lin * err_y_body
    mag = math.hypot(vx_body, vy_body)
    if mag > max_lin:
        scale = max_lin / mag
        vx_body *= scale
        vy_body *= scale

    omega = max(-max_ang, min(max_ang, k_ang * dtheta))
    return vx_body, vy_body, omega, False


__all__ = [
    "WHEEL_NAMES",
    "WHEEL_STEER_JOINTS",
    "WHEEL_ROLL_JOINTS",
    "WheelCommand",
    "SwerveCommand",
    "compute_swerve",
    "waypoint_body_velocity",
]
