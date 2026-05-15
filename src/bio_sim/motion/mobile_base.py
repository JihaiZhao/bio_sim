"""SE(2) teleport-style mobile base controller.

Mirrors genie_sim's ``APICore._update_robot_base`` (api_core.py:1504): each
sim step we directly write the articulation root pose (and zero its velocity
so PhysX doesn't keep a residual). This decouples base motion from cuRobo
arm planning — cuRobo continues to plan in the (moving) base frame.

For the omnidirectional G2 chassis, "drive to a waypoint" is just linear
interpolation toward the target with separate caps on linear and angular
speed. Holonomic — no need to align heading first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _wrap_pi(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass
class MobileBaseState:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # yaw, radians
    z: float = 0.0      # base height (kept fixed; configured at construction)


class MobileBase:
    """Kinematic SE(2) teleport for an Isaac Lab Articulation root.

    The articulation must be spawned with ``fix_root_link=False`` — otherwise
    PhysX welds the root and the pose writes have no visible effect.
    """

    def __init__(
        self,
        articulation,
        x: float = 0.0,
        y: float = 0.0,
        theta: float = 0.0,
        z: float = 0.18,
        max_lin_speed: float = 0.5,
        max_ang_speed: float = 1.0,
        device: str | None = None,
    ) -> None:
        self.articulation = articulation
        self.state = MobileBaseState(x=x, y=y, theta=theta, z=z)
        self.max_lin_speed = max_lin_speed
        self.max_ang_speed = max_ang_speed
        self._device = device or str(articulation.data.root_pos_w.device)
        # Pre-allocated tensors for write_*_to_sim (shape (1, 7) and (1, 6)).
        self._pose_buf = torch.zeros((1, 7), device=self._device, dtype=torch.float32)
        self._vel_buf = torch.zeros((1, 6), device=self._device, dtype=torch.float32)
        self.write_to_sim()

    @staticmethod
    def _yaw_to_quat_wxyz(theta: float) -> tuple[float, float, float, float]:
        """yaw → (w, x, y, z) about world Z."""
        half = 0.5 * theta
        return (math.cos(half), 0.0, 0.0, math.sin(half))

    def set_pose(self, x: float, y: float, theta: float) -> None:
        """Snap to (x, y, theta) immediately."""
        self.state.x = x
        self.state.y = y
        self.state.theta = _wrap_pi(theta)
        self.write_to_sim()

    def write_to_sim(self) -> None:
        """Push the current state to the articulation (call every sim step)."""
        s = self.state
        qw, qx, qy, qz = self._yaw_to_quat_wxyz(s.theta)
        self._pose_buf[0, 0] = s.x
        self._pose_buf[0, 1] = s.y
        self._pose_buf[0, 2] = s.z
        self._pose_buf[0, 3] = qw
        self._pose_buf[0, 4] = qx
        self._pose_buf[0, 5] = qy
        self._pose_buf[0, 6] = qz
        self.articulation.write_root_pose_to_sim(self._pose_buf)
        # Zero root velocity so the chassis doesn't drift under gravity or
        # accumulated PhysX residuals — this is what makes the base behave
        # kinematically despite fix_root_link=False.
        self.articulation.write_root_velocity_to_sim(self._vel_buf)

    def step_toward(
        self,
        target_x: float,
        target_y: float,
        target_theta: float,
        dt: float,
        pos_tol: float = 0.01,
        ang_tol: float = 0.02,
    ) -> bool:
        """Advance the state one ``dt`` toward the target. Returns True if reached.

        Independent caps on linear and angular speed; no coupling. Linear
        motion is clamped to ``max_lin_speed * dt``; angular to
        ``max_ang_speed * dt``.
        """
        dx = target_x - self.state.x
        dy = target_y - self.state.y
        dtheta = _wrap_pi(target_theta - self.state.theta)
        dist = math.hypot(dx, dy)
        ang_err = abs(dtheta)

        if dist < pos_tol and ang_err < ang_tol:
            self.state.x = target_x
            self.state.y = target_y
            self.state.theta = _wrap_pi(target_theta)
            self.write_to_sim()
            return True

        max_lin_step = self.max_lin_speed * dt
        if dist > 1e-9:
            lin_step = min(dist, max_lin_step)
            self.state.x += lin_step * dx / dist
            self.state.y += lin_step * dy / dist

        max_ang_step = self.max_ang_speed * dt
        if ang_err > 1e-9:
            ang_step = min(ang_err, max_ang_step) * (1.0 if dtheta > 0 else -1.0)
            self.state.theta = _wrap_pi(self.state.theta + ang_step)

        self.write_to_sim()
        return False
