#
# NavigateTo: drive the kinematic swerve base to a world (x, y, yaw).
#
# The base is actually stepped by SimApp.run -> robot.base_hold every sim
# step (P-control toward the active goal). This skill just sets the goal and
# polls arrival, so it composes cleanly with the shared step loop.
#

from __future__ import annotations

import math

from .skill import Skill, SkillContext, Status


class NavigateTo(Skill):
    def __init__(self, x: float, y: float, yaw: float = 0.0, marker: str | None = None):
        self._x, self._y, self._yaw = x, y, yaw
        self._marker = marker
        self.name = f"NavigateTo({marker or f'{x:.2f},{y:.2f}'})"

    @classmethod
    def to_marker(cls, scene_marker_name: str) -> "NavigateTo":
        # resolved at start() once the scene is in ctx
        s = cls(0.0, 0.0, 0.0, marker=scene_marker_name)
        return s

    def start(self, ctx: SkillContext) -> None:
        if self._marker is not None:
            self._x, self._y, self._yaw = ctx.scene.marker_pose(self._marker)
        ctx.robot.base.set_goal(self._x, self._y, self._yaw)

    def update(self, ctx: SkillContext) -> Status:
        if ctx.robot.base.arrived():
            ctx.robot.base.clear_goal()
            return Status.SUCCESS
        return Status.RUNNING


# --- relative, scripted base moves (computed from the LIVE base pose at
#     start, then handed to NavController as an absolute goal) ------------

class FaceYaw(Skill):
    """Rotate the base IN PLACE to an absolute world yaw (radians)."""

    def __init__(self, yaw: float, label: str | None = None):
        self._yaw = float(yaw)
        self.name = f"FaceYaw({label or f'{math.degrees(self._yaw):.0f}deg'})"

    def start(self, ctx: SkillContext) -> None:
        x, y, _z, _yaw = ctx.robot.base.base_pose()
        ctx.robot.base.set_goal(x, y, self._yaw)

    def update(self, ctx: SkillContext) -> Status:
        if ctx.robot.base.arrived():
            ctx.robot.base.clear_goal()
            return Status.SUCCESS
        return Status.RUNNING


class DriveStraight(Skill):
    """Drive the base `distance` metres along its CURRENT heading. The yaw
    is unchanged. reverse=True backs straight up (no turn-around) instead."""

    def __init__(self, distance: float, reverse: bool = False,
                 label: str | None = None):
        self._d = float(distance)
        self._rev = bool(reverse)
        tag = label or (f"{'back' if reverse else 'fwd'} {self._d:.2f}m")
        self.name = f"DriveStraight({tag})"

    def start(self, ctx: SkillContext) -> None:
        x, y, _z, yaw = ctx.robot.base.base_pose()
        s = -1.0 if self._rev else 1.0
        gx = x + s * self._d * math.cos(yaw)
        gy = y + s * self._d * math.sin(yaw)
        ctx.robot.base.set_goal(gx, gy, yaw, reverse=self._rev)

    def update(self, ctx: SkillContext) -> Status:
        if ctx.robot.base.arrived():
            ctx.robot.base.clear_goal()
            return Status.SUCCESS
        return Status.RUNNING
