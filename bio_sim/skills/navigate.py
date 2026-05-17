#
# NavigateTo: drive the kinematic swerve base to a world (x, y, yaw).
#
# The base is actually stepped by SimApp.run -> robot.base_hold every sim
# step (P-control toward the active goal). This skill just sets the goal and
# polls arrival, so it composes cleanly with the shared step loop.
#

from __future__ import annotations

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
