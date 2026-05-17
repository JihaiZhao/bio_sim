#
# Grasp / Release: kinematic attach/detach (see robot/gripper.py for why the
# G2 omnipicker fingers are not actuated in the MVP). A few settle ticks let
# the attach/detach register before the next skill plans.
#

from __future__ import annotations

from .skill import Skill, SkillContext, Status

_SETTLE_TICKS = 5


class Grasp(Skill):
    def __init__(self, obj: str):
        self._obj = obj
        self._t = 0
        self.name = f"Grasp({obj})"

    def start(self, ctx: SkillContext) -> None:
        self._t = 0
        ctx.robot.gripper.grasp(ctx, self._obj)
        ctx.blackboard["held"] = self._obj

    def update(self, ctx: SkillContext) -> Status:
        self._t += 1
        return Status.SUCCESS if self._t >= _SETTLE_TICKS else Status.RUNNING


class Release(Skill):
    def __init__(self, obj: str, place_xyz=None):
        self._obj = obj
        self._place_xyz = place_xyz
        self._t = 0
        self.name = f"Release({obj})"

    def start(self, ctx: SkillContext) -> None:
        self._t = 0
        ctx.robot.gripper.release(ctx, place_xyz=self._place_xyz)
        ctx.blackboard.pop("held", None)

    def update(self, ctx: SkillContext) -> Status:
        self._t += 1
        return Status.SUCCESS if self._t >= _SETTLE_TICKS else Status.RUNNING
