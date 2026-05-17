#
# Grasp / Release.
#
# Grasp now actually closes the omnipicker fingers (drive the single command
# joint; USD mimic joints follow), waits for them to physically contact the
# cube, THEN forms the fixed joint (a slip backup) + cuRobo payload attach.
# Release detaches, deletes the joint, and opens the fingers.
#

from __future__ import annotations

from .skill import Skill, SkillContext, Status

_CLOSE_TICKS = 110  # sim steps to close + clamp firmly before the lift
_SETTLE_TICKS = 8    # post-release settle


class Grasp(Skill):
    def __init__(self, obj: str):
        self._obj = obj
        self._t = 0
        self._ok = None
        self.name = f"Grasp({obj})"

    def start(self, ctx: SkillContext) -> None:
        self._t = 0
        self._ok = None
        # Force-close the fingers onto the cube (it's sitting on the table at
        # the reached grasp pose). No weld: friction holds it.
        ctx.robot.set_gripper(close=True)

    def update(self, ctx: SkillContext) -> Status:
        self._t += 1
        if self._t < _CLOSE_TICKS:
            return Status.RUNNING  # fingers closing + gripping under force
        if self._ok is None:
            # fingers are gripping: sanity-check reach + cuRobo attach.
            self._ok = ctx.robot.gripper.grasp(ctx, self._obj)
            if self._ok:
                ctx.blackboard["held"] = self._obj
        return Status.SUCCESS if self._ok else Status.FAILURE


class Release(Skill):
    def __init__(self, obj: str):
        self._obj = obj
        self._t = 0
        self.name = f"Release({obj})"

    def start(self, ctx: SkillContext) -> None:
        self._t = 0
        ctx.robot.gripper.release(ctx)  # detach + delete joint + open fingers
        ctx.blackboard.pop("held", None)

    def update(self, ctx: SkillContext) -> Status:
        self._t += 1
        return Status.SUCCESS if self._t >= _SETTLE_TICKS else Status.RUNNING
