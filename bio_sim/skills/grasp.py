#
# Grasp / Release.
#
# Grasp closes the omnipicker fingers (drive the single command joint; USD
# mimic joints follow), waits for them to physically contact the cube, then
# hands off to Gripper.grasp: cuRobo payload attach + friction vice-hold,
# and (only when cfg grasp_mode == "assist") a UsdPhysics.FixedJoint weld
# so the kinematic base can't fling the cube. Release reverses it.
#

from __future__ import annotations

from .skill import Skill, SkillContext, Status

# Close-window length depends on the grasp mode:
#   physics: need a full close + clamp window so finger friction stabilizes
#            on the cube (it's the ONLY load path; the cube must be firmly
#            held when the lift starts).
#   assist : as soon as the FixedJoint is in, the weld takes the load and
#            any further finger close just squeezes a rigidly-bonded body.
#            BUT the friction-only seconds BEFORE the weld are where the
#            real damage happens -- the closing fingers slide / rotate the
#            (thin, lightly-friction-supported) plate around on the riser,
#            and that slid+tilted state is the one the weld then locks
#            forever. Shorten the window aggressively: fingers should just
#            barely touch before _weld snapshots the relative pose.
_CLOSE_TICKS_PHYSICS = 110
_CLOSE_TICKS_ASSIST = 20
_SETTLE_TICKS = 15    # post-release settle


class Grasp(Skill):
    def __init__(self, obj: str):
        self._obj = obj
        self._t = 0
        self._ok = None
        self._close_ticks = _CLOSE_TICKS_PHYSICS
        self.name = f"Grasp({obj})"

    def start(self, ctx: SkillContext) -> None:
        self._t = 0
        self._ok = None
        mode = getattr(ctx.robot.gripper, "mode", "physics")
        self._close_ticks = (_CLOSE_TICKS_ASSIST if mode == "assist"
                             else _CLOSE_TICKS_PHYSICS)
        # Force-close the fingers onto the cube (it's sitting on the table at
        # the reached grasp pose). No weld: friction holds it.
        ctx.robot.set_gripper(close=True)

    def update(self, ctx: SkillContext) -> Status:
        self._t += 1
        if self._t < self._close_ticks:
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
