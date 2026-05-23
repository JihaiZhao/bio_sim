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

# Close-window length: how long set_gripper(close=True) runs in friction
# mode before gripper.grasp() is invoked (which then runs clamp_hold + the
# optional FixedJoint weld). Both modes share this -- the weld in assist
# mode still needs the fingers to physically reach + press on the cube
# first, because _weld captures the LIVE relative pose: weld too early and
# the relative pose is "plate at rest, fingers wide open above" -> through
# the carry plate floats at that offset under the still-opening fingers,
# looking exactly like a slip. Travel from q=0.8 (open) to q~0.05 (cube
# contact for a 14.6 mm plate) at GRIP_CLOSE_VEL = -0.6 rad/s takes ~75
# ticks; 110 leaves a comfortable margin + a few ticks of force-build to
# stabilize the contact friction before clamp_hold latches it.
_CLOSE_TICKS_PHYSICS = 110
_CLOSE_TICKS_ASSIST = 15
# Release is now PHASED: phase 1 just opens the fingers (weld still holds
# the plate), phase 2 deletes the weld. _OPEN_TICKS controls how long we
# wait for the fingers to physically clear the plate before _unweld lets
# the plate fall -- otherwise a still-in-contact finger PD-opening swing
# can flick the plate sideways on release.
# Finger travel from clamp_hold target (~0.02) to clearly-off-plate
# (~0.15) is ~0.13 rad; at terminal vel 0.3 rad/s = ~26 ticks. 60 leaves
# margin + lets the plate settle a touch before unweld.
_RELEASE_OPEN_TICKS = 25
# Total Release skill duration: open + unweld + finger-fully-open + plate-settle.
# Sized so the fingers reach GRIP_OPEN_Q (0.8) BEFORE retreat starts -- otherwise
# the arm lifts while the fingers are still PD-opening, which looks like a
# second "open" mid-retreat. Finger travel ~0.05 -> 0.8 rad at terminal vel
# 0.3 rad/s = ~150 ticks; 180 leaves margin so retreat starts on fully-open.
_SETTLE_TICKS = 40


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
        # Diagnostic: snapshot BOTH the driven (outer) and the mimic (inner)
        # finger joint state RIGHT before set_gripper(close=True). Lets us
        # compare first-run vs after-reset: if the reset failed to snap
        # inner to -outer or to zero finger velocities, the diff shows up
        # here and explains why the same _CLOSE_TICKS lands the fingers in
        # different places on run 1 vs run 2 ("premature close" symptom).
        try:
            r = ctx.robot._robot
            sjs = r.get_joints_state()
            keys = [n for n in r.dof_names
                    if "gripper_" in n
                    and ("_outer_joint1" in n or "_inner_joint1" in n)]
            parts = []
            for n in keys:
                i = r.get_dof_index(n)
                parts.append(f"{n[-25:]} p={float(sjs.positions[i]):+.4f}"
                             f" v={float(sjs.velocities[i]):+.4f}")
            print("[grasp.start DIAG]\n  " + "\n  ".join(parts))
        except Exception as e:  # noqa: BLE001
            print(f"[grasp.start DIAG] failed: {e}")
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
        self._unwelded = False
        self.name = f"Release({obj})"

    def start(self, ctx: SkillContext) -> None:
        self._t = 0
        self._unwelded = False
        # Phase 1: start opening the fingers WHILE the weld still holds the
        # plate. Plate is rigid -> the fingers can swing off it without
        # dragging it laterally. The weld stays in place.
        ctx.robot.gripper.release_open(ctx)
        ctx.blackboard.pop("held", None)

    def update(self, ctx: SkillContext) -> Status:
        self._t += 1
        # Phase 2: once the fingers have cleared the plate, delete the
        # FixedJoint and detach the cuRobo payload. Plate becomes a free
        # dynamic body and falls the remaining clearance onto the surface.
        if not self._unwelded and self._t >= _RELEASE_OPEN_TICKS:
            ctx.robot.gripper.release_unweld(ctx)
            self._unwelded = True
        return Status.SUCCESS if self._t >= _SETTLE_TICKS else Status.RUNNING
