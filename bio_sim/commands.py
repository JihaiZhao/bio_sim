#
# Action bus. The functions here are the single source of truth for
# "what does pressing R / hitting POST /reset / etc. actually do"
# inside the sim. Both _ResetKey (carb keyboard, in sim/app.py) and
# bio_sim.server (FastAPI) call into here -- no duplication.
#
# All of these MUST run on the sim main thread (the same loop that
# drives world.step). The HTTP layer enqueues them; the sim loop
# drains the queue between physics steps via server.drain_queue().
# Calling these directly from a request handler will race PhysX and
# is wrong.
#

from __future__ import annotations


def reset_env(ctx, runner) -> dict:
    """Drop any held payload, snap arm/gripper/base back to their
    validated start poses, resample + reposition the manipulated
    object, and rewind the runner. The skill list itself is left
    intact -- only the runner index is reset."""
    try:
        ctx.robot.gripper.release(ctx)
    except Exception:
        pass  # nothing held; fine
    try:
        ctx.robot.reset_arm()
        ctx.robot.reset_gripper()
        ctx.robot.base.reset_pose(
            *getattr(ctx.robot, "base_start", (0.0, 0.0, 0.0))
        )
        name = ctx.scene.objects[0].name
        ctx.scene.randomize_plate()
        ctx.scene.reset_object(name)
        ctx.blackboard.pop("held", None)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    runner.restart()
    return {"ok": True}


def status(ctx, runner) -> dict:
    """Snapshot of runner + sim state. Cheap; safe to poll at any rate."""
    current = runner.current
    return {
        "step_index": int(ctx.world.step_index),
        "skill": current.name if current is not None else None,
        "skill_idx": int(runner.index),
        "skill_total": int(runner.total),
        "done": bool(runner.done),
        "failed": bool(runner.failed),
        "held": ctx.blackboard.get("held") is not None,
    }
