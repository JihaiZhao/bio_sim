#
# Gripper / object attachment.
#
# The G2 omnipicker's gripper joints are locked (passive) in the cuRobo
# config, so for the MVP "grasp" is a KINEMATIC attach, not finger closure:
#   - on grasp: freeze the object's rigid-body dynamics and remember its pose
#     relative to the gripper center link; each step it follows the gripper.
#   - on release: snap it down onto the place target and re-enable dynamics.
#
# TODO(payload-aware planning): also call motion_gen.attach_objects_to_robot()
# so cuRobo plans the transport leg with the object's collision volume. Until
# then, keep place/lift offsets generous.
#

from __future__ import annotations

import numpy as np

from curobo.types.math import Pose


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def _quat_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float64)


class Gripper:
    def __init__(self, robot):
        self._robot = robot
        self._held = None          # object name
        self._held_prim = None
        self._rel_p = None         # object pos in gripper frame
        self._rel_q = None         # object quat in gripper frame

    @property
    def holding(self) -> str | None:
        return self._held

    def grasp(self, ctx, obj_name: str) -> None:
        prim = ctx.scene.object_prim(obj_name)
        gp, gq = self._robot.ee_world_pose(ctx)          # gripper center, world
        op, oq = ctx.scene.object_pose(obj_name)

        # rel = inv(gripper) * object
        gq_c = _quat_conj(gq)
        dp = np.asarray(op) - np.asarray(gp)
        # rotate dp by conj(gq)
        rp = _quat_mul(_quat_mul(gq_c, np.array([0.0, *dp])), gq)[1:]
        self._rel_p = rp
        self._rel_q = _quat_mul(gq_c, oq)

        self._set_dynamics(prim, enabled=False)
        self._held, self._held_prim = obj_name, prim
        print(f"[gripper] grasped {obj_name}")

    def release(self, ctx, place_xyz=None) -> None:
        if self._held is None:
            return
        if place_xyz is not None:
            self._held_prim.set_world_pose(
                position=np.asarray(place_xyz, dtype=np.float32)
            )
        self._set_dynamics(self._held_prim, enabled=True)
        print(f"[gripper] released {self._held}")
        self._held = self._held_prim = None
        self._rel_p = self._rel_q = None

    def hold_step(self, ctx) -> None:
        """Keep the held object glued to the gripper. Call every sim step."""
        if self._held is None:
            return
        gp, gq = self._robot.ee_world_pose(ctx)
        # object_world = gripper * rel
        rp_w = _quat_mul(_quat_mul(gq, np.array([0.0, *self._rel_p])),
                         _quat_conj(gq))[1:]
        op = np.asarray(gp) + rp_w
        oq = _quat_mul(gq, self._rel_q)
        self._held_prim.set_world_pose(
            position=op.astype(np.float32), orientation=oq.astype(np.float32)
        )

    @staticmethod
    def _set_dynamics(prim, enabled: bool) -> None:
        # Best-effort: keep the cube from falling while carried.
        for attr in ("disable_rigid_body_physics", "enable_rigid_body_physics"):
            fn = getattr(prim, attr, None)
            if fn is None:
                continue
            if (attr.startswith("disable") and not enabled) or (
                attr.startswith("enable") and enabled
            ):
                try:
                    fn()
                    return
                except Exception:
                    pass
