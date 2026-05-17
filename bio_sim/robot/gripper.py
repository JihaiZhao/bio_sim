#
# Gripper = force-controlled friction grasp (genie_sim ParallelGripper),
# NOT a fixed joint. The fingers close under a capped force; PhysX contact
# friction holds the cube. cuRobo attach is planning-only (so the carried
# legs avoid self-collision with the payload) and does NOT touch physics.
#
# The actual drive control (zero stiffness + capped max force + closing
# velocity, reasserted every step) lives in G2Robot.set_gripper /
# _apply_gripper. This class just sequences attach/detach + bookkeeping.
#

from __future__ import annotations

import numpy as np

GRASP_GAP_TOL = 0.10  # m; loose sanity (force-closure tolerates cm-level
#                       arm error; only fail a wildly-missed reach)


class Gripper:
    def __init__(self, robot):
        self._robot = robot
        self._held = None

    @property
    def holding(self) -> str | None:
        return self._held

    def grasp(self, ctx, obj_name: str) -> bool:
        """Called AFTER the fingers have closed (force mode). Sanity-checks
        the reach, then cuRobo-attaches the payload for planning."""
        ee_p, _ = self._robot.ee_world_pose(ctx)
        obj_p, _ = ctx.scene.object_pose(obj_name)
        gap = float(np.linalg.norm(np.asarray(ee_p) - np.asarray(obj_p)))
        print(f"[gripper] grasp gap |ee - object| = {gap:.4f} m")
        # Did the fingers actually move? open target = GRIP_OPEN_Q (0.8);
        # if idx81 is still ~0.8 the drive isn't actuating; if ~0 it closed.
        gi, gpos = self._robot.gripper_joint_state()
        ee_w, _ = self._robot.ee_world_pose(ctx)
        gpos_s = ("n/a" if gpos is None
                  else "[" + ", ".join(f"{p:.4f}" for p in gpos) + "]")
        print(f"[gripper] DIAG cmd-joint(idx81) dof={gi} pos={gpos_s} "
              f"(open~0.8, closed~0) "
              f"ee_world={np.round(np.asarray(ee_w),3).tolist()} "
              f"obj_world={np.round(np.asarray(obj_p),3).tolist()}")
        if gap > GRASP_GAP_TOL:
            print(f"[gripper] gap exceeds {GRASP_GAP_TOL} m -> reach missed")
            return False

        # cuRobo payload-aware planning for the carried legs (planning only;
        # no physics constraint -- the grip is held by finger friction).
        cu_js = self._robot.read_cu_js()
        p_w, q_w = ctx.scene.object_pose(obj_name)
        p_b, q_b = self._robot.base.world_to_base(p_w, q_w)
        dims = ctx.scene.object_dims(obj_name)
        self._robot.arm.attach_payload(
            cu_js, obj_name, dims,
            [float(p_b[0]), float(p_b[1]), float(p_b[2]),
             float(q_b[0]), float(q_b[1]), float(q_b[2]), float(q_b[3])],
        )
        # Lock the fingers as a stiff vice at the contacted aperture so the
        # carry (base accel, arm motion) can't ratchet them shut and eject
        # the thin box. Still pure friction -- no joint to the object.
        self._robot.clamp_hold()
        self._held = obj_name
        print(f"[gripper] grasped {obj_name} (friction vice-hold; cuRobo attached)")
        return True

    def release(self, ctx) -> None:
        if self._held is None:
            return
        # Carry-integrity check: if the cube slipped out during the carry,
        # |ee - object| will be large / object z near the floor.
        ee_p, _ = self._robot.ee_world_pose(ctx)
        obj_p, _ = ctx.scene.object_pose(self._held)
        d = float(np.linalg.norm(np.asarray(ee_p) - np.asarray(obj_p)))
        print(f"[gripper] at release: |ee-object|={d:.4f} m  "
              f"object_z={float(obj_p[2]):.3f} (held OK if small & z>0.5)")
        self._robot.arm.detach_payload()
        self._robot.set_gripper(close=False)  # open the fingers (position)
        print(f"[gripper] released {self._held}")
        self._held = None
