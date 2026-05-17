#
# MoveArmTo: plan the active arm to a world pose, then stream the trajectory.
#
# Pose source is a callable resolved at start() so it can track a live object
# (e.g. grasp = object's current pose, pre-grasp = object + z offset).
#
# Lifecycle: wait until the robot is quasi-static -> plan once -> stream one
# interpolated waypoint per tick until exhausted.
#

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from .skill import Skill, SkillContext, Status

PoseFn = Callable[[SkillContext], Tuple[np.ndarray, np.ndarray]]


class MoveArmTo(Skill):
    def __init__(self, pose_fn: PoseFn, label: str = "pose",
                 kinematic: bool = False):
        # kinematic=True: hard joint streaming (mm-accurate, for centering
        # the grasp). kinematic=False: PD streaming (smooth, so a friction-
        # held object follows the hand during the carry).
        self._pose_fn = pose_fn
        self._kinematic = kinematic
        self._planned = False
        self._failed = False
        self.name = f"MoveArmTo({label})"

    # ---- common pose providers ---------------------------------------
    @staticmethod
    def grasp_pose(name: str, dz: float = 0.0):
        """Object position (+dz) with the scene's reachable grasp orientation
        (the object spawns at identity quaternion, which may be infeasible)."""
        def fn(ctx):
            p, _ = ctx.scene.object_pose(name)
            p = np.array(p, dtype=np.float64)
            p[2] += dz
            return p, np.asarray(ctx.scene.grasp_q, dtype=np.float64)
        return fn

    @staticmethod
    def scene_place(dz: float = 0.0):
        """Place point from the scene (mirrors the grasp in B's frame)."""
        def fn(ctx):
            p = np.array(ctx.scene.place_xyz, dtype=np.float64)
            p[2] += dz
            return p, np.asarray(ctx.scene.grasp_q, dtype=np.float64)
        return fn

    @staticmethod
    def world(p, q):
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        return lambda ctx: (p, q)

    @staticmethod
    def gripper_offset(dz: float):
        """Target = current gripper world pose shifted by dz (lift/retreat)."""
        def fn(ctx):
            p, q = ctx.robot.ee_world_pose(ctx)
            p = np.array(p, dtype=np.float64)
            p[2] += dz
            return p, np.asarray(q, dtype=np.float64)
        return fn

    # ---- lifecycle ----------------------------------------------------
    def start(self, ctx: SkillContext) -> None:
        ctx.robot.set_arm_mode(kinematic=self._kinematic)
        self._planned = False
        self._failed = False
        self._streamed = False
        self._settle = 0

    def update(self, ctx: SkillContext) -> Status:
        if self._failed:
            return Status.FAILURE

        if not self._planned:
            if not ctx.robot.robot_static():
                return Status.RUNNING  # wait for the base/arm to settle
            self._goal = self._pose_fn(ctx)
            if not ctx.robot.plan_arm_to(*self._goal):
                self._failed = True
                return Status.FAILURE
            self._planned = True
            return Status.RUNNING

        if not self._streamed:
            # stream the interpolated trajectory, one waypoint per tick
            if ctx.robot.advance_arm_plan(ctx.world):
                self._streamed = True
            return Status.RUNNING

        # SETTLE: the trajectory points were only *commanded*; the loaded PD
        # drive still has to physically converge. Low velocity alone is a
        # FALSE "arrived" signal -- the cuRobo trajectory ends on a ~zero-
        # velocity waypoint, so the instant streaming finishes the arm is
        # briefly slow while STILL at the start pose; declaring SUCCESS there
        # deferred the whole motion into the next skill (the carried cube
        # got jerked out during NavigateTo). Gate on the ee actually
        # reaching the goal position; keep a cap purely as a deadlock guard.
        self._settle += 1
        try:
            ee_p, _ = ctx.robot.ee_world_pose(ctx)
            dist = float(np.linalg.norm(
                np.asarray(ee_p) - np.asarray(self._goal[0])))
        except Exception:  # noqa: BLE001
            dist = 0.0
        arrived = dist < _ARRIVE_TOL and ctx.robot.robot_static()
        if arrived or self._settle > _SETTLE_CAP:
            if not arrived:
                print(f"[move_arm] {self.name} settle CAP hit "
                      f"(ee {dist:.3f} m from goal) -- proceeding anyway")
            return Status.SUCCESS
        return Status.RUNNING


_ARRIVE_TOL = 0.04   # m; ee must physically reach the goal, not just slow
_SETTLE_CAP = 1500   # deadlock guard only (loaded PD convergence is slow)
