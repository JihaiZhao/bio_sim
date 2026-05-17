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
    def __init__(self, pose_fn: PoseFn, label: str = "pose"):
        self._pose_fn = pose_fn
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
        self._planned = False
        self._failed = False

    def update(self, ctx: SkillContext) -> Status:
        if self._failed:
            return Status.FAILURE

        if not self._planned:
            if not ctx.robot.robot_static():
                return Status.RUNNING  # wait for the base/arm to settle
            p, q = self._pose_fn(ctx)
            if not ctx.robot.plan_arm_to(p, q):
                self._failed = True
                return Status.FAILURE
            self._planned = True
            return Status.RUNNING

        done = ctx.robot.advance_arm_plan(ctx.world)
        return Status.SUCCESS if done else Status.RUNNING
