#
# ArmPlanner: thin wrapper over cuRobo MotionGen, ported faithfully from the
# validated g2_motion_gen_reacher.py.
#
# G2 is a 30+ DOF dual-arm robot; the config locks torso/head/passive-gripper
# joints leaving two 7-DOF arms, and declares BOTH gripper centers as target
# links. plan(): the active arm tracks a world goal (transformed into the
# current kinematic-base frame) while the idle arm's center link is pinned to
# its current FK so it holds station instead of drifting.
#

from __future__ import annotations

from typing import Optional

import numpy as np


class ArmPlanner:
    def __init__(self, robot_cfg: dict, world_cfg, ee_link: str, idle_link: str,
                 reactive: bool = False):
        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.types.base import TensorDeviceType
        from curobo.wrap.reacher.motion_gen import (
            MotionGen,
            MotionGenConfig,
            MotionGenPlanConfig,
        )

        self.tensor_args = TensorDeviceType()
        self.ee_link = ee_link
        self.idle_link = idle_link
        self._reactive = reactive

        n_obstacle_cuboids = 30
        n_obstacle_mesh = 100

        if reactive:
            trajopt_tsteps, trajopt_dt = 40, 0.04
            optimize_dt, max_attempts = False, 1
            trim_steps, interpolation_dt = [1, None], 0.04
            enable_finetune_trajopt = False
        else:
            trajopt_tsteps, trajopt_dt = 32, None
            optimize_dt, max_attempts = True, 4
            trim_steps, interpolation_dt = None, 0.05
            enable_finetune_trajopt = True

        cfg = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            self.tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            num_trajopt_seeds=12,
            num_graph_seeds=12,
            interpolation_dt=interpolation_dt,
            collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
            optimize_dt=optimize_dt,
            trajopt_dt=trajopt_dt,
            trajopt_tsteps=trajopt_tsteps,
            trim_steps=trim_steps,
        )
        self.motion_gen = MotionGen(cfg)
        self.plan_config = MotionGenPlanConfig(
            enable_graph=False,
            enable_graph_attempt=2,
            max_attempts=max_attempts,
            enable_finetune_trajopt=enable_finetune_trajopt,
            time_dilation_factor=1.0 if reactive else 0.5,
        )

    def warmup(self) -> None:
        if not self._reactive:
            print("[arm] cuRobo warmup (G2 is large; ~1 min)...")
            self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
        print("[arm] cuRobo ready")

    @property
    def joint_names(self):
        return self.motion_gen.kinematics.joint_names

    def sync_world(self, obstacles) -> None:
        self.motion_gen.update_world(obstacles)

    def compute_kinematics(self, cu_js):
        return self.motion_gen.compute_kinematics(cu_js)

    def retract_link_pose(self, retract_config, j_names, link: str):
        """FK of `link` at the retract config, in base_link frame.

        Used to derive a provably-reachable validation scene: the grasp
        target IS this pose, so IK cannot fail on the nominal grasp.
        Returns (p[3], q[4]) numpy, quaternion (w, x, y, z).
        """
        from curobo.types.state import JointState

        js = JointState.from_position(
            self.tensor_args.to_device(retract_config).view(1, -1),
            joint_names=j_names,
        )
        kin = self.motion_gen.compute_kinematics(js)
        pose = kin.ee_pose if link == self.ee_link else kin.link_poses[link]
        p = pose.position.cpu().numpy().ravel().astype(np.float64)
        q = pose.quaternion.cpu().numpy().ravel().astype(np.float64)
        return p, q

    def fk_link_pose(self, cu_js, link: str):
        kin = self.motion_gen.compute_kinematics(cu_js)
        if link == self.ee_link:
            return kin.ee_pose
        return kin.link_poses[link]

    def ik_ok(self, p, q) -> bool:
        """True if `p,q` (base_link frame) is IK-feasible for the active arm.

        Used at scene setup to pick a provably reachable validation anchor
        (FK of the retract config alone can sit on a reach boundary, so an
        approach offset above it may be infeasible).
        """
        from curobo.types.math import Pose

        pose = Pose(
            position=self.tensor_args.to_device(np.asarray(p, dtype=np.float64)),
            quaternion=self.tensor_args.to_device(np.asarray(q, dtype=np.float64)),
        )
        res = self.motion_gen.solve_ik(pose)
        try:
            return bool(res.success.item())
        except Exception:
            return bool(np.any(res.success.cpu().numpy()))

    def plan_to_world_pose(self, cu_js, p_world, q_world, base) -> Optional[object]:
        """Plan the active (ee_link) arm to a world pose; idle arm pinned.

        Returns an interpolated, full JointState ready to stream to Isaac, or
        None if planning failed.
        """
        from curobo.types.math import Pose

        cu_js = cu_js.unsqueeze(0)

        # idle arm: pin its center link to current FK so it holds station.
        cur_kin = self.motion_gen.compute_kinematics(cu_js)
        idle_pose = cur_kin.link_poses[self.idle_link].clone()

        # cuRobo plans in base_link frame; goal is world -> transform into the
        # current kinematic-base frame.
        p_b, q_b = base.world_to_base(p_world, q_world)
        goal_pose = Pose(
            position=self.tensor_args.to_device(np.asarray(p_b)),
            quaternion=self.tensor_args.to_device(np.asarray(q_b)),
        )
        link_poses = {self.idle_link: idle_pose}

        result = self.motion_gen.plan_single(
            cu_js, goal_pose, self.plan_config, link_poses=link_poses
        )
        if not result.success.item():
            print(f"[arm] plan failed: {result.status}")
            return None

        plan = self.motion_gen.get_full_js(result.get_interpolated_plan())
        return plan
