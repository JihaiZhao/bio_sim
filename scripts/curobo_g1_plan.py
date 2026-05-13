"""Headless cuRobo smoke test for Agibot G1 (no Isaac Sim required).

Validates:
  1. agibot_g1_curobo.yml loads under cuRobo V2 with our path-substituting loader.
  2. MotionPlanner warms up and plans a pose target for the right gripper.
  3. The retract config in the YAML is reachable / IK seeds find a solution.

Use this before adding Isaac Sim — if cuRobo can't plan, no sim integration
will help.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from bio_sim.motion import planner as p  # noqa: E402

G1_CUROBO_YAML = REPO_ROOT / "src/bio_sim/robot/agibot_g1_curobo.yml"


def main() -> int:
    print(f"loading {G1_CUROBO_YAML.relative_to(REPO_ROOT)}")
    # Right-arm-only smoke test: override YAML's dual-arm tool_frames.
    handle = p.build(
        G1_CUROBO_YAML,
        tool_frames=["gripper_r_center_link"],
        num_ik_seeds=32,
        warmup_iterations=2,
    )
    print(f"  joint_names ({len(handle.joint_names)}): {handle.joint_names[:6]} ...")
    print(f"  tool_frames: {handle.tool_frames}")

    # Forward-kinematics the default pose to find a reachable anchor for the
    # right gripper, then offset slightly.
    fk = handle.planner.compute_kinematics(handle.default_joint_state)
    home_pose = fk.tool_poses
    home_pos = home_pose.position.squeeze().detach().cpu().numpy()
    home_quat = home_pose.quaternion.squeeze().detach().cpu().numpy()
    print(f"  home tool pos: {home_pos.tolist()}")
    print(f"  home tool quat: {home_quat.tolist()}")

    # Target: a clearly different right-gripper pose — pull the arm forward
    # and toward the body centerline.
    target_pos = (0.35, -0.30, 0.85)
    target_quat = tuple(float(x) for x in home_quat)
    print(f"target_pos={target_pos} target_quat={target_quat}")

    result = p.plan_arm_pose(
        handle,
        target_position=target_pos,
        target_quaternion=target_quat,
    )

    if result is None or not result.success.any().item():
        print("plan FAILED")
        if result is not None and hasattr(result, "status"):
            print(f"  status: {result.status}")
        return 1

    positions, joint_names, dt = p.trajectory_to_numpy(result, handle.planner)
    print(f"trajectory shape: {positions.shape}, dt={dt:.4f}s")
    n_steps = positions.shape[0] if positions.ndim >= 2 else 1
    print(f"plan succeeded: {n_steps} waypoints, duration={n_steps * dt:.2f}s")
    if positions.ndim >= 2:
        arm_r_idxs = [(j, joint_names.index(j)) for j in joint_names if "arm_r" in j]
        q0 = [f"{float(positions[0, i]):+.3f}" for _, i in arm_r_idxs[:3]]
        qN = [f"{float(positions[-1, i]):+.3f}" for _, i in arm_r_idxs[:3]]
        print(f"  q0[arm_r joints 1-3]: {q0}")
        print(f"  qN[arm_r joints 1-3]: {qN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
