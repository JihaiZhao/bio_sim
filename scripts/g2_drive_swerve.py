"""Method (B): physics-driven swerve drive on G2.

Same UX as ``g2_drive_to_waypoint.py`` (method A), but the chassis moves
because the wheels push it via friction, not because we teleport the
root. Each sim step:

  1. Read current chassis pose from the articulation.
  2. Compute desired body-frame velocity (vx, vy, omega) toward the
     active waypoint.
  3. Inverse-kinematics through ``compute_swerve`` to per-wheel
     (steer, roll_speed).
  4. Write the steer angles and roll velocities to the joint targets.

If the chassis slips or oscillates, raise wheel/ground friction or lower
``--max-lin``. Defaults are intentionally conservative.

Pre-req: run ``scripts/patch_g2_drive_usd.py`` once to produce the drive
USD. (That patcher copies the working ``G2_omnipicker`` asset tree to
``G2_omnipicker_drive`` and adds wheel colliders + opens rolling-joint
limits — pure USD-API edits, no Isaac Sim required.)
"""

from __future__ import annotations

import argparse
import select
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Warp / Isaac Sim compat shim must import before AppLauncher.
from bio_sim.motion import planner as _p  # noqa: E402,F401

from isaaclab.app import AppLauncher  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-lin", type=float, default=0.4,
                        help="Body-frame linear speed cap (m/s).")
    parser.add_argument("--max-ang", type=float, default=0.8,
                        help="Yaw rate cap (rad/s).")
    parser.add_argument("--friction-static", type=float, default=1.0)
    parser.add_argument("--friction-dynamic", type=float, default=0.9)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _readline_nonblocking() -> str | None:
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().rstrip("\n")
    return None


def _parse_waypoint(line: str) -> tuple[float, float, float] | None:
    import math as _m
    parts = line.strip().split()
    if len(parts) not in (2, 3):
        return None
    try:
        x = float(parts[0])
        y = float(parts[1])
        theta = _m.radians(float(parts[2])) if len(parts) == 3 else 0.0
    except ValueError:
        return None
    return x, y, theta


def _quat_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    """yaw (about world Z) from a wxyz quaternion."""
    import math as _m
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return _m.atan2(siny_cosp, cosy_cosp)


def main() -> int:
    args = parse_args()
    launcher = AppLauncher(args)
    sim_app = launcher.app

    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

    from bio_sim.motion.swerve_drive import (
        WHEEL_NAMES, WHEEL_ROLL_JOINTS, WHEEL_STEER_JOINTS,
        compute_swerve, waypoint_body_velocity,
    )
    from bio_sim.robot.agibot_g2_drive_cfg import (
        AGIBOT_G2_DRIVE_CFG, AGIBOT_G2_DRIVE_USD_PATH,
    )

    if not AGIBOT_G2_DRIVE_USD_PATH.exists():
        print(f"missing drive USD: {AGIBOT_G2_DRIVE_USD_PATH}", file=sys.stderr)
        print("  Run: python scripts/patch_g2_drive_usd.py", file=sys.stderr)
        sim_app.close()
        return 1

    print("[1/3] building scene (G2 drive + ground + dome light)")
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0)
    scene_cfg.dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )
    # Ground with high friction so the wheels can grip.
    scene_cfg.ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=args.friction_static,
                dynamic_friction=args.friction_dynamic,
                restitution=0.0,
            ),
        ),
    )
    scene_cfg.robot = AGIBOT_G2_DRIVE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    sim_cfg = sim_utils.SimulationCfg(dt=0.005)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    dt = sim.get_physics_dt()
    sim_joint_names = list(robot.data.joint_names)
    print(f"[2/3] robot loaded: {len(sim_joint_names)} joints, dt={dt}s")

    # Locate steer / roll joint indices in the articulation's joint order.
    steer_idx = {n: sim_joint_names.index(WHEEL_STEER_JOINTS[n]) for n in WHEEL_NAMES}
    roll_idx = {n: sim_joint_names.index(WHEEL_ROLL_JOINTS[n]) for n in WHEEL_NAMES}

    # Hold the rest of the robot at its init pose; only wheel targets change.
    pos_target = robot.data.default_joint_pos.clone()
    vel_target = torch.zeros_like(robot.data.default_joint_vel)
    robot.set_joint_position_target(pos_target)
    robot.set_joint_velocity_target(vel_target)

    print("[3/3] ready.\n")
    print("  Type a target as: x y theta_deg     (e.g. '1.0 0.5 90')\n"
          "  Or just 'x y'                       (theta defaults to 0)\n"
          "  Type 'q' (or close the window) to quit.\n"
          f"  Speed caps: lin={args.max_lin} m/s, ang={args.max_ang} rad/s\n"
          f"  Ground friction: static={args.friction_static}, "
          f"dynamic={args.friction_dynamic}\n", flush=True)

    target: tuple[float, float, float] | None = None
    prev_steer: dict[str, float] = {n: 0.0 for n in WHEEL_NAMES}
    print("waypoint> ", end="", flush=True)

    try:
        while sim_app.is_running():
            # Current chassis pose.
            pos_w = robot.data.root_pos_w[0].detach().cpu().numpy()
            quat_w = robot.data.root_quat_w[0].detach().cpu().numpy()
            cur_x, cur_y = float(pos_w[0]), float(pos_w[1])
            cur_theta = _quat_to_yaw(*[float(q) for q in quat_w])

            if target is not None:
                vx, vy, w, reached = waypoint_body_velocity(
                    target[0], target[1], target[2],
                    cur_x, cur_y, cur_theta,
                    max_lin=args.max_lin, max_ang=args.max_ang,
                )
                if reached:
                    print(f"\n  [reached] pose=({cur_x:.3f}, {cur_y:.3f}, "
                          f"{cur_theta:.3f} rad)")
                    target = None
                    print("waypoint> ", end="", flush=True)
                    vx = vy = w = 0.0
            else:
                vx = vy = w = 0.0

            cmd = compute_swerve(vx, vy, w, prev_steer=prev_steer)
            for name in WHEEL_NAMES:
                wc = cmd.wheels[name]
                pos_target[0, steer_idx[name]] = wc.steer_angle
                vel_target[0, roll_idx[name]] = wc.roll_speed
                prev_steer[name] = wc.steer_angle
            robot.set_joint_position_target(pos_target)
            robot.set_joint_velocity_target(vel_target)

            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)

            line = _readline_nonblocking()
            if line is None:
                continue
            line = line.strip()
            if line.lower() in ("q", "quit", "exit"):
                print("  exiting.")
                break
            if not line:
                print("waypoint> ", end="", flush=True)
                continue
            parsed = _parse_waypoint(line)
            if parsed is None:
                print("  parse error — expected 'x y' or 'x y theta_deg'.")
                print("waypoint> ", end="", flush=True)
                continue
            target = parsed
            print(f"  driving from ({cur_x:.3f}, {cur_y:.3f}, "
                  f"{cur_theta:.3f}) → "
                  f"({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})...")
    finally:
        sim_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
