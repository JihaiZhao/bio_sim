"""Interactive: drive G2's mobile base to user-specified waypoints.

Spawns G2 on a ground plane and prompts in the terminal for waypoint
targets. Each target is "x y theta_deg" (theta in degrees, world-frame
yaw). The base teleports there at capped linear/angular speeds via
:class:`bio_sim.motion.mobile_base.MobileBase` (SE(2) prim teleport,
mirroring genie_sim's _update_robot_base). Arms are held at the init pose.

Type ``q`` (or close the viewport) to quit.
"""

from __future__ import annotations

import argparse
import select
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from isaaclab.app import AppLauncher  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-lin", type=float, default=0.5, help="Max linear speed (m/s)."
    )
    parser.add_argument(
        "--max-ang", type=float, default=1.0, help="Max angular speed (rad/s)."
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _readline_nonblocking() -> str | None:
    """Return one line from stdin if available, else None. Strips newline."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().rstrip("\n")
    return None


def _parse_waypoint(line: str) -> tuple[float, float, float] | None:
    """Parse 'x y theta_deg' → (x, y, theta_rad). Returns None on error."""
    import math

    parts = line.strip().split()
    if len(parts) not in (2, 3):
        return None
    try:
        x = float(parts[0])
        y = float(parts[1])
        theta_deg = float(parts[2]) if len(parts) == 3 else 0.0
    except ValueError:
        return None
    return x, y, math.radians(theta_deg)


def main() -> int:
    args = parse_args()
    launcher = AppLauncher(args)
    sim_app = launcher.app

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

    from bio_sim.motion.mobile_base import MobileBase
    from bio_sim.robot.agibot_g2_cfg import AGIBOT_G2_CFG, AGIBOT_G2_USD_PATH

    if not AGIBOT_G2_USD_PATH.exists():
        print(f"missing G2 USD: {AGIBOT_G2_USD_PATH}", file=sys.stderr)
        sim_app.close()
        return 1

    print("[1/3] building scene (G2 + ground + dome light)")
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0)
    scene_cfg.dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )
    scene_cfg.ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    scene_cfg.robot = AGIBOT_G2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    dt = sim.get_physics_dt()
    print(f"[2/3] robot loaded: {len(robot.data.joint_names)} joints, dt={dt}s")

    # Hold arms/body/head at the init pose throughout, so the chassis is
    # the only thing that visibly moves.
    hold_target = robot.data.default_joint_pos.clone()
    robot.set_joint_position_target(hold_target)

    base = MobileBase(
        articulation=robot,
        x=0.0, y=0.0, theta=0.0,
        z=0.18,
        max_lin_speed=args.max_lin,
        max_ang_speed=args.max_ang,
    )

    print("[3/3] ready.\n")
    print(
        "  Type a target as: x y theta_deg     (e.g. '1.0 0.5 90')\n"
        "  Or just 'x y'                       (theta defaults to 0)\n"
        "  Type 'q' (or close the window) to quit.\n"
        f"  Speed caps: lin={args.max_lin} m/s, ang={args.max_ang} rad/s\n",
        flush=True,
    )

    target: tuple[float, float, float] | None = None
    print("waypoint> ", end="", flush=True)

    try:
        while sim_app.is_running():
            # Hold arms/body steady; only the base teleports.
            robot.set_joint_position_target(hold_target)

            if target is not None:
                reached = base.step_toward(
                    target[0], target[1], target[2], dt
                )
                if reached:
                    s = base.state
                    print(
                        f"\n  [reached] pose=({s.x:.3f}, {s.y:.3f}, "
                        f"{s.theta:.3f} rad)"
                    )
                    target = None
                    print("waypoint> ", end="", flush=True)
            else:
                # No active waypoint — still write the base pose every step
                # so PhysX residuals (or anything else acting on the free
                # root) can't drift the chassis between commands.
                base.write_to_sim()

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
                print(
                    "  parse error — expected 'x y' or 'x y theta_deg' "
                    "(numbers in metres / degrees)."
                )
                print("waypoint> ", end="", flush=True)
                continue
            target = parsed
            s = base.state
            print(
                f"  driving from ({s.x:.3f}, {s.y:.3f}, {s.theta:.3f}) → "
                f"({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})..."
            )
    finally:
        sim_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
