"""M1 smoke test — spawn Dexmate Vega-1 alone on a ground plane.

Used to verify that the Dexmate USD imports cleanly into IsaacLab with the
ArticulationCfg authored in `bio_sim.robot.dexmate_cfg`. No motion planning yet.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="bio_sim: load Dexmate on a ground plane")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app

    try:
        # TODO(M1):
        #   1. Build a SimulationContext.
        #   2. Spawn GroundPlaneCfg.
        #   3. Spawn DEXMATE_VEGA_CFG (from bio_sim.robot.dexmate_cfg).
        #   4. Step the sim for ~5 s; assert the robot remains within tolerance of its initial pose.
        while sim_app.is_running():
            sim_app.update()
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
