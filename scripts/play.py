"""Main entry point: launch Isaac Sim and run the scripted pick-and-place demo.

Milestones:
  M0 — launch an empty stage and exit cleanly.
  M1 — load Dexmate Vega-1 on a ground plane (delegated to inspect_robot.py for now).
  M2 — full bio-lab scene + cuRobo-driven pick-and-place via bio_sim.pipeline.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="bio_sim: scripted pick-and-place demo")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app

    try:
        # M0: empty stage smoke test. The loop exits when the window is closed.
        # M2: replace with bio_sim.pipeline.run(...).
        while sim_app.is_running():
            sim_app.update()
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
