"""M1 smoke test — run a minimal cuRobo plan on Dexmate Vega-1.

Verifies that cuRobo is installed, that `bio_sim/robot/curobo_robot.yml` parses,
and that MotionGen can produce a collision-free trajectory between two joint
configurations with no scene obstacles.

Run via Isaac Sim's `omni_python` once cuRobo's Isaac Sim install is wired up.
"""

from __future__ import annotations


def main() -> None:
    # TODO(M1):
    #   1. Load curobo.types.robot.RobotConfig from bio_sim/robot/curobo_robot.yml.
    #   2. Build a MotionGenConfig with an empty WorldConfig.
    #   3. Pick two joint configurations (home pose, slight arm offset).
    #   4. Call motion_gen.plan_single(...) and print success + trajectory length.
    raise NotImplementedError("M1: cuRobo plan with Dexmate Vega-1")


if __name__ == "__main__":
    main()
