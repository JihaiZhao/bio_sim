#
# Robot registry: one source of truth for "which robot are we running".
#
# Before this, "which robot" was encoded in three independent selectors:
#   * play.py --r1 flag        (selects R1ProRobot vs G2Robot)
#   * play.py --robot yml      (cuRobo planner config; default G2 swapped
#                                to R1 by a hard-coded startswith("R1Pro")
#                                test in play.py)
#   * load_full_cfg(<key>)     (task-config overlay file under
#                                bio_sim/config/robots/<key>.yaml)
# Adding a third robot meant editing all three. A `RobotSpec` bundles
# them, and `resolve(name)` is the only thing entry points call.
#
# Adding a new robot:
#   1. Subclass RobotBase (cf. bio_sim/robot/g2.py, r1pro.py).
#   2. Drop a config overlay at bio_sim/config/robots/<name>.yaml
#      (cube_xyz, grasp_quat, init_arm_pose, ...).
#   3. Append a RobotSpec to REGISTRY below.
#
# Import ordering: this module imports the robot classes at load time,
# which transitively pulls cuRobo / IsaacSim. play.py imports it from
# inside main(), AFTER SimApp(...) has booted -- same constraint as the
# existing `from bio_sim.robot import G2Robot` line. Do not import this
# at the top of a module that runs before SimApp.
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Type

from .g2 import G2Robot
from .r1pro import R1ProRobot
from .robot_base import RobotBase


@dataclass(frozen=True)
class RobotSpec:
    name: str
    cls: Type[RobotBase]
    default_curobo_yml: str   # default --robot value if the user omits it
    cfg_overlay: str          # filename stem under bio_sim/config/robots/


REGISTRY: dict[str, RobotSpec] = {
    "g2": RobotSpec(
        name="g2",
        cls=G2Robot,
        default_curobo_yml="G2_omnipicker_fixed_dual.yml",
        cfg_overlay="g2",
    ),
    "r1pro": RobotSpec(
        name="r1pro",
        cls=R1ProRobot,
        default_curobo_yml="R1Pro_arm_no_torso.yml",
        cfg_overlay="r1pro",
    ),
}


def resolve(name: str) -> RobotSpec:
    if name not in REGISTRY:
        raise SystemExit(
            f"unknown robot {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]


__all__ = ["REGISTRY", "RobotSpec", "resolve"]
