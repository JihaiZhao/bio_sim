#
# Single source of truth for "what can I run": robot / scene / task registries.
#
# Each spec is METADATA ONLY -- the implementation class / builder function
# is referenced by a "module:attr" string and resolved lazily via load_ref()
# AFTER SimApp boots. This is the boot-ordering constraint: importing G2Robot
# / R1ProRobot / etc. transitively pulls cuRobo and isaacsim.core, which crash
# without a SimulationApp constructed first. So `python -m bio_sim list` can
# read every spec's metadata for free; only `run` pays the import cost.
#
# Adding a new robot / scene / task = append one Spec line below + provide its
# yaml/builder. cli.py does NOT need to change -- its argparse Enums are
# derived from these dicts dynamically.
#

from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class RobotSpec:
    name: str
    description: str
    cls_ref: str              # e.g. "bio_sim.robot.g2:G2Robot"
    default_curobo_yml: str   # default cuRobo planner yml if --robot-yml omitted
    cfg_overlay: str          # filename stem under bio_sim/config/robots/


@dataclass(frozen=True)
class SceneSpec:
    name: str
    description: str
    cls_ref: str              # e.g. "bio_sim.scene.ot_one_scene:OtOneScene"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    description: str
    builder_ref: str          # e.g. "bio_sim.tasks.pick_place:build_pick_place"
    config_file: str          # filename under bio_sim/config/ (task-level yaml)
    compatible_scenes: tuple[str, ...]  # which SceneSpec.name values are valid


ROBOTS: dict[str, RobotSpec] = {
    "g2": RobotSpec(
        name="g2",
        description="G2 dual-arm omnipicker (kinematic base)",
        cls_ref="bio_sim.robot.g2:G2Robot",
        default_curobo_yml="G2_omnipicker_fixed_dual.yml",
        cfg_overlay="g2",
    ),
    "r1pro": RobotSpec(
        name="r1pro",
        description="R1 Pro (BEHAVIOR-1K holonomic base)",
        cls_ref="bio_sim.robot.r1pro:R1ProRobot",
        default_curobo_yml="R1Pro_arm_no_torso.yml",
        cfg_overlay="r1pro",
    ),
}


SCENES: dict[str, SceneSpec] = {
    "bio": SceneSpec(
        name="bio",
        description="Two-table A/B layout (plate on table top)",
        cls_ref="bio_sim.scene.bio_scene:BioScene",
    ),
    "ot_one": SceneSpec(
        name="ot_one",
        description="BioScene + plate on OT-One deck at table A",
        cls_ref="bio_sim.scene.ot_one_scene:OtOneScene",
    ),
}


TASKS: dict[str, TaskSpec] = {
    "pick_place": TaskSpec(
        name="pick_place",
        description="Pick from table A, scripted detour, place at table B",
        builder_ref="bio_sim.tasks.pick_place:build_pick_place",
        config_file="task_pick_place.yaml",
        compatible_scenes=("bio", "ot_one"),
    ),
}


DEFAULTS = {"robot": "g2", "scene": "ot_one", "task": "pick_place"}


def load_ref(ref: str):
    """Lazy import 'pkg.mod:attr'. MUST be called AFTER SimApp has booted
    (the targets pull cuRobo / isaacsim.core / torch in their import chains)."""
    mod, attr = ref.split(":")
    return getattr(importlib.import_module(mod), attr)


__all__ = [
    "DEFAULTS",
    "ROBOTS",
    "RobotSpec",
    "SCENES",
    "SceneSpec",
    "TASKS",
    "TaskSpec",
    "load_ref",
]
