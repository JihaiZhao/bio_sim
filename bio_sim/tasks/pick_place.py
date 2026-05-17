#
# PickAndPlace = an ordered list of skills. Geometry is deterministic and
# reachable: scene.place_for_validation() derives the grasp pose from the
# robot's retract FK and mirrors it at B. Offsets come from
# config/task_pick_place.yaml; the structure is generic.
#

from __future__ import annotations

import os
from typing import List

import yaml

from ..skills.grasp import Grasp, Release
from ..skills.move_arm import MoveArmTo
from ..skills.navigate import NavigateTo
from ..skills.skill import Skill

_CFG = os.path.join(os.path.dirname(__file__), "..", "config", "task_pick_place.yaml")


def load_cfg(path: str | None = None) -> dict:
    with open(path or _CFG, "r") as f:
        return yaml.safe_load(f)


def build_pick_place(cfg: dict | None = None) -> List[Skill]:
    cfg = cfg or load_cfg()
    obj = cfg["object"]
    pre_dz = cfg["pre_grasp_dz"]
    lift_dz = cfg["lift_dz"]
    retreat_dz = cfg["retreat_dz"]

    return [
        NavigateTo.to_marker(cfg["pick_marker"]),
        MoveArmTo(MoveArmTo.grasp_pose(obj, dz=pre_dz), label="pre-grasp"),
        MoveArmTo(MoveArmTo.grasp_pose(obj, dz=0.0), label="grasp"),
        Grasp(obj),
        MoveArmTo(MoveArmTo.gripper_offset(lift_dz), label="lift"),
        NavigateTo.to_marker(cfg["place_marker"]),
        MoveArmTo(MoveArmTo.scene_place(dz=pre_dz), label="pre-place"),
        MoveArmTo(MoveArmTo.scene_place(dz=0.0), label="place"),
        Release(obj),
        MoveArmTo(MoveArmTo.gripper_offset(retreat_dz), label="retreat"),
    ]
