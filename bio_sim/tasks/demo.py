#
# Demo task: pure-arm pick-and-place. Base does not move -- robot_start IS
# the work pose, and the skill list contains zero FaceYaw / DriveStraight.
#
# Skill list:
#   pre-grasp -> grasp -> Grasp -> lift -> pre-place -> place -> Release -> retreat
#
# All targets come from scene.grasp_xyz / scene.place_xyz, which DemoScene
# sets to:
#   * grasp_xyz = plate's spawn pose on the table top (-X side of the table)
#   * place_xyz = OT-One deck centre (+X side of the table) + place_offset_xy
#
# Phased arm control mirrors pick_place: KINEMATIC for the centre-on-cube
# moves so the closing fingers don't swipe the plate; PD for everything
# after the grip so a friction-held / welded payload follows smoothly.
#
# Config is loaded the same way as pick_place (shared task yaml + per-robot
# overlay). Driven by bio_sim/config/task_demo.yaml + bio_sim/config/robots/
# <robot>.yaml; the cli routes the right file via TaskSpec.config_file.
#

from __future__ import annotations

from typing import List

from ..skills.grasp import Grasp, Release
from ..skills.move_arm import MoveArmTo
from ..skills.skill import Skill


def build_demo(cfg: dict | None = None) -> List[Skill]:
    cfg = cfg or {}
    obj = cfg["object"]
    pre_dz = cfg["pre_grasp_dz"]
    lift_dz = cfg["lift_dz"]
    retreat_dz = cfg["retreat_dz"]
    place_dz = cfg["place_clearance_dz"]
    # grasp_dz < 0 sinks EE BELOW the object's top face. Needed for thin
    # objects (e.g. 1.46cm well plate) where the omnipicker fingertip
    # closing arc passes at gripper_center height -- with EE at plate top
    # the fingertips clear the plate sides entirely and close on empty
    # air. -0.008 puts EE 8mm into the plate body so fingertips engage
    # the side wall mid-height.
    grasp_dz = float(cfg.get("grasp_dz", 0.0))

    return [
        # --- pick on the table ---
        MoveArmTo(MoveArmTo.grasp_pose(obj, dz=pre_dz),
                  label="pre-grasp", kinematic=True),
        MoveArmTo(MoveArmTo.grasp_pose(obj, dz=grasp_dz),
                  label="grasp", kinematic=True),
        Grasp(obj),
        MoveArmTo(MoveArmTo.gripper_offset(lift_dz),
                  label="lift", kinematic=False),
        # --- place into the OT-One deck ---
        MoveArmTo(MoveArmTo.scene_place(dz=pre_dz),
                  label="pre-place", kinematic=False),
        MoveArmTo(MoveArmTo.scene_place(dz=place_dz),
                  label="place", kinematic=True),
        Release(obj),
        MoveArmTo(MoveArmTo.gripper_offset(retreat_dz),
                  label="retreat", kinematic=False),
    ]
