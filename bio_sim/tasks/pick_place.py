#
# PickAndPlace = an ordered list of skills. Geometry is deterministic and
# reachable: scene.place_for_validation() derives the grasp pose from the
# robot's retract FK and mirrors it at B. Offsets come from
# config/task_pick_place.yaml; the structure is generic.
#
# The base path is now SCRIPTED with relative moves (FaceYaw / DriveStraight)
# instead of NavigateTo-marker, per the user's spec. The numbers are tied to
# the validated layout (see the table below): the base must end EXACTLY at
# the validated grasp pose (0,0) and place pose (nav_dx,0) or the arm IK
# fails -- those constraints fix every distance.
#
#   spawn (robot_start = 0, 1.5, yaw 0)
#   FaceYaw(face_yaw)             -> face the table        (0, 1.5, fy)
#   DriveStraight(1.5)            -> validated grasp pose   (0, 0,  fy)
#   <pre-grasp, grasp, Grasp, lift>
#   DriveStraight(1.0, reverse)   -> back off, still facing (0, 1.0, fy)
#   FaceYaw(0)                    -> face +x                (0, 1.0, 0)
#   DriveStraight(NAV_DX=2.5)     -> detour at y=1.0 around (2.5,1.0, 0)
#                                    the bio_optica (y=-0.7)
#   FaceYaw(face_yaw)             -> face the table         (2.5,1.0, fy)
#   DriveStraight(1.0)            -> validated place pose    (2.5,0,  fy)
#   <pre-place, place, Release, retreat>
#

from __future__ import annotations

import math
import os
from typing import List

import yaml

from ..skills.grasp import Grasp, Release
from ..skills.move_arm import MoveArmTo
from ..skills.navigate import DriveStraight, FaceYaw
from ..skills.skill import Skill

_CFG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")
_CFG = os.path.join(_CFG_DIR, "task_pick_place.yaml")
_ROBOT_CFG_DIR = os.path.join(_CFG_DIR, "robots")

# Scripted base-path distances (metres). Forward-to-A and the traverse are
# fixed by the validated layout (robot_start.y and nav_dx); the back-off and
# final approach are the user's spec. Tune here if you change robot_start /
# nav_dx so the base still lands on the validated grasp/place poses.
_FWD_TO_A = 1.0      # spawn y=1.5 -> grasp pose y=0
_BACK_OFF = 1.0      # pure reverse off the A table (still facing it)
# Last forward leg onto B. GEOMETRICALLY LOCKED to _BACK_OFF: the traverse
# leg keeps y constant, so the only way the base lands back on the validated
# place pose (y=0) is final-approach == back-off. NOT an independent knob --
# derived so it can never desync (a 0.2 m mismatch silently stalls pre-place
# in the 1500-tick SETTLE deadlock guard, no FAILURE emitted).
_FINAL_APPROACH = 1.0


def load_cfg(path: str | None = None) -> dict:
    with open(path or _CFG, "r") as f:
        return yaml.safe_load(f)


def load_robot_cfg(name: str, path: str | None = None) -> dict:
    # Per-robot overlay (cube/grasp/init pose), e.g. name='g2' or 'r1pro'.
    p = path or os.path.join(_ROBOT_CFG_DIR, f"{name}.yaml")
    with open(p, "r") as f:
        return yaml.safe_load(f) or {}


def load_full_cfg(robot: str) -> dict:
    # Shared task config + per-robot overlay (overlay keys win).
    cfg = load_cfg()
    cfg.update(load_robot_cfg(robot))
    return cfg


def build_pick_place(cfg: dict | None = None) -> List[Skill]:
    cfg = cfg or load_cfg()
    obj = cfg["object"]
    pre_dz = cfg["pre_grasp_dz"]
    lift_dz = cfg["lift_dz"]
    retreat_dz = cfg["retreat_dz"]
    place_dz = cfg["place_clearance_dz"]
    face_yaw = math.radians(cfg.get("robot_face_yaw_deg", -90.0))
    traverse = float(cfg.get("nav_dx", 2.5))  # detour leg == A<->B spacing

    # Phased arm control (Option B): KINEMATIC for the approach so the
    # gripper centers on the cube to ~mm (a few cm of PD error makes the
    # closing fingers swipe a free cube away); PD for everything after the
    # grip so the carried cube follows the hand smoothly.
    return [
        # --- go to A (scripted) ---
        FaceYaw(face_yaw, label="face-table-A"),
        DriveStraight(_FWD_TO_A, label="approach-A"),
        # --- pick ---
        MoveArmTo(MoveArmTo.grasp_pose(obj, dz=pre_dz),
                  label="pre-grasp", kinematic=True),
        MoveArmTo(MoveArmTo.grasp_pose(obj, dz=0.0),
                  label="grasp", kinematic=True),
        Grasp(obj),
        MoveArmTo(MoveArmTo.gripper_offset(lift_dz),
                  label="lift", kinematic=False),
        # --- A -> B (scripted detour at y=1.0, around the bio_optica) ---
        DriveStraight(_BACK_OFF, reverse=True, label="back-off-A"),
        FaceYaw(0.0, label="face-+x"),
        DriveStraight(traverse, label="traverse"),
        FaceYaw(face_yaw, label="face-table-B"),
        DriveStraight(_FINAL_APPROACH, label="approach-B"),
        # --- place ---
        MoveArmTo(MoveArmTo.scene_place(dz=pre_dz),
                  label="pre-place", kinematic=False),
        MoveArmTo(MoveArmTo.scene_place(dz=place_dz),
                  label="place", kinematic=False),
        Release(obj),
        MoveArmTo(MoveArmTo.gripper_offset(retreat_dz),
                  label="retreat", kinematic=False),
    ]
