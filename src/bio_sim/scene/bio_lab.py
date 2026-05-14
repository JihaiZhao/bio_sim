"""Bio-lab scene composition for bio_sim.

Provides IsaacLab ``AssetBaseCfg`` / ``RigidObjectCfg`` instances for the
Thorlabs optical table (the bench) and a 500 mL beaker (the pickable item),
plus a matching cuRobo ``scene_model`` dict with the bench as a cuboid
obstacle.

Lab-asset USDs live under ``src/bio_sim/assets/`` (gitignored — placed there
by the user). All coordinates are in the world frame (robot is spawned at the
origin facing +X via ``bio_sim.robot.agibot_g1_cfg.AGIBOT_G1_CFG``).
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg


_ASSETS = Path(__file__).resolve().parents[2] / "bio_sim" / "assets"

# --- Bench: Thorlabs 75x90 optical table -------------------------------------
# USD default prim /table_with_mount: Z-up, m/unit=1, top at local z=0, legs to
# local z=-0.795. x range (-0.176, 0.724), y range (-0.379, 0.379). USD origin
# is offset from the table's geometric centre: top centre is at local
# (+0.274, 0, 0).
BENCH_USD = _ASSETS / "infrastructure/tables/table-thorlabs-75x90/table.usda"
# The Thorlabs USD's origin is offset from the bench's geometric centre: the
# bench extends from x = BENCH_POS.x - 0.176 (back) to BENCH_POS.x + 0.724
# (front). The G1 base_link's collision spheres reach to roughly x = +0.32.
# Push BENCH_POS so the back edge lands at x ≈ 0.50, comfortably clear of the
# robot's footprint — otherwise cuRobo sees the start state in collision and
# silently fails to plan.
BENCH_POS = (0.68, 0.0, 0.79)           # back edge at world x ≈ 0.504
BENCH_TOP_CENTER = (BENCH_POS[0] + 0.274, BENCH_POS[1], BENCH_POS[2])
BENCH_TOP_DIMS = (0.9, 0.76, 0.05)     # approx top slab (X, Y, Z)


# --- Pickable: 500 mL beaker --------------------------------------------------
# USD default prim /Root: Y-up, m/unit=1, small (≈5 cm wide, 6 cm tall along Y).
# We rotate +90° around the world X axis so the beaker's Y axis (its long
# cylinder axis) lines up with world Z and the beaker stands upright on the
# bench top.
BEAKER_USD = _ASSETS / "labware/beaker500ml/beaker-500ml.usda"
BEAKER_HEIGHT = 0.055
# Place on the back third of the bench, right side — keeps the beaker close
# to the robot so the right arm doesn't have to fully extend.
BEAKER_POS = (BENCH_POS[0] - 0.08, -0.25, BENCH_POS[2] + BEAKER_HEIGHT / 2)
# ≈ (0.60, -0.25, 0.818)
# wxyz = (cos 45°, sin 45°, 0, 0) — 90° around +X axis.
BEAKER_QUAT_Y_UP_TO_Z_UP = (0.70710678, 0.70710678, 0.0, 0.0)


def make_bench_cfg(prim_path: str = "{ENV_REGEX_NS}/Bench") -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(BENCH_USD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=BENCH_POS),
    )


def make_beaker_cfg(prim_path: str = "{ENV_REGEX_NS}/Beaker") -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(BEAKER_USD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=BEAKER_POS,
            rot=BEAKER_QUAT_Y_UP_TO_Z_UP,
        ),
    )


def cuRobo_obstacles() -> dict:
    """cuRobo ``scene_model`` describing the bench as a single cuboid.

    The beaker is intentionally NOT in the obstacle set — it's the target to
    grasp, and cuRobo would otherwise refuse to enter contact with it.
    """
    cx, cy, cz = BENCH_TOP_CENTER
    dx, dy, dz = BENCH_TOP_DIMS
    return {
        "cuboid": {
            "bench_top": {
                "dims": [dx, dy, dz],
                # top centre is at (cx, cy, cz); pose specifies the cuboid centre,
                # which sits at cz - dz/2 because dz is the slab thickness.
                "pose": [cx, cy, cz - dz / 2, 1.0, 0.0, 0.0, 0.0],
            }
        }
    }


# Grasp geometry (relative to the beaker's world pose).
GRASP_APPROACH_HEIGHT = 0.18   # pre-grasp hover offset above beaker top (m)
GRASP_LIFT_HEIGHT = 0.20       # lift offset above beaker top after closing (m)
