#
# BioScene: declarative environment.
#
# Validation scene (before real bio assets): a big FixedCuboid = table, a
# small DynamicCuboid = object, two floor markers A/B. Every pose is KNOWN
# and, more importantly, DERIVED from the robot's retract-config forward
# kinematics in place_for_validation(): the nominal grasp target equals the
# retract EE pose, so IK cannot fail on it. The place leg mirrors the grasp
# in the B base frame, so it is equally reachable.
#
# Adding real assets later = extend OBJECTS / TABLES / MARKERS (or subclass);
# the loop never changes.
#
# Frame: world frame, quaternions (w, x, y, z) (cuRobo / Isaac convention).
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class ObjectSpec:
    name: str
    position: Tuple[float, float, float]
    size: float = 0.05
    color: Tuple[float, float, float] = (0.9, 0.1, 0.1)


@dataclass
class TableSpec:
    name: str
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    color: Tuple[float, float, float] = (0.45, 0.3, 0.2)


@dataclass
class Marker:
    name: str
    position: Tuple[float, float, float]
    yaw: float = 0.0
    color: Tuple[float, float, float] = (0.2, 0.8, 0.2)


# Placeholders; place_for_validation() overwrites every pose from the robot's
# real workspace once the arm planner exists.
DEFAULT_OBJECTS = [ObjectSpec(name="object_a", position=(0.5, 0.0, 0.5), size=0.05)]
DEFAULT_TABLES = [
    TableSpec("table_A", center=(0.5, 0.0, 0.2), size=(0.4, 0.4, 0.4)),
    TableSpec("table_B", center=(1.5, 0.0, 0.2), size=(0.4, 0.4, 0.4)),
]
DEFAULT_MARKERS = [
    Marker("A", position=(0.0, 0.0, 0.02), yaw=0.0, color=(0.1, 0.8, 0.1)),
    Marker("B", position=(1.0, 0.0, 0.02), yaw=0.0, color=(0.1, 0.4, 0.9)),
]

OBJECT_HALF = 0.025          # half of object size (0.05)
GRASP_CLEARANCE = 0.001      # object bottom barely above table top
SLAB_THICKNESS = 0.04        # table is a thin slab, not a floor monolith
NAV_DX = 1.0                 # B is A + this many metres along world +x


class BioScene:
    def __init__(self, objects=None, tables=None, markers=None):
        self.objects: List[ObjectSpec] = objects or list(DEFAULT_OBJECTS)
        self.tables: List[TableSpec] = tables or list(DEFAULT_TABLES)
        self.markers: List[Marker] = markers or list(DEFAULT_MARKERS)
        self._obj_prims: Dict[str, object] = {}
        self._table_prims: Dict[str, object] = {}
        self._marker_prims: Dict[str, object] = {}
        self._marker_by_name = {m.name: m for m in self.markers}
        self._usd_help = None
        self._world_cfg = None
        self._last_sync = -1
        # filled by place_for_validation()
        self.place_xyz: Tuple[float, float, float] | None = None
        self.grasp_q = np.array([1.0, 0.0, 0.0, 0.0])

    # ---- build --------------------------------------------------------
    def build(self, sim) -> None:
        from isaacsim.core.api.objects import (
            DynamicCuboid,
            FixedCuboid,
            VisualCuboid,
        )

        sim.world.scene.add_default_ground_plane()

        for t in self.tables:
            self._table_prims[t.name] = sim.world.scene.add(FixedCuboid(
                prim_path=f"/World/{t.name}", name=t.name,
                position=np.array(t.center, dtype=np.float32),
                scale=np.array(t.size, dtype=np.float32),
                color=np.array(t.color),
            ))

        for o in self.objects:
            self._obj_prims[o.name] = sim.world.scene.add(DynamicCuboid(
                prim_path=f"/World/{o.name}", name=o.name,
                position=np.array(o.position, dtype=np.float32),
                size=o.size, color=np.array(o.color),
            ))

        for m in self.markers:
            self._marker_prims[m.name] = VisualCuboid(
                prim_path=f"/World/marker_{m.name}", name=f"marker_{m.name}",
                position=np.array(m.position, dtype=np.float32),
                size=0.08, color=np.array(m.color),
            )

        self._build_curobo_world()

    def _build_curobo_world(self) -> None:
        # Initial placeholder world; the periodic stage resync (maybe_sync)
        # replaces it with the real table geometry read off the stage.
        from curobo.geom.types import WorldConfig
        from curobo.util.usd_helper import UsdHelper
        from curobo.util_file import get_world_configs_path, join_path, load_yaml

        table = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
        )
        table.cuboid[0].pose[2] -= 0.02
        mesh_world = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
        ).get_mesh_world()
        mesh_world.mesh[0].name += "_mesh"
        mesh_world.mesh[0].pose[2] = -10.5
        self._world_cfg = WorldConfig(cuboid=table.cuboid, mesh=mesh_world.mesh)
        self._usd_help = UsdHelper()

    @property
    def curobo_world(self):
        return self._world_cfg

    def attach_to_stage(self, sim) -> None:
        self._usd_help.load_stage(sim.world.stage)
        self._usd_help.add_world_to_stage(self._world_cfg, base_frame="/World")

    # ---- deterministic, reachable placement ---------------------------
    def place_for_validation(self, robot, pre_dz: float, lift_dz: float) -> None:
        """Snap table/object/markers to a provably reachable workspace anchor.

        Robot starts at world origin (A), so base frame == world frame here.
        We keep the retract EE orientation and SEARCH for a position where
        the grasp, the pre-grasp (+pre_dz) and the lift/retreat (+lift_dz)
        are ALL IK-feasible -- retract FK alone can sit on a reach boundary.
        B := A + (NAV_DX, 0); place mirrors the grasp in B's frame (pure
        translation, yaw 0) so it is equally reachable.
        """
        import numpy as _np

        p_ee, q_ee = robot.arm.retract_link_pose(
            robot.retract_config, robot.j_names, robot.ee_link
        )
        self.grasp_q = q_ee.copy()

        clearance = max(lift_dz, pre_dz)
        anchor = None
        for shrink in (1.0, 0.85, 0.7, 0.55):
            for dz_down in _np.arange(0.0, 0.65, 0.05):
                c = _np.array([p_ee[0] * shrink, p_ee[1] * shrink,
                               p_ee[2] - dz_down], dtype=_np.float64)
                if (robot.arm.ik_ok(c, q_ee)
                        and robot.arm.ik_ok(c + [0, 0, pre_dz], q_ee)
                        and robot.arm.ik_ok(c + [0, 0, clearance], q_ee)):
                    anchor = c
                    break
            if anchor is not None:
                break
        if anchor is None:
            anchor = _np.array(p_ee, dtype=_np.float64)
            print("[scene] WARNING: no fully-reachable anchor found; "
                  "falling back to retract FK (pre-grasp/lift may fail)")

        ox, oy, oz = float(anchor[0]), float(anchor[1]), float(anchor[2])
        obj = self.objects[0]
        obj.position = (ox, oy, oz)

        # table = a THIN slab whose top sits just under the object. A
        # floor-to-top monolith would intersect the robot's own collision
        # spheres (the anchor is only ~0.5 m from the base) and fail IK.
        top = oz - OBJECT_HALF - GRASP_CLEARANCE
        tcx, tcy = ox, oy
        tA = self.tables[0]
        tA.center = (tcx, tcy, top - SLAB_THICKNESS / 2.0)
        tA.size = (0.40, 0.40, SLAB_THICKNESS)

        # B and table_B: same geometry shifted +NAV_DX in world x
        self._marker_by_name["A"].position = (0.0, 0.0, 0.02)
        self._marker_by_name["A"].yaw = 0.0
        self._marker_by_name["B"].position = (NAV_DX, 0.0, 0.02)
        self._marker_by_name["B"].yaw = 0.0
        self.place_xyz = (ox + NAV_DX, oy, oz)
        tB = self.tables[1]
        tB.center = (tcx + NAV_DX, tcy, top - SLAB_THICKNESS / 2.0)
        tB.size = tA.size

        # push prims to their resolved poses
        self._obj_prims[obj.name].set_world_pose(
            position=np.array(obj.position, dtype=np.float32)
        )
        for t in (tA, tB):
            self._table_prims[t.name].set_world_pose(
                position=np.array(t.center, dtype=np.float32)
            )
            self._table_prims[t.name].set_local_scale(
                np.array(t.size, dtype=np.float32)
            )
        for name in ("A", "B"):
            m = self._marker_by_name[name]
            self._marker_prims[name].set_world_pose(
                position=np.array(m.position, dtype=np.float32)
            )

        print(f"[scene] validation layout: object/grasp @ "
              f"({ox:.3f},{oy:.3f},{oz:.3f})  "
              f"table_top={top:.3f}  place @ {self.place_xyz}")

    # ---- queries ------------------------------------------------------
    def object_pose(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        p, q = self._obj_prims[name].get_world_pose()
        return np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)

    def object_prim(self, name: str):
        return self._obj_prims[name]

    def marker_pose(self, name: str) -> Tuple[float, float, float]:
        m = self._marker_by_name[name]
        return m.position[0], m.position[1], m.yaw

    # ---- periodic cuRobo obstacle resync ------------------------------
    def maybe_sync(self, step_index: int, arm, robot_prim_path: str) -> None:
        if not (step_index == 50 or step_index % 1000 == 0):
            return
        if step_index == self._last_sync:
            return
        self._last_sync = step_index
        ignore = [robot_prim_path, "/World/defaultGroundPlane", "/curobo"]
        ignore += [f"/World/{o.name}" for o in self.objects]
        ignore += [f"/World/marker_{m.name}" for m in self.markers]
        obstacles = self._usd_help.get_obstacles_from_stage(
            only_paths=["/World"],
            reference_prim_path=robot_prim_path,
            ignore_substring=ignore,
        ).get_collision_check_world()
        arm.sync_world(obstacles)
        print(f"[scene] cuRobo world resynced @ step {step_index}")
