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
    # If set, a rectangular box (x, y, z) instead of a cube -- a thin
    # elongated "rod"/tube the omnipicker can actually cage, vs a fat cube
    # its fingers just swipe.
    scale: Tuple[float, float, float] | None = None
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
# A thin object makes the omnipicker over-close (idx81 ~0 = pinching near
# nothing -> no grip). A CHUNKY box fills the gripper's grasp so the
# fingers stop at a firm mid-range clamp. (Earlier big-cube "swipe" was the
# centering / fast-close bugs, since fixed.)
DEFAULT_OBJECTS = [ObjectSpec(name="object_a", position=(0.5, 0.0, 0.5),
                              scale=(0.03, 0.1, 0.1))]
DEFAULT_TABLES = [
    TableSpec("table_A", center=(0.5, 0.0, 0.2), size=(0.4, 0.4, 0.4)),
    TableSpec("table_B", center=(1.5, 0.0, 0.2), size=(0.4, 0.4, 0.4)),
]
DEFAULT_MARKERS = [
    Marker("A", position=(0.0, 0.0, 0.02), yaw=0.0, color=(0.1, 0.8, 0.1)),
    Marker("B", position=(1.0, 0.0, 0.02), yaw=0.0, color=(0.1, 0.4, 0.9)),
]

OBJECT_HALF = 0.04           # z half-extent of the box (0.08/2)
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
        self._sim = None
        # filled by place_for_validation()
        self.place_xyz: Tuple[float, float, float] | None = None
        self.grasp_q = np.array([1.0, 0.0, 0.0, 0.0])

    # ---- build --------------------------------------------------------
    def build(self, sim) -> None:
        # Only the ground + cuRobo world here. The object/tables/markers are
        # spawned LATER (place_for_validation) directly at their resolved
        # poses: Isaac records a prim's default state at add() time and
        # snaps back to it on Play, so spawning at the default pose and
        # moving afterwards drops the cube on the floor on a windowed Play.
        self._sim = sim
        sim.world.scene.add_default_ground_plane()
        self._build_curobo_world()

    def _spawn_props(self) -> None:
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.api.objects import (
            DynamicCuboid,
            FixedCuboid,
            VisualCuboid,
        )

        sim = self._sim
        for t in self.tables:
            self._table_prims[t.name] = sim.world.scene.add(FixedCuboid(
                prim_path=f"/World/{t.name}", name=t.name,
                position=np.array(t.center, dtype=np.float32),
                scale=np.array(t.size, dtype=np.float32),
                color=np.array(t.color),
            ))
        # High-friction material so the force-closed fingers can hold it.
        grip_mat = PhysicsMaterial(
            prim_path="/World/PhysicsMaterials/grip",
            static_friction=1.2, dynamic_friction=1.0, restitution=0.1,
        )
        for o in self.objects:
            # Mass = 0.01 kg, matching genie_sim's ACTUAL value
            # (G2_omnipicker client default 0.01 kg). The earlier note
            # here claimed "genie uses ~0.2 kg" -- that was wrong; genie
            # never validated pure-friction carry of a heavy object, it
            # used a 10 g toy mass. This underactuated mimic gripper holds
            # by fingertip friction only, so the payload must be light;
            # the old "0.2 kg gets punched away if light" worry is moot
            # now that the base accel-slew, vice-hold, and arrive-gated
            # settle remove the impulsive disturbances.
            kw = {}
            if o.scale is not None:
                kw["scale"] = np.array(o.scale, dtype=np.float32)
            else:
                kw["size"] = o.size
            cube = sim.world.scene.add(DynamicCuboid(
                prim_path=f"/World/{o.name}", name=o.name,
                position=np.array(o.position, dtype=np.float32),
                color=np.array(o.color), mass=0.3, **kw,
            ))
            try:
                cube.apply_physics_material(grip_mat)
            except Exception as exc:  # noqa: BLE001
                print(f"[scene] friction material skipped: {exc}")
            self._obj_prims[o.name] = cube
        for m in self.markers:
            self._marker_prims[m.name] = VisualCuboid(
                prim_path=f"/World/marker_{m.name}", name=f"marker_{m.name}",
                position=np.array(m.position, dtype=np.float32),
                size=0.08, color=np.array(m.color),
            )

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
    def place_for_validation(self, robot, cfg: dict) -> None:
        """Snap table/object/markers to a provably reachable workspace anchor.

        Robot starts at world origin (A), so base frame == world frame here.
        Keep the (IK-checked) retract EE orientation and search a grid of
        candidate positions; among those where grasp, pre-grasp (+pre_dz)
        and lift/retreat (+clearance) are ALL IK-feasible, pick the one whose
        height is CLOSEST to cfg['grasp_height'] (so the robot grasps low at
        a realistic table height, not up near its shoulder).
        B := A + (NAV_DX, 0); place mirrors the grasp in B's frame.
        """
        import numpy as _np

        pre_dz = cfg["pre_grasp_dz"]
        lift_dz = cfg["lift_dz"]
        want_z = cfg.get("grasp_height", 0.75)

        p_ee, q_ee = robot.arm.retract_link_pose(
            robot.retract_config, robot.j_names, robot.ee_link
        )
        self.grasp_q = q_ee.copy()
        # make ik_ok enforce the same idle-arm pin plan_single uses
        robot.arm.compute_idle_retract_pin(robot.retract_config, robot.j_names)

        clearance = max(lift_dz, pre_dz)
        # Build candidates, try them ordered by closeness to the desired
        # height; accept the FIRST where grasp + pre-grasp + lift all
        # actually plan (real plan_single, idle pinned) and early-exit.
        z_lo = max(0.30, want_z - 0.20)
        z_hi = min(float(p_ee[2]) + 1e-6, want_z + 0.35)
        cands = []
        for shrink in (1.0, 0.85, 0.7, 0.55):
            for z in _np.arange(z_lo, z_hi, 0.05):
                cands.append(_np.array(
                    [p_ee[0] * shrink, p_ee[1] * shrink, z], dtype=_np.float64))
        cands.sort(key=lambda c: (abs(c[2] - want_z), -c[0]))

        anchor = None
        for ci, c in enumerate(cands):
            if (robot.arm.plan_ok(c, q_ee)
                    and robot.arm.plan_ok(c + [0, 0, pre_dz], q_ee)
                    and robot.arm.plan_ok(c + [0, 0, clearance], q_ee)):
                anchor = c
                print(f"[scene] anchor found after {ci + 1} candidates; "
                      f"z={c[2]:.3f} (target {want_z})")
                break
        if anchor is None:
            anchor = _np.array(p_ee, dtype=_np.float64)
            print("[scene] WARNING: no plannable anchor found; "
                  "falling back to retract FK")

        # `anchor` is a BASE-frame target validated by plan_ok. Runtime reads
        # the object's WORLD pose and converts via world_to_base, which
        # subtracts the base standing height. So place the object in WORLD at
        # anchor + BASE_STAND_Z; then world_to_base recovers exactly `anchor`
        # (base at A is x=y=yaw=0, z=BASE_STAND_Z). Without this the runtime
        # grasp goal is BASE_STAND_Z below the validated pose -> IK_FAIL.
        from ..robot.base import BASE_STAND_Z

        ox, oy = float(anchor[0]), float(anchor[1])
        oz = float(anchor[2]) + BASE_STAND_Z
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

        # Specs are now final -> spawn the prims AT these poses, so Isaac
        # records them as the default state (no Play snap-back / floor drop).
        self._spawn_props()

        print(f"[scene] validation layout: object/grasp @ "
              f"({ox:.3f},{oy:.3f},{oz:.3f})  "
              f"table_top={top:.3f}  place @ {self.place_xyz}")

    # ---- queries ------------------------------------------------------
    def object_pose(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        p, q = self._obj_prims[name].get_world_pose()
        return np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)

    def object_prim(self, name: str):
        return self._obj_prims[name]

    def object_dims(self, name: str):
        """(x, y, z) extents for the cuRobo payload box."""
        for o in self.objects:
            if o.name == name:
                if o.scale is not None:
                    return list(o.scale)
                return [o.size, o.size, o.size]
        return [0.05, 0.05, 0.05]

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
        # The validation table is a PHYSICAL rest surface, not a planning
        # obstacle: as a cuRobo obstacle the thin slab right under the cube
        # blocks the hand from descending to a low grasp (IK_FAIL). Excluding
        # it also makes setup-time IK match runtime planning.
        ignore += [f"/World/{t.name}" for t in self.tables]
        obstacles = self._usd_help.get_obstacles_from_stage(
            only_paths=["/World"],
            reference_prim_path=robot_prim_path,
            ignore_substring=ignore,
        ).get_collision_check_world()
        arm.sync_world(obstacles)
        print(f"[scene] cuRobo world resynced @ step {step_index}")
