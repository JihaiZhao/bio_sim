#
# BioScene: declarative environment.
#
# Validation scene (before real bio assets): a big FixedCuboid = table, a
# small DynamicCuboid = object. Every pose is KNOWN and, more importantly,
# DERIVED from the robot's retract-config forward kinematics in
# place_for_validation(): the nominal grasp target equals the retract EE
# pose, so IK cannot fail on it. The place leg mirrors the grasp in the B
# base frame, so it is equally reachable.
#
# Adding real assets later = extend OBJECTS / TABLES (or subclass); the
# loop never changes.
#
# Frame: world frame, quaternions (w, x, y, z) (cuRobo / Isaac convention).
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..asset_lib import load_object


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
    # genie-style root-relative dir of a USD-backed asset (e.g.
    # "objects/well_plate_96"). When set, _spawn_props references the USD
    # instead of building a DynamicCuboid; the USD must carry its own
    # RigidBodyAPI + collider (see assets/objects/<name>/object_parameters.json
    # for the metadata used here: scaled-size for layout, mass not needed).
    asset: str | None = None


@dataclass
class TableSpec:
    name: str
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    color: Tuple[float, float, float] = (0.45, 0.3, 0.2)


@dataclass
class FixtureSpec:
    """A static scene prop loaded from the asset library by its genie-style
    root-relative `asset` directory (data_info_dir). Pose is WORLD-frame;
    quaternion is (w, x, y, z). Not a physics body -- furniture/equipment
    the robot works *at*, mirroring genie's scene/background objects."""

    name: str
    asset: str                              # e.g. "objects/bio_optica_aus240plus"
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    # float -> uniform; 3-tuple -> per-axis (x, y, z); None -> sidecar scale.
    scale: float | Tuple[float, float, float] | None = None


def _scale3(s) -> Tuple[float, float, float]:
    if isinstance(s, (int, float)):
        return (float(s), float(s), float(s))
    return (float(s[0]), float(s[1]), float(s[2]))


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

GRASP_CLEARANCE = 0.002      # object bottom barely above table top
SLAB_THICKNESS = 0.04        # table is a thin slab, not a floor monolith
# Layout (cube/table world poses, robot facing, A<->B spacing, grasp
# orientation) is now HARD-CODED via task_pick_place.yaml -- no IK search.
# These are only fallbacks if a yaml key is missing.
NAV_DX = 1.0                 # B is A + this many metres along world +x


class BioScene:
    def __init__(self, objects=None, tables=None, fixtures=None):
        self.objects: List[ObjectSpec] = objects or list(DEFAULT_OBJECTS)
        self.tables: List[TableSpec] = tables or list(DEFAULT_TABLES)
        # Small "risers" (one per table) sit on the table top and lift the
        # plate so the gripper fingers can wrap its edge -- see riser block
        # in task_pick_place.yaml. Populated by place_for_validation; empty
        # = disabled (plate rests directly on the table).
        self.risers: List[TableSpec] = []
        self.fixtures: List[FixtureSpec] = fixtures or []
        self._fixture_prims: Dict[str, object] = {}
        self._obj_prims: Dict[str, object] = {}
        self._table_prims: Dict[str, object] = {}
        self._riser_prims: Dict[str, object] = {}
        self._usd_help = None
        self._world_cfg = None
        self._last_sync = -1
        self._sim = None
        # Lighting + room cfg subdicts (or None). Populated by from_cfg.
        self._lighting_cfg: dict | None = None
        self._room_cfg: dict | None = None
        # filled by place_for_validation()
        self.place_xyz: Tuple[float, float, float] | None = None
        self.grasp_xyz: Tuple[float, float, float] | None = None
        self.grasp_q = np.array([1.0, 0.0, 0.0, 0.0])
        # Cube spawn orientation (w,x,y,z). Was hard-IDENTITY (thin 3cm axis
        # locked to world X); now a yaml knob so the cube can be rotated to
        # present its 3cm face to whatever grasp_quat the gripper uses.
        self.cube_quat = np.array([1.0, 0.0, 0.0, 0.0])
        # Per-reset plate randomization. Scenes that opt in set _plate_base_xy
        # and _plate_random_dxy in place_for_validation; randomize_plate then
        # resamples grasp_xyz around the base. Default: disabled (no-op).
        self._plate_base_xy: Tuple[float, float] | None = None
        self._plate_random_dxy: Tuple[float, float] = (0.0, 0.0)
        self._plate_rng = np.random.default_rng()

    @classmethod
    def from_cfg(cls, cfg: dict) -> "BioScene":
        """Build from a declarative recipe (genie-style): the `scene.fixtures`
        list references assets by root-relative `asset` dir + a pose; this
        layer never hardcodes prim geometry for them.

        `scene.objects` (optional) overrides DEFAULT_OBJECTS: each entry may
        carry an `asset` field to spawn a USD-backed rigid body (see
        ObjectSpec.asset). place_for_validation still owns the runtime pose
        (read from cube_xyz / cube_quat), so per-entry `position` is unused
        for the first object.
        """
        recipe = (cfg or {}).get("scene", {}) or {}
        fixtures = [
            FixtureSpec(
                name=f["name"],
                asset=f["asset"],
                position=tuple(f.get("position", (0.0, 0.0, 0.0))),
                quaternion=tuple(f.get("quaternion", (1.0, 0.0, 0.0, 0.0))),
                scale=f.get("scale"),
            )
            for f in recipe.get("fixtures", [])
        ]
        objects = None
        if recipe.get("objects"):
            objects = [
                ObjectSpec(
                    name=o["name"],
                    position=tuple(o.get("position", (0.0, 0.0, 0.0))),
                    size=float(o.get("size", 0.05)),
                    scale=tuple(o["scale"]) if o.get("scale") else None,
                    asset=o.get("asset"),
                )
                for o in recipe["objects"]
            ]
        scene = cls(objects=objects, fixtures=fixtures)
        scene._lighting_cfg = (cfg or {}).get("lighting")
        scene._room_cfg = (cfg or {}).get("room")
        return scene

    # ---- build --------------------------------------------------------
    def build(self, sim) -> None:
        # Only the ground + cuRobo world here. The object/tables are
        # spawned LATER (place_for_validation) directly at their resolved
        # poses: Isaac records a prim's default state at add() time and
        # snaps back to it on Play, so spawning at the default pose and
        # moving afterwards drops the cube on the floor on a windowed Play.
        self._sim = sim
        sim.world.scene.add_default_ground_plane()
        self._build_curobo_world()
        self._apply_lighting(sim)
        self._apply_room(sim)

    def _apply_lighting(self, sim) -> None:
        # Opt-in: tasks without a `lighting:` block in yaml keep the
        # default Isaac look. Tasks with one get USD-authored dome/sun/fill
        # under /World/_lighting/, plus an initial render mode.
        cfg = getattr(self, "_lighting_cfg", None)
        if not cfg:
            return
        from . import lighting
        lighting.remove_default_dome(sim.world.stage)
        lighting.apply_preset(sim.world.stage, "/World",
                              cfg.get("preset", "studio"),
                              override_cfg=cfg)
        render = cfg.get("render") or {}
        lighting.set_render_mode(render.get("mode", "RealTime"),
                                 spp=int(render.get("spp", 4)))

    def _apply_room(self, sim) -> None:
        # Opt-in: tasks without a `room:` block keep an open-air scene.
        cfg = getattr(self, "_room_cfg", None)
        if not cfg:
            return
        from . import room
        room.apply_walls(sim, "/World", cfg)

    def _spawn_props(self) -> None:
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.api.objects import (
            DynamicCuboid,
            FixedCuboid,
        )

        sim = self._sim
        for t in self.tables:
            self._table_prims[t.name] = sim.world.scene.add(FixedCuboid(
                prim_path=f"/World/{t.name}", name=t.name,
                position=np.array(t.center, dtype=np.float32),
                scale=np.array(t.size, dtype=np.float32),
                color=np.array(t.color),
            ))
        # Risers (optional): same primitive type as tables, just smaller and
        # sat on top. Spawned BEFORE the plate so the plate's RigidBody
        # settles correctly onto the riser top.
        for r in self.risers:
            self._riser_prims[r.name] = sim.world.scene.add(FixedCuboid(
                prim_path=f"/World/{r.name}", name=r.name,
                position=np.array(r.center, dtype=np.float32),
                scale=np.array(r.size, dtype=np.float32),
                color=np.array(r.color),
            ))
        # High-friction material so the force-closed fingers can hold it.
        grip_mat = PhysicsMaterial(
            prim_path="/World/PhysicsMaterials/grip",
            static_friction=5.2, dynamic_friction=5.0, restitution=0.1,
        )
        for o in self.objects:
            if o.asset is not None:
                self._obj_prims[o.name] = self._spawn_usd_object(o, grip_mat)
                continue
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
                orientation=np.array(self.cube_quat, dtype=np.float32),
                color=np.array(o.color), mass=0.5, **kw,
            ))
            try:
                cube.apply_physics_material(grip_mat)
            except Exception as exc:  # noqa: BLE001
                print(f"[scene] friction material skipped: {exc}")
            self._obj_prims[o.name] = cube
        self._spawn_fixtures()

    def _spawn_usd_object(self, o: ObjectSpec, grip_mat):
        """Spawn a USD-backed dynamic object: reference the library asset,
        wrap it in SingleRigidPrim so the runtime get/set_world_pose +
        velocity API stays uniform with the DynamicCuboid path, and apply
        the same grip-friction material as the cube. The referenced USD
        must already carry RigidBodyAPI + a collider (e.g. baked via the
        well_plate_96 setup script).

        SingleRigidPrim has NO apply_physics_material method (only the
        DynamicCuboid MRO picks it up via SingleGeometryPrim), so we bind
        the friction material at the USD level here. strongerThanDescendants
        overrides any baked DefaultMaterial binding on the collider sub-prim
        (e.g. well_plate's /World/mesh) -- without this override PhysX uses
        the baked-empty material (mu~=0.5) and the thin plate slips out of
        the friction grasp.

        We also force the collider approximation to `convexHull` on every
        Mesh under the rigid root. The shipped well_plate USD uses
        `convexDecomposition` against a 380k-vertex / 126k-face VISUAL
        mesh, which (a) blows past PhysX's per-hull vertex / hull-count
        budgets and (b) decomposes the 96 punched-through wells into a
        chaos of micro / degenerate convex pieces -- fingers slip into
        the gaps and pass through the plate (the observed pure-physics
        "passes through, doesn't lift" bug). convexHull collapses the
        whole mesh into a single outer hull; the 96 wells are filled
        solid, but friction grasp contacts the SIDE WALLS where hull and
        true geometry coincide, so the simplification doesn't lose the
        grasp surface."""
        from isaacsim.core.prims import SingleRigidPrim
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import Usd, UsdPhysics, UsdShade

        asset = load_object(o.asset)
        prim_path = f"/World/{o.name}"
        add_reference_to_stage(usd_path=asset.usd_path, prim_path=prim_path)
        rigid = SingleRigidPrim(
            prim_path=prim_path, name=o.name,
            position=np.array(o.position, dtype=np.float32),
            orientation=np.array(self.cube_quat, dtype=np.float32),
        )
        self._sim.world.scene.add(rigid)
        prim = self._sim.world.stage.GetPrimAtPath(prim_path)
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(
            grip_mat.material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )
        hulls = 0
        for p in Usd.PrimRange(prim):
            if not p.HasAPI(UsdPhysics.MeshCollisionAPI):
                continue
            mc = UsdPhysics.MeshCollisionAPI(p)
            attr = mc.GetApproximationAttr()
            if not attr:
                attr = mc.CreateApproximationAttr()
            attr.Set(UsdPhysics.Tokens.convexHull)
            hulls += 1
        print(f"[scene] object '{o.name}' <- {asset.data_info_dir} "
              f"size={asset.size} mass={asset.mass} "
              f"grip_mat bound, {hulls} mesh collider(s) -> convexHull")
        return rigid

    def _spawn_fixtures(self) -> None:
        """Reference each library asset's USD onto the stage as a STATIC prop.

        Pose is authored as USD xform ops (version-independent, and not
        subject to the World physics-reset snap-back that bit the dynamic
        cube -- a referenced Xform with no rigid body just stays put)."""
        if not self.fixtures:
            return
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import Gf, Usd, UsdGeom

        stage = self._sim.world.stage
        for fx in self.fixtures:
            asset = load_object(fx.asset)
            prim_path = f"/World/{fx.name}"
            add_reference_to_stage(usd_path=asset.usd_path,
                                   prim_path=prim_path)
            prim = stage.GetPrimAtPath(prim_path)
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            px, py, pz = (float(v) for v in fx.position)
            qw, qx, qy, qz = (float(v) for v in fx.quaternion)
            sx, sy, sz = _scale3(
                fx.scale if fx.scale is not None else asset.scale)

            # Author orient + scale FIRST with NO translate, so the prim's
            # world AABB == the oriented/scaled mesh anchored at the origin.
            # Many lab USDs (this AUS240) have their geometry FAR from the
            # prim origin -> placing by raw translate puts the visible /
            # cuRobo-collision mesh nowhere near `position` (the bug: place
            # IK_FAIL, board never under it). So measure the real mesh AABB
            # and recenter: `position` then means the mesh FOOTPRINT CENTRE
            # (x,y) with its BASE on z -- regardless of the asset's origin.
            t_op = xf.AddTranslateOp()
            t_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
            xf.AddOrientOp().Set(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))
            xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))

            bbox = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                includedPurposes=[UsdGeom.Tokens.default_,
                                  UsdGeom.Tokens.render],
                useExtentsHint=True,
            ).ComputeWorldBound(prim).ComputeAlignedRange()
            mn, mx = bbox.GetMin(), bbox.GetMax()
            cx, cy = 0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1])
            minz = mn[2]
            t_op.Set(Gf.Vec3d(px - cx, py - cy, pz - minz))

            self._fixture_prims[fx.name] = prim
            print(f"[scene] fixture '{fx.name}' <- {asset.data_info_dir} "
                  f"recentred: mesh AABB c=({cx:.2f},{cy:.2f}) minz={minz:.2f}"
                  f" -> footprint @ ({px:.2f},{py:.2f}) base z={pz:.2f} "
                  f"scale=({sx},{sy},{sz})")

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
        # load_stage = bind cuRobo's UsdHelper to Isaac's live stage so the
        # runtime maybe_sync() can traverse it. We deliberately do NOT call
        # add_world_to_stage(self._world_cfg, ...) here -- that authors a
        # /World/obstacles viz subtree of the INITIAL cuRobo world cfg
        # which (a) nothing reads, (b) ends up double-counted by
        # get_obstacles_from_stage during sync. Pure curobo-example
        # leftover.
        self._usd_help.load_stage(sim.world.stage)

    # ---- hard-coded layout (no IK search) -----------------------------
    def place_for_validation(self, robot, cfg: dict) -> None:
        """Place cube / tables from HARD-CODED config (no IK
        search). Everything is read straight from task_pick_place.yaml so
        you tune it by editing numbers, not code:

          cube_xyz          cube world position [x, y, z]
          grasp_quat        gripper world orientation at grasp [w, x, y, z]
          robot_face_yaw_deg robot yaw at A and B (degrees)
          table_size        board [len_x, len_y, thickness]
          nav_dx            B = A + (nav_dx, 0)

        `robot` is unused -- reachability is now YOUR responsibility (tune
        cube_xyz / grasp_quat / face yaw until the runtime plan succeeds);
        the old auto-search that "guaranteed" it was removed by request.
        """
        ox, oy, oz_cfg = cfg.get("cube_xyz", [0.0, -0.50, 0.84])
        gq = cfg.get("grasp_quat", [-0.2706, 0.6533, -0.6533, 0.2706])
        self.cube_quat = np.asarray(
            cfg.get("cube_quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        tsx, tsy, tsz = cfg.get(
            "table_size", [0.40, 0.40, SLAB_THICKNESS])
        nav_dx = float(cfg.get("nav_dx", NAV_DX))

        # Optional riser block under the plate (see task_pick_place.yaml
        # `riser`). Plate is AUTO-LIFTED by riser_h so its bottom rests on
        # the riser top, while the table top stays at its no-riser height.
        riser_cfg = cfg.get("riser") or {}
        riser_size = tuple(riser_cfg.get("size") or (0.0, 0.0, 0.0))
        riser_h = float(riser_size[2])
        riser_color = tuple(riser_cfg.get("color", (0.85, 0.85, 0.85)))
        oz = oz_cfg + riser_h

        self.grasp_q = np.asarray(gq, dtype=np.float64)

        obj = self.objects[0]
        obj.position = (ox, oy, oz)

        # Board top sits just under the object so it rests flush on the
        # board. Half-extent comes from: USD asset sidecar (size in m) for
        # asset-backed objects, otherwise the procedural cube's scale/size.
        if obj.asset is not None:
            obj_half_z = float(load_object(obj.asset).size[2]) / 2.0
        elif obj.scale is not None:
            obj_half_z = obj.scale[2] / 2.0
        else:
            obj_half_z = obj.size / 2.0
        # `top` = table top z. With no riser this is just plate_bottom -
        # GRASP_CLEARANCE (the old formula). With a riser, the table stays
        # PUT (table top unchanged) and the riser fills the gap up to the
        # lifted plate bottom -- so subtract riser_h to undo the oz lift.
        top = oz - obj_half_z - GRASP_CLEARANCE - riser_h
        tA = self.tables[0]
        tA.center = (ox, oy, top - tsz / 2.0)
        tA.size = (tsx, tsy, tsz)

        self.place_xyz = (ox + nav_dx, oy, oz)
        self.grasp_xyz = (ox, oy, oz)  # A-side point (R-key reset)
        tB = self.tables[1]
        tB.center = (ox + nav_dx, oy, top - tsz / 2.0)
        tB.size = tA.size

        # Riser specs: one per table, footprint = riser size (smaller than
        # the plate), bottom flush with table top. Populated only when the
        # riser is configured; empty list disables the feature.
        self.risers = []
        if riser_h > 0:
            rsx, rsy, rsz = float(riser_size[0]), float(riser_size[1]), riser_h
            riser_cz = top + rsz / 2.0
            self.risers = [
                TableSpec(name="riser_A",
                          center=(ox, oy, riser_cz),
                          size=(rsx, rsy, rsz),
                          color=riser_color),
                TableSpec(name="riser_B",
                          center=(ox + nav_dx, oy, riser_cz),
                          size=(rsx, rsy, rsz),
                          color=riser_color),
            ]

        # Specs are final -> spawn the prims AT these poses (no snap-back).
        self._spawn_props()

        print(f"[scene] HARD-CODED layout: cube @ ({ox:.3f},{oy:.3f},"
              f"{oz:.3f})  grasp_q={list(np.round(self.grasp_q, 4))}  "
              f"face_yaw={cfg.get('robot_face_yaw_deg', -90.0)}deg  "
              f"place @ {self.place_xyz}")

    # ---- queries ------------------------------------------------------
    def object_pose(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        p, q = self._obj_prims[name].get_world_pose()
        return np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)

    def object_prim(self, name: str):
        return self._obj_prims[name]

    def reset_object(self, name: str) -> None:
        """Snap the cube back to its validated A-side grasp pose with zero
        velocity (so a looped run repeats the exact validated scenario)."""
        prim = self._obj_prims[name]
        p = np.array(self.grasp_xyz, dtype=np.float32)
        q = np.array(self.cube_quat, dtype=np.float32)  # same as spawn
        prim.set_world_pose(position=p, orientation=q)
        for setter in ("set_linear_velocity", "set_angular_velocity"):
            try:
                getattr(prim, setter)(np.zeros(3, dtype=np.float32))
            except Exception:  # noqa: BLE001
                pass

    def randomize_plate(self) -> None:
        """Resample the first object's xy uniformly in [-hx, +hx] x [-hy, +hy]
        around _plate_base_xy and update grasp_xyz so the next reset spawns
        the plate at the new pose. Pick targets follow automatically because
        MoveArmTo.grasp_pose reads the live object pose; place_xyz is left
        untouched. No-op when the range is unset / zero."""
        hx, hy = self._plate_random_dxy
        if (hx == 0.0 and hy == 0.0) or self._plate_base_xy is None:
            return
        dx = float(self._plate_rng.uniform(-hx, hx))
        dy = float(self._plate_rng.uniform(-hy, hy))
        bx, by = self._plate_base_xy
        nx, ny = bx + dx, by + dy
        z = self.grasp_xyz[2]
        self.grasp_xyz = (nx, ny, z)
        self.objects[0].position = (nx, ny, z)
        print(f"[scene] plate randomized -> ({nx:.3f}, {ny:.3f}) "
              f"d=({dx:+.3f}, {dy:+.3f})")

    def object_dims(self, name: str):
        """(x, y, z) extents for the cuRobo payload box."""
        for o in self.objects:
            if o.name == name:
                if o.asset is not None:
                    return list(load_object(o.asset).size)
                if o.scale is not None:
                    return list(o.scale)
                return [o.size, o.size, o.size]
        return [0.05, 0.05, 0.05]

    # ---- periodic cuRobo obstacle resync ------------------------------
    # First-sync step. MUST be AFTER the base finishes navigating in
    # pick_place: get_obstacles_from_stage(reference_prim_path=robot) bakes
    # obstacle poses into the BASE frame AT SYNC TIME. If the base then
    # moves before MoveArmTo plans, the stored obstacle frame is stale and
    # cuRobo plans straight through the real obstacle. Step 50 is
    # empirically after FaceYaw + DriveStraight complete. Do NOT lower
    # for navigating tasks. The 1000-step periodic resync handles long
    # tasks where the base drifts during execution. (Sub-classes that
    # know they have no nav phase -- e.g. DemoScene -- override the
    # whole method with an earlier first sync.)
    _SYNC_STEP_FIRST = 50
    _SYNC_STEP_PERIOD = 1000

    def maybe_sync(self, step_index: int, arm, robot_prim_path: str) -> None:
        if not (step_index == self._SYNC_STEP_FIRST
                or step_index % self._SYNC_STEP_PERIOD == 0):
            return
        if step_index == self._last_sync:
            return
        self._last_sync = step_index
        ignore = [robot_prim_path, "/World/defaultGroundPlane", "/curobo",
                  "/World/_room", "/World/_lighting"]
        ignore += [f"/World/{o.name}" for o in self.objects]
        # Tables WERE ignored historically: the thin slab right under the
        # cube blocked the old sideways grasp from descending to a low pose
        # (IK_FAIL). With the new top-down grasp_quat the lower finger now
        # sweeps INTO the board if cuRobo doesn't see it, so the table is
        # included as an obstacle again. If a future low-side grasp comes
        # back, re-evaluate per-grasp instead of toggling here globally.
        obstacles = self._usd_help.get_obstacles_from_stage(
            only_paths=["/World"],
            reference_prim_path=robot_prim_path,
            ignore_substring=ignore,
        ).get_collision_check_world()
        arm.sync_world(obstacles)
        print(f"[scene] cuRobo world resynced @ step {step_index}")
