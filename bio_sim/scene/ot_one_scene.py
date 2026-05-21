#
# OtOneScene: pick-and-place across TWO OT-Ones on Thorlabs lab tables.
#
# Scene composition (FIXED, ignores scene.fixtures + scene.objects in cfg):
#   * Two Thorlabs 75x90 tables (assets/objects/table-thorlabs-75x90/), one
#     at marker A and one at marker B, nav_dx apart along world +X.
#   * Two Opentrons OT-One frames (assets/objects/ot_one/), one mounted on
#     each table. Each has 286 mesh colliders for cuRobo obstacle world.
#   * One 96-well plate (assets/objects/well_plate_96/) spawned ON the
#     deck of OT-One A. OT-One B starts EMPTY -- the pick-place runner
#     drops the plate inside it.
#
# No bio_optica stainers, no risers -- those live in BioScene's task.
#
# Subclasses BioScene so the entire pick-place API (markers, grasp_q,
# grasp_xyz, place_xyz, object_pose/reset_object, maybe_sync) is inherited;
# only the LAYOUT is overridden in place_for_validation.
#
# Frame: world frame, quaternions (w, x, y, z).
#

from __future__ import annotations

import numpy as np

from ..asset_lib import load_object
from .bio_scene import (
    BioScene,
    FixtureSpec,
    GRASP_CLEARANCE,
    NAV_DX,
)


# ---------------------------------------------------------------------------
# OT-One USD geometry constants.
#   * source units : mm labelled as m  -> sidecar scale 0.001
#   * source up    : Y                 -> rotate +90 deg about X to bring it
#                                         to Z-up. Verified: asset Y_min has
#                                         feet/L-pieces, Y_max has 550 mm
#                                         X/Y rails (gantry top), so +Y is
#                                         up in the asset frame.
#   * deck offset  : 0.0635 m above the OT-One base (Object_222/Object_220
#                    deck plate top vs AABB min_y) after recentering.
# ---------------------------------------------------------------------------

# Asset's upAxis is now claimed as Z (we re-saved both ot_one*.usd with
# UpAxis=Z), so Isaac no longer applies the implicit Y-up -> Z-up rotation
# that was firing inconsistently across the two references. The asset's
# geometry is STILL internally Y-up, so we apply +90 deg about X EXPLICITLY
# here to bring it upright. Bake any additional horizontal yaw into this
# same quat (e.g. compose with +90 deg about Z if you want it rotated 90
# in the XY plane).
OT_ONE_QUAT = (0.0, 0.0, 0.0, 1.0)   # 180 deg about world +Z
DECK_TOP_ABOVE_BASE = 0.0635
# How far the plate sits IN FRONT of the OT-One deck centre, in world +Y
# (toward the robot when robot faces -Y). Deck is ~0.6 m square so the
# plate can be offset up to ~0.26 m before it falls off; 0.2 m keeps it
# clearly forward while leaving margin. OT-One itself stays centred on the
# table -- only the plate's xy is offset.
PLATE_FORWARD_OFFSET_Y = 0.20
OT_ONE_ASSET = "objects/ot_one"
# Hydra applies a one-shot UpAxis auto-rotation on the FIRST reference of a
# Y-up USD file -- subsequent references to the same file reuse the cached
# render-state and SKIP the auto-rotation, so two references of ot_one.usd
# come out rotated 90 deg apart (USD attrs are identical; only render path
# differs). Workaround: point the second OT-One at a duplicate file so it's
# a "different asset" from Hydra's POV. Same geometry as objects/ot_one/.
OT_ONE_B_ASSET = "objects/ot_one_b"

# Thorlabs 75x90 lab table: 0.9 m x 0.76 m, 0.795 m tall, Z-up, m-per-unit
# 1.0. After _spawn_fixtures recentering, the BASE is at the supplied pz
# and the TOP is at pz + 0.795.
THORLABS_ASSET = "objects/table-thorlabs-75x90"
THORLABS_HEIGHT = 0.795


class OtOneScene(BioScene):
    def place_for_validation(self, robot, cfg: dict) -> None:
        """Build the two-table OT-One layout from cfg.

        Cfg keys honoured (same names as BioScene):
          cube_xyz             plate world position [x, y, z]
          grasp_quat           gripper world orientation [w,x,y,z]
          cube_quat            plate spawn orientation
          robot_face_yaw_deg   robot yaw at A and B (degrees)
          nav_dx               world +X offset from A to B

        Cfg keys DROPPED in this scene (they belong to BioScene's bench):
          table_size           tables are Thorlabs USDs, not FixedCuboids
          riser                no riser -- plate rests directly on the
                               OT-One deck plate (with GRASP_CLEARANCE)
          scene.fixtures       fixed layout = two OT-Ones; user fixtures
                               from cfg are NOT honoured here
        """
        ox, oy, _oz_cfg = cfg.get("cube_xyz", [0.0, -0.50, 0.84])
        gq = cfg.get("grasp_quat", [-0.2706, 0.6533, -0.6533, 0.2706])
        self.cube_quat = np.asarray(
            cfg.get("cube_quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        face_yaw = float(np.radians(cfg.get("robot_face_yaw_deg", -90.0)))
        nav_dx = float(cfg.get("nav_dx", NAV_DX))

        self.grasp_q = np.asarray(gq, dtype=np.float64)

        # Plate half-extent (z) lookup, same convention as BioScene.
        obj = self.objects[0]
        if obj.asset is not None:
            obj_half_z = float(load_object(obj.asset).size[2]) / 2.0
        elif obj.scale is not None:
            obj_half_z = obj.scale[2] / 2.0
        else:
            obj_half_z = obj.size / 2.0

        # Stack z layout is DERIVED, not from cfg: table BASE sits on the
        # floor (z=0), OT-One sits on the table top, plate sits on the
        # OT-One deck plate. We IGNORE cfg's cube_xyz.z (it was tuned for
        # the BioScene 0.4 m FixedCuboid; thorlabs is 0.795 m so the
        # numbers don't transfer). cfg.cube_xyz xy is still honoured.
        table_base_z = 0.0
        table_top_z = THORLABS_HEIGHT
        deck_top_z = table_top_z + DECK_TOP_ABOVE_BASE
        oz = deck_top_z + obj_half_z + GRASP_CLEARANCE
        # Plate xy is OT-One xy + a forward offset (toward robot, +Y world).
        # Table and OT-One stay at cfg's cube_xyz xy (centred on world xy);
        # only the plate sits closer to the front of the deck so the arm
        # doesn't need to reach all the way to the deck centre.
        plate_x = ox
        plate_y = oy + PLATE_FORWARD_OFFSET_Y
        obj.position = (plate_x, plate_y, oz)

        # Markers: A at world origin, B at (nav_dx, 0). Same as BioScene.
        self._marker_by_name["A"].position = (0.0, 0.0, 0.02)
        self._marker_by_name["A"].yaw = face_yaw
        self._marker_by_name["B"].position = (nav_dx, 0.0, 0.02)
        self._marker_by_name["B"].yaw = face_yaw
        # Grasp/place at the actual plate xy (incl. forward offset), not at
        # the OT-One/table centre. Same offset applies at marker B so the
        # release point lines up with the deck-front position on table B.
        self.grasp_xyz = (plate_x, plate_y, oz)
        self.place_xyz = (plate_x + nav_dx, plate_y, oz)

        # Wipe the BioScene-style FixedCuboid tables, BioScene's riser
        # block, and any cfg-supplied fixtures (bio_optica etc.). This
        # scene's layout is fixed: two Thorlabs tables + two OT-Ones.
        self.tables = []
        self.risers = []
        self.fixtures = [
            FixtureSpec(name="table_A",  asset=THORLABS_ASSET,
                        position=(ox, oy, table_base_z)),
            FixtureSpec(name="table_B",  asset=THORLABS_ASSET,
                        position=(ox + nav_dx, oy, table_base_z)),
            FixtureSpec(name="ot_one_A", asset=OT_ONE_ASSET,
                        position=(ox, oy, table_top_z),
                        quaternion=OT_ONE_QUAT),
            FixtureSpec(name="ot_one_B", asset=OT_ONE_B_ASSET,
                        position=(ox + nav_dx, oy, table_top_z),
                        quaternion=OT_ONE_QUAT),
        ]

        self._spawn_props()
        self._make_thorlabs_static()

        print(f"[ot_one_scene] layout: plate @ ({ox:.3f},{oy:.3f},{oz:.3f}) "
              f"grasp_q={list(np.round(self.grasp_q, 4))} "
              f"face_yaw={cfg.get('robot_face_yaw_deg', -90.0)}deg "
              f"place_xyz={self.place_xyz}  "
              f"deck_top_z={deck_top_z:.3f}  table_top_z={table_top_z:.3f}  "
              f"table_base_z={table_base_z:.3f} (floats above floor by this much)")

    def _spawn_fixtures(self) -> None:
        """Same recentre-by-AABB logic as BioScene._spawn_fixtures, with one
        extra detail this scene's assets need: the Thorlabs table USD ships
        xformOp:translate / orient / scale pre-declared in DOUBLE precision
        (quatd / double3), so AddOrientOp() and AddScaleOp() with their
        default Float precision raise a USD typeName/precision error. We
        probe each xformOp:* attribute on the freshly-referenced prim,
        match the existing precision, and feed Gf.Quatd / Gf.Vec3d where
        appropriate. Fixtures that DON'T pre-declare these ops (bio_optica,
        ot_one) keep going through Float just like BioScene does.
        """
        if not self.fixtures:
            return
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import Gf, Usd, UsdGeom

        DBL = UsdGeom.XformOp.PrecisionDouble
        FLT = UsdGeom.XformOp.PrecisionFloat

        def _precision(prim, op_name: str):
            attr = prim.GetAttribute(f"xformOp:{op_name}")
            if attr and attr.IsValid():
                t = attr.GetTypeName().type.typeName  # "GfQuatd" / "GfVec3d" / ...
                if t.endswith("d") or t.endswith("Double"):
                    return DBL
            return FLT

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
            s = fx.scale if fx.scale is not None else asset.scale

            tp, op, sp = _precision(prim, "translate"), _precision(prim, "orient"), _precision(prim, "scale")
            t_op = xf.AddTranslateOp(precision=tp)
            t_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
            quat = (Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz)) if op == DBL
                    else Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))
            xf.AddOrientOp(precision=op).Set(quat)
            scale_v = (Gf.Vec3d(float(s), float(s), float(s)) if sp == DBL
                       else Gf.Vec3f(float(s), float(s), float(s)))
            xf.AddScaleOp(precision=sp).Set(scale_v)

            bbox = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                includedPurposes=[UsdGeom.Tokens.default_,
                                  UsdGeom.Tokens.render],
                useExtentsHint=True,
            ).ComputeWorldBound(prim).ComputeAlignedRange()
            mn, mx = bbox.GetMin(), bbox.GetMax()
            cx, cy = 0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1])
            t_op.Set(Gf.Vec3d(px - cx, py - cy, pz - mn[2]))

            self._fixture_prims[fx.name] = prim
            print(f"[ot_one_scene] fixture '{fx.name}' <- {asset.data_info_dir} "
                  f"recentred: AABB c=({cx:.2f},{cy:.2f}) minz={mn[2]:.2f} "
                  f"-> footprint @ ({px:.2f},{py:.2f}) base z={pz:.2f} scale={s}")

    def _make_thorlabs_static(self) -> None:
        """Disable physics on the Thorlabs table prims.

        The Thorlabs USD declares PhysicsArticulationRootAPI on the outer
        Xform and PhysicsRigidBodyAPI on the inner `table` Xform. Without
        intervention the rigid body would free-fall under gravity. We set
        `physics:kinematicEnabled = True` on every RigidBody we find under
        each table prim so it stays put while still acting as a collider.
        """
        from pxr import Usd, UsdPhysics
        stage = self._sim.world.stage
        for name in ("table_A", "table_B"):
            root = stage.GetPrimAtPath(f"/World/{name}")
            if not root.IsValid():
                continue
            for p in Usd.PrimRange(root):
                if not p.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                rb = UsdPhysics.RigidBodyAPI(p)
                attr = rb.GetKinematicEnabledAttr()
                if not attr:
                    attr = rb.CreateKinematicEnabledAttr(True, writeSparsely=False)
                attr.Set(True)

    def maybe_sync(self, step_index: int, arm, robot_prim_path: str) -> None:
        """Same as BioScene.maybe_sync. The two OT-One fixtures and the
        two Thorlabs tables are NOT in the ignore list -- they become
        cuRobo obstacles. Markers + plate stay ignored (graspable target /
        navigation visualisation).
        """
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
        print(f"[ot_one_scene] cuRobo world resynced @ step {step_index}")
