#
# DemoScene: single-OT-One pick-and-place inside ONE Thorlabs table.
#
# Purpose: a clean, base-doesn't-move scenario for cinematic demo capture.
# Robot stands on one side of a Thorlabs lab table; an OT-One occupies the
# +X half of the deck; the 96-well plate spawns on the -X half of the deck;
# the arm picks the plate from the table-top free area and places it INTO
# the OT-One. base_hold keeps the chassis pinned to robot_start the entire
# task -- no FaceYaw / DriveStraight in the skill list.
#
# Layout (all values are world-frame, z-up, quat (w,x,y,z)):
#   * 1 Thorlabs 75x90 table at (table_cx, table_cy), rotated 90 deg about
#     Z so its long edge (0.9 m) runs along world Y (depth from the robot);
#     top at z = THORLABS_HEIGHT
#   * 1 OT-One centred on the table x-axis, slightly forward of the table
#     centre (OT_Y_FORWARD = 0.05 m) so it reads as the focal point
#   * 96-well plate offset PLATE_X_RIGHT in front-AND-right of the OT-One
#     (robot's right when facing -Y), so the right arm grasps it straight
#     ahead-and-right without crossing the body
#   * place_xyz = OT-One deck centre + cfg.place_offset_xy
#
# Multi-env (visual backdrop, driven by `num_envs` / `env_spacing`):
#   * env 0 is the cuRobo-driven primary -- this is the real, moving scene
#   * envs 1..N-1 each get a STATIC copy of (table + OT-One + plate) placed
#     at (i * env_spacing, 0, 0) -- no robot, no physics on the plate, just
#     visual scale for the demo background. cuRobo ignores them entirely
#     so collision-world stays cheap.
#   * N synchronised moving robots is NOT implemented here -- that needs
#     Isaac-Lab-style batched ArticulationView + Cloner replicate_physics
#     restructuring of robot loading. Ship Phase B separately.
#
# Re-uses OtOneScene's cage trick: replace OT-One's 286 visual meshes with a
# 7-cuboid hollow cage in cuRobo's collision world.
#
# Frame: world frame, quaternions (w, x, y, z).
#

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..asset_lib import load_object
from .bio_scene import (
    BioScene,
    FixtureSpec,
    GRASP_CLEARANCE,
    _scale3,
)


# ---------------------------------------------------------------------------
# Layout constants (re-using the proven OT-One numbers from OtOneScene).
# ---------------------------------------------------------------------------
OT_ONE_QUAT = (0.0, 0.0, 0.0, 1.0)               # 180 deg about world +Z
DECK_TOP_ABOVE_BASE = 0.0635                      # OT-One deck plate above base
OT_ONE_ASSET = "objects/ot_one"
OT_ONE_B_ASSET = "objects/ot_one_b"               # same geometry, distinct path
THORLABS_ASSET = "objects/table-thorlabs-75x90"
# Flat tabletop (densest vertex plane) is at asset Z=-0.0127m, while the
# AABB max is Z=0 because of a 245x203mm, 12.7mm-tall mounting block sitting
# in the middle-left of the breadboard surface. Use the FLAT-top height for
# stacking, not the AABB max -- otherwise anything placed on top floats by
# 12.7mm (and the block protrudes harmlessly into the OT-One's open lower
# frame, which is realistic).
THORLABS_HEIGHT = 0.782                           # FLAT tabletop above base
DEMO_TABLE_Z_SCALE = 1.2                          # demo wants a taller table

# Thorlabs table is rotated 90 deg about Z so its long edge (0.900 m) runs
# along world Y (depth, into the scene away from the robot). Short edge
# (0.759 m) becomes the world-X width. This puts plate + OT-One IN A LINE
# perpendicular to the robot (plate FRONT, OT-One BACK), which both reads
# better on camera and keeps the OT-One off the robot's right shoulder.
TABLE_QUAT_90Z = (0.70710678, 0.0, 0.0, 0.70710678)
# Post-rotation table extents (asset is X=0.900, Y=0.759 unrotated -> swap
# after 90 deg about Z). Keep as constants so the layout maths reads clean.
TABLE_X_AFTER_ROT = 0.759
TABLE_Y_AFTER_ROT = 0.900

# OT-One sits NEAR the table centre, biased SLIGHTLY toward the robot so it
# reads as the focal point of the demo (was previously pushed all the way to
# the back of the table, which made it both visually far and out of arm
# reach). Plate sits forward of the OT-One AND laterally offset onto the
# robot's right side (world -X when robot faces -Y) so the right arm grasps
# straight ahead-and-right without crossing the body.
OT_Y_FORWARD = -0.1    # OT-One centre at (table_cx, table_cy + OT_Y_FORWARD)
PLATE_Y_FORWARD = 0.33 # plate centre y = table_cy + PLATE_Y_FORWARD
PLATE_X_RIGHT = 0.0   # plate centre x = table_cx - PLATE_X_RIGHT (robot's right)

# Demo table sits on the FLOOR (no shim) -- a floating table read as a bug
# in viewport reviews. The earlier 6.35 cm lift was an attempt to match
# OtOneScene's plate z so the validated grasp_quat would IK-OK, but the
# G2 pre-grasp IK_FAILs across all scenes today (separate issue), so the
# lift bought nothing while making the viewport ugly.
TABLE_LIFT_Z = 0.0

# cuRobo cage thicknesses. The old 5 cm base slab left the entire region
# from 5 cm to deck top (6.35 cm) collider-free, and below it the lower
# compartment from 0 to 5 cm was the ONLY block -- side-entry trajectories
# at deck height threaded straight through the OT-One's lower body and the
# deck plate (which is NOT in any of the 26 CollisionAPI meshes either).
# Now: base slab covers OT-One's lower compartment up to JUST BELOW the
# deck top, leaving a ~1 cm slice so the gripper can still descend to
# grasp/place plates that sit ON the deck without the cage blocking the
# legal end pose. Sides above this stay open between corner pillars so
# above-deck side entry remains valid for the pre-grasp pose.
OT_CAGE_PILLAR_T = 0.05
OT_CAGE_BASE_T = DECK_TOP_ABOVE_BASE - 0.01      # ~5.35 cm: just below deck
OT_CAGE_TOP_T = 0.06


def _mirror_otone_asset(env_idx: int) -> str:
    """Pick a USD path for the i-th mirror OT-One. env_0 uses OT_ONE_ASSET,
    env_1 uses OT_ONE_B_ASSET (already shipped as a duplicate to dodge the
    Hydra UpAxis cache bug; see OtOneScene header). env_2+ alternates --
    safe because the cached files now have explicit UpAxis=Z so the cache
    miss is harmless on subsequent references, but we keep alternating
    out of paranoia for non-Z assets a future user might swap in."""
    return OT_ONE_B_ASSET if env_idx % 2 == 1 else OT_ONE_ASSET


class DemoScene(BioScene):
    """Single-table demo: plate on table free area, place into OT-One deck.

    Honoured cfg keys (per-robot overlay):
      cube_xyz             plate spawn world position [x, y, z] (z auto-
                           derived from table+plate geometry; xy honoured)
      grasp_quat           gripper world orientation for both grasp & place
      cube_quat            plate spawn orientation [w,x,y,z]
      robot_face_yaw_deg   robot yaw at the (fixed) stance pose, degrees
      robot_start          [x, y, yaw_rad] -- ALREADY at the work pose;
                           there is no scripted base path in the demo task

    Demo-specific cfg keys (task yaml):
      place_offset_xy      [dx, dy] from OT-One centre to the deck place
                           target (e.g. [0.0, -0.05] -> 5 cm toward the
                           robot from deck centre)
      num_envs             number of copies arranged along +X (visual only
                           for envs >= 1)
      env_spacing          metres between adjacent env_i and env_{i+1}

    Cfg keys IGNORED in this scene (they belong to BioScene's bench):
      nav_dx, table_size, riser, scene.fixtures -- layout here is fixed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # OT-One world AABBs (footprint center xy, base z, world extents)
        # cached at spawn time; keyed by fixture name. Used by maybe_sync
        # to build the cuRobo cage.
        self._otone_aabbs: Dict[
            str, Tuple[float, float, float, float, float, float]] = {}
        # Multi-env knobs (read from cfg in place_for_validation).
        self._num_envs = 1
        self._env_spacing = 0.0
        # Default deck-place offset relative to OT-One centre (overridden
        # by cfg.place_offset_xy). 0 in both axes = dead-centre on the deck.
        self._place_offset_xy: Tuple[float, float] = (0.0, 0.0)

    # ----- main build hook --------------------------------------------------
    def place_for_validation(self, robot, cfg: dict) -> None:
        # cube_xyz is read as the PLATE world xy. Z is derived from the
        # table-top stack (cfg.z is ignored on purpose -- it was tuned for
        # a free-floor cube and won't match the table+OT-One height).
        ox, oy, _ = cfg.get("cube_xyz", [-0.50, -0.40, 0.0])
        gq = cfg.get("grasp_quat", [0.0, 1.0, 0.0, 0.0])
        self.cube_quat = np.asarray(
            cfg.get("cube_quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        self._place_offset_xy = tuple(
            cfg.get("place_offset_xy", self._place_offset_xy))
        self._num_envs = max(1, int(cfg.get("num_envs", 1)))
        self._env_spacing = float(cfg.get("env_spacing", 4.0))

        self.grasp_q = np.asarray(gq, dtype=np.float64)

        # Plate half-extent (z) for clearance math.
        obj = self.objects[0]
        if obj.asset is not None:
            obj_half_z = float(load_object(obj.asset).size[2]) / 2.0
        elif obj.scale is not None:
            obj_half_z = obj.scale[2] / 2.0
        else:
            obj_half_z = obj.size / 2.0

        # Vertical layout is DERIVED from the asset stack. The table is
        # LIFTED by TABLE_LIFT_Z so its top sits at the SAME world z as the
        # OT-One deck does in OtOneScene -- the per-robot grasp_quat was
        # validated at that height. Without the lift the plate would be ~6
        # cm lower and the same grasp_quat IK-FAILs at the lower target.
        table_base_z = TABLE_LIFT_Z
        table_top_z = table_base_z + THORLABS_HEIGHT * DEMO_TABLE_Z_SCALE
        deck_top_z = table_top_z + DECK_TOP_ABOVE_BASE
        plate_z = table_top_z + obj_half_z + GRASP_CLEARANCE

        # Layout (post-rotation table, OT-One CENTRAL / plate FRONT-RIGHT):
        # cube_xyz xy is read as the PLATE position. Table is sized 0.76 m
        # (X) x 0.9 m (Y) after the 90 deg Z rotation; OT-One sits just
        # forward of table centre, plate is offset +x_right and +y_forward
        # from the OT-One so the right arm reaches it straight-ahead-right.
        table_cx = ox + PLATE_X_RIGHT
        table_cy = oy - PLATE_Y_FORWARD
        otone_cx = table_cx
        otone_cy = table_cy + OT_Y_FORWARD

        obj.position = (ox, oy, plate_z)

        # Place target = OT-One deck centre + place_offset_xy, lifted to the
        # deck top + plate half-extent + the same clearance the grasp uses.
        pox, poy = self._place_offset_xy
        place_x = otone_cx + float(pox)
        place_y = otone_cy + float(poy)
        place_z = deck_top_z + obj_half_z + GRASP_CLEARANCE
        self.grasp_xyz = (ox, oy, plate_z)
        self.place_xyz = (place_x, place_y, place_z)

        self.tables = []
        self.risers = []
        self.fixtures = [
            FixtureSpec(name="demo_table", asset=THORLABS_ASSET,
                        position=(table_cx, table_cy, table_base_z),
                        quaternion=TABLE_QUAT_90Z,
                        scale=(1.0, 1.0, DEMO_TABLE_Z_SCALE)),
            FixtureSpec(name="ot_one_demo", asset=OT_ONE_ASSET,
                        position=(otone_cx, otone_cy, table_top_z),
                        quaternion=OT_ONE_QUAT),
        ]

        self._spawn_props()
        self._make_thorlabs_static()
        self._spawn_visual_mirrors(table_cx, table_cy, otone_cx, otone_cy,
                                   table_base_z, table_top_z, plate_z)

        print(f"[demo_scene] env_0: plate @ ({ox:.3f},{oy:.3f},{plate_z:.3f})"
              f" place @ ({place_x:.3f},{place_y:.3f},{place_z:.3f}) "
              f"table_centre=({table_cx:.3f},{table_cy:.3f}) "
              f"OT-One_centre=({otone_cx:.3f},{otone_cy:.3f}) "
              f"deck_top_z={deck_top_z:.3f} num_envs={self._num_envs}"
              f" (mirrors visual-only)")

    # ----- spawn helpers (same precision dance as OtOneScene) ---------------
    def _spawn_fixtures(self) -> None:
        if not self.fixtures:
            return
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import Gf, Usd, UsdGeom

        DBL = UsdGeom.XformOp.PrecisionDouble
        FLT = UsdGeom.XformOp.PrecisionFloat

        def _precision(prim, op_name: str):
            attr = prim.GetAttribute(f"xformOp:{op_name}")
            if attr and attr.IsValid():
                t = attr.GetTypeName().type.typeName
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
            sx, sy, sz = _scale3(
                fx.scale if fx.scale is not None else asset.scale)

            tp = _precision(prim, "translate")
            op = _precision(prim, "orient")
            sp = _precision(prim, "scale")
            t_op = xf.AddTranslateOp(precision=tp)
            t_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
            quat = (Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz)) if op == DBL
                    else Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))
            xf.AddOrientOp(precision=op).Set(quat)
            scale_v = (Gf.Vec3d(sx, sy, sz) if sp == DBL
                       else Gf.Vec3f(sx, sy, sz))
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
            if fx.name.startswith("ot_one"):
                self._otone_aabbs[fx.name] = (
                    float(px), float(py), float(pz),
                    float(mx[0] - mn[0]),
                    float(mx[1] - mn[1]),
                    float(mx[2] - mn[2]),
                )
            print(f"[demo_scene] fixture '{fx.name}' <- {asset.data_info_dir}"
                  f" recentred: AABB c=({cx:.2f},{cy:.2f}) minz={mn[2]:.2f}"
                  f" -> footprint @ ({px:.2f},{py:.2f}) base z={pz:.2f} "
                  f"scale=({sx},{sy},{sz})")

    def _make_thorlabs_static(self) -> None:
        """Pin every Thorlabs table (env_0 + visual mirrors) to kinematic."""
        from pxr import Usd, UsdPhysics
        stage = self._sim.world.stage
        names = ["demo_table"]
        names += [f"env_{i}/demo_table" for i in range(1, self._num_envs)]
        for name in names:
            root = stage.GetPrimAtPath(f"/World/{name}")
            if not root.IsValid():
                continue
            for p in Usd.PrimRange(root):
                if not p.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                rb = UsdPhysics.RigidBodyAPI(p)
                attr = rb.GetKinematicEnabledAttr()
                if not attr:
                    attr = rb.CreateKinematicEnabledAttr(
                        True, writeSparsely=False)
                attr.Set(True)

    # ----- multi-env visual backdrop ---------------------------------------
    def _spawn_visual_mirrors(self, table_cx, table_cy, otone_cx, otone_cy,
                              table_base_z, table_top_z, plate_z) -> None:
        """Drop N-1 STATIC copies of (table + OT-One + plate) along +X.

        Everything is kinematic / static -- no physics on the mirror plates,
        no articulations, no per-tick sync. They're scene decoration to make
        the recording look like a lab full of robots, while env_0 is the
        only one that actually executes the task.

        Why so minimal: a true N-robot synced demo needs Isaac-Lab-style
        Cloner.replicate_physics + a batched ArticulationView pump driven
        from env_0's joint state, plus refactoring G2Robot.load_into to
        spawn under /World/env_0/ instead of /World/. That's a separate
        delivery -- this one just gets the demo OFF the ground.
        """
        if self._num_envs <= 1:
            return
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import Sdf, Usd, UsdPhysics

        stage = self._sim.world.stage
        plate_asset_dir = self.objects[0].asset
        plate_usd = (load_object(plate_asset_dir).usd_path
                     if plate_asset_dir is not None else None)

        for i in range(1, self._num_envs):
            dx = float(i * self._env_spacing)
            env_root = f"/World/env_{i}"
            stage.DefinePrim(env_root, "Xform")
            # Mirror table (same 90 deg Z rotation as env_0).
            t_path = f"{env_root}/demo_table"
            add_reference_to_stage(
                usd_path=load_object(THORLABS_ASSET).usd_path,
                prim_path=t_path)
            self._author_static_xform(
                stage, t_path,
                pos=(table_cx + dx, table_cy, table_base_z),
                quat=TABLE_QUAT_90Z, recenter=True,
                scale=(1.0, 1.0, DEMO_TABLE_Z_SCALE))
            # Mirror OT-One.
            o_asset = _mirror_otone_asset(i)
            o_path = f"{env_root}/ot_one"
            add_reference_to_stage(usd_path=load_object(o_asset).usd_path,
                                   prim_path=o_path)
            self._author_static_xform(
                stage, o_path,
                pos=(otone_cx + dx, otone_cy, table_top_z),
                quat=OT_ONE_QUAT, recenter=True)
            self._set_subtree_kinematic(stage, o_path)
            # Mirror plate (static, kinematic so gravity won't take it).
            if plate_usd is not None:
                p_path = f"{env_root}/plate"
                add_reference_to_stage(usd_path=plate_usd, prim_path=p_path)
                self._author_static_xform(
                    stage, p_path,
                    pos=(self.grasp_xyz[0] + dx, self.grasp_xyz[1], plate_z),
                    quat=tuple(self.cube_quat), recenter=False)
                self._set_subtree_kinematic(stage, p_path)
            print(f"[demo_scene] env_{i} visual mirror @ dx={dx:+.2f}")

    def _author_static_xform(self, stage, prim_path: str, pos, quat,
                             recenter: bool, scale=None) -> None:
        """Author translate+orient on a referenced prim. recenter=True does
        the same AABB-based recentering as _spawn_fixtures so OT-One/table
        prims land at footprint-centre coords (not the raw asset origin).
        scale: float | 3-tuple | None (None -> identity)."""
        from pxr import Gf, Usd, UsdGeom

        DBL = UsdGeom.XformOp.PrecisionDouble
        FLT = UsdGeom.XformOp.PrecisionFloat

        prim = stage.GetPrimAtPath(prim_path)
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()

        def _prec(name: str):
            a = prim.GetAttribute(f"xformOp:{name}")
            if a and a.IsValid():
                t = a.GetTypeName().type.typeName
                if t.endswith("d") or t.endswith("Double"):
                    return DBL
            return FLT

        tp, op = _prec("translate"), _prec("orient")
        t_op = xf.AddTranslateOp(precision=tp)
        t_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
        qw, qx, qy, qz = (float(v) for v in quat)
        qatt = (Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz)) if op == DBL
                else Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))
        xf.AddOrientOp(precision=op).Set(qatt)
        if scale is not None:
            sp = _prec("scale")
            sx, sy, sz = _scale3(scale)
            scale_v = (Gf.Vec3d(sx, sy, sz) if sp == DBL
                       else Gf.Vec3f(sx, sy, sz))
            xf.AddScaleOp(precision=sp).Set(scale_v)

        px, py, pz = (float(v) for v in pos)
        if recenter:
            bbox = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                includedPurposes=[UsdGeom.Tokens.default_,
                                  UsdGeom.Tokens.render],
                useExtentsHint=True,
            ).ComputeWorldBound(prim).ComputeAlignedRange()
            mn, mx = bbox.GetMin(), bbox.GetMax()
            cx, cy = 0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1])
            t_op.Set(Gf.Vec3d(px - cx, py - cy, pz - mn[2]))
        else:
            t_op.Set(Gf.Vec3d(px, py, pz))

    def _set_subtree_kinematic(self, stage, root_path: str) -> None:
        from pxr import Usd, UsdPhysics
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            return
        for p in Usd.PrimRange(root):
            if not p.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            rb = UsdPhysics.RigidBodyAPI(p)
            attr = rb.GetKinematicEnabledAttr()
            if not attr:
                attr = rb.CreateKinematicEnabledAttr(True, writeSparsely=False)
            attr.Set(True)

    # ----- cuRobo cage (lifted from OtOneScene) ----------------------------
    def _build_otone_curobo_cages(self) -> list:
        from curobo.geom.types import Cuboid

        cubes: list = []
        for name, (px, py, pz, w, d, h) in self._otone_aabbs.items():
            pt, bt, tt = OT_CAGE_PILLAR_T, OT_CAGE_BASE_T, OT_CAGE_TOP_T
            for sx in (-1, 1):
                for sy in (-1, 1):
                    cubes.append(Cuboid(
                        name=f"{name}_pillar_{'p' if sx > 0 else 'n'}"
                             f"{'p' if sy > 0 else 'n'}",
                        pose=[
                            px + sx * (w / 2 - pt / 2),
                            py + sy * (d / 2 - pt / 2),
                            pz + h / 2,
                            1.0, 0.0, 0.0, 0.0,
                        ],
                        dims=[pt, pt, h],
                    ))
            cubes.append(Cuboid(
                name=f"{name}_base",
                pose=[px, py, pz + bt / 2, 1.0, 0.0, 0.0, 0.0],
                dims=[w, d, bt],
            ))
            for sx in (-1, 1):
                cubes.append(Cuboid(
                    name=f"{name}_top_{'p' if sx > 0 else 'n'}",
                    pose=[
                        px + sx * (w / 2 - tt / 2),
                        py,
                        pz + h - tt / 2,
                        1.0, 0.0, 0.0, 0.0,
                    ],
                    dims=[tt, d, tt],
                ))
        return cubes

    # cuRobo sync step. OtOneScene uses 50 because pick_place spends those
    # frames in FaceYaw/DriveStraight (the arm hasn't started planning yet).
    # The demo task has NO navigation -- MoveArmTo(pre-grasp) starts planning
    # on step 1, so an unsynced cuRobo world means the first trajectory is
    # planned against an EMPTY world and slices straight through the table.
    # Sync at step 5 (tiny warmup so USD prims are finalised, well before
    # MoveArmTo.start() builds its first plan).
    _SYNC_STEP = 5

    def maybe_sync(self, step_index: int, arm, robot_prim_path: str) -> None:
        """One-shot collision-world build at _SYNC_STEP. cuRobo sees env_0
        geometry only -- visual mirrors are ignored via the env_*/ prefix
        so they don't bloat the collision world."""
        if step_index != self._SYNC_STEP or step_index == self._last_sync:
            return
        self._last_sync = step_index
        ignore = [robot_prim_path, "/World/defaultGroundPlane", "/curobo"]
        ignore += [f"/World/{o.name}" for o in self.objects]
        ignore += [f"/World/{name}" for name in self._otone_aabbs]
        ignore += [f"/World/env_{i}" for i in range(1, self._num_envs)]
        obstacles = self._usd_help.get_obstacles_from_stage(
            only_paths=["/World"],
            reference_prim_path=robot_prim_path,
            ignore_substring=ignore,
        ).get_collision_check_world()

        cage = self._build_otone_curobo_cages()
        if obstacles.cuboid is None:
            obstacles.cuboid = []
        obstacles.cuboid.extend(cage)

        meshes = obstacles.mesh or []
        cuboids = obstacles.cuboid or []
        print(f"[demo_scene] cuRobo world: meshes={len(meshes)} "
              f"cuboids={len(cuboids)} cage={len(cage)}")
        arm.sync_world(obstacles)
        print(f"[demo_scene] cuRobo world resynced @ step {step_index}")
