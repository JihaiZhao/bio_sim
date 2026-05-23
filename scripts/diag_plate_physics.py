"""Headless diagnostic: dump the well_plate_96 USD's baked physics AND
the post-spawn live state (collider geometry, bound physicsMaterial after
our strongerThanDescendants binding, mesh structure). Replicates the
exact spawn path bio_scene._spawn_usd_object uses, so the live readout
reflects what PhysX will actually see at sim time.

Run: uv run python scripts/diag_plate_physics.py
Output: /tmp/diag_plate.out (stdout is captured by Isaac kit log).
"""

from __future__ import annotations

from isaacsim import SimulationApp

_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

ASSET = "assets/objects/well_plate_96/well_plate_96.usd"
OUT = open("/tmp/diag_plate.out", "w")


def _p(*a):
    OUT.write(" ".join(str(x) for x in a) + "\n")
    OUT.flush()


def _aabb(prim):
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return r.GetMin(), r.GetMax()


def main():
    stage = Usd.Stage.Open(ASSET)
    _p(f"========== {ASSET} ==========")
    _p(f"upAxis        : {UsdGeom.GetStageUpAxis(stage)}")
    _p(f"metersPerUnit : {UsdGeom.GetStageMetersPerUnit(stage)}")
    _p(f"defaultPrim   : {stage.GetDefaultPrim().GetPath()}")

    rb_count = mass_count = coll_count = mat_count = 0
    _p("\n--- RigidBody / Mass prims ---")
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb_count += 1
            rb = UsdPhysics.RigidBodyAPI(prim)
            kin = rb.GetKinematicEnabledAttr().Get()
            _p(f"  RigidBody : {prim.GetPath()}  kinematic={kin}")
            mn, mx = _aabb(prim)
            _p(f"    AABB    : min={tuple(round(v,4) for v in mn)} "
               f"max={tuple(round(v,4) for v in mx)}")
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass_count += 1
            m = UsdPhysics.MassAPI(prim)
            _p(f"  Mass      : {prim.GetPath()}")
            _p(f"    mass    : {m.GetMassAttr().Get()}")
            _p(f"    density : {m.GetDensityAttr().Get()}")
            _p(f"    com     : {m.GetCenterOfMassAttr().Get()}")
            _p(f"    diaI    : {m.GetDiagonalInertiaAttr().Get()}")

    _p("\n--- CollisionAPI prims ---")
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            coll_count += 1
            c = UsdPhysics.CollisionAPI(prim)
            mc = UsdPhysics.MeshCollisionAPI(prim) if prim.HasAPI(
                UsdPhysics.MeshCollisionAPI) else None
            approx = (mc.GetApproximationAttr().Get()
                      if mc and mc.GetApproximationAttr() else None)
            mat_rel = c.GetSimulationOwnerRel()
            _p(f"  Collider  : {prim.GetPath()}  "
               f"approx={approx}  type={prim.GetTypeName()}")
            # Material binding
            bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(
                materialPurpose="physics")
            mb = bound[0].GetPath() if bound and bound[0] else None
            _p(f"    physicsMaterial : {mb}")

    _p("\n--- PhysicsMaterials in stage ---")
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            mat_count += 1
            m = UsdPhysics.MaterialAPI(prim)
            _p(f"  Material  : {prim.GetPath()}")
            _p(f"    staticFriction  : {m.GetStaticFrictionAttr().Get()}")
            _p(f"    dynamicFriction : {m.GetDynamicFrictionAttr().Get()}")
            _p(f"    restitution     : {m.GetRestitutionAttr().Get()}")
            _p(f"    density         : {m.GetDensityAttr().Get()}")

    _p(f"\nsummary: rb={rb_count} mass={mass_count} coll={coll_count} mat={mat_count}")


def live_spawn_probe():
    """Spawn the plate into a live Isaac stage the same way bio_scene does
    (add_reference_to_stage + SingleRigidPrim + UsdShade binding) and dump
    the LIVE collider + bound material on /World/plate AND its mesh
    sub-prims. This is the only way to confirm the strongerThanDescendants
    physics binding actually wins over the baked /World/Looks/DefaultMaterial
    binding on the mesh sub-prim."""
    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.prims import SingleRigidPrim
    from isaacsim.core.utils.stage import add_reference_to_stage

    _p("\n\n========== LIVE SPAWN PROBE (replicates bio_scene path) ==========")
    world = World(physics_dt=1.0 / 240.0, rendering_dt=1.0 / 60.0)
    stage = world.stage

    grip_mat = PhysicsMaterial(
        prim_path="/World/PhysicsMaterials/grip",
        static_friction=5.2, dynamic_friction=5.0, restitution=0.1)

    prim_path = "/World/plate"
    add_reference_to_stage(usd_path=ASSET, prim_path=prim_path)
    rigid = SingleRigidPrim(prim_path=prim_path, name="plate",
                            position=np.array([0.0, 0.0, 0.5]),
                            orientation=np.array([1.0, 0.0, 0.0, 0.0]))
    world.scene.add(rigid)
    prim = stage.GetPrimAtPath(prim_path)

    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(grip_mat.material,
                 bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                 materialPurpose="physics")
    _p(f"applied strongerThanDescendants binding @ {prim_path}")
    _p(f"  -> grip_mat: {grip_mat.material.GetPath()}")

    # Force collider approximation -> convexHull (matches bio_scene patch).
    for p in Usd.PrimRange(prim):
        if not p.HasAPI(UsdPhysics.MeshCollisionAPI):
            continue
        mc = UsdPhysics.MeshCollisionAPI(p)
        a = mc.GetApproximationAttr() or mc.CreateApproximationAttr()
        a.Set(UsdPhysics.Tokens.convexHull)
        _p(f"  -> set approximation=convexHull on {p.GetPath()}")

    # Walk the spawned subtree and report.
    _p("\n--- post-spawn subtree ---")
    for p in Usd.PrimRange(prim):
        apis = [a for a in p.GetAppliedSchemas()
                if any(k in a for k in
                       ("Physics", "Mass", "RigidBody", "Collision",
                        "MaterialBinding"))]
        line = f"  {p.GetPath()}  type={p.GetTypeName()}"
        if apis:
            line += f"  APIs={apis}"
        _p(line)
        if p.HasAPI(UsdPhysics.CollisionAPI):
            c = UsdPhysics.CollisionAPI(p)
            mc = (UsdPhysics.MeshCollisionAPI(p)
                  if p.HasAPI(UsdPhysics.MeshCollisionAPI) else None)
            approx = (mc.GetApproximationAttr().Get() if mc else None)
            enabled = c.GetCollisionEnabledAttr().Get()
            _p(f"    collision_enabled={enabled}  approx={approx}")
        if p.HasAPI(UsdPhysics.MeshCollisionAPI):
            # Resolve bound physics material at this prim.
            mb = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial(
                materialPurpose="physics")
            mp = mb[0].GetPath() if mb and mb[0] else None
            _p(f"    *** ComputeBoundMaterial(physics) -> {mp}")
            # Also check what staticFriction/dynamicFriction PhysX will see.
            if mp:
                m_prim = stage.GetPrimAtPath(mp)
                if m_prim.HasAPI(UsdPhysics.MaterialAPI):
                    api = UsdPhysics.MaterialAPI(m_prim)
                    _p(f"    *** MaterialAPI: static={api.GetStaticFrictionAttr().Get()} "
                       f"dynamic={api.GetDynamicFrictionAttr().Get()} "
                       f"restitution={api.GetRestitutionAttr().Get()}")
                else:
                    _p(f"    *** WARN bound material has NO MaterialAPI applied "
                       f"-> PhysX uses default (mu~0.5)")
        if p.GetTypeName() == "Mesh":
            m = UsdGeom.Mesh(p)
            v = m.GetPointsAttr().Get()
            f = m.GetFaceVertexCountsAttr().Get()
            _p(f"    Mesh: vertices={len(v) if v else 0} faces={len(f) if f else 0}")

    world.stop()


main()
live_spawn_probe()
_app.close()
