"""Headless diagnostic: open ot_one*.usd + well_plate_96.usd and report
stage metadata + per-prim geometry/physics summary. Boots SimulationApp
because pxr lives inside isaacsim's extscache (no plain-Python access).

Isaac redirects sys.stdout into its kit log, so we write the diagnostic
report to /tmp/diag_otone.out directly.

Run: uv run python scripts/diag_otone_usd.py
"""

from __future__ import annotations

import sys

from isaacsim import SimulationApp

_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

ASSETS = [
    "assets/objects/ot_one/ot_one.usd",
    "assets/objects/ot_one_b/ot_one.usd",
    "assets/objects/well_plate_96/well_plate_96.usd",
]

OUT = open("/tmp/diag_otone.out", "w")


def _p(*a):
    line = " ".join(str(x) for x in a)
    OUT.write(line + "\n")
    OUT.flush()


def _world_aabb(prim):
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return rng.GetMin(), rng.GetMax()


def _summarize(path: str) -> None:
    _p(f"\n========== {path} ==========")
    stage = Usd.Stage.Open(path)
    if not stage:
        _p("FAILED to open")
        return

    up = UsdGeom.GetStageUpAxis(stage)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    default = stage.GetDefaultPrim()
    _p(f"upAxis              : {up}")
    _p(f"metersPerUnit       : {mpu}")
    _p(f"defaultPrim         : {default.GetPath() if default else '<NONE>'}")

    root = default if default else stage.GetPseudoRoot()
    mn, mx = _world_aabb(root)
    ext = tuple(mx[i] - mn[i] for i in range(3))
    _p(f"AABB min            : ({mn[0]:+.4f}, {mn[1]:+.4f}, {mn[2]:+.4f})")
    _p(f"AABB max            : ({mx[0]:+.4f}, {mx[1]:+.4f}, {mx[2]:+.4f})")
    _p(f"AABB extents (W,D,H): ({ext[0]:.4f}, {ext[1]:.4f}, {ext[2]:.4f})")

    n_mesh = 0
    n_collision = 0
    n_rigid = 0
    n_xform = 0
    for p in Usd.PrimRange(root):
        t = p.GetTypeName()
        if t == "Mesh":
            n_mesh += 1
        if t == "Xform":
            n_xform += 1
        if p.HasAPI(UsdPhysics.CollisionAPI):
            n_collision += 1
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            n_rigid += 1
    _p(f"Mesh prims          : {n_mesh}")
    _p(f"Xform prims         : {n_xform}")
    _p(f"CollisionAPI prims  : {n_collision}")
    _p(f"RigidBodyAPI prims  : {n_rigid}")

    shown = 0
    for p in Usd.PrimRange(root):
        if not p.HasAPI(UsdPhysics.CollisionAPI):
            continue
        attr = p.GetAttribute("physics:approximation")
        approx = (attr.Get() if attr and attr.IsValid() else None) or "default"
        _p(f"  collider[{shown}] {p.GetPath()} approx={approx}")
        shown += 1
        if shown >= 5:
            break


def main() -> int:
    for a in ASSETS:
        try:
            _summarize(a)
        except Exception as exc:  # noqa: BLE001
            _p(f"\n[ERR] {a}: {exc}")
    OUT.close()
    _app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
