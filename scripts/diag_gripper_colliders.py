"""Headless diagnostic: enumerate every prim under gripper_r_* / gripper_l_*
in the G2 robot.usda and report which ones actually have a working
CollisionAPI (collision_enabled=True + an approximation that produces a
real PhysX shape). The "fingers pass through the plate" symptom is most
likely caused by the fingertip RigidBody links having NO collider geometry
under them -- so PhysX has nothing to test plate contact against.

Run: uv run python scripts/diag_gripper_colliders.py
Output: /tmp/diag_gripper.out
"""

from __future__ import annotations

from isaacsim import SimulationApp

_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

ASSET = "assets/robot/G2/robot.usda"
OUT = open("/tmp/diag_gripper.out", "w")


def _p(*a):
    OUT.write(" ".join(str(x) for x in a) + "\n")
    OUT.flush()


def main():
    stage = Usd.Stage.Open(ASSET)
    _p(f"========== {ASSET} ==========")
    _p(f"defaultPrim = {stage.GetDefaultPrim().GetPath()}")

    # Walk only the gripper subtrees.
    rigid_links = []        # gripper_*_link* prims with RigidBodyAPI
    for prim in stage.Traverse():
        name = prim.GetName()
        if not (name.startswith("gripper_r_") or name.startswith("gripper_l_")):
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_links.append(prim)

    _p(f"\nfound {len(rigid_links)} gripper rigid links")
    for link in rigid_links:
        _p(f"\n--- {link.GetPath()} ---")
        # Walk descendants for any CollisionAPI prim.
        coll_found = []
        for d in Usd.PrimRange(link):
            if not d.HasAPI(UsdPhysics.CollisionAPI):
                continue
            c = UsdPhysics.CollisionAPI(d)
            enabled = c.GetCollisionEnabledAttr().Get()
            approx = None
            if d.HasAPI(UsdPhysics.MeshCollisionAPI):
                mc = UsdPhysics.MeshCollisionAPI(d)
                a = mc.GetApproximationAttr()
                approx = a.Get() if a else None
            tname = d.GetTypeName()
            # Try to read mesh vertex count.
            vcount = None
            if tname == "Mesh":
                m = UsdGeom.Mesh(d)
                v = m.GetPointsAttr().Get()
                vcount = len(v) if v else 0
            coll_found.append((d.GetPath(), tname, enabled, approx, vcount))
        if not coll_found:
            _p("  !! NO CollisionAPI prim under this rigid link")
            continue
        for path, tname, enabled, approx, vcount in coll_found:
            extra = f" verts={vcount}" if vcount is not None else ""
            _p(f"  collider: {path}  type={tname}  enabled={enabled}  "
               f"approx={approx}{extra}")


main()
_app.close()
