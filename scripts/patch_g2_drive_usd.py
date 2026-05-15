"""Build the drive-capable G2 USD by patching the working SE(2) USD.

We can't regenerate the G2 USD from scratch via UrdfConverter — the
gripper FBX/STL meshes referenced by the URDF were never checked in (the
working ``G2_omnipicker/robot.usd`` has them baked into 55MB of layered
USDs from a prior conversion done elsewhere). Regenerating produces an
invisible robot.

Instead we COPY the working asset tree to ``G2_omnipicker_drive/`` and
apply two USD-level patches that make the wheels physically drivable:

1. **Rolling-joint limits.** The 4 rolling joints (idx112/132/142/122)
   are PhysicsRevoluteJoint with ``lower=upper=0`` — effectively locked.
   We override both limits to ±1e6 (wide-open / continuous) on the root
   layer so the wheel can spin.

2. **Wheel collision shapes.** The wheel link2 prims have visuals but no
   collider, so the chassis can't be supported by ground contact. We
   add a child Sphere prim (radius matching the URDF wheel radius) at
   each wheel link2's origin, mark it ``CollisionAPI`` + invisible, and
   give it a high-friction physics material. A sphere is simpler and
   more reliable than a cylinder for rolling on flat ground (single
   contact point, no cylinder-approximation issues).

Idempotent — re-running overwrites the patched USD.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src/bio_sim/assets/robot/G2_omnipicker"
DST_DIR = REPO_ROOT / "src/bio_sim/assets/robot/G2_omnipicker_drive"

# Wheel geometry — keep in sync with agibot_g2_drive_cfg.WHEEL_RADIUS.
WHEEL_RADIUS = 0.07

ROLLING_JOINTS = [
    "/genie/joints/idx112_chassis_lwheel_front_joint2",
    "/genie/joints/idx132_chassis_rwheel_front_joint2",
    "/genie/joints/idx142_chassis_rwheel_rear_joint2",
    "/genie/joints/idx122_chassis_lwheel_rear_joint2",
]

WHEEL_LINK2_PRIMS = [
    "/genie/chassis_lwheel_front_link2",
    "/genie/chassis_rwheel_front_link2",
    "/genie/chassis_lwheel_rear_link2",
    "/genie/chassis_rwheel_rear_link2",
]


def main() -> int:
    if not (SRC_DIR / "robot.usd").exists():
        print(f"ERROR: source USD not found: {SRC_DIR}/robot.usd")
        return 1

    print(f"copying {SRC_DIR.name}/ → {DST_DIR.name}/ ...")
    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)
    shutil.copytree(SRC_DIR, DST_DIR)
    # Drop the cached asset hash so Isaac Lab re-evaluates the USD.
    hash_file = DST_DIR / ".asset_hash"
    if hash_file.exists():
        hash_file.unlink()

    usd_path = DST_DIR / "robot.usd"
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        print(f"ERROR: failed to open {usd_path}")
        return 1

    # Ensure overrides land in the top-level robot.usd, not the layered
    # configuration sub-USDs (which we want to leave bit-identical to the
    # source).
    stage.SetEditTarget(stage.GetRootLayer())

    # ---- Patch 1: rolling joints become unlimited ----------------------------
    print("\n[1/2] opening rolling joints (lower=upper=0 → ±1e6) ...")
    for jpath in ROLLING_JOINTS:
        prim = stage.GetPrimAtPath(jpath)
        if not prim or not prim.IsValid():
            print(f"  [warn] joint not found: {jpath}")
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        # Override the low/high limit attrs on the root layer.
        joint.CreateLowerLimitAttr(-1.0e6, writeSparsely=False)
        joint.CreateUpperLimitAttr(+1.0e6, writeSparsely=False)
        print(f"  [ok] {jpath}")

    # ---- Patch 2: wheel collision spheres ------------------------------------
    print("\n[2/2] adding sphere colliders to wheel link2 prims ...")
    # Single shared physics material with high friction.
    mat_path = "/genie/Looks/wheel_friction_material"
    mat_prim = stage.OverridePrim(mat_path)
    UsdPhysics.MaterialAPI.Apply(mat_prim)
    mat_api = UsdPhysics.MaterialAPI(mat_prim)
    mat_api.CreateStaticFrictionAttr(1.2)
    mat_api.CreateDynamicFrictionAttr(1.0)
    mat_api.CreateRestitutionAttr(0.0)

    for link_path in WHEEL_LINK2_PRIMS:
        link_prim = stage.GetPrimAtPath(link_path)
        if not link_prim or not link_prim.IsValid():
            print(f"  [warn] wheel link not found: {link_path}")
            continue

        coll_path = f"{link_path}/wheel_collider"
        # Use Define so it's typed correctly even if the override prim path
        # didn't exist yet.
        sphere = UsdGeom.Sphere.Define(stage, coll_path)
        sphere.CreateRadiusAttr(WHEEL_RADIUS)
        # Hide the collider; the URDF visual mesh stays the only thing the
        # user sees on the wheel.
        UsdGeom.Imageable(sphere).MakeInvisible()

        coll_prim = sphere.GetPrim()
        UsdPhysics.CollisionAPI.Apply(coll_prim)
        # Bind the friction material to this collider.
        UsdShade.MaterialBindingAPI.Apply(coll_prim).Bind(
            UsdShade.Material(stage.GetPrimAtPath(mat_path)),
            bindingStrength=UsdShade.Tokens.weakerThanDescendants,
            materialPurpose="physics",
        )
        print(f"  [ok] {coll_path}  (r={WHEEL_RADIUS})")

    stage.GetRootLayer().Save()
    print(f"\nwrote {usd_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
