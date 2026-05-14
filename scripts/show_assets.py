"""Spawn every lab asset side-by-side in Isaac Sim for visual inspection.

No robot, no planning, no physics interaction — just a static gallery. The
Thorlabs bench is laid out in the centre; the bench-scale items (Sartorius
balance, IKA hotplate, 500 mL beaker, bottle) sit on top of it; the PureLab
glovebox is parked off to the side because it dominates the scene.

Y-up assets (beaker, IKA, bottle) are rotated +90° around X so they stand
upright in our Z-up world.

Run:
    python scripts/show_assets.py
    python scripts/show_assets.py --headless   # logs only, no viewer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from isaaclab.app import AppLauncher  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab-asset gallery")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


_ASSETS = REPO_ROOT / "src" / "bio_sim" / "assets"
_Y_UP_TO_Z_UP = (0.70710678, 0.70710678, 0.0, 0.0)  # wxyz: +90° around X
# OT's converted glb has glb-+X as the base direction (the asset is on its
# side after a Y→Z swap); rotating +90° around Y instead puts glb-+X at world-Z.
_OT_BASE_DOWN = (0.70710678, 0.0, 0.70710678, 0.0)  # wxyz: +90° around Y
_MM_TO_M = (0.001, 0.001, 0.001)  # mm-unit USDs need this to render at real size
_CM_TO_M = (0.01, 0.01, 0.01)     # cm-unit USDs (OT.usdz)


# (name, path-under-assets, world position, rot, scale)
# - rot is wxyz or None for identity
# - scale is (sx, sy, sz) or None for unit scale
_ITEMS = [
    # Big infrastructure, parked off to the side. Glovebox and Sartorius are
    # authored with metersPerUnit=0.001 but raw vertex values in mm — Isaac
    # Sim ignores metersPerUnit on referenced layers, so we scale by 0.001
    # manually to get a sane physical size. The upstream glovebox uses a
    # frosted-glass material with opacity 0.49 — we override to opaque so its
    # geometry is visible.
    ("glovebox", "infrastructure/glovebox/purelab/glovebox-inert-purelab.usda",
        (-3.0, 0.0, 0.0), None, _MM_TO_M),

    # Bench, centred at the origin (its top ends up at world z = 0.79).
    ("bench", "infrastructure/tables/table-thorlabs-75x90/table.usda",
        (0.0, 0.0, 0.79), None, None),

    # Items on the bench. The bench's top spans world x ≈ [-0.176, 0.724] and
    # y ≈ [-0.379, 0.379], all at z = 0.79.
    ("sartorius", "equipment/balance/sartorius/sartorius_entris_ii.usda",
        (0.55, -0.20, 0.797), None, _MM_TO_M),

    # IKA: Y-up metadata is wrong; geometry is actually Z-up. Bbox extends
    # x∈[-0.164, 0.185], y∈[-0.165, 0.175], z∈[-0.004, 0.074], so we drop the
    # spawn z to 0.794 (flush with the bench top at 0.79) and centre the x/y
    # so the asset doesn't hang off the bench edge. Material override because
    # the external texture pack (../LabTools/Scale-IKA.usdz) is missing.
    ("ika", "equipment/balance-heater-stirrer/scale-IKA.usda",
        (0.30,  0.15, 0.794), None, None),

    # Beaker uses OmniGlass material (transparent) — overridden to a solid
    # tint so it's obvious where it sits in the "what do I have" view.
    ("beaker", "labware/beaker500ml/beaker-500ml.usda",
        (0.10, -0.20, 0.827), _Y_UP_TO_Z_UP, None),

    ("bottle", "labware/bottle/bottle_new.usda",
        (0.10,  0.20, 0.790), _Y_UP_TO_Z_UP, None),

    # OT (GUI-converted from mix.bio_and_the_ot.one.glb). 286 meshes. The
    # GUI converter applied the Y-up→Z-up correction inline, so we use
    # identity rotation. Raw bbox y∈[-252, +378] stage units → after scale
    # 0.001 and auto-Y→Z, the asset extends 0.252 m below / 0.378 m above
    # the spawn origin. Spawn z = 0.252 lands the base on the ground.
    ("ot", "equipment/ot_one.usd",
        (-1.5, -1.5, 0.252), None, (0.001, 0.001, 0.001)),
]


def main() -> int:
    args = parse_args()
    launcher = AppLauncher(args)
    sim_app = launcher.app

    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.assets import AssetBaseCfg

    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )
    scene_cfg.ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    missing = []
    for name, rel_path, pos, rot, scale in _ITEMS:
        usd = _ASSETS / rel_path
        if not usd.exists():
            missing.append((name, usd))
            continue
        spawn_kwargs = {"usd_path": str(usd)}
        if scale is not None:
            spawn_kwargs["scale"] = scale
        # Override materials on assets that would otherwise be hard to see:
        #   - IKA's external texture is missing (renders black)
        #   - glovebox uses frosted glass (opacity 0.49)
        #   - beaker uses OmniGlass (fully transparent)
        _material_override = {
            "ika":      sim_utils.PreviewSurfaceCfg(diffuse_color=(0.65, 0.65, 0.68), metallic=0.3, roughness=0.4),
            "glovebox": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.85, 0.88), roughness=0.6),
            "beaker":   sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.55, 0.90), roughness=0.4),
            # ot_one.usd was GUI-converted from a glb with textures embedded
            # inline — no override needed; original PBR materials should
            # render correctly.
        }
        if name in _material_override:
            spawn_kwargs["visual_material"] = _material_override[name]
        cfg = AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=sim_utils.UsdFileCfg(**spawn_kwargs),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=pos, rot=rot or (1.0, 0.0, 0.0, 0.0)
            ),
        )
        setattr(scene_cfg, name, cfg)
        marks = []
        if rot: marks.append("Y→Z")
        if scale: marks.append(f"scale={scale[0]}")
        suffix = f" [{', '.join(marks)}]" if marks else ""
        print(f"  + {name:10s}  pos={pos}{suffix}  ({rel_path})")

    if missing:
        print("\nmissing assets (skipped):", file=sys.stderr)
        for name, path in missing:
            print(f"  - {name}: {path}", file=sys.stderr)

    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    print("\nLoaded. Close the window (or Ctrl-C in headless) to exit.")
    try:
        while sim_app.is_running():
            sim.step()
            scene.update(sim.get_physics_dt())
    finally:
        sim_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
