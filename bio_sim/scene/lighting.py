#
# Lighting & render presets for bio_sim demo capture.
#
# Authors UsdLux dome + distant + (optional) rect lights under
# {env_root}/_lighting/ and toggles RTX render mode via carb.settings.
#
# Three built-in presets approximate the reference video moods:
#   studio  -> clean white photo studio (image 1)
#   warm    -> golden-hour warm interior (image 2)
#   night   -> dim low-key evening      (image 3)
#
# Default render mode is RealTime (fast viewport for demo runs); switch
# to PathTracing manually (e.g. via the P key) before recording.
#
# Frame: world frame, Z-up. Elevation is degrees above the horizon;
# azimuth is degrees CCW from +X around +Z.
#

from __future__ import annotations

import math
import os
from typing import Optional


_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _hdri(name: str) -> str:
    return os.path.join(_PROJECT_ROOT, "assets", "lighting", name)


# Tunable starting points; iterate in the viewport until the mood matches.
PRESETS: dict[str, dict] = {
    "studio": {
        "dome": {"hdri": _hdri("photo_studio.hdr"), "intensity": 600.0,
                 "color_temperature": 6500.0, "rotation_deg": [0.0, 0.0, 0.0]},
        "sun":  {"intensity": 2000.0, "color_temperature": 5500.0,
                 "angle_deg": 8.0, "elevation_deg": 45.0, "azimuth_deg": 120.0},
        "fill": None,
    },
    "warm": {
        "dome": {"hdri": _hdri("sunflowers.hdr"), "intensity": 400.0,
                 "color_temperature": 3500.0, "rotation_deg": [0.0, 0.0, 0.0]},
        "sun":  {"intensity": 1500.0, "color_temperature": 3000.0,
                 "angle_deg": 10.0, "elevation_deg": 30.0, "azimuth_deg": 90.0},
        "fill": {"intensity": 500.0, "color_temperature": 3200.0,
                 "size": [0.8, 1.2], "position": [-1.0, 0.0, 1.5]},
    },
    "night": {
        "dome": {"hdri": _hdri("car_light.hdr"), "intensity": 80.0,
                 "color_temperature": 2800.0, "rotation_deg": [0.0, 0.0, 0.0]},
        "sun":  {"intensity": 300.0, "color_temperature": 2400.0,
                 "angle_deg": 15.0, "elevation_deg": 20.0, "azimuth_deg": 160.0},
        "fill": {"intensity": 200.0, "color_temperature": 2700.0,
                 "size": [0.6, 0.9], "position": [-0.8, 0.0, 1.2]},
    },
}


# ---- math helpers ---------------------------------------------------------


def _sun_direction(elevation_deg: float, azimuth_deg: float
                   ) -> tuple[float, float, float]:
    """Unit vector that the LIGHT travels along. Sun at (elev, az) emits
    toward the origin -- elev=90 -> straight down (0, 0, -1); elev=0,
    az=0 -> horizontal -X (-1, 0, 0)."""
    el = math.radians(elevation_deg)
    az = math.radians(azimuth_deg)
    return (-math.cos(el) * math.cos(az),
            -math.cos(el) * math.sin(az),
            -math.sin(el))


def _quat_aim_neg_z(dx: float, dy: float, dz: float
                    ) -> tuple[float, float, float, float]:
    """Quat (w, x, y, z) that rotates local -Z onto the unit vector
    (dx, dy, dz). UsdLux.DistantLight + RectLight emit along local -Z."""
    sx, sy, sz = 0.0, 0.0, -1.0
    ax = sy * dz - sz * dy
    ay = sz * dx - sx * dz
    az = sx * dy - sy * dx
    dot = sx * dx + sy * dy + sz * dz
    if dot > 0.9999:
        return (1.0, 0.0, 0.0, 0.0)
    if dot < -0.9999:
        return (0.0, 1.0, 0.0, 0.0)
    w = 1.0 + dot
    n = math.sqrt(w * w + ax * ax + ay * ay + az * az)
    return (w / n, ax / n, ay / n, az / n)


# ---- USD authoring --------------------------------------------------------


def _ensure_xform(stage, path: str):
    from pxr import UsdGeom
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    return prim


def _set_xform(stage, prim_path: str, translate=None, orient_quat=None,
               rotate_xyz_deg=None) -> None:
    from pxr import Gf, UsdGeom
    prim = stage.GetPrimAtPath(prim_path)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    if translate is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*translate))
    if orient_quat is not None:
        w, x, y, z = orient_quat
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    elif rotate_xyz_deg is not None:
        rx, ry, rz = rotate_xyz_deg
        xf.AddRotateXYZOp().Set(Gf.Vec3f(float(rx), float(ry), float(rz)))


def _author_dome(stage, prim_path: str, hdri: str, intensity: float,
                 color_temperature: float, rotation_deg=None) -> None:
    from pxr import Sdf, UsdLux
    light = UsdLux.DomeLight.Define(stage, prim_path)
    light.CreateIntensityAttr(float(intensity))
    light.CreateColorTemperatureAttr(float(color_temperature))
    light.CreateEnableColorTemperatureAttr(True)
    if hdri and os.path.exists(hdri):
        light.CreateTextureFileAttr(Sdf.AssetPath(hdri))
    else:
        if hdri:
            print(f"[lighting] WARN dome HDRI not found: {hdri} (using untextured dome)")
    if rotation_deg is not None:
        _set_xform(stage, prim_path, rotate_xyz_deg=rotation_deg)


def _author_sun(stage, prim_path: str, intensity: float,
                color_temperature: float, angle_deg: float,
                elevation_deg: float, azimuth_deg: float) -> None:
    from pxr import UsdLux
    light = UsdLux.DistantLight.Define(stage, prim_path)
    light.CreateIntensityAttr(float(intensity))
    light.CreateColorTemperatureAttr(float(color_temperature))
    light.CreateEnableColorTemperatureAttr(True)
    light.CreateAngleAttr(float(angle_deg))
    dx, dy, dz = _sun_direction(elevation_deg, azimuth_deg)
    q = _quat_aim_neg_z(dx, dy, dz)
    _set_xform(stage, prim_path, orient_quat=q)


def _author_fill(stage, prim_path: str, intensity: float,
                 color_temperature: float, size, position) -> None:
    from pxr import UsdLux
    light = UsdLux.RectLight.Define(stage, prim_path)
    light.CreateIntensityAttr(float(intensity))
    light.CreateColorTemperatureAttr(float(color_temperature))
    light.CreateEnableColorTemperatureAttr(True)
    w, h = size
    light.CreateWidthAttr(float(w))
    light.CreateHeightAttr(float(h))
    px, py, pz = position
    # Aim at the world origin's tabletop area (~ z=0.8). Light travels
    # FROM position TO target = (0, 0, 0.8); normalize as -Z aim vector.
    tx, ty, tz = 0.0, 0.0, 0.8
    dx, dy, dz = tx - px, ty - py, tz - pz
    n = math.sqrt(dx * dx + dy * dy + dz * dz)
    if n > 1e-6:
        dx, dy, dz = dx / n, dy / n, dz / n
    q = _quat_aim_neg_z(dx, dy, dz)
    _set_xform(stage, prim_path, translate=position, orient_quat=q)


def remove_default_dome(stage) -> None:
    """Isaac's add_default_ground_plane embeds a DistantLight + SphereLight
    pair so the default scene isn't pitch black. They clash with our custom
    presets -- zero their intensity so only our lights drive the look. We
    don't delete the prims since the ground plane geometry is wired up
    through the same path."""
    from pxr import UsdLux
    zeroed = 0
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        path = prim.GetPath().pathString
        if "defaultGroundPlane" not in path:
            continue
        api = UsdLux.LightAPI(prim) if prim.HasAPI(UsdLux.LightAPI) else None
        if api is None:
            for cls in (UsdLux.DistantLight, UsdLux.SphereLight, UsdLux.DomeLight,
                        UsdLux.RectLight, UsdLux.DiskLight, UsdLux.CylinderLight):
                if prim.IsA(cls):
                    api = UsdLux.LightAPI(prim)
                    break
        if api is None:
            continue
        attr = api.GetIntensityAttr()
        if attr and attr.IsValid():
            attr.Set(0.0)
            zeroed += 1
    if zeroed:
        print(f"[lighting] zeroed {zeroed} default light(s) under defaultGroundPlane")


# ---- public API -----------------------------------------------------------


def apply_preset(stage, env_root: str, name: str,
                 override_cfg: Optional[dict] = None) -> None:
    """Author dome + sun (+ optional fill) under {env_root}/_lighting/.

    name           preset name; one of PRESETS or 'custom'
    override_cfg   when name is a preset: per-subdict (dome/sun/fill) value
                   merge over the preset. when name == 'custom': whole config.
    """
    if name == "custom":
        if not override_cfg:
            raise ValueError(
                "lighting preset='custom' requires lighting.dome/sun in cfg")
        cfg = override_cfg
    else:
        if name not in PRESETS:
            raise ValueError(
                f"unknown lighting preset '{name}'; have {list(PRESETS)}")
        cfg = {k: (dict(v) if isinstance(v, dict) else v)
               for k, v in PRESETS[name].items()}
        if override_cfg:
            for k in ("dome", "sun", "fill"):
                v = override_cfg.get(k)
                if isinstance(v, dict):
                    if cfg.get(k) is None:
                        cfg[k] = {}
                    cfg[k].update(v)
    apply_custom(stage, env_root, cfg)
    print(f"[lighting] preset '{name}' applied under {env_root}/_lighting")


def apply_custom(stage, env_root: str, cfg: dict) -> None:
    light_root = f"{env_root}/_lighting"
    _ensure_xform(stage, light_root)
    dome = cfg.get("dome")
    if dome:
        _author_dome(stage, f"{light_root}/dome",
                     hdri=dome.get("hdri", ""),
                     intensity=dome.get("intensity", 500.0),
                     color_temperature=dome.get("color_temperature", 6500.0),
                     rotation_deg=dome.get("rotation_deg"))
    sun = cfg.get("sun")
    if sun:
        _author_sun(stage, f"{light_root}/sun",
                    intensity=sun.get("intensity", 1500.0),
                    color_temperature=sun.get("color_temperature", 5500.0),
                    angle_deg=sun.get("angle_deg", 5.0),
                    elevation_deg=sun.get("elevation_deg", 45.0),
                    azimuth_deg=sun.get("azimuth_deg", 120.0))
    fill = cfg.get("fill")
    if fill:
        _author_fill(stage, f"{light_root}/fill",
                     intensity=fill.get("intensity", 500.0),
                     color_temperature=fill.get("color_temperature", 3200.0),
                     size=fill.get("size", [0.8, 1.2]),
                     position=fill.get("position", [-1.0, 0.0, 1.5]))


# ---- render mode ----------------------------------------------------------

_MODE_TO_RTX = {
    "RealTime": "RaytracedLighting",
    "PathTracing": "PathTracing",
}


def set_render_mode(mode: str, spp: int = 4) -> None:
    if mode not in _MODE_TO_RTX:
        raise ValueError(
            f"unknown render mode '{mode}'; expected RealTime or PathTracing")
    import carb
    s = carb.settings.get_settings()
    s.set_string("/rtx/rendermode", _MODE_TO_RTX[mode])
    if mode == "PathTracing":
        s.set_int("/rtx/pathtracing/spp", int(spp))
        s.set_int("/rtx/pathtracing/totalSpp", 64)
    extra = f" (spp={spp})" if mode == "PathTracing" else ""
    print(f"[lighting] render mode -> {mode}{extra}")


def toggle_render_mode(current: str, spp: int = 4) -> str:
    new_mode = "PathTracing" if current == "RealTime" else "RealTime"
    set_render_mode(new_mode, spp=spp)
    return new_mode
