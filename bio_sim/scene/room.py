#
# Room shell (walls + optional ceiling) for the demo backdrop.
#
# Authors each wall as a FixedCuboid under {env_root}/_room/. FixedCuboid
# is the same primitive bio_scene._spawn_props uses for tables -- static
# rigid body with a colored visual material, no extra material wiring
# needed. Walls are kept OFF cuRobo's collision world by demo_scene's
# maybe_sync ignore list so they don't slow down planning.
#
# Opt-in via the yaml `room:` block (no block -> no walls authored, old
# tasks unchanged).
#
# Frame: world frame, Z-up. `pos` is the wall centre, `size` is the
# axis-aligned extent (sx, sy, sz) -- so a back wall facing the camera
# is e.g. size=[wide, thin, tall]=[8.0, 0.1, 3.0].
#

from __future__ import annotations

from typing import Optional

import numpy as np


def apply_walls(sim, env_root: str, cfg: Optional[dict]) -> int:
    """Author walls + optional ceiling. Returns count of boxes added."""
    if not cfg:
        return 0
    walls = cfg.get("walls") or []
    ceiling = cfg.get("ceiling")
    if not walls and not ceiling:
        return 0

    from isaacsim.core.api.objects import FixedCuboid

    n = 0
    for w in walls:
        name = w["name"]
        pos = np.array(w["pos"], dtype=np.float32)
        size = np.array(w["size"], dtype=np.float32)
        color = np.array(w.get("color", [0.92, 0.88, 0.82]),
                         dtype=np.float32)
        sim.world.scene.add(FixedCuboid(
            prim_path=f"{env_root}/_room/wall_{name}",
            name=f"_room_wall_{name}",
            position=pos,
            scale=size,
            color=color,
        ))
        n += 1

    if ceiling:
        pos = np.array(ceiling["pos"], dtype=np.float32)
        size = np.array(ceiling["size"], dtype=np.float32)
        color = np.array(ceiling.get("color", [0.95, 0.95, 0.95]),
                         dtype=np.float32)
        sim.world.scene.add(FixedCuboid(
            prim_path=f"{env_root}/_room/ceiling",
            name="_room_ceiling",
            position=pos,
            scale=size,
            color=color,
        ))
        n += 1

    print(f"[room] authored {n} box(es) under {env_root}/_room")
    return n
