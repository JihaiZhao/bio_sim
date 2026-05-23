#
# WebRTC livestream smoke test.
#
# Boots a headless SimulationApp with omni.kit.livestream.webrtc enabled,
# builds a minimal stage (ground + a distinctly coloured cube so the
# operator can confirm pixels are actually flowing), and pumps the sim
# forever. The point is to PROVE the streaming server comes up cleanly
# in this pip install BEFORE bolting any bio_sim / curobo / FastAPI on
# top of it.
#
# Usage:
#   uv run python scripts/livestream_smoke.py
#
# Then connect a WebRTC client to ws://<host>:49100 (the default signaling
# port baked into the extension config). For a quick eyeball test use
# NVIDIA's Omniverse Streaming Client (download from the Launcher). A
# browser-side webrtc-client.js will need to be sourced separately --
# Isaac Sim's pip install does NOT bundle one.
#
# Memory: this script is the resolution of bio-sim-isaac-headless-debug
# (the older livestream-ext crash). If this script exits cleanly with
# "[smoke] streaming server live" in the log, the extension stack is
# healthy and frontend integration can proceed.
#

from __future__ import annotations

# CRITICAL: SimulationApp() must be the first Isaac-touching import. Do
# NOT add `import torch` / `import isaacsim.core.*` above this line.
from isaacsim import SimulationApp

# Headless on purpose -- this is the deployment mode the frontend will
# consume. extra_args is the Isaac 5.1 lever for enabling kit extensions
# at boot (the old `livestream=2` SimulationApp kwarg was removed in 5.x).
CFG = {
    "headless": True,
    "width": 1920,
    "height": 1080,
    # headless+hide_ui=None auto-hides the UI; for streaming we want it
    # visible so the browser actually sees menus / viewport chrome.
    "hide_ui": False,
    "extra_args": [
        "--enable", "omni.kit.livestream.webrtc",
    ],
}

app = SimulationApp(CFG)

# Safe to import isaacsim.core now.
import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import DynamicCuboid  # noqa: E402

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# Bright magenta cube at (0, 0, 0.5) -- something obviously not-default so
# we can tell the stream from a stock empty viewport at a glance.
world.scene.add(
    DynamicCuboid(
        prim_path="/World/smoke_cube",
        name="smoke_cube",
        position=np.array([0.0, 0.0, 0.5]),
        scale=np.array([0.3, 0.3, 0.3]),
        color=np.array([1.0, 0.0, 1.0]),
    )
)

world.reset()
world.play()

print("=" * 60)
print("[smoke] streaming server live")
print("[smoke]   signaling: ws://<host>:49100")
print("[smoke]   client:    NVIDIA Omniverse Streaming Client (native)")
print("[smoke]              -- pip install does NOT bundle a browser client")
print("[smoke] Ctrl+C to stop.")
print("=" * 60)

try:
    while app.is_running():
        world.step(render=True)
except KeyboardInterrupt:
    print("[smoke] caught Ctrl+C, shutting down")

app.close()
