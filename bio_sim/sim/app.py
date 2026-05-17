#
# SimApp: Isaac Sim 5.1 runtime. Owns the SimulationApp + World and the single
# step loop that everything else is pumped from.
#
# CRITICAL ORDERING: SimulationApp() must be constructed before torch / curobo
# / any isaacsim.core import (the RTX + Kit plugins initialize in the ctor).
# So play.py builds SimApp first, then imports robot/scene/skill modules.
#
# Isaac Sim 5.1 native namespace is isaacsim.core.* (the old omni.isaac.core
# survives only as a deprecated shim). New code uses the native namespace.
#

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Settle window: physics needs a few steps to stabilize and the articulation
# view to come up before we touch DOFs or start planning.
ROBOT_INIT_UNTIL = 10
SETTLE_UNTIL = 20


class SimApp:
    def __init__(self, headless: str | None = None, width: int = 1920, height: int = 1080):
        from isaacsim import SimulationApp  # noqa: WPS433

        self._headless = headless
        self._app = SimulationApp(
            {"headless": headless is not None, "width": str(width), "height": str(height)}
        )

        # Safe to import isaacsim.core now.
        from isaacsim.core.api import World  # noqa: WPS433

        self.world = World(stage_units_in_meters=1.0)
        self.stage = self.world.stage
        xform = self.stage.DefinePrim("/World", "Xform")
        self.stage.SetDefaultPrim(xform)
        self.stage.DefinePrim("/curobo", "Xform")

    # ---- lifecycle ----------------------------------------------------
    @property
    def physics_dt(self) -> float:
        return self.world.get_physics_dt()

    @property
    def step_index(self) -> int:
        return self.world.current_time_step_index

    def is_running(self) -> bool:
        return self._app.is_running()

    def step(self, render: bool = True) -> None:
        self.world.step(render=render)

    def add_extensions(self) -> None:
        """curobo's isaac_sim example helper (asset import extensions).

        Always pass headless_mode=None: the helper would otherwise enable
        omni.kit.livestream.<mode>, which is absent in this pip install and
        shuts the app down. We don't stream the viewport for validation.
        """
        sys.path.append(
            os.path.join(_PROJECT_ROOT, "third_party", "curobo", "examples", "isaac_sim")
        )
        from helper import add_extensions  # noqa: WPS433

        add_extensions(self._app, None)

    def close(self) -> None:
        self._app.close()

    # ---- main loop ----------------------------------------------------
    def run(self, ctx, runner, on_world_sync=None) -> None:
        """Pump the sim. Per step, after physics settles:

        1. robot.ensure_initialized(ctx)  -- DOF map, drive modes (once)
        2. robot.base_hold(ctx)           -- keep kinematic base from drifting
        3. on_world_sync(step_index)      -- periodic cuRobo obstacle resync
        4. runner.tick(ctx)               -- advance the active skill
        """
        if self._headless is not None:
            self.world.play()  # no Play button in headless
        else:
            # Windowed: make sure Isaac does NOT auto-start the timeline.
            # Stop it explicitly so nothing (init / settle / task) runs until
            # the user actually clicks Play.
            try:
                import omni.timeline

                omni.timeline.get_timeline_interface().stop()
            except Exception as exc:  # noqa: BLE001
                print(f"[sim] could not stop timeline: {exc}")
            print("\n" + "=" * 60)
            print("  Ready. Press the PLAY button in the viewport to start.")
            print("=" * 60 + "\n")

        printed_play_hint = False
        while self.is_running():
            self.step(render=True)

            if not self.world.is_playing():
                if not printed_play_hint:
                    print("**** Waiting for Play... ****")
                    printed_play_hint = True
                continue

            si = self.step_index

            if si < ROBOT_INIT_UNTIL:
                ctx.robot.ensure_initialized(ctx)

            # Hold/teleport the kinematic base every step from the moment the
            # controller exists so the freed base never free-falls.
            if ctx.robot.base_ready:
                ctx.robot.base_hold(ctx)

            if si < SETTLE_UNTIL:
                continue

            if on_world_sync is not None:
                on_world_sync(si)

            runner.tick(ctx)

            if runner.done:
                status = "FAILED" if runner.failed else "COMPLETE"
                print(f"[sim] task {status}")
                if self._headless is not None:
                    break  # automated run: exit so the result is reported
                # windowed: keep rendering so the result stays visible
                print("[sim] idling; close the window to exit")
                while self.is_running():
                    self.step(render=True)
                break

        self.close()
