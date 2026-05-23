#
# SimApp: Isaac Sim 5.1 runtime. Owns the SimulationApp + World and the single
# step loop that everything else is pumped from.
#
# CRITICAL ORDERING: SimulationApp() must be constructed before torch / curobo
# / any isaacsim.core import (the RTX + Kit plugins initialize in the ctor).
# So bio_sim/cli.py builds SimApp first, then load_ref()'s the registry-
# selected robot/scene/task classes (which are what transitively import the
# heavy deps). Same constraint applies to grasp_probe.py / nav_probe.py.
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


class _ResetKey:
    """Press R in the viewport to reset the env (cube + base back to the
    validated start) and re-arm the task so the user can replay it without
    relaunching. Windowed only -- carb keyboard needs the app window."""

    def __init__(self, ctx, runner):
        import carb.input
        import omni.appwindow

        self._ctx = ctx
        self._runner = runner
        self._kbd = carb.input
        app_window = omni.appwindow.get_default_app_window()
        self._keyboard = app_window.get_keyboard()
        self._input = carb.input.acquire_input_interface()
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_kbd
        )
        print("[reset] press  R  to reset the env (then it re-runs the task)")

    def _on_kbd(self, e):
        et = self._kbd.KeyboardEventType
        K = self._kbd.KeyboardInput
        if e.type == et.KEY_PRESS and e.input == K.R:
            self._reset()
        return True

    def _reset(self):
        ctx = self._ctx
        try:
            ctx.robot.gripper.release(ctx)  # detach payload + open fingers
        except Exception:  # noqa: BLE001
            pass
        try:
            ctx.robot.reset_arm()                # arm back to retract/init
            ctx.robot.reset_gripper()            # SNAP fingers open (PD lag would
                                                 # otherwise push plate on rerun)
            ctx.robot.base.reset_pose(
                *getattr(ctx.robot, "base_start", (0.0, 0.0, 0.0)))
            name = ctx.scene.objects[0].name
            # Resample plate xy first (no-op when randomization is disabled)
            # so reset_object snaps the cube to the new grasp_xyz.
            ctx.scene.randomize_plate()
            ctx.scene.reset_object(name)
            ctx.blackboard.pop("held", None)
        except Exception as exc:  # noqa: BLE001
            print(f"[reset] env reset failed: {exc}")
            return
        self._runner.restart()
        print("[reset] env reset done")

    def close(self):
        try:
            self._input.unsubscribe_to_keyboard_events(
                self._keyboard, self._sub)
        except Exception:  # noqa: BLE001
            pass


class _RenderKey:
    """Press P in the viewport to toggle RTX RealTime <-> PathTracing.
    Path tracing accumulates samples for a couple seconds after the switch
    -- intended use is: leave RealTime for normal demo runs, flip to PT
    right before recording a take, flip back after."""

    def __init__(self, spp: int = 4):
        import carb.input
        import omni.appwindow

        self._spp = spp
        self._kbd = carb.input
        app_window = omni.appwindow.get_default_app_window()
        self._keyboard = app_window.get_keyboard()
        self._input = carb.input.acquire_input_interface()
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_kbd)
        print("[render] press  P  to toggle RealTime <-> PathTracing")

    def _on_kbd(self, e):
        et = self._kbd.KeyboardEventType
        K = self._kbd.KeyboardInput
        if e.type == et.KEY_PRESS and e.input == K.P:
            self._toggle()
        return True

    def _toggle(self):
        import carb
        from bio_sim.scene import lighting

        s = carb.settings.get_settings()
        current_rtx = s.get_as_string("/rtx/rendermode")
        current_mode = ("PathTracing" if current_rtx == "PathTracing"
                        else "RealTime")
        lighting.toggle_render_mode(current_mode, spp=self._spp)

    def close(self):
        try:
            self._input.unsubscribe_to_keyboard_events(
                self._keyboard, self._sub)
        except Exception:  # noqa: BLE001
            pass


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

        # Windowed only: load the GUI extensions that give us a full File menu
        # (Save / Export / Collect As) and the Script Editor. The default
        # bio_sim startup is a minimal Kit profile that omits these.
        if self._headless is None:
            import omni.kit.app  # noqa: WPS433

            mgr = omni.kit.app.get_app().get_extension_manager()
            for ext in (
                "omni.kit.window.file",       # File > Save / Save As / Export
                "omni.kit.tool.collect",      # File > Collect As...
                "omni.kit.window.script_editor",
                "omni.physx.commands",        # right-click > Add > Physics > Colliders Preset
                "omni.physx.ui",
            ):
                try:
                    mgr.set_extension_enabled_immediate(ext, True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[sim] could not enable {ext}: {exc}")

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
        reset_key = None
        render_key = None
        if self._headless is not None:
            self.world.play()  # no Play button in headless
        else:
            reset_key = _ResetKey(ctx, runner)
            render_key = _RenderKey()
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
        printed_done = False
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
                if not printed_done:
                    status = "FAILED" if runner.failed else "COMPLETE"
                    print(f"[sim] task {status}")
                    printed_done = True
                if self._headless is not None:
                    break  # automated run: exit so the result is reported
                # Windowed: keep pumping the sim so the result stays visible
                # AND the R key can reset + re-run the task (runner.restart()
                # clears .done, so the loop picks the task back up).
            else:
                printed_done = False

        if reset_key is not None:
            reset_key.close()
        if render_key is not None:
            render_key.close()
        self.close()
