#
# FastAPI control plane for `bio_sim serve`.
#
# Threading model -- the critical bit:
#
#   - uvicorn runs in a daemon thread (start_in_thread). FastAPI handlers
#     execute on uvicorn's worker pool, NOT on the sim main thread.
#   - Touching USD / PhysX / Articulation state from a worker thread races
#     world.step() and corrupts state. So handlers DO NOT do that. They
#     enqueue a Command and block on its Future.
#   - drain_queue(ctx, runner) is called from the sim main thread at the
#     top of every step. It pops commands, dispatches to handlers in
#     bio_sim.commands (sim-thread safe), and fulfils the Future, which
#     unblocks the HTTP caller with the result.
#
# Lifecycle: server starts listening as soon as start_in_thread returns;
# any commands that arrive before the sim is up just queue and are
# processed once the sim loop reaches its first drain_queue call.
#

from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from typing import Any, Callable

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import commands


class Command:
    __slots__ = ("op", "args", "future")

    def __init__(self, op: str, args: dict | None = None) -> None:
        self.op = op
        self.args = args or {}
        self.future: Future = Future()


# Single shared queue. The HTTP layer pushes; the sim loop drains. Unbounded
# is fine -- we'd OOM long before queueing 1e9 reset requests.
_queue: "queue.Queue[Command]" = queue.Queue()

# op -> sim-thread handler. Handlers MUST accept (ctx, runner, **args)
# and return a JSON-serialisable dict. See bio_sim/commands.py.
_DISPATCH: dict[str, Callable[..., dict]] = {
    "status": commands.status,
    "reset":  commands.reset_env,
}


def make_app() -> FastAPI:
    """Build a fresh FastAPI app. Routes are thin: enqueue + wait."""
    app = FastAPI(
        title="bio_sim control plane",
        description=(
            "HTTP/JSON commands for a running bio_sim instance. Pair with "
            "the WebRTC livestream on :49100 for video."
        ),
    )

    # Dev default: open CORS. Tighten when deploying.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/status")
    def get_status() -> dict:
        return _enqueue_and_wait("status", timeout=5.0)

    @app.post("/reset")
    def post_reset() -> dict:
        return _enqueue_and_wait("reset", timeout=10.0)

    @app.get("/tasks")
    def list_tasks() -> dict:
        # Metadata only -- no sim involvement, answer immediately.
        from bio_sim.specs import TASKS

        return {
            "tasks": [
                {"name": n, "description": s.description}
                for n, s in TASKS.items()
            ]
        }

    @app.get("/healthz")
    def healthz() -> dict:
        # Returns immediately. Does NOT prove the sim is healthy -- only
        # that uvicorn is up. Use /status for sim liveness.
        return {"ok": True, "queue_depth": _queue.qsize()}

    return app


def _enqueue_and_wait(op: str, args: dict | None = None,
                      timeout: float = 10.0) -> dict:
    cmd = Command(op, args)
    _queue.put(cmd)
    try:
        return cmd.future.result(timeout=timeout)
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"sim did not process {op!r} within {timeout}s "
                   f"(queue depth={_queue.qsize()}); is it paused or stalled?",
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


def start_in_thread(port: int = 8000, host: str = "0.0.0.0") -> threading.Thread:
    """Spawn uvicorn in a daemon thread. The thread exits with the
    process; no explicit shutdown call needed for a Ctrl+C teardown."""
    app = make_app()
    config = uvicorn.Config(
        app, host=host, port=port,
        log_level="info", access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(
        target=server.run, name="bio_sim.server", daemon=True,
    )
    thread.start()
    print(f"[server] uvicorn listening on http://{host}:{port}")
    print(f"[server]   GET  /status   POST /reset   GET /tasks   GET /healthz")
    print(f"[server]   docs at        http://{host}:{port}/docs")
    return thread


def drain_queue(ctx, runner) -> None:
    """Sim-thread pump. Drains every pending command in this tick and
    fulfils each Future. Idempotent on an empty queue (returns at once)."""
    while True:
        try:
            cmd = _queue.get_nowait()
        except queue.Empty:
            return
        handler = _DISPATCH.get(cmd.op)
        if handler is None:
            cmd.future.set_exception(KeyError(f"unknown op: {cmd.op}"))
            continue
        try:
            result = handler(ctx, runner, **cmd.args)
            cmd.future.set_result(result)
        except Exception as exc:  # noqa: BLE001
            cmd.future.set_exception(exc)
