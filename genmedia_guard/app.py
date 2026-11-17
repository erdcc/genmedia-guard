"""Plain-ASGI HTTP front end for ``GuardService`` (no web framework needed).

Run:  uvicorn genmedia_guard.app:create_app --factory --port 8000

Endpoints
    POST /generate        {"prompt": str, "seed": int}   -> model output metadata
    POST /canary/start    {"scenario": "quality-drift"}  -> begin a canary rollout
    GET  /canary          current rollout status
    GET  /metrics         Prometheus exposition
    GET  /healthz         liveness / readiness

Environment
    GUARD_SLEEP=1               really sleep for the simulated model latency
    GUARD_EVAL_INTERVAL_S=5     how often the controller evaluates the canary
    GUARD_STEP_DURATION_S=60    observation window per traffic step
    GUARD_MIN_REQUESTS=100      canary requests required before gates apply
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import replace
from typing import Any, Awaitable, Callable, Dict, Optional

from .canary import Action, CanaryConfig
from .fake_model import SCENARIOS
from .service import GuardService

log = logging.getLogger("genmedia_guard")

Scope = Dict[str, Any]
Receive = Callable[[], Awaitable[Dict[str, Any]]]
Send = Callable[[Dict[str, Any]], Awaitable[None]]


def config_from_env() -> CanaryConfig:
    cfg = CanaryConfig()
    return replace(
        cfg,
        step_duration_s=float(os.getenv("GUARD_STEP_DURATION_S", cfg.step_duration_s)),
        min_requests=int(os.getenv("GUARD_MIN_REQUESTS", cfg.min_requests)),
    )


async def _read_json(receive: Receive) -> Any:
    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body"):
            break
    return json.loads(body) if body else {}


async def _respond(send: Send, status: int, payload: Any, content_type: str = "application/json") -> None:
    body = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", content_type.encode()),
                            (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


def create_app(
    service: Optional[GuardService] = None,
    eval_interval_s: Optional[float] = None,
    simulate_latency: Optional[bool] = None,
):
    svc = service or GuardService(config=config_from_env())
    interval = eval_interval_s if eval_interval_s is not None else float(
        os.getenv("GUARD_EVAL_INTERVAL_S", "5"))
    sleep = simulate_latency if simulate_latency is not None else os.getenv("GUARD_SLEEP") == "1"

    async def eval_loop() -> None:
        while True:
            await asyncio.sleep(interval)
            verdict = svc.evaluate()
            if verdict.action != Action.HOLD:
                log.warning("canary %s: %s", verdict.action.value, "; ".join(verdict.reasons))

    async def lifespan(receive: Receive, send: Send) -> None:
        task: Optional[asyncio.Task] = None
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                task = asyncio.create_task(eval_loop())
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                if task:
                    task.cancel()
                await send({"type": "lifespan.shutdown.complete"})
                return

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await lifespan(receive, send)
            return
        if scope["type"] != "http":
            return

        method, path = scope["method"], scope["path"]
        try:
            if method == "GET" and path == "/healthz":
                await _respond(send, 200, {"status": "ok"})
            elif method == "GET" and path == "/metrics":
                await _respond(send, 200, svc.registry.render(),
                               "text/plain; version=0.0.4; charset=utf-8")
            elif method == "GET" and path == "/canary":
                await _respond(send, 200, svc.controller.status())
            elif method == "POST" and path == "/canary/start":
                data = await _read_json(receive)
                scenario = data.get("scenario", "")
                if scenario not in SCENARIOS:
                    await _respond(send, 400, {"error": f"scenario must be one of {sorted(SCENARIOS)}"})
                else:
                    await _respond(send, 200, svc.start_canary(scenario))
            elif method == "POST" and path == "/generate":
                data = await _read_json(receive)
                prompt, seed = data.get("prompt"), data.get("seed", 0)
                if not isinstance(prompt, str) or not isinstance(seed, int):
                    await _respond(send, 400, {"error": "need string 'prompt' and integer 'seed'"})
                    return
                result = svc.generate(prompt, seed)
                if sleep and "latency_ms" in result:
                    await asyncio.sleep(float(result["latency_ms"]) / 1000.0)
                await _respond(send, int(result["status"]), result)
            else:
                await _respond(send, 404, {"error": "not found"})
        except json.JSONDecodeError:
            await _respond(send, 400, {"error": "invalid JSON body"})

    return app
