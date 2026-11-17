import asyncio
import json

from genmedia_guard.app import create_app
from genmedia_guard.canary import CanaryConfig
from genmedia_guard.service import GuardService


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def call(app, method, path, body=None):
    """Drive the ASGI app directly, no server needed."""
    sent = []
    raw = json.dumps(body).encode() if body is not None else b""

    async def receive():
        return {"type": "http.request", "body": raw, "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(app({"type": "http", "method": method, "path": path}, receive, send))
    status = sent[0]["status"]
    payload = sent[1]["body"].decode()
    return status, payload


def make_app():
    clock = Clock()
    svc = GuardService(CanaryConfig(step_duration_s=30, min_requests=50), clock=clock, baseline_size=200)
    return create_app(svc), svc, clock


def test_healthz_and_unknown_route():
    app, _, _ = make_app()
    assert call(app, "GET", "/healthz") == (200, '{"status": "ok"}')
    assert call(app, "GET", "/nope")[0] == 404


def test_generate_validates_input_and_returns_metadata():
    app, _, _ = make_app()
    assert call(app, "POST", "/generate", {"prompt": 5})[0] == 400
    status, body = call(app, "POST", "/generate", {"prompt": "a cat", "seed": 3})
    data = json.loads(body)
    assert status == 200 and data["version"] == "v1-stable" and data["quality"]["ok"]


def test_invalid_json_is_a_400():
    app, _, _ = make_app()

    async def receive():
        return {"type": "http.request", "body": b"{oops", "more_body": False}

    sent = []

    async def send(m):
        sent.append(m)

    asyncio.run(app({"type": "http", "method": "POST", "path": "/generate"}, receive, send))
    assert sent[0]["status"] == 400


def test_unknown_scenario_is_rejected():
    app, _, _ = make_app()
    assert call(app, "POST", "/canary/start", {"scenario": "bogus"})[0] == 400


def test_metrics_endpoint_exposes_prometheus_text():
    app, _, _ = make_app()
    call(app, "POST", "/generate", {"prompt": "x", "seed": 1})
    status, text = call(app, "GET", "/metrics")
    assert status == 200
    assert "genmedia_requests_total" in text
    assert 'genmedia_request_duration_seconds_bucket{version="v1-stable",le="+Inf"}' in text
    assert "genmedia_canary_weight" in text


def test_bad_canary_gets_fallback_and_is_rolled_back():
    app, svc, clock = make_app()
    call(app, "POST", "/canary/start", {"scenario": "silent-collapse"})
    # Force every request to the canary so the collapse shows up quickly.
    svc.pick_role = lambda: "canary"
    fallbacks = 0
    for i in range(400):
        clock.t = i * 0.05
        data = svc.generate("p", i)
        assert data["status"] in (200, 502)  # users are still served (or a clean 502)
        fallbacks += data.get("fallback", False)
        if i % 20 == 0:
            svc.evaluate()
    assert fallbacks > 0
    assert svc.controller.state.value == "rolled_back"
    assert svc.m_rollbacks.value() == 1
    status, text = call(app, "GET", "/canary")
    assert json.loads(text)["state"] == "rolled_back"


def test_healthy_canary_is_promoted_and_becomes_stable():
    app, svc, clock = make_app()
    call(app, "POST", "/canary/start", {"scenario": "healthy"})
    for second in range(1, 200):
        for i in range(60):
            clock.t = second - 1 + i / 60
            svc.generate("p", second * 100 + i)
        clock.t = second
        if second % 5 == 0:
            svc.evaluate()
        if svc.controller.state.value == "promoted":
            break
    assert svc.controller.state.value == "promoted"
    assert svc.stable_model.version.name == "v2-healthy"
    assert svc.m_promotions.value() == 1
