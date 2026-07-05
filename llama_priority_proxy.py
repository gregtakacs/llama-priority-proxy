#!/usr/bin/env python3
"""
llama_priority_proxy.py — the "JSON config orchestrator" entrypoint.sh has been
waiting for. Replaces Ollama's role entirely: sits in front of multiple
llama-server child processes it spawns/evicts itself, and routes OpenAI-
compatible requests to the right one based on scenarios (named, prioritized
groups of models meant to be resident together — see config/scenario_*.json).

PHASE 3 STATE (see /home/greg/.claude/plans/tidy-petting-nebula.md): config
loading, auth, /status, on-demand spawn, priority-gated scenario switching,
fallback_for aliasing, and shared-model dedup across a switch are implemented.
Capacity queuing (/slots pre-check, SSE heartbeat, max-wait) and keep_alive
idle-eviction are NOT yet implemented (phases 4/5).

Config read at startup from --config-dir (default config/):
  model_options.json       - parallel/min_ctx/max_ctx/extra_args/n_gpu_layers/
                              label/keep_alive per model (see its own _comment)
  model_vram_registry.json - base_vram_bytes/bytes_per_ctx_token/max_ctx/
                              parallel_tested per model (from `bench`)
  scenario_*.json           - one file per scenario: name/priority/default/
                              port(s)/models (each with ctx, slot, fallback_for,
                              optional label override)
  standalone_models.json    - always-on models outside the scenario system
                              (nomic-embed-text), loaded once at startup

Ports are fixed (see scenario/standalone config, not chosen dynamically here):
  11444 (this proxy, 0.0.0.0) / 11445 (standalone, 127.0.0.1) /
  11446+11447 (scenario primary/secondary slots, 127.0.0.1)
"""

import argparse
import asyncio
import json
import os
import re
import time

import aiohttp
from aiohttp import web

from llama_process import (
    format_bytes, gpu_total_bytes, gpu_used_bytes,
    launch_server, max_ctx_for_budget, predicted_vram, shutdown_server, wait_for_health,
)

DEFAULT_HEADROOM_GB = 1.0
DEFAULT_LOAD_TIMEOUT_S = 300
DEFAULT_MAX_WAIT_S = 60
DEFAULT_KEEP_ALIVE_S = 300  # Ollama's own default idle timeout, used for blank/unset keep_alive
IDLE_SWEEP_INTERVAL_S = 10

_KEEP_ALIVE_RE = re.compile(r"^(\d+(?:\.\d+)?)(s|m|h)$")
_KEEP_ALIVE_UNITS = {"s": 1, "m": 60, "h": 3600}


def parse_keep_alive(raw):
    """Ollama-style keep_alive string -> idle-timeout seconds, or None for
    "-1" (never evict on idle — still subject to priority preemption, see
    activate_scenario's docstring). Blank/unset uses Ollama's own default."""
    raw = (raw or "").strip()
    if raw == "":
        return DEFAULT_KEEP_ALIVE_S
    if raw == "-1":
        return None
    match = _KEEP_ALIVE_RE.match(raw)
    if match:
        value, unit = match.groups()
        return float(value) * _KEEP_ALIVE_UNITS[unit]
    print(f"[proxy] WARNING: could not parse keep_alive '{raw}' — using default ({DEFAULT_KEEP_ALIVE_S}s)")
    return DEFAULT_KEEP_ALIVE_S


# ---------------------------------------------------------------------------
# Auth — Bearer token matching OpenAI's own convention, checked at the proxy's
# external edge only (backend llama-server children bind 127.0.0.1, never
# exposed, so there's no second boundary to protect).
# ---------------------------------------------------------------------------

def read_secret_file(key_env_var):
    """Checks <VAR>_FILE first (Docker secret path), falls back to plain <VAR>.
    Ported from OllamaModelProxy.py's _read_secret_file — same convention
    entrypoint.sh already uses for llama_api_key."""
    file_path = os.environ.get(f"{key_env_var}_FILE", "")
    if file_path:
        try:
            with open(file_path) as f:
                return f.read().strip()
        except OSError as e:
            print(f"[proxy] WARNING: failed to read secret from {file_path}: {e}")
    return os.environ.get(key_env_var, "")


def openai_error(status, message, err_type="invalid_request_error", code=None):
    """Every proxy-originated error (auth failure, unknown model, etc.) uses this
    shape — so a request we handle ourselves looks exactly like one a real
    OpenAI-compatible backend would have produced, not a proxy-specific quirk."""
    return web.json_response(
        {"error": {"message": message, "type": err_type, "param": None, "code": code}},
        status=status,
    )


@web.middleware
async def auth_middleware(request, handler):
    # /health is deliberately exempt: Docker's HEALTHCHECK can't authenticate,
    # and liveness alone doesn't leak anything /status would (model/scenario
    # state stays behind the key like every other route).
    api_key = request.app["proxy_api_key"]
    if api_key and request.path != "/health":
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {api_key}":
            return openai_error(401, "Incorrect API key provided.", code="invalid_api_key")
    return await handler(request)


async def handle_health(request):
    return web.json_response({"status": "ok"})


async def handle_models(request):
    """GET /v1/models — Open WebUI (and any other OpenAI-compatible client)
    calls this to populate its model picker; without it there's nothing to
    select in the UI at all, regardless of whether routing itself works."""
    state = request.app["state"]
    now = int(time.time())
    return web.json_response({
        "object": "list",
        "data": [{"id": label, "object": "model", "created": now, "owned_by": "llama-priority-proxy"}
                 for label in state.label_index],
    })


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def load_config(config_dir):
    options = load_json(os.path.join(config_dir, "model_options.json"), {}).get("options", {})
    registry = {m["name"]: m for m in
                load_json(os.path.join(config_dir, "model_vram_registry.json"), {"models": []})["models"]}
    standalone = load_json(os.path.join(config_dir, "standalone_models.json"), {"models": []})["models"]

    scenarios = {}
    for fname in sorted(os.listdir(config_dir)):
        if fname.startswith("scenario_") and fname.endswith(".json"):
            scenario = load_json(os.path.join(config_dir, fname), None)
            if scenario and "name" in scenario:
                scenarios[scenario["name"]] = scenario

    default_scenarios = [s for s in scenarios.values() if s.get("default")]
    if len(default_scenarios) != 1:
        raise RuntimeError(f"expected exactly one scenario with \"default\": true, "
                            f"found {len(default_scenarios)}")

    return options, registry, standalone, scenarios


def model_label(name, options, override=None):
    """A scenario-model's own 'label' override wins; otherwise the model's
    global label from model_options.json; otherwise the model's own name."""
    if override:
        return override
    return options.get(name, {}).get("label") or name


def build_label_index(scenarios, standalone, options):
    """Maps every externally-visible label -> a descriptor telling the request
    handler how to route it. Raises on a duplicate label, since that's always
    a config mistake (two things silently fighting over one routing key)."""
    index = {}
    for scenario in scenarios.values():
        for m in scenario["models"]:
            label = model_label(m["name"], options, m.get("label"))
            if label in index:
                raise RuntimeError(f"label '{label}' is claimed by more than one model — "
                                    f"give one of them an explicit distinct 'label'")
            index[label] = {
                "scenario": scenario["name"],
                "name": m["name"],
                "slot": m.get("slot", "primary"),
                "ctx_spec": m["ctx"],
                "fallback_for": m.get("fallback_for", []),
            }
    for m in standalone:
        label = model_label(m["name"], options)
        if label in index:
            raise RuntimeError(f"label '{label}' is claimed by more than one model")
        index[label] = {"scenario": None, "name": m["name"], "port": m["port"], "ctx_spec": m["ctx"]}
    return index


def solve_scenario_sizes(scenario, registry, budget_bytes):
    """Group-solve every member's ctx size at once (fixed members first, then
    whichever one is "auto" gets whatever budget remains) — the same math as
    benchmark_vram.py's cmd_solve_scenario, just returning values instead of
    printing them. Computed once when a scenario activates, not per-model, so
    load order never matters (see the plan's "load-order thrash" discussion)."""
    fixed_total = 0
    sizes = {}
    auto_name = None
    for m in scenario["models"]:
        if m["ctx"] == "auto":
            auto_name = m["name"]
        else:
            ctx = int(m["ctx"])
            sizes[m["name"]] = ctx
            fixed_total += predicted_vram(registry[m["name"]], ctx)
    if auto_name:
        entry = registry[auto_name]
        parallel = entry["parallel_tested"]
        remaining = budget_bytes - fixed_total
        sizes[auto_name] = max_ctx_for_budget(entry, remaining, ctx_cap=entry["max_ctx"] * parallel)
    return sizes


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------

class LoadedModel:
    def __init__(self, handle, name, port, ctx_size, keep_alive, parallel):
        self.handle = handle
        self.name = name
        self.port = port
        self.ctx_size = ctx_size
        self.keep_alive = keep_alive
        self.parallel = parallel
        self.last_used = time.monotonic()


class ProxyState:
    def __init__(self):
        self.options = {}
        self.registry = {}
        self.scenarios = {}
        self.standalone = []
        self.label_index = {}
        self.active_scenario = None
        self.loaded = {}          # model name -> LoadedModel (scenario-owned)
        self.standalone_loaded = {}  # model name -> LoadedModel
        self.reserved_ctx = {}    # model name -> ctx size, for the active scenario
        self.http_session = None
        self.spawn_lock = asyncio.Lock()  # serialize scenario switches/spawns


# ---------------------------------------------------------------------------
# Model lifecycle — launch_server/wait_for_health/shutdown_server are blocking
# (subprocess + time.sleep polling); run them in the default executor so a
# 10-60s model load doesn't stall the whole event loop.
# ---------------------------------------------------------------------------

async def spawn_model(state, name, port, ctx, load_timeout_s):
    opt = state.options.get(name, {})
    path = os.path.join(state.models_dir, f"{name}.gguf")
    parallel = opt.get("parallel", 1)
    loop = asyncio.get_event_loop()

    handle = await loop.run_in_executor(
        None, launch_server, path, ctx, parallel, port,
        opt.get("n_gpu_layers", 99), opt.get("extra_args", []), {"kind": "native"},
    )
    ok, reason = await loop.run_in_executor(None, wait_for_health, handle, load_timeout_s)
    if not ok:
        await loop.run_in_executor(None, shutdown_server, handle)
        raise RuntimeError(f"'{name}' failed to become healthy: {reason}")

    return LoadedModel(handle, name, port, ctx, opt.get("keep_alive", ""), parallel)


async def evict_model(lm):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, shutdown_server, lm.handle)


async def activate_scenario(state, scenario_name, load_timeout_s):
    """Evict whatever the new scenario doesn't want, group-solve ctx sizes for
    all its members at once, and switch. A model shared by both the outgoing
    and incoming scenario (same name — see build_label_index's dedup-by-name)
    is kept running across the switch rather than torn down and relaunched,
    but only if its already-loaded ctx-size matches what the new scenario
    would reserve for it: llama-server's ctx is fixed at launch, so a size
    mismatch means a relaunch is unavoidable, not just an optimization we're
    skipping."""
    scenario = state.scenarios[scenario_name]
    gpu_total = gpu_total_bytes(state.gpu_index)
    # Standalone models (nomic) are always resident and never evicted — their
    # VRAM is permanently spoken for, so the scenario's own group-sizing must
    # treat it the same as headroom, not as capacity it could ever claim.
    standalone_vram = sum(predicted_vram(state.registry[lm.name], lm.ctx_size)
                           for lm in state.standalone_loaded.values())
    budget = gpu_total - int(state.headroom_gb * (1024 ** 3)) - standalone_vram
    new_sizes = solve_scenario_sizes(scenario, state.registry, budget)

    new_names = {m["name"] for m in scenario["models"]}
    for name, lm in list(state.loaded.items()):
        if name in new_names and lm.ctx_size == new_sizes.get(name):
            continue  # shared with the new scenario at the same size — keep it running
        await evict_model(lm)
        del state.loaded[name]

    state.reserved_ctx = new_sizes
    state.active_scenario = scenario_name
    await eager_load_pinned_members(state, scenario_name, load_timeout_s)


async def eager_load_pinned_members(state, scenario_name, load_timeout_s):
    """keep_alive: -1 on a scenario member means eager-load it the moment its
    scenario becomes active, not lazily on first request — see the "default
    scenario + keep_alive: -1" design discussion. Applies on any activation,
    not just reversion to the default scenario."""
    scenario = state.scenarios[scenario_name]
    for m in scenario["models"]:
        if m["name"] in state.loaded:
            continue
        keep_alive = state.options.get(m["name"], {}).get("keep_alive", "")
        if parse_keep_alive(keep_alive) is not None:
            continue  # not pinned — stays lazy
        port = scenario["port"] if m.get("slot", "primary") == "primary" else scenario["port_secondary"]
        ctx = state.reserved_ctx[m["name"]]
        try:
            lm = await spawn_model(state, m["name"], port, ctx, load_timeout_s)
            state.loaded[m["name"]] = lm
            print(f"[proxy] eager-loaded pinned (keep_alive=-1) model '{m['name']}' for scenario '{scenario_name}'")
        except RuntimeError as e:
            print(f"[proxy] WARNING: failed to eager-load pinned model '{m['name']}': {e}")


def find_fallback(state, scenario_name):
    """A resident model in the currently ACTIVE scenario tagged to cover
    `scenario_name`'s role — used to serve a same-or-lower-priority request
    from what's already loaded instead of thrashing. None if nothing resident
    qualifies (caller falls back to a real switch in that case)."""
    if state.active_scenario is None:
        return None
    active_def = state.scenarios[state.active_scenario]
    candidates = [m for m in active_def["models"]
                  if scenario_name in m.get("fallback_for", []) and m["name"] in state.loaded]
    if not candidates:
        return None
    candidates.sort(key=lambda m: 0 if m.get("slot") == "primary" else 1)
    return state.loaded[candidates[0]["name"]]


async def maybe_evict_idle_scenario(state):
    """If the active scenario is non-default and EVERY one of its loaded
    members has gone idle past its own keep_alive, evict it and reactivate
    the default scenario. A single keep_alive=-1 member is enough to keep the
    whole scenario resident (it's pinned, so the scenario can't be "all idle"
    while it's still running)."""
    if state.active_scenario is None:
        return
    if state.scenarios[state.active_scenario].get("default"):
        return  # already the baseline — nothing to revert to
    if not state.loaded:
        return  # nothing actually loaded for it yet

    now = time.monotonic()
    for lm in state.loaded.values():
        timeout = parse_keep_alive(lm.keep_alive)
        if timeout is None or (now - lm.last_used) < timeout:
            return  # pinned, or still within its own idle window

    default_name = next(name for name, s in state.scenarios.items() if s.get("default"))
    print(f"[proxy] '{state.active_scenario}' idle past keep_alive — reverting to default scenario '{default_name}'")
    await activate_scenario(state, default_name, state.load_timeout_s)


async def idle_eviction_sweep(state):
    """Background task: periodically checks whether the active scenario has
    gone idle long enough to revert to the default one. Runs under the same
    lock as request handling so it never races a switch/spawn in progress."""
    while True:
        await asyncio.sleep(IDLE_SWEEP_INTERVAL_S)
        async with state.spawn_lock:
            await maybe_evict_idle_scenario(state)


# ---------------------------------------------------------------------------
# Capacity queuing — llama-server's own GET /slots?fail_on_no_slot=1 is a
# purpose-built pre-check (200 = a slot is free, 503 = full) rather than
# something we need to infer from is_processing counts ourselves. --no-
# context-shift means a genuinely full --kv-unified pool gets REJECTED, not
# silently evicted, so we poll this instead of just firing the real request
# and hoping. Bounded by one global max-wait; a streaming request gets SSE
# keep-alive comments during the wait so the client's own stream-inactivity
# timeout doesn't fire before we ever get to the real response.
# ---------------------------------------------------------------------------

async def wait_for_capacity(request, state, port, heartbeat_response, max_wait_s):
    """Returns True (room found), False (max-wait elapsed), or 'disconnected'
    (client gave up first)."""
    deadline = time.monotonic() + max_wait_s
    url = f"http://127.0.0.1:{port}/slots?fail_on_no_slot=1"
    while True:
        # aiohttp has no Starlette-style request.is_disconnected() — the
        # transport itself reports whether the underlying connection is
        # closing, which is the equivalent signal here.
        if request.transport is not None and request.transport.is_closing():
            return "disconnected"
        try:
            async with state.http_session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    return True
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return True  # /slots unreachable/unsupported (e.g. --no-slots) — don't block on an unknown
        if time.monotonic() >= deadline:
            return False
        if heartbeat_response is not None:
            await heartbeat_response.write(b": ping\n\n")
        await asyncio.sleep(2)


# ---------------------------------------------------------------------------
# Request forwarding — parses SSE line-by-line (aiohttp's StreamReader already
# splits on \n reliably across TCP chunk boundaries, so no manual blank-line
# buffering is needed) and rewrites the `model` field back to whatever label
# the client actually asked for, so aliasing (phase 3) is invisible to it.
# ---------------------------------------------------------------------------

async def forward(request, state, body, port, client_label, real_name, max_wait_s):
    body["model"] = real_name
    is_stream = bool(body.get("stream"))
    url = f"http://127.0.0.1:{port}{request.path}"

    # A streaming response's status line commits to 200 the moment we
    # prepare() it, which has to happen before contacting the backend if we
    # want to heartbeat during a capacity wait — same convention OpenAI's own
    # streaming API uses: errors after that point ride inside the stream as a
    # data event, never as a change to the already-sent HTTP status.
    response = None
    if is_stream:
        response = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)

    capacity = await wait_for_capacity(request, state, port, response, max_wait_s)
    if capacity is not True:
        if capacity == "disconnected":
            if response is not None:
                await response.write_eof()
            return response or web.Response(status=499)
        reason = f"timed out after {max_wait_s}s waiting for a free slot"
        err_body = {"error": {"message": f"No capacity available: {reason}.",
                               "type": "server_error", "code": "no_capacity"}}
        if response is not None:
            await response.write(f"data: {json.dumps(err_body)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response
        return web.json_response(err_body, status=503)

    async with state.http_session.post(url, json=body) as resp:
        if not is_stream:
            data = await resp.json()
            if isinstance(data, dict) and data.get("model") == real_name:
                data["model"] = client_label
            return web.json_response(data, status=resp.status)

        async for line_bytes in resp.content:
            line = line_bytes.decode("utf-8", errors="replace")
            if line.startswith("data: ") and real_name != client_label:
                payload = line[len("data: "):].strip()
                if payload and payload != "[DONE]":
                    try:
                        obj = json.loads(payload)
                        if obj.get("model") == real_name:
                            obj["model"] = client_label
                        line = f"data: {json.dumps(obj)}\n"
                    except json.JSONDecodeError:
                        pass
            await response.write(line.encode())
        await response.write_eof()
        return response


# ---------------------------------------------------------------------------
# HTTP handlers
# ---------------------------------------------------------------------------

async def handle_completion(request):
    state = request.app["state"]
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return openai_error(400, "Invalid JSON body.")

    requested_label = body.get("model")
    target = state.label_index.get(requested_label)
    if target is None:
        return openai_error(404, f"Unknown model '{requested_label}'.", code="model_not_found")

    if target["scenario"] is None:
        lm = state.standalone_loaded.get(target["name"])
        if lm is None:
            return openai_error(503, f"Standalone model '{target['name']}' is not loaded.")
        lm.last_used = time.monotonic()
        return await forward(request, state, body, lm.port, requested_label, target["name"], state.max_wait_s)

    scenario_name = target["scenario"]
    async with state.spawn_lock:
        if state.active_scenario != scenario_name:
            active_priority = (state.scenarios[state.active_scenario]["priority"]
                                if state.active_scenario else -1)
            outranks_active = state.scenarios[scenario_name]["priority"] > active_priority
            if not outranks_active:
                alias = find_fallback(state, scenario_name)
                if alias is not None:
                    alias.last_used = time.monotonic()
                    return await forward(request, state, body, alias.port, requested_label, alias.name, state.max_wait_s)
                # Nothing resident can cover it — no cheaper option than switching.
            await activate_scenario(state, scenario_name, state.load_timeout_s)

        lm = state.loaded.get(target["name"])
        if lm is None:
            scenario = state.scenarios[scenario_name]
            port = scenario["port"] if target["slot"] == "primary" else scenario["port_secondary"]
            ctx = state.reserved_ctx[target["name"]]
            try:
                lm = await spawn_model(state, target["name"], port, ctx, state.load_timeout_s)
            except RuntimeError as e:
                return openai_error(503, str(e))
            state.loaded[target["name"]] = lm

    lm.last_used = time.monotonic()
    return await forward(request, state, body, lm.port, requested_label, target["name"], state.max_wait_s)


async def handle_status(request):
    state = request.app["state"]
    now = time.monotonic()
    gpu_total = gpu_total_bytes(state.gpu_index)
    gpu_used = gpu_used_bytes(state.gpu_index)
    return web.json_response({
        "active_scenario": state.active_scenario,
        "loaded_models": [
            {"name": lm.name, "port": lm.port, "ctx_size": lm.ctx_size,
             "idle_for_s": round(now - lm.last_used, 1)}
            for lm in state.loaded.values()
        ],
        "standalone_models": [
            {"name": lm.name, "port": lm.port, "ctx_size": lm.ctx_size}
            for lm in state.standalone_loaded.values()
        ],
        "gpu": {
            "total": format_bytes(gpu_total),
            "used": format_bytes(gpu_used),
            "free": format_bytes(gpu_total - gpu_used) if gpu_used is not None else None,
        },
        "scenarios": [
            {"name": s["name"], "priority": s["priority"], "default": s.get("default", False),
             "active": s["name"] == state.active_scenario}
            for s in state.scenarios.values()
        ],
    })


# ---------------------------------------------------------------------------
# App wiring
# ---------------------------------------------------------------------------

async def on_startup(app):
    state = app["state"]
    state.http_session = aiohttp.ClientSession()
    for m in state.standalone:
        lm = await spawn_model(state, m["name"], m["port"], m["ctx"], state.load_timeout_s)
        state.standalone_loaded[m["name"]] = lm
        print(f"[proxy] standalone '{m['name']}' loaded on port {m['port']} (ctx={m['ctx']:,})")

    # The default scenario is the baseline everything reverts to — activate it
    # immediately at boot (eager-loading any keep_alive=-1 member right away)
    # rather than waiting for the first request to discover it.
    default_name = next(name for name, s in state.scenarios.items() if s.get("default"))
    await activate_scenario(state, default_name, state.load_timeout_s)

    state.idle_sweep_task = asyncio.create_task(idle_eviction_sweep(state))


async def on_cleanup(app):
    state = app["state"]
    state.idle_sweep_task.cancel()
    for lm in list(state.loaded.values()) + list(state.standalone_loaded.values()):
        await evict_model(lm)
    await state.http_session.close()


def build_app(state, proxy_api_key):
    app = web.Application(middlewares=[auth_middleware])
    app["state"] = state
    app["proxy_api_key"] = proxy_api_key
    app.router.add_post("/v1/chat/completions", handle_completion)
    app.router.add_post("/v1/completions", handle_completion)
    app.router.add_post("/v1/embeddings", handle_completion)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/status", handle_status)
    app.router.add_get("/health", handle_health)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", default=os.environ.get("MODELS_DIR", "/models"))
    parser.add_argument("--config-dir", default=os.environ.get("CONFIG_DIR", "config"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PROXY_PORT", "11444")))
    parser.add_argument("--gpu-index", type=int, default=int(os.environ.get("GPU_INDEX", "0")))
    parser.add_argument("--headroom-gb", type=float,
                         default=float(os.environ.get("PROXY_HEADROOM_GB", str(DEFAULT_HEADROOM_GB))))
    parser.add_argument("--load-timeout", type=int,
                         default=int(os.environ.get("PROXY_LOAD_TIMEOUT_S", str(DEFAULT_LOAD_TIMEOUT_S))))
    parser.add_argument("--max-wait", type=int,
                         default=int(os.environ.get("PROXY_MAX_WAIT_S", str(DEFAULT_MAX_WAIT_S))),
                         help="Max seconds to wait for a free slot before returning a real error "
                              "(one global setting for all connections, per the design discussion).")
    args = parser.parse_args()

    state = ProxyState()
    state.models_dir = args.models_dir
    state.gpu_index = args.gpu_index
    state.headroom_gb = args.headroom_gb
    state.load_timeout_s = args.load_timeout
    state.max_wait_s = args.max_wait
    state.options, state.registry, state.standalone, state.scenarios = load_config(args.config_dir)
    state.label_index = build_label_index(state.scenarios, state.standalone, state.options)

    proxy_api_key = read_secret_file("PROXY_API_KEY")
    app = build_app(state, proxy_api_key)
    web.run_app(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
