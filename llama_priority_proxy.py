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
    # /health and /dashboard (the static HTML/JS shell only — no data) are
    # exempt: Docker's HEALTHCHECK can't authenticate, and the dashboard page
    # itself has nothing to hide — it can't see anything until its own JS
    # supplies this same Bearer key to /status and /slots/*, which ARE
    # protected below like every other route.
    # Exact-match only, NEVER startswith: "/" as a startswith-prefix matches
    # every path (everything starts with "/"), which previously exempted the
    # entire API — see the incident this comment is here to prevent recurring.
    api_key = request.app["proxy_api_key"]
    exempt_paths = {"/health", "/dashboard", "/dashboard/", "/"}
    if api_key and request.path not in exempt_paths:
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

_SAMPLING_FLAGS = {
    "temp": "--temp",
    "top_k": "--top-k",
    "top_p": "--top-p",
    "min_p": "--min-p",
    "repeat_penalty": "--repeat-penalty",
    "repeat_last_n": "--repeat-last-n",
    "presence_penalty": "--presence-penalty",
    "frequency_penalty": "--frequency-penalty",
}


def sampling_args(opt):
    """Translate model_options.json's named sampling fields into llama-server
    CLI flags. These only seed the server-side default — a client that sends
    its own temperature/top_p/etc. per-request still overrides them."""
    args = []
    for key, flag in _SAMPLING_FLAGS.items():
        if key in opt:
            args += [flag, str(opt[key])]
    return args


async def spawn_model(state, name, port, ctx, load_timeout_s):
    opt = state.options.get(name, {})
    path = os.path.join(state.models_dir, f"{name}.gguf")
    parallel = opt.get("parallel", 1)
    loop = asyncio.get_event_loop()

    handle = await loop.run_in_executor(
        None, launch_server, path, ctx, parallel, port,
        opt.get("n_gpu_layers", 99), opt.get("extra_args", []) + sampling_args(opt), {"kind": "native"},
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
        print(f"[proxy] activate_scenario('{scenario_name}'): evicting '{name}' (not in new scenario, or ctx-size mismatch)")
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


def model_supports_vision(state, name):
    return "--mmproj" in state.options.get(name, {}).get("extra_args", [])


def find_fallback(state, scenario_name, require_vision=False):
    """A resident model in the currently ACTIVE scenario tagged to cover
    `scenario_name`'s role — used to serve a same-or-lower-priority request
    from what's already loaded instead of thrashing. None if nothing resident
    qualifies (caller falls back to a real switch in that case).

    require_vision excludes a candidate that lacks --mmproj: fallback aliasing
    is invisible to the client (see forward()'s docstring), so silently
    routing an image request to a text-only alias would just crash inside
    llama-server instead of failing loudly — better to force a real switch to
    a model that can actually see the image."""
    if state.active_scenario is None:
        return None
    active_def = state.scenarios[state.active_scenario]
    candidates = [m for m in active_def["models"]
                  if scenario_name in m.get("fallback_for", []) and m["name"] in state.loaded
                  and (not require_vision or model_supports_vision(state, m["name"]))]
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

    # aiohttp.ClientSession()'s default total timeout is 300s — long enough to
    # feel "safe" in testing, short enough that a legitimately large/slow
    # generation under concurrent load can exceed it. That used to look
    # identical to a dead backend (both raise a ClientConnectionError
    # subclass) and would evict a perfectly healthy, still-generating model.
    # Bounded generously here instead of left at the session default so real
    # crashes are still caught without punishing slow-but-alive requests.
    forward_timeout = aiohttp.ClientTimeout(total=1800)
    async with state.http_session.post(url, json=body, timeout=forward_timeout) as resp:
        if not is_stream:
            data = await resp.json()
            if isinstance(data, dict) and "model" in data:
                data["model"] = client_label
            return web.json_response(data, status=resp.status)

        async for line_bytes in resp.content:
            line = line_bytes.decode("utf-8", errors="replace")
            if line.startswith("data: ") and real_name != client_label:
                payload = line[len("data: "):].strip()
                if payload and payload != "[DONE]":
                    try:
                        obj = json.loads(payload)
                        if "model" in obj:
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

_IMAGE_CONTENT_TYPES = {"image_url", "input_image", "image"}


def request_has_image(body):
    """True if any message's content includes an image part (OpenAI multimodal
    chat format) — checked so an image request can be kept off a model/alias
    that was never launched with --mmproj, instead of reaching llama-server
    and failing there with an opaque backend error."""
    for message in body.get("messages", []):
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in _IMAGE_CONTENT_TYPES:
                    return True
    return False


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

    has_image = request_has_image(body)
    if has_image and not model_supports_vision(state, target["name"]):
        return openai_error(400, f"Model '{requested_label}' does not accept image input "
                                  f"(no vision projector configured for it).", code="unsupported_content")

    if target["scenario"] is None:
        lm = state.standalone_loaded.get(target["name"])
        if lm is None:
            return openai_error(503, f"Standalone model '{target['name']}' is not loaded.")
        try:
            result = await forward(request, state, body, lm.port, requested_label, target["name"], state.max_wait_s)
            lm.last_used = time.monotonic()  # set on completion, not dispatch — see forward()'s call sites
            return result
        except asyncio.CancelledError:
            print(f"[proxy] request for '{target['name']}' (standalone) cancelled (client disconnected?) — not evicting")
            raise
        except aiohttp.ClientConnectorError as e:
            # A genuine failure to even establish a NEW connection — the
            # child is actually down. Popping the dict alone would just
            # forget it while it keeps running and holding VRAM forever;
            # evict_model actually tells it to shut down (a safe no-op if
            # it's already dead — shutdown_server checks poll() first).
            print(f"[proxy] '{target['name']}' (standalone) evicted after {type(e).__name__}: {e}")
            async with state.spawn_lock:
                stale = state.standalone_loaded.pop(target["name"], None)
                if stale is not None:
                    await evict_model(stale)
            return openai_error(503, f"Model '{requested_label}' is not responding (it may have crashed).",
                                 code="model_unavailable")
        except aiohttp.ClientConnectionError as e:
            # Broader than ClientConnectorError — also covers a reset/closed
            # transport mid-request, which is exactly what happens when the
            # INCOMING client (e.g. an IDE autocomplete cancelling on every
            # keystroke) drops its own request: that tears down our outbound
            # write to the backend without the backend being unhealthy at
            # all. Don't evict a perfectly good model over someone else's
            # cancel — see the "loaded then unloaded" incident this guards.
            print(f"[proxy] '{target['name']}' (standalone) request failed after {type(e).__name__}: {e} (not evicting)")
            return openai_error(503, f"Request to '{requested_label}' failed: {e}", code="request_failed")

    scenario_name = target["scenario"]
    async with state.spawn_lock:
        if state.active_scenario != scenario_name:
            active_priority = (state.scenarios[state.active_scenario]["priority"]
                                if state.active_scenario else -1)
            outranks_active = state.scenarios[scenario_name]["priority"] > active_priority
            if not outranks_active:
                alias = find_fallback(state, scenario_name, require_vision=has_image)
                if alias is not None:
                    try:
                        result = await forward(request, state, body, alias.port, requested_label, alias.name, state.max_wait_s)
                        alias.last_used = time.monotonic()
                        return result
                    except asyncio.CancelledError:
                        print(f"[proxy] request for '{alias.name}' (fallback alias) cancelled (client disconnected?) — not evicting")
                        raise
                    except aiohttp.ClientConnectorError as e:
                        print(f"[proxy] '{alias.name}' (fallback alias) evicted after {type(e).__name__}: {e}")
                        # Already holding spawn_lock here — mutate directly, don't re-acquire.
                        # See the standalone-path comment: must actually evict
                        # (kill), not just forget, or a still-alive process
                        # orphans and its VRAM is never reclaimed.
                        stale = state.loaded.pop(alias.name, None)
                        if stale is not None:
                            await evict_model(stale)
                        return openai_error(503, f"Model '{requested_label}' is not responding (it may have crashed).",
                                             code="model_unavailable")
                    except aiohttp.ClientConnectionError as e:
                        # See the standalone-path comment — a reset/closed
                        # transport mid-request is very often just the
                        # incoming client cancelling, not a dead backend.
                        print(f"[proxy] '{alias.name}' (fallback alias) request failed after {type(e).__name__}: {e} (not evicting)")
                        return openai_error(503, f"Request to '{requested_label}' failed: {e}", code="request_failed")
                # Nothing resident can cover it (or, for an image request, nothing
                # resident that both covers it and can actually see) — no cheaper
                # option than switching.
            await activate_scenario(state, scenario_name, state.load_timeout_s)

        lm = state.loaded.get(target["name"])
        if lm is None:
            scenario = state.scenarios[scenario_name]
            port = scenario["port"] if target["slot"] == "primary" else scenario["port_secondary"]
            ctx = state.reserved_ctx[target["name"]]
            print(f"[proxy] on-demand spawning '{target['name']}' on port {port} (ctx={ctx:,}) for request to '{requested_label}'")
            try:
                lm = await spawn_model(state, target["name"], port, ctx, state.load_timeout_s)
            except RuntimeError as e:
                return openai_error(503, str(e))
            state.loaded[target["name"]] = lm

    try:
        result = await forward(request, state, body, lm.port, requested_label, target["name"], state.max_wait_s)
        lm.last_used = time.monotonic()  # on completion, not dispatch — "idle" should mean actually idle
        return result
    except asyncio.CancelledError:
        print(f"[proxy] request for '{target['name']}' cancelled (client disconnected?) — not evicting")
        raise
    except aiohttp.ClientConnectorError as e:
        # A genuine failure to even establish a NEW connection — the child
        # actually died (e.g. a CUDA OOM abort under concurrent load) and
        # left state.loaded stale, so every future request would hit the
        # same dead port forever. evict_model actually shuts it down (a
        # safe no-op if already dead) instead of just forgetting it.
        print(f"[proxy] '{target['name']}' evicted after {type(e).__name__}: {e}")
        async with state.spawn_lock:
            stale = state.loaded.pop(target["name"], None)
            if stale is not None:
                await evict_model(stale)
        return openai_error(503, f"Model '{requested_label}' is not responding (it may have crashed) — "
                                  f"it will reload on the next request.", code="model_unavailable")
    except aiohttp.ClientConnectionError as e:
        # Broader than ClientConnectorError — also covers a reset/closed
        # transport mid-request. That's exactly what happens when the
        # INCOMING client (e.g. Continue's autocomplete, which cancels and
        # resends on every keystroke) drops its own request: it tears down
        # our outbound write to the backend without the backend being
        # unhealthy at all. Treating this as a crash was the actual cause of
        # the "model loads then immediately unloads" incident — the model
        # got killed once per keystroke for no real reason. Don't evict.
        print(f"[proxy] '{target['name']}' request failed after {type(e).__name__}: {e} (not evicting)")
        return openai_error(503, f"Request to '{requested_label}' failed: {e}", code="request_failed")


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


async def handle_activate_scenario(request):
    """POST /scenarios/{name}/activate — manual override from the dashboard.
    Unlike a normal request, this ignores priority entirely: the operator
    explicitly asked for this scenario, so it switches regardless of whether
    it would "outrank" whatever's currently active. That's only a one-time
    nudge, not a pin — the very next incoming request still routes by the
    usual priority/fallback rules, so a higher-priority scenario reclaims
    its spot the moment it's actually needed again."""
    state = request.app["state"]
    name = request.match_info["name"]
    if name not in state.scenarios:
        return openai_error(404, f"Unknown scenario '{name}'.", code="scenario_not_found")
    async with state.spawn_lock:
        if state.active_scenario != name:
            try:
                await activate_scenario(state, name, state.load_timeout_s)
            except RuntimeError as e:
                return openai_error(503, str(e))
    return web.json_response({"active_scenario": state.active_scenario})


async def handle_slots_port(request):
    """GET /slots/{port} — proxy to the child llama-server process.
    This is served by the proxy itself (same-origin), avoiding the
    cross-origin CORS issues of hitting child servers directly from the
    browser. Behind the same Bearer-key check as every other data route."""
    state = request.app["state"]
    port = request.match_info["port"]
    try:
        async with state.http_session.get(f"http://127.0.0.1:{port}/slots",
                                           timeout=aiohttp.ClientTimeout(total=5)) as resp:
            data = await resp.json()
            return web.json_response(data, status=resp.status)
    except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError):
        return web.json_response([], status=200)  # empty slots if child unreachable


async def handle_dashboard(request):
    """Serve the embedded dashboard HTML — no auth required."""
    return web.Response(text=DASHBOARD_HTML, content_type="text/html")


# ---------------------------------------------------------------------------
# Dashboard HTML (embedded — served at /dashboard, no auth needed)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Llama Priority Proxy — Dashboard</title>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #222633;
    --border: #2e3345;
    --text: #e1e4ed;
    --text-dim: #8b8fa3;
    --accent: #6c8cff;
    --accent-dim: #3a4a8c;
    --green: #4ade80;
    --yellow: #facc15;
    --red: #f87171;
    --orange: #fb923c;
    --radius: 10px;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 24px;
    min-height: 100vh;
  }
  h1 {
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 20px;
    letter-spacing: -0.02em;
  }
  h1 span { color: var(--accent); }
  .grid { display: grid; gap: 16px; }
  .grid-2 { grid-template-columns: 1fr 1fr; }
  .grid-3 { grid-template-columns: 2fr 1fr 1fr; }
  @media (max-width: 900px) {
    .grid-2, .grid-3 { grid-template-columns: 1fr; }
  }
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px;
  }
  .card-title {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-dim);
    margin-bottom: 12px;
    font-weight: 600;
  }
  .settings-row {
    display: flex;
    gap: 10px;
    align-items: center;
    flex-wrap: wrap;
  }
  .settings-row input { flex: 1; min-width: 200px; }
  input, select, button {
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 14px;
    font-size: 0.875rem;
    outline: none;
  }
  input:focus { border-color: var(--accent); }
  input::placeholder { color: var(--text-dim); }
  button {
    cursor: pointer;
    font-weight: 500;
    transition: all 0.15s;
    border: 1px solid var(--border);
  }
  button:hover { border-color: var(--accent); }
  .btn-primary { background: var(--accent); color: #fff; border-color: var(--accent); }
  .btn-primary:hover { background: #5a7ae8; }
  .btn-sm { padding: 4px 10px; font-size: 0.78rem; }
  .status-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 20px;
    font-size: 0.8rem; font-weight: 600;
  }
  .status-ok { background: rgba(74,222,128,0.12); color: var(--green); }
  .status-err { background: rgba(248,113,113,0.12); color: var(--red); }
  .status-dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
  .scenario-name { font-size: 1.6rem; font-weight: 700; color: var(--accent); }
  .scenario-meta { color: var(--text-dim); font-size: 0.82rem; margin-top: 4px; }
  .gpu-bar-wrap { height: 14px; background: var(--bg); border-radius: 7px; overflow: hidden; margin-top: 8px; }
  .gpu-bar { height: 100%; border-radius: 7px; transition: width 0.6s ease; background: linear-gradient(90deg, var(--green), var(--yellow)); }
  .gpu-text { margin-top: 6px; font-size: 0.85rem; color: var(--text-dim); }
  .gpu-text strong { color: var(--text); }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th { text-align: left; padding: 8px 10px; color: var(--text-dim); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; border-bottom: 1px solid var(--border); }
  td { padding: 8px 10px; border-bottom: 1px solid var(--border); vertical-align: middle; }
  tr:last-child td { border-bottom: none; }
  .model-name { font-weight: 600; }
  .model-name small { display: block; color: var(--text-dim); font-weight: 400; font-size: 0.75rem; }
  .proc-indicators { display: inline-flex; gap: 3px; align-items: center; }
  .proc-dot { width: 10px; height: 10px; border-radius: 3px; background: var(--border); transition: background 0.2s; }
  .proc-dot.active { background: var(--accent); }
  .proc-label { margin-left: 4px; font-size: 0.75rem; color: var(--text-dim); }
  .scenario-item {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 12px; border-radius: 6px;
    margin-bottom: 6px; font-size: 0.85rem;
  }
  .scenario-item.active { background: rgba(108,140,255,0.1); border: 1px solid var(--accent-dim); }
  .scenario-item:not(.active) { border: 1px solid var(--border); }
  .scenario-item .name { font-weight: 600; flex: 1; }
  .scenario-item .priority { color: var(--text-dim); font-size: 0.78rem; }
  .scenario-item .default-badge {
    background: rgba(74,222,128,0.12); color: var(--green);
    font-size: 0.68rem; padding: 2px 7px; border-radius: 10px; font-weight: 600;
  }
  .empty { color: var(--text-dim); font-size: 0.85rem; padding: 8px 0; }
  .refresh-info { font-size: 0.75rem; color: var(--text-dim); }
  .hidden { display: none; }
  .error-msg { color: var(--red); font-size: 0.82rem; margin-top: 8px; }
  .scenario-item[data-scenario] { cursor: pointer; }
  .scenario-item[data-scenario]:hover { border-color: var(--accent); }
  .modal-overlay {
    position: fixed; inset: 0; background: rgba(0,0,0,0.6);
    display: flex; align-items: center; justify-content: center; z-index: 100;
  }
  .modal-overlay.hidden { display: none; }
  .modal-card { max-width: 380px; width: 90%; }
  .modal-actions { display: flex; gap: 10px; justify-content: flex-end; margin-top: 16px; }
</style>
</head>
<body>

<h1>🐫 <span>Llama</span> Priority Proxy</h1>

<div id="loginGate" class="card" style="max-width:360px;margin:64px auto;text-align:center;">
  <div class="card-title">Enter Proxy API Key</div>
  <div class="settings-row" style="justify-content:center;">
    <input type="password" id="apiKeyInput" placeholder="API key" autocomplete="current-password" style="max-width:220px;">
    <button class="btn-primary" id="apiKeySubmit">Unlock</button>
  </div>
  <div class="error-msg hidden" id="loginError"></div>
</div>

<div id="dashboardApp" style="display:none;">

<div class="card" style="margin-bottom: 16px;">
  <div class="settings-row">
    <select id="refreshRate">
      <option value="1">1s</option>
      <option value="2">2s</option>
      <option value="3" selected>3s</option>
      <option value="5">5s</option>
      <option value="10">10s</option>
      <option value="30">30s</option>
    </select>
    <span class="refresh-info" id="refreshInfo">Refresh: auto</span>
    <button class="btn-sm" id="logoutBtn" style="margin-left:auto;">Log out</button>
  </div>
  <div class="error-msg hidden" id="errorMsg"></div>
</div>

<div class="grid grid-2" id="mainContent" style="display:none;">

  <div style="display:flex;flex-direction:column;gap:16px;">
    <div class="card">
      <div class="card-title">Active Scenario</div>
      <div class="scenario-name" id="activeScenario">—</div>
      <div class="scenario-meta" id="scenarioMeta"></div>
    </div>

    <div class="card">
      <div class="card-title">Loaded Models</div>
      <div id="modelsTable"></div>
    </div>

    <div class="card">
      <div class="card-title">All Scenarios</div>
      <div id="scenariosList"></div>
    </div>
  </div>

  <div style="display:flex;flex-direction:column;gap:16px;">
    <div class="card">
      <div class="card-title">Status</div>
      <div id="statusBadge"></div>
      <div class="scenario-meta" id="lastUpdate"></div>
    </div>

    <div class="card">
      <div class="card-title">GPU VRAM</div>
      <div class="gpu-text"><strong id="gpuUsed">—</strong> used of <strong id="gpuTotal">—</strong></div>
      <div class="gpu-bar-wrap"><div class="gpu-bar" id="gpuBar"></div></div>
      <div class="gpu-text" id="gpuFree" style="margin-top:4px;"></div>
    </div>

    <div class="card">
      <div class="card-title">Standalone Models</div>
      <div id="standaloneModels"></div>
    </div>
  </div>
</div>

<div id="emptyState" class="card" style="text-align:center;padding:48px 24px;display:none;">
  <div style="color:var(--text-dim);font-size:0.95rem;">Unable to connect to proxy.</div>
</div>

<div id="switchModal" class="modal-overlay hidden">
  <div class="card modal-card">
    <div class="card-title">Confirm Scenario Switch</div>
    <div style="margin:12px 0;font-size:0.9rem;">
      Switch active scenario to <strong id="switchModalName"></strong>?
      This evicts the currently loaded model(s) and may take up to a minute
      while the new scenario's models load.
    </div>
    <div class="error-msg hidden" id="switchModalError"></div>
    <div class="modal-actions">
      <button id="switchModalCancel">Cancel</button>
      <button class="btn-primary" id="switchModalConfirm">Switch</button>
    </div>
  </div>
</div>

</div>

<script>
// ---- Auth (Bearer key stored for this browser tab session only — cleared
// on tab/window close, never persisted to localStorage/disk) ----
const AUTH_STORAGE_KEY = 'llamaProxyApiKey';

function getStoredKey() { return sessionStorage.getItem(AUTH_STORAGE_KEY) || ''; }
function setStoredKey(k) { sessionStorage.setItem(AUTH_STORAGE_KEY, k); }
function clearStoredKey() { sessionStorage.removeItem(AUTH_STORAGE_KEY); }

function authHeaders() {
  const key = getStoredKey();
  return key ? { 'Authorization': `Bearer ${key}` } : {};
}

function showLogin(message) {
  stopPolling();
  document.getElementById('loginGate').classList.remove('hidden');
  document.getElementById('dashboardApp').style.display = 'none';
  const el = document.getElementById('loginError');
  if (message) {
    el.textContent = message;
    el.classList.remove('hidden');
  } else {
    el.classList.add('hidden');
  }
}

function showApp() {
  document.getElementById('loginGate').classList.add('hidden');
  document.getElementById('dashboardApp').style.display = '';
}

async function trySubmitKey(key) {
  setStoredKey(key);
  try {
    await apiFetch('/status');  // throws + clears key + re-shows login on 401
    showApp();
    startPolling();
  } catch (e) {
    if (getStoredKey()) {  // still set => not the 401 path, which already re-prompted
      showLogin(e.message);
    }
  }
}

document.getElementById('apiKeySubmit').addEventListener('click', () => {
  const val = document.getElementById('apiKeyInput').value.trim();
  if (val) trySubmitKey(val);
});
document.getElementById('apiKeyInput').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') document.getElementById('apiKeySubmit').click();
});
document.getElementById('logoutBtn').addEventListener('click', () => {
  clearStoredKey();
  showLogin();
  document.getElementById('apiKeyInput').value = '';
});

// ---- API (relative paths — same origin; Authorization header carries the
// session key set up by the login gate above) ----
async function apiFetch(path) {
  const resp = await fetch(path, { headers: authHeaders() });
  if (resp.status === 401) {
    clearStoredKey();
    showLogin('Invalid or expired API key.');
    throw new Error('Unauthorized');
  }
  if (!resp.ok) {
    const errBody = await resp.text().catch(() => '');
    throw new Error(`HTTP ${resp.status}: ${errBody || resp.statusText}`);
  }
  return resp.json();
}

// ---- Polling ----
let pollTimer = null;

async function fetchAndRender() {
  try {
    const status = await apiFetch('/status');
    showStatusOk('Connected');
    hideError();
    // Show main content on first success
    document.getElementById('mainContent').style.display = '';
    document.getElementById('emptyState').style.display = 'none';
    renderStatus(status);
  } catch(e) {
    showStatusErr('Disconnected');
    showError(e.message);
  }
  document.getElementById('refreshInfo').textContent = `Refresh: ${getRefreshRateSeconds()}s`;
}

function startPolling() {
  fetchAndRender();
  pollTimer = setInterval(fetchAndRender, getRefreshRate());
}

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

function getRefreshRate() {
  return parseInt(document.getElementById('refreshRate').value) * 1000;
}

function getRefreshRateSeconds() {
  return document.getElementById('refreshRate').value;
}

document.getElementById('refreshRate').addEventListener('change', () => {
  stopPolling();
  startPolling();
});

// ---- Render ----
function formatBytes(b) {
  if (b == null) return 'N/A';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let v = parseFloat(b);
  let i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(2)} ${units[i]}`;
}

function fmtNum(n) { return n.toLocaleString(); }

function fmtIdle(secs) {
  if (secs < 1) return 'just now';
  if (secs < 60) return `${Math.round(secs)}s`;
  if (secs < 3600) return `${(secs/60).toFixed(0)}m`;
  return `${(secs/3600).toFixed(1)}h`;
}

function parseByteStr(s) {
  if (!s) return 0;
  const m = s.match(/([\d.]+)\s*(B|KB|MB|GB|TB)/i);
  if (!m) return 0;
  const v = parseFloat(m[1]);
  const u = m[2].toUpperCase();
  const mult = {B:1, KB:1024, MB:1024*1024, GB:1024*1024*1024, TB:1024*1024*1024*1024}[u] || 1;
  return v * mult;
}

function showStatusOk(msg) {
  document.getElementById('statusBadge').innerHTML =
    `<span class="status-badge status-ok"><span class="status-dot"></span>${msg}</span>`;
}
function showStatusErr(msg) {
  document.getElementById('statusBadge').innerHTML =
    `<span class="status-badge status-err"><span class="status-dot"></span>${msg}</span>`;
}
function showError(msg) {
  const el = document.getElementById('errorMsg');
  el.textContent = msg;
  el.classList.remove('hidden');
}
function hideError() {
  document.getElementById('errorMsg').classList.add('hidden');
}

function renderStatus(data) {
  document.getElementById('activeScenario').textContent = data.active_scenario || 'None';
  const activeScenarioDef = data.scenarios?.find(s => s.active);
  const activeModels = data.loaded_models || [];
  const activeModelNames = activeModels.map(m => m.name);
  let metaParts = [];
  if (activeScenarioDef) {
    metaParts.push(`Priority: ${activeScenarioDef.priority}`);
    if (activeScenarioDef.default) metaParts.push('Default');
  }
  metaParts.push(`${activeModelNames.length} model${activeModelNames.length !== 1 ? 's' : ''} in scenario`);
  document.getElementById('scenarioMeta').textContent = metaParts.join(' · ');

  document.getElementById('lastUpdate').textContent = `Last update: ${new Date().toLocaleTimeString()}`;

  if (data.gpu) {
    document.getElementById('gpuTotal').textContent = data.gpu.total;
    document.getElementById('gpuUsed').textContent = data.gpu.used;
    if (data.gpu.free) document.getElementById('gpuFree').textContent = `Free: ${data.gpu.free}`;
    else document.getElementById('gpuFree').textContent = '';
    const totalBytes = parseByteStr(data.gpu.total);
    const usedBytes = parseByteStr(data.gpu.used);
    const pct = totalBytes > 0 ? Math.min(100, (usedBytes / totalBytes) * 100) : 0;
    const bar = document.getElementById('gpuBar');
    bar.style.width = pct + '%';
    if (pct > 80) bar.style.background = 'linear-gradient(90deg, #f87171, #fb923c)';
    else if (pct > 50) bar.style.background = 'linear-gradient(90deg, #facc15, #fb923c)';
    else bar.style.background = 'linear-gradient(90deg, #4ade80, #facc15)';
  }

  renderModelsTable(data.loaded_models || [], 'modelsTable', 'modelSlotCache');
  renderModelsTable(data.standalone_models || [], 'standaloneModels', 'standaloneSlotCache');
  renderScenariosList(data.scenarios || []);
}

function renderModelsTable(models, containerId, cacheKey) {
  const container = document.getElementById(containerId);
  if (models.length === 0) {
    container.innerHTML = '<div class="empty">No models loaded</div>';
    return;
  }

  // Build or update table rows — only rebuild if model set changed
  const slotCache = window[cacheKey] = window[cacheKey] || {};

  // Check if we need to rebuild (model set changed)
  const currentModels = new Set();
  container.querySelectorAll('tr[data-model]').forEach(tr => {
    currentModels.add(tr.getAttribute('data-model'));
  });
  const newModels = new Set(models.map(m => `${m.name}|${m.port}`));
  const needRebuild = currentModels.size !== newModels.size ||
                      ![...newModels].every(k => currentModels.has(k)) ||
                      ![...currentModels].every(k => newModels.has(k));

  if (needRebuild) {
    let html = `<table><thead><tr><th>Model</th><th>Port</th><th>Context</th><th>Processes</th><th>Idle</th></tr></thead><tbody>`;
    for (const m of models) {
      const slotInfo = slotCache[m.port] || { slots: 0, running: 0, loaded: false };
      const numDots = slotInfo.slots || 1;
      const indicators = Array.from({length: numDots}, (_, i) => {
        const active = i < slotInfo.running;
        return `<div class="proc-dot${active ? ' active' : ''}"></div>`;
      }).join('');
      const procLabel = slotInfo.loaded ? `${slotInfo.running}/${slotInfo.slots}` : 'loading...';
      html += `<tr data-model="${m.name}|${m.port}">
        <td class="model-name">${m.name}<small>ctx ${fmtNum(m.ctx_size)}</small></td>
        <td>${m.port}</td>
        <td>${fmtNum(m.ctx_size)}</td>
        <td class="proc-cell" data-port="${m.port}"><div class="proc-indicators">${indicators}<span class="proc-label">${procLabel}</span></div></td>
        <td class="idle-cell" data-idle-s="${m.idle_for_s != null ? m.idle_for_s : ''}">${m.idle_for_s != null ? fmtIdle(m.idle_for_s) : '—'}</td>
      </tr>`;
    }
    html += '</tbody></table>';
    container.innerHTML = html;
  } else {
    // Update existing rows without rebuilding
    for (const m of models) {
      const row = container.querySelector(`tr[data-model="${m.name}|${m.port}"]`);
      if (row) {
        const idleCell = row.querySelector('.idle-cell');
        if (idleCell) {
          idleCell.dataset.idleS = m.idle_for_s != null ? m.idle_for_s : '';
          if (!idleCell.classList.contains('is-active')) {
            idleCell.textContent = m.idle_for_s != null ? fmtIdle(m.idle_for_s) : '—';
          }
        }
      }
    }
  }

  // Update process indicators from slot polling
  for (const m of models) {
    fetchSlots(m.port);
  }
}

async function fetchSlots(port) {
  try {
    const resp = await fetch(`/slots/${port}`, { headers: authHeaders() });
    if (resp.ok) {
      const data = await resp.json();
      const slots = Array.isArray(data) ? data : (data.slots || []);
      const total = slots.length;
      const active = new Array(total).fill(false);
      slots.forEach((s, i) => {
        const idx = (typeof s.id === 'number') ? s.id : i;
        if (idx >= 0 && idx < total) active[idx] = !!s.is_processing;
      });
      updateProcIndicators(port, active, total);
    } else {
      updateProcIndicators(port, [], 0);
    }
  } catch(e) {
    updateProcIndicators(port, [], 0);
  }
}

function updateProcIndicators(port, active, total) {
  const procCell = document.querySelector(`.proc-cell[data-port="${port}"]`);
  if (!procCell) return;
  const slots = total || 1;
  const running = active.filter(Boolean).length;
  const dots = Array.from({length: slots}, (_, i) => {
    return `<div class="proc-dot${active[i] ? ' active' : ''}"></div>`;
  }).join('');
  procCell.innerHTML = `<div class="proc-indicators">${dots}<span class="proc-label">${running}/${slots}</span></div>`;

  const idleCell = procCell.parentElement?.querySelector('.idle-cell');
  if (idleCell) {
    if (running > 0) {
      idleCell.classList.add('is-active');
      idleCell.textContent = 'active';
    } else {
      idleCell.classList.remove('is-active');
      const raw = idleCell.dataset.idleS;
      idleCell.textContent = (raw !== '' && raw != null) ? fmtIdle(parseFloat(raw)) : '—';
    }
  }
}

function renderScenariosList(scenarios) {
  const container = document.getElementById('scenariosList');
  if (scenarios.length === 0) {
    container.innerHTML = '<div class="empty">No scenarios</div>';
    return;
  }
  let html = '';
  for (const s of scenarios) {
    const cls = s.active ? 'active' : '';
    const defaultBadge = s.default ? '<span class="default-badge">default</span>' : '';
    const clickAttr = s.active ? '' : ` data-scenario="${s.name}"`;
    html += `<div class="scenario-item ${cls}"${clickAttr}>
      ${s.active ? '<span style="color:var(--accent);font-size:0.8rem;">●</span>' : '<span style="color:var(--text-dim);font-size:0.8rem;">○</span>'}
      <span class="name">${s.name} ${defaultBadge}</span>
      <span class="priority">#${s.priority}</span>
    </div>`;
  }
  container.innerHTML = html;
}

// ---- Scenario switching (manual override, confirmed via modal) ----
let pendingScenario = null;

function openSwitchModal(name) {
  pendingScenario = name;
  document.getElementById('switchModalName').textContent = name;
  document.getElementById('switchModalError').classList.add('hidden');
  const btn = document.getElementById('switchModalConfirm');
  btn.disabled = false;
  btn.textContent = 'Switch';
  document.getElementById('switchModal').classList.remove('hidden');
}

function closeSwitchModal() {
  document.getElementById('switchModal').classList.add('hidden');
  pendingScenario = null;
}

document.getElementById('scenariosList').addEventListener('click', (e) => {
  const item = e.target.closest('.scenario-item[data-scenario]');
  if (!item) return;
  openSwitchModal(item.getAttribute('data-scenario'));
});

document.getElementById('switchModalCancel').addEventListener('click', closeSwitchModal);

document.getElementById('switchModalConfirm').addEventListener('click', async () => {
  if (!pendingScenario) return;
  const btn = document.getElementById('switchModalConfirm');
  const errEl = document.getElementById('switchModalError');
  btn.disabled = true;
  btn.textContent = 'Switching… (can take up to a minute)';
  errEl.classList.add('hidden');
  try {
    const resp = await fetch(`/scenarios/${encodeURIComponent(pendingScenario)}/activate`,
                              { method: 'POST', headers: authHeaders() });
    if (resp.status === 401) {
      clearStoredKey();
      closeSwitchModal();
      showLogin('Invalid or expired API key.');
      return;
    }
    if (!resp.ok) {
      const body = await resp.json().catch(() => null);
      throw new Error(body?.error?.message || `HTTP ${resp.status}`);
    }
    closeSwitchModal();
    await fetchAndRender();
  } catch (e) {
    errEl.textContent = e.message;
    errEl.classList.remove('hidden');
    btn.disabled = false;
    btn.textContent = 'Switch';
  }
});

// Init
document.getElementById('refreshInfo').textContent = 'Refresh: auto';

async function init() {
  const storedKey = getStoredKey();
  if (storedKey) {
    await trySubmitKey(storedKey);
    return;
  }
  // No stored key yet — try without one first. If PROXY_API_KEY isn't set
  // on the proxy at all, auth is off for every route (not just this one),
  // so an anonymous /status call already succeeds and there's nothing to
  // log in to. Only fall back to the login gate if that actually fails,
  // instead of always prompting for a "key" that might not even exist.
  try {
    const resp = await fetch('/status');
    if (resp.ok) {
      showApp();
      startPolling();
      return;
    }
  } catch (e) {
    // network error — fall through to the login gate below
  }
  showLogin();
}
init();
</script>
</body>
</html>"""


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
    # aiohttp's default client_max_size (1 MiB) is smaller than a single
    # base64-encoded image in a chat completion request — raised here so
    # multimodal requests don't get rejected with 413 before handle_completion
    # ever sees them.
    app = web.Application(middlewares=[auth_middleware], client_max_size=64 * 1024 * 1024)
    app["state"] = state
    app["proxy_api_key"] = proxy_api_key
    app.router.add_post("/v1/chat/completions", handle_completion)
    app.router.add_post("/v1/completions", handle_completion)
    app.router.add_post("/v1/embeddings", handle_completion)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/status", handle_status)
    app.router.add_post("/scenarios/{name}/activate", handle_activate_scenario)
    app.router.add_get("/slots/{port}", handle_slots_port)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/dashboard", handle_dashboard)
    app.router.add_get("/", handle_dashboard)  # / → dashboard
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