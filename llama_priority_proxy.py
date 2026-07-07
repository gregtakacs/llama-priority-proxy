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
  image_backend.json        - optional: ComfyUI reverse-proxy + coexistence
                              settings (base_url/coexist_scenario/poll+revert
                              delays) — absent means /comfyui/* returns 503

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
DEFAULT_EVICT_DRAIN_TIMEOUT_S = 40  # see drain_in_flight_before_evict -- ~8k tokens at 200 tok/s

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


def preempted_error(requested_label):
    """Specific, friendly error for a request whose model was killed by
    drain_in_flight_before_evict's forced-eviction path (see LoadedModel.
    preempted_reason) — distinct from the generic "model crashed" error so
    the caller (and, ultimately, the chat UI) can tell "the GPU was needed
    for image generation, just retry" apart from an actual backend failure."""
    return openai_error(503, f"'{requested_label}' was interrupted because image generation needed the GPU. "
                              f"Please retry your message.", code="image_generation_preempted")


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
    # /comfyui/* is also exempt: ComfyUI has no auth of its own, and OpenWebUI's
    # ComfyUI client may not attach the Authorization header to its websocket
    # handshake — same trust posture as the llama-server children today
    # (protected by network exposure, not an app-layer check).
    if api_key and request.path not in exempt_paths and not request.path.startswith("/comfyui/"):
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
    # Optional — the image-generation feature is entirely inert (routes under
    # /comfyui return 503) if this file doesn't exist.
    image_cfg = load_json(os.path.join(config_dir, "image_backend.json"), None)

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

    return options, registry, standalone, scenarios, image_cfg


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
        # Count of forward() calls currently using this model, so
        # activate_scenario's eviction loop (see drain_in_flight_before_evict)
        # can wait for them to finish instead of killing the process out from
        # under a live request/stream.
        self.in_flight = 0
        # Set by drain_in_flight_before_evict when this exact model gets
        # force-evicted with requests still in flight (drain timeout
        # exceeded). The in-flight request's own exception handler in
        # handle_completion checks this (via its already-held `lm`/`alias`
        # reference, which still points at this same object after eviction)
        # to return a specific, friendly error instead of a generic
        # "model crashed" one.
        self.preempted_reason = None


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
        self.image_cfg = None            # parsed config/image_backend.json, or None if absent
        self.pre_comfy_scenario = None   # scenario active right before comfy_coexist took over
        self.pre_comfy_last_used = {}    # model name -> last_used snapshot from right before that, see activate_comfy_coexist
        self.comfy_revert_task = None    # single in-flight watch_comfy_queue_and_revert task, or None
        self.comfy_activated_at = None   # monotonic timestamp of the most recent real comfy_coexist activation


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


async def drain_in_flight_before_evict(lm, name, timeout_s, poll_interval_s=0.2):
    """Wait for any forward() calls currently using `lm` to finish before it
    gets evicted. forward() deliberately runs WITHOUT holding state.spawn_lock
    (a long chat stream can't hold the lock for its whole duration — see
    handle_completion), which means activate_scenario's eviction loop (which
    DOES hold the lock) previously had zero visibility into whether the model
    it was about to kill was mid-request. In practice this model is very
    often the exact one an image-generation tool call's own conversation
    turn is using (or a concurrent title/tags/follow-up background call to
    the same model) — evicting out from under it doesn't fail cleanly, it
    kills the connection with a raw disconnect (ServerDisconnectedError /
    ClientConnectionResetError), silently losing that turn's response even
    though the image generation itself goes on to succeed independently.
    Bounded: comfy_coexist genuinely needs the VRAM and can't wait forever,
    so if requests are still in flight after timeout_s, evict anyway with a
    loud warning rather than block image generation indefinitely -- but flag
    `lm` first so the request(s) still in flight get a clean, specific error
    (see handle_completion's ClientConnectorError/ClientConnectionError
    handlers) instead of an opaque raw disconnect."""
    if lm.in_flight <= 0:
        return
    deadline = time.monotonic() + timeout_s
    while lm.in_flight > 0 and time.monotonic() < deadline:
        await asyncio.sleep(poll_interval_s)
    if lm.in_flight > 0:
        print(f"[proxy] WARNING: evicting '{name}' with {lm.in_flight} request(s) still in flight "
              f"after waiting {timeout_s}s for them to drain — proceeding anyway, this will disconnect them")
        lm.preempted_reason = "image_generation"


# ---------------------------------------------------------------------------
# Image-generation coexistence (ComfyUI) — the "small model" is just another
# scenario (see config/scenario_comfy_coexist.json), reusing activate_scenario
# entirely rather than needing separate eviction/spawn logic. See
# handle_comfyui_proxy for where activate_comfy_coexist is actually triggered
# (the moment a POST /comfyui/prompt comes through) and
# watch_comfy_queue_and_revert for how the switch back is triggered.
# ---------------------------------------------------------------------------

async def activate_comfy_coexist(state):
    """No-op if already active. Remembers whatever was actually active so
    watch_comfy_queue_and_revert can restore it later instead of assuming
    "default" — a chat could've been happening in a non-default scenario when
    the image request interrupted it. Also snapshots each currently-loaded
    model's last_used (state.loaded only ever holds the outgoing scenario's
    own members at this point) so that restore can resume each one's idle
    clock from where it actually was instead of granting a fresh full
    keep_alive window just for having been caught in an unrelated image
    generation — see restore_pre_comfy_state."""
    coexist_name = state.image_cfg["coexist_scenario"]
    if state.active_scenario == coexist_name:
        return
    state.pre_comfy_scenario = state.active_scenario
    state.pre_comfy_last_used = {name: lm.last_used for name, lm in state.loaded.items()}
    await activate_scenario(state, coexist_name, state.load_timeout_s)
    state.comfy_activated_at = time.monotonic()


async def free_comfyui_memory(state):
    """Best-effort — logs and swallows errors/timeouts, never raises (mirrors
    the tolerant style wait_for_capacity already uses for /slots
    unreachability): a ComfyUI hiccup here shouldn't block an LLM spawn."""
    url = f"{state.image_cfg['base_url']}/free"
    try:
        async with state.http_session.post(url, json={"unload_models": True, "free_memory": True},
                                            timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                print(f"[proxy] WARNING: ComfyUI /free returned HTTP {resp.status}")
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        print(f"[proxy] WARNING: failed to free ComfyUI memory: {e}")


async def comfy_queue_busy(state):
    """True only if ComfyUI's own queue actually has something running/pending
    RIGHT NOW. Used by watch_comfy_queue_and_revert to detect a natural,
    patient drain (generation actually finished) — NOT used to decide whether
    a competing request may forcibly preempt comfy_coexist early; see
    scenario_fits_after_comfy_evict for that (a live VRAM check is the more
    honest signal there, since queue state alone doesn't say whether ComfyUI
    is still mid-load and about to claim more memory than it has yet).
    Unreachable/unparseable ComfyUI reads as "not busy" — don't block
    indefinitely on an unknown, same tolerant style as elsewhere here."""
    cfg = state.image_cfg
    try:
        async with state.http_session.get(f"{cfg['base_url']}/queue",
                                           timeout=aiohttp.ClientTimeout(total=5)) as resp:
            data = await resp.json()
            return bool(data.get("queue_running") or data.get("queue_pending"))
    except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError):
        return False


async def scenario_fits_after_comfy_evict(state, scenario_name):
    """Empirical (live nvidia-smi) check for whether `scenario_name` could
    actually load right now if the comfy_coexist model were evicted (or as-is
    if it's already gone, e.g. crashed) — ground truth for "is it actually
    safe to preempt comfy_coexist", rather than inferring it from an indirect
    signal like ComfyUI's own queue state. Reuses solve_scenario_sizes, but
    with a LIVE-measured budget instead of the static headroom-based one
    activate_scenario itself uses for a normal switch: ComfyUI's real
    footprint at any given moment isn't something this proxy predicts —
    nvidia-smi already knows it exactly. state.loaded only ever holds
    comfy_coexist's own member(s) while this scenario is active, so summing
    all of it is exactly "what evicting comfy_coexist would free"."""
    gpu_total = gpu_total_bytes(state.gpu_index)
    gpu_used = gpu_used_bytes(state.gpu_index)
    if gpu_used is None:
        return True  # can't measure — don't block on an unknown
    freed = sum(predicted_vram(state.registry[lm.name], lm.ctx_size) for lm in state.loaded.values())
    available = gpu_total - gpu_used + freed - int(state.headroom_gb * (1024 ** 3))
    sizes = solve_scenario_sizes(state.scenarios[scenario_name], state.registry, available)
    total_needed = sum(predicted_vram(state.registry[name], ctx) for name, ctx in sizes.items())
    return total_needed <= available and all(ctx > 0 for ctx in sizes.values())


def _pre_comfy_last_used_if_valid(state, name, now):
    """Returns the pre-interruption last_used snapshot for `name` if it hasn't
    already idled past its own keep_alive since being snapshotted, else None
    (either it was never loaded before the interruption, or it's since expired).
    Shared by scenario_has_restorable_members and restore_pre_comfy_state so
    the two can't disagree about what still counts as "worth restoring"."""
    saved_last_used = state.pre_comfy_last_used.get(name)
    if saved_last_used is None:
        return None
    keep_alive = state.options.get(name, {}).get("keep_alive", "")
    timeout = parse_keep_alive(keep_alive)
    if timeout is not None and (now - saved_last_used) >= timeout:
        return None
    return saved_last_used


def scenario_has_restorable_members(state, scenario_name):
    """True if reactivating scenario_name would actually resume at least one
    previously-resident model. False means every one of its members either
    wasn't loaded before the interruption or has since idled out DURING it —
    i.e. there's nothing left to resume, so reactivating it would just relabel
    an empty shell (see watch_comfy_queue_and_revert, which falls back to the
    default scenario instead when this is False)."""
    now = time.monotonic()
    return any(
        _pre_comfy_last_used_if_valid(state, m["name"], now) is not None
        for m in state.scenarios[scenario_name]["models"]
    )


async def restore_pre_comfy_state(state, scenario_name, load_timeout_s):
    """Called right after activate_scenario reactivates the scenario comfy_coexist
    interrupted. activate_scenario itself only lazily restores members that are
    permanently pinned/eager (see eager_load_pinned_members) — everything else
    would otherwise sit unloaded until the next real request, which is exactly
    the "no warm-loading benefit at all" bug this fixes. For every member that
    was ACTUALLY resident right before the interruption (state.pre_comfy_last_used,
    snapshotted in activate_comfy_coexist) and hasn't since idled out, reload it
    here and restore its last_used to that snapshot — so its idle-eviction clock
    resumes from where it genuinely was instead of getting a fresh full
    keep_alive window just for having been caught in an unrelated image
    generation."""
    scenario = state.scenarios[scenario_name]
    now = time.monotonic()
    for m in scenario["models"]:
        name = m["name"]
        saved_last_used = _pre_comfy_last_used_if_valid(state, name, now)
        if saved_last_used is None:
            continue
        if name in state.loaded:
            state.loaded[name].last_used = saved_last_used
            continue
        port = scenario["port"] if m.get("slot", "primary") == "primary" else scenario["port_secondary"]
        ctx = state.reserved_ctx[name]
        try:
            lm = await spawn_model(state, name, port, ctx, load_timeout_s)
            lm.last_used = saved_last_used
            state.loaded[name] = lm
            keep_alive = state.options.get(name, {}).get("keep_alive", "")
            timeout = parse_keep_alive(keep_alive)
            remaining = f"{timeout - (now - saved_last_used):.0f}s left on its keep_alive" if timeout is not None else "no keep_alive limit"
            print(f"[proxy] restored '{name}' for scenario '{scenario_name}' ({remaining})")
        except RuntimeError as e:
            print(f"[proxy] WARNING: failed to restore model '{name}': {e}")
    state.pre_comfy_last_used = {}


async def watch_comfy_queue_and_revert(state):
    """Started after every forwarded /prompt (see handle_comfyui_proxy), which
    cancels any previous instance first — only one of these runs at a time.
    Polls ComfyUI's queue until it drains, waits a settle window to absorb
    back-to-back submissions, then frees ComfyUI and restores whatever
    scenario was active before comfy_coexist took over."""
    cfg = state.image_cfg
    while True:
        await asyncio.sleep(cfg["queue_poll_interval_s"])
        if not await comfy_queue_busy(state):
            break  # drained (or unreachable) — either way, don't hold the coexistence model hostage indefinitely

    await asyncio.sleep(cfg["revert_delay_s"])
    async with state.spawn_lock:
        if state.active_scenario != cfg["coexist_scenario"]:
            return  # already legitimately preempted by something else (e.g. a vision request)
        # activate_scenario's own hook frees ComfyUI's memory as part of
        # leaving comfy_coexist — no need to duplicate that call here.
        default_name = next(name for name, s in state.scenarios.items() if s.get("default"))
        target = state.pre_comfy_scenario or default_name
        if target != default_name and not scenario_has_restorable_members(state, target):
            # Everything pre_comfy_scenario had running idled out DURING the
            # interruption itself (image generation can run well past a short
            # scenario's own keep_alive) -- reactivating it would just relabel
            # an empty shell with nothing loaded and no path back to default
            # until a real request forces a switch (see maybe_evict_idle_scenario's
            # own "if not state.loaded: return" guard, which can't help here since
            # it never fires for an already-empty scenario). Equivalent to there
            # having been no prior scenario at all: go straight to default.
            print(f"[proxy] '{target}' fully idled out during image generation — reverting to default '{default_name}' instead")
            target = default_name
        print(f"[proxy] ComfyUI queue drained — reverting to '{target}'")
        await activate_scenario(state, target, state.load_timeout_s)
        await restore_pre_comfy_state(state, target, state.load_timeout_s)


async def activate_scenario(state, scenario_name, load_timeout_s):
    """Evict whatever the new scenario doesn't want, group-solve ctx sizes for
    all its members at once, and switch. A model shared by both the outgoing
    and incoming scenario (same name — see build_label_index's dedup-by-name)
    is kept running across the switch rather than torn down and relaunched,
    but only if its already-loaded ctx-size matches what the new scenario
    would reserve for it: llama-server's ctx is fixed at launch, so a size
    mismatch means a relaunch is unavoidable, not just an optimization we're
    skipping."""
    if (state.image_cfg is not None and scenario_name != state.active_scenario
            and state.active_scenario == state.image_cfg.get("coexist_scenario")):
        # Leaving comfy_coexist via some path other than the queue-watcher
        # (e.g. a vision request forcing a real switch) — free ComfyUI too,
        # since the coexistence model is being evicted anyway.
        await free_comfyui_memory(state)

    scenario = state.scenarios[scenario_name]
    gpu_total = gpu_total_bytes(state.gpu_index)
    # Standalone models (nomic) are always resident and never evicted — their
    # VRAM is permanently spoken for, so the scenario's own group-sizing must
    # treat it the same as headroom, not as capacity it could ever claim.
    standalone_vram = sum(predicted_vram(state.registry[lm.name], lm.ctx_size)
                           for lm in state.standalone_loaded.values())
    budget = (gpu_total - int(state.headroom_gb * (1024 ** 3)) - standalone_vram
              - scenario.get("comfy_reserved_vram_bytes", 0))
    new_sizes = solve_scenario_sizes(scenario, state.registry, budget)

    new_names = {m["name"] for m in scenario["models"]}
    for name, lm in list(state.loaded.items()):
        if name in new_names and lm.ctx_size == new_sizes.get(name):
            continue  # shared with the new scenario at the same size — keep it running
        await drain_in_flight_before_evict(lm, name, state.evict_drain_timeout_s)
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
    not just reversion to the default scenario.

    A member can also be marked "eager": true directly in the scenario's own
    config, independent of the model's global keep_alive — used by
    comfy_coexist (see its _comment) so its one model loads the instant that
    scenario activates rather than waiting for a chat request to trigger the
    normal on-demand spawn path, without changing that same model's (lazy)
    behavior in any other scenario it also appears in."""
    scenario = state.scenarios[scenario_name]
    for m in scenario["models"]:
        if m["name"] in state.loaded:
            continue
        keep_alive = state.options.get(m["name"], {}).get("keep_alive", "")
        if parse_keep_alive(keep_alive) is not None and not m.get("eager", False):
            continue  # not pinned, and not scenario-forced-eager — stays lazy
        port = scenario["port"] if m.get("slot", "primary") == "primary" else scenario["port_secondary"]
        ctx = state.reserved_ctx[m["name"]]
        try:
            lm = await spawn_model(state, m["name"], port, ctx, load_timeout_s)
            state.loaded[m["name"]] = lm
            print(f"[proxy] eager-loaded model '{m['name']}' for scenario '{scenario_name}'")
        except RuntimeError as e:
            print(f"[proxy] WARNING: failed to eager-load model '{m['name']}': {e}")


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
    if state.image_cfg is not None and state.active_scenario == state.image_cfg.get("coexist_scenario"):
        # comfy_coexist has its own dedicated, ComfyUI-aware revert path (see
        # watch_comfy_queue_and_revert) — the coexistence model looking idle
        # from a CHAT perspective says nothing about whether ComfyUI itself
        # is still mid-generation. Without this exemption, a long generation
        # with no chat traffic in the meantime would get forcibly (and
        # incorrectly) reverted here well before ComfyUI actually finishes —
        # see the "5 minutes idle, comfy busy for 15" incident this guards.
        return
    # No "if not state.loaded: return" guard here on purpose: every path that
    # calls activate_scenario (and its internal eager-loading) does so while
    # holding state.spawn_lock, and this function is only ever reached via
    # idle_eviction_sweep, which needs that same lock first -- so there's no
    # window where a scenario switch is "still starting up" when this runs.
    # An empty state.loaded here means every member died individually (e.g.
    # the ClientConnectorError crash-recovery path in handle_completion, which
    # pops a dead model without touching active_scenario) while this scenario
    # was still nominally active. Treating that as "not idle yet" left the
    # proxy stuck showing an active-but-empty scenario forever, with no path
    # back to default until some unrelated request happened to force a real
    # switch. The loop below already does the right thing on an empty dict —
    # it just never runs, and falls through to reverting to default.
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
                # resident that both covers it and can actually see) — normally
                # "no cheaper option than switching", EXCEPT when comfy_coexist
                # is active: preempting it isn't automatically safe, so decide
                # with two checks instead of just assuming either way.
            if state.image_cfg is not None and state.active_scenario == state.image_cfg.get("coexist_scenario"):
                grace_s = state.image_cfg.get("startup_grace_s", 60)
                within_grace = (state.comfy_activated_at is not None
                                 and time.monotonic() - state.comfy_activated_at < grace_s)
                # 1. Startup grace period: comfy_coexist may have JUST activated
                #    and ComfyUI may still be mid-load — nvidia-smi's live
                #    reading would understate its eventual footprint, so don't
                #    even attempt a fit-check yet; assume busy unconditionally.
                # 2. Otherwise, check empirically (live nvidia-smi, not an
                #    inferred/stale signal) whether the requested scenario
                #    would actually fit if the coexistence model were evicted
                #    — see scenario_fits_after_comfy_evict's own docstring for
                #    why this replaced trusting active_scenario alone (that
                #    flag can outlive the coexistence model actually being
                #    loaded, e.g. if it crashed, which previously left every
                #    request stuck here forever with no path back to normal).
                if within_grace or not await scenario_fits_after_comfy_evict(state, scenario_name):
                    return openai_error(503, f"Image generation is active and holding GPU memory — "
                                              f"'{requested_label}' can't be loaded until it finishes.",
                                         code="image_generation_busy")
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

    # in_flight is read by activate_scenario (via drain_in_flight_before_evict)
    # to avoid killing this model out from under this exact request — spawn_lock
    # is deliberately NOT held across forward() (a long chat stream can't hold
    # it for its whole duration), so this counter is the only thing standing
    # between a concurrent scenario switch (e.g. an image-generation tool call
    # activating comfy_coexist) and evicting a perfectly healthy, in-flight model.
    lm.in_flight += 1
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
        if lm.preempted_reason == "image_generation":
            return preempted_error(requested_label)
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
        # BUT: if drain_in_flight_before_evict already flagged this exact
        # model as force-evicted for image generation, that's not a spurious
        # client-side cancel -- return the specific, actionable error instead.
        print(f"[proxy] '{target['name']}' request failed after {type(e).__name__}: {e} (not evicting)")
        if lm.preempted_reason == "image_generation":
            return preempted_error(requested_label)
        return openai_error(503, f"Request to '{requested_label}' failed: {e}", code="request_failed")
    finally:
        lm.in_flight -= 1


async def handle_status(request):
    state = request.app["state"]
    now = time.monotonic()
    gpu_total = gpu_total_bytes(state.gpu_index)
    gpu_used = gpu_used_bytes(state.gpu_index)
    # Idle-eviction only ever reverts a non-default scenario (see
    # maybe_evict_idle_scenario) — while the default scenario is active,
    # nothing gets evicted on idle no matter how long it's been sitting there.
    active_is_default = (state.active_scenario is not None
                          and state.scenarios[state.active_scenario].get("default", False))
    scenario_models = []
    if state.active_scenario is not None:
        scenario = state.scenarios[state.active_scenario]
        for m in scenario["models"]:
            lm = state.loaded.get(m["name"])
            port = scenario["port"] if m.get("slot", "primary") == "primary" else scenario["port_secondary"]
            keep_alive = state.options.get(m["name"], {}).get("keep_alive", "")
            scenario_models.append({
                "name": m["name"],
                "label": m.get("label", m["name"]),
                "slot": m.get("slot", "primary"),
                "port": port,
                "ctx_size": state.reserved_ctx.get(m["name"]),
                "loaded": lm is not None,
                "keep_alive": keep_alive,
                "idle_for_s": None if lm is None else round(now - lm.last_used, 1),
                "evict_in_s": (None if lm is None or active_is_default or parse_keep_alive(keep_alive) is None
                               else round(max(0.0, parse_keep_alive(keep_alive) - (now - lm.last_used)), 1)),
            })
    return web.json_response({
        "active_scenario": state.active_scenario,
        "scenario_models": scenario_models,
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
        "image_backend": (None if state.image_cfg is None else {
            "active": state.active_scenario == state.image_cfg["coexist_scenario"],
            "pre_comfy_scenario": state.pre_comfy_scenario,
        }),
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


async def handle_load_model(request):
    """POST /models/{name}/load — manual force-load from the dashboard. Only
    valid for a member of the currently active scenario: that's what pins down
    which port/ctx-size to launch it on (state.reserved_ctx, solved once at
    scenario activation — see solve_scenario_sizes). A no-op if already
    loaded, same idempotent shape as handle_activate_scenario."""
    state = request.app["state"]
    name = request.match_info["name"]
    if state.active_scenario is None:
        return openai_error(409, "No active scenario.", code="no_active_scenario")
    scenario = state.scenarios[state.active_scenario]
    model_def = next((m for m in scenario["models"] if m["name"] == name), None)
    if model_def is None:
        return openai_error(404, f"'{name}' is not a member of the active scenario '{state.active_scenario}'.",
                             code="model_not_found")
    async with state.spawn_lock:
        if name not in state.loaded:
            port = scenario["port"] if model_def.get("slot", "primary") == "primary" else scenario["port_secondary"]
            ctx = state.reserved_ctx[name]
            try:
                state.loaded[name] = await spawn_model(state, name, port, ctx, state.load_timeout_s)
            except RuntimeError as e:
                return openai_error(503, str(e))
    return web.json_response({"name": name, "loaded": True})


async def handle_evict_model(request):
    """POST /models/{name}/evict — manual force-evict from the dashboard.
    A no-op if already not loaded. Doesn't touch state.active_scenario or
    state.reserved_ctx — the model simply stays absent until the next
    scenario activation (or a future force-load) brings it back."""
    state = request.app["state"]
    name = request.match_info["name"]
    async with state.spawn_lock:
        lm = state.loaded.pop(name, None)
        if lm is not None:
            await evict_model(lm)
    return web.json_response({"name": name, "loaded": False})


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


_HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
                "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length"}


async def _relay_ws(source, dest):
    """Pumps frames from source to dest, preserving TEXT vs BINARY (ComfyUI's
    progress socket sends binary preview-image frames) until either side
    closes. A dumb passthrough — completion detection for the coexistence
    swap is done independently via watch_comfy_queue_and_revert, not by
    inspecting these frames."""
    async for msg in source:
        if msg.type == aiohttp.WSMsgType.TEXT:
            await dest.send_str(msg.data)
        elif msg.type == aiohttp.WSMsgType.BINARY:
            await dest.send_bytes(msg.data)
        elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
            break


async def handle_comfyui_proxy(request):
    """Reverse-proxies everything under /comfyui/* to the real ComfyUI
    instance. Deliberately NOT the JSON/SSE-aware forward() used for chat:
    ComfyUI's own protocol (POST /prompt, GET /history/{id}, GET /view, and a
    progress websocket at /ws) isn't OpenAI-shaped, and /view returns raw
    image bytes.

    The one place this does more than blind passthrough: a POST to /prompt
    (the only call that actually triggers ComfyUI loading a checkpoint)
    activates the comfy_coexist scenario first, guaranteeing VRAM headroom
    exists before ComfyUI ever starts loading — see activate_comfy_coexist."""
    state = request.app["state"]
    if state.image_cfg is None:
        return openai_error(503, "Image generation is not configured (no config/image_backend.json).")

    tail = request.match_info["tail"]
    if request.method == "POST" and tail == "prompt":
        if state.comfy_revert_task is not None and not state.comfy_revert_task.done():
            state.comfy_revert_task.cancel()
        async with state.spawn_lock:
            try:
                await activate_comfy_coexist(state)
            except RuntimeError as e:
                return openai_error(503, str(e))

    target_url = f"{state.image_cfg['base_url']}/{tail}"
    if request.query_string:
        target_url += f"?{request.query_string}"

    if request.headers.get("Upgrade", "").lower() == "websocket":
        ws_server = web.WebSocketResponse()
        await ws_server.prepare(request)
        async with state.http_session.ws_connect(target_url) as ws_client:
            tasks = [asyncio.create_task(_relay_ws(ws_server, ws_client)),
                     asyncio.create_task(_relay_ws(ws_client, ws_server))]
            _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        if not ws_server.closed:
            await ws_server.close()
        return ws_server

    headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_HEADERS}
    body = await request.read()
    async with state.http_session.request(request.method, target_url, headers=headers, data=body,
                                           timeout=aiohttp.ClientTimeout(total=1800)) as resp:
        response = web.StreamResponse(
            status=resp.status,
            headers={"Content-Type": resp.headers.get("Content-Type", "application/octet-stream")},
        )
        await response.prepare(request)
        async for chunk in resp.content.iter_chunked(65536):
            await response.write(chunk)
        await response.write_eof()

    if request.method == "POST" and tail == "prompt":
        state.comfy_revert_task = asyncio.create_task(watch_comfy_queue_and_revert(state))

    return response


async def handle_dashboard(request):
    """Serve the dashboard HTML (see dashboard.html) — no auth required."""
    return web.Response(text=DASHBOARD_HTML, content_type="text/html")


# ---------------------------------------------------------------------------
# Dashboard HTML (served at /dashboard, no auth needed) -- kept in its own
# dashboard.html file alongside this one rather than embedded as a giant
# triple-quoted string; loaded once at import time since it never changes
# at runtime.
# ---------------------------------------------------------------------------

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")) as _f:
    DASHBOARD_HTML = _f.read()


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
    if state.comfy_revert_task is not None and not state.comfy_revert_task.done():
        state.comfy_revert_task.cancel()
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
    app.router.add_post("/models/{name}/load", handle_load_model)
    app.router.add_post("/models/{name}/evict", handle_evict_model)
    app.router.add_get("/slots/{port}", handle_slots_port)
    app.router.add_route("*", "/comfyui/{tail:.*}", handle_comfyui_proxy)
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
    parser.add_argument("--evict-drain-timeout", type=int,
                         default=int(os.environ.get("PROXY_EVICT_DRAIN_TIMEOUT_S", str(DEFAULT_EVICT_DRAIN_TIMEOUT_S))),
                         help="Max seconds an automatic scenario switch (e.g. comfy_coexist activating) "
                              "will wait for in-flight requests on the model being evicted to finish "
                              "before killing it anyway.")
    args = parser.parse_args()

    state = ProxyState()
    state.models_dir = args.models_dir
    state.gpu_index = args.gpu_index
    state.headroom_gb = args.headroom_gb
    state.load_timeout_s = args.load_timeout
    state.max_wait_s = args.max_wait
    state.evict_drain_timeout_s = args.evict_drain_timeout
    state.options, state.registry, state.standalone, state.scenarios, state.image_cfg = load_config(args.config_dir)
    state.label_index = build_label_index(state.scenarios, state.standalone, state.options)

    proxy_api_key = read_secret_file("PROXY_API_KEY")
    app = build_app(state, proxy_api_key)
    web.run_app(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()