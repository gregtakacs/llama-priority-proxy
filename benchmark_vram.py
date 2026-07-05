#!/usr/bin/env python3
"""
benchmark_vram.py — Empirically measure llama.cpp (llama-server) VRAM footprint
per model and solve for the maximum context length(s) that fit a VRAM budget.

Why not brute-force every context size (like the Ollama measure_models.py does)?
Reloading a 35B model over and over to bisect a working ctx-size is slow. Instead
this tool measures VRAM at TWO context sizes per model, fits a line:

    vram_bytes(ctx) = base_vram_bytes + bytes_per_ctx_token * ctx

(KV cache scales linearly with context length; base_vram_bytes covers model
weights + compute buffers at the tested --parallel value) and then solves that
line analytically for whatever budget you give it — including "solve" scenarios
where several models must fit in VRAM at the same time, e.g.:

  Scenario 1 (chat): one MoE model, --parallel 4, shared (--kv-unified) context.
    How large can that shared context be in 32GB?

  Scenario 2 (coding): 3 models resident together (main coding model, a small
    autocomplete model, an embedding model). Two of them run at a small fixed
    ctx; how much of the remaining VRAM can the main model's context use?

Run this script itself on your HOST — plain `python3`, no Docker wrapper needed.
It launches/tears down llama-server itself, one of two ways (--backend):
  "docker" (default): spins up a throwaway --gpus all container per trial from
    --image, bind-mounting --models-dir; no manual `docker run`/mounts to
    remember, just point it at your models dir and image name.
  "native": llama-server is already on PATH (bare metal, or you're running
    this script inside a container that already has it) — execs it directly.
Either way, nvidia-smi (also run on the host) resolves the same host-visible
PID for VRAM measurement, so the rest of the tool doesn't care which backend
launched the process.

Usage:
    # 0. Recommended first step: enumerate what's in your models dir (reads each
    #    GGUF's own header for architecture/params/native context length, no
    #    llama-server launch required) AND write/refresh config/model_options.json
    #    with a default {"parallel": 1} entry per model. Edit that file afterward
    #    to bump parallel for whichever models need it (see its own _comment).
    python3 benchmark_vram.py inspect --models-dir ~/docker/appdata/llm-models \\
        --write-options config/model_options.json

    # 1. Benchmark actual VRAM usage. Models are auto-discovered from *.gguf
    #    files in --models-dir (name/path/max_ctx all inferred from the file
    #    itself — see discover_models()); --options applies whatever you edited
    #    into model_options.json above.
    python3 benchmark_vram.py bench --models-dir ~/docker/appdata/llm-models \\
        --image llama-cpp-priority-proxy \\
        --options config/model_options.json --output config/model_vram_registry.json

    # 2. Solve a single-model scenario (always uses the entry's own
    #    parallel_tested — want a different parallel? Change it in
    #    model_options.json and re-run bench; config drift auto re-measures):
    python3 benchmark_vram.py solve --registry config/model_vram_registry.json \\
        --budget-gb 31 --model Qwen3.6-35B-A3B-UD-Q4_K_XL

    # 3. Solve a multi-model concurrent scenario from a file:
    python3 benchmark_vram.py solve --registry config/model_vram_registry.json \\
        --budget-gb 31 --scenario config/scenario_coding.json

Assumptions:
  - Single GPU (index 0 by default; override with --gpu-index).
  - Docker backend: `docker` CLI works without sudo for the invoking user, and
    the image has --gpus/nvidia-container-toolkit support already (see
    Dockerfile.llama-cpp-priority-proxy). Native backend: `llama-server` is on
    PATH (override with LLAMA_SERVER_BIN env var).
  - Nothing else is using the GPU while benchmarking runs.
"""

import argparse
import json
import os
import re
import struct
import subprocess
import time

from llama_process import (
    format_bytes, gpu_total_bytes, gpu_used_bytes,
    handle_pid, launch_server, max_ctx_for_budget, predicted_vram,
    print_log_tail, shutdown_server, wait_for_health,
)

# Fallback ctx sizes to retry with (largest first) if a sample point OOMs.
CTX_RETRY_FALLBACKS = [131072, 65536, 32768, 16384, 8192, 4096, 2048, 1024]

DEFAULT_HEADROOM_GB = 1.0  # reserved for driver/desktop/other overhead

# Fallback used only when a GGUF has no discoverable context_length metadata.
DEFAULT_MAX_CTX = 32768


# ---------------------------------------------------------------------------
# GGUF metadata reading (pure stdlib — just the header/kv/tensor-info section,
# never the multi-GB tensor data) so models are self-describing: no more
# hand-typed max_ctx guesses or path/name typos in a config file.
# ---------------------------------------------------------------------------

_GGUF_FIXED_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_GGUF_STRUCT_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                     6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}
_GGUF_STRING, _GGUF_ARRAY = 8, 9


def _gguf_read_string(f):
    (length,) = struct.unpack("<Q", f.read(8))
    return f.read(length).decode("utf-8", errors="replace")


def _gguf_read_value(f, value_type):
    if value_type == _GGUF_STRING:
        return _gguf_read_string(f)
    if value_type == _GGUF_ARRAY:
        (elem_type,) = struct.unpack("<I", f.read(4))
        (length,) = struct.unpack("<Q", f.read(8))
        return [_gguf_read_value(f, elem_type) for _ in range(length)]
    size = _GGUF_FIXED_SIZES.get(value_type)
    if size is None:
        raise ValueError(f"unsupported GGUF value type {value_type}")
    return struct.unpack(_GGUF_STRUCT_FMT[value_type], f.read(size))[0]


def read_gguf_info(path):
    """Read just the GGUF header/metadata/tensor-info (not tensor data) to get
    the model's declared architecture, native context length, and true
    parameter count (summed from tensor dims, so quantization packing can't
    make it look smaller — see the NVFP4 'phantom param count' confusion)."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"not a GGUF file: {path}")
        (_version,) = struct.unpack("<I", f.read(4))
        (tensor_count,) = struct.unpack("<Q", f.read(8))
        (kv_count,) = struct.unpack("<Q", f.read(8))

        metadata = {}
        for _ in range(kv_count):
            key = _gguf_read_string(f)
            (value_type,) = struct.unpack("<I", f.read(4))
            metadata[key] = _gguf_read_value(f, value_type)

        total_params = 0
        for _ in range(tensor_count):
            _name = _gguf_read_string(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
            f.read(4)   # ggml tensor dtype — unused
            f.read(8)   # offset — unused
            n = 1
            for d in dims:
                n *= d
            total_params += n

    arch = metadata.get("general.architecture", "unknown")
    ctx_len = metadata.get(f"{arch}.context_length")
    pooling_type = metadata.get(f"{arch}.pooling_type")
    is_embedding = (
        pooling_type is not None
        or "embed" in os.path.basename(path).lower()
        or "bert" in arch.lower()
    )
    return {
        "architecture": arch,
        "context_length": int(ctx_len) if ctx_len is not None else None,
        "n_params": total_params,
        "file_size_bytes": os.path.getsize(path),
        "is_embedding": is_embedding,
    }


_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def _is_mmproj_file(fname):
    """mmproj-*.gguf (vision projector) files are real GGUFs — read_gguf_info
    parses them fine — but they're a companion to another model's --mmproj
    flag (see config/model_options.json's extra_args), not a launchable model
    in their own right. llama.cpp's own ecosystem convention is consistently
    to put "mmproj" somewhere in the filename (mmproj-F16.gguf, mmproj-BF16.gguf,
    <model>-mmproj-f16.gguf, ...), so filter on that rather than trying to
    infer it from GGUF metadata."""
    return "mmproj" in fname.lower()


def discovered_model_names(models_dir):
    """Just the name-deriving pass of discover_models(), for building/refreshing
    a model_options.json without needing to read every GGUF's metadata twice."""
    names = []
    for fname in sorted(os.listdir(models_dir)):
        if not fname.lower().endswith(".gguf"):
            continue
        if _is_mmproj_file(fname):
            continue
        shard_match = _SHARD_RE.search(fname)
        if shard_match and shard_match.group(1) != "00001":
            continue
        names.append(fname[: shard_match.start()] if shard_match else fname[: -len(".gguf")])
    return names


def discover_models(models_dir, options=None):
    """Scan models_dir for *.gguf files and build model_cfg entries automatically —
    name/path come from the filename, max_ctx from the file's own GGUF metadata.
    Multi-part shards (name-00001-of-00003.gguf) are collapsed to their first part.
    `options` (keyed by discovered name, see config/model_options.json) can set
    parallel/min_ctx/max_ctx/extra_args/n_gpu_layers for the handful of things that
    aren't inherent to the file itself (e.g. how many parallel slots you intend to
    run a given model with)."""
    options = options or {}
    configs = []
    for fname in sorted(os.listdir(models_dir)):
        if not fname.lower().endswith(".gguf"):
            continue
        if _is_mmproj_file(fname):
            continue  # vision projector, not a launchable model — see _is_mmproj_file
        shard_match = _SHARD_RE.search(fname)
        if shard_match and shard_match.group(1) != "00001":
            continue  # only the first shard is a load target; rest follow automatically
        name = fname[: shard_match.start()] if shard_match else fname[: -len(".gguf")]
        path = os.path.join(models_dir, fname)

        try:
            info = read_gguf_info(path)
        except (OSError, ValueError, struct.error) as e:
            print(f"[warn] skipping '{fname}': could not read GGUF metadata ({e})")
            continue

        opt = options.get(name, {})
        max_ctx = opt.get("max_ctx") or info["context_length"] or DEFAULT_MAX_CTX
        if info["context_length"] is None:
            print(f"[warn] '{name}': no context_length in GGUF metadata — "
                  f"falling back to {DEFAULT_MAX_CTX:,}. Set 'max_ctx' in model_options.json if known.")

        cfg = {
            "name": name,
            "path": path,
            "parallel": opt.get("parallel", 1),
            "max_ctx": max_ctx,
            "architecture": info["architecture"],
            "n_params": info["n_params"],
        }
        if "min_ctx" in opt:
            cfg["min_ctx"] = opt["min_ctx"]
        if "n_gpu_layers" in opt:
            cfg["n_gpu_layers"] = opt["n_gpu_layers"]
        if info["is_embedding"]:
            cfg["extra_args"] = opt.get("extra_args", ["--embedding"])
        elif "extra_args" in opt:
            cfg["extra_args"] = opt["extra_args"]
        configs.append(cfg)
    return configs


# ---------------------------------------------------------------------------
# GPU helpers (benchmarking-specific; format_bytes/gpu_total_bytes/gpu_used_bytes
# and all llama-server process management now live in llama_process.py)
# ---------------------------------------------------------------------------

def process_vram_bytes(pid, gpu_index=0):
    """VRAM attributed to a specific pid, or None if not found (yet)."""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits",
         f"--id={gpu_index}"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            line_pid, used_mb = int(parts[0]), float(parts[1])
        except ValueError:
            continue
        if line_pid == pid:
            return int(used_mb * 1024 * 1024)
    return None


def wait_for_vram_settle(pid, gpu_index, poll_s=2, stable_reads=2, max_wait_s=90):
    """Poll process VRAM until two consecutive reads agree within 1% (or 64MB)."""
    deadline = time.time() + max_wait_s
    last = None
    stable_count = 0
    while time.time() < deadline:
        cur = process_vram_bytes(pid, gpu_index)
        if cur is not None and last is not None:
            delta = abs(cur - last)
            if delta <= max(64 * 1024 * 1024, 0.01 * last):
                stable_count += 1
                if stable_count >= stable_reads:
                    return cur
            else:
                stable_count = 0
        last = cur
        time.sleep(poll_s)
    return last


def wait_for_baseline_clear(baseline_bytes, gpu_index, tolerance_mb=200, max_wait_s=30):
    """After killing a server, wait for GPU used memory to drop back near baseline
    before starting the next trial (driver cleanup can lag slightly)."""
    deadline = time.time() + max_wait_s
    tol = tolerance_mb * 1024 * 1024
    while time.time() < deadline:
        cur = gpu_used_bytes(gpu_index)
        if cur is not None and cur <= baseline_bytes + tol:
            return
        time.sleep(1)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def measure_point(model_cfg, ctx, port, gpu_index, load_timeout_s, backend):
    """Launch llama-server at a given ctx, measure settled VRAM, tear down.
    Returns vram_bytes on success, or None if the server OOM'd / failed to start."""
    extra_args = model_cfg.get("extra_args", [])
    n_gpu_layers = model_cfg.get("n_gpu_layers", 99)
    parallel = model_cfg["parallel"]

    print(f"    launching ctx={ctx:,} parallel={parallel} ({backend['kind']}) ...")
    pre_launch_baseline = gpu_used_bytes(gpu_index)
    handle = launch_server(model_cfg["path"], ctx, parallel, port, n_gpu_layers, extra_args, backend)
    try:
        ok, reason = wait_for_health(handle, load_timeout_s)
        if not ok:
            print(f"    ✗ failed to become healthy: {reason}")
            print_log_tail(handle)
            return None
        pid = handle_pid(handle)
        if pid is None:
            print("    ✗ could not resolve a PID for VRAM measurement")
            print_log_tail(handle)
            return None
        vram = wait_for_vram_settle(pid, gpu_index)
        if vram is None:
            print("    ✗ could not read per-process VRAM from nvidia-smi")
            print_log_tail(handle)
            return None
        print(f"    ✓ settled at {format_bytes(vram)}")
        return vram
    finally:
        shutdown_server(handle)
        if pre_launch_baseline is not None:
            wait_for_baseline_clear(pre_launch_baseline, gpu_index)


def linear_fit(points):
    """points: list of (ctx, vram_bytes). Least-squares fit -> (slope, intercept)."""
    n = len(points)
    xbar = sum(p[0] for p in points) / n
    ybar = sum(p[1] for p in points) / n
    num = sum((x - xbar) * (y - ybar) for x, y in points)
    den = sum((x - xbar) ** 2 for x, y in points)
    if den == 0:
        raise ValueError("need at least two distinct ctx sample points")
    slope = num / den
    intercept = ybar - slope * xbar
    return slope, intercept


def pick_sample_ctxs(model_cfg):
    max_ctx = model_cfg.get("max_ctx", 32768)
    min_ctx = model_cfg.get("min_ctx")
    if min_ctx is not None:
        pts = sorted(set(c for c in (min_ctx, max_ctx) if c <= max_ctx))
        if len(pts) >= 2:
            return pts
        print(f"    [warn] configured min_ctx={min_ctx} gave <2 usable points "
              f"(max_ctx={max_ctx:,}) — falling back to auto-picked points")
    # Test AT the real ceiling, not an arbitrarily smaller stand-in — otherwise
    # a large max_ctx (native or an explicit override) never actually gets
    # empirically verified, just extrapolated from smaller points via the linear
    # fit. If this large point OOMs, the fallback ladder in benchmark_model()
    # bisects downward until something loads, so this is safe to always try.
    large = max_ctx
    # Derived from max_ctx//2, not a fixed 4096 — for small-context models
    # (e.g. an embedding model with max_ctx=2048), a fixed 4096 would collapse
    # to the same value as `large` and leave only one (unfittable) point.
    small = min(4096, max(1, max_ctx // 2))
    if small >= large:
        small = max(1, large // 4)
    return sorted(set([small, large]))


def config_signature(model_cfg):
    """Snapshot of every model_cfg field that affects the measurement itself.
    Stored alongside each registry entry so `bench` can tell whether a config
    change (e.g. bumping parallel in model_options.json) means the existing
    measurement is stale, without needing --force."""
    return {
        "parallel": model_cfg["parallel"],
        "max_ctx": model_cfg.get("max_ctx"),
        "min_ctx": model_cfg.get("min_ctx"),
        "n_gpu_layers": model_cfg.get("n_gpu_layers", 99),
        "extra_args": model_cfg.get("extra_args", []),
    }


def benchmark_model(model_cfg, port, gpu_index, load_timeout_s, backend):
    name = model_cfg["name"]
    print(f"\n{'#'*60}\n# Benchmarking: {name}\n{'#'*60}")

    candidates = pick_sample_ctxs(model_cfg)
    points = []
    for ctx in candidates:
        vram = measure_point(model_cfg, ctx, port, gpu_index, load_timeout_s, backend)
        if vram is not None:
            points.append((ctx, vram))
            continue
        # OOM'd — bisect downward using the fallback ladder until we recover
        # a usable point, so a model with a too-ambitious min_ctx/max_ctx still
        # yields a fit instead of aborting outright.
        for fb in CTX_RETRY_FALLBACKS:
            if fb >= ctx:
                continue
            print(f"    retrying at fallback ctx={fb:,}")
            vram = measure_point(model_cfg, fb, port, gpu_index, load_timeout_s, backend)
            if vram is not None:
                points.append((fb, vram))
                break

    if len(points) < 2:
        print(f"  ✗ Only got {len(points)} usable point(s) — cannot fit a line. Skipping '{name}'.")
        return None

    slope, intercept = linear_fit(points)
    print(f"  ✓ fit: base={format_bytes(intercept)}  +{format_bytes(slope)}/token")
    return {
        "name": name,
        "parallel_tested": model_cfg["parallel"],
        "max_ctx": model_cfg.get("max_ctx", 131072),
        "base_vram_bytes": round(intercept),
        "bytes_per_ctx_token": slope,
        "config_signature": config_signature(model_cfg),
        "samples": [{"ctx": c, "vram_bytes": v} for c, v in points],
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Registry I/O
# ---------------------------------------------------------------------------

def resolve_config_path(path):
    """Every generated/hand-written config file in this project lives under
    config/ — if a bare filename doesn't exist as given, check config/<basename>
    before giving up, since typing the bare name out of habit is an easy slip."""
    if path and not os.path.exists(path):
        alt = os.path.join("config", os.path.basename(path))
        if os.path.exists(alt):
            print(f"[info] '{path}' not found — using '{alt}' instead")
            return alt
    return path


def load_registry(path):
    try:
        with open(path) as f:
            data = json.load(f)
        return data if "models" in data else {"models": []}
    except (FileNotFoundError, json.JSONDecodeError):
        return {"models": []}


def save_registry(data, path):
    data["models"].sort(key=lambda m: m["name"])
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved {len(data['models'])} model(s) to '{path}'")


def upsert(registry, entry):
    for i, m in enumerate(registry["models"]):
        if m["name"] == entry["name"]:
            registry["models"][i] = entry
            return
    registry["models"].append(entry)


def check_config_drift(registry, options_path):
    """Compare each registry entry's stored config_signature against what's
    CURRENTLY in model_options.json — pure JSON-to-JSON comparison, no GGUF
    reads or --models-dir needed. parallel/min_ctx/n_gpu_layers all have
    deterministic defaults purely from what is (or isn't) in the options file,
    matching discover_models()'s own fallback logic exactly. max_ctx/extra_args
    fall back to the registry's own last-known value when not explicitly set in
    options, since resolving their "native" default requires the GGUF itself —
    that class of drift (the underlying file changing without the option being
    touched) is a rarer case `bench` already catches directly on its own."""
    resolved_options = resolve_config_path(options_path)
    if not resolved_options or not os.path.exists(resolved_options):
        return
    with open(resolved_options) as f:
        options = json.load(f).get("options", {})

    drifted = []
    for entry in registry.get("models", []):
        stored_sig = entry.get("config_signature")
        opt = options.get(entry["name"])
        if stored_sig is None or opt is None:
            continue
        current_sig = {
            "parallel": opt.get("parallel", 1),
            "max_ctx": opt.get("max_ctx", stored_sig.get("max_ctx")),
            "min_ctx": opt.get("min_ctx"),
            "n_gpu_layers": opt.get("n_gpu_layers", 99),
            "extra_args": opt.get("extra_args", stored_sig.get("extra_args")),
        }
        if current_sig != stored_sig:
            drifted.append((entry["name"], stored_sig, current_sig))

    if drifted:
        print(f"[warn] {len(drifted)} model(s) are stale — model_options.json has changed "
              f"since the last `bench` run for them:")
        for name, old, new in drifted:
            print(f"  - {name}: benchmarked with {old}")
            print(f"      now configured as {new}")
        print("  Run `bench` to refresh before trusting these numbers.\n")


# ---------------------------------------------------------------------------
# Solve: max ctx for a VRAM budget
# ---------------------------------------------------------------------------

def cmd_solve_single(registry, name, budget_bytes, target_ctx_per_slot=None):
    entry = next((m for m in registry["models"] if m["name"] == name), None)
    if entry is None:
        print(f"[error] '{name}' not found in registry. Run 'bench' first.")
        return
    # Always the parallel it was actually benchmarked at — bytes_per_ctx_token/
    # base_vram_bytes are fitted for that specific compute-buffer overhead, so
    # solving against a different parallel would be internally inconsistent.
    # Want a different parallel? Change it in model_options.json and re-run
    # `bench` — config drift auto-triggers a re-measurement.
    parallel = entry["parallel_tested"]

    # Pool cap: with --kv-unified the shared pool isn't limited to a single
    # conversation's ceiling (entry["max_ctx"]) — it needs to hold `parallel`
    # concurrent conversations, each up to that ceiling. So the pool itself can
    # legitimately be sized up to parallel * max_ctx; going beyond that buys
    # nothing (no single conversation should exceed max_ctx, and there are only
    # `parallel` slots to fill).
    pool_cap = entry["max_ctx"] * parallel
    max_possible = max_ctx_for_budget(entry, budget_bytes, ctx_cap=pool_cap)

    if target_ctx_per_slot is not None:
        # The actual question: "I want every one of `parallel` concurrent users
        # guaranteed >= target_ctx_per_slot tokens — what --ctx-size do I pass?"
        # With --kv-unified you pass the FULL total, never divided down.
        if target_ctx_per_slot > entry["max_ctx"]:
            # This is the one real architectural ceiling: no single conversation
            # should exceed the model's own trained length, regardless of VRAM
            # or how many parallel slots you have.
            print(f"\n{name}: target {target_ctx_per_slot:,} tokens/slot exceeds this model's "
                  f"own max_ctx ({entry['max_ctx']:,}) — unreachable regardless of VRAM or parallel.")
            return

        requested_total = target_ctx_per_slot * parallel
        used = predicted_vram(entry, requested_total)
        print(f"\n{name}: target {target_ctx_per_slot:,} tokens/slot x parallel={parallel} "
              f"= {requested_total:,} tokens needed, budget={format_bytes(budget_bytes)}:")
        if used <= budget_bytes:
            print(f"  ✓ fits — pass --ctx-size {requested_total:,} to llama-server "
                  f"(predicted VRAM: {format_bytes(used)}, headroom: {format_bytes(budget_bytes - used)})")
            if max_possible > requested_total:
                print(f"  You could go as high as --ctx-size {max_possible:,} and still fit — "
                      f"that raises every slot's guaranteed floor to {max_possible // parallel:,} "
                      f"tokens instead of just {target_ctx_per_slot:,}"
                      f"{' (the model max — no point going higher)' if max_possible == pool_cap else ''}.")
        else:
            print(f"  ✗ does not fit — needs {format_bytes(used)}, budget is only {format_bytes(budget_bytes)}")
            print(f"  Max --ctx-size that DOES fit: {max_possible:,} tokens -> only "
                  f"{max_possible // parallel:,} tokens/slot guaranteed at parallel={parallel} "
                  f"(short of your {target_ctx_per_slot:,} target).")
            print(f"  To hit {target_ctx_per_slot:,}/slot, lower 'parallel' for this model in "
                  f"model_options.json (and re-run bench) or free up VRAM.")
        return

    ctx = max_possible
    used = predicted_vram(entry, ctx)
    print(f"\n{name} (parallel={parallel}), budget={format_bytes(budget_bytes)}:")
    print(f"  max shared ctx-size: {ctx:,} tokens  <- pass this to --ctx-size"
          f"{' (model max for ' + str(parallel) + ' slots — VRAM allows more but there is no benefit)' if ctx == pool_cap else ''}")
    if parallel > 1:
        print(f"  worst-case per-slot: {ctx // parallel:,} tokens "
              f"(guaranteed floor if all {parallel} slots are simultaneously busy — "
              f"--no-context-shift means a request fails rather than evicting another slot)")
    print(f"  predicted VRAM use:  {format_bytes(used)}")
    print(f"  headroom left:       {format_bytes(budget_bytes - used)}")


def cmd_solve_scenario(registry, scenario_path, budget_bytes):
    with open(scenario_path) as f:
        scenario = json.load(f)

    by_name = {m["name"]: m for m in registry["models"]}
    fixed_total = 0
    auto_entry = None
    resolved = []

    for item in scenario["models"]:
        entry = by_name.get(item["name"])
        if entry is None:
            print(f"[error] '{item['name']}' not found in registry. Run 'bench' first.")
            return
        if item.get("ctx") == "auto":
            if auto_entry is not None:
                print("[error] only one model may be 'auto' per scenario — "
                      "give the others a fixed 'ctx' value.")
                return
            auto_entry = (item["name"], entry)
        else:
            ctx = int(item["ctx"])
            vram = predicted_vram(entry, ctx)
            fixed_total += vram
            resolved.append((item["name"], ctx, vram))

    print(f"\nScenario '{scenario_path}', budget={format_bytes(budget_bytes)}:")
    for name, ctx, vram in resolved:
        print(f"  {name}: fixed ctx={ctx:,} -> {format_bytes(vram)}")
    print(f"  fixed-model subtotal: {format_bytes(fixed_total)}")

    if auto_entry is None:
        print(f"  remaining headroom: {format_bytes(budget_bytes - fixed_total)}")
        return

    name, entry = auto_entry
    remaining = budget_bytes - fixed_total
    parallel = entry["parallel_tested"]
    ctx = max_ctx_for_budget(entry, remaining, ctx_cap=entry["max_ctx"] * parallel)
    used = predicted_vram(entry, ctx)
    print(f"  {name}: auto -> max ctx {ctx:,} tokens ({format_bytes(used)})")
    if parallel > 1:
        print(f"    worst-case per-slot: {ctx // parallel:,} tokens "
              f"(guaranteed floor if all {parallel} slots are simultaneously busy)")
    print(f"  total predicted VRAM: {format_bytes(fixed_total + used)}")
    print(f"  headroom left:        {format_bytes(budget_bytes - fixed_total - used)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_bench_models(args):
    options = {}
    options_path = resolve_config_path(args.options)
    if options_path and os.path.exists(options_path):
        with open(options_path) as f:
            options = json.load(f).get("options", {})
    elif options_path:
        print(f"[info] '{options_path}' not found — using defaults (parallel=1) for every model. "
              f"Run `inspect --models-dir {args.models_dir} --write-options {options_path}` to generate it.")
    return discover_models(args.models_dir, options)


_BENCH_OPTIONS_COMMENT = [
    "Per-model options for `bench`, keyed by filename without .gguf.",
    "Auto-populated by `inspect --write-options`; re-running it only ADDS newly",
    "discovered models (or backfills label/keep_alive if missing) — it never",
    "touches/overwrites values you've already edited here.",
    "",
    "Fields (all optional except parallel):",
    "  parallel     (default 1)   how many concurrent slots to benchmark this model with.",
    "                             This is the one you'll most likely change: bump it to",
    "                             match how many parallel requests you actually intend to",
    "                             serve — it shifts the measured VRAM baseline and can't",
    "                             be corrected after the fact by `solve`.",
    "  min_ctx      (optional)    override the small ctx sample point used for the fit.",
    "  max_ctx      (optional)    override the detected native context ceiling; also used",
    "                             as the large sample point when min_ctx is set.",
    "  extra_args   (optional)    extra llama-server flags (auto-set to [\"--embedding\"]",
    "                             for detected embedding models).",
    "  n_gpu_layers (default 99)  GPU offload layer count.",
    "  label        (default: the model's own name) friendly nickname the future",
    "                             priority proxy will expose instead of this raw",
    "                             filename-derived name. Purely cosmetic/routing —",
    "                             does not affect benchmarking or trigger a re-bench.",
    "  keep_alive   (default: blank) how long to keep this model loaded when idle",
    "                             before eviction, Ollama-style: a duration string",
    "                             (\"5m\", \"10s\", \"66h\") or \"-1\" for never evict.",
    "                             NOT wired up yet — reserved for the proxy; changing",
    "                             it does not affect benchmarking or trigger a re-bench.",
]

# Cosmetic/proxy-facing fields — deliberately NOT part of config_signature, since
# neither affects VRAM measurement. Backfilled onto existing entries (not just new
# ones) by write_options_file() if missing, without touching anything else.
_OPTIONS_METADATA_DEFAULTS = {
    "label": None,       # None here means "use the model's own name" — see write_options_file
    "keep_alive": "",
}


def write_options_file(path, names, embedding_names=None):
    """Create or refresh model_options.json: add a default entry for any newly
    discovered model name, and backfill label/keep_alive onto EXISTING entries
    if either is missing — but never touch any value you've already set,
    including a label/keep_alive you've already customized.

    `embedding_names`: names GGUF metadata says are embedding models (see
    read_gguf_info's is_embedding) — these get extra_args: ["--embedding"]
    backfilled too if not already set, since the live proxy only ever reads
    this file (never the GGUF itself) and has no other way to know a model
    needs that flag to serve embeddings correctly."""
    embedding_names = embedding_names or set()
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        data.setdefault("options", {})
    else:
        data = {"options": {}}
    data["_comment"] = _BENCH_OPTIONS_COMMENT  # always refresh — it's generated docs, not user data

    added = [name for name in names if name not in data["options"]]
    for name in added:
        entry = {"parallel": 1, "label": name, "keep_alive": ""}
        if name in embedding_names:
            entry["extra_args"] = ["--embedding"]
        data["options"][name] = entry

    backfilled = []
    for name in names:
        if name in added:
            continue
        entry = data["options"][name]
        changed = False
        for field, default in _OPTIONS_METADATA_DEFAULTS.items():
            if field not in entry:
                entry[field] = name if field == "label" else default
                changed = True
        if "extra_args" not in entry and name in embedding_names:
            entry["extra_args"] = ["--embedding"]
            changed = True
        if changed:
            backfilled.append(name)

    data["options"] = dict(sorted(data["options"].items()))  # keep the file alphabetical, always
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return added, backfilled


def cmd_inspect(args):
    """Fast, GPU-free enumeration: read each GGUF's own metadata (architecture,
    native context length, true parameter count) without launching llama-server."""
    rows = []
    for fname in sorted(os.listdir(args.models_dir)):
        if not fname.lower().endswith(".gguf"):
            continue
        if _is_mmproj_file(fname):
            continue  # vision projector, not a launchable model — see _is_mmproj_file
        shard_match = _SHARD_RE.search(fname)
        if shard_match and shard_match.group(1) != "00001":
            continue
        path = os.path.join(args.models_dir, fname)
        try:
            info = read_gguf_info(path)
        except (OSError, ValueError, struct.error) as e:
            print(f"[warn] skipping '{fname}': could not read GGUF metadata ({e})")
            continue
        rows.append((fname, info))

    name_w = max([len(r[0]) for r in rows] + [10])
    arch_w = max([len(r[1]["architecture"]) for r in rows] + [12])
    header = f"{'file':<{name_w}}  {'architecture':<{arch_w}}  {'params':>10}  {'native ctx':>12}  {'file size':>10}  embed?"
    print(header)
    print("-" * len(header))
    for fname, info in rows:
        params = info["n_params"]
        params_s = f"{params/1e9:.1f}B" if params >= 1e9 else f"{params/1e6:.0f}M"
        ctx_s = f"{info['context_length']:,}" if info["context_length"] else "unknown"
        print(f"{fname:<{name_w}}  {info['architecture']:<{arch_w}}  {params_s:>10}  "
              f"{ctx_s:>12}  {format_bytes(info['file_size_bytes']):>10}  "
              f"{'yes' if info['is_embedding'] else ''}")

    if args.write_options:
        names = discovered_model_names(args.models_dir)
        embedding_names = set()
        for fname, info in rows:
            if not info["is_embedding"]:
                continue
            shard_match = _SHARD_RE.search(fname)
            name = fname[: shard_match.start()] if shard_match else fname[: -len(".gguf")]
            embedding_names.add(name)
        added, backfilled = write_options_file(args.write_options, names, embedding_names)
        if added:
            print(f"\nAdded {len(added)} new model(s) to '{args.write_options}' "
                  f"(parallel: 1, label: <name>, keep_alive: blank): {', '.join(added)}")
        if backfilled:
            print(f"Backfilled missing label/keep_alive on {len(backfilled)} existing "
                  f"entry/entries: {', '.join(backfilled)}")
        if not added and not backfilled:
            print(f"\n'{args.write_options}' already covers every discovered model — nothing to do.")
        print(f"Edit it (parallel/min_ctx/max_ctx/label/keep_alive/etc.) before running `bench`.")


def cmd_bench(args):
    backend = {"kind": args.backend, "gpu_index": args.gpu_index}
    if args.backend == "docker":
        backend["image"] = args.image
        backend["models_dir"] = args.models_dir

    model_cfgs = _load_bench_models(args)

    registry = load_registry(args.output)
    by_name = {m["name"]: m for m in registry["models"]}

    for model_cfg in model_cfgs:
        if args.model and model_cfg["name"] != args.model:
            continue
        existing = by_name.get(model_cfg["name"])
        if existing is not None and not args.force:
            sig = config_signature(model_cfg)
            if existing.get("config_signature") == sig:
                print(f"Skipping '{model_cfg['name']}' (already benchmarked, config unchanged)")
                continue
            print(f"Re-benchmarking '{model_cfg['name']}' — config changed since last measurement "
                  f"(was {existing.get('config_signature')}, now {sig})")
        entry = benchmark_model(model_cfg, args.port, args.gpu_index, args.load_timeout, backend)
        if entry is not None:
            upsert(registry, entry)
            save_registry(registry, args.output)  # save incrementally


def cmd_solve(args):
    registry = load_registry(resolve_config_path(args.registry))
    check_config_drift(registry, args.options)
    budget_bytes = int(args.budget_gb * (1024 ** 3))
    if args.scenario:
        cmd_solve_scenario(registry, resolve_config_path(args.scenario), budget_bytes)
    elif args.model:
        cmd_solve_single(registry, args.model, budget_bytes, args.target_ctx_per_slot)
    else:
        print("[error] 'solve' needs either --model NAME, or --scenario FILE")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench", help="Measure VRAM footprint of models via llama-server")
    b.add_argument("--models-dir", required=True, help="Auto-discover *.gguf files here (name/path/max_ctx all inferred)")
    b.add_argument("--options", default="config/model_options.json",
                   help="model_options.json — per-model options keyed by discovered name "
                        "(parallel/min_ctx/max_ctx/extra_args/n_gpu_layers). "
                        "Generate/refresh one with `inspect --write-options`. "
                        "(default: %(default)s)")
    b.add_argument("--output", default="config/model_vram_registry.json", help="Registry output path")
    b.add_argument("--model", help="Only benchmark this one model name")
    b.add_argument("--force", action="store_true", help="Re-measure even if already in registry")
    b.add_argument("--port", type=int, default=18080, help="Scratch port for the benchmark server")
    b.add_argument("--gpu-index", type=int, default=0)
    b.add_argument("--load-timeout", type=int, default=300, help="Seconds to wait for /health per trial")
    b.add_argument("--backend", choices=["docker", "native"], default="docker",
                   help="'docker' (default): this script launches/tears down a throwaway "
                        "llama-server container per trial itself — no manual `docker run` needed. "
                        "'native': llama-server is already on PATH (bare metal, or you're running "
                        "this script inside the same container that has it).")
    b.add_argument("--image", default="llama-cpp-priority-proxy",
                   help="Docker image to run llama-server from (--backend docker only)")
    b.set_defaults(func=cmd_bench)

    i = sub.add_parser("inspect", help="List *.gguf files with metadata-derived params/native context (no GPU needed)")
    i.add_argument("--models-dir", required=True)
    i.add_argument("--write-options", default="config/model_options.json",
                   help="Create/refresh this model_options.json with a default {parallel: 1} entry "
                        "for every discovered model — the recommended first step before `bench`. "
                        "Existing entries are left untouched. (default: %(default)s)")
    i.set_defaults(func=cmd_inspect)

    s = sub.add_parser("solve", help="Solve max ctx-size for a VRAM budget")
    s.add_argument("--registry", default="config/model_vram_registry.json")
    s.add_argument("--budget-gb", type=float, help="VRAM budget in GB (default: detected total - headroom)")
    s.add_argument("--headroom-gb", type=float, default=DEFAULT_HEADROOM_GB)
    s.add_argument("--gpu-index", type=int, default=0)
    s.add_argument("--model", help="Single-model mode: registry entry name")
    s.add_argument("--target-ctx-per-slot", type=int,
                   help="Single-model mode: 'I want every one of this model's parallel_tested "
                        "concurrent users guaranteed this many tokens.' Reports the exact "
                        "--ctx-size to pass llama-server (target * parallel_tested) and whether "
                        "it fits the budget — the actual question you're usually asking, vs. the "
                        "default 'maximize total ctx-size' behavior when this is omitted.")
    s.add_argument("--scenario", help="Multi-model scenario JSON (see config/scenario_coding.json)")
    s.add_argument("--options", default="config/model_options.json",
                   help="Checked against the registry's stored config_signature on every run — "
                        "warns if model_options.json has changed since a model was last "
                        "benchmarked (default: %(default)s). Pass --options '' to skip this check.")
    s.set_defaults(func=cmd_solve)

    args = parser.parse_args()
    if args.cmd == "solve" and args.budget_gb is None:
        total = gpu_total_bytes(args.gpu_index)
        args.budget_gb = (total - args.headroom_gb * (1024 ** 3)) / (1024 ** 3)
        print(f"[info] auto budget: {args.budget_gb:.2f} GB (detected {format_bytes(total)} - "
              f"{args.headroom_gb} GB headroom)")

    args.func(args)


if __name__ == "__main__":
    main()
