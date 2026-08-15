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

    # 4. Ground truth: actually LAUNCH the whole scenario (every model solve
    #    predicted, all resident together, exactly like the live proxy would)
    #    and hit the primary with a real prompt. `bench`'s single-model 2-point
    #    fit can be several GB off run-to-run on a noisy/shared GPU (confirmed
    #    in practice) -- solve's predictions are only as good as that fit, so
    #    "solve says it fits" is not the same as "it actually works." This is
    #    the only way to know for sure a scenario both fits AND stays
    #    performant (i.e. doesn't spill into system/shared memory) under real
    #    load, not just on paper:
    python3 benchmark_vram.py validate --models-dir ~/docker/appdata/llm-models \\
        --image llama-cpp-priority-proxy --scenario config/scenario_coding.json

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
    print_log_tail, send_completion_prompt, send_warmup_prompt, shutdown_server, wait_for_health,
)

# Fallback ctx sizes to retry with (largest first) if a sample point OOMs.
CTX_RETRY_FALLBACKS = [131072, 65536, 32768, 16384, 8192, 4096, 2048, 1024]

DEFAULT_HEADROOM_GB = 1.0  # reserved for driver/desktop/other overhead

# Default size of the synthetic /completion warmup prompt each measure_point()
# trial sends before reading "settled" VRAM (see send_warmup_prompt in
# llama_process.py). Empirically motivated: a model that measured as fitting
# via the OLD health-check-only methodology (near-zero predicted headroom,
# but nominally fitting) turned out to spill several GB into shared/system
# memory the moment a real ~2,823-token prompt hit it, tanking prefill from
# 1000+ tok/s to ~34 tok/s. 3072 comfortably exceeds that reproduction case.
DEFAULT_WARMUP_PROMPT_TOKENS = 3072

# Anchored to this script's own directory rather than left as bare relative
# paths — otherwise `--write-options`'s default silently resolves against
# whatever the shell's cwd happens to be, and a script that's supposed to be a
# safe no-op on an existing config instead creates a fresh, defaults-only file
# there (this is what wiped out hand-edited sampling params once already).
_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")


def _default_config_path(filename):
    return os.path.join(_CONFIG_DIR, filename)

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


def _is_mmproj_info(info):
    """mmproj-*.gguf (vision projector) files are real, readable GGUFs — but a
    companion to some OTHER model's --mmproj flag (see config/model_options.json's
    extra_args), not a launchable model in their own right. llama.cpp itself
    writes general.architecture = "clip" into every projector it produces —
    confirmed against a real mmproj-F16.gguf (architecture 'clip', context_length
    None, ~447M "params" that are really vision-tower weights). Decided purely
    from GGUF metadata — no filename heuristic."""
    return info["architecture"].lower() == "clip"


def discovered_model_names(models_dir):
    """Just the name-deriving pass of discover_models(), for building/refreshing
    a model_options.json. Still reads each file's header (read_gguf_info doesn't
    touch tensor data, so this is cheap even for huge models) — needed to filter
    out mmproj files by architecture, not just by name."""
    names = []
    for fname in sorted(os.listdir(models_dir)):
        if not fname.lower().endswith(".gguf"):
            continue
        shard_match = _SHARD_RE.search(fname)
        if shard_match and shard_match.group(1) != "00001":
            continue
        path = os.path.join(models_dir, fname)
        try:
            info = read_gguf_info(path)
        except (OSError, ValueError, struct.error) as e:
            print(f"[warn] skipping '{fname}': could not read GGUF metadata ({e})")
            continue
        if _is_mmproj_info(info):
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
        if _is_mmproj_info(info):
            continue  # vision projector, not a launchable model — see _is_mmproj_info

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


def wait_for_aggregate_vram_settle(pre_launch_baseline, gpu_index, poll_s=2, stable_reads=2, max_wait_s=90):
    """Fallback for wait_for_vram_settle when nvidia-smi's per-process
    accounting isn't available (observed under WSL2: --query-compute-apps
    and the 'Processes' table are always empty there, even with a live
    --gpus all container running, while aggregate --query-gpu=memory.used
    works fine). Polls aggregate GPU memory.used until it settles, then
    attributes the delta over pre_launch_baseline to this trial -- accurate
    as long as nothing else is using the GPU concurrently, same assumption
    wait_for_baseline_clear below already relies on between trials."""
    deadline = time.time() + max_wait_s
    last = None
    stable_count = 0
    while time.time() < deadline:
        cur = gpu_used_bytes(gpu_index)
        if cur is not None and last is not None:
            delta = abs(cur - last)
            if delta <= max(64 * 1024 * 1024, 0.01 * last):
                stable_count += 1
                if stable_count >= stable_reads:
                    return cur - pre_launch_baseline if pre_launch_baseline is not None else cur
            else:
                stable_count = 0
        last = cur
        time.sleep(poll_s)
    if last is None or pre_launch_baseline is None:
        return None
    return last - pre_launch_baseline


def wait_for_baseline_clear(baseline_bytes, gpu_index, tolerance_mb=200, max_wait_s=120):
    """After killing a server, wait for GPU used memory to drop back near baseline
    before starting the next trial (driver cleanup can lag slightly).

    Returns True once cleared, False on timeout. The timeout case matters more
    than it looks: if this gives up early and the NEXT trial's pre_launch_baseline
    is captured while stale VRAM from THIS trial hasn't actually been released
    yet, and the driver then finishes releasing it mid-way through the next
    trial's own (now much longer, since bench sends a real warmup prompt)
    measurement window, the next trial's aggregate-diff reading silently
    UNDERCOUNTS its own usage by whatever amount got freed during that window —
    caught in practice as a Q3_K_XL entry with an impossibly low ~2.9GB base for
    a 13GB+ weight file, once warmup made trials long enough to expose it. Print
    a warning on timeout instead of failing silently, so a bad reading is at
    least visible instead of quietly poisoning the fit."""
    deadline = time.time() + max_wait_s
    tol = tolerance_mb * 1024 * 1024
    while time.time() < deadline:
        cur = gpu_used_bytes(gpu_index)
        if cur is not None and cur <= baseline_bytes + tol:
            return True
        time.sleep(1)
    cur = gpu_used_bytes(gpu_index)
    print(f"    [warn] GPU memory didn't clear back to baseline within {max_wait_s}s "
          f"(baseline={format_bytes(baseline_bytes)}, still at {format_bytes(cur) if cur is not None else 'unknown'}) "
          f"-- next trial's measurement may be corrupted by this trial's not-yet-released VRAM")
    return False


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

# How close a "settled" reading can get to the card's total physical VRAM
# before it's flagged as untrustworthy -- see _warn_if_near_ceiling's
# docstring for why a near-ceiling reading is actively misleading, not just
# imprecise.
NEAR_CEILING_MARGIN_FRAC = 0.05


def _warn_if_near_ceiling(vram_bytes, gpu_index):
    """A 'settled' VRAM reading close to the card's total physical capacity
    isn't a real measurement of what the model needs -- it's the ceiling
    nvidia-smi is physically capable of reporting. `nvidia-smi`'s
    memory.used only ever reports DEDICATED VRAM; on Windows/WSL2, when a
    config's true requirement exceeds the physical card, the driver
    silently backs the overflow with system RAM (WDDM's "shared GPU
    memory") instead of erroring -- invisible to nvidia-smi entirely, and
    catastrophically slower (confirmed: ~34 tok/s prefill vs. 1000+ normal).

    Caught concretely in this project's own history: at ctx=262,144, bench
    repeatedly measured ~23.2GB (suspiciously close to this card's ~23.9GB
    total) while every OTHER sample point at smaller ctx agreed tightly on a
    consistent 65.00 KB/token slope that, extrapolated to 262,144, predicts
    ~29-30GB -- a 6-7GB gap matching almost exactly the shared-memory
    spillover Windows Task Manager reported when a config fit from that
    262,144 sample was trusted and failed live. The 262,144 reading wasn't
    noise; it was truncated at the physical ceiling with no signal that
    anything was wrong.

    Returns True if `vram_bytes` is within NEAR_CEILING_MARGIN_FRAC of the
    card's total (and prints a loud warning); False otherwise."""
    total = gpu_total_bytes(gpu_index)
    if total is None:
        return False
    threshold = total * (1 - NEAR_CEILING_MARGIN_FRAC)
    if vram_bytes < threshold:
        return False
    print(f"    [WARN] settled reading {format_bytes(vram_bytes)} is within "
          f"{NEAR_CEILING_MARGIN_FRAC*100:.0f}% of this card's total VRAM ({format_bytes(total)}) "
          f"-- likely TRUNCATED, not a real measurement. The true requirement may be several GB "
          f"higher, silently spilling into system-backed shared GPU memory (invisible to "
          f"nvidia-smi, catastrophically slow). Do not trust this sample point; prefer a smaller "
          f"max_ctx that measures comfortably below the physical ceiling.")
    return True


def measure_point(model_cfg, ctx, port, gpu_index, load_timeout_s, backend, warmup_prompt_tokens=0):
    """Launch llama-server at a given ctx, measure settled VRAM, tear down.
    Returns (vram_bytes, near_ceiling: bool) on success, or (None, False) if
    the server OOM'd / failed to start. near_ceiling True means don't trust
    vram_bytes as the model's true requirement -- see _warn_if_near_ceiling."""
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
            return None, False
        pid = handle_pid(handle)
        if pid is None:
            print("    ✗ could not resolve a PID for VRAM measurement")
            print_log_tail(handle)
            return None, False
        if warmup_prompt_tokens > 0:
            print(f"    sending ~{warmup_prompt_tokens:,}-token warmup prompt "
                  f"(exercises real compute buffers, not just llama.cpp's own tiny startup warmup) ...")
            ok, reason = send_warmup_prompt(handle, warmup_prompt_tokens, ctx=ctx)
            if not ok:
                print(f"    [warn] warmup prompt failed ({reason}) -- measuring VRAM from "
                      f"bare health-check state instead, result may understate real usage")
        vram = wait_for_vram_settle(pid, gpu_index)
        if vram is None:
            # Per-process accounting unavailable (e.g. WSL2 -- see
            # wait_for_aggregate_vram_settle's docstring) -- fall back to
            # aggregate-usage diffing against the pre-launch baseline.
            vram = wait_for_aggregate_vram_settle(pre_launch_baseline, gpu_index)
            if vram is None:
                print("    ✗ could not read per-process OR aggregate VRAM from nvidia-smi")
                print_log_tail(handle)
                return None, False
            print(f"    ✓ settled at {format_bytes(vram)} (aggregate-diff fallback)")
            return vram, _warn_if_near_ceiling(vram, gpu_index)
        print(f"    ✓ settled at {format_bytes(vram)}")
        return vram, _warn_if_near_ceiling(vram, gpu_index)
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


def benchmark_model(model_cfg, port, gpu_index, load_timeout_s, backend, warmup_prompt_tokens=0):
    name = model_cfg["name"]
    print(f"\n{'#'*60}\n# Benchmarking: {name}\n{'#'*60}")

    candidates = pick_sample_ctxs(model_cfg)
    points = []  # (ctx, vram_bytes, near_ceiling)
    for ctx in candidates:
        vram, near_ceiling = measure_point(model_cfg, ctx, port, gpu_index, load_timeout_s, backend, warmup_prompt_tokens)
        if vram is not None:
            points.append((ctx, vram, near_ceiling))
            continue
        # OOM'd — bisect downward using the fallback ladder until we recover
        # a usable point, so a model with a too-ambitious min_ctx/max_ctx still
        # yields a fit instead of aborting outright.
        for fb in CTX_RETRY_FALLBACKS:
            if fb >= ctx:
                continue
            print(f"    retrying at fallback ctx={fb:,}")
            vram, near_ceiling = measure_point(model_cfg, fb, port, gpu_index, load_timeout_s, backend, warmup_prompt_tokens)
            if vram is not None:
                points.append((fb, vram, near_ceiling))
                break

    if len(points) < 2:
        print(f"  ✗ Only got {len(points)} usable point(s) — cannot fit a line. Skipping '{name}'.")
        return None

    slope, intercept = linear_fit([(ctx, vram) for ctx, vram, _ in points])
    any_near_ceiling = any(near_ceiling for _, _, near_ceiling in points)
    print(f"  ✓ fit: base={format_bytes(intercept)}  +{format_bytes(slope)}/token"
          f"{'  [UNRELIABLE -- see near-ceiling warning above]' if any_near_ceiling else ''}")
    return {
        "name": name,
        "parallel_tested": model_cfg["parallel"],
        "max_ctx": model_cfg.get("max_ctx", 131072),
        "base_vram_bytes": round(intercept),
        "bytes_per_ctx_token": slope,
        "config_signature": config_signature(model_cfg),
        "warmup_prompt_tokens": warmup_prompt_tokens,
        "measurement_may_be_capped": any_near_ceiling,
        "samples": [{"ctx": c, "vram_bytes": v, "near_ceiling": nc} for c, v, nc in points],
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
        data = data if "models" in data else {"models": []}
    except (FileNotFoundError, json.JSONDecodeError):
        return {"models": []}
    capped = [m["name"] for m in data["models"] if m.get("measurement_may_be_capped")]
    if capped:
        print(f"[WARN] registry has {len(capped)} model(s) whose measurement may be truncated by "
              f"the physical VRAM ceiling (see _warn_if_near_ceiling) -- don't trust these for "
              f"'auto' sizing without re-benching at a smaller max_ctx: {', '.join(capped)}")
    return data


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


def resolve_scenario_sizes(registry, scenario, budget_bytes):
    """Shared by cmd_solve_scenario (prints a report) and cmd_validate (actually
    launches the result and checks it live). Returns (resolved, auto_result, error):
      resolved    -- list of (name, ctx, vram_bytes) for every FIXED-ctx model
      auto_result -- (name, ctx, vram_bytes, parallel) for the one 'auto' model,
                      or None if the scenario has no 'auto' model
      error       -- error message string, or None on success (resolved/auto_result
                      are meaningless if error is set)
    """
    by_name = {m["name"]: m for m in registry["models"]}
    fixed_total = 0
    auto_entry = None
    resolved = []

    for item in scenario["models"]:
        entry = by_name.get(item["name"])
        if entry is None:
            return None, None, f"'{item['name']}' not found in registry. Run 'bench' first."
        if item.get("ctx") == "auto":
            if auto_entry is not None:
                return None, None, "only one model may be 'auto' per scenario — give the others a fixed 'ctx' value."
            auto_entry = (item["name"], entry)
        else:
            ctx = int(item["ctx"])
            vram = predicted_vram(entry, ctx)
            fixed_total += vram
            resolved.append((item["name"], ctx, vram))

    if auto_entry is None:
        return resolved, None, None

    name, entry = auto_entry
    remaining = budget_bytes - fixed_total
    parallel = entry["parallel_tested"]
    ctx = max_ctx_for_budget(entry, remaining, ctx_cap=entry["max_ctx"] * parallel)
    used = predicted_vram(entry, ctx)
    return resolved, (name, ctx, used, parallel), None


def cmd_solve_scenario(registry, scenario_path, budget_bytes):
    with open(scenario_path) as f:
        scenario = json.load(f)

    resolved, auto_result, error = resolve_scenario_sizes(registry, scenario, budget_bytes)
    if error:
        print(f"[error] {error}")
        return

    fixed_total = sum(vram for _, _, vram in resolved)
    print(f"\nScenario '{scenario_path}', budget={format_bytes(budget_bytes)}:")
    for name, ctx, vram in resolved:
        print(f"  {name}: fixed ctx={ctx:,} -> {format_bytes(vram)}")
    print(f"  fixed-model subtotal: {format_bytes(fixed_total)}")

    if auto_result is None:
        print(f"  remaining headroom: {format_bytes(budget_bytes - fixed_total)}")
        return

    name, ctx, used, parallel = auto_result
    print(f"  {name}: auto -> max ctx {ctx:,} tokens ({format_bytes(used)})")
    if parallel > 1:
        print(f"    worst-case per-slot: {ctx // parallel:,} tokens "
              f"(guaranteed floor if all {parallel} slots are simultaneously busy)")
    print(f"  total predicted VRAM: {format_bytes(fixed_total + used)}")
    print(f"  headroom left:        {format_bytes(budget_bytes - fixed_total - used)}")


# ---------------------------------------------------------------------------
# Scenario validation -- actually load the whole scenario (every model
# `solve` predicted, all resident together, exactly like the live proxy
# would) and hit it with a real prompt. Motivated directly by two real
# failures where a single-model bench/solve prediction ("this fits, headroom
# X") did not survive contact with an actual multi-thousand-token request --
# solve's math is only ever as good as the two-point fit behind it, and this
# is the only way to get ground truth instead of another prediction.
# ---------------------------------------------------------------------------

def _model_cfg_for(name, model_cfgs_by_name):
    cfg = model_cfgs_by_name.get(name)
    if cfg is None:
        raise KeyError(name)
    return cfg


def cmd_validate(args):
    backend = {"kind": args.backend, "gpu_index": args.gpu_index}
    if args.backend == "docker":
        backend["image"] = args.image
        backend["models_dir"] = args.models_dir

    scenario_path = resolve_config_path(args.scenario)
    with open(scenario_path) as f:
        scenario = json.load(f)

    registry = load_registry(resolve_config_path(args.registry))
    check_config_drift(registry, args.options)

    standalone_path = resolve_config_path(args.standalone)
    standalone = []
    if standalone_path and os.path.exists(standalone_path):
        with open(standalone_path) as f:
            standalone = json.load(f).get("models", [])

    if args.budget_gb is None:
        total = gpu_total_bytes(args.gpu_index)
        args.budget_gb = (total - args.headroom_gb * (1024 ** 3)) / (1024 ** 3)
        print(f"[info] auto budget: {args.budget_gb:.2f} GB (detected {format_bytes(total)} - "
              f"{args.headroom_gb} GB headroom)")
    budget_bytes = int(args.budget_gb * (1024 ** 3))

    # Standalone models are always-resident in the real proxy and DO eat into
    # the same budget the scenario's own 'auto' sizing assumes -- see
    # llama_priority_proxy.py's own budget calc. Subtract them here too so
    # this test's resolved ctx matches what the live proxy would actually pick.
    by_name = {m["name"]: m for m in registry["models"]}
    standalone_vram = 0
    for m in standalone:
        entry = by_name.get(m["name"])
        if entry is None:
            print(f"[error] standalone model '{m['name']}' not found in registry. Run 'bench' first.")
            return
        standalone_vram += predicted_vram(entry, int(m["ctx"]))
    scenario_budget = budget_bytes - standalone_vram

    resolved, auto_result, error = resolve_scenario_sizes(registry, scenario, scenario_budget)
    if error:
        print(f"[error] {error}")
        return

    plan = list(resolved)  # (name, ctx, vram)
    primary_name = None
    for item in scenario["models"]:
        if item.get("slot") == "primary":
            primary_name = item["name"]
    if auto_result is not None:
        name, ctx, used, _parallel = auto_result
        plan.append((name, ctx, used))
        if primary_name is None:
            primary_name = name  # the 'auto' model is virtually always the primary
    if primary_name is None:
        primary_name = plan[0][0]  # last resort: just pick the first model

    print(f"\nValidating scenario '{scenario_path}' live (budget={format_bytes(budget_bytes)}, "
          f"standalone subtotal={format_bytes(standalone_vram)}, primary='{primary_name}'):")
    for name, ctx, vram in plan:
        marker = " [primary]" if name == primary_name else ""
        print(f"  {name}: ctx={ctx:,} (predicted {format_bytes(vram)}){marker}")
    for m in standalone:
        print(f"  {m['name']}: ctx={int(m['ctx']):,} (standalone, predicted "
              f"{format_bytes(predicted_vram(by_name[m['name']], int(m['ctx'])))})")

    model_cfgs_by_name = {cfg["name"]: cfg for cfg in _load_bench_models(args)}
    to_launch = [(name, ctx) for name, ctx, _ in plan] + [(m["name"], int(m["ctx"])) for m in standalone]

    handles = []
    port = args.port_base
    pre_launch_baseline = gpu_used_bytes(args.gpu_index)
    try:
        for name, ctx in to_launch:
            try:
                cfg = _model_cfg_for(name, model_cfgs_by_name)
            except KeyError:
                print(f"  ✗ '{name}' not found under --models-dir {args.models_dir} -- aborting")
                return
            print(f"  launching '{name}' ctx={ctx:,} on scratch port {port} ({backend['kind']}) ...")
            handle = launch_server(cfg["path"], ctx, cfg.get("parallel", 1), port,
                                    cfg.get("n_gpu_layers", 99), cfg.get("extra_args", []), backend)
            handles.append((name, handle))
            port += 1

        for name, handle in handles:
            ok, reason = wait_for_health(handle, args.load_timeout)
            if not ok:
                print(f"  ✗ FAIL: '{name}' never became healthy: {reason}")
                print_log_tail(handle)
                return
        print("  all models healthy — sending real test prompt to primary ...")

        primary_handle = next(h for name, h in handles if name == primary_name)
        ok, result = send_completion_prompt(primary_handle, args.warmup_prompt_tokens, timeout_s=args.request_timeout)
        if not ok:
            print(f"  ✗ FAIL: test prompt to '{primary_name}' did not complete: {result}")
            print(f"    (this is the same failure mode as a real client hitting a scenario that "
                  f"looks fine on paper but hasn't actually been load-tested)")
            return

        vram_with_load = gpu_used_bytes(args.gpu_index)
        total_bytes = gpu_total_bytes(args.gpu_index)
        free_bytes = (total_bytes - vram_with_load) if (total_bytes and vram_with_load is not None) else None

        timings = (result or {}).get("timings", {})
        prompt_tps = timings.get("prompt_per_second")
        predicted_tps = timings.get("predicted_per_second")

        print(f"\n  Result:")
        print(f"    prefill:    {prompt_tps:.1f} tok/s" if prompt_tps is not None else "    prefill:    unknown")
        print(f"    decode:     {predicted_tps:.1f} tok/s" if predicted_tps is not None else "    decode:     unknown")
        print(f"    live VRAM used: {format_bytes(vram_with_load)}" if vram_with_load is not None else "    live VRAM used: unknown")
        print(f"    live VRAM free: {format_bytes(free_bytes)}" if free_bytes is not None else "    live VRAM free: unknown")

        fails = []
        if prompt_tps is None or prompt_tps < args.min_prefill_tps:
            fails.append(f"prefill {prompt_tps if prompt_tps is not None else '?'} tok/s "
                         f"< required {args.min_prefill_tps} tok/s")
        min_free_bytes = args.min_free_mb * 1024 * 1024
        if free_bytes is None or free_bytes < min_free_bytes:
            fails.append(f"free VRAM {format_bytes(free_bytes) if free_bytes is not None else '?'} "
                         f"< required {args.min_free_mb} MB")

        if fails:
            print(f"\n  ✗ FAIL — scenario fits on paper but is not performant/safe under real load:")
            for f in fails:
                print(f"    - {f}")
            print(f"    This means the config would likely reproduce shared-memory spillover / "
                  f"drastic slowdown under real traffic -- lower max_ctx and re-validate.")
        else:
            print(f"\n  ✓ PASS — fits and performs well under a real "
                  f"~{args.warmup_prompt_tokens:,}-token request.")
    finally:
        print("\n  tearing down ...")
        for _name, handle in handles:
            shutdown_server(handle)
        if pre_launch_baseline is not None:
            wait_for_baseline_clear(pre_launch_baseline, args.gpu_index, max_wait_s=60)


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
    "  max_ctx      (default: this model's own advertised native context length,",
    "                             from its GGUF metadata — omitted if the GGUF doesn't",
    "                             advertise one) hard cap on this model's context: also",
    "                             the large sample point when min_ctx is set, AND the",
    "                             ceiling `solve_scenario_sizes`'s \"auto\" ctx-sizing will",
    "                             never exceed even with VRAM to spare. Lower it here (then",
    "                             re-run `bench` — it's part of the config signature, so",
    "                             this alone triggers an automatic re-measurement) if you",
    "                             don't want a model ballooning to its full native ceiling.",
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
    "",
    "Sampling defaults (all optional, all passed straight through as llama-server",
    "CLI flags at spawn — see spawn_model()/sampling_args() in llama_priority_proxy.py):",
    "  temp, top_k, top_p, min_p, repeat_penalty, repeat_last_n,",
    "  presence_penalty, frequency_penalty",
    "These only set the server-side DEFAULT — a client that sends its own",
    "temperature/top_p/etc. in the request body still overrides them per-request.",
    "Meaningless for embedding models (no token sampling happens); omit for those.",
    "llama-server's own built-in defaults (temp=0.8, top_k=40, top_p=0.95, min_p=0.05,",
    "repeat_penalty=1.0 i.e. DISABLED, no presence/frequency penalty) are why an",
    "unconfigured model can fall into a repeat loop — nothing discourages it.",
]

# Cosmetic/proxy-facing fields — deliberately NOT part of config_signature, since
# neither affects VRAM measurement. Backfilled onto existing entries (not just new
# ones) by write_options_file() if missing, without touching anything else.
_OPTIONS_METADATA_DEFAULTS = {
    "label": None,       # None here means "use the model's own name" — see write_options_file
    "keep_alive": "",
}


def write_options_file(path, names, embedding_names=None, context_lengths=None):
    """Create or refresh model_options.json: add a default entry for any newly
    discovered model name, and backfill label/keep_alive onto EXISTING entries
    if either is missing — but never touch any value you've already set,
    including a label/keep_alive (or max_ctx) you've already customized.

    `embedding_names`: names GGUF metadata says are embedding models (see
    read_gguf_info's is_embedding) — these get extra_args: ["--embedding"]
    backfilled too if not already set, since the live proxy only ever reads
    this file (never the GGUF itself) and has no other way to know a model
    needs that flag to serve embeddings correctly.

    `context_lengths`: name -> the GGUF's own advertised context_length (or
    None/missing if that metadata wasn't present) — written as this model's
    default max_ctx so the cap the proxy's "auto" ctx-sizing will respect is
    visible and editable up front, rather than silently inherited from
    discover_models()'s own `opt.get("max_ctx") or info["context_length"]`
    fallback. Omitted entirely (not written as null) when the GGUF has no
    advertised length to offer — nothing to default to."""
    embedding_names = embedding_names or set()
    context_lengths = context_lengths or {}
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
        if context_lengths.get(name) is not None:
            entry["max_ctx"] = context_lengths[name]
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
        if "max_ctx" not in entry and context_lengths.get(name) is not None:
            entry["max_ctx"] = context_lengths[name]
            changed = True
        if "extra_args" not in entry and name in embedding_names:
            entry["extra_args"] = ["--embedding"]
            changed = True
        if changed:
            backfilled.append(name)

    # Rebuild with a fixed key order (_comment then options) rather than relying
    # on whatever order the dict happened to accumulate keys in — a freshly
    # created file (no os.path.exists(path) above) would otherwise get
    # "options" first and "_comment" appended after, since that's insertion
    # order for a dict built as {"options": {}} then assigned "_comment" later.
    data = {"_comment": data["_comment"], "options": dict(sorted(data["options"].items()))}
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
        shard_match = _SHARD_RE.search(fname)
        if shard_match and shard_match.group(1) != "00001":
            continue
        path = os.path.join(args.models_dir, fname)
        try:
            info = read_gguf_info(path)
        except (OSError, ValueError, struct.error) as e:
            print(f"[warn] skipping '{fname}': could not read GGUF metadata ({e})")
            continue
        if _is_mmproj_info(info):
            continue  # vision projector, not a launchable model — see _is_mmproj_info
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
        context_lengths = {}
        for fname, info in rows:
            shard_match = _SHARD_RE.search(fname)
            name = fname[: shard_match.start()] if shard_match else fname[: -len(".gguf")]
            context_lengths[name] = info["context_length"]
            if info["is_embedding"]:
                embedding_names.add(name)
        added, backfilled = write_options_file(args.write_options, names, embedding_names, context_lengths)
        if added:
            print(f"\nAdded {len(added)} new model(s) to '{args.write_options}' "
                  f"(parallel: 1, label: <name>, keep_alive: blank, "
                  f"max_ctx: <advertised native ctx, if known>): {', '.join(added)}")
        if backfilled:
            print(f"Backfilled missing label/keep_alive/max_ctx on {len(backfilled)} existing "
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
            existing_warmup = existing.get("warmup_prompt_tokens", 0)
            sig_matches = existing.get("config_signature") == sig
            warmup_matches = existing_warmup == args.warmup_prompt_tokens
            if sig_matches and warmup_matches:
                print(f"Skipping '{model_cfg['name']}' (already benchmarked, config unchanged)")
                continue
            if sig_matches:
                print(f"Re-benchmarking '{model_cfg['name']}' — warmup_prompt_tokens changed "
                      f"(was {existing_warmup}, now {args.warmup_prompt_tokens})")
            else:
                print(f"Re-benchmarking '{model_cfg['name']}' — config changed since last measurement "
                      f"(was {existing.get('config_signature')}, now {sig})")
        entry = benchmark_model(model_cfg, args.port, args.gpu_index, args.load_timeout, backend,
                                 args.warmup_prompt_tokens)
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

    i = sub.add_parser("inspect", help="List *.gguf files with metadata-derived params/native context (no GPU needed)")
    i.add_argument("--models-dir", required=True)
    i.add_argument("--write-options", default=_default_config_path("model_options.json"),
                   help="Create/refresh this model_options.json with a default {parallel: 1} entry "
                        "for every discovered model — the recommended first step before `bench`. "
                        "Existing entries are left untouched. (default: %(default)s)")
    i.set_defaults(func=cmd_inspect)

    b = sub.add_parser("bench", help="Measure VRAM footprint of models via llama-server")
    b.add_argument("--models-dir", required=True, help="Auto-discover *.gguf files here (name/path/max_ctx all inferred)")
    b.add_argument("--options", default=_default_config_path("model_options.json"),
                   help="model_options.json — per-model options keyed by discovered name "
                        "(parallel/min_ctx/max_ctx/extra_args/n_gpu_layers). "
                        "Generate/refresh one with `inspect --write-options`. "
                        "(default: %(default)s)")
    b.add_argument("--output", default=_default_config_path("model_vram_registry.json"), help="Registry output path")
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
    b.add_argument("--warmup-prompt-tokens", type=int, default=DEFAULT_WARMUP_PROMPT_TOKENS,
                   help="Send a synthetic prompt of roughly this many tokens through /completion "
                        "at each sample point before measuring 'settled' VRAM, so the compute "
                        "buffers a genuinely-sized real request needs get captured. llama.cpp's own "
                        "tiny startup warmup alone was found to understate real usage by several GB "
                        "at large context sizes -- see send_warmup_prompt() in llama_process.py for "
                        "the reproduction case. Pass 0 to disable (old, faster-but-riskier behavior; "
                        "only the health-check is used). Stored per-entry in the registry, so changing "
                        "this value triggers an automatic re-measurement same as any other config drift. "
                        "(default: %(default)s)")
    b.set_defaults(func=cmd_bench)

    s = sub.add_parser("solve", help="Solve max ctx-size for a VRAM budget")
    s.add_argument("--registry", default=_default_config_path("model_vram_registry.json"))
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
    s.add_argument("--options", default=_default_config_path("model_options.json"),
                   help="Checked against the registry's stored config_signature on every run — "
                        "warns if model_options.json has changed since a model was last "
                        "benchmarked (default: %(default)s). Pass --options '' to skip this check.")
    s.set_defaults(func=cmd_solve)

    v = sub.add_parser("validate", help="Actually launch a whole scenario (every model 'solve' would "
                                          "predict, all resident together) and hit the primary with a "
                                          "real prompt -- ground truth for 'does this fit AND perform "
                                          "well', not another prediction. See resolve_scenario_sizes's "
                                          "docstring / cmd_validate's module comment for why this exists.")
    v.add_argument("--scenario", required=True, help="Scenario JSON to validate (see config/scenario_coding.json)")
    v.add_argument("--models-dir", required=True, help="Auto-discover *.gguf files here (name/path/max_ctx all inferred)")
    v.add_argument("--options", default=_default_config_path("model_options.json"))
    v.add_argument("--registry", default=_default_config_path("model_vram_registry.json"))
    v.add_argument("--standalone", default=_default_config_path("standalone_models.json"),
                   help="Always-on models (e.g. an embedding model) that also eat into the same "
                        "live budget the real proxy computes -- see standalone_models.json. Pass "
                        "--standalone '' to ignore.")
    v.add_argument("--budget-gb", type=float, help="VRAM budget in GB (default: detected total - headroom)")
    v.add_argument("--headroom-gb", type=float, default=DEFAULT_HEADROOM_GB)
    v.add_argument("--gpu-index", type=int, default=0)
    v.add_argument("--port-base", type=int, default=18090, help="First scratch port; each launched model gets the next one")
    v.add_argument("--load-timeout", type=int, default=300, help="Seconds to wait for /health per model")
    v.add_argument("--request-timeout", type=int, default=240, help="Seconds to wait for the real test prompt to complete")
    v.add_argument("--warmup-prompt-tokens", type=int, default=DEFAULT_WARMUP_PROMPT_TOKENS,
                   help="Size of the real test prompt sent to the scenario's primary model. "
                        "(default: %(default)s)")
    v.add_argument("--min-prefill-tps", type=float, default=200.0,
                   help="Fail if measured prefill (prompt_per_second) drops below this -- normal is "
                        "1000+ tok/s on this project's hardware, degraded/shared-memory-spillover "
                        "cases measured ~34 tok/s, so 200 cleanly separates the two. (default: %(default)s)")
    v.add_argument("--min-free-mb", type=float, default=512.0,
                   help="Fail if live free VRAM (after the test prompt, with every model in the "
                        "scenario resident) drops below this. (default: %(default)s)")
    v.add_argument("--backend", choices=["docker", "native"], default="docker")
    v.add_argument("--image", default="llama-cpp-priority-proxy",
                   help="Docker image to run llama-server from (--backend docker only)")
    v.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    if args.cmd in ("solve", "validate") and args.budget_gb is None:
        total = gpu_total_bytes(args.gpu_index)
        args.budget_gb = (total - args.headroom_gb * (1024 ** 3)) / (1024 ** 3)
        print(f"[info] auto budget: {args.budget_gb:.2f} GB (detected {format_bytes(total)} - "
              f"{args.headroom_gb} GB headroom)")

    args.func(args)


if __name__ == "__main__":
    main()
