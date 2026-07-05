#!/usr/bin/env python3
"""
llama_process.py — shared llama-server process management: launch, health-check,
shutdown, and PID resolution, for either a native subprocess or a throwaway
docker container.

Used by:
  - benchmark_vram.py, which supports both backends ("docker": spins up a
    throwaway container per benchmark trial from the host; "native": execs
    llama-server directly, e.g. from inside the same container that has it).
  - llama_priority_proxy.py, which only ever uses "native" — it runs inside
    the same container as llama-server and manages long-lived child processes
    rather than one-off benchmark trials.

Both backends are driven through the same ServerHandle/launch_server/
wait_for_health/handle_pid/shutdown_server functions — only their internals
differ per backend ("kind" field on ServerHandle).

Also holds the VRAM-fit math (max_ctx_for_budget/predicted_vram) shared between
benchmark_vram.py's `solve` command and the live proxy's scenario-group sizing —
small and dependency-free enough not to warrant a third module.
"""

import os
import signal
import subprocess
import time
from urllib import request as urlrequest
from urllib.error import URLError

LLAMA_SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN", "llama-server")
LOG_DIR = "/tmp/llama_bench_logs"

_LLAMA_SERVER_FLAGS = ["--flash-attn", "on", "--kv-unified", "--cont-batching", "--no-context-shift"]


def format_bytes(b):
    if b is None:
        return "N/A"
    b = float(b)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(b) < 1024.0:
            return f"{b:.2f} {unit}"
        b /= 1024.0
    return f"{b:.2f} PB"


def gpu_total_bytes(gpu_index=0):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits",
         f"--id={gpu_index}"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"nvidia-smi failed to report total VRAM: {result.stderr}")
    return int(float(result.stdout.strip().splitlines()[0])) * 1024 * 1024


def gpu_used_bytes(gpu_index=0):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
         f"--id={gpu_index}"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return int(float(result.stdout.strip().splitlines()[0])) * 1024 * 1024


class ServerHandle:
    """kind='native': proc + log_path are set. kind='docker': container_name is
    set (None if `docker run` itself failed to even start, in which case
    docker_launch_error holds its stderr)."""
    def __init__(self, kind, port, proc=None, log_path=None,
                 container_name=None, docker_launch_error=None, gpu_index=0):
        self.kind = kind
        self.port = port
        self.proc = proc
        self.log_path = log_path
        self.container_name = container_name
        self.docker_launch_error = docker_launch_error
        self.gpu_index = gpu_index


def launch_server(model_path, ctx, parallel, port, n_gpu_layers=99, extra_args=None, backend=None):
    backend = backend or {"kind": "native"}
    llama_args = [
        "--host", "127.0.0.1", "--port", str(port),
        "--ctx-size", str(ctx), "--parallel", str(parallel),
        "--n-gpu-layers", str(n_gpu_layers),
    ] + _LLAMA_SERVER_FLAGS + (extra_args or [])

    if backend["kind"] == "native":
        args = [LLAMA_SERVER_BIN, "--model", model_path] + llama_args
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, f"{os.path.basename(model_path)}_ctx{ctx}_p{parallel}.log")
        proc = subprocess.Popen(args, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
        return ServerHandle("native", port, proc=proc, log_path=log_path)

    # docker backend: model_path is a HOST path (as discovered from --models-dir);
    # translate it to the in-container mount path.
    container_name = f"llama-bench-{os.getpid()}-{int(time.time() * 1000) % 1_000_000}"
    gpu_flag = "all" if backend["gpu_index"] == 0 else f"device={backend['gpu_index']}"
    container_model_path = f"/models/{os.path.basename(model_path)}"
    cmd = [
        "docker", "run", "-d",
        "--gpus", gpu_flag,
        "--network", "host",
        "--name", container_name,
        "--entrypoint", "llama-server",
        "-v", f"{backend['models_dir']}:/models:ro",
        backend["image"],
        "--model", container_model_path,
    ] + llama_args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return ServerHandle("docker", port, docker_launch_error=result.stderr.strip(),
                             gpu_index=backend["gpu_index"])
    return ServerHandle("docker", port, container_name=container_name, gpu_index=backend["gpu_index"])


def _docker_inspect(container_name, fmt):
    result = subprocess.run(["docker", "inspect", "--format", fmt, container_name],
                             capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def print_log_tail(handle, n=20):
    if handle.kind == "native":
        try:
            with open(handle.log_path) as f:
                lines = f.readlines()[-n:]
        except OSError:
            print(f"    (could not read log at {handle.log_path})")
            return
        source = handle.log_path
    elif handle.docker_launch_error is not None:
        lines = handle.docker_launch_error.splitlines(keepends=True)[-n:]
        source = "docker run stderr (container never started)"
    else:
        result = subprocess.run(["docker", "logs", "--tail", str(n), handle.container_name],
                                 capture_output=True, text=True)
        lines = (result.stdout + result.stderr).splitlines(keepends=True)
        source = f"docker logs {handle.container_name}"
    print(f"    --- last {len(lines)} line(s) of {source} ---")
    for line in lines:
        print(f"    | {line.rstrip()}")


def wait_for_health(handle, timeout_s):
    """Returns (True, None) once /health responds, or (False, reason) on failure —
    reason is 'died' (process/container exited — check the log, could be OOM or
    anything else: bad GGUF, arg error, etc.) or 'timeout' (still starting)."""
    if handle.kind == "docker" and handle.docker_launch_error is not None:
        return False, f"docker run failed to start container: {handle.docker_launch_error}"

    url = f"http://127.0.0.1:{handle.port}/health"
    start = time.time()
    deadline = start + timeout_s
    while time.time() < deadline:
        alive, exit_code = _handle_alive(handle)
        if not alive:
            return False, f"exited (code {exit_code}) after {time.time() - start:.1f}s"
        try:
            with urlrequest.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return True, None
        except (URLError, OSError, TimeoutError):
            pass
        time.sleep(1)
    return False, f"timed out after {timeout_s}s (still starting)"


def _handle_alive(handle):
    """Returns (running: bool, exit_code: int|None)."""
    if handle.kind == "native":
        exit_code = handle.proc.poll()
        return exit_code is None, exit_code
    running = _docker_inspect(handle.container_name, "{{.State.Running}}")
    if running is None:
        return False, None  # container gone entirely
    if running == "true":
        return True, None
    exit_code = _docker_inspect(handle.container_name, "{{.State.ExitCode}}")
    return False, (int(exit_code) if exit_code else None)


def handle_pid(handle):
    """The host-namespace PID nvidia-smi would report for this process —
    works the same whether it's a native subprocess or a docker container,
    since nvidia-smi always reports the host-visible PID."""
    if handle.kind == "native":
        return handle.proc.pid
    pid = _docker_inspect(handle.container_name, "{{.State.Pid}}")
    return int(pid) if pid and pid != "0" else None


def shutdown_server(handle, timeout_s=15):
    if handle.kind == "native":
        if handle.proc.poll() is not None:
            return
        handle.proc.send_signal(signal.SIGTERM)
        try:
            handle.proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            handle.proc.kill()
            handle.proc.wait(timeout=10)
        return

    if handle.container_name is None:
        return  # docker run itself never created a container
    subprocess.run(["docker", "stop", "--time", str(timeout_s), handle.container_name],
                    capture_output=True)
    subprocess.run(["docker", "rm", "-f", handle.container_name], capture_output=True)


def max_ctx_for_budget(entry, budget_bytes, round_to=256, ctx_cap=None):
    """Given a fitted registry entry, find the largest ctx (<= ctx_cap) whose
    predicted vram fits budget_bytes. ctx_cap defaults to entry["max_ctx"] (the
    model's own per-conversation ceiling) — pass parallel * max_ctx instead when
    sizing a --kv-unified POOL meant to hold `parallel` full-length conversations
    at once; the pool isn't limited to a single conversation's ceiling."""
    slope, base = entry["bytes_per_ctx_token"], entry["base_vram_bytes"]
    cap = entry["max_ctx"] if ctx_cap is None else ctx_cap
    if slope <= 0:
        return cap if base <= budget_bytes else 0
    raw = (budget_bytes - base) / slope
    ctx = int(raw // round_to) * round_to
    return max(0, min(ctx, cap))


def predicted_vram(entry, ctx):
    return entry["base_vram_bytes"] + entry["bytes_per_ctx_token"] * ctx
