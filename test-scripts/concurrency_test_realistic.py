import os
import time
import json
import uuid
import threading
import requests

API_KEY_PATH = os.path.expanduser("~/docker/secrets/ollama_token")
BASE_URL = "http://localhost:11444"
MODEL_NAME = "ornith-35b"

with open(API_KEY_PATH, "r") as f:
    api_key = f.read().strip()

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

start_time = None
results = {}
lock = threading.Lock()


def build_haystack():
    # Random prefix busts prompt-cache reuse from any previous test run,
    # forcing a genuinely cold prefill this time. Applied ONCE, outside the
    # repeated filler — putting it inside the multiplied string bloats the
    # token count by ~20 tokens x 27000 repeats.
    cache_buster = uuid.uuid4().hex
    filler = "The sky is blue and grass is green. " * 27000
    needle = "\n\nThe secret code word is: PINEAPPLE-7742.\n\n"
    haystack = filler[: len(filler) // 10] + needle + filler[len(filler) // 10 :]
    return f"[session-{cache_buster}]\n\n" + haystack


def stream_request(label, prompt, max_tokens):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }

    t_request_start = time.time() - start_time
    first_token_time = None
    last_token_time = None
    token_count = 0

    with requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=headers,
        json=payload,
        stream=True,
        timeout=900,
    ) as resp:
        if resp.status_code >= 400:
            print(f"  [{label}] ERROR {resp.status_code}: {resp.text}")
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data_str = line[len("data: "):]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content") or delta.get("reasoning_content")
            if content:
                now = time.time() - start_time
                if first_token_time is None:
                    first_token_time = now
                    print(f"  [{label}] first token at t={now:.2f}s")
                last_token_time = now
                token_count += 1

    with lock:
        ttft = round(first_token_time - t_request_start, 3) if first_token_time else None
        total_duration = round(last_token_time - t_request_start, 3) if last_token_time else None
        decode_duration = (
            round(last_token_time - first_token_time, 3)
            if first_token_time and last_token_time and token_count > 1
            else None
        )
        # Decode-phase throughput: tokens/sec during actual generation,
        # excluding TTFT/prefill wait. More meaningful than overall throughput
        # for comparing generation speed independent of queueing delays.
        decode_tps = (
            round((token_count - 1) / decode_duration, 2)
            if decode_duration and decode_duration > 0
            else None
        )
        # Overall throughput: tokens/sec including the wait for first token.
        # This is what the requester actually experienced end-to-end.
        overall_tps = (
            round(token_count / total_duration, 2)
            if total_duration and total_duration > 0
            else None
        )

        results[label] = {
            "request_start": round(t_request_start, 3),
            "first_token_at": round(first_token_time, 3) if first_token_time else None,
            "last_token_at": round(last_token_time, 3) if last_token_time else None,
            "ttft": ttft,
            "total_duration": total_duration,
            "token_count": token_count,
            "decode_tokens_per_sec": decode_tps,
            "overall_tokens_per_sec": overall_tps,
        }


def main():
    global start_time

    print("Building fresh ~240K-token haystack (cache-busted)...")
    haystack = build_haystack()

    start_time = time.time()

    # LONG: simulates your real coding session — big context (~240K tokens),
    # cold prefill, plus a generation ask on top.
    long_thread = threading.Thread(
        target=stream_request,
        args=(
            "LONG (coding session, big context)",
            haystack + "\n\nSummarize this document in 200 words, then mention the secret code word.",
            400,
        ),
    )

    # SHORT: your wife's quick WebUI lookup, tiny prompt.
    short_thread = threading.Thread(
        target=stream_request,
        args=("SHORT (wife's query)", "What is the capital of France? Answer in one word.", 20),
    )

    print("Starting LONG request (cold ~240K-token prefill)...")
    long_thread.start()

    time.sleep(2)  # let the long request's prefill get underway first
    print("Starting SHORT request while LONG is still prefilling...")
    short_thread.start()

    long_thread.join()
    short_thread.join()

    print("\n=== Results ===")
    for label, r in results.items():
        print(f"\n{label}:")
        for k, v in r.items():
            print(f"  {k}: {v}")

    print("\n=== Verdict ===")
    short = results.get("SHORT (wife's query)")
    long = results.get("LONG (coding session, big context)")
    if short and long and short["ttft"] is not None:
        print(f"Short request TTFT: {short['ttft']}s")
        print(f"Long request TTFT (its own prefill time): {long['ttft']}s")
        if short["ttft"] < long["ttft"] * 0.5:
            print("=> Short request got a fast response DESPITE the long cold prefill in flight.")
            print("   Chunked prefill / continuous batching is protecting interactive latency.")
        else:
            print("=> Short request's TTFT tracked close to the long request's prefill time.")
            print("   This suggests the short query had to wait behind (or alongside) the")
            print("   long request's prefill rather than being prioritized ahead of it.")


if __name__ == "__main__":
    main()