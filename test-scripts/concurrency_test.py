import os
import time
import json
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
        timeout=600,
    ) as resp:
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
                last_token_time = now
                token_count += 1

    with lock:
        results[label] = {
            "request_start": round(t_request_start, 3),
            "first_token_at": round(first_token_time, 3) if first_token_time else None,
            "last_token_at": round(last_token_time, 3) if last_token_time else None,
            "ttft": round(first_token_time - t_request_start, 3) if first_token_time else None,
            "total_duration": round(last_token_time - t_request_start, 3) if last_token_time else None,
            "token_count": token_count,
        }


def main():
    global start_time
    start_time = time.time()

    # Long request: simulates your heavy coding-session generation
    long_thread = threading.Thread(
        target=stream_request,
        args=("LONG (coding session)", "Write a detailed 500-word story about a robot exploring an old library.", 600),
    )

    # Short request: simulates your wife's quick WebUI lookup
    short_thread = threading.Thread(
        target=stream_request,
        args=("SHORT (wife's query)", "What is the capital of France? Answer in one word.", 20),
    )

    long_thread.start()
    time.sleep(2)  # let the long request establish itself first, mimicking mid-session interruption
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
    long = results.get("LONG (coding session)")
    if short and long and short["ttft"] is not None:
        if short["first_token_at"] < long["last_token_at"]:
            print("Short request's first token arrived BEFORE the long request finished.")
            print("=> Continuous batching is interleaving requests. Concurrency is working.")
        else:
            print("Short request's first token only arrived AFTER the long request finished.")
            print("=> Requests appear to be serialized, not interleaved.")
        print(f"Short request time-to-first-token: {short['ttft']}s")


if __name__ == "__main__":
    main()