import os
import time
import json
import requests

API_KEY_PATH = os.path.expanduser("~/docker/secrets/ollama_token")
BASE_URL = "http://localhost:11446"

with open(API_KEY_PATH, "r") as f:
    api_key = f.read().strip()

headers = {"Authorization": f"Bearer {api_key}"}


def poll_slots(duration=90, interval=0.5, outfile="slots_trace.json"):
    log = []
    t0 = time.time()
    print(f"Polling /slots every {interval}s for {duration}s... (Ctrl+C to stop early)")
    try:
        while time.time() - t0 < duration:
            r = requests.get(f"{BASE_URL}/slots", headers=headers, timeout=5).json()
            elapsed = round(time.time() - t0, 2)
            # Print a compact one-line summary per poll so you can watch it live
            summary = [
                f"slot{s['id']}:{'BUSY' if s['is_processing'] else 'idle'}"
                for s in r
            ]
            print(f"t={elapsed:>6}s  {'  '.join(summary)}")
            log.append({"t": elapsed, "slots": r})
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Stopped early.")

    with open(outfile, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nWrote {len(log)} samples to {outfile}")


if __name__ == "__main__":
    poll_slots(duration=90, interval=0.5)