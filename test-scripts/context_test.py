import os
import requests

API_KEY_PATH = os.path.expanduser("~/docker/secrets/ollama_token")
BASE_URL = "http://localhost:11444"
MODEL_NAME = "ornith-35b"

with open(API_KEY_PATH, "r") as f:
    api_key = f.read().strip()

headers = {"Authorization": f"Bearer {api_key}"}

# Build a haystack with a unique needle buried near the start
filler = "The sky is blue and grass is green. " * 27000  # ~243K tokens, leaving headroom
needle = "\n\nThe secret code word is: PINEAPPLE-7742.\n\n"
haystack = filler[: len(filler) // 10] + needle + filler[len(filler) // 10 :]

# 1. Confirm actual token count
tok_resp = requests.post(
    f"{BASE_URL}/tokenize",
    headers=headers,
    json={"content": haystack},
    timeout=300,
)
tok_resp.raise_for_status()
n_tokens = len(tok_resp.json()["tokens"])
print(f"Haystack token count: {n_tokens}")

# 2. Ask the model to recall the needle
question = "\n\nWhat is the secret code word mentioned earlier in this document? Reply with just the code word."
chat_resp = requests.post(
    f"{BASE_URL}/v1/chat/completions",
    headers=headers,
    json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": haystack + question}],
        "max_tokens": 1000,
    },
    timeout=600,
)
print(chat_resp.status_code, chat_resp.text)
chat_resp.raise_for_status()
answer = chat_resp.json()["choices"][0]["message"]["content"]
print(f"Model answer: {answer}")

if "PINEAPPLE-7742" in answer:
    print("PASS: needle recalled correctly")
else:
    print("FAIL: needle not found in response")