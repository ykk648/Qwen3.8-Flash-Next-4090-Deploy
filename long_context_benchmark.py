#!/usr/bin/env python3
import json
import secrets
import time
import urllib.request
from pathlib import Path


TARGETS = (8_000, 32_000, 64_000, 120_000)


def post(url: str, api_key: str, payload: dict):
    request = urllib.request.Request(
        url,
        json.dumps(payload).encode(),
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    return urllib.request.urlopen(request, timeout=1800)


def build_prompt(target_tokens: int, passkey: str) -> str:
    prefix = (
        "Read the entire context and remember the passkey. "
        f"The passkey is {passkey}. Do not change it.\n"
    )
    suffix = "\nQuestion: What is the passkey? Output only the digits.\nAnswer:"
    filler_tokens = max(target_tokens - 48, 1)
    return prefix + (" alpha" * filler_tokens) + suffix


def run(url: str, api_key: str, target_tokens: int) -> dict:
    passkey = str(secrets.randbelow(90000000) + 10000000)
    prompt = build_prompt(target_tokens, passkey)
    payload = {
        "model": "qwen3.8-flash-next",
        "prompt": prompt,
        "max_tokens": 64,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_prompt": False,
    }
    started = time.perf_counter()
    first = None
    text = []
    usage = {}
    with post(f"{url}/v1/completions", api_key, payload) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("choices"):
                chunk = event["choices"][0].get("text", "")
                if chunk and first is None:
                    first = time.perf_counter()
                text.append(chunk)
            if event.get("usage"):
                usage = event["usage"]
    finished = time.perf_counter()
    output = "".join(text)
    completion_tokens = usage.get("completion_tokens", 0)
    decode_seconds = finished - first if first is not None else 0
    return {
        "target_tokens": target_tokens,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "ttft_s": first - started if first is not None else None,
        "total_s": finished - started,
        "decode_tps": (completion_tokens - 1) / decode_seconds if decode_seconds else 0,
        "passkey_retrieved": passkey in output,
        "output_preview": output[-160:].replace("\n", "\\n"),
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    api_key = (root / ".api-key").read_text().strip()
    for target_tokens in TARGETS:
        print(json.dumps(run("http://127.0.0.1:8001", api_key, target_tokens)), flush=True)


if __name__ == "__main__":
    main()
