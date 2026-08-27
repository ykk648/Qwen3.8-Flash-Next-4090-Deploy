#!/usr/bin/env python3
import json
import time
import urllib.request
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    api_key = (root / ".api-key").read_text().strip()
    payload = json.dumps(
        {
            "model": "qwen3.8-flash-next",
            "input": (" alpha" * 7_950) + "\nReply with exactly OK.",
            "max_output_tokens": 32,
            "stream": False,
        }
    ).encode()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    for run_number in (1, 2):
        request = urllib.request.Request(
            "http://127.0.0.1:8001/v1/responses", payload, headers
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=1800) as response:
            result = json.load(response)
        elapsed = time.perf_counter() - started
        details = result.get("usage", {}).get("input_tokens_details", {})
        print(
            json.dumps(
                {
                    "run": run_number,
                    "elapsed_s": elapsed,
                    "input_tokens": result.get("usage", {}).get("input_tokens"),
                    "cached_tokens": details.get("cached_tokens"),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
