#!/usr/bin/env python3
import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path


def run_once(url: str, api_key: str, output_tokens: int) -> dict[str, float]:
    payload = json.dumps(
        {
            "model": "qwen3.8-flash-next",
            "prompt": "Explain B-tree insertion, page splits, and lookup complexity.",
            "max_tokens": output_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(f"{url}/v1/completions", payload, headers)
    started = time.perf_counter()
    first = None
    completion_tokens = 0
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("choices") and event["choices"][0].get("text") and first is None:
                first = time.perf_counter()
            if event.get("usage"):
                completion_tokens = event["usage"].get("completion_tokens", completion_tokens)
    finished = time.perf_counter()
    if first is None:
        raise RuntimeError("No output tokens")
    decode_tokens = max(completion_tokens - 1, 0)
    return {
        "ttft_s": first - started,
        "total_s": finished - started,
        "completion_tokens": completion_tokens,
        "output_tps": decode_tokens / (finished - first),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    key_file = Path(__file__).with_name(".api-key")
    api_key = key_file.read_text().strip() if key_file.exists() else ""
    results = []
    for run_number in range(1, args.runs + 1):
        result = run_once(f"http://127.0.0.1:{args.port}", api_key, args.output_tokens)
        result["run"] = run_number
        results.append(result)
        print(json.dumps(result), flush=True)
    print(json.dumps({
        "runs": args.runs,
        "median_ttft_s": statistics.median(item["ttft_s"] for item in results),
        "median_output_tps": statistics.median(item["output_tps"] for item in results),
        "mean_output_tps": statistics.mean(item["output_tps"] for item in results),
    }))


if __name__ == "__main__":
    main()
