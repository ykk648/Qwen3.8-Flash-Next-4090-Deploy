#!/usr/bin/env python3
import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any


def run_once(
    url: str, api_key: str, output_tokens: int, prompt: str
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": "qwen3.8-flash-next",
            "prompt": prompt,
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
    timings = {}
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
            if event.get("timings"):
                timings = event["timings"]
    finished = time.perf_counter()
    if first is None:
        raise RuntimeError("No output tokens")
    decode_tokens = max(completion_tokens - 1, 0)
    result = {
        "ttft_s": first - started,
        "total_s": finished - started,
        "completion_tokens": completion_tokens,
        "output_tps": decode_tokens / (finished - first),
    }
    for key in (
        "prompt_n",
        "prompt_ms",
        "prompt_per_second",
        "predicted_n",
        "predicted_ms",
        "predicted_per_second",
        "draft_n",
        "draft_n_accepted",
    ):
        if key in timings:
            result[f"server_{key}"] = timings[key]
    if timings.get("draft_n"):
        result["draft_acceptance"] = (
            timings.get("draft_n_accepted", 0) / timings["draft_n"]
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--url")
    parser.add_argument("--output")
    parser.add_argument(
        "--prompt",
        default="Explain B-tree insertion, page splits, and lookup complexity.",
    )
    args = parser.parse_args()
    key_file = Path(__file__).with_name(".api-key")
    api_key = key_file.read_text().strip() if key_file.exists() else ""
    url = args.url or f"http://127.0.0.1:{args.port}"
    results: list[dict[str, Any]] = []
    for run_number in range(1, args.runs + 1):
        result = run_once(url, api_key, args.output_tokens, args.prompt)
        result["run"] = run_number
        results.append(result)
        print(json.dumps(result), flush=True)
    summary = {
        "runs": args.runs,
        "median_ttft_s": statistics.median(item["ttft_s"] for item in results),
        "median_output_tps": statistics.median(item["output_tps"] for item in results),
        "mean_output_tps": statistics.mean(item["output_tps"] for item in results),
    }
    server_tps = [item["server_predicted_per_second"] for item in results if "server_predicted_per_second" in item]
    acceptance = [item["draft_acceptance"] for item in results if "draft_acceptance" in item]
    if server_tps:
        summary["median_server_output_tps"] = statistics.median(server_tps)
    if acceptance:
        summary["mean_draft_acceptance"] = statistics.mean(acceptance)
    print(json.dumps(summary))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"results": results, "summary": summary}, indent=2) + "\n")


if __name__ == "__main__":
    main()
