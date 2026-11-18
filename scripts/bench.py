"""Drive Lab 08 with a reproducible local Qwen3 workload."""

import argparse
import dataclasses
import json
import random
import sys
from pathlib import Path

import torch

from tinyserve.bench import Arrival, run_trace, summarize
from tinyserve.engine import Engine
from tinyserve.hf_io import load_model
from tinyserve.requests import Request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--max-active", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=640)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--batch-tokens", type=int, default=256)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--gap-ms", type=float, default=0.0)
    parser.add_argument("--profile", choices=("short", "mixed"), default="mixed")
    args = parser.parse_args()
    if args.requests < 1 or args.output_tokens < 1 or args.gap_ms < 0:
        parser.error("requests and output tokens must be positive; gap must be nonnegative")
    dtype = getattr(torch, args.dtype)
    model = load_model(args.model_dir, device=args.device, dtype=dtype)
    engine = Engine(
        model,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_active=args.max_active,
        max_batch_tokens=args.batch_tokens,
        max_prefill_chunk=args.chunk,
    )
    rng = random.Random(4080)
    trace = []
    for index in range(args.requests):
        length = 1024 if args.profile == "mixed" and index % 4 == 0 else 128
        prompt = tuple(rng.randrange(10_000) for _ in range(length))
        trace.append(Arrival(index * args.gap_ms / 1000, Request(str(index), prompt, args.output_tokens)))
    warmup = [
        Arrival(0, Request(f"warmup-{i}", arrival.request.prompt_ids, min(8, args.output_tokens)))
        for i, arrival in enumerate(trace[: args.max_active])
    ]
    run_trace(engine, warmup)
    cuda = torch.device(args.device).type == "cuda"
    if cuda:
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    samples = run_trace(engine, trace)
    report = {
        "config": vars(args) | {"model_dir": str(args.model_dir), "seed": 4080},
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "metrics": dataclasses.asdict(summarize(samples)),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device) if cuda else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device) if cuda else None,
        "free_blocks_after_run": engine.free_blocks,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
