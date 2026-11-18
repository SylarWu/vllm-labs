"""Serve a local Qwen3 snapshot through the Lab 07 HTTP interface."""

import argparse
from pathlib import Path

import torch
import uvicorn

from tinyserve.api import EngineChatBackend, create_app
from tinyserve.engine import Engine
from tinyserve.hf_io import HfTokenizer, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    model = load_model(args.model_dir, device=args.device, dtype=getattr(torch, args.dtype))
    tokenizer = HfTokenizer(args.model_dir)
    engine = Engine(model, num_blocks=640, block_size=16, max_active=8, max_batch_tokens=256, max_prefill_chunk=128)
    backend = EngineChatBackend(engine, tokenizer, model="qwen3-lab", eos_token_id=tokenizer.eos_token_id)
    uvicorn.run(create_app(backend), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
