"""Serve a local Qwen3 snapshot through the Lab 07 HTTP interface."""

import argparse
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse

from tinyserve.api import ChatBackend, EngineChatBackend, create_app
from tinyserve.engine import Engine
from tinyserve.hf_io import HfTokenizer, load_model


def create_serve_app(backend: ChatBackend) -> FastAPI:
    app = create_app(backend)

    @app.get("/chat", response_class=FileResponse, include_in_schema=False)
    async def chat_page() -> FileResponse:
        return FileResponse(Path(__file__).with_name("chat.html"), media_type="text/html")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--port", type=int, default=7789)
    args = parser.parse_args()

    model = load_model(args.model_dir, device=args.device, dtype=getattr(torch, args.dtype))
    tokenizer = HfTokenizer(args.model_dir)
    engine = Engine(model, num_blocks=640, block_size=16, max_active=8, max_batch_tokens=256, max_prefill_chunk=128)
    backend = EngineChatBackend(engine, tokenizer, model="qwen3-lab", eos_token_id=tokenizer.eos_token_id)
    uvicorn.run(create_serve_app(backend), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
