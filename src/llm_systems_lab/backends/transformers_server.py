"""Transparent, batch-one Transformers baseline with an OpenAI-style API.

This is intentionally a reference baseline, not a production server. A process
lock serializes GPU generation so concurrency creates queueing rather than
unsafe simultaneous mutation of model state.
"""

import argparse
import json
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Iterator, List


def create_app(model_id: str, revision: str, dtype_name: str) -> Any:
    try:
        import torch
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install the exp002-transformers dependencies before starting this server."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 002 requires a CUDA GPU for formal measurements.")
    dtype_by_name = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    if dtype_name not in dtype_by_name:
        raise ValueError(f"unsupported dtype: {dtype_name}")

    state: Dict[str, Any] = {}
    generation_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: Any) -> Iterator[None]:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=dtype_by_name[dtype_name],
            device_map={"": 0},
            attn_implementation="sdpa",
        )
        model.eval()
        state.update(tokenizer=tokenizer, model=model)
        yield
        state.clear()
        torch.cuda.empty_cache()

    app = FastAPI(title="Transformers Experiment 002 Baseline", lifespan=lifespan)

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok" if state else "loading"}

    @app.get("/v1/models")
    def models() -> Dict[str, List[Dict[str, str]]]:
        return {"data": [{"id": model_id, "object": "model"}]}

    @app.post("/v1/chat/completions")
    def chat_completions(payload: Dict[str, Any]) -> Any:
        if not payload.get("stream", False):
            raise HTTPException(status_code=400, detail="stream=true is required")
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages are required")
        max_tokens = int(payload.get("max_tokens", 1))
        if max_tokens <= 0:
            raise HTTPException(status_code=400, detail="max_tokens must be positive")

        tokenizer = state["tokenizer"]
        model = state["model"]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        request_id = f"chatcmpl-{uuid.uuid4().hex}"

        def events() -> Iterator[str]:
            completion_tokens = 0
            with generation_lock, torch.inference_mode():
                input_ids = encoded["input_ids"].to("cuda:0")
                attention_mask = encoded["attention_mask"].to("cuda:0")
                past_key_values = None
                for _ in range(max_tokens):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    past_key_values = outputs.past_key_values
                    token_id = int(next_token.item())
                    content = tokenizer.decode(
                        [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
                    )
                    completion_tokens += 1
                    event = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {"content": content}}],
                    }
                    yield f"data: {json.dumps(event)}\n\n"
                    input_ids = next_token
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones(
                                (attention_mask.shape[0], 1),
                                dtype=attention_mask.dtype,
                                device=attention_mask.device,
                            ),
                        ],
                        dim=-1,
                    )

                usage = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
                yield f"data: {json.dumps(usage)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(
        create_app(args.model, args.revision, args.dtype),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
