"""A small observable gateway for overload and canary experiments."""

import argparse
import asyncio
import json
import logging
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, AsyncIterator, Dict

from .canary import assign_lane


LOGGER = logging.getLogger("llm-systems-gateway")


class GatewayState:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.slots = asyncio.Semaphore(int(config["max_in_flight"]))
        self.counters = Counter()
        self.latencies = defaultdict(list)
        self.in_flight = 0

    def metrics_text(self) -> str:
        lines = [
            "# HELP llm_gateway_requests_total Completed requests by lane and status.",
            "# TYPE llm_gateway_requests_total counter",
        ]
        for (lane, status), value in sorted(self.counters.items(), key=str):
            if lane in {"stable", "canary"}:
                lines.append(f'llm_gateway_requests_total{{lane="{lane}",status="{status}"}} {value}')
        lines.extend(["# TYPE llm_gateway_in_flight gauge", f"llm_gateway_in_flight {self.in_flight}"])
        for lane, values in sorted(self.latencies.items()):
            lines.append(f'llm_gateway_latency_ms_count{{lane="{lane}"}} {len(values)}')
            lines.append(f'llm_gateway_latency_ms_sum{{lane="{lane}"}} {sum(values):.3f}')
        lines.append(f"llm_gateway_rejections_total {self.counters['system', 'rejected']}")
        lines.append(f"llm_gateway_timeouts_total {self.counters['system', 'timeout']}")
        return "\n".join(lines) + "\n"


def create_app(config: Dict[str, Any]):
    try:
        import httpx
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
    except ImportError as exc:
        raise RuntimeError("Install the production optional dependencies") from exc

    app = FastAPI(title="LLM Systems Lab Gateway")
    state = GatewayState(config)
    client = httpx.AsyncClient(timeout=float(config["request_timeout_seconds"]))

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(state.metrics_text(), media_type="text/plain; version=0.0.4")

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        lane = assign_lane(request_id, float(config["canary_fraction"]))
        base_url = config[f"{lane}_base_url"].rstrip("/")
        try:
            await asyncio.wait_for(state.slots.acquire(), timeout=float(config["queue_timeout_ms"]) / 1000)
        except asyncio.TimeoutError:
            state.counters["system", "rejected"] += 1
            return JSONResponse(status_code=429, content={"error": {"type": "overloaded", "message": "admission queue deadline exceeded", "request_id": request_id}})

        state.in_flight += 1
        started = time.perf_counter()
        body = await request.body()
        upstream = None
        try:
            upstream_request = client.build_request("POST", f"{base_url}/chat/completions", content=body, headers={"content-type": "application/json", "x-request-id": request_id})
            upstream = await client.send(upstream_request, stream=True)
        except httpx.TimeoutException:
            state.counters["system", "timeout"] += 1
            state.in_flight -= 1
            state.slots.release()
            return JSONResponse(status_code=504, content={"error": {"type": "upstream_timeout", "message": "upstream deadline exceeded", "request_id": request_id}})
        except httpx.HTTPError:
            state.counters[lane, "upstream_error"] += 1
            state.in_flight -= 1
            state.slots.release()
            return JSONResponse(status_code=502, content={"error": {"type": "upstream_error", "message": "upstream request failed", "request_id": request_id}})

        async def relay() -> AsyncIterator[bytes]:
            status = str(upstream.status_code)
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await upstream.aclose()
                elapsed = (time.perf_counter() - started) * 1000
                state.counters[lane, status] += 1
                state.latencies[lane].append(elapsed)
                state.in_flight -= 1
                state.slots.release()
                LOGGER.info("request_complete request_id=%s lane=%s status=%s elapsed_ms=%.3f", request_id, lane, status, elapsed)

        return StreamingResponse(relay(), status_code=upstream.status_code, media_type=upstream.headers.get("content-type", "text/event-stream"), headers={"x-request-id": request_id, "x-model-lane": lane})

    @app.on_event("shutdown")
    async def shutdown():
        await client.aclose()

    return app


def serve(config_path: Path, host: str, port: int) -> None:
    import uvicorn

    config = json.loads(config_path.read_text(encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uvicorn.run(create_app(config), host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    serve(args.config, args.host, args.port)


if __name__ == "__main__":
    main()
