"""Command-line entry point for reproducible experiments."""

import argparse
import json
from pathlib import Path
from typing import List, Optional

from .environment import capture_environment
from .experiment001 import run_experiment
from .experiment002 import run_backend_matrix
from .experiment_config import expand_axes, load_matrix
from .matrix_runner import run_endpoint_matrix
from .metrics import summarize
from .mock_server import MockServerConfig, serve_forever
from .models import BenchmarkMetadata, BenchmarkResult, RequestTrace
from .online import run_online_benchmark
from .quality import load_jsonl, score_predictions
from .reporting import load_json, write_json, write_markdown
from .synthetic import generate_traces


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llms-lab")
    commands = parser.add_subparsers(dest="command", required=True)

    benchmark = commands.add_parser(
        "benchmark", help="Run a deterministic synthetic pipeline validation."
    )
    benchmark.add_argument("--requests", type=int, default=20)
    benchmark.add_argument("--seed", type=int, default=7)
    benchmark.add_argument("--ttft-slo-ms", type=float, default=250.0)
    benchmark.add_argument("--latency-slo-ms", type=float, default=2500.0)
    benchmark.add_argument("--output", type=Path, default=Path("results/synthetic.json"))
    benchmark.add_argument("--report", type=Path, default=Path("reports/synthetic.md"))

    summarize_command = commands.add_parser(
        "summarize", help="Aggregate a JSON array of raw request traces."
    )
    summarize_command.add_argument("--input", type=Path, required=True)
    summarize_command.add_argument("--wall-time-ms", type=float, required=True)
    summarize_command.add_argument("--ttft-slo-ms", type=float)
    summarize_command.add_argument("--latency-slo-ms", type=float)

    server = commands.add_parser(
        "mock-server", help="Run the controlled OpenAI-compatible SSE server."
    )
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8000)
    server.add_argument("--prefill-ms", type=float, default=20.0)
    server.add_argument("--token-delay-ms", type=float, default=5.0)
    server.add_argument("--output-tokens", type=int, default=8)
    server.add_argument("--cold-start-ms", type=float, default=0.0)
    server.add_argument("--tail-every", type=int, default=0)
    server.add_argument("--tail-delay-ms", type=float, default=0.0)

    online = commands.add_parser(
        "benchmark-online", help="Benchmark an OpenAI-compatible streaming endpoint."
    )
    online.add_argument("--base-url", required=True)
    online.add_argument("--model", required=True)
    online.add_argument("--requests", type=int, default=30)
    online.add_argument("--concurrency", type=int, default=1)
    online.add_argument("--input-tokens", type=int, default=128)
    online.add_argument("--output-tokens", type=int, default=8)
    online.add_argument("--timeout-seconds", type=float, default=30.0)
    online.add_argument("--ttft-slo-ms", type=float)
    online.add_argument("--latency-slo-ms", type=float)
    online.add_argument("--output", type=Path, default=Path("results/online.json"))
    online.add_argument("--report", type=Path, default=Path("reports/online.md"))

    experiment = commands.add_parser(
        "experiment-001", help="Run the complete controlled measurement experiment."
    )
    experiment.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/exp001_measurement_basics/results"),
    )

    experiment002 = commands.add_parser(
        "experiment-002-backend",
        help="Run the Experiment 002 matrix against one live backend.",
    )
    experiment002.add_argument("--backend", required=True)
    experiment002.add_argument("--base-url", required=True)
    experiment002.add_argument("--config", type=Path, required=True)
    experiment002.add_argument("--output-dir", type=Path, required=True)
    experiment002.add_argument("--backend-version", default="unknown")

    matrix = commands.add_parser(
        "experiment-matrix", help="Run a repeated endpoint-backed experiment matrix."
    )
    matrix.add_argument("--experiment", required=True)
    matrix.add_argument("--variant", required=True)
    matrix.add_argument("--base-url", required=True)
    matrix.add_argument("--config", type=Path, required=True)
    matrix.add_argument("--output-dir", type=Path, required=True)
    matrix.add_argument("--backend-version", default="unknown")

    quality = commands.add_parser(
        "score-quality", help="Score saved deterministic predictions."
    )
    quality.add_argument("--dataset", type=Path, required=True)
    quality.add_argument("--predictions", type=Path, required=True)
    quality.add_argument("--output", type=Path, required=True)

    gateway = commands.add_parser(
        "production-gateway", help="Run the overload-safe canary gateway."
    )
    gateway.add_argument("--config", type=Path, required=True)
    gateway.add_argument("--host", default="127.0.0.1")
    gateway.add_argument("--port", type=int, default=9000)
    return parser


def _run_benchmark(args: argparse.Namespace) -> int:
    if args.requests <= 0:
        raise SystemExit("--requests must be positive")
    traces = generate_traces(args.requests, args.seed)
    # Synthetic requests represent a serial validation run. Real concurrent
    # drivers must measure elapsed wall-clock time around the complete run.
    wall_time_ms = sum(trace.total_latency_ms for trace in traces)
    metrics = summarize(
        traces,
        wall_time_ms=wall_time_ms,
        ttft_slo_ms=args.ttft_slo_ms,
        total_latency_slo_ms=args.latency_slo_ms,
    )
    metadata = BenchmarkMetadata(
        experiment_id="pipeline-validation",
        backend="synthetic",
        model="none",
        hardware="none",
        precision="none",
        concurrency=1,
        source="synthetic",
        notes="Deterministic data for validating metrics and reports.",
    )
    result = BenchmarkResult.create(metadata, metrics, traces)
    write_json(result, args.output)
    write_markdown(result, args.report)
    print(json.dumps(metrics, indent=2))
    print(f"\nWrote raw result to {args.output}")
    print(f"Wrote report to {args.report}")
    return 0


def _run_summarize(args: argparse.Namespace) -> int:
    raw = load_json(args.input)
    if not isinstance(raw, list):
        raise SystemExit("Input must be a JSON array of request trace objects.")
    traces = [RequestTrace.from_dict(item) for item in raw]
    metrics = summarize(
        traces,
        wall_time_ms=args.wall_time_ms,
        ttft_slo_ms=args.ttft_slo_ms,
        total_latency_slo_ms=args.latency_slo_ms,
    )
    print(json.dumps(metrics, indent=2))
    return 0


def _run_online(args: argparse.Namespace) -> int:
    run = run_online_benchmark(
        args.base_url,
        args.model,
        args.requests,
        args.concurrency,
        args.input_tokens,
        args.output_tokens,
        args.timeout_seconds,
    )
    metrics = summarize(
        run.traces,
        wall_time_ms=run.wall_time_ms,
        ttft_slo_ms=args.ttft_slo_ms,
        total_latency_slo_ms=args.latency_slo_ms,
    )
    result = BenchmarkResult.create(
        BenchmarkMetadata(
            experiment_id="online-benchmark",
            backend="openai-compatible",
            model=args.model,
            hardware="unspecified",
            precision="unspecified",
            concurrency=args.concurrency,
            source="measured",
            notes="Content-event timestamps are not necessarily token timestamps on real servers.",
            environment=capture_environment(),
            workload={
                "requests": args.requests,
                "input_size_hint": args.input_tokens,
                "output_tokens_requested": args.output_tokens,
            },
        ),
        metrics,
        run.traces,
    )
    write_json(result, args.output)
    write_markdown(result, args.report)
    print(json.dumps(metrics, indent=2))
    return 0


def _run_matrix(args: argparse.Namespace) -> int:
    required = {
        "model", "model_revision", "input_size_hints", "output_tokens",
        "concurrency", "warmup_requests", "measured_requests", "repetitions",
        "timeout_seconds", "seed",
    }
    config = load_matrix(args.config, required)
    cells = expand_axes({
        "input_size_hint": config["input_size_hints"],
        "concurrency": config["concurrency"],
    })
    run_endpoint_matrix(
        args.experiment, args.variant, args.base_url, config["model"],
        config["model_revision"], config.get("precision", config.get("dtype", "unknown")),
        args.backend_version, cells, args.output_dir,
        warmup_requests=config["warmup_requests"],
        measured_requests=config["measured_requests"], repetitions=config["repetitions"],
        output_tokens=config["output_tokens"], timeout_seconds=config["timeout_seconds"],
        ttft_slo_ms=config.get("ttft_slo_ms", 1000),
        e2e_slo_ms=config.get("e2e_slo_ms", 15000), seed=config["seed"],
    )
    return 0


def _score_quality(args: argparse.Namespace) -> int:
    dataset = load_jsonl(args.dataset)
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    result = score_predictions(dataset, predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "benchmark":
        return _run_benchmark(args)
    if args.command == "summarize":
        return _run_summarize(args)
    if args.command == "mock-server":
        serve_forever(
            MockServerConfig(
                prefill_ms=args.prefill_ms,
                token_delay_ms=args.token_delay_ms,
                output_tokens=args.output_tokens,
                cold_start_ms=args.cold_start_ms,
                tail_every=args.tail_every,
                tail_delay_ms=args.tail_delay_ms,
            ),
            args.host,
            args.port,
        )
        return 0
    if args.command == "benchmark-online":
        return _run_online(args)
    if args.command == "experiment-001":
        run_experiment(args.output_dir)
        print(f"Wrote Experiment 001 results to {args.output_dir}")
        return 0
    if args.command == "experiment-002-backend":
        run_backend_matrix(
            args.backend,
            args.base_url,
            args.config,
            args.output_dir,
            args.backend_version,
        )
        print(f"Wrote {args.backend} results to {args.output_dir}")
        return 0
    if args.command == "experiment-matrix":
        return _run_matrix(args)
    if args.command == "score-quality":
        return _score_quality(args)
    if args.command == "production-gateway":
        from .production_gateway import serve

        serve(args.config, args.host, args.port)
        return 0
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
