# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create and merge portable phase-action cost bundles.

This tool performs no policy training. It converts CUDA-event observations into
bounded raw samples keyed only by deployment and action shape. Runtime startup
may scale these samples with node-local anchor probes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
ACTION_ORDER = {
    "encoder": 0,
    "prefill": 1,
    "decode": 2,
    "encoder_prefill": 3,
    "encoder_decode": 4,
    "prefill_decode": 5,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _empty_capability() -> dict[str, Any]:
    return {
        "max_prefill_batch_size": 0,
        "max_decode_batch_size": 0,
        "max_encoder_batch_size": 0,
        "prefill_chunk_tokens": 0,
        "max_kv_cache_capacity": 0,
        "kv_page_tokens": 128,
        "kv_bytes_per_token": 0,
        "vision_output_bytes_per_token": 0,
        "graph_shapes": [],
    }


def _fingerprint_from_engine(args: argparse.Namespace) -> dict[str, Any]:
    engine_dir = args.engine_dir.resolve()
    config_path = engine_dir / "config.json"
    engine_path = engine_dir / "llm.engine"
    if not config_path.is_file() or not engine_path.is_file():
        raise ValueError(f"engine directory must contain config.json and llm.engine: {engine_dir}")
    config = _read_json(config_path)
    builder = config.get("builder_config", config)
    capability = _empty_capability()
    capability.update(
        {
            "max_prefill_batch_size": builder.get(
                "max_prefill_batch_size", builder.get("max_batch_size", 0)
            ),
            "max_decode_batch_size": builder.get(
                "max_decode_batch_size", builder.get("max_batch_size", 0)
            ),
            "max_encoder_batch_size": builder.get("max_encoder_batch_size", 0),
            "prefill_chunk_tokens": builder.get(
                "max_packed_prefill_chunk_tokens", builder.get("max_input_len", 0)
            ),
            "max_kv_cache_capacity": builder.get("max_kv_cache_capacity", 0),
        }
    )
    external = engine_dir / "embedding.safetensors"
    return {
        "model_hash": args.model_hash or "",
        "onnx_hash": args.onnx_hash or "",
        "engine_hash": _sha256(engine_path),
        "external_weight_hash": _sha256(external) if external.is_file() else "",
        "precision": args.precision or config.get("dtype", ""),
        "kv_dtype": config.get("kv_cache_dtype", ""),
        "capability": capability,
        "gpu": {
            "compute_capability": args.compute_capability or "",
            "sm_count": args.sm_count,
            "memory_bytes": args.memory_bytes,
            "product_name": args.gpu_name or "",
            "uuid": args.gpu_uuid or "",
        },
        "software": {
            "tensorrt": args.tensorrt or "",
            "cuda": args.cuda or "",
            "driver": args.driver or "",
            "plugin_hash": _sha256(args.plugin) if args.plugin else "",
        },
    }


def _bucket(tokens: int, bucket_tokens: int) -> int:
    return (max(0, tokens) + bucket_tokens - 1) // bucket_tokens


def _action_key(metric: dict[str, Any], context_bucket_tokens: int) -> dict[str, Any] | None:
    action = metric.get("global_action", "")
    if action not in ACTION_ORDER:
        return None
    prefill_batch = int(metric.get("prefill_batch", 0))
    decode_batch = int(metric.get("decode_batch", 0))
    if action == "prefill":
        primary_batch, secondary_batch = prefill_batch, 0
        primary_context = _bucket(int(metric.get("prefill_past_kv_max", 0)), context_bucket_tokens)
        secondary_context = 0
    elif action == "decode":
        primary_batch, secondary_batch = decode_batch, 0
        primary_context = _bucket(int(metric.get("decode_context_max", 0)), context_bucket_tokens)
        secondary_context = 0
    elif action == "prefill_decode":
        primary_batch, secondary_batch = prefill_batch, decode_batch
        primary_context = _bucket(int(metric.get("prefill_past_kv_max", 0)), context_bucket_tokens)
        secondary_context = _bucket(int(metric.get("decode_context_max", 0)), context_bucket_tokens)
    else:
        return None
    return {
        "action": action,
        "primary_batch_size": primary_batch,
        "secondary_batch_size": secondary_batch,
        "chunk_length": int(metric.get("prefill_chunk_length", 0)),
        "primary_context_bucket": primary_context,
        "secondary_context_bucket": secondary_context,
        "execution_variant": metric.get("global_execution_variant", "eager"),
        "residual_augmentation": False,
    }


def _canonical_key(key: dict[str, Any]) -> str:
    return json.dumps(key, sort_keys=True, separators=(",", ":"))


def _metric_lines(paths: Iterable[Path]) -> Iterable[tuple[str, dict[str, Any]]]:
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line in source:
                line = line.strip()
                if not line:
                    continue
                if "\t" in line:
                    prefix, payload = line.split("\t", 1)
                    if prefix not in {"PHASE_METRIC", "PHASE_ENCODER_METRIC"}:
                        continue
                else:
                    prefix, payload = "PHASE_METRIC", line
                yield prefix, json.loads(payload)


def _records_from_metrics(paths: list[Path], max_samples: int,
                          context_bucket_tokens: int = 512) -> list[dict[str, Any]]:
    observations: dict[str, deque[dict[str, float]]] = defaultdict(lambda: deque(maxlen=max_samples))
    keys: dict[str, dict[str, Any]] = {}
    timestamps: dict[str, int] = {}
    for prefix, metric in _metric_lines(paths):
        if prefix == "PHASE_ENCODER_METRIC":
            key = {
                "action": "encoder",
                "primary_batch_size": int(metric["batch_size"]),
                "secondary_batch_size": 0,
                "chunk_length": 0,
                "primary_context_bucket": _bucket(int(metric.get("input_tokens", 0)), 1024),
                "secondary_context_bucket": 0,
                "execution_variant": "eager",
                "residual_augmentation": False,
            }
            reference = makespan = float(metric["gpu_ms"])
        else:
            if not metric.get("global_decision_applied", False):
                continue
            key = _action_key(metric, context_bucket_tokens)
            if key is None:
                continue
            reference = float(metric.get("global_reference_work_ms", 0.0))
            makespan = float(metric.get("makespan_gpu_ms", 0.0))
            if reference <= 0.0 or makespan <= 0.0:
                continue
        encoded = _canonical_key(key)
        keys[encoded] = key
        timestamps[encoded] = time.time_ns()
        observations[encoded].append({"reference_work_ms": reference, "makespan_ms": makespan})
    return [
        {
            "key": keys[encoded],
            "observed_at_unix_ns": timestamps[encoded],
            "observations": list(values),
        }
        for encoded, values in sorted(
            observations.items(), key=lambda item: (ACTION_ORDER[keys[item[0]]["action"]], item[0])
        )
    ]


def _command_fingerprint(args: argparse.Namespace) -> None:
    _write_json(args.output, _fingerprint_from_engine(args))


def _command_build(args: argparse.Namespace) -> None:
    fingerprint = _read_json(args.fingerprint)
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "bundle_version": args.bundle_version,
        "source": args.source,
        "created_at_unix_ns": time.time_ns(),
        "deployment": fingerprint,
        "records": _records_from_metrics(args.metrics, args.max_samples, args.context_bucket_tokens),
    }
    if not bundle["records"]:
        raise ValueError("no action-fidelity CUDA observations found")
    _write_json(args.output, bundle)


def _shape_contract(deployment: dict[str, Any]) -> tuple[Any, ...]:
    capability = deployment["capability"]
    return (
        deployment.get("model_hash", ""),
        deployment.get("precision", ""),
        deployment.get("kv_dtype", ""),
        capability.get("max_prefill_batch_size", 0),
        capability.get("max_decode_batch_size", 0),
        capability.get("max_encoder_batch_size", 0),
        capability.get("prefill_chunk_tokens", 0),
        capability.get("max_kv_cache_capacity", 0),
    )


def _command_aggregate(args: argparse.Namespace) -> None:
    bundles = [_read_json(path) for path in args.inputs]
    target = _read_json(args.fingerprint)
    target_contract = _shape_contract(target)
    merged: dict[str, deque[dict[str, float]]] = defaultdict(lambda: deque(maxlen=args.max_samples))
    keys: dict[str, dict[str, Any]] = {}
    timestamps: dict[str, int] = {}
    for bundle in bundles:
        if bundle.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("cannot aggregate a different cost bundle schema")
        if _shape_contract(bundle["deployment"]) != target_contract:
            raise ValueError("cannot aggregate heterogeneous phase shape contracts")
        for record in bundle["records"]:
            encoded = _canonical_key(record["key"])
            keys[encoded] = record["key"]
            timestamps[encoded] = max(timestamps.get(encoded, 0), record.get("observed_at_unix_ns", 0))
            merged[encoded].extend(record["observations"])
    records = [
        {
            "key": keys[encoded],
            "observed_at_unix_ns": timestamps[encoded],
            "observations": list(values),
        }
        for encoded, values in sorted(merged.items())
    ]
    _write_json(
        args.output,
        {
            "schema_version": SCHEMA_VERSION,
            "bundle_version": args.bundle_version,
            "source": "fleet",
            "created_at_unix_ns": time.time_ns(),
            "deployment": target,
            "records": records,
        },
    )


def _fingerprint_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--model-hash")
    parser.add_argument("--onnx-hash")
    parser.add_argument("--precision")
    parser.add_argument("--compute-capability")
    parser.add_argument("--sm-count", type=int, default=0)
    parser.add_argument("--memory-bytes", type=int, default=0)
    parser.add_argument("--gpu-name")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--tensorrt")
    parser.add_argument("--cuda")
    parser.add_argument("--driver")
    parser.add_argument("--plugin", type=Path)
    parser.add_argument("--output", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    fingerprint = commands.add_parser("fingerprint")
    _fingerprint_arguments(fingerprint)
    fingerprint.set_defaults(run=_command_fingerprint)

    build = commands.add_parser("build")
    build.add_argument("--metrics", type=Path, nargs="+", required=True)
    build.add_argument("--fingerprint", type=Path, required=True)
    build.add_argument("--source", choices=("build", "node"), default="build")
    build.add_argument("--bundle-version", default="build-v1")
    build.add_argument("--max-samples", type=int, default=32)
    build.add_argument("--context-bucket-tokens", type=int, default=512)
    build.add_argument("--output", type=Path, required=True)
    build.set_defaults(run=_command_build)

    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--inputs", type=Path, nargs="+", required=True)
    aggregate.add_argument("--fingerprint", type=Path, required=True)
    aggregate.add_argument("--bundle-version", default="fleet-v1")
    aggregate.add_argument("--max-samples", type=int, default=32)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.set_defaults(run=_command_aggregate)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if getattr(args, "max_samples", 1) <= 0:
        raise ValueError("max samples must be positive")
    if getattr(args, "context_bucket_tokens", 1) <= 0:
        raise ValueError("context bucket tokens must be positive")
    args.run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
