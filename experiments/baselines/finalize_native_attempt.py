#!/usr/bin/env python3
"""Validate one native experiment attempt, write provenance, then mark DONE."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capture_provenance import runtime_snapshot
from llmrankers.experiment_controls import (
    atomic_write_json,
    atomic_write_text,
    sha256_file,
    validate_telemetry,
)


def read_first_stage_pools(
    path: Path, *, pool_cap: int, query_limit: int = 0
) -> dict[str, list[str]]:
    if pool_cap < 1:
        raise ValueError("First-stage pool cap must be positive.")
    pools: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(f"Malformed first-stage row at {path}:{line_number}")
            docs = pools.setdefault(fields[0], [])
            if len(docs) < pool_cap:
                docs.append(fields[2])
    short = {qid: len(docids) for qid, docids in pools.items() if len(docids) != pool_cap}
    if short:
        raise ValueError(f"First-stage run does not provide depth {pool_cap}: {short}")
    duplicates = {qid: len(docids) for qid, docids in pools.items() if len(set(docids)) != pool_cap}
    if duplicates:
        raise ValueError(f"First-stage top-{pool_cap} contains duplicate documents: {duplicates}")
    if query_limit:
        pools = dict(list(pools.items())[:query_limit])
    return pools


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--condition-dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-objective", required=True, choices=["top10", "full"])
    parser.add_argument("--output-depth", type=int, required=True)
    parser.add_argument("--run-path", type=Path, required=True)
    parser.add_argument("--prompt-variant", default="canonical")
    parser.add_argument("--label-scheme", default="sequential")
    parser.add_argument("--seed", type=int, default=929)
    parser.add_argument("--query-limit", type=int, default=0)
    args = parser.parse_args()
    if args.query_limit < 0:
        raise ValueError("--query-limit must be non-negative.")
    run_file = args.attempt_dir / f"{args.method}.txt"
    telemetry_file = args.attempt_dir / f"{args.method}_telemetry.jsonl"
    checkpoint_protocol = args.attempt_dir / "checkpoints" / "protocol.json"
    summary_file = args.attempt_dir / f"{args.method}_summary.json"
    metadata_file = args.attempt_dir / "experiment_metadata.json"
    for path in (run_file, telemetry_file, checkpoint_protocol, summary_file, metadata_file, args.run_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty required artifact: {path}")

    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    pool_cap = int(metadata.get("prompt_contract", {}).get("pool", 0))
    if pool_cap < args.output_depth:
        raise ValueError(
            f"Invalid launch pool/output depths: pool={pool_cap}, output={args.output_depth}"
        )
    rankings = {}
    with open(run_file, encoding="utf-8") as stream:
        for line in stream:
            qid, _, docid, *_ = line.split()
            rankings.setdefault(qid, []).append(docid)
    expected = read_first_stage_pools(
        args.run_path, pool_cap=pool_cap, query_limit=args.query_limit
    )
    successful = {}
    for line_number, line in enumerate(telemetry_file.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        row = json.loads(line)
        try:
            validate_telemetry(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid telemetry at {telemetry_file}:{line_number}: {exc}"
            ) from exc
        if row["status"] not in {"ok", "error"}:
            raise ValueError(
                f"Invalid telemetry status at {telemetry_file}:{line_number}: "
                f"{row['status']!r}"
            )
        cross_field_mismatches = {
            key: {"expected": value, "actual": row.get(key)}
            for key, value in {
                "method": args.method,
                "model": args.model,
                "dataset": args.dataset,
                "output_objective": args.output_objective,
                "output_depth": args.output_depth,
                "pool_cap": pool_cap,
            }.items()
            if row.get(key) != value
        }
        if cross_field_mismatches:
            raise ValueError(
                f"Telemetry/launch mismatch at {telemetry_file}:{line_number}: "
                f"{cross_field_mismatches}"
            )
        if row["qid"] not in expected:
            raise ValueError(
                f"Telemetry qid={row['qid']!r} is outside the selected first-stage queries."
            )
        if row.get("live_pool_size") != len(expected[row["qid"]]):
            raise ValueError(
                f"qid={row['qid']!r}: telemetry live_pool_size does not match first-stage pool."
            )
        if row["status"] == "ok":
            if row["qid"] in successful:
                raise ValueError(f"Duplicate successful telemetry qid={row['qid']!r}")
            successful[row["qid"]] = row
    if set(rankings) != set(successful) or set(rankings) != set(expected):
        raise ValueError("Run, telemetry, and first-stage qid sets differ.")
    for qid, docids in rankings.items():
        if len(docids) != args.output_depth or len(set(docids)) != args.output_depth:
            raise ValueError(f"qid={qid}: output depth/uniqueness failure")
        if not set(docids).issubset(set(expected[qid])):
            raise ValueError(f"qid={qid}: output contains a document outside the first-stage pool")
    protocol = json.loads(checkpoint_protocol.read_text(encoding="utf-8"))
    runtime = runtime_snapshot()
    source_sha = runtime["source_snapshot"]["sha256"]
    if metadata.get("source_snapshot_sha256") != source_sha:
        raise ValueError("Experiment source snapshot changed while the attempt was running.")
    if protocol.get("protocol", {}).get("metadata") != metadata:
        raise ValueError("Checkpoint protocol and experiment metadata differ.")
    if metadata.get("first_stage_sha256") != sha256_file(args.run_path):
        raise ValueError("First-stage run changed while the attempt was running.")
    if str(metadata.get("prompt_contract", {}).get("direction", "")).startswith("maxcontext_"):
        template = metadata.get("prompt_template_contract")
        if not isinstance(template, dict) or not template.get("rendered_chat_prompt"):
            raise ValueError("MaxContext attempt is missing the rendered prompt-template contract.")
        rendered_hash = hashlib.sha256(
            template["rendered_chat_prompt"].encode("utf-8")
        ).hexdigest()
        if template.get("rendered_chat_prompt_sha256") != rendered_hash:
            raise ValueError("Rendered prompt-template hash mismatch.")
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    expected_summary = {
        "queries": len(rankings),
        "output_depth": args.output_depth,
        "llm_calls": sum(int(row["llm_calls"]) for row in successful.values()),
        "actual_prompt_tokens": sum(int(row["actual_prompt_tokens"]) for row in successful.values()),
        "actual_completion_tokens": sum(int(row["actual_completion_tokens"]) for row in successful.values()),
    }
    mismatches = {
        key: {"expected": value, "actual": summary.get(key)}
        for key, value in expected_summary.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Native summary mismatch: {mismatches}")
    manifest = {
        "schema_version": 1,
        "protocol_hash": protocol["protocol_hash"],
        "method": args.method,
        "model": args.model,
        "dataset": args.dataset,
        "output_objective": args.output_objective,
        "output_depth": args.output_depth,
        "run_file": run_file.name,
        "first_stage_run": str(args.run_path.resolve()),
        "first_stage_sha256": metadata["first_stage_sha256"],
        "output_run_sha256": sha256_file(run_file),
        "telemetry_sha256": sha256_file(telemetry_file),
        "qrels_id": metadata["qrels_id"],
        "qrels_sha256": metadata["qrels_sha256"],
        "ir_dataset_name": metadata["ir_dataset_name"],
        "prompt_variant": args.prompt_variant,
        "prompt_contract": metadata["prompt_contract"],
        "prompt_contract_sha256": metadata["prompt_contract_sha256"],
        "prompt_template_contract": metadata.get("prompt_template_contract"),
        "label_scheme": args.label_scheme,
        "seed": args.seed,
        "model_revision": metadata["model_revision"],
        "tokenizer_revision": metadata["tokenizer_revision"],
        "source_snapshot": runtime["source_snapshot"],
        "git_commit": runtime["git_commit"],
        "git_dirty": runtime["git_dirty"],
        "git_status_sha256": runtime["git_status_sha256"],
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "runtime": runtime,
        "command_argv": protocol["protocol"]["run"],
        "ranker_argv": protocol["protocol"]["ranker"],
        "parallelism": {
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started_at_utc": metadata["started_at_utc"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "queries": len(rankings),
        "baseline_category": "native",
        "query_limit": args.query_limit or None,
    }
    atomic_write_json(args.attempt_dir / "protocol_manifest.json", manifest)
    atomic_write_text(args.attempt_dir / "DONE", "ok\n")
    args.condition_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(args.condition_dir / "LATEST", args.attempt_dir.name + "\n")


if __name__ == "__main__":
    main()
