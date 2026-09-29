#!/usr/bin/env python3
"""Validate experiment artifacts and emit tidy tables plus paired statistics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path


def paired_bootstrap(differences, samples: int = 10_000, seed: int = 929):
    values = list(differences)
    if not values:
        raise ValueError("Paired bootstrap needs at least one difference.")
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    means.sort()
    return {
        "mean_difference": sum(values) / len(values),
        "ci95": [means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]],
    }


def sign_flip_pvalue(differences, samples: int = 10_000, seed: int = 929, shift: float = 0.0):
    values = [float(value) + shift for value in differences]
    observed = sum(values) / len(values)
    rng = random.Random(seed)
    exceed = 0
    for _ in range(samples):
        randomized = sum(value if rng.random() < 0.5 else -value for value in values) / len(values)
        if randomized >= observed:
            exceed += 1
    return (exceed + 1) / (samples + 1)


def holm_adjust(pvalues):
    indexed = sorted(enumerate(pvalues), key=lambda item: item[1])
    adjusted = [0.0] * len(indexed)
    running = 0.0
    total = len(indexed)
    for order, (original_index, pvalue) in enumerate(indexed):
        running = max(running, min(1.0, (total - order) * pvalue))
        adjusted[original_index] = running
    return adjusted


def validate_native_row(row):
    required = (
        "output_objective", "output_depth", "nominal_prompt_tokens", "retry_prompt_tokens",
        "actual_prompt_tokens", "nominal_completion_tokens", "retry_completion_tokens",
        "actual_completion_tokens",
        "retry_inclusive_llm_calls", "wall_seconds", "retry_seconds",
        "retry_excluded_wall_seconds",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"Native telemetry missing fields: {missing}")
    if row["actual_prompt_tokens"] != row["nominal_prompt_tokens"] + row["retry_prompt_tokens"]:
        raise ValueError("Prompt-token identity failed.")
    if row["actual_completion_tokens"] != row["nominal_completion_tokens"] + row["retry_completion_tokens"]:
        raise ValueError("Completion-token identity failed.")


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def collect(root: Path, *, include_smoke: bool = False):
    rows = []
    discrepancies = []
    for latest_path in sorted(root.rglob("LATEST")):
        if not include_smoke and any(part.startswith("smoke-q") for part in latest_path.parts):
            continue
        attempt_name = latest_path.read_text(encoding="utf-8").strip()
        if not attempt_name or Path(attempt_name).name != attempt_name:
            discrepancies.append(
                {"attempt": str(latest_path.parent), "reason": "invalid_LATEST_pointer"}
            )
            continue
        attempt = latest_path.parent / attempt_name
        manifest_path = attempt / "protocol_manifest.json"
        if not manifest_path.is_file():
            discrepancies.append({"attempt": str(attempt), "reason": "missing_manifest"})
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        telemetry_paths = sorted(attempt.glob("*_telemetry.jsonl"))
        done = (attempt / "DONE").is_file()
        if not done:
            discrepancies.append({"attempt": str(attempt), "reason": "incomplete_no_DONE"})
            continue
        if not telemetry_paths:
            discrepancies.append({"attempt": str(attempt), "reason": "missing_telemetry"})
            continue
        for telemetry_path in telemetry_paths:
            for row in read_jsonl(telemetry_path):
                if "nominal_prompt_tokens" in row:
                    validate_native_row(row)
                rows.append(
                    {
                        "attempt": str(attempt),
                        "done": done,
                        "baseline_category": manifest.get("baseline_category", "native"),
                        "dataset": row.get("dataset", manifest.get("dataset")),
                        "method": row.get("method", manifest.get("baseline")),
                        "model": row.get("model", manifest.get("model")),
                        "qid": row.get("qid"),
                        "output_objective": row.get("output_objective", manifest.get("output_objective")),
                        "output_depth": row.get("output_depth", manifest.get("output_depth")),
                        "llm_calls": row.get("llm_calls", row.get("total_calls")),
                        "retry_inclusive_llm_calls": row.get(
                            "retry_inclusive_llm_calls",
                            (row.get("llm_calls", row.get("total_calls", 0)) or 0)
                            + (row.get("retry_calls", 0) or 0),
                        ),
                        "actual_prompt_tokens": row.get("actual_prompt_tokens", row.get("prompt_tokens")),
                        "actual_completion_tokens": row.get(
                            "actual_completion_tokens", row.get("completion_tokens")
                        ),
                        "wall_seconds": row.get("wall_seconds", row.get("query_wall_seconds")),
                        "retry_seconds": row.get("retry_seconds", 0),
                        "retry_excluded_wall_seconds": row.get(
                            "retry_excluded_wall_seconds",
                            row.get("wall_seconds", row.get("query_wall_seconds")),
                        ),
                        "batch_wall_seconds": row.get("batch_wall_seconds"),
                        "status": row.get("status"),
                    }
                )
    return rows, discrepancies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/paper"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args()
    rows, discrepancies = collect(args.root, include_smoke=args.include_smoke)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "experiment_results.json").write_text(
        json.dumps({"rows": rows, "discrepancies": discrepancies}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = [
        "attempt", "done", "baseline_category", "dataset", "method", "model", "qid",
        "output_objective", "output_depth", "llm_calls", "retry_inclusive_llm_calls",
        "actual_prompt_tokens", "actual_completion_tokens", "wall_seconds", "retry_seconds",
        "retry_excluded_wall_seconds", "batch_wall_seconds", "status",
    ]
    with open(args.output_dir / "experiment_results.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    grouped = {}
    for row in rows:
        key = (
            row["baseline_category"], row["dataset"], row["method"], row["model"],
            row["output_objective"], row["output_depth"],
        )
        group = grouped.setdefault(
            key, {"queries": 0, "nominal_calls": 0, "actual_calls": 0, "tokens": 0}
        )
        if row["status"] == "ok":
            group["queries"] += 1
            group["nominal_calls"] += row["llm_calls"] or 0
            group["actual_calls"] += row["retry_inclusive_llm_calls"] or 0
            group["tokens"] += (row["actual_prompt_tokens"] or 0) + (row["actual_completion_tokens"] or 0)
    lines = ["# Experiment summary", "", "| Category | Dataset | Method | Model | Objective | Depth | Queries | Nominal calls | Actual calls | Tokens |", "|---|---|---|---|---|---:|---:|---:|---:|---:|"]
    for key, value in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {key[4]} | {key[5]} | "
            f"{value['queries']} | {value['nominal_calls']} | {value['actual_calls']} | "
            f"{value['tokens']} |"
        )
    if discrepancies:
        lines.extend(["", "## Discrepancies", ""])
        lines.extend(f"- `{item['attempt']}`: {item['reason']}" for item in discrepancies)
    (args.output_dir / "experiment_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
