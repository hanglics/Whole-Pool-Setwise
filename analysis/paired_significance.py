#!/usr/bin/env python3
"""Paired effectiveness tests for declared comparison families."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def read_per_query_eval(path: Path, metric: str) -> dict[str, float]:
    values = {}
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if len(fields) < 3 or fields[0] != metric or fields[1] == "all":
                continue
            if fields[1] in values:
                raise ValueError(f"Duplicate {metric}/{fields[1]} at {path}:{line_number}")
            values[fields[1]] = float(fields[2])
    if not values:
        raise ValueError(f"No per-query rows for metric={metric!r} in {path}; evaluate with -q.")
    return values


def paired_bootstrap(values: list[float], samples: int, seed: int) -> tuple[float, float, float]:
    if not values:
        raise ValueError("Paired bootstrap requires at least one query.")
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    means.sort()
    return (
        sum(values) / len(values),
        means[int(0.025 * samples)],
        means[min(samples - 1, int(0.975 * samples))],
    )


def sign_flip_pvalue(
    values: list[float], *, samples: int, seed: int, shift: float = 0.0, two_sided: bool
) -> float:
    shifted = [value + shift for value in values]
    observed = sum(shifted) / len(shifted)
    rng = random.Random(seed)
    exceed = 0
    for _ in range(samples):
        randomized = sum(value if rng.random() < 0.5 else -value for value in shifted) / len(shifted)
        if ((abs(randomized) >= abs(observed)) if two_sided else (randomized >= observed)):
            exceed += 1
    return (exceed + 1) / (samples + 1)


def holm_adjust(pvalues: list[float]) -> list[float]:
    indexed = sorted(enumerate(pvalues), key=lambda item: item[1])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for order, (original_index, pvalue) in enumerate(indexed):
        running = max(running, min(1.0, (len(pvalues) - order) * pvalue))
        adjusted[original_index] = running
    return adjusted


def parse_spec(value: str) -> dict:
    fields = value.split("|")
    if len(fields) != 6:
        raise argparse.ArgumentTypeError(
            "comparison must be NAME|FAMILY|SYSTEM_EVAL|REFERENCE_EVAL|METRIC|DIRECTION"
        )
    name, family, system, reference, metric, direction = fields
    if direction not in {"higher", "lower"}:
        raise argparse.ArgumentTypeError("DIRECTION must be higher or lower")
    return {
        "name": name,
        "family": family,
        "system_eval": Path(system),
        "reference_eval": Path(reference),
        "metric": metric,
        "direction": direction,
    }


def analyze(specs: list[dict], samples: int, seed: int, delta: float) -> list[dict]:
    rows = []
    for index, spec in enumerate(specs):
        system = read_per_query_eval(spec["system_eval"], spec["metric"])
        reference = read_per_query_eval(spec["reference_eval"], spec["metric"])
        if set(system) != set(reference):
            raise ValueError(f"Paired qid mismatch for comparison {spec['name']!r}")
        multiplier = 1.0 if spec["direction"] == "higher" else -1.0
        differences = [multiplier * (system[qid] - reference[qid]) for qid in sorted(system)]
        mean, ci_low, ci_high = paired_bootstrap(differences, samples, seed + index)
        rows.append(
            {
                "name": spec["name"],
                "family": spec["family"],
                "metric": spec["metric"],
                "direction": spec["direction"],
                "queries": len(differences),
                "mean_difference": mean,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "randomization_p_raw": sign_flip_pvalue(
                    differences, samples=samples, seed=seed + index, two_sided=True
                ),
                "noninferiority_margin": delta,
                "noninferiority_p_raw": sign_flip_pvalue(
                    differences,
                    samples=samples,
                    seed=seed + index,
                    shift=delta,
                    two_sided=False,
                ),
                "noninferiority_ci_pass": ci_low > -delta,
                "system_eval": str(spec["system_eval"]),
                "reference_eval": str(spec["reference_eval"]),
            }
        )
    by_family = defaultdict(list)
    for idx, row in enumerate(rows):
        by_family[row["family"]].append(idx)
    for indices in by_family.values():
        superiority = holm_adjust([rows[idx]["randomization_p_raw"] for idx in indices])
        noninferiority = holm_adjust([rows[idx]["noninferiority_p_raw"] for idx in indices])
        for idx, p_superiority, p_noninferiority in zip(indices, superiority, noninferiority):
            rows[idx]["randomization_p_holm"] = p_superiority
            rows[idx]["noninferiority_p_holm"] = p_noninferiority
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", action="append", type=parse_spec, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=929)
    parser.add_argument("--noninferiority-delta", type=float, default=0.01)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    rows = analyze(args.comparison, args.samples, args.seed, args.noninferiority_delta)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with open(args.output_csv, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "samples": args.samples,
                "seed": args.seed,
                "noninferiority_delta": args.noninferiority_delta,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
