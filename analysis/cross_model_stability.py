#!/usr/bin/env python3
"""Aggregate EMNLP repeated-run stability outputs across model families."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from repeated_run_stability import DEFAULT_PRIMARY_METRIC, run_stability_analysis


DEFAULT_ROOT_GLOB = Path("results/maxcontext_dualend/emnlp_phase_c_required")
DEFAULT_OUTPUT_DIR = Path("results/emnlp/analysis/cross_model_stability")
MAXCONTEXT_METHODS = {"maxctx_topdown", "dualend", "maxctx_bottomup"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        default=None,
        help="One or more stability-test-runs roots. Defaults to all EMNLP Phase C roots.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--primary-metric", default=DEFAULT_PRIMARY_METRIC)
    parser.add_argument(
        "--include-all-methods",
        action="store_true",
        help="Include any non-MaxContext stability outputs found under the roots.",
    )
    return parser.parse_args()


def discover_roots() -> list[Path]:
    if not DEFAULT_ROOT_GLOB.exists():
        return []
    return sorted(DEFAULT_ROOT_GLOB.glob("*-dl19/stability-test-runs"), key=lambda path: path.as_posix())


def family_from_model_tag(model_tag: str) -> str:
    normalized = model_tag.lower().replace(".", "-")
    if normalized.startswith("qwen3-5-"):
        return "qwen3.5"
    if normalized.startswith("meta-llama-3-1-") or normalized.startswith("llama-3-1-"):
        return "llama3.1"
    if normalized.startswith("ministral-3-") or normalized.startswith("mistral-3-"):
        return "mistral3"
    if normalized.startswith("qwen3-"):
        return "qwen3"
    return "unknown"


def model_tag_from_root(root: Path) -> str:
    model_dir = root.parent.name
    if model_dir.endswith("-dl19"):
        return model_dir[:-5]
    if model_dir.endswith("-dl20"):
        return model_dir[:-5]
    return model_dir


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def topk_sort_key(topk: str) -> int:
    match = re.fullmatch(r"top(\d+)", topk)
    if match is None:
        return 10**9
    return int(match.group(1))


def write_tex(path: Path, rows: list[dict[str, object]], primary_metric: str, topk: str) -> None:
    primary_rows = [
        row for row in rows
        if row.get("metric") == primary_metric and row.get("topk") == topk
    ]
    primary_rows.sort(key=lambda row: (
        str(row.get("model_family", "")),
        str(row.get("model", "")),
        str(row.get("method", "")),
    ))
    lines = [
        "\\begin{tabular}{lllrr}",
        "\\toprule",
        "Family & Model & Method & Mean & Std \\\\",
        "\\midrule",
    ]
    for row in primary_rows:
        mean = float(row["mean"]) if row.get("mean") not in {None, ""} else 0.0
        std = float(row["sample_std"]) if row.get("sample_std") not in {None, ""} else 0.0
        lines.append(
            f"{row['model_family']} & {row['model']} & {row['method']} & {mean:.4f} & {std:.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    roots = args.roots if args.roots is not None else discover_roots()
    if not roots:
        raise SystemExit("No EMNLP stability roots found.")

    all_rows: list[dict[str, object]] = []
    for root in roots:
        analysis = run_stability_analysis(root, primary_metric=args.primary_metric)
        model_tag = model_tag_from_root(root)
        model_family = family_from_model_tag(model_tag)
        for row in analysis["aggregate_rows"]:
            if not args.include_all_methods and row.get("method") not in MAXCONTEXT_METHODS:
                continue
            enriched = dict(row)
            enriched["model_tag"] = model_tag
            enriched["model_family"] = model_family
            all_rows.append(enriched)

    fieldnames = [
        "model_family",
        "model_tag",
        "model",
        "method_family",
        "method",
        "topk",
        "metric",
        "n_runs",
        "mean",
        "sample_std",
        "range",
        "coefficient_of_variation",
        "missing_run_ids",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "stability_summary.csv", all_rows, fieldnames)
    topks = sorted(
        {
            str(row.get("topk", ""))
            for row in all_rows
            if re.fullmatch(r"top\d+", str(row.get("topk", "")))
        },
        key=topk_sort_key,
    )
    for topk in topks:
        write_tex(args.output_dir / f"stability_summary_{topk}.tex", all_rows, args.primary_metric, topk)
    top50_path = args.output_dir / "stability_summary_top50.tex"
    if top50_path.exists():
        (args.output_dir / "stability_summary.tex").write_text(top50_path.read_text())
    print(f"Wrote {len(all_rows)} stability rows to {args.output_dir}")


if __name__ == "__main__":
    main()
