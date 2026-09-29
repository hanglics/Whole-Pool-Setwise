#!/usr/bin/env python3
"""Manifest-first audit of the 108 existing BEIR experiment cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from paired_significance import analyze, read_per_query_eval


METHODS = ("maxcontext_topdown", "maxcontext_dualend")
DATASETS = (
    "beir-nfcorpus",
    "beir-touche2020",
    "beir-fiqa",
    "beir-dbpedia",
    "beir-trec-covid",
    "beir-scifact",
)
FIRST_STAGE_RUNS = {
    "beir-nfcorpus": "runs/bm25/run.beir.bm25-flat.nfcorpus.txt",
    "beir-touche2020": "runs/bm25/run.beir.bm25-flat.webis-touche2020.txt",
    "beir-fiqa": "runs/bm25/run.beir.bm25-flat.fiqa.txt",
    "beir-dbpedia": "runs/bm25/run.beir.bm25-flat.dbpedia-entity.txt",
    "beir-trec-covid": "runs/bm25/run.beir.bm25-flat.trec-covid.txt",
    "beir-scifact": "runs/bm25/run.beir.bm25-flat.scifact.txt",
}
REQUIRED_METRICS = ("ndcg_cut_10", "ndcg_cut_100", "map_cut_100")
QRELS = {
    "beir-nfcorpus": "beir-v1.0.0-nfcorpus-test",
    "beir-touche2020": "beir-v1.0.0-webis-touche2020-test",
    "beir-fiqa": "beir-v1.0.0-fiqa-test",
    "beir-dbpedia": "beir-v1.0.0-dbpedia-entity-test",
    "beir-trec-covid": "beir-v1.0.0-trec-covid-test",
    "beir-scifact": "beir-v1.0.0-scifact-test",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_expected_pools(path: Path, cap: int = 100) -> dict[str, list[str]]:
    pools: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(f"Malformed first-stage row at {path}:{line_number}")
            rows = pools.setdefault(fields[0], [])
            if len(rows) < cap:
                rows.append(fields[2])
    return pools


def audit_run(path: Path, expected_pools: dict[str, list[str]]) -> dict:
    qids = {}
    errors = []
    if not path.is_file():
        return {
            "queries": 0,
            "min_depth": 0,
            "max_depth": 0,
            "errors": ["missing_run"],
            "missing_qids": sorted(expected_pools),
            "unexpected_qids": [],
        }
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if len(fields) != 6:
                errors.append(f"malformed_line:{line_number}")
                continue
            qid, docid = fields[0], fields[2]
            qids.setdefault(qid, []).append(docid)
    for qid, docids in qids.items():
        if len(docids) != len(set(docids)):
            errors.append(f"duplicate_docids:{qid}")
        expected_docs = expected_pools.get(qid)
        if expected_docs is not None and len(docids) != len(expected_docs):
            errors.append(f"wrong_depth:{qid}:{len(docids)}:expected_{len(expected_docs)}")
        if expected_docs is not None and set(docids) != set(expected_docs):
            errors.append(f"pool_membership_mismatch:{qid}")
    missing_qids = sorted(set(expected_pools) - set(qids))
    unexpected_qids = sorted(set(qids) - set(expected_pools))
    if missing_qids:
        errors.append(f"missing_qids:{len(missing_qids)}")
    if unexpected_qids:
        errors.append(f"unexpected_qids:{len(unexpected_qids)}")
    depths = [len(docids) for docids in qids.values()]
    return {
        "queries": len(qids),
        "min_depth": min(depths, default=0),
        "max_depth": max(depths, default=0),
        "errors": errors,
        "missing_qids": missing_qids,
        "unexpected_qids": unexpected_qids,
    }


def parse_eval(path: Path) -> dict:
    metrics = {}
    if not path.is_file():
        return metrics
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[1] == "all":
            try:
                metrics[fields[0]] = float(fields[2])
            except ValueError:
                pass
    return metrics


def evaluate_bm25(run_path: Path, dataset: str, output_path: Path) -> dict:
    """Create a persisted per-query/aggregate BM25 evaluation for BEIR."""
    command = [
        sys.executable,
        "-m",
        "pyserini.eval.trec_eval",
        "-q",
        "-l",
        "1",
        "-m",
        "ndcg_cut.10,100",
        "-m",
        "map_cut.100",
        QRELS[dataset],
        str(run_path),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    if not completed.stdout.strip():
        raise ValueError(f"BM25 evaluation produced no output for {dataset}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(completed.stdout, encoding="utf-8")
    metrics = parse_eval(output_path)
    missing = [metric for metric in REQUIRED_METRICS if metric not in metrics]
    if missing:
        raise ValueError(f"BM25 evaluation for {dataset} misses metrics: {missing}")
    return {"command": command, "metrics": metrics}


def parse_fallbacks(path: Path) -> dict:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    specs = {
        "parse_fallbacks": ("Avg parse fallbacks", "average_per_query"),
        "strict_parse_fallbacks": ("Avg strict parse fallbacks", "average_per_query"),
        "unparseable_after_exhaustion_fallbacks": (
            "Avg unparseable after exhaustion fallbacks",
            "average_per_query",
        ),
        "bm25_bypasses": ("Avg BM25 bypass", "average_per_query"),
        "parse_failure_bm25_fallback": (
            "Avg parse_failure_bm25_fallback",
            "average_per_query",
        ),
        "retries_fired": ("Total retries fired", "total"),
    }
    parsed = {}
    for name, (label, aggregation) in specs.items():
        match = re.search(
            rf"{re.escape(label)}:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
            text,
            flags=re.IGNORECASE,
        )
        parsed[name] = {
            "value": float(match.group(1)) if match else (0.0 if name == "retries_fired" else None),
            "aggregation": aggregation,
        }
    return parsed


def completion_marker_path(*paths: Path) -> Path | None:
    """Prefer a complete .log; otherwise accept one complete scheduler output."""
    candidates = []
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        timing = "Avg time per query:" in text or "Avg wall-clock time per query:" in text
        counters = all(label in text for label in (
            "Avg comparisons:", "Avg prompt tokens:", "Avg completion tokens:",
        ))
        if timing and counters:
            if path.suffix == ".log":
                return path
            candidates.append(path)
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous complete execution logs: {candidates}")
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/emnlp/main/phase_b2_beir"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bm25-eval-dir", type=Path, help="Reuse saved per-query BM25 evaluations instead of invoking trec_eval.")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=929)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = sorted(path.name for path in args.root.iterdir() if path.is_dir())
    if len(models) != 9:
        raise ValueError(f"Expected 9 BEIR model directories, found {len(models)}: {models}")
    expected_pools = {
        dataset: read_expected_pools(Path(FIRST_STAGE_RUNS[dataset])) for dataset in DATASETS
    }
    bm25_evaluations = {}
    bm25_manifest_rows = []
    for dataset in DATASETS:
        run_path = Path(FIRST_STAGE_RUNS[dataset])
        eval_path = args.output_dir / "bm25_eval" / f"{dataset}.eval"
        if args.bm25_eval_dir is None:
            evaluation = evaluate_bm25(run_path, dataset, eval_path)
        else:
            source_eval = args.bm25_eval_dir / f"{dataset}.eval"
            metrics = parse_eval(source_eval)
            for metric in REQUIRED_METRICS:
                if metric not in metrics:
                    raise ValueError(f"Missing {metric} in {source_eval}")
                if set(read_per_query_eval(source_eval, metric)) != set(expected_pools[dataset]):
                    raise ValueError(f"BM25 qid mismatch for {dataset}/{metric}")
            eval_path.parent.mkdir(parents=True, exist_ok=True)
            if source_eval.resolve() != eval_path.resolve():
                eval_path.write_bytes(source_eval.read_bytes())
            evaluation = {"command": None, "metrics": metrics}
        bm25_evaluations[dataset] = {**evaluation, "eval_path": eval_path}
        depths = [len(rows) for rows in expected_pools[dataset].values()]
        bm25_manifest_rows.append(
            {
                "schema_version": 1,
                "dataset": dataset,
                "method": "bm25",
                "run": str(run_path),
                "run_sha256": sha256_file(run_path),
                "qrels_id": QRELS[dataset],
                "evaluation_command": evaluation["command"],
                "eval": str(eval_path),
                "eval_sha256": sha256_file(eval_path),
                "queries": len(expected_pools[dataset]),
                "min_depth_at_100_cap": min(depths, default=0),
                "max_depth_at_100_cap": max(depths, default=0),
                "metrics": evaluation["metrics"],
                "status": "complete",
            }
        )
    rows = []
    discrepancies = []
    for model in models:
        for dataset in DATASETS:
            for method in METHODS:
                cell = args.root / model / dataset / method / "pool100"
                run_path = cell / f"{method}.txt"
                eval_path = cell / f"{method}.eval"
                log_path = cell / f"{method}.log"
                slurm_paths = sorted(cell.glob("slurm-*.out"))
                run_audit = audit_run(run_path, expected_pools[dataset])
                metrics = parse_eval(eval_path)
                missing_metrics = [metric for metric in REQUIRED_METRICS if metric not in metrics]
                completion_path = completion_marker_path(log_path, *slurm_paths)
                completion_marker = completion_path is not None
                artifact_sha256 = {
                    name: sha256_file(path) if path.is_file() else None
                    for name, path in (("run", run_path), ("eval", eval_path), ("log", log_path))
                }
                artifact_sha256["log"] = sha256_file(completion_path) if completion_path else None
                artifact_sha256["completion"] = (
                    sha256_file(completion_path) if completion_path is not None else None
                )
                row = {
                    "model": model,
                    "dataset": dataset,
                    "method": method,
                    "cell": str(cell),
                    "run_present": run_path.is_file(),
                    "eval_present": eval_path.is_file(),
                    "log_present": completion_path is not None,
                    "log_artifact": str(completion_path) if completion_path else None,
                    "log_format": completion_path.suffix if completion_path else None,
                    "completion_marker": completion_marker,
                    "completion_artifact": str(completion_path) if completion_path else None,
                    "artifact_sha256": artifact_sha256,
                    "first_stage_run": FIRST_STAGE_RUNS[dataset],
                    "first_stage_sha256": sha256_file(Path(FIRST_STAGE_RUNS[dataset])),
                    **run_audit,
                    "metrics": metrics,
                    "missing_metrics": missing_metrics,
                    "fallbacks": parse_fallbacks(
                        completion_path
                    ) if completion_path is not None else {},
                    "status": "complete",
                }
                if (
                    not all((row["run_present"], row["eval_present"], row["log_present"]))
                    or not completion_marker
                    or missing_metrics
                    or row["errors"]
                ):
                    row["status"] = "discrepant"
                    row["discrepancy_reasons"] = [
                        *row["errors"],
                        *([] if row["run_present"] else ["missing_run"]),
                        *([] if row["eval_present"] else ["missing_eval"]),
                        *([] if row["log_present"] else ["missing_log"]),
                        *([] if completion_marker else ["missing_completion_marker"]),
                        *[f"missing_metric:{metric}" for metric in missing_metrics],
                    ]
                    discrepancies.append(row)
                else:
                    row["discrepancy_reasons"] = []
                rows.append(row)
    if len(rows) != 108:
        raise AssertionError(f"Expected 108 cells, built {len(rows)}")

    with open(args.output_dir / "beir_manifest.jsonl", "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    with open(args.output_dir / "bm25_manifest.jsonl", "w", encoding="utf-8") as stream:
        for row in bm25_manifest_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    summary_rows = []
    for row in rows:
        summary_rows.append(
            {
                "model": row["model"],
                "model_subset": "all9" if row["model"] == "qwen3-5-0-8b" else "all9;ge2b",
                "dataset": row["dataset"],
                "method": row["method"],
                "status": row["status"],
                "queries": row["queries"],
                "min_depth": row["min_depth"],
                "max_depth": row["max_depth"],
                "ndcg_cut_10": row["metrics"].get("ndcg_cut_10"),
                "ndcg_cut_100": row["metrics"].get("ndcg_cut_100"),
                "map_cut_100": row["metrics"].get("map_cut_100"),
            }
        )
    for model in models:
        for dataset in DATASETS:
            metrics = bm25_evaluations[dataset]["metrics"]
            depths = [len(rows) for rows in expected_pools[dataset].values()]
            summary_rows.append(
                {
                    "model": model,
                    "model_subset": "all9" if model == "qwen3-5-0-8b" else "all9;ge2b",
                    "dataset": dataset,
                    "method": "bm25",
                    "status": "complete",
                    "queries": len(expected_pools[dataset]),
                    "min_depth": min(depths, default=0),
                    "max_depth": max(depths, default=0),
                    "ndcg_cut_10": metrics["ndcg_cut_10"],
                    "ndcg_cut_100": metrics["ndcg_cut_100"],
                    "map_cut_100": metrics["map_cut_100"],
                }
            )
    for subset, subset_models in (
        ("macro_all9", set(models)),
        ("macro_ge2b", set(models) - {"qwen3-5-0-8b"}),
    ):
        for dataset in DATASETS:
            for method in ("bm25", *METHODS):
                members = [
                    row for row in summary_rows
                    if row["model"] in subset_models
                    and row["dataset"] == dataset
                    and row["method"] == method
                ]
                summary_rows.append(
                    {
                        "model": f"__{subset}__",
                        "model_subset": subset,
                        "dataset": dataset,
                        "method": method,
                        "status": "complete" if all(row["status"] == "complete" for row in members) else "discrepant",
                        "queries": sum(row["queries"] for row in members),
                        "min_depth": min((row["min_depth"] for row in members), default=0),
                        "max_depth": max((row["max_depth"] for row in members), default=0),
                        **{
                            metric: (
                                sum(row[metric] for row in members) / len(members)
                                if members and all(row.get(metric) is not None for row in members)
                                else None
                            )
                            for metric in ("ndcg_cut_10", "ndcg_cut_100", "map_cut_100")
                        },
                    }
                )
    with open(args.output_dir / "beir_summary.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "model", "model_subset", "dataset", "method", "status", "queries",
                "min_depth", "max_depth", "ndcg_cut_10", "ndcg_cut_100", "map_cut_100",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    with open(args.output_dir / "beir_fallbacks.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "dataset", "method", "counter", "value", "aggregation"])
        for row in rows:
            for counter, parsed in row["fallbacks"].items():
                writer.writerow(
                    [
                        row["model"], row["dataset"], row["method"], counter,
                        parsed["value"], parsed["aggregation"],
                    ]
                )
    significance_specs = []
    for model in models:
        for dataset in DATASETS:
            wp_t_eval = args.root / model / dataset / "maxcontext_topdown" / "pool100" / "maxcontext_topdown.eval"
            wp_de_eval = args.root / model / dataset / "maxcontext_dualend" / "pool100" / "maxcontext_dualend.eval"
            bm25_eval = bm25_evaluations[dataset]["eval_path"]
            for name, system_eval, reference_eval in (
                ("wp-de-vs-wp-t", wp_de_eval, wp_t_eval),
                ("wp-t-vs-bm25", wp_t_eval, bm25_eval),
                ("wp-de-vs-bm25", wp_de_eval, bm25_eval),
            ):
                significance_specs.append(
                    {
                        "name": f"{model}:{dataset}:{name}",
                        "family": f"beir:{dataset}:ndcg_cut_10",
                        "system_eval": system_eval,
                        "reference_eval": reference_eval,
                        "metric": "ndcg_cut_10",
                        "direction": "higher",
                    }
                )
    significance = analyze(
        significance_specs, samples=args.samples, seed=args.seed, delta=0.01
    )
    with open(args.output_dir / "beir_significance.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(significance[0]))
        writer.writeheader()
        writer.writerows(significance)
    lines = ["# BEIR audit", ""]
    if discrepancies:
        lines.extend(
            f"- `{row['cell']}`: {row['status']} {row['discrepancy_reasons']}"
            for row in discrepancies
        )
    else:
        lines.append("No run/eval/log presence or run-structure discrepancies found.")
    lines.extend(
        [
            "",
            "BM25 per-query evaluations correspond to the six frozen first-stage runs.",
            "Paired significance includes WP-DE vs WP-T and each reranker vs BM25.",
        ]
    )
    (args.output_dir / "beir_discrepancies.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
