#!/usr/bin/env python3
"""BEIR qrels/trec_eval gate for the paper experiments."""
from __future__ import annotations

import argparse
import csv
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASETS = [
    ("beir-dbpedia", "beir-v1.0.0-dbpedia-entity-test", "runs/bm25/run.beir.bm25-flat.dbpedia-entity.txt"),
    ("beir-nfcorpus", "beir-v1.0.0-nfcorpus-test", "runs/bm25/run.beir.bm25-flat.nfcorpus.txt"),
    ("beir-scifact", "beir-v1.0.0-scifact-test", "runs/bm25/run.beir.bm25-flat.scifact.txt"),
    ("beir-trec-covid", "beir-v1.0.0-trec-covid-test", "runs/bm25/run.beir.bm25-flat.trec-covid.txt"),
    ("beir-touche2020", "beir-v1.0.0-webis-touche2020-test", "runs/bm25/run.beir.bm25-flat.webis-touche2020.txt"),
    ("beir-fiqa", "beir-v1.0.0-fiqa-test", "runs/bm25/run.beir.bm25-flat.fiqa.txt"),
]
DATASET_TAGS = [dataset for dataset, _, _ in DATASETS]
FIELDS = "dataset,qrels_label,bm25_run,level,metric,ndcg_cut_10,exit_code,verdict,fail_reason".split(",")
NDCG_RE = re.compile(r"^ndcg_cut_10\s+all\s+([0-9]*\.?[0-9]+)$", re.MULTILINE)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASET_TAGS, default=None)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--metric", default="ndcg_cut.10")
    parser.add_argument("--bm25-runs-dir", type=Path, default=Path("runs/bm25"))
    parser.add_argument("--output", type=Path, default=Path("results/probe_beir_eval.csv"))
    parser.add_argument("--pyserini-cmd", default="python -m pyserini.eval.trec_eval")
    return parser.parse_args()
def repo_path(path):
    return path if path.is_absolute() else REPO_ROOT / path
def fail(row, reason):
    row.update({"verdict": "FAIL", "fail_reason": " ".join(str(reason).split())})
    return row
def output_excerpt(completed):
    text = completed.stderr or completed.stdout
    return " ".join(text[-200:].split()) or "<empty>"
def selected_datasets(args):
    datasets = [row for row in DATASETS if args.dataset is None or row[0] == args.dataset]
    if not datasets:
        raise SystemExit("No BEIR datasets matched the requested subset.")
    return datasets
def probe(dataset, qrels_label, bm25_run, args):
    bm25_run_path = repo_path(args.bm25_runs_dir) / Path(bm25_run).name
    row = dict.fromkeys(FIELDS, "")
    row.update({
        "dataset": dataset, "qrels_label": qrels_label, "bm25_run": str(bm25_run_path),
        "level": args.level, "metric": args.metric, "verdict": "FAIL",
    })
    if not bm25_run_path.is_file():
        return fail(row, f"bm25_run_missing: {bm25_run_path}")
    cmd = shlex.split(args.pyserini_cmd) + [
        "-q", "-l", str(args.level), "-m", args.metric, qrels_label, str(bm25_run_path),
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        row["exit_code"] = 127
        return fail(row, f"trec_eval_failed: exit_code=127 output={exc}")
    row["exit_code"] = completed.returncode
    match = NDCG_RE.search(completed.stdout)
    if match is not None:
        row["ndcg_cut_10"] = match.group(1)
    if completed.returncode != 0:
        return fail(row, f"trec_eval_failed: exit_code={completed.returncode} output={output_excerpt(completed)}")
    if match is None:
        return fail(row, f"ndcg_cut_10_missing: exit_code={completed.returncode} output={output_excerpt(completed)}")
    if float(match.group(1)) <= 0:
        return fail(row, f"ndcg_cut_10_nonpositive: exit_code={completed.returncode} output={output_excerpt(completed)}")
    row["verdict"] = "PASS"
    return row
def main():
    args = parse_args()
    rows = [probe(dataset, qrels_label, bm25_run, args) for dataset, qrels_label, bm25_run in selected_datasets(args)]
    output = repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            "{dataset}: {verdict} qrels={qrels_label} ndcg_cut_10={ndcg_cut_10} "
            "exit={exit_code}".format(**row)
        )
    return 0 if all(row["verdict"] == "PASS" for row in rows) else 1
if __name__ == "__main__":
    raise SystemExit(main())
