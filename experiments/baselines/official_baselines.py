#!/usr/bin/env python3
"""Non-algorithmic adapters shared by the pinned official baseline runners."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import types
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
UPSTREAM_ROOT = HERE / ".upstreams"
MODEL_LOCK_PATH = HERE / "model_revisions.lock.json"
TOURRANK_PARSER_INPUT_POLICY = (
    "decode_api_content_without_special_tokens_then_official_parser"
)
TOURRANK_NO_DOCUMENT_POLICY = "fail_query_without_calling_official_parser"
TOURRANK_DEBUG_OUTPUT_POLICY = "attempt_local_relative_upstream_sidecar"
TOURRANK_DUPLICATE_SELECTION_POLICY = (
    "preserve_official_dictionary_scoring_without_deduplication_or_backfill"
)
TOURRANK_SELECTION_CARDINALITY_POLICY = (
    "preserve_official_parser_count_without_retry_backfill_or_rejection"
)
TOURRANK_QUERY_LIMIT_POLICY = "first_qids_in_local_first_stage_run_order"
TOURRANK_MALFORMED_ITEM_POLICY = (
    "preserve_official_valueerror_substitution_and_indexerror_ignore_semantics"
)
TOURRANK_FORMAT_REPAIR_POLICY = (
    "one_deterministic_corrective_turn_on_missing_document_then_fail"
)
TOURRANK_FORMAT_REPAIR_PROMPT_TEMPLATE = (
    "Your previous response did not follow the required output format. Even if "
    "none of the documents is perfectly relevant, select the relatively best "
    "{m} documents from the provided set. Return exactly {m} comma-separated "
    "labels, each written as Document X where X is a number from 1 through {n}. "
    "Output only those labels. Do not explain or refuse."
)


@contextmanager
def isolated_upstream_imports():
    """Force pinned upstream imports to source via an empty cache namespace."""
    previous_write_policy = sys.dont_write_bytecode
    previous_cache_prefix = sys.pycache_prefix
    with tempfile.TemporaryDirectory(prefix="wps-upstream-pycache-") as cache:
        sys.dont_write_bytecode = True
        sys.pycache_prefix = cache
        try:
            yield
        finally:
            sys.pycache_prefix = previous_cache_prefix
            sys.dont_write_bytecode = previous_write_policy


def paper_model_revisions() -> dict[str, str]:
    """Return the only open-weight model snapshots permitted for paper experiments."""
    locked = json.loads(MODEL_LOCK_PATH.read_text(encoding="utf-8"))
    if not locked or not all(
        isinstance(model, str)
        and isinstance(revision, str)
        and len(revision) == 40
        for model, revision in locked.items()
    ):
        raise ValueError(f"Invalid paper-model revision lock: {MODEL_LOCK_PATH}")
    return locked


def require_paper_model_revision(model: str, revision: str) -> None:
    """Reject an unapproved model or mutable/mismatched model snapshot."""
    locked = paper_model_revisions()
    if model not in locked:
        raise ValueError(f"Model is not in the paper model lock: {model!r}")
    if revision != locked[model]:
        raise ValueError(
            f"Revision mismatch for {model}: expected {locked[model]}, got {revision}"
        )


def sanitized_protocol_argv(argv: Sequence[str]) -> list[str]:
    """Remove execution-control flags while retaining every scientific option."""
    excluded = {"--resume", "--validate-only"}
    return [str(value) for value in argv if str(value) not in excluded]


def normalized_text_hash(text: str) -> str:
    normalized = " ".join(str(text).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def hash_ir_dataset_qrels(dataset_name: str) -> str:
    """Hash qrels canonically so manifests bind the exact evaluation judgments."""
    import ir_datasets

    rows = sorted(
        (
            str(row.query_id),
            str(row.doc_id),
            int(row.relevance),
            str(getattr(row, "iteration", "0")),
        )
        for row in ir_datasets.load(dataset_name).qrels_iter()
    )
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_trec(path: str | Path, depth: int = 100) -> OrderedDict[str, list[dict[str, Any]]]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(f"Malformed TREC row at {path}:{line_number}")
            qid, _, docid, rank, score, _ = fields
            rows = grouped.setdefault(qid, [])
            if len(rows) < depth:
                rows.append({"docid": docid, "rank": int(rank), "score": float(score)})
    for qid, rows in grouped.items():
        if len(rows) != depth or len({row["docid"] for row in rows}) != depth:
            raise ValueError(f"qid={qid}: expected {depth} unique documents, got {len(rows)}")
    return grouped


def load_tourrank_jsonl(path: str | Path) -> OrderedDict[str, dict[str, Any]]:
    records: OrderedDict[str, dict[str, Any]] = OrderedDict()
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            hits = record.get("hits", [])
            qids = {str(hit["qid"]) for hit in hits}
            if len(qids) != 1:
                raise ValueError(f"TourRank row {line_number} has inconsistent qids.")
            qid = qids.pop()
            if qid in records:
                raise ValueError(f"Duplicate TourRank qid={qid}")
            if len(hits) != 100 or len({str(hit["docid"]) for hit in hits}) != 100:
                raise ValueError(f"TourRank qid={qid} does not contain 100 unique hits.")
            records[qid] = record
    return records


def validate_ordered_pools(
    official: Mapping[str, Sequence[Mapping[str, Any]]],
    local: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    score_tolerance: float = 0.0,
) -> list[dict[str, Any]]:
    discrepancies = []
    for qid in sorted(set(official) | set(local)):
        if qid not in official or qid not in local:
            discrepancies.append({"qid": qid, "field": "qid_presence"})
            continue
        official_rows, local_rows = official[qid], local[qid]
        if len(official_rows) != len(local_rows):
            discrepancies.append(
                {"qid": qid, "field": "depth", "official": len(official_rows), "local": len(local_rows)}
            )
            continue
        for index, (left, right) in enumerate(zip(official_rows, local_rows), start=1):
            if str(left["docid"]) != str(right["docid"]):
                discrepancies.append(
                    {"qid": qid, "rank": index, "field": "docid", "official": left["docid"], "local": right["docid"]}
                )
            if "rank" in left or "rank" in right:
                try:
                    official_rank = int(left["rank"])
                    local_rank = int(right["rank"])
                except (KeyError, TypeError, ValueError):
                    official_rank = left.get("rank")
                    local_rank = right.get("rank")
                if official_rank != index or local_rank != index:
                    discrepancies.append(
                        {
                            "qid": qid,
                            "position": index,
                            "field": "rank",
                            "expected": index,
                            "official": official_rank,
                            "local": local_rank,
                        }
                    )
            if "score" in left and "score" in right and abs(float(left["score"]) - float(right["score"])) > score_tolerance:
                discrepancies.append(
                    {"qid": qid, "rank": index, "field": "score", "official": left["score"], "local": right["score"]}
                )
            if "content" in left and "content" in right:
                official_hash = normalized_text_hash(left["content"])
                local_hash = normalized_text_hash(right["content"])
                if official_hash != local_hash:
                    discrepancies.append(
                        {
                            "qid": qid,
                            "rank": index,
                            "field": "content",
                            "official_hash": official_hash,
                            "local_hash": local_hash,
                        }
                    )
    return discrepancies


def tourrank_pools(records: Mapping[str, Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {qid: list(record["hits"]) for qid, record in records.items()}


def write_trec(path: str | Path, rankings: Mapping[str, Sequence[str]], tag: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as stream:
        for qid, docids in rankings.items():
            if len(set(docids)) != len(docids):
                raise ValueError(f"Duplicate output document for qid={qid}")
            for rank, docid in enumerate(docids, start=1):
                stream.write(f"{qid}\tQ0\t{docid}\t{rank}\t{-rank}\t{tag}\n")


def import_file(module_name: str, path: str | Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    with isolated_upstream_imports():
        spec.loader.exec_module(module)
    return module


def load_tourrank_module():
    """Load TourRank without requiring or initializing an external API client."""
    stub = types.ModuleType("openai")

    class DisabledOpenAI:
        def __init__(self, *_args, **_kwargs):
            pass

        def __getattr__(self, name):
            raise RuntimeError(f"External OpenAI client access is disabled: {name}")

    stub.OpenAI = DisabledOpenAI
    stub.InternalServerError = RuntimeError
    stub.RateLimitError = RuntimeError
    previous = sys.modules.get("openai")
    sys.modules["openai"] = stub
    try:
        return import_file(
            "wps_official_tourrank",
            UPSTREAM_ROOT / "tourrank" / "TourRank_multiprocessing.py",
        )
    finally:
        if previous is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = previous


def add_fullrank_to_path() -> Path:
    checkout = UPSTREAM_ROOT / "liu_fullrank"
    resolved = str(checkout.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    return checkout


def adaptation_manifest(baseline: str, commit: str, adaptations: Iterable[dict[str, Any]]) -> dict[str, Any]:
    adaptations = list(adaptations)
    algorithmic_fields = {"ranking_logic", "prompt_text", "parsing", "generation"}
    changed = [item for item in adaptations if item.get("field") in algorithmic_fields and item.get("changed")]
    return {
        "schema_version": 1,
        "baseline": baseline,
        "upstream_commit": commit,
        "adaptations": adaptations,
        "faithful": not changed,
    }


def ordered_score_ranking(module, docids: Sequence[str], score_dicts: Sequence[Mapping[str, int]]) -> list[str]:
    aggregate = OrderedDict((docid, 0) for docid in docids)
    for score_dict in score_dicts:
        for docid, score in score_dict.items():
            aggregate[docid] += score
    # Delegate stable descending sorting to the upstream helper.
    return module.sort_docs_by_relevance(list(aggregate), list(aggregate.values()))
