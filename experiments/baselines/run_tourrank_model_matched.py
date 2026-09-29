#!/usr/bin/env python3
"""Run pinned TourRank logic with an paper open-model backend."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capture_provenance import source_snapshot
from llmrankers.experiment_controls import (
    append_jsonl,
    atomic_write_json,
    protocol_hash,
    sha256_file,
)
from official_baselines import (
    HERE,
    UPSTREAM_ROOT,
    adaptation_manifest,
    hash_ir_dataset_qrels,
    load_tourrank_jsonl,
    load_tourrank_module,
    load_trec,
    ordered_score_ranking,
    paper_model_revisions,
    require_paper_model_revision,
    sanitized_protocol_argv,
    TOURRANK_DEBUG_OUTPUT_POLICY,
    TOURRANK_DUPLICATE_SELECTION_POLICY,
    TOURRANK_FORMAT_REPAIR_POLICY,
    TOURRANK_FORMAT_REPAIR_PROMPT_TEMPLATE,
    TOURRANK_MALFORMED_ITEM_POLICY,
    TOURRANK_NO_DOCUMENT_POLICY,
    TOURRANK_PARSER_INPUT_POLICY,
    TOURRANK_QUERY_LIMIT_POLICY,
    TOURRANK_SELECTION_CARDINALITY_POLICY,
    tourrank_pools,
    validate_ordered_pools,
    write_trec,
)
from open_model_backend import OpenModelBackend


UPSTREAM_COMMIT = "bab7db3510eb9a0b7c78112ce7fe002a36144ea8"
EXPECTED_CALLS_PER_TOURNAMENT = 13
GENERATION_BUDGET = 512
PASSAGE_LENGTH = 512
MAX_FORMAT_RETRIES = 1
PAPER_MODELS = frozenset(paper_model_revisions())


class TourRankFormatError(ValueError):
    """The local model did not produce an official-parser-shaped response."""


@contextmanager
def isolated_upstream_workdir(path: Path):
    """Contain the pinned parser's relative debug sidecar in one attempt."""
    previous = Path.cwd()
    target = path.resolve()
    os.chdir(target)
    try:
        yield
    finally:
        os.chdir(previous)


class _TeeStdout:
    """Capture pinned-parser diagnostics without hiding them from the job log."""

    def __init__(self, original):
        self.original = original
        self.capture = io.StringIO()

    def write(self, value):
        self.capture.write(value)
        return self.original.write(value)

    def flush(self):
        self.capture.flush()
        return self.original.flush()


def parser_debug_artifact(path: Path) -> dict:
    payload = path.read_bytes() if path.is_file() else b""
    text = payload.decode("utf-8", errors="replace")
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "entries": sum(line == "New Error: " for line in text.splitlines()),
    }


class InlineManager:
    """Run the official manager-list code path without GPU multiprocessing."""

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    @staticmethod
    def list(values=None):
        return list(values or [])


class InlineProcess:
    """Execute the official Process target inline so one local model owns the GPU."""

    def __init__(self, target, args):
        self.target = target
        self.args = args
        self.exitcode = None

    def start(self):
        try:
            self.target(*self.args)
        except Exception:
            self.exitcode = 1
            raise
        self.exitcode = 0

    def join(self):
        return None


def infer_prompt_shape(messages) -> tuple[int | None, int | None]:
    if len(messages) < 2:
        return None, None
    match = re.search(
        r"query and (\d+) documents.*select the (\d+) documents",
        messages[1]["content"],
        re.IGNORECASE | re.DOTALL,
    )
    return (int(match.group(1)), int(match.group(2))) if match else (None, None)


def prepare_tourrank_parser_input(
    generated: dict, n_selected: int | None
) -> tuple[str, dict]:
    """Adapt local tokenizer output to the official API-content boundary.

    Raw tokenizer text remains immutable telemetry. TourRank's released parser
    consumed OpenAI ``message.content``, which never includes tokenizer control
    tokens, so it receives the special-token-free decode. A response with no
    ``Document`` line is never passed to the parser or converted into a
    synthetic ranking; the responder may issue the single declared corrective
    turn before treating a second such response as fatal.
    """
    raw_output = generated.get("raw_output")
    content_output = generated.get("content_output")
    if not isinstance(raw_output, str) or not isinstance(content_output, str):
        raise TypeError("TourRank backend must return string raw/content outputs.")
    if not isinstance(n_selected, int) or n_selected < 1:
        raise ValueError("TourRank could not infer the requested selection count.")

    has_document_line = any(
        "Document" in line for line in content_output.splitlines()
    )
    return content_output, {
        "special_tokens_removed": raw_output != content_output,
        "format_valid": has_document_line,
        "parser_input_policy": TOURRANK_PARSER_INPUT_POLICY,
        "no_document_policy": TOURRANK_NO_DOCUMENT_POLICY,
    }


def format_repair_messages(
    messages: list[dict[str, str]], previous_output: str, n: int, m: int
) -> list[dict[str, str]]:
    """Append one format-only correction without changing candidates or query."""
    return [
        *copy.deepcopy(messages),
        {"role": "assistant", "content": previous_output},
        {
            "role": "user",
            "content": TOURRANK_FORMAT_REPAIR_PROMPT_TEMPLATE.format(n=n, m=m),
        },
    ]


class InstrumentedResponder:
    def __init__(self, backend: OpenModelBackend):
        self.backend = backend
        self.tournament = 0
        self.request_index = 0
        self.events: list[dict] = []

    def begin_tournament(self, tournament: int) -> None:
        self.tournament = tournament
        self.request_index = 0

    def record_parser_outcome(
        self,
        *,
        n: int,
        m: int,
        selected,
        captured_stdout: str,
        debug_delta: bytes,
        error: Exception | None = None,
    ) -> dict:
        if not self.events:
            raise RuntimeError("TourRank parser outcome has no generation event.")
        event = self.events[-1]
        if event.get("parser_status") is not None:
            raise RuntimeError("TourRank generation received multiple parser outcomes.")
        if event.get("n") != n or event.get("m") != m:
            raise RuntimeError("TourRank parser shape differs from its generation prompt.")
        event.update(
            {
                "parser_status": "failed" if error is not None else "ok",
                "parser_value_error_fallbacks": captured_stdout.count(
                    "ValueError occured in score. (get_top_M)"
                ),
                "parser_index_error_fallbacks": captured_stdout.count(
                    "IndexError occured in doc. (get_top_M)"
                ),
                "parser_debug_delta_sha256": hashlib.sha256(debug_delta).hexdigest(),
                "parser_debug_delta_bytes": len(debug_delta),
                "parser_selected_count": len(selected) if selected is not None else None,
                "parser_unique_selected_count": (
                    len(set(map(str, selected))) if selected is not None else None
                ),
            }
        )
        if selected is not None:
            event["parser_duplicate_selection_slots"] = (
                event["parser_selected_count"]
                - event["parser_unique_selected_count"]
            )
            event["parser_selection_unique"] = (
                event["parser_duplicate_selection_slots"] == 0
            )
        else:
            event["parser_duplicate_selection_slots"] = None
            event["parser_selection_unique"] = None
        event["duplicate_selection_policy"] = TOURRANK_DUPLICATE_SELECTION_POLICY
        if selected is not None:
            count_delta = event["parser_selected_count"] - m
            event["parser_selection_count_delta"] = count_delta
            event["parser_under_selection_slots"] = max(-count_delta, 0)
            event["parser_over_selection_slots"] = max(count_delta, 0)
            event["parser_selection_count_matches_requested"] = count_delta == 0
        else:
            event["parser_selection_count_delta"] = None
            event["parser_under_selection_slots"] = None
            event["parser_over_selection_slots"] = None
            event["parser_selection_count_matches_requested"] = None
        event["selection_cardinality_policy"] = (
            TOURRANK_SELECTION_CARDINALITY_POLICY
        )
        event["malformed_item_policy"] = TOURRANK_MALFORMED_ITEM_POLICY
        if error is not None:
            event["parser_exception_class"] = type(error).__name__
            event["parser_exception_message"] = str(error)
        event["parser_contract_valid"] = (
            error is None
            and event["parser_selected_count"] is not None
        )
        return event

    def _event_from_attempts(
        self,
        *,
        n_docs: int | None,
        n_selected: int | None,
        started: float,
        attempts: list[dict],
        retry_calls: int,
        status: str,
        error: Exception | None = None,
    ) -> dict:
        event = {
            "tournament": self.tournament,
            "request_index": self.request_index,
            "n": n_docs,
            "m": n_selected,
            "started_at": started,
            "ended_at": time.time(),
            "duration_seconds": sum(
                float(attempt["duration_seconds"]) for attempt in attempts
            ),
            "prompt_tokens": sum(int(attempt["prompt_tokens"]) for attempt in attempts),
            "completion_tokens": sum(
                int(attempt["completion_tokens"]) for attempt in attempts
            ),
            "generation_budget": GENERATION_BUDGET,
            "generation_attempt_count": len(attempts),
            "generation_attempts": copy.deepcopy(attempts),
            "retry_calls": retry_calls,
            "retry_seconds": sum(
                float(attempt["duration_seconds"]) for attempt in attempts[1:]
            ),
            "format_repair_triggered": retry_calls > 0,
            "format_repair_succeeded": status == "ok" and retry_calls > 0,
            "format_repair_policy": TOURRANK_FORMAT_REPAIR_POLICY,
            "initial_format_valid": (
                attempts[0].get("format_valid") if attempts else None
            ),
            "status": status,
        }
        if attempts:
            final_attempt = attempts[-1]
            for field in (
                "rendered_prompt_sha256",
                "raw_output",
                "content_output",
                "parser_input",
                "special_tokens_removed",
                "format_valid",
                "parser_input_policy",
                "no_document_policy",
            ):
                event[field] = final_attempt.get(field)
        if error is not None:
            event["exception_class"] = type(error).__name__
            event["exception_message"] = str(error)
        return event

    @staticmethod
    def _generation_attempt(
        generated: dict,
        parser_input: str,
        parser_metadata: dict,
        *,
        attempt_index: int,
    ) -> dict:
        return {
            "attempt_index": attempt_index,
            "attempt_kind": "initial" if attempt_index == 0 else "format_repair",
            "status": "ok" if parser_metadata["format_valid"] else "format_invalid",
            "duration_seconds": generated["wall_seconds"],
            "prompt_tokens": generated["prompt_tokens"],
            "completion_tokens": generated["completion_tokens"],
            "generation_budget": GENERATION_BUDGET,
            "rendered_prompt_sha256": generated["rendered_prompt_sha256"],
            "raw_output": generated["raw_output"],
            "content_output": generated["content_output"],
            "parser_input": parser_input,
            **parser_metadata,
        }

    def __call__(self, messages):
        n_docs, n_selected = infer_prompt_shape(messages)
        started = time.time()
        attempts: list[dict] = []
        active_messages = copy.deepcopy(messages)
        retry_calls = 0

        for attempt_index in range(MAX_FORMAT_RETRIES + 1):
            if attempt_index:
                retry_calls += 1
            try:
                generated = self.backend.generate(
                    active_messages, max_new_tokens=GENERATION_BUDGET
                )
                parser_input, parser_metadata = prepare_tourrank_parser_input(
                    generated, n_selected
                )
            except Exception as exc:
                event = self._event_from_attempts(
                    n_docs=n_docs,
                    n_selected=n_selected,
                    started=started,
                    attempts=attempts,
                    retry_calls=retry_calls,
                    status="failed",
                    error=exc,
                )
                self.events.append(event)
                raise

            attempts.append(
                self._generation_attempt(
                    generated,
                    parser_input,
                    parser_metadata,
                    attempt_index=attempt_index,
                )
            )
            if parser_metadata["format_valid"]:
                event = self._event_from_attempts(
                    n_docs=n_docs,
                    n_selected=n_selected,
                    started=started,
                    attempts=attempts,
                    retry_calls=retry_calls,
                    status="ok",
                )
                self.request_index += 1
                self.events.append(event)
                return parser_input

            if attempt_index < MAX_FORMAT_RETRIES:
                active_messages = format_repair_messages(
                    list(messages), parser_input, n_docs, n_selected
                )
                continue

            error = TourRankFormatError(
                "TourRank response still contains no Document line after one "
                "bounded format-repair generation; execution does not synthesize "
                "a ranking."
            )
            event = self._event_from_attempts(
                n_docs=n_docs,
                n_selected=n_selected,
                started=started,
                attempts=attempts,
                retry_calls=retry_calls,
                status="failed",
                error=error,
            )
            self.events.append(event)
            raise error

        raise AssertionError("unreachable TourRank format-repair loop")


class InstrumentedOfficialParser:
    """Call the exact pinned parser while observing its fallback side effects."""

    def __init__(self, parser, responder: InstrumentedResponder, debug_path: Path):
        self.parser = parser
        self.responder = responder
        self.debug_path = debug_path

    def __call__(self, answer, N=10, M=5, groups_docid=[]):
        before = self.debug_path.stat().st_size if self.debug_path.is_file() else 0
        tee = _TeeStdout(sys.stdout)
        try:
            with redirect_stdout(tee):
                selected = self.parser(
                    answer, N=N, M=M, groups_docid=groups_docid
                )
        except Exception as exc:
            payload = self.debug_path.read_bytes() if self.debug_path.is_file() else b""
            self.responder.record_parser_outcome(
                n=N,
                m=M,
                selected=None,
                captured_stdout=tee.capture.getvalue(),
                debug_delta=payload[before:],
                error=exc,
            )
            raise
        payload = self.debug_path.read_bytes() if self.debug_path.is_file() else b""
        self.responder.record_parser_outcome(
            n=N,
            m=M,
            selected=selected,
            captured_stdout=tee.capture.getvalue(),
            debug_delta=payload[before:],
        )
        return selected


def tourrank_seed(base_seed: int, qid: str, tournament: int) -> int:
    payload = f"{base_seed}\0{qid}\0{tournament}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def record_tourrank_failure(
    path: Path,
    *,
    qid: str,
    model: str,
    protocol_hash_value: str,
    events: list[dict],
    wall_seconds: float,
    error: Exception,
) -> None:
    append_jsonl(
        path,
        {
            "schema_version": 2,
            "qid": qid,
            "method": "TourRank-2 model-matched",
            "model": model,
            "protocol_hash": protocol_hash_value,
            "status": "failed",
            "query_wall_seconds": wall_seconds,
            "events": events,
            "exception_class": type(error).__name__,
            "exception_message": str(error),
            "occurred_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def annotate_stages(events: list[dict]) -> None:
    expected = [(20, 10)] * 5 + [(10, 4)] * 5 + [(20, 10), (10, 5), (5, 2)]
    stage_names = ["100_to_50"] * 5 + ["50_to_20"] * 5 + [
        "20_to_10", "10_to_5", "5_to_2"
    ]
    for tournament in sorted({event["tournament"] for event in events}):
        rows = sorted(
            (event for event in events if event["tournament"] == tournament),
            key=lambda event: event["request_index"],
        )
        if [(row["n"], row["m"]) for row in rows] != expected:
            raise RuntimeError(f"TourRank schedule mismatch for tournament {tournament}")
        for row, stage in zip(rows, stage_names):
            row["stage"] = stage


def build_telemetry(
    qid: str,
    model: str,
    tournaments: int,
    events: list[dict],
    wall_seconds: float,
) -> dict:
    annotate_stages(events)
    retry_calls = sum(int(event.get("retry_calls", 0)) for event in events)
    retry_seconds = sum(float(event.get("retry_seconds", 0)) for event in events)
    return {
        "schema_version": 2,
        "qid": qid,
        "method": f"TourRank-{tournaments} model-matched",
        "model": model,
        "output_objective": "top10",
        "output_depth": 10,
        "tournaments": tournaments,
        "nominal_calls": len(events),
        "total_calls": len(events),
        "retry_calls": retry_calls,
        "retry_inclusive_llm_calls": len(events) + retry_calls,
        "generation_attempt_count": len(events) + retry_calls,
        "prompt_tokens": sum(event["prompt_tokens"] for event in events),
        "completion_tokens": sum(event["completion_tokens"] for event in events),
        "request_seconds_sum": sum(event["duration_seconds"] for event in events),
        "query_wall_seconds": wall_seconds,
        "retry_seconds": retry_seconds,
        "retry_excluded_wall_seconds": max(wall_seconds - retry_seconds, 0.0),
        "max_concurrent_requests": 1,
        "execution_mode": "serial_single_gpu_backend_adaptation",
        "official_parallel_latency_comparable": False,
        "special_token_stripped_calls": sum(
            bool(event.get("special_tokens_removed")) for event in events
        ),
        "special_token_stripped_generations": sum(
            bool(attempt.get("special_tokens_removed"))
            for event in events
            for attempt in event.get("generation_attempts", [])
        ),
        "format_failure_calls": sum(
            not bool(event.get("format_valid")) for event in events
        ),
        "initial_format_failure_calls": sum(
            event.get("initial_format_valid") is False for event in events
        ),
        "format_repair_calls": retry_calls,
        "format_repair_successes": sum(
            bool(event.get("format_repair_succeeded")) for event in events
        ),
        "parser_value_error_fallbacks": sum(
            int(event.get("parser_value_error_fallbacks", 0)) for event in events
        ),
        "parser_index_error_fallbacks": sum(
            int(event.get("parser_index_error_fallbacks", 0)) for event in events
        ),
        "parser_duplicate_selection_calls": sum(
            int(event.get("parser_duplicate_selection_slots", 0) > 0)
            for event in events
        ),
        "parser_duplicate_selection_slots": sum(
            int(event.get("parser_duplicate_selection_slots", 0))
            for event in events
        ),
        "parser_under_selection_calls": sum(
            int(event.get("parser_under_selection_slots", 0) > 0)
            for event in events
        ),
        "parser_under_selection_slots": sum(
            int(event.get("parser_under_selection_slots", 0)) for event in events
        ),
        "parser_over_selection_calls": sum(
            int(event.get("parser_over_selection_slots", 0) > 0)
            for event in events
        ),
        "parser_over_selection_slots": sum(
            int(event.get("parser_over_selection_slots", 0)) for event in events
        ),
        "parser_input_policy": TOURRANK_PARSER_INPUT_POLICY,
        "no_document_policy": TOURRANK_NO_DOCUMENT_POLICY,
        "debug_output_policy": TOURRANK_DEBUG_OUTPUT_POLICY,
        "duplicate_selection_policy": TOURRANK_DUPLICATE_SELECTION_POLICY,
        "selection_cardinality_policy": TOURRANK_SELECTION_CARDINALITY_POLICY,
        "malformed_item_policy": TOURRANK_MALFORMED_ITEM_POLICY,
        "format_repair_policy": TOURRANK_FORMAT_REPAIR_POLICY,
        "events": events,
        "status": "ok",
    }


def local_tourrank_records(dataset_name: str, run_path: Path) -> tuple[dict, dict]:
    import ir_datasets

    dataset = ir_datasets.load(dataset_name)
    queries = {str(query.query_id): query.text for query in dataset.queries_iter()}
    docstore = dataset.docs_store()
    pools = load_trec(run_path, 100)
    local = {}
    for qid, rows in pools.items():
        output = []
        for row in rows:
            doc = docstore.get(row["docid"])
            text = doc.text
            if hasattr(doc, "title") and doc.title:
                text = f"{doc.title} {text}"
            output.append({**row, "content": text})
        local[qid] = output
    return queries, local


def use_local_passages(
    official_records: dict, local_pools: dict
) -> dict:
    """Keep the official schedule inputs but substitute our model-facing text."""
    records = {}
    for qid, official_record in official_records.items():
        local_rows = local_pools[qid]
        hits = []
        for official_hit, local_row in zip(official_record["hits"], local_rows):
            hits.append({**official_hit, "content": local_row["content"]})
        records[qid] = {"query": official_record["query"], "hits": hits}
    return records


def inference_records_sha256(records: dict) -> str:
    digest = hashlib.sha256()
    for qid in sorted(records):
        payload = {"qid": qid, **records[qid]}
        digest.update(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def select_query_limit(
    records: dict, max_queries: int | None, qid_order: list[str] | None = None
) -> dict:
    if max_queries is None:
        return records
    qids = list(records) if qid_order is None else [str(qid) for qid in qid_order]
    selected_qids = qids[:max_queries]
    missing = [qid for qid in selected_qids if qid not in records]
    if missing:
        raise ValueError(f"Query-limit order contains unknown qids: {missing}")
    if len(selected_qids) != max_queries:
        raise ValueError(
            f"Query-limit requested {max_queries} qids, found {len(selected_qids)}."
        )
    return {qid: records[qid] for qid in selected_qids}


def truncate_local_passages(
    records: dict, ranker, passage_length: int
) -> tuple[dict, dict]:
    """Apply the same active-tokenizer passage cap as the native experiments."""
    truncated = copy.deepcopy(records)
    passage_count = 0
    truncated_passage_count = 0
    max_untruncated_passage_tokens = 0
    for record in truncated.values():
        for hit in record["hits"]:
            passage_count += 1
            token_count = len(ranker.tokenizer.tokenize(hit["content"]))
            max_untruncated_passage_tokens = max(
                max_untruncated_passage_tokens, token_count
            )
            if token_count > passage_length:
                truncated_passage_count += 1
            hit["content"] = ranker.truncate(hit["content"], passage_length)
    return truncated, {
        "passage_count": passage_count,
        "truncated_passage_count": truncated_passage_count,
        "max_untruncated_passage_tokens": max_untruncated_passage_tokens,
        "passage_length": passage_length,
        "truncation_method": (
            "active_tokenizer_tokenize_then_convert_tokens_to_string"
        ),
    }


def verify_input(
    dataset_tag: str, official_records: dict, local_run: Path, discrepancy_path: Path
) -> tuple[dict, dict, list[dict]]:
    names = {
        "dl19": "msmarco-passage/trec-dl-2019/judged",
        "dl20": "msmarco-passage/trec-dl-2020/judged",
    }
    queries, local = local_tourrank_records(names[dataset_tag], local_run)
    discrepancies = validate_ordered_pools(
        tourrank_pools(official_records), local, score_tolerance=1e-6
    )
    for qid, record in official_records.items():
        if qid not in queries or record["query"] != queries[qid]:
            discrepancies.append({"qid": qid, "field": "query"})
    fatal = [row for row in discrepancies if row["field"] != "content"]
    if fatal:
        raise ValueError(f"TourRank input fidelity failed; see {discrepancy_path}")
    inference_records = use_local_passages(official_records, local)
    candidate_count = sum(len(record["hits"]) for record in official_records.values())
    content_discrepancies = sum(
        row["field"] == "content" for row in discrepancies
    )
    validation = {
        "discrepancies": len(discrepancies),
        "content_discrepancies": content_discrepancies,
        "content_discrepancy_rate": (
            content_discrepancies / candidate_count if candidate_count else 0.0
        ),
        "structural_discrepancies": len(fatal),
        "candidate_count": candidate_count,
        "preflight_query_count": len(official_records),
        "preflight_candidate_count": candidate_count,
        "official_json_used_for_inference": False,
        "local_ir_datasets_content_used_for_inference": True,
        "passage_source": names[dataset_tag],
        "local_untruncated_input_sha256": inference_records_sha256(inference_records),
    }
    return validation, inference_records, discrepancies


def persist_discrepancy_report(path: Path, discrepancies: list[dict]) -> None:
    """Write once, or prove an existing attempt report is byte-equivalent data."""
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != discrepancies:
            raise ValueError(
                f"TourRank discrepancy report drift at {path}; use a new attempt."
            )
        return
    atomic_write_json(path, discrepancies)


def assert_validate_only_output_safe(output_dir: Path) -> None:
    if (output_dir / "checkpoints" / "protocol.json").exists():
        raise ValueError(
            "--validate-only refuses an experiment attempt directory; "
            "use a dedicated preflight directory."
        )


def checkpoint_path(root: Path, qid: str) -> Path:
    digest = hashlib.sha256(qid.encode("utf-8")).hexdigest()[:16]
    return root / f"query-{digest}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["dl19", "dl20"])
    parser.add_argument("--local-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=sorted(PAPER_MODELS))
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--tournaments", type=int, default=2, choices=[2])
    parser.add_argument("--seed", type=int, default=929)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()
    if args.max_queries is not None and args.max_queries < 1:
        raise ValueError("--max-queries must be positive when provided.")
    require_paper_model_revision(args.model, args.model_revision)

    subprocess.run(
        [sys.executable, str(HERE / "fetch_official_baselines.py"), "--baseline", "tourrank", "--verify-only"],
        check=True,
    )
    source = UPSTREAM_ROOT / "tourrank" / "data" / f"bm25_{args.dataset}_top100.jsonl"
    official_records = load_tourrank_jsonl(source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    discrepancy_path = args.output_dir / "input_discrepancies.json"
    input_validation, records, discrepancies = verify_input(
        args.dataset,
        official_records,
        args.local_run,
        discrepancy_path,
    )
    if args.validate_only:
        assert_validate_only_output_safe(args.output_dir)
        persist_discrepancy_report(discrepancy_path, discrepancies)
        print(
            f"Validated {len(records)} TourRank queries; "
            f"content discrepancies={input_validation['content_discrepancies']}; "
            "local ir_datasets passage text selected for inference."
        )
        return
    local_qid_order = list(load_trec(args.local_run, 100))
    records = select_query_limit(records, args.max_queries, local_qid_order)

    backend = OpenModelBackend(
        args.model, args.model_revision, cache_dir=args.cache_dir, device="cuda"
    )
    records, passage_truncation = truncate_local_passages(
        records, backend.ranker, PASSAGE_LENGTH
    )
    backend_contract = backend.contract()
    checkpoint_root = args.output_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    protocol_path = checkpoint_root / "protocol.json"
    protocol = {
        "schema_version": 2,
        "baseline": "TourRank model-matched adaptation",
        "upstream_commit": UPSTREAM_COMMIT,
        "dataset": args.dataset,
        "model": args.model,
        "model_revision": args.model_revision,
        "tokenizer_revision": args.model_revision,
        "backend_contract": backend_contract,
        "tournaments": args.tournaments,
        "generation_budget": GENERATION_BUDGET,
        "parser_input_policy": TOURRANK_PARSER_INPUT_POLICY,
        "no_document_policy": TOURRANK_NO_DOCUMENT_POLICY,
        "debug_output_policy": TOURRANK_DEBUG_OUTPUT_POLICY,
        "duplicate_selection_policy": TOURRANK_DUPLICATE_SELECTION_POLICY,
        "selection_cardinality_policy": TOURRANK_SELECTION_CARDINALITY_POLICY,
        "query_limit_policy": TOURRANK_QUERY_LIMIT_POLICY,
        "malformed_item_policy": TOURRANK_MALFORMED_ITEM_POLICY,
        "format_repair_policy": TOURRANK_FORMAT_REPAIR_POLICY,
        "max_format_retries": MAX_FORMAT_RETRIES,
        "format_repair_prompt_template": TOURRANK_FORMAT_REPAIR_PROMPT_TEMPLATE,
        "format_repair_prompt_sha256": hashlib.sha256(
            TOURRANK_FORMAT_REPAIR_PROMPT_TEMPLATE.encode("utf-8")
        ).hexdigest(),
        "seed": args.seed,
        "rng_policy": "sha256(base_seed, qid, tournament)",
        "source_sha256": sha256_file(source),
        "official_structure_source_sha256": sha256_file(source),
        "local_run_sha256": sha256_file(args.local_run),
        "inference_input_sha256": inference_records_sha256(records),
        "inference_qids": list(records),
        "inference_query_count": len(records),
        "inference_candidate_count": sum(
            len(record["hits"]) for record in records.values()
        ),
        "passage_source": input_validation["passage_source"],
        "passage_source_policy": (
            "local_ir_datasets_active_tokenizer_512_for_cross_method_comparability"
        ),
        "passage_length": PASSAGE_LENGTH,
        "passage_truncation": passage_truncation,
        "output_objective": "top10",
        "output_depth": 10,
        "query_limit": args.max_queries,
        "command_argv": sanitized_protocol_argv(sys.argv),
        "source_snapshot_sha256": source_snapshot()["sha256"],
        "qrels_id": "dl19-passage" if args.dataset == "dl19" else "dl20-passage",
        "qrels_sha256": hash_ir_dataset_qrels(
            "msmarco-passage/trec-dl-2019/judged"
            if args.dataset == "dl19"
            else "msmarco-passage/trec-dl-2020/judged"
        ),
    }
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["started_at_utc"] = existing.get("started_at_utc")
        protocol["protocol_hash"] = protocol_hash(protocol)
        if existing != protocol:
            raise ValueError("TourRank checkpoint protocol mismatch; use a new attempt.")
        if not args.resume:
            raise FileExistsError("TourRank checkpoints exist; pass --resume.")
    else:
        protocol["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        protocol["protocol_hash"] = protocol_hash(protocol)
        atomic_write_json(protocol_path, protocol)
    persist_discrepancy_report(discrepancy_path, discrepancies)

    official_module = load_tourrank_module()
    official_module.Manager = InlineManager
    official_module.Process = InlineProcess
    responder = InstrumentedResponder(backend)
    official_module.get_response = responder
    debug_path = args.output_dir.resolve() / "debug.txt"
    official_module.get_top_M = InstrumentedOfficialParser(
        official_module.get_top_M, responder, debug_path
    )
    rankings = {}
    checkpoint_paths = {}
    failure_path = args.output_dir / "tourrank_failures.jsonl"
    for qid, record in records.items():
        path = checkpoint_path(checkpoint_root, qid)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("protocol_hash") != protocol["protocol_hash"]:
                raise ValueError(f"TourRank checkpoint hash mismatch for qid={qid}")
            rankings[qid] = payload["docids"]
            checkpoint_paths[qid] = path
            continue
        docids = [str(hit["docid"]) for hit in record["hits"]]
        contents = {str(hit["docid"]): hit["content"] for hit in record["hits"]}
        tournament_scores = []
        responder.events = []
        started = time.perf_counter()
        try:
            for tournament in range(args.tournaments):
                random.seed(tourrank_seed(args.seed, qid, tournament))
                responder.begin_tournament(tournament)
                with isolated_upstream_workdir(args.output_dir):
                    official_module.filter_processing(
                        tournament,
                        record["query"],
                        copy.deepcopy(docids),
                        contents,
                        tournament_scores,
                    )
            wall = time.perf_counter() - started
            expected_calls = EXPECTED_CALLS_PER_TOURNAMENT * args.tournaments
            if len(tournament_scores) != args.tournaments or len(responder.events) != expected_calls:
                raise RuntimeError(f"TourRank qid={qid}: incomplete official schedule")
            ranking = ordered_score_ranking(official_module, docids, tournament_scores)[:10]
            telemetry = build_telemetry(
                qid, args.model, args.tournaments, responder.events, wall
            )
        except Exception as exc:
            record_tourrank_failure(
                failure_path,
                qid=qid,
                model=args.model,
                protocol_hash_value=protocol["protocol_hash"],
                events=copy.deepcopy(responder.events),
                wall_seconds=time.perf_counter() - started,
                error=exc,
            )
            raise
        atomic_write_json(
            path,
            {
                "schema_version": 2,
                "qid": qid,
                "protocol_hash": protocol["protocol_hash"],
                "docids": ranking,
                "telemetry": telemetry,
            },
        )
        rankings[qid] = ranking
        checkpoint_paths[qid] = path

    telemetry_path = args.output_dir / "tourrank_telemetry.jsonl"
    telemetry_path.write_text(
        "".join(
            json.dumps(
                json.loads(checkpoint_paths[qid].read_text(encoding="utf-8"))["telemetry"],
                sort_keys=True,
            )
            + "\n"
            for qid in records
        ),
        encoding="utf-8",
    )
    write_trec(args.output_dir / "tourrank.txt", rankings, f"TourRank-{args.tournaments}-matched")
    manifest = adaptation_manifest(
        "TourRank model-matched adaptation",
        UPSTREAM_COMMIT,
        [
            {"field": "ranking_logic", "changed": False, "reason": "official filter_processing executed"},
            {
                "field": "primary_prompt_text",
                "changed": False,
                "reason": "official prompt helpers executed unchanged",
            },
            {
                "field": "format_repair_prompt",
                "changed": True,
                "reason": (
                    "one deterministic corrective turn is appended only after "
                    "a response with no Document line; candidates and query are "
                    "unchanged and every retry is fully charged"
                ),
            },
            {"field": "parsing", "changed": False, "reason": "official get_top_M executed"},
            {
                "field": "duplicate_selection_semantics",
                "changed": False,
                "reason": (
                    "repeated labels are retained exactly as returned by official "
                    "get_top_M; official dictionary point assignment collapses them "
                    "without deduplication or backfill"
                ),
            },
            {
                "field": "selection_cardinality_semantics",
                "changed": False,
                "reason": (
                    "official get_top_M cardinality is preserved when the model "
                    "returns fewer or more valid labels than requested"
                ),
            },
            {
                "field": "malformed_item_fallback_semantics",
                "changed": False,
                "reason": (
                    "official get_top_M ValueError substitution and IndexError "
                    "ignore branches are preserved exactly and recorded"
                ),
            },
            {
                "field": "response_transport",
                "changed": True,
                "reason": (
                    "raw tokenizer output retained for telemetry; special-token-free "
                    "API-equivalent content passed to the official parser; an initial "
                    "response without a Document line receives one bounded corrective "
                    "turn and a second failure stops without synthetic selections"
                ),
            },
            {
                "field": "debug_output_plumbing",
                "changed": True,
                "reason": (
                    "official relative debug.txt side effects are isolated inside "
                    "the immutable attempt directory"
                ),
            },
            {"field": "generation", "changed": True, "reason": "paper open model with native chat template"},
            {"field": "parallelism", "changed": True, "reason": "official process targets serialized on one GPU"},
            {"field": "randomness_seeding", "changed": True, "reason": "nondeterministic shuffle replaced by qid/tournament-derived deterministic seed"},
            {
                "field": "input_plumbing",
                "changed": True,
                "reason": (
                    "official candidate structure validated; local ir_datasets "
                    "passage text used for cross-method comparability; query-limited "
                    "smoke cells follow local first-stage run order"
                ),
            },
        ],
    )
    manifest.update(
        {
            "algorithm_faithful": True,
            "official_model_reproduction": False,
            "external_api": False,
            "dataset": args.dataset,
            "tournaments": args.tournaments,
            "generation_budget": GENERATION_BUDGET,
            "parser_input_policy": TOURRANK_PARSER_INPUT_POLICY,
            "no_document_policy": TOURRANK_NO_DOCUMENT_POLICY,
            "debug_output_policy": TOURRANK_DEBUG_OUTPUT_POLICY,
            "duplicate_selection_policy": TOURRANK_DUPLICATE_SELECTION_POLICY,
            "selection_cardinality_policy": TOURRANK_SELECTION_CARDINALITY_POLICY,
            "query_limit_policy": TOURRANK_QUERY_LIMIT_POLICY,
            "malformed_item_policy": TOURRANK_MALFORMED_ITEM_POLICY,
            "format_repair_policy": TOURRANK_FORMAT_REPAIR_POLICY,
            "max_format_retries": protocol["max_format_retries"],
            "format_repair_prompt_template": protocol[
                "format_repair_prompt_template"
            ],
            "format_repair_prompt_sha256": protocol[
                "format_repair_prompt_sha256"
            ],
            "rng_policy": protocol["rng_policy"],
            "model": args.model,
            "model_revision": args.model_revision,
            "tokenizer_revision": args.model_revision,
            "backend_contract": backend_contract,
            "output_objective": "top10",
            "output_depth": 10,
            "query_limit": args.max_queries,
            "baseline_category": "model_matched_adapted",
            "protocol_hash": protocol["protocol_hash"],
            "source_sha256": protocol["source_sha256"],
            "official_structure_source_sha256": protocol[
                "official_structure_source_sha256"
            ],
            "local_run_sha256": protocol["local_run_sha256"],
            "inference_input_sha256": protocol["inference_input_sha256"],
            "inference_qids": protocol["inference_qids"],
            "inference_query_count": protocol["inference_query_count"],
            "inference_candidate_count": protocol["inference_candidate_count"],
            "passage_source": protocol["passage_source"],
            "passage_source_policy": protocol["passage_source_policy"],
            "passage_length": protocol["passage_length"],
            "passage_truncation": protocol["passage_truncation"],
            "source_snapshot_sha256": protocol["source_snapshot_sha256"],
            "qrels_id": protocol["qrels_id"],
            "qrels_sha256": protocol["qrels_sha256"],
            "command_argv": protocol["command_argv"],
            "started_at_utc": protocol["started_at_utc"],
            "input_validation": input_validation,
            "parser_debug_artifact": parser_debug_artifact(debug_path),
        }
    )
    atomic_write_json(args.output_dir / "protocol_manifest.json", manifest)


if __name__ == "__main__":
    main()
