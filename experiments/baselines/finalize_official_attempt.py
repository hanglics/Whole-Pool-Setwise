#!/usr/bin/env python3
"""Validate an official-baseline attempt before publishing DONE/LATEST."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capture_provenance import runtime_snapshot
from official_baselines import (
    load_trec,
    paper_model_revisions,
    TOURRANK_DEBUG_OUTPUT_POLICY,
    TOURRANK_DUPLICATE_SELECTION_POLICY,
    TOURRANK_MALFORMED_ITEM_POLICY,
    TOURRANK_FORMAT_REPAIR_POLICY,
    TOURRANK_FORMAT_REPAIR_PROMPT_TEMPLATE,
    TOURRANK_NO_DOCUMENT_POLICY,
    TOURRANK_PARSER_INPUT_POLICY,
    TOURRANK_QUERY_LIMIT_POLICY,
    TOURRANK_SELECTION_CARDINALITY_POLICY,
)
from llmrankers.experiment_controls import atomic_write_json, atomic_write_text, sha256_file


TOURRANK_PASSAGE_POLICY = (
    "local_ir_datasets_active_tokenizer_512_for_cross_method_comparability"
)
LIU_PASSAGE_POLICY = (
    "local_ir_datasets_official_liu_100_word_prompt_pipeline_for_"
    "cross_method_comparability"
)


def _read_run(path: Path) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(f"Malformed TREC row at {path}:{line_number}")
            rankings.setdefault(fields[0], []).append(fields[2])
    return rankings


def _read_telemetry(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("qid"))
            if row.get("status") != "ok":
                raise ValueError(f"Non-success telemetry at {path}:{line_number}")
            if qid in rows:
                raise ValueError(f"Duplicate telemetry qid={qid!r}")
            rows[qid] = row
    return rows


def validate_tourrank_provenance(
    manifest: dict, *, query_ids: set[str], discrepancy_path: Path
) -> None:
    """Reject a TourRank result that is not bound to our model-facing text."""
    query_count = len(query_ids)
    if manifest.get("passage_source_policy") != TOURRANK_PASSAGE_POLICY:
        raise ValueError("TourRank manifest has the wrong passage-source policy.")
    if manifest.get("passage_length") != 512:
        raise ValueError("TourRank manifest must use passage_length=512.")
    if manifest.get("query_limit_policy") != TOURRANK_QUERY_LIMIT_POLICY:
        raise ValueError("TourRank manifest has the wrong query-limit policy.")
    digest = manifest.get("inference_input_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("TourRank manifest has no valid inference-input hash.")
    if manifest.get("inference_query_count") != query_count:
        raise ValueError("TourRank inference-query count does not match its output.")
    inference_qids = manifest.get("inference_qids")
    if (
        not isinstance(inference_qids, list)
        or not all(isinstance(qid, str) for qid in inference_qids)
        or len(inference_qids) != len(set(inference_qids))
        or set(inference_qids) != query_ids
    ):
        raise ValueError("TourRank inference qids do not match its output.")
    candidate_count = query_count * 100
    if manifest.get("inference_candidate_count") != candidate_count:
        raise ValueError("TourRank inference-candidate count does not match top-100 input.")

    validation = manifest.get("input_validation") or {}
    if validation.get("structural_discrepancies") != 0:
        raise ValueError("TourRank input has structural discrepancies.")
    if validation.get("official_json_used_for_inference") is not False:
        raise ValueError("TourRank must not use released passage text for inference.")
    if validation.get("local_ir_datasets_content_used_for_inference") is not True:
        raise ValueError("TourRank must use local ir_datasets passage text.")
    if not discrepancy_path.is_file():
        raise FileNotFoundError(
            f"Missing TourRank input-discrepancy artifact: {discrepancy_path}"
        )
    discrepancies = json.loads(discrepancy_path.read_text(encoding="utf-8"))
    if len(discrepancies) != validation.get("discrepancies"):
        raise ValueError("TourRank discrepancy artifact does not match its manifest.")
    if any(row.get("field") != "content" for row in discrepancies):
        raise ValueError("TourRank discrepancy artifact contains structural drift.")
    if len(discrepancies) != validation.get("content_discrepancies"):
        raise ValueError("TourRank content-discrepancy count does not match its manifest.")

    truncation = manifest.get("passage_truncation") or {}
    if truncation.get("passage_count") != candidate_count:
        raise ValueError("TourRank truncation count does not match model-facing input.")
    if truncation.get("passage_length") != 512:
        raise ValueError("TourRank truncation metadata has the wrong passage length.")
    truncated_count = truncation.get("truncated_passage_count")
    if not isinstance(truncated_count, int) or not 0 <= truncated_count <= candidate_count:
        raise ValueError("TourRank truncated-passage count is invalid.")
    if truncation.get("truncation_method") != (
        "active_tokenizer_tokenize_then_convert_tokens_to_string"
    ):
        raise ValueError("TourRank truncation method is not the native tokenizer policy.")


def validate_tourrank_format_repair_event(
    qid: str,
    event: dict,
    *,
    format_repair_policy: str | None,
    generation_budget: int | None,
) -> dict:
    """Validate retry visibility and return event-level accounting totals."""
    if format_repair_policy is None:
        if event.get("format_repair_policy") is not None:
            raise ValueError(f"qid={qid}: legacy TourRank event declares repair")
        if event.get("generation_attempts") is not None:
            raise ValueError(f"qid={qid}: legacy TourRank event hides a repair schema")
        if int(event.get("retry_calls", 0)) != 0:
            raise ValueError(f"qid={qid}: legacy TourRank event has retry calls")
        return {
            "retry_calls": 0,
            "retry_seconds": 0.0,
            "generation_attempts": 1,
            "stripped_generations": int(bool(event.get("special_tokens_removed"))),
            "initial_format_failures": int(event.get("format_valid") is False),
            "format_repair_successes": 0,
        }

    if event.get("format_repair_policy") != TOURRANK_FORMAT_REPAIR_POLICY:
        raise ValueError(f"qid={qid}: TourRank event repair policy mismatch")
    attempts = event.get("generation_attempts")
    if not isinstance(attempts, list) or len(attempts) not in (1, 2):
        raise ValueError(f"qid={qid}: invalid TourRank generation-attempt list")
    if event.get("generation_attempt_count") != len(attempts):
        raise ValueError(f"qid={qid}: TourRank generation-attempt count mismatch")

    prompt_tokens = 0
    completion_tokens = 0
    request_seconds = 0.0
    stripped_generations = 0
    for index, attempt in enumerate(attempts):
        if attempt.get("attempt_index") != index:
            raise ValueError(f"qid={qid}: TourRank generation-attempt index mismatch")
        expected_kind = "initial" if index == 0 else "format_repair"
        if attempt.get("attempt_kind") != expected_kind:
            raise ValueError(f"qid={qid}: TourRank generation-attempt kind mismatch")
        raw_output = attempt.get("raw_output")
        content_output = attempt.get("content_output")
        parser_input = attempt.get("parser_input")
        if not all(
            isinstance(value, str)
            for value in (raw_output, content_output, parser_input)
        ):
            raise ValueError(f"qid={qid}: TourRank generation-attempt views are missing")
        if parser_input != content_output:
            raise ValueError(f"qid={qid}: TourRank repair attempt altered parser input")
        expected_stripped = raw_output != content_output
        if attempt.get("special_tokens_removed") is not expected_stripped:
            raise ValueError(f"qid={qid}: TourRank repair control-token mismatch")
        stripped_generations += expected_stripped
        format_valid = attempt.get("format_valid")
        if not isinstance(format_valid, bool):
            raise ValueError(f"qid={qid}: TourRank repair format flag is missing")
        has_document = any("Document" in line for line in parser_input.splitlines())
        if format_valid is not has_document:
            raise ValueError(f"qid={qid}: TourRank repair format flag mismatch")
        expected_status = "ok" if format_valid else "format_invalid"
        if attempt.get("status") != expected_status:
            raise ValueError(f"qid={qid}: TourRank repair attempt status mismatch")
        if attempt.get("parser_input_policy") != TOURRANK_PARSER_INPUT_POLICY:
            raise ValueError(f"qid={qid}: TourRank repair parser policy mismatch")
        if attempt.get("no_document_policy") != TOURRANK_NO_DOCUMENT_POLICY:
            raise ValueError(f"qid={qid}: TourRank repair no-Document policy mismatch")
        if attempt.get("generation_budget") != generation_budget:
            raise ValueError(f"qid={qid}: TourRank repair generation budget mismatch")
        rendered_hash = attempt.get("rendered_prompt_sha256")
        if not isinstance(rendered_hash, str) or re.fullmatch(
            r"[0-9a-f]{64}", rendered_hash
        ) is None:
            raise ValueError(f"qid={qid}: TourRank repair prompt hash is invalid")
        attempt_prompt_tokens = attempt.get("prompt_tokens")
        attempt_completion_tokens = attempt.get("completion_tokens")
        attempt_seconds = attempt.get("duration_seconds")
        if not isinstance(attempt_prompt_tokens, int) or attempt_prompt_tokens < 0:
            raise ValueError(f"qid={qid}: TourRank repair prompt tokens are invalid")
        if (
            not isinstance(attempt_completion_tokens, int)
            or attempt_completion_tokens < 0
        ):
            raise ValueError(f"qid={qid}: TourRank repair completion tokens are invalid")
        if not isinstance(attempt_seconds, (int, float)) or attempt_seconds < 0:
            raise ValueError(f"qid={qid}: TourRank repair duration is invalid")
        prompt_tokens += attempt_prompt_tokens
        completion_tokens += attempt_completion_tokens
        request_seconds += float(attempt_seconds)

    retry_calls = len(attempts) - 1
    if retry_calls and attempts[0]["format_valid"] is not False:
        raise ValueError(f"qid={qid}: TourRank repair ran after a valid response")
    if attempts[-1]["format_valid"] is not True:
        raise ValueError(f"qid={qid}: successful TourRank event has invalid repair output")
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
        if event.get(field) != final_attempt.get(field):
            raise ValueError(f"qid={qid}: TourRank final repair view mismatch for {field}")
    if event.get("retry_calls") != retry_calls:
        raise ValueError(f"qid={qid}: TourRank event retry count mismatch")
    expected_retry_seconds = sum(
        float(attempt["duration_seconds"]) for attempt in attempts[1:]
    )
    if not math.isclose(
        float(event.get("retry_seconds", -1)), expected_retry_seconds, abs_tol=1e-9
    ):
        raise ValueError(f"qid={qid}: TourRank event retry time mismatch")
    if event.get("format_repair_triggered") is not (retry_calls > 0):
        raise ValueError(f"qid={qid}: TourRank repair-trigger flag mismatch")
    if event.get("format_repair_succeeded") is not (retry_calls > 0):
        raise ValueError(f"qid={qid}: TourRank repair-success flag mismatch")
    if event.get("initial_format_valid") is not attempts[0]["format_valid"]:
        raise ValueError(f"qid={qid}: TourRank initial format flag mismatch")
    if int(event.get("prompt_tokens", -1)) != prompt_tokens:
        raise ValueError(f"qid={qid}: TourRank event prompt-token mismatch")
    if int(event.get("completion_tokens", -1)) != completion_tokens:
        raise ValueError(f"qid={qid}: TourRank event completion-token mismatch")
    if not math.isclose(
        float(event.get("duration_seconds", -1)), request_seconds, abs_tol=1e-9
    ):
        raise ValueError(f"qid={qid}: TourRank event request-time mismatch")
    return {
        "retry_calls": retry_calls,
        "retry_seconds": expected_retry_seconds,
        "generation_attempts": len(attempts),
        "stripped_generations": stripped_generations,
        "initial_format_failures": int(attempts[0]["format_valid"] is False),
        "format_repair_successes": retry_calls,
    }


def validate_tourrank_generation_contract(
    manifest: dict, telemetry: dict[str, dict], debug_path: Path
) -> None:
    """Reject hidden control-token corruption or undisclosed format repair."""
    if manifest.get("parser_input_policy") != TOURRANK_PARSER_INPUT_POLICY:
        raise ValueError("TourRank manifest has the wrong parser-input policy.")
    if manifest.get("no_document_policy") != TOURRANK_NO_DOCUMENT_POLICY:
        raise ValueError("TourRank manifest has the wrong no-Document policy.")
    if manifest.get("debug_output_policy") != TOURRANK_DEBUG_OUTPUT_POLICY:
        raise ValueError("TourRank manifest has the wrong debug-output policy.")
    if (
        manifest.get("duplicate_selection_policy")
        != TOURRANK_DUPLICATE_SELECTION_POLICY
    ):
        raise ValueError("TourRank manifest has the wrong duplicate-selection policy.")
    if (
        manifest.get("selection_cardinality_policy")
        != TOURRANK_SELECTION_CARDINALITY_POLICY
    ):
        raise ValueError("TourRank manifest has the wrong cardinality policy.")
    malformed_item_policy = manifest.get("malformed_item_policy")
    if malformed_item_policy not in (None, TOURRANK_MALFORMED_ITEM_POLICY):
        raise ValueError("TourRank manifest has the wrong malformed-item policy.")
    format_repair_policy = manifest.get("format_repair_policy")
    if format_repair_policy not in (None, TOURRANK_FORMAT_REPAIR_POLICY):
        raise ValueError("TourRank manifest has the wrong format-repair policy.")
    if format_repair_policy is not None:
        if manifest.get("max_format_retries") != 1:
            raise ValueError("TourRank manifest must allow exactly one format retry.")
        if (
            manifest.get("format_repair_prompt_template")
            != TOURRANK_FORMAT_REPAIR_PROMPT_TEMPLATE
        ):
            raise ValueError("TourRank manifest has the wrong format-repair prompt.")
        expected_repair_prompt_hash = hashlib.sha256(
            TOURRANK_FORMAT_REPAIR_PROMPT_TEMPLATE.encode("utf-8")
        ).hexdigest()
        if manifest.get("format_repair_prompt_sha256") != expected_repair_prompt_hash:
            raise ValueError("TourRank format-repair prompt hash mismatch.")

    expected_debug_entries = 0
    for qid, row in telemetry.items():
        if row.get("parser_input_policy") != TOURRANK_PARSER_INPUT_POLICY:
            raise ValueError(f"qid={qid}: TourRank telemetry parser policy mismatch")
        if row.get("no_document_policy") != TOURRANK_NO_DOCUMENT_POLICY:
            raise ValueError(f"qid={qid}: TourRank telemetry fallback policy mismatch")
        if row.get("debug_output_policy") != TOURRANK_DEBUG_OUTPUT_POLICY:
            raise ValueError(f"qid={qid}: TourRank telemetry debug policy mismatch")
        if (
            row.get("duplicate_selection_policy")
            != TOURRANK_DUPLICATE_SELECTION_POLICY
        ):
            raise ValueError(
                f"qid={qid}: TourRank duplicate-selection policy mismatch"
            )
        if (
            row.get("selection_cardinality_policy")
            != TOURRANK_SELECTION_CARDINALITY_POLICY
        ):
            raise ValueError(f"qid={qid}: TourRank cardinality policy mismatch")
        if row.get("malformed_item_policy") != malformed_item_policy:
            raise ValueError(f"qid={qid}: TourRank malformed-item policy mismatch")
        if row.get("format_repair_policy") != format_repair_policy:
            raise ValueError(f"qid={qid}: TourRank format-repair policy mismatch")
        events = row.get("events")
        if not isinstance(events, list) or len(events) != int(row.get("total_calls", -1)):
            raise ValueError(f"qid={qid}: TourRank generation events are incomplete")

        stripped = 0
        format_failures = 0
        value_error_fallbacks = 0
        index_error_fallbacks = 0
        duplicate_selection_calls = 0
        duplicate_selection_slots = 0
        under_selection_calls = 0
        under_selection_slots = 0
        over_selection_calls = 0
        over_selection_slots = 0
        retry_calls = 0
        retry_seconds = 0.0
        generation_attempts = 0
        stripped_generations = 0
        initial_format_failures = 0
        format_repair_successes = 0
        prompt_tokens = 0
        completion_tokens = 0
        request_seconds = 0.0
        for event in events:
            if event.get("status") != "ok":
                raise ValueError(f"qid={qid}: TourRank event is not successful")
            repair_accounting = validate_tourrank_format_repair_event(
                qid,
                event,
                format_repair_policy=format_repair_policy,
                generation_budget=manifest.get("generation_budget"),
            )
            retry_calls += repair_accounting["retry_calls"]
            retry_seconds += repair_accounting["retry_seconds"]
            generation_attempts += repair_accounting["generation_attempts"]
            stripped_generations += repair_accounting["stripped_generations"]
            initial_format_failures += repair_accounting["initial_format_failures"]
            format_repair_successes += repair_accounting[
                "format_repair_successes"
            ]
            prompt_tokens += int(event.get("prompt_tokens", 0))
            completion_tokens += int(event.get("completion_tokens", 0))
            request_seconds += float(event.get("duration_seconds", 0))
            raw_output = event.get("raw_output")
            content_output = event.get("content_output")
            parser_input = event.get("parser_input")
            if not all(
                isinstance(value, str)
                for value in (raw_output, content_output, parser_input)
            ):
                raise ValueError(f"qid={qid}: TourRank output views are missing")
            expected_stripped = raw_output != content_output
            if event.get("special_tokens_removed") is not expected_stripped:
                raise ValueError(f"qid={qid}: TourRank control-token flag mismatch")
            stripped += expected_stripped

            format_valid = event.get("format_valid")
            if not isinstance(format_valid, bool):
                raise ValueError(f"qid={qid}: TourRank format-valid flag is missing")
            format_failures += not format_valid
            if event.get("parser_input_policy") != TOURRANK_PARSER_INPUT_POLICY:
                raise ValueError(f"qid={qid}: TourRank event parser policy mismatch")
            if event.get("parser_status") != "ok":
                raise ValueError(f"qid={qid}: TourRank parser did not complete")
            value_errors = event.get("parser_value_error_fallbacks")
            index_errors = event.get("parser_index_error_fallbacks")
            if not isinstance(value_errors, int) or value_errors < 0:
                raise ValueError(f"qid={qid}: invalid TourRank ValueError count")
            if not isinstance(index_errors, int) or index_errors < 0:
                raise ValueError(f"qid={qid}: invalid TourRank IndexError count")
            value_error_fallbacks += value_errors
            index_error_fallbacks += index_errors
            expected_debug_entries += value_errors + index_errors
            debug_bytes = event.get("parser_debug_delta_bytes")
            debug_digest = event.get("parser_debug_delta_sha256")
            if not isinstance(debug_bytes, int) or debug_bytes < 0:
                raise ValueError(f"qid={qid}: invalid TourRank debug delta size")
            if not isinstance(debug_digest, str) or re.fullmatch(
                r"[0-9a-f]{64}", debug_digest
            ) is None:
                raise ValueError(f"qid={qid}: invalid TourRank debug delta hash")
            if debug_bytes == 0 and debug_digest != hashlib.sha256(b"").hexdigest():
                raise ValueError(f"qid={qid}: empty TourRank debug delta hash mismatch")
            if (value_errors + index_errors > 0) != (debug_bytes > 0):
                raise ValueError(f"qid={qid}: TourRank warning/debug mismatch")
            if not format_valid:
                raise ValueError(f"qid={qid}: successful TourRank row has a format failure")
            if parser_input != content_output:
                raise ValueError(f"qid={qid}: TourRank parser input was altered")
            if event.get("no_document_policy") != TOURRANK_NO_DOCUMENT_POLICY:
                raise ValueError(f"qid={qid}: TourRank no-Document policy drift")
            if not any("Document" in line for line in parser_input.splitlines()):
                raise ValueError(f"qid={qid}: TourRank parser input has no Document line")
            if event.get("parser_contract_valid") is not True:
                raise ValueError(f"qid={qid}: TourRank parser contract is invalid")
            if event.get("malformed_item_policy") != malformed_item_policy:
                raise ValueError(f"qid={qid}: TourRank event malformed-item policy mismatch")
            requested_count = event.get("m")
            selected_count = event.get("parser_selected_count")
            if not isinstance(requested_count, int) or requested_count < 1:
                raise ValueError(f"qid={qid}: invalid TourRank requested count")
            if not isinstance(selected_count, int) or selected_count < 0:
                raise ValueError(f"qid={qid}: invalid TourRank selected count")
            unique_count = event.get("parser_unique_selected_count")
            if (
                not isinstance(unique_count, int)
                or unique_count < 0
                or unique_count > selected_count
            ):
                raise ValueError(f"qid={qid}: invalid TourRank unique-selection count")
            expected_duplicate_slots = selected_count - unique_count
            if (
                event.get("parser_duplicate_selection_slots")
                != expected_duplicate_slots
            ):
                raise ValueError(f"qid={qid}: TourRank duplicate-slot count mismatch")
            if (
                event.get("parser_selection_unique")
                is not (expected_duplicate_slots == 0)
            ):
                raise ValueError(f"qid={qid}: TourRank selection-unique flag mismatch")
            if (
                event.get("duplicate_selection_policy")
                != TOURRANK_DUPLICATE_SELECTION_POLICY
            ):
                raise ValueError(f"qid={qid}: TourRank event duplicate policy mismatch")
            expected_count_delta = selected_count - requested_count
            expected_under_slots = max(-expected_count_delta, 0)
            expected_over_slots = max(expected_count_delta, 0)
            if event.get("parser_selection_count_delta") != expected_count_delta:
                raise ValueError(f"qid={qid}: TourRank selection-count delta mismatch")
            if event.get("parser_under_selection_slots") != expected_under_slots:
                raise ValueError(f"qid={qid}: TourRank under-selection mismatch")
            if event.get("parser_over_selection_slots") != expected_over_slots:
                raise ValueError(f"qid={qid}: TourRank over-selection mismatch")
            if (
                event.get("parser_selection_count_matches_requested")
                is not (expected_count_delta == 0)
            ):
                raise ValueError(f"qid={qid}: TourRank count-match flag mismatch")
            if (
                event.get("selection_cardinality_policy")
                != TOURRANK_SELECTION_CARDINALITY_POLICY
            ):
                raise ValueError(f"qid={qid}: TourRank event cardinality policy mismatch")
            duplicate_selection_calls += expected_duplicate_slots > 0
            duplicate_selection_slots += expected_duplicate_slots
            under_selection_calls += expected_under_slots > 0
            under_selection_slots += expected_under_slots
            over_selection_calls += expected_over_slots > 0
            over_selection_slots += expected_over_slots

        if int(row.get("special_token_stripped_calls", -1)) != stripped:
            raise ValueError(f"qid={qid}: TourRank stripped-call count mismatch")
        if int(row.get("format_failure_calls", -1)) != format_failures:
            raise ValueError(f"qid={qid}: TourRank format-failure count mismatch")
        if format_failures:
            raise ValueError(f"qid={qid}: successful TourRank telemetry has format failures")
        if format_repair_policy is not None:
            nominal_calls = len(events)
            if int(row.get("nominal_calls", -1)) != nominal_calls:
                raise ValueError(f"qid={qid}: TourRank nominal-call count mismatch")
            if int(row.get("retry_calls", -1)) != retry_calls:
                raise ValueError(f"qid={qid}: TourRank retry-call aggregate mismatch")
            if (
                int(row.get("retry_inclusive_llm_calls", -1))
                != nominal_calls + retry_calls
            ):
                raise ValueError(
                    f"qid={qid}: TourRank retry-inclusive call count mismatch"
                )
            if int(row.get("generation_attempt_count", -1)) != generation_attempts:
                raise ValueError(
                    f"qid={qid}: TourRank generation-attempt aggregate mismatch"
                )
            if (
                int(row.get("special_token_stripped_generations", -1))
                != stripped_generations
            ):
                raise ValueError(
                    f"qid={qid}: TourRank stripped-generation count mismatch"
                )
            if (
                int(row.get("initial_format_failure_calls", -1))
                != initial_format_failures
            ):
                raise ValueError(
                    f"qid={qid}: TourRank initial-format aggregate mismatch"
                )
            if int(row.get("format_repair_calls", -1)) != retry_calls:
                raise ValueError(f"qid={qid}: TourRank repair-call aggregate mismatch")
            if (
                int(row.get("format_repair_successes", -1))
                != format_repair_successes
            ):
                raise ValueError(
                    f"qid={qid}: TourRank repair-success aggregate mismatch"
                )
            if int(row.get("prompt_tokens", -1)) != prompt_tokens:
                raise ValueError(f"qid={qid}: TourRank prompt-token aggregate mismatch")
            if int(row.get("completion_tokens", -1)) != completion_tokens:
                raise ValueError(
                    f"qid={qid}: TourRank completion-token aggregate mismatch"
                )
            if not math.isclose(
                float(row.get("request_seconds_sum", -1)),
                request_seconds,
                abs_tol=1e-9,
            ):
                raise ValueError(f"qid={qid}: TourRank request-time aggregate mismatch")
            if not math.isclose(
                float(row.get("retry_seconds", -1)), retry_seconds, abs_tol=1e-9
            ):
                raise ValueError(f"qid={qid}: TourRank retry-time aggregate mismatch")
            query_wall_seconds = float(row.get("query_wall_seconds", -1))
            expected_retry_excluded = max(query_wall_seconds - retry_seconds, 0.0)
            if not math.isclose(
                float(row.get("retry_excluded_wall_seconds", -1)),
                expected_retry_excluded,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"qid={qid}: TourRank retry-excluded time mismatch"
                )
        if int(row.get("parser_value_error_fallbacks", -1)) != value_error_fallbacks:
            raise ValueError(f"qid={qid}: TourRank ValueError aggregate mismatch")
        if int(row.get("parser_index_error_fallbacks", -1)) != index_error_fallbacks:
            raise ValueError(f"qid={qid}: TourRank IndexError aggregate mismatch")
        if (
            int(row.get("parser_duplicate_selection_calls", -1))
            != duplicate_selection_calls
        ):
            raise ValueError(f"qid={qid}: TourRank duplicate-call aggregate mismatch")
        if (
            int(row.get("parser_duplicate_selection_slots", -1))
            != duplicate_selection_slots
        ):
            raise ValueError(f"qid={qid}: TourRank duplicate-slot aggregate mismatch")
        if int(row.get("parser_under_selection_calls", -1)) != under_selection_calls:
            raise ValueError(f"qid={qid}: TourRank under-call aggregate mismatch")
        if int(row.get("parser_under_selection_slots", -1)) != under_selection_slots:
            raise ValueError(f"qid={qid}: TourRank under-slot aggregate mismatch")
        if int(row.get("parser_over_selection_calls", -1)) != over_selection_calls:
            raise ValueError(f"qid={qid}: TourRank over-call aggregate mismatch")
        if int(row.get("parser_over_selection_slots", -1)) != over_selection_slots:
            raise ValueError(f"qid={qid}: TourRank over-slot aggregate mismatch")

    payload = debug_path.read_bytes() if debug_path.is_file() else b""
    text = payload.decode("utf-8", errors="replace")
    actual_debug = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "entries": sum(line == "New Error: " for line in text.splitlines()),
    }
    if manifest.get("parser_debug_artifact") != actual_debug:
        raise ValueError("TourRank parser debug artifact does not match its manifest.")
    if actual_debug["entries"] != expected_debug_entries:
        raise ValueError("TourRank parser debug entries do not match telemetry.")


def validate_liu_provenance(
    manifest: dict, *, query_count: int, discrepancy_path: Path
) -> None:
    """Reject a Liu result that is not bound to our local passage source."""
    if manifest.get("passage_source_policy") != LIU_PASSAGE_POLICY:
        raise ValueError("Liu manifest has the wrong passage-source policy.")
    if manifest.get("max_passage_length") != 100:
        raise ValueError("Liu manifest must preserve the official 100-word cap.")
    digest = manifest.get("inference_input_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("Liu manifest has no valid inference-input hash.")
    if manifest.get("inference_query_count") != query_count:
        raise ValueError("Liu inference-query count does not match its output.")
    candidate_count = query_count * 100
    if manifest.get("inference_candidate_count") != candidate_count:
        raise ValueError("Liu inference-candidate count does not match top-100 input.")
    if manifest.get("official_index_content_used_for_inference") is not False:
        raise ValueError("Liu must not use released-index passage text for inference.")
    if manifest.get("local_ir_datasets_content_used_for_inference") is not True:
        raise ValueError("Liu must use local ir_datasets passage text.")

    validation = manifest.get("input_validation") or {}
    if validation.get("structural_discrepancies") != 0:
        raise ValueError("Liu input has structural discrepancies.")
    if validation.get("official_index_content_used_for_inference") is not False:
        raise ValueError("Liu validation does not exclude released-index content.")
    if validation.get("local_ir_datasets_content_used_for_inference") is not True:
        raise ValueError("Liu validation does not confirm local passage content.")
    if not discrepancy_path.is_file():
        raise FileNotFoundError(
            f"Missing Liu input-discrepancy artifact: {discrepancy_path}"
        )
    discrepancies = json.loads(discrepancy_path.read_text(encoding="utf-8"))
    if len(discrepancies) != validation.get("discrepancies"):
        raise ValueError("Liu discrepancy artifact does not match its manifest.")
    if any(row.get("field") != "content" for row in discrepancies):
        raise ValueError("Liu discrepancy artifact contains structural drift.")
    if len(discrepancies) != validation.get("content_discrepancies"):
        raise ValueError("Liu content-discrepancy count does not match its manifest.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-dir", required=True, type=Path)
    parser.add_argument("--condition-dir", required=True, type=Path)
    parser.add_argument("--run-file", required=True)
    parser.add_argument("--telemetry-file", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--input-run", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-objective", required=True, choices=["top10", "full"])
    parser.add_argument("--output-depth", required=True, type=int)
    parser.add_argument("--expected-calls-per-query", type=int)
    parser.add_argument("--expected-category", required=True)
    parser.add_argument("--query-limit", type=int, default=0)
    args = parser.parse_args()
    if args.query_limit < 0:
        raise ValueError("--query-limit must be non-negative.")

    run_path = args.attempt_dir / args.run_file
    telemetry_path = args.attempt_dir / args.telemetry_file
    manifest_path = args.attempt_dir / "protocol_manifest.json"
    for path in (run_path, telemetry_path, manifest_path, args.input_run):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty required artifact: {path}")

    expected = load_trec(args.input_run, 100)
    if args.query_limit:
        expected = dict(list(expected.items())[: args.query_limit])
    rankings = _read_run(run_path)
    telemetry = _read_telemetry(telemetry_path)
    if set(rankings) != set(expected) or set(telemetry) != set(expected):
        raise ValueError("Baseline output, telemetry, and first-stage qid sets differ.")
    for qid, docids in rankings.items():
        if len(docids) != args.output_depth or len(set(docids)) != args.output_depth:
            raise ValueError(f"qid={qid}: output depth/uniqueness failure")
        if not set(docids).issubset({row["docid"] for row in expected[qid]}):
            raise ValueError(f"qid={qid}: output contains a document outside the input pool")
        row = telemetry[qid]
        if row.get("output_objective") != args.output_objective:
            raise ValueError(f"qid={qid}: telemetry objective mismatch")
        if int(row.get("output_depth", -1)) != args.output_depth:
            raise ValueError(f"qid={qid}: telemetry depth mismatch")
        calls = row.get("total_calls", row.get("llm_calls"))
        if calls is None or int(calls) <= 0:
            raise ValueError(f"qid={qid}: missing positive LLM-call count")
        if args.expected_calls_per_query is not None and int(calls) != args.expected_calls_per_query:
            raise ValueError(
                f"qid={qid}: expected {args.expected_calls_per_query} calls, got {calls}"
            )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    locked_models = paper_model_revisions()
    manifest_model = manifest.get("model")
    if manifest_model not in locked_models:
        raise ValueError(f"Baseline model is not an paper model: {manifest_model!r}")
    locked_revision = locked_models[manifest_model]
    if manifest.get("model_revision") != locked_revision:
        raise ValueError("Baseline model revision does not match model_revisions.lock.json.")
    if manifest.get("tokenizer_revision") != locked_revision:
        raise ValueError("Baseline tokenizer revision does not match model_revisions.lock.json.")
    backend_contract = manifest.get("backend_contract") or {}
    if backend_contract.get("model") != manifest_model:
        raise ValueError("Backend contract model does not match the locked manifest model.")
    if backend_contract.get("model_revision") != locked_revision:
        raise ValueError("Backend contract model revision does not match the lock.")
    if backend_contract.get("tokenizer_revision") != locked_revision:
        raise ValueError("Backend contract tokenizer revision does not match the lock.")
    if any(row.get("model") != manifest_model for row in telemetry.values()):
        raise ValueError("Telemetry model does not match the locked manifest model.")
    required_manifest = {
        "dataset": args.dataset,
        "output_objective": args.output_objective,
        "output_depth": args.output_depth,
        "baseline_category": args.expected_category,
        "query_limit": args.query_limit or None,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in required_manifest.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Baseline manifest mismatch: {mismatches}")
    if manifest.get("algorithm_faithful") is not True:
        raise ValueError("Model-matched baseline is not marked algorithm_faithful.")
    if manifest.get("official_model_reproduction") is not False:
        raise ValueError("Model-matched baseline must not claim official-model reproduction.")
    if manifest.get("external_api") is not False:
        raise ValueError("Revision baselines must use local open-model inference only.")
    if not isinstance(manifest.get("command_argv"), list) or not manifest["command_argv"]:
        raise ValueError("Baseline manifest is missing sanitized command_argv.")
    if not manifest.get("started_at_utc"):
        raise ValueError("Baseline manifest is missing started_at_utc.")
    if args.output_objective == "full" and manifest.get("setting") != "zero_shot":
        raise ValueError("The Liu full-ranking baseline must be explicitly zero-shot.")
    if args.output_objective == "full":
        validate_liu_provenance(
            manifest,
            query_count=len(rankings),
            discrepancy_path=args.attempt_dir / "input_discrepancies.json",
        )
    if args.output_objective == "top10":
        validate_tourrank_provenance(
            manifest,
            query_ids=set(rankings),
            discrepancy_path=args.attempt_dir / "input_discrepancies.json",
        )
        validate_tourrank_generation_contract(
            manifest, telemetry, args.attempt_dir / "debug.txt"
        )

    runtime = runtime_snapshot()
    if manifest.get("source_snapshot_sha256") != runtime["source_snapshot"]["sha256"]:
        raise ValueError("Experiment source snapshot changed while the baseline attempt was running.")
    prompt_tokens = sum(int(row.get("prompt_tokens", 0)) for row in telemetry.values())
    completion_tokens = sum(int(row.get("completion_tokens", 0)) for row in telemetry.values())
    calls = sum(int(row.get("total_calls", row.get("llm_calls", 0))) for row in telemetry.values())
    retry_calls = sum(int(row.get("retry_calls", 0)) for row in telemetry.values())
    batch_walls = {
        row.get("batch_id"): row.get("batch_wall_seconds")
        for row in telemetry.values()
        if row.get("batch_id") is not None
    }
    summary = {
        "schema_version": 1,
        "queries": len(rankings),
        "output_objective": args.output_objective,
        "output_depth": args.output_depth,
        "llm_calls": calls,
        "retry_calls": retry_calls,
        "retry_inclusive_llm_calls": calls + retry_calls,
        "actual_prompt_tokens": prompt_tokens,
        "actual_completion_tokens": completion_tokens,
        "retry_seconds": sum(
            float(row.get("retry_seconds", 0) or 0) for row in telemetry.values()
        ),
        "query_wall_seconds_sum": sum(
            float(row.get("query_wall_seconds", 0) or 0) for row in telemetry.values()
        ),
        "batch_wall_seconds_sum": sum(float(value or 0) for value in batch_walls.values()),
    }
    if args.output_objective == "top10":
        summary.update(
            {
                "special_token_stripped_calls": sum(
                    int(row.get("special_token_stripped_calls", 0))
                    for row in telemetry.values()
                ),
                "format_failure_calls": sum(
                    int(row.get("format_failure_calls", 0))
                    for row in telemetry.values()
                ),
                "initial_format_failure_calls": sum(
                    int(row.get("initial_format_failure_calls", 0))
                    for row in telemetry.values()
                ),
                "format_repair_calls": sum(
                    int(row.get("format_repair_calls", 0))
                    for row in telemetry.values()
                ),
                "format_repair_successes": sum(
                    int(row.get("format_repair_successes", 0))
                    for row in telemetry.values()
                ),
                "parser_value_error_fallbacks": sum(
                    int(row.get("parser_value_error_fallbacks", 0))
                    for row in telemetry.values()
                ),
                "parser_index_error_fallbacks": sum(
                    int(row.get("parser_index_error_fallbacks", 0))
                    for row in telemetry.values()
                ),
                "parser_duplicate_selection_calls": sum(
                    int(row.get("parser_duplicate_selection_calls", 0))
                    for row in telemetry.values()
                ),
                "parser_duplicate_selection_slots": sum(
                    int(row.get("parser_duplicate_selection_slots", 0))
                    for row in telemetry.values()
                ),
                "parser_under_selection_calls": sum(
                    int(row.get("parser_under_selection_calls", 0))
                    for row in telemetry.values()
                ),
                "parser_under_selection_slots": sum(
                    int(row.get("parser_under_selection_slots", 0))
                    for row in telemetry.values()
                ),
                "parser_over_selection_calls": sum(
                    int(row.get("parser_over_selection_calls", 0))
                    for row in telemetry.values()
                ),
                "parser_over_selection_slots": sum(
                    int(row.get("parser_over_selection_slots", 0))
                    for row in telemetry.values()
                ),
                "parser_input_policy": TOURRANK_PARSER_INPUT_POLICY,
                "no_document_policy": TOURRANK_NO_DOCUMENT_POLICY,
                "duplicate_selection_policy": TOURRANK_DUPLICATE_SELECTION_POLICY,
                "selection_cardinality_policy": TOURRANK_SELECTION_CARDINALITY_POLICY,
                "malformed_item_policy": TOURRANK_MALFORMED_ITEM_POLICY,
                "format_repair_policy": manifest.get("format_repair_policy"),
                "parser_debug_artifact": manifest["parser_debug_artifact"],
            }
        )
    manifest.update(
        {
            "runtime": runtime,
            "source_snapshot": runtime["source_snapshot"],
            "git_commit": runtime["git_commit"],
            "git_dirty": runtime["git_dirty"],
            "git_status_sha256": runtime["git_status_sha256"],
            "hostname": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "parallelism": {
                "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "queries": len(rankings),
            "query_limit": args.query_limit or None,
            "run_file": args.run_file,
        }
    )
    atomic_write_json(args.attempt_dir / args.summary_file, summary)
    manifest.update(
        {
            "output_run_sha256": sha256_file(run_path),
            "telemetry_sha256": sha256_file(telemetry_path),
            "summary_sha256": sha256_file(args.attempt_dir / args.summary_file),
        }
    )
    atomic_write_json(manifest_path, manifest)

    atomic_write_text(args.attempt_dir / "DONE", "ok\n")
    atomic_write_text(args.condition_dir / "LATEST", args.attempt_dir.name + "\n")


if __name__ == "__main__":
    main()
