import logging
import ir_datasets
from pyserini.search.lucene import LuceneSearcher
from pyserini.search._base import get_topics
from llmrankers.rankers import SearchResult
from llmrankers.pointwise import PointwiseLlmRanker, MonoT5LlmRanker
from llmrankers.setwise import SetwiseLlmRanker, OpenAiSetwiseLlmRanker, _is_multimodal_config
from llmrankers.setwise_extended import (
    BiasAwareDualEndSetwiseLlmRanker,
    BidirectionalEnsembleRanker,
    BottomUpSetwiseLlmRanker,
    DualEndSetwiseLlmRanker,
    MaxContextBottomUpSetwiseLlmRanker,
    MaxContextDualEndSetwiseLlmRanker,
    MaxContextTopDownSetwiseLlmRanker,
    SameCallRegularizedSetwiseLlmRanker,
    SelectiveDualEndSetwiseLlmRanker,
)
from llmrankers.experiment_controls import (
    QueryCheckpointStore,
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    read_jsonl,
    sha256_file,
    validate_telemetry,
)
from llmrankers.pairwise import PairwiseLlmRanker, DuoT5LlmRanker, OpenAiPairwiseLlmRanker
from llmrankers.listwise import OpenAiListwiseLlmRanker, ListwiseLlmRanker
from tqdm import tqdm
from transformers import AutoConfig
import argparse
import hashlib
import os
import sys
import json
import time
import random
import torch
random.seed(929)
logger = logging.getLogger(__name__)


MAXCONTEXT_DIRECTIONS = {
    "maxcontext_dualend",
    "maxcontext_topdown",
    "maxcontext_bottomup",
}


def validate_character_scheme_args(args):
    if args.setwise and args.setwise.character_scheme != 'letters_a_w':
        if args.setwise.direction not in ('topdown', 'bottomup'):
            raise ValueError(
                f"--character_scheme {args.setwise.character_scheme} requires "
                f"--direction topdown or bottomup; got --direction {args.setwise.direction}."
            )
        if args.run.openai_key is not None:
            raise ValueError(
                f"--character_scheme {args.setwise.character_scheme} is not supported with --openai_key."
            )


def parse_args(parser, commands):
    # Divide argv by commands
    split_argv = [[]]
    for c in sys.argv[1:]:
        if c in commands.choices:
            split_argv.append([c])
        else:
            split_argv[-1].append(c)
    # Initialize namespace
    args = argparse.Namespace()
    for c in commands.choices:
        setattr(args, c, None)
    # Parse each command
    parser.parse_args(split_argv[0], namespace=args)  # Without command
    for argv in split_argv[1:]:  # Commands
        n = argparse.Namespace()
        setattr(args, argv[0], n)
        parser.parse_args(argv, namespace=n)
    return args




def write_run_file(path, results, tag, output_depth=None):
    with open(path, 'w') as f:
        for qid, _, ranking in results:
            rank = 1
            for doc in ranking[:output_depth]:
                docid = doc.docid
                score = doc.score
                f.write(f"{qid}\tQ0\t{docid}\t{rank}\t{score}\t{tag}\n")
                rank += 1


def derive_sidecar_path(save_path, suffix):
    if save_path.endswith(".txt"):
        return save_path[:-4] + suffix
    return save_path + suffix


def _load_experiment_metadata(path):
    if path is None:
        return {}
    with open(path, encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict):
        raise ValueError("--experiment_metadata must contain one JSON object.")
    return metadata


def _validate_raw_capture(path, telemetry):
    """Prove that comparison logs account for every nominal/retry generation."""
    rows = read_jsonl(path)
    primary = [row for row in rows if row.get("type") != "dual_worst"]
    attempts = []
    for row in primary:
        raw_attempts = row.get("raw_attempts")
        if not isinstance(raw_attempts, list) or not raw_attempts:
            raise ValueError(f"Missing raw-attempt list for qid={telemetry['qid']!r}.")
        for expected_index, attempt in enumerate(raw_attempts):
            required = {
                "attempt_index", "raw_output", "prompt_tokens", "completion_tokens",
                "accepted", "parse_result", "parse_reason",
            }
            if not required.issubset(attempt):
                raise ValueError(f"Incomplete raw-attempt record for qid={telemetry['qid']!r}.")
            if attempt["attempt_index"] != expected_index:
                raise ValueError(f"Non-sequential raw-attempt indices for qid={telemetry['qid']!r}.")
            if expected_index < len(raw_attempts) - 1 and attempt["accepted"]:
                raise ValueError(f"A retried generation is marked accepted for qid={telemetry['qid']!r}.")
            attempts.append(attempt)
    if len(attempts) != telemetry["retry_inclusive_llm_calls"]:
        raise ValueError(
            f"Raw-capture call mismatch for qid={telemetry['qid']!r}: "
            f"captured={len(attempts)}, charged={telemetry['retry_inclusive_llm_calls']}"
        )
    if sum(int(row["prompt_tokens"]) for row in attempts) != telemetry["actual_prompt_tokens"]:
        raise ValueError(f"Raw-capture prompt-token mismatch for qid={telemetry['qid']!r}.")
    if sum(int(row["completion_tokens"]) for row in attempts) != telemetry["actual_completion_tokens"]:
        raise ValueError(f"Raw-capture completion-token mismatch for qid={telemetry['qid']!r}.")


def _experiment_protocol(args, metadata):
    run_options = dict(vars(args.run))
    # Never persist or hash a secret. API experiments have their own official
    # wrapper and the presence/absence of a key is not a ranking protocol.
    run_options.pop("openai_key", None)
    # Resuming changes execution control, not the experiment protocol.
    run_options.pop("resume", None)
    return {
        "schema_version": 1,
        "run": run_options,
        "ranker": dict(vars(args.setwise)) if args.setwise is not None else None,
        "first_stage_sha256": sha256_file(args.run.run_path),
        "metadata": metadata,
    }


def _attach_experiment_prompt_template_contract(args, ranker, metadata):
    """Freeze one fully rendered, model-specific prompt-shape probe in metadata."""
    if (
        args.run.experiment_metadata is None
        or args.setwise is None
        or args.setwise.direction not in MAXCONTEXT_DIRECTIONS
    ):
        return metadata
    probe_docs = [
        SearchResult(
            docid=f"__template_doc_{index + 1}__",
            score=float(args.run.hits - index),
            text=f"{{{{ passage_{index + 1}_text }}}}",
        )
        for index in range(args.run.hits)
    ]
    saved = {
        "characters": list(ranker.CHARACTERS),
        "label_scheme": getattr(ranker, "label_scheme", None),
        "qid": getattr(ranker, "_current_qid", None),
    }
    try:
        ranker._current_qid = "__template_query__"
        if args.setwise.direction == 'maxcontext_dualend':
            builder = "dual"
            instruction = ranker._build_dual_prompt_text("{{ query_text }}", probe_docs)
            rendered = ranker._build_chat_prompt([{"role": "user", "content": instruction}])
        else:
            builder = "worst" if args.setwise.direction == "maxcontext_bottomup" else "best"
            prompt_builder = ranker._build_worst_prompt if builder == "worst" else ranker._build_best_prompt
            instruction = prompt_builder("{{ query_text }}", probe_docs)
            rendered = ranker._build_chat_prompt([{"role": "user", "content": instruction}])
            rendered += " Passage:"
    finally:
        ranker.CHARACTERS = saved["characters"]
        ranker.label_scheme = saved["label_scheme"]
        ranker._current_qid = saved["qid"]
    contract = {
        "schema_version": 1,
        "builder": builder,
        "pool_size": args.run.hits,
        "prompt_variant": "canonical",
        "label_scheme": "sequential",
        "instruction_prompt": instruction,
        "rendered_chat_prompt": rendered,
        "rendered_chat_prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    }
    updated = dict(metadata)
    updated["prompt_template_contract"] = contract
    atomic_write_json(args.run.experiment_metadata, updated)
    return updated


_PARSE_COUNTERS = (
    "total_parse_fallback",
    "total_lenient_fallback",
    "total_strict_parse_fallback",
    "total_lexical_refusal_fallback",
    "total_numeric_out_of_range_fallback",
    "total_degenerate_repetition_fallback",
    "total_unparseable_after_exhaustion_fallback",
    "total_duplicate_label_fallback",
    "total_out_of_window_label_fallback",
    "total_parse_failure_strict",
    "total_parse_failure_bm25_fallback",
)


def _build_experiment_telemetry(
    args,
    ranker,
    metadata,
    *,
    qid,
    live_pool_size,
    wall_seconds,
    peak_allocated,
    peak_reserved,
    status="ok",
    error=None,
):
    nominal_prompt = int(getattr(ranker, "total_prompt_tokens", 0))
    nominal_completion = int(getattr(ranker, "total_completion_tokens", 0))
    retry_prompt = int(getattr(ranker, "total_retry_prompt_tokens", 0))
    retry_completion = int(getattr(ranker, "total_retry_completion_tokens", 0))
    llm_calls = int(getattr(ranker, "total_compare", 0))
    retries = int(getattr(ranker, "total_retries", 0))
    bypasses = int(
        getattr(ranker, "total_non_llm_bypasses", getattr(ranker, "total_bm25_bypass", 0))
    )
    selection_steps = int(getattr(ranker, "total_selection_steps", llm_calls + bypasses))
    if selection_steps != llm_calls + bypasses:
        # Non-MaxContext rankers do not expose algorithmic steps. Preserve a
        # self-consistent conservative definition rather than inventing depth.
        selection_steps = llm_calls + bypasses
    output_depth = args.run.output_depth or min(args.run.hits, getattr(args.setwise, "k", args.run.hits))
    objective = metadata.get("output_objective")
    if objective is None:
        objective = "top10" if output_depth == 10 and output_depth < args.run.hits else "full"
    parse = {name.removeprefix("total_"): int(getattr(ranker, name, 0)) for name in _PARSE_COUNTERS}
    row = {
        "schema_version": 1,
        "qid": str(qid),
        "method": metadata.get(
            "method", getattr(args.setwise, "direction", type(ranker).__name__)
        ),
        "model": args.run.model_name_or_path,
        "dataset": metadata.get("dataset"),
        "pool_cap": args.run.hits,
        "live_pool_size": live_pool_size,
        "output_objective": objective,
        "output_depth": output_depth,
        "llm_calls": llm_calls,
        "selection_steps": selection_steps,
        "non_llm_bypasses": bypasses,
        "nominal_serial_depth": selection_steps,
        "observed_serial_depth": llm_calls,
        "retry_inclusive_llm_calls": llm_calls + retries,
        "retry_inclusive_serial_depth": selection_steps + retries,
        "nominal_prompt_tokens": nominal_prompt,
        "nominal_completion_tokens": nominal_completion,
        "nominal_total_tokens": nominal_prompt + nominal_completion,
        "retry_prompt_tokens": retry_prompt,
        "retry_completion_tokens": retry_completion,
        "retry_total_tokens": retry_prompt + retry_completion,
        "actual_prompt_tokens": nominal_prompt + retry_prompt,
        "actual_completion_tokens": nominal_completion + retry_completion,
        "actual_total_tokens": nominal_prompt + nominal_completion + retry_prompt + retry_completion,
        "wall_seconds": wall_seconds,
        "retry_seconds": float(getattr(ranker, "total_retry_overhead_seconds", 0.0)),
        "retry_excluded_wall_seconds": max(
            0.0,
            wall_seconds - float(getattr(ranker, "total_retry_overhead_seconds", 0.0)),
        ),
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "context_limit": getattr(ranker, "max_input_tokens", None),
        "max_rendered_prompt_tokens": getattr(ranker, "max_rendered_prompt_tokens", None),
        "min_context_fit_margin": getattr(ranker, "min_context_fit_margin", None),
        "rendered_prompt_hashes": list(getattr(ranker, "rendered_prompt_hashes", [])),
        "batch_id": None,
        "batch_size": None,
        "batch_wall_seconds": None,
        "amortized_seconds_per_query": None,
        "status": status,
        "error": error,
        "parse": parse,
        "provenance": metadata,
    }
    validate_telemetry(row)
    return row


def main(args):
    # Preserve compatibility with programmatic/legacy callers constructed before
    # the experiment CLI fields existed. argparse namespaces already contain
    # these values; this only fills absent attributes on external namespaces.
    for name, default in {
        "max_queries": None,
        "output_depth": None,
        "resume": False,
        "checkpoint_dir": None,
        "capture_raw_responses": False,
        "experiment_metadata": None,
        "telemetry_path": None,
        "model_revision": None,
        "tokenizer_revision": None,
    }.items():
        if not hasattr(args.run, name):
            setattr(args.run, name, default)
    if args.run.max_queries is not None and args.run.max_queries < 1:
        raise ValueError("--max_queries must be positive when provided.")
    if args.run.output_depth is not None:
        if not 1 <= args.run.output_depth <= args.run.hits:
            raise ValueError("--output_depth must satisfy 1 <= output_depth <= hits.")
        if args.setwise is not None and args.run.output_depth > args.setwise.k:
            raise ValueError("--output_depth cannot exceed setwise --k.")
    if args.run.resume and args.run.checkpoint_dir is None:
        raise ValueError("--resume requires --checkpoint_dir.")
    if args.run.capture_raw_responses and not args.run.log_comparisons:
        raise ValueError("--capture_raw_responses requires --log_comparisons.")
    if args.run.capture_raw_responses and args.run.checkpoint_dir is None:
        raise ValueError("--capture_raw_responses requires --checkpoint_dir for per-query auditing.")
    if args.setwise is not None and args.setwise.direction in MAXCONTEXT_DIRECTIONS:
        if args.run.openai_key is not None:
            raise ValueError(
                f"--direction {args.setwise.direction} is not supported with --openai_key. "
                f"MaxContext requires a local Qwen3 / Qwen3.5 / Llama-3.1 / Ministral-3 model."
            )

    if args.run.shuffle and args.run.reverse:
        raise SystemExit("Error: --shuffle and --reverse are mutually exclusive.")

    if args.run.shuffle or args.run.reverse:
        setwise_args = getattr(args, "setwise", None)
        direction = getattr(setwise_args, "direction", None) if setwise_args is not None else None
        if direction not in MAXCONTEXT_DIRECTIONS:
            raise SystemExit(
                "Error: --shuffle / --reverse only supported for MaxContext setwise "
                f"directions ({sorted(MAXCONTEXT_DIRECTIONS)}). Got direction={direction!r}."
            )

    peek_config = None

    def validate_local_multimodal_config():
        nonlocal peek_config
        if args.run.openai_key is not None:
            return None
        if peek_config is None:
            peek_config = AutoConfig.from_pretrained(
                args.run.model_name_or_path,
                cache_dir=args.run.cache_dir,
                revision=args.run.model_revision,
                trust_remote_code=True,
            )
        if _is_multimodal_config(peek_config):
            if args.run.scoring == "likelihood":
                raise SystemExit(
                    f"Error: --scoring likelihood is not supported for multimodal "
                    f"model_type={peek_config.model_type!r}. Use --scoring generation."
                )
            logger.warning(
                "Model type %s is multimodal; vision inputs are unused for text-only IR reranking.",
                peek_config.model_type,
            )
        return peek_config

    if args.pointwise:
        validate_local_multimodal_config()
        if 'monot5' in args.run.model_name_or_path:
            ranker = MonoT5LlmRanker(model_name_or_path=args.run.model_name_or_path,
                                     tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                     device=args.run.device,
                                     cache_dir=args.run.cache_dir,
                                     method=args.pointwise.method,
                                     batch_size=args.pointwise.batch_size)
        else:
            ranker = PointwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                        tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                        device=args.run.device,
                                        cache_dir=args.run.cache_dir,
                                        method=args.pointwise.method,
                                        batch_size=args.pointwise.batch_size)

    elif args.setwise:
        if args.run.openai_key:
            ranker = OpenAiSetwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                            api_key=args.run.openai_key,
                                            num_child=args.setwise.num_child,
                                            method=args.setwise.method,
                                            k=args.setwise.k)
        elif args.setwise.direction == 'topdown':
            validate_local_multimodal_config()
            ranker = SetwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                      tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                      device=args.run.device,
                                      cache_dir=args.run.cache_dir,
                                      num_child=args.setwise.num_child,
                                      scoring=args.run.scoring,
                                      character_scheme=args.setwise.character_scheme,
                                      method=args.setwise.method,
                                      num_permutation=args.setwise.num_permutation,
                                      k=args.setwise.k,
                                      model_revision=args.run.model_revision,
                                      tokenizer_revision=args.run.tokenizer_revision)
        elif args.setwise.direction == 'bottomup':
            validate_local_multimodal_config()
            ranker = BottomUpSetwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                              tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                              device=args.run.device,
                                              cache_dir=args.run.cache_dir,
                                              num_child=args.setwise.num_child,
                                              scoring=args.run.scoring,
                                              character_scheme=args.setwise.character_scheme,
                                              method=args.setwise.method,
                                              num_permutation=args.setwise.num_permutation,
                                              k=args.setwise.k)
        elif args.setwise.direction == 'dualend':
            validate_local_multimodal_config()
            ranker = DualEndSetwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                             tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                             device=args.run.device,
                                             cache_dir=args.run.cache_dir,
                                             num_child=args.setwise.num_child,
                                             scoring=args.run.scoring,
                                             method=args.setwise.method,
                                             num_permutation=args.setwise.num_permutation,
                                             k=args.setwise.k)
        elif args.setwise.direction == 'selective_dualend':
            validate_local_multimodal_config()
            ranker = SelectiveDualEndSetwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                                      tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                                      device=args.run.device,
                                                      cache_dir=args.run.cache_dir,
                                                      num_child=args.setwise.num_child,
                                                      scoring=args.run.scoring,
                                                      method=args.setwise.method,
                                                      num_permutation=args.setwise.num_permutation,
                                                      k=args.setwise.k,
                                                      gate_strategy=args.setwise.gate_strategy,
                                                      shortlist_size=args.setwise.shortlist_size,
                                                      margin_threshold=args.setwise.margin_threshold,
                                                      uncertainty_percentile=args.setwise.uncertainty_percentile)
        elif args.setwise.direction == 'bias_aware_dualend':
            if args.setwise.method == 'heapsort':
                raise ValueError(
                    'bias_aware_dualend supports only bubblesort and selection; '
                    'heapsort bypasses the order-robust joint prompting path.'
                )
            validate_local_multimodal_config()
            ranker = BiasAwareDualEndSetwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                                      tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                                      device=args.run.device,
                                                      cache_dir=args.run.cache_dir,
                                                      num_child=args.setwise.num_child,
                                                      scoring=args.run.scoring,
                                                      method=args.setwise.method,
                                                      num_permutation=args.setwise.num_permutation,
                                                      k=args.setwise.k,
                                                      gate_strategy=args.setwise.gate_strategy,
                                                      shortlist_size=args.setwise.shortlist_size,
                                                      margin_threshold=args.setwise.margin_threshold,
                                                      uncertainty_percentile=args.setwise.uncertainty_percentile,
                                                      order_robust_orderings=args.setwise.order_robust_orderings)
        elif args.setwise.direction == 'samecall_regularized':
            validate_local_multimodal_config()
            ranker = SameCallRegularizedSetwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                                         tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                                         device=args.run.device,
                                                         cache_dir=args.run.cache_dir,
                                                         num_child=args.setwise.num_child,
                                                         scoring=args.run.scoring,
                                                         method=args.setwise.method,
                                                         num_permutation=args.setwise.num_permutation,
                                                         k=args.setwise.k)
        elif args.setwise.direction == 'bidirectional':
            validate_local_multimodal_config()
            ranker = BidirectionalEnsembleRanker(model_name_or_path=args.run.model_name_or_path,
                                                tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                                device=args.run.device,
                                                num_child=args.setwise.num_child,
                                                k=args.setwise.k,
                                                scoring=args.run.scoring,
                                                method=args.setwise.method,
                                                num_permutation=args.setwise.num_permutation,
                                                fusion=args.setwise.fusion,
                                                alpha=args.setwise.alpha,
                                                cache_dir=args.run.cache_dir)
        elif args.setwise.direction == 'maxcontext_dualend':
            if args.run.hits != args.setwise.k:
                raise ValueError(
                    f"{args.setwise.direction} requires --hits == --k (pool_size)."
                )
            if args.run.scoring != "generation":
                raise ValueError(
                    f"{args.setwise.direction} requires --scoring generation."
                )
            if args.setwise.num_permutation != 1:
                raise ValueError(
                    f"{args.setwise.direction} requires --num_permutation 1."
                )
            if args.setwise.method != "selection":
                raise ValueError(
                    f"{args.setwise.direction} requires --method selection."
                )
            validate_local_multimodal_config()
            ranker_class = MaxContextDualEndSetwiseLlmRanker
            ranker = ranker_class(
                model_name_or_path=args.run.model_name_or_path,
                tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                device=args.run.device,
                cache_dir=args.run.cache_dir,
                num_child=args.setwise.num_child,
                scoring=args.run.scoring,
                method=args.setwise.method,
                num_permutation=args.setwise.num_permutation,
                k=args.setwise.k,
                pool_size=args.setwise.k,
                shuffle=args.run.shuffle,
                reverse=args.run.reverse,
                allow_parse_failure_bm25_fallback=args.run.allow_parse_failure_bm25_fallback,
                output_depth=args.run.output_depth,
                capture_raw_responses=args.run.capture_raw_responses,
                model_revision=args.run.model_revision,
                tokenizer_revision=args.run.tokenizer_revision,
            )
        elif args.setwise.direction == 'maxcontext_topdown':
            if args.run.hits != args.setwise.k:
                raise ValueError(
                    "maxcontext_topdown requires --hits == --k (pool_size)."
                )
            if args.run.scoring != "generation":
                raise ValueError(
                    "maxcontext_topdown requires --scoring generation."
                )
            if args.setwise.num_permutation != 1:
                raise ValueError(
                    "maxcontext_topdown requires --num_permutation 1."
                )
            if args.setwise.method != "selection":
                raise ValueError(
                    "maxcontext_topdown requires --method selection."
                )
            validate_local_multimodal_config()
            ranker = MaxContextTopDownSetwiseLlmRanker(
                model_name_or_path=args.run.model_name_or_path,
                tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                device=args.run.device,
                cache_dir=args.run.cache_dir,
                num_child=args.setwise.num_child,
                scoring=args.run.scoring,
                method=args.setwise.method,
                num_permutation=args.setwise.num_permutation,
                k=args.setwise.k,
                pool_size=args.setwise.k,
                shuffle=args.run.shuffle,
                reverse=args.run.reverse,
                allow_parse_failure_bm25_fallback=args.run.allow_parse_failure_bm25_fallback,
                output_depth=args.run.output_depth,
                capture_raw_responses=args.run.capture_raw_responses,
                model_revision=args.run.model_revision,
                tokenizer_revision=args.run.tokenizer_revision,
            )
        elif args.setwise.direction == 'maxcontext_bottomup':
            if args.run.hits != args.setwise.k:
                raise ValueError(
                    "maxcontext_bottomup requires --hits == --k (pool_size)."
                )
            if args.run.scoring != "generation":
                raise ValueError(
                    "maxcontext_bottomup requires --scoring generation."
                )
            if args.setwise.num_permutation != 1:
                raise ValueError(
                    "maxcontext_bottomup requires --num_permutation 1."
                )
            if args.setwise.method != "selection":
                raise ValueError(
                    "maxcontext_bottomup requires --method selection."
                )
            validate_local_multimodal_config()
            ranker = MaxContextBottomUpSetwiseLlmRanker(
                model_name_or_path=args.run.model_name_or_path,
                tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                device=args.run.device,
                cache_dir=args.run.cache_dir,
                num_child=args.setwise.num_child,
                scoring=args.run.scoring,
                method=args.setwise.method,
                num_permutation=args.setwise.num_permutation,
                k=args.setwise.k,
                pool_size=args.setwise.k,
                shuffle=args.run.shuffle,
                reverse=args.run.reverse,
                allow_parse_failure_bm25_fallback=args.run.allow_parse_failure_bm25_fallback,
                output_depth=args.run.output_depth,
                capture_raw_responses=args.run.capture_raw_responses,
                model_revision=args.run.model_revision,
                tokenizer_revision=args.run.tokenizer_revision,
            )
        else:
            raise ValueError(f'Unknown direction: {args.setwise.direction}')

    elif args.pairwise:
        if args.pairwise.method != 'allpair':
            args.pairwise.batch_size = 2
            logger.info(f'Setting batch_size to 2.')

        if args.run.openai_key:
            ranker = OpenAiPairwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                             api_key=args.run.openai_key,
                                             method=args.pairwise.method,
                                             k=args.pairwise.k)

        elif 'duot5' in args.run.model_name_or_path:
            validate_local_multimodal_config()
            ranker = DuoT5LlmRanker(model_name_or_path=args.run.model_name_or_path,
                                    tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                    device=args.run.device,
                                    cache_dir=args.run.cache_dir,
                                    method=args.pairwise.method,
                                    batch_size=args.pairwise.batch_size,
                                    k=args.pairwise.k)
        else:
            validate_local_multimodal_config()
            ranker = PairwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                       tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                       device=args.run.device,
                                       cache_dir=args.run.cache_dir,
                                       method=args.pairwise.method,
                                       batch_size=args.pairwise.batch_size,
                                       k=args.pairwise.k)

    elif args.listwise:
        if args.run.openai_key:
            ranker = OpenAiListwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                             api_key=args.run.openai_key,
                                             window_size=args.listwise.window_size,
                                             step_size=args.listwise.step_size,
                                             num_repeat=args.listwise.num_repeat)
        else:
            validate_local_multimodal_config()
            ranker = ListwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                       tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                       device=args.run.device,
                                       cache_dir=args.run.cache_dir,
                                       window_size=args.listwise.window_size,
                                       step_size=args.listwise.step_size,
                                       scoring=args.run.scoring,
                                       num_repeat=args.listwise.num_repeat)
    else:
        raise ValueError('Must specify either --pointwise, --setwise, --pairwise or --listwise.')

    experiment_metadata = _attach_experiment_prompt_template_contract(
        args, ranker, _load_experiment_metadata(args.run.experiment_metadata)
    )

    # Set up comparison logging for position bias analysis
    comparison_output_path = args.run.log_comparisons
    if args.run.log_comparisons:
        if args.run.checkpoint_dir is not None:
            # Checkpointed runs log to per-qid fragments. Failed-query fragments
            # are replaced on resume and only successful fragments are merged.
            ranker._comparison_log_path = None
        elif args.run.resume:
            ranker._comparison_log_path = args.run.log_comparisons
            os.makedirs(os.path.dirname(os.path.abspath(args.run.log_comparisons)), exist_ok=True)
            open(args.run.log_comparisons, 'a').close()
        else:
            # Preserve legacy overwrite behavior for non-resumed runs.
            ranker._comparison_log_path = args.run.log_comparisons
            open(args.run.log_comparisons, 'w').close()
    else:
        ranker._comparison_log_path = None

    query_map = {}
    if args.run.ir_dataset_name is not None:
        dataset = ir_datasets.load(args.run.ir_dataset_name)
        for query in dataset.queries_iter():
            qid = query.query_id
            text = query.text
            query_map[qid] = ranker.truncate(text, args.run.query_length)
        dataset = ir_datasets.load(args.run.ir_dataset_name)
        docstore = dataset.docs_store()
    else:
        topics = get_topics(args.run.pyserini_index+'-test')
        for topic_id in list(topics.keys()):
            text = topics[topic_id]['title']
            query_map[str(topic_id)] = ranker.truncate(text, args.run.query_length)
        docstore = LuceneSearcher.from_prebuilt_index(args.run.pyserini_index+'.flat')

    logger.info(f'Loading first stage run from {args.run.run_path}.')
    first_stage_rankings = []
    with open(args.run.run_path, 'r') as f:
        current_qid = None
        current_ranking = []
        for line in tqdm(f):
            qid, _, docid, _, score, _ = line.strip().split()
            if qid != current_qid:
                if current_qid is not None:
                    first_stage_rankings.append((current_qid, query_map[current_qid], current_ranking[:args.run.hits]))
                current_ranking = []
                current_qid = qid
            if len(current_ranking) >= args.run.hits:
                continue
            if args.run.ir_dataset_name is not None:
                text = docstore.get(docid).text
                if 'title' in dir(docstore.get(docid)):
                    text = f'{docstore.get(docid).title} {text}'
            else:
                data = json.loads(docstore.doc(docid).raw())
                text = data['text']
                if 'title' in data:
                    text = f'{data["title"]} {text}'
            text = ranker.truncate(text, args.run.passage_length)
            current_ranking.append(SearchResult(docid=docid, score=float(score), text=text))
        first_stage_rankings.append((current_qid, query_map[current_qid], current_ranking[:args.run.hits]))

    if args.run.max_queries is not None:
        first_stage_rankings = first_stage_rankings[:args.run.max_queries]

    effective_output_depth = args.run.output_depth or args.run.hits
    checkpoint_store = None
    if args.run.checkpoint_dir is not None:
        checkpoint_store = QueryCheckpointStore.open(
            args.run.checkpoint_dir,
            _experiment_protocol(args, experiment_metadata),
            resume=args.run.resume,
        )

    telemetry_by_qid = {}
    if args.run.telemetry_path and os.path.exists(args.run.telemetry_path):
        existing_rows = read_jsonl(args.run.telemetry_path)
        for row in existing_rows:
            validate_telemetry(row)
            if row["status"] == "ok":
                if row["qid"] in telemetry_by_qid:
                    raise ValueError(f"Duplicate completed telemetry qid: {row['qid']}")
                telemetry_by_qid[row["qid"]] = row
        if existing_rows and not args.run.resume:
            raise FileExistsError(
                f"Telemetry path {args.run.telemetry_path} is non-empty; use --resume or a new attempt."
            )

    reranked_results = []
    total_comparisons = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_bm25_bypasses = 0
    total_retries_across_queries = 0
    total_retry_overhead_seconds = 0.0
    total_query_wall_seconds = 0.0
    optional_stat_labels = {
        "total_dual_invocations": "dual invocations",
        "total_single_invocations": "single invocations",
        "total_order_robust_windows": "order-robust windows",
        "total_extra_orderings": "extra orderings",
        "total_regularized_worst_moves": "regularized worst moves",
        "total_parse_fallback": "parse fallbacks",
        "total_lenient_fallback": "lenient fallbacks",
        "total_strict_parse_fallback": "strict parse fallbacks",
        "total_lexical_refusal_fallback": "lexical refusal fallbacks",
        "total_numeric_out_of_range_fallback": "numeric out-of-range fallbacks",
        "total_degenerate_repetition_fallback": "degenerate repetition fallbacks",
        "total_unparseable_after_exhaustion_fallback": "unparseable after exhaustion fallbacks",
        "total_duplicate_label_fallback": "duplicate label fallbacks",
        "total_out_of_window_label_fallback": "out-of-window label fallbacks",
        "total_parse_failure_strict": "parse_failure_strict",
        "total_parse_failure_bm25_fallback": "parse_failure_bm25_fallback",
    }
    optional_stat_totals = {
        attr: 0.0
        for attr in optional_stat_labels
        if hasattr(ranker, attr)
    }

    tic = time.time()
    # Section G: log the model's effective repetition_penalty so we can
    # correlate sticky-loop behavior with model-family generation_config defaults.
    if hasattr(ranker, "llm") and hasattr(ranker.llm, "generation_config"):
        rep_pen = getattr(ranker.llm.generation_config, "repetition_penalty", None)
        print(
            f"[run] model={args.run.model_name_or_path!r} configured repetition_penalty={rep_pen!r}",
            file=sys.stderr,
        )

    for qid, query, ranking in tqdm(first_stage_rankings):
        checkpoint = None
        checkpoint_depth = min(effective_output_depth, len(ranking))
        if checkpoint_store is not None and args.run.resume:
            checkpoint = checkpoint_store.load(qid, checkpoint_depth)
        if checkpoint is not None:
            row = checkpoint["telemetry"]
            if args.run.capture_raw_responses:
                _validate_raw_capture(
                    checkpoint_store.path_for(qid).with_suffix(".comparisons.jsonl"), row
                )
            if qid in telemetry_by_qid and telemetry_by_qid[qid] != row:
                raise ValueError(f"Telemetry/checkpoint mismatch for qid={qid!r}.")
            if args.run.telemetry_path and qid not in telemetry_by_qid:
                append_jsonl(args.run.telemetry_path, row)
                telemetry_by_qid[qid] = row
            restored = [
                SearchResult(docid=docid, score=-rank, text=None)
                for rank, docid in enumerate(checkpoint["docids"], start=1)
            ]
            reranked_results.append((qid, query, restored))
            total_comparisons += row["llm_calls"]
            total_prompt_tokens += row["nominal_prompt_tokens"]
            total_completion_tokens += row["nominal_completion_tokens"]
            total_retries_across_queries += row["retry_inclusive_llm_calls"] - row["llm_calls"]
            total_retry_overhead_seconds += row["retry_seconds"]
            total_query_wall_seconds += row["wall_seconds"]
            for attr in optional_stat_totals:
                optional_stat_totals[attr] += row.get("parse", {}).get(attr.removeprefix("total_"), 0)
            total_bm25_bypasses += row["non_llm_bypasses"]
            continue
        if args.run.shuffle_ranking is not None:
            if args.run.shuffle_ranking == 'random':
                random.shuffle(ranking)
            elif args.run.shuffle_ranking == 'inverse':
                ranking = ranking[::-1]
            else:
                raise ValueError(f'Invalid shuffle ranking method: {args.run.shuffle_ranking}.')
        ranker._current_qid = qid
        if checkpoint_store is not None and comparison_output_path:
            comparison_fragment = checkpoint_store.path_for(qid).with_suffix(".comparisons.jsonl")
            comparison_fragment.parent.mkdir(parents=True, exist_ok=True)
            if comparison_fragment.exists():
                comparison_fragment.unlink()
            comparison_fragment.touch()
            ranker._comparison_log_path = str(comparison_fragment)
        cuda_active = args.run.device.startswith("cuda") and torch.cuda.is_available()
        if cuda_active:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        query_start = time.perf_counter()
        try:
            reranked = ranker.rerank(query, ranking)
            if cuda_active:
                torch.cuda.synchronize()
            query_wall = time.perf_counter() - query_start
            peak_allocated = torch.cuda.max_memory_allocated() if cuda_active else 0
            peak_reserved = torch.cuda.max_memory_reserved() if cuda_active else 0
            telemetry = _build_experiment_telemetry(
                args,
                ranker,
                experiment_metadata,
                qid=qid,
                live_pool_size=len(ranking),
                wall_seconds=query_wall,
                peak_allocated=peak_allocated,
                peak_reserved=peak_reserved,
            )
            if args.run.capture_raw_responses:
                raw_capture_path = (
                    comparison_fragment
                    if checkpoint_store is not None
                    else args.run.log_comparisons
                )
                _validate_raw_capture(raw_capture_path, telemetry)
            emitted = reranked[:checkpoint_depth]
            if checkpoint_store is not None:
                checkpoint_store.save(qid, [doc.docid for doc in emitted], telemetry)
            if args.run.telemetry_path:
                if qid in telemetry_by_qid:
                    raise ValueError(f"Refusing duplicate telemetry qid={qid!r}.")
                append_jsonl(args.run.telemetry_path, telemetry)
                telemetry_by_qid[qid] = telemetry
            reranked_results.append((qid, query, reranked))
        except Exception as exc:
            if cuda_active:
                torch.cuda.synchronize()
            query_wall = time.perf_counter() - query_start
            if args.run.telemetry_path:
                failure = _build_experiment_telemetry(
                    args,
                    ranker,
                    experiment_metadata,
                    qid=qid,
                    live_pool_size=len(ranking),
                    wall_seconds=query_wall,
                    peak_allocated=torch.cuda.max_memory_allocated() if cuda_active else 0,
                    peak_reserved=torch.cuda.max_memory_reserved() if cuda_active else 0,
                    status="error",
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
                append_jsonl(args.run.telemetry_path, failure)
            if isinstance(exc, ValueError) and (
                "parse failed" in str(exc) or "Duplicate best/worst label" in str(exc)
            ):
                completed_queries = len(reranked_results)
                cumulative_retries = total_retries_across_queries + getattr(ranker, "total_retries", 0)
                print(
                    f"\n[run.py] strict parse failure on qid={qid!r} after "
                    f"{completed_queries} completed queries. Cumulative retries: "
                    f"{cumulative_retries}.",
                    file=sys.stderr,
                )
            raise
        total_query_wall_seconds += query_wall
        total_comparisons += ranker.total_compare
        total_prompt_tokens += ranker.total_prompt_tokens
        total_completion_tokens += ranker.total_completion_tokens
        total_retries_across_queries += getattr(ranker, "total_retries", 0)
        total_retry_overhead_seconds += getattr(ranker, "total_retry_overhead_seconds", 0.0)
        total_bm25_bypasses += getattr(ranker, "total_bm25_bypass", 0)
        for attr in optional_stat_totals:
            optional_stat_totals[attr] += getattr(ranker, attr, 0.0)
    toc = time.time()

    print(f'Avg comparisons: {total_comparisons/len(reranked_results)}')
    print(f'Avg prompt tokens: {total_prompt_tokens/len(reranked_results)}')
    print(f'Avg completion tokens: {total_completion_tokens/len(reranked_results)}')
    if total_bm25_bypasses > 0:
        print(f'Avg BM25 bypass: {total_bm25_bypasses / len(reranked_results)}')
    # Restored queries retain their measured time, even when resume skips inference.
    elapsed = total_query_wall_seconds if args.run.resume else toc - tic
    query_count = len(reranked_results)
    wall_clock_per_query = elapsed / query_count
    retry_overhead_per_query = total_retry_overhead_seconds / query_count
    retry_excluded_per_query = max(0.0, elapsed - total_retry_overhead_seconds) / query_count
    retries_per_query = total_retries_across_queries / query_count
    # Keep the legacy label for existing collectors, but make its semantics raw
    # wall-clock time and emit explicit timing diagnostics for ECIR analysis.
    print(f'Avg time per query: {wall_clock_per_query}')
    print(f'Avg wall-clock time per query: {wall_clock_per_query}')
    print(f'Avg retry-excluded time per query: {retry_excluded_per_query}')
    print(f'Avg retry overhead seconds per query: {retry_overhead_per_query}')
    print(f'Avg retries per query: {retries_per_query}')
    if total_retries_across_queries > 0:
        print(f'Total retries fired: {total_retries_across_queries}')
    for attr, total in optional_stat_totals.items():
        print(f'Avg {optional_stat_labels[attr]}: {total/len(reranked_results)}')

    write_run_file(
        args.run.save_path,
        reranked_results,
        'LLMRankers',
        output_depth=args.run.output_depth,
    )
    if checkpoint_store is not None and comparison_output_path:
        chunks = []
        for qid, _, _ in first_stage_rankings:
            fragment = checkpoint_store.path_for(qid).with_suffix(".comparisons.jsonl")
            if fragment.exists():
                chunks.append(fragment.read_text(encoding="utf-8"))
        atomic_write_text(comparison_output_path, "".join(chunks))
    if args.run.telemetry_path:
        all_rows = read_jsonl(args.run.telemetry_path)
        ok_rows = [row for row in all_rows if row["status"] == "ok"]
        if len(ok_rows) != len(first_stage_rankings):
            raise ValueError(
                f"Telemetry completeness failure: {len(ok_rows)} successful rows for "
                f"{len(first_stage_rankings)} queries."
            )
        summary = {
            "schema_version": 1,
            "queries": len(ok_rows),
            "output_depth": effective_output_depth,
            "llm_calls": sum(row["llm_calls"] for row in ok_rows),
            "actual_prompt_tokens": sum(row["actual_prompt_tokens"] for row in ok_rows),
            "actual_completion_tokens": sum(row["actual_completion_tokens"] for row in ok_rows),
            "wall_seconds": sum(row["wall_seconds"] for row in ok_rows),
        }
        atomic_write_json(derive_sidecar_path(args.run.save_path, "_summary.json"), summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(title='sub-commands')

    run_parser = commands.add_parser('run')
    run_parser.add_argument('--run_path', type=str, help='Path to the first stage run file (TREC format) to rerank.')
    run_parser.add_argument('--save_path', type=str, help='Path to save the reranked run file (TREC format).')
    run_parser.add_argument('--model_name_or_path', type=str,
                            help='Path to the pretrained model or model identifier from huggingface.co/models')
    run_parser.add_argument('--tokenizer_name_or_path', type=str, default=None,
                            help='Path to the pretrained tokenizer or tokenizer identifier from huggingface.co/tokenizers')
    run_parser.add_argument('--model_revision', type=str, default=None,
                            help='Immutable Hugging Face model commit used for loading and resume hashing.')
    run_parser.add_argument('--tokenizer_revision', type=str, default=None,
                            help='Immutable tokenizer commit; defaults to --model_revision in experiment launchers.')
    run_parser.add_argument('--ir_dataset_name', type=str, default=None)
    run_parser.add_argument('--pyserini_index', type=str, default=None)
    run_parser.add_argument('--hits', type=int, default=100)
    run_parser.add_argument('--max_queries', type=int, default=None,
                            help='Smoke runs: process the first N first-stage queries.')
    run_parser.add_argument('--query_length', type=int, default=128)
    run_parser.add_argument('--passage_length', type=int, default=128)
    run_parser.add_argument('--device', type=str, default='cuda')
    run_parser.add_argument('--cache_dir', type=str, default=None)
    run_parser.add_argument('--openai_key', type=str, default=None)
    run_parser.add_argument('--scoring', type=str, default='generation', choices=['generation', 'likelihood'])
    run_parser.add_argument('--shuffle_ranking', type=str, default=None, choices=['inverse', 'random'])
    run_parser.add_argument('--shuffle', action='store_true', default=False,
                            help='Per-round shuffle of remaining pool for MaxContext methods (fixed seed 929).')
    run_parser.add_argument('--reverse', action='store_true', default=False,
                            help='Per-round reverse ordering of remaining pool for MaxContext methods.')
    run_parser.add_argument('--log_comparisons', type=str, default=None,
                            help='Path to write per-comparison JSONL log for position bias analysis')
    run_parser.add_argument('--telemetry_path', type=str, default=None,
                            help='Experiments: append one validated JSON row per query.')
    run_parser.add_argument('--output_depth', type=int, default=None,
                            help='Stop compatible MaxContext rankers at this head depth and slice output.')
    run_parser.add_argument('--experiment_metadata', type=str, default=None,
                            help='JSON file containing dataset/provenance fields for experiment telemetry.')
    run_parser.add_argument('--checkpoint_dir', type=str, default=None,
                            help='Directory for atomic per-query checkpoints.')
    run_parser.add_argument('--resume', action='store_true', default=False,
                            help='Resume only checkpoints with an identical protocol hash.')
    run_parser.add_argument('--capture_raw_responses', action='store_true', default=False,
                            help='Include every raw generation in --log_comparisons.')
    run_parser.add_argument('--allow_parse_failure_bm25_fallback', action='store_true', default=False,
                            help='MaxContext only: when LLM label parsing fails for a single query, '
                                 'fall back to BM25 (first-stage) ordering for that query and increment '
                                 'total_parse_failure_bm25_fallback. Off by default — strict-raise behavior '
                                 'is preserved so smoke runs surface new parse-failure modes.')

    pointwise_parser = commands.add_parser('pointwise')
    pointwise_parser.add_argument('--method', type=str, default='yes_no',
                                  choices=['qlm', 'yes_no'])
    pointwise_parser.add_argument('--batch_size', type=int, default=2)

    pairwise_parser = commands.add_parser('pairwise')
    pairwise_parser.add_argument('--method', type=str, default='allpair',
                                 choices=['allpair', 'heapsort', 'bubblesort'])
    pairwise_parser.add_argument('--batch_size', type=int, default=2)
    pairwise_parser.add_argument('--k', type=int, default=10)

    setwise_parser = commands.add_parser('setwise')
    setwise_parser.add_argument('--num_child', type=int, default=3)
    setwise_parser.add_argument('--method', type=str, default='heapsort',
                                choices=['heapsort', 'bubblesort', 'selection'])
    setwise_parser.add_argument('--k', type=int, default=10)
    setwise_parser.add_argument('--num_permutation', type=int, default=1)
    setwise_parser.add_argument('--direction', type=str, default='topdown',
                                choices=['topdown', 'bottomup', 'dualend', 'selective_dualend',
                                         'bias_aware_dualend', 'samecall_regularized', 'bidirectional',
                                         'maxcontext_dualend', 'maxcontext_topdown',
                                         'maxcontext_bottomup'],
                                help='Ranking direction: topdown (standard), bottomup (reverse), '
                                     'dualend (simultaneous best-worst), selective_dualend '
                                     '(TopDown with selective joint prompting), bias_aware_dualend '
                                     '(order-robust joint prompting), samecall_regularized '
                                     '(TopDown with worst-signal regularization), bidirectional (ensemble), '
                                     'maxcontext_dualend (full-pool numeric DualEnd selection), '
                                     'maxcontext_topdown (full-pool numeric best-only selection), '
                                     'maxcontext_bottomup (full-pool numeric worst-only selection)')
    setwise_parser.add_argument('--character_scheme', type=str, default='letters_a_w',
                                choices=['letters_a_w', 'bigrams_aa_zz'])
    setwise_parser.add_argument('--fusion', type=str, default='rrf',
                                choices=['rrf', 'combsum', 'weighted'],
                                help='Fusion method for bidirectional ensemble')
    setwise_parser.add_argument('--alpha', type=float, default=0.5,
                                help='Weight for top-down in weighted fusion (bidirectional only)')
    setwise_parser.add_argument('--gate_strategy', type=str, default='hybrid',
                                choices=['off', 'shortlist', 'uncertain', 'hybrid'],
                                help='When to invoke extra DualEnd logic for selective/bias-aware variants '
                                     '(shortlist routing is ignored for selective heapsort)')
    setwise_parser.add_argument('--shortlist_size', type=int, default=20,
                                help='Prefix depth treated as near the top-k boundary for selective/bias-aware variants')
    setwise_parser.add_argument('--margin_threshold', type=float, default=0.15,
                                help='Backward-compatible alias for the uncertainty percentile; '
                                     '0.15 means the tightest 15%% of query-local BM25-spread windows')
    setwise_parser.add_argument('--uncertainty_percentile', type=float, default=None,
                                help='Query-local percentile cutoff for uncertainty gating; '
                                     '0.15 means the tightest 15%% of BM25-spread windows')
    setwise_parser.add_argument('--order_robust_orderings', type=int, default=3,
                                help='Number of controlled orderings for bias-aware DualEnd windows '
                                     '(bubblesort/selection only)')

    listwise_parser = commands.add_parser('listwise')
    listwise_parser.add_argument('--window_size', type=int, default=3)
    listwise_parser.add_argument('--step_size', type=int, default=1)
    listwise_parser.add_argument('--num_repeat', type=int, default=1)

    args = parse_args(parser, commands)

    if args.run.ir_dataset_name is not None and args.run.pyserini_index is not None:
        raise ValueError('Must specify either --ir_dataset_name or --pyserini_index, not both.')

    arg_dict = vars(args)
    if arg_dict['run'] is None or sum(arg_dict[arg] is not None for arg in arg_dict) != 2:
        raise ValueError(
            'Need to set --run and can only set one of --pointwise, --pairwise, '
            '--setwise or --listwise'
        )
    validate_character_scheme_args(args)
    main(args)
