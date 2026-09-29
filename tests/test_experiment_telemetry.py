import json
from types import MethodType, SimpleNamespace
from unittest import mock

import pytest
import torch

from llmrankers.experiment_controls import QueryCheckpointStore, validate_telemetry
from llmrankers.rankers import SearchResult
from llmrankers.setwise import SetwiseLlmRanker


def telemetry_row(qid="q1"):
    return {
        "schema_version": 1,
        "qid": qid,
        "method": "wp_t",
        "output_objective": "top10",
        "output_depth": 2,
        "llm_calls": 2,
        "retry_inclusive_llm_calls": 3,
        "selection_steps": 2,
        "non_llm_bypasses": 0,
        "nominal_prompt_tokens": 20,
        "nominal_completion_tokens": 2,
        "nominal_total_tokens": 22,
        "retry_prompt_tokens": 10,
        "retry_completion_tokens": 1,
        "retry_total_tokens": 11,
        "actual_prompt_tokens": 30,
        "actual_completion_tokens": 3,
        "actual_total_tokens": 33,
        "wall_seconds": 2.0,
        "retry_seconds": 0.5,
        "retry_excluded_wall_seconds": 1.5,
        "status": "ok",
    }


def test_retry_token_work_survives_nominal_counter_restore():
    ranker = SetwiseLlmRanker.__new__(SetwiseLlmRanker)
    ranker.total_compare = 3
    ranker.total_prompt_tokens = 20
    ranker.total_completion_tokens = 2
    ranker.total_parse_failure_strict = 0
    ranker.total_retry_prompt_tokens = 0
    ranker.total_retry_completion_tokens = 0
    snapshot = ranker._snapshot_retry_invisible_counters()
    ranker.total_compare += 1
    ranker.total_prompt_tokens += 11
    ranker.total_completion_tokens += 4
    ranker._restore_retry_invisible_counters(snapshot)
    assert (ranker.total_compare, ranker.total_prompt_tokens, ranker.total_completion_tokens) == (3, 20, 2)
    assert (ranker.total_retry_prompt_tokens, ranker.total_retry_completion_tokens) == (11, 4)


def test_failed_attempt_not_accepted_retry_is_charged_to_retry_time():
    ranker = SetwiseLlmRanker.__new__(SetwiseLlmRanker)
    ranker.num_permutation = 1
    ranker.scoring = "generation"
    ranker.config = SimpleNamespace(model_type="qwen3")
    ranker.CHARACTERS = ["1", "2"]
    ranker.total_compare = 0
    ranker.total_prompt_tokens = 0
    ranker.total_completion_tokens = 0
    ranker.total_parse_failure_strict = 0
    ranker.total_retry_prompt_tokens = 0
    ranker.total_retry_completion_tokens = 0
    ranker.total_retry_overhead_seconds = 0.0
    ranker.total_retries = 0
    ranker.capture_raw_responses = False
    ranker.strict_no_parse_fallback = True
    ranker._parse_failure_max_retries = 1
    ranker.label_scheme = "numeric_1_based"
    ranker._build_best_prompt = MethodType(lambda self, query, docs: "prompt", ranker)
    ranker._uses_chat_template = MethodType(lambda self: True, ranker)
    ranker._build_chat_prompt = MethodType(lambda self, messages: "chat", ranker)
    ranker._generation_budget = MethodType(lambda self, mode: 4, ranker)
    ranker._log_comparison = MethodType(lambda self, *args, **kwargs: None, ranker)

    class Tokenizer:
        all_special_tokens = []

        def __init__(self):
            self.outputs = iter(["not a label", "2"])

        def decode(self, *_args, **_kwargs):
            return next(self.outputs)

    ranker.tokenizer = Tokenizer()
    inputs = SimpleNamespace(input_ids=torch.tensor([[10, 11]]))
    ranker._tokenize_inputs = MethodType(
        lambda self, prompt, padding=False: inputs, ranker
    )
    ranker._generate = MethodType(
        lambda self, model_inputs, max_new_tokens, decoder_input_ids=None: torch.tensor(
            [[10, 11, 12]]
        ),
        ranker,
    )
    documents = [
        SearchResult(docid="d1", score=2.0, text="one"),
        SearchResult(docid="d2", score=1.0, text="two"),
    ]
    with mock.patch(
        "llmrankers.setwise.time.perf_counter", side_effect=[1.0, 1.4, 10.0]
    ):
        assert ranker.compare("query", documents) == "2"
    assert ranker.total_retries == 1
    assert ranker.total_retry_overhead_seconds == pytest.approx(0.4)
    assert ranker.total_retry_prompt_tokens == 2
    assert ranker.total_retry_completion_tokens == 1
    assert ranker.total_prompt_tokens == 2
    assert ranker.total_completion_tokens == 1


def test_checkpoint_requires_identical_protocol_and_valid_depth(tmp_path):
    store = QueryCheckpointStore.open(tmp_path / "checkpoint", {"cell": 1}, resume=False)
    row = telemetry_row()
    validate_telemetry(row)
    store.save("q1", ["d1", "d2"], row)
    assert store.load("q1", 2)["docids"] == ["d1", "d2"]
    resumed = QueryCheckpointStore.open(tmp_path / "checkpoint", {"cell": 1}, resume=True)
    assert resumed.load("q1", 2) is not None
    with pytest.raises(ValueError, match="protocol hash"):
        QueryCheckpointStore.open(tmp_path / "checkpoint", {"cell": 2}, resume=True)
    with pytest.raises(ValueError, match="row-count"):
        resumed.load("q1", 1)


def test_invalid_token_identity_is_rejected():
    row = telemetry_row()
    row["actual_total_tokens"] += 1
    with pytest.raises(ValueError):
        validate_telemetry(row)
