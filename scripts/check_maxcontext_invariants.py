#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import random
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _missing_dependency(*_args, **_kwargs):
    raise RuntimeError("Optional runtime dependency is not available in this test environment.")


try:
    import ir_datasets  # noqa: F401
except ModuleNotFoundError:
    ir_datasets_stub = ModuleType("ir_datasets")
    ir_datasets_stub.load = _missing_dependency
    sys.modules["ir_datasets"] = ir_datasets_stub

try:
    from pyserini.search.lucene import LuceneSearcher as _LuceneSearcher  # noqa: F401
    from pyserini.search._base import get_topics as _get_topics  # noqa: F401
except ModuleNotFoundError:
    pyserini_stub = ModuleType("pyserini")
    pyserini_search_stub = ModuleType("pyserini.search")
    pyserini_lucene_stub = ModuleType("pyserini.search.lucene")
    pyserini_base_stub = ModuleType("pyserini.search._base")

    class _LuceneSearcherStub:
        from_prebuilt_index = staticmethod(_missing_dependency)

    pyserini_lucene_stub.LuceneSearcher = _LuceneSearcherStub
    pyserini_base_stub.get_topics = _missing_dependency
    sys.modules["pyserini"] = pyserini_stub
    sys.modules["pyserini.search"] = pyserini_search_stub
    sys.modules["pyserini.search.lucene"] = pyserini_lucene_stub
    sys.modules["pyserini.search._base"] = pyserini_base_stub

try:
    import tiktoken  # noqa: F401
except ModuleNotFoundError:
    sys.modules["tiktoken"] = ModuleType("tiktoken")

try:
    import openai  # noqa: F401
except ModuleNotFoundError:
    sys.modules["openai"] = ModuleType("openai")

try:
    import transformers  # noqa: F401
except ModuleNotFoundError:
    transformers_stub = ModuleType("transformers")

    class _AutoStub:
        from_pretrained = staticmethod(_missing_dependency)

    transformers_stub.AutoConfig = _AutoStub
    transformers_stub.AutoTokenizer = _AutoStub
    transformers_stub.AutoModelForCausalLM = _AutoStub
    transformers_stub.AutoModelForImageTextToText = _AutoStub
    transformers_stub.AutoProcessor = _AutoStub
    transformers_stub.T5Tokenizer = _AutoStub
    transformers_stub.T5ForConditionalGeneration = _AutoStub
    transformers_stub.DataCollatorWithPadding = object
    sys.modules["transformers"] = transformers_stub

import run as run_module
import llmrankers.setwise as setwise_module
from llmrankers.rankers import SearchResult
from llmrankers._processor_adapter import ProcessorTokenizerAdapter
from llmrankers.setwise import (
    MULTIMODAL_MODEL_TYPES,
    SetwiseLlmRanker,
    THINKING_BUDGET_MODEL_TYPES,
    THINKING_DISABLE_MODEL_TYPES,
    TRUST_REMOTE_CODE_MODEL_TYPES,
    _is_multimodal_config,
    compute_max_fit_window,
)
from llmrankers.setwise_extended import (
    BiasAwareDualEndSetwiseLlmRanker,
    BottomUpSetwiseLlmRanker,
    DualEndSetwiseLlmRanker,
    MAXCONTEXT_SHUFFLE_SEED,
    MAXCONTEXT_ALLOWED_MODEL_TYPES,
    MaxContextBottomUpSetwiseLlmRanker,
    MaxContextDualEndSetwiseLlmRanker,
    MaxContextTopDownSetwiseLlmRanker,
    SameCallRegularizedSetwiseLlmRanker,
    SelectiveDualEndSetwiseLlmRanker,
    _assert_maxcontext_bottomup_fits,
    _assert_maxcontext_topdown_fits,
    _resolve_maxcontext_label_index,
)


class DummyBatch(dict):
    def __getattr__(self, item):
        return self[item]

    def to(self, _device):
        return self


class DummyTokenizer:
    pad_token = None
    eos_token = "</s>"
    pad_token_id = 0
    eos_token_id = 2
    model_max_length = 4096
    padding_side = "right"

    def __init__(self):
        self.chat_template = None
        self.use_default_system_prompt = True

    def encode(self, text, add_special_tokens=True, return_tensors=None, **_kwargs):
        ids = list(range(self._length(text, add_special_tokens)))
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def __call__(
        self,
        inputs,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors=None,
        truncation=False,
        max_length=None,
        padding=False,
    ):
        single_input = isinstance(inputs, str)
        texts = [inputs] if single_input else list(inputs)
        token_lists = [list(range(self._length(text, add_special_tokens))) for text in texts]

        if truncation and max_length is not None:
            token_lists = [tokens[:max_length] for tokens in token_lists]

        if return_tensors == "pt":
            max_len = max(len(tokens) for tokens in token_lists)
            padded_ids = []
            padded_masks = []
            for tokens in token_lists:
                pad_len = max_len - len(tokens)
                padded_ids.append(tokens + ([0] * pad_len))
                padded_masks.append(([1] * len(tokens)) + ([0] * pad_len))
            return DummyBatch(
                {
                    "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(padded_masks, dtype=torch.long),
                }
            )

        if single_input:
            return {
                "input_ids": token_lists[0],
                "attention_mask": [1] * len(token_lists[0]),
            }
        return {
            "input_ids": token_lists,
            "attention_mask": [[1] * len(tokens) for tokens in token_lists],
        }

    @staticmethod
    def _length(text, add_special_tokens):
        base = len(str(text).split())
        return base + (1 if add_special_tokens else 0)

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return " ".join(str(token) for token in ids)

    def batch_decode(self, rows, skip_special_tokens=True):
        return [self.decode(row, skip_special_tokens=skip_special_tokens) for row in rows]

    def batch_encode_plus(self, texts, return_tensors=None, add_special_tokens=True, padding=False):
        encoded = [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]
        if padding:
            width = max(len(row) for row in encoded)
            encoded = [row + [self.pad_token_id] * (width - len(row)) for row in encoded]
        input_ids = torch.tensor(encoded, dtype=torch.long) if return_tensors == "pt" else encoded
        return SimpleNamespace(input_ids=input_ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **_kwargs):
        rendered = "\n".join(message["content"] for message in messages)
        if add_generation_prompt:
            rendered += "\nassistant:"
        if tokenize:
            return self.encode(rendered)
        return rendered

    def tokenize(self, text):
        return str(text).split()

    def convert_tokens_to_ids(self, tokens):
        return list(range(len(tokens)))

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)


class DummyMistralTokenizer(DummyTokenizer):
    def __init__(self):
        self.use_default_system_prompt = True

    def __setattr__(self, name, value):
        if name == "chat_template":
            raise AttributeError("MistralCommonBackend does not support chat_template assignment")
        super().__setattr__(name, value)

    @property
    def chat_template(self):
        return None

    @chat_template.setter
    def chat_template(self, value):
        raise AttributeError("MistralCommonBackend does not support chat_template assignment")

    def convert_tokens_to_string(self, tokens):
        raise NotImplementedError("MistralCommonBackend gap")


class DummyQwen35Tokenizer(DummyTokenizer):
    pass


class DummyProcessor:
    def __init__(self, tokenizer=None, *, fail_apply_chat_template_once=False):
        self.tokenizer = tokenizer if tokenizer is not None else DummyQwen35Tokenizer()
        self.fail_apply_chat_template_once = fail_apply_chat_template_once

    def apply_chat_template(self, *args, **kwargs):
        if self.fail_apply_chat_template_once:
            self.fail_apply_chat_template_once = False
            raise TypeError("processor rejected plain text message shape")
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def __call__(self, text=None, images=None, return_tensors=None):
        return self.tokenizer(text or "", return_tensors=return_tensors)


class DummyCausalModel:
    def eval(self):
        return self


def expect_raises(fn, exc_type, contains: str):
    try:
        fn()
    except exc_type as exc:
        if contains and contains not in str(exc):
            raise AssertionError(f"Expected {contains!r} in {exc!r}") from exc
        return
    except Exception as exc:  # pragma: no cover - defensive
        raise AssertionError(f"Expected {exc_type.__name__}, got {type(exc).__name__}: {exc}") from exc
    raise AssertionError(f"Expected {exc_type.__name__} to be raised.")


def make_run_args(
    *,
    hits=10,
    scoring="generation",
    num_permutation=1,
    method="selection",
    k=10,
    direction="maxcontext_dualend",
    openai_key=None,
):
    return SimpleNamespace(
        run=SimpleNamespace(
            model_name_or_path="Qwen/Qwen3-4B",
            tokenizer_name_or_path=None,
            device="cpu",
            cache_dir=None,
            openai_key=openai_key,
            scoring=scoring,
            hits=hits,
            query_length=128,
            passage_length=512,
            log_comparisons=None,
            ir_dataset_name=None,
            pyserini_index=None,
            run_path="unused.txt",
            save_path="unused.txt",
            shuffle_ranking=None,
            shuffle=False,
            reverse=False,
        ),
        setwise=SimpleNamespace(
            direction=direction,
            num_child=3,
            method=method,
            k=k,
            num_permutation=num_permutation,
            character_scheme="letters_a_w",
            fusion="rrf",
            alpha=0.5,
            gate_strategy="hybrid",
            shortlist_size=20,
            margin_threshold=0.15,
            uncertainty_percentile=None,
            order_robust_orderings=3,
        ),
        pointwise=None,
        pairwise=None,
        listwise=None,
    )


def fake_super_init_factory(model_type, *, max_input_tokens=32768, tokenizer=None):
    def fake_super_init(self, *args, **kwargs):
        self.config = SimpleNamespace(model_type=model_type)
        self.scoring = kwargs["scoring"]
        self.k = kwargs["k"]
        self.num_permutation = kwargs["num_permutation"]
        self.method = kwargs["method"]
        self.max_input_tokens = max_input_tokens
        self.tokenizer = DummyTokenizer() if tokenizer is None else tokenizer
        self._comparison_log_path = None
        self._current_qid = None

    return fake_super_init


def instantiate_maxcontext_variant(
    ranker_cls,
    init_owner,
    *,
    model_name_or_path="Qwen/Qwen3-4B",
    model_type="qwen3",
    scoring="generation",
    pool_size=10,
    method="selection",
    num_permutation=1,
    k=None,
):
    if k is None:
        k = pool_size
    with mock.patch.object(init_owner, "__init__", new=fake_super_init_factory(model_type)):
        return ranker_cls(
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=None,
            device="cpu",
            cache_dir=None,
            num_child=3,
            scoring=scoring,
            method=method,
            num_permutation=num_permutation,
            k=k,
            pool_size=pool_size,
        )


def make_docs(count):
    return [
        SearchResult(docid=f"d{i}", score=float(-i), text=f"doc {i}")
        for i in range(1, count + 1)
    ]


def make_docs_with_scores(scores):
    return [
        SearchResult(docid=f"d{i}", score=score, text=f"doc {i}")
        for i, score in enumerate(scores, start=1)
    ]


def assert_materialized_rerank(results, docs):
    assert len(results) == len(docs)
    assert [doc.score for doc in results] == [-rank for rank in range(1, len(docs) + 1)]
    assert all(doc.text is None for doc in results)
    output_docids = [doc.docid for doc in results]
    input_docids = [doc.docid for doc in docs]
    assert set(output_docids) == set(input_docids)
    assert len(output_docids) == len(set(output_docids))


CLASSIFIER_FALLBACK_COUNTERS = (
    "total_parse_fallback",
    "total_out_of_window_label_fallback",
    "total_lexical_refusal_fallback",
    "total_numeric_out_of_range_fallback",
    "total_degenerate_repetition_fallback",
    "total_unparseable_after_exhaustion_fallback",
    "total_duplicate_label_fallback",
)
ALL_FALLBACK_COUNTERS = CLASSIFIER_FALLBACK_COUNTERS + (
    "total_lenient_fallback",
    "total_strict_parse_fallback",
)


def assert_fallback_counters_zero(ranker, counters=ALL_FALLBACK_COUNTERS):
    for counter in counters:
        assert hasattr(ranker, counter), counter
        assert getattr(ranker, counter) == 0, counter


def model_name_for_type(model_type: str) -> str:
    if model_type == "llama":
        return "meta-llama/Meta-Llama-3.1-8B-Instruct"
    if model_type == "qwen3_5":
        return "Qwen/Qwen3.5-9B"
    if model_type == "qwen3_5_moe":
        return "Qwen/Qwen3.5-35B-A3B"
    if model_type in {"mistral", "mistral3", "ministral"}:
        return "mistralai/Ministral-3-8B-Instruct-2512"
    return "Qwen/Qwen3-4B"


def assert_maxcontext_numeric_attrs(ranker, pool_size: int) -> None:
    assert ranker.CHARACTERS == [str(i + 1) for i in range(pool_size)]
    assert ranker.num_child == pool_size - 1
    assert ranker.method == "selection"
    assert ranker.strict_no_truncation is True
    assert ranker.strict_no_parse_fallback is True
    assert ranker._parse_failure_max_retries == setwise_module.MAX_PARSE_FAILURE_RETRIES
    assert ranker.total_lenient_fallback == 0
    assert ranker.total_strict_parse_fallback == 0
    assert ranker.label_scheme == "numeric_1_based"


def build_strict_topdown_compare_stub():
    ranker = object.__new__(MaxContextTopDownSetwiseLlmRanker)
    ranker.total_compare = 0
    ranker.num_permutation = 1
    ranker.total_prompt_tokens = 0
    ranker.total_completion_tokens = 0
    ranker.scoring = "generation"
    ranker.config = SimpleNamespace(model_type="qwen3")
    ranker.tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: "garbage")
    ranker.CHARACTERS = ["1", "2", "3"]
    ranker.strict_no_parse_fallback = True
    ranker.total_strict_parse_fallback = 0
    ranker.total_parse_failure_strict = 0
    ranker.total_retries = 0
    ranker.total_retry_overhead_seconds = 0.0
    ranker._comparison_log_path = None
    ranker._current_qid = "q1"
    ranker._build_best_prompt = lambda query, docs: "prompt"
    ranker._build_chat_prompt = lambda messages: messages[0]["content"]
    ranker._tokenize_inputs = lambda prompt: DummyBatch(
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }
    )
    ranker._generate = lambda inputs, max_new_tokens: torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    ranker._parse_single_label = lambda raw, characters: None
    ranker._is_supported_causal_model = lambda: True
    return ranker


def build_strict_bottomup_compare_stub():
    ranker = object.__new__(MaxContextBottomUpSetwiseLlmRanker)
    ranker.total_compare = 0
    ranker.num_permutation = 1
    ranker.total_prompt_tokens = 0
    ranker.total_completion_tokens = 0
    ranker.scoring = "generation"
    ranker.config = SimpleNamespace(model_type="qwen3")
    ranker.tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: "garbage")
    ranker.CHARACTERS = ["1", "2", "3"]
    ranker.strict_no_parse_fallback = True
    ranker.total_strict_parse_fallback = 0
    ranker.total_parse_failure_strict = 0
    ranker.total_retries = 0
    ranker.total_retry_overhead_seconds = 0.0
    ranker._comparison_log_path = None
    ranker._current_qid = "q1"
    ranker._build_worst_prompt = lambda query, docs: "prompt"
    ranker._build_chat_prompt = lambda messages: messages[0]["content"]
    ranker._tokenize_inputs = lambda prompt: DummyBatch(
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }
    )
    ranker._generate = lambda inputs, max_new_tokens: torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    ranker._parse_single_label = lambda raw, characters: None
    ranker._is_supported_causal_model = lambda: True
    return ranker


def build_maxcontext_numeric_compare_stub(raw_outputs, *, pool_size):
    ranker = object.__new__(MaxContextBottomUpSetwiseLlmRanker)
    outputs = iter(raw_outputs)
    ranker.total_compare = 0
    ranker.total_parse_fallback = 0
    ranker.total_lexical_refusal_fallback = 0
    ranker.total_numeric_out_of_range_fallback = 0
    ranker.total_degenerate_repetition_fallback = 0
    ranker.total_unparseable_after_exhaustion_fallback = 0
    ranker.total_duplicate_label_fallback = 0
    ranker.total_lenient_fallback = 0
    # Parse-failure hotfix counters (default 0 / off — match _setup_maxcontext_numeric_attrs).
    ranker.total_parse_failure_strict = 0
    ranker.total_parse_failure_bm25_fallback = 0
    ranker.total_strict_parse_fallback = 0
    ranker.total_retries = 0
    ranker.total_retry_overhead_seconds = 0.0
    ranker._allow_parse_failure_bm25_fallback = False
    ranker.num_permutation = 1
    ranker.total_prompt_tokens = 0
    ranker.total_completion_tokens = 0
    ranker.scoring = "generation"
    ranker.config = SimpleNamespace(model_type="qwen3")
    ranker.tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: next(outputs))
    ranker.CHARACTERS = [str(i + 1) for i in range(pool_size)]
    ranker.strict_no_parse_fallback = True
    ranker.label_scheme = "numeric_1_based"
    ranker._comparison_log_path = None
    ranker._current_qid = None
    ranker._build_best_prompt = lambda query, docs: "prompt"
    ranker._build_worst_prompt = lambda query, docs: "prompt"
    ranker._build_chat_prompt = lambda messages: messages[0]["content"]
    ranker._tokenize_inputs = lambda prompt: DummyBatch(
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }
    )
    ranker._generate = lambda inputs, max_new_tokens: torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    ranker._is_supported_causal_model = lambda: True
    return ranker


def build_maxcontext_dualend_compare_stub(raw_outputs, *, pool_size, log_path=None):
    ranker = object.__new__(MaxContextDualEndSetwiseLlmRanker)
    outputs = iter(raw_outputs)
    ranker.total_compare = 0
    ranker.total_parse_fallback = 0
    ranker.total_lexical_refusal_fallback = 0
    ranker.total_numeric_out_of_range_fallback = 0
    ranker.total_degenerate_repetition_fallback = 0
    ranker.total_unparseable_after_exhaustion_fallback = 0
    ranker.total_duplicate_label_fallback = 0
    ranker.total_lenient_fallback = 0
    # Parse-failure hotfix counters (default 0 / off — match _setup_maxcontext_numeric_attrs).
    ranker.total_parse_failure_strict = 0
    ranker.total_parse_failure_bm25_fallback = 0
    ranker.total_strict_parse_fallback = 0
    ranker.total_retries = 0
    ranker.total_retry_overhead_seconds = 0.0
    ranker._allow_parse_failure_bm25_fallback = False
    ranker.num_permutation = 1
    ranker.total_prompt_tokens = 0
    ranker.total_completion_tokens = 0
    ranker.scoring = "generation"
    ranker.config = SimpleNamespace(model_type="qwen3")
    ranker.tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: next(outputs))
    ranker.CHARACTERS = [str(i + 1) for i in range(pool_size)]
    ranker.strict_no_parse_fallback = True
    ranker.label_scheme = "numeric_1_based"
    ranker._comparison_log_path = log_path
    ranker._current_qid = "q1"
    ranker._format_passages = lambda docs: "\n".join(doc.text for doc in docs)
    ranker._build_chat_prompt = lambda messages: messages[0]["content"]
    ranker._tokenize_inputs = lambda prompt: DummyBatch(
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }
    )
    ranker._generate = lambda inputs, max_new_tokens: torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    ranker._is_supported_causal_model = lambda: True
    return ranker


def build_standard_topdown_compare_stub(raw_outputs, *, characters, label_scheme=None):
    ranker = object.__new__(SetwiseLlmRanker)
    outputs = iter(raw_outputs)
    ranker.total_compare = 0
    ranker.total_parse_fallback = 0
    ranker.total_out_of_window_label_fallback = 0
    ranker.total_lexical_refusal_fallback = 0
    ranker.total_numeric_out_of_range_fallback = 0
    ranker.total_degenerate_repetition_fallback = 0
    ranker.total_unparseable_after_exhaustion_fallback = 0
    ranker.total_duplicate_label_fallback = 0
    ranker.total_lenient_fallback = 0
    ranker.total_strict_parse_fallback = 0
    ranker.total_retries = 0
    ranker.total_retry_overhead_seconds = 0.0
    ranker.num_permutation = 1
    ranker.total_prompt_tokens = 0
    ranker.total_completion_tokens = 0
    ranker.scoring = "generation"
    ranker.config = SimpleNamespace(model_type="qwen3")
    ranker.tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: next(outputs))
    ranker.CHARACTERS = characters
    ranker.strict_no_parse_fallback = False
    if label_scheme is not None:
        ranker.label_scheme = label_scheme
    ranker._comparison_log_path = None
    ranker._current_qid = "q1"
    ranker._build_best_prompt = lambda query, docs: "prompt"
    ranker._build_chat_prompt = lambda messages: messages[0]["content"]
    ranker._tokenize_inputs = lambda prompt: DummyBatch(
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }
    )
    ranker._generate = lambda inputs, max_new_tokens: torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    ranker._is_supported_causal_model = lambda: True
    return ranker


def build_standard_bottomup_compare_stub(raw_outputs, *, characters, label_scheme=None):
    ranker = object.__new__(BottomUpSetwiseLlmRanker)
    outputs = iter(raw_outputs)
    ranker.total_compare = 0
    ranker.total_parse_fallback = 0
    ranker.total_out_of_window_label_fallback = 0
    ranker.total_lexical_refusal_fallback = 0
    ranker.total_numeric_out_of_range_fallback = 0
    ranker.total_degenerate_repetition_fallback = 0
    ranker.total_unparseable_after_exhaustion_fallback = 0
    ranker.total_duplicate_label_fallback = 0
    ranker.total_lenient_fallback = 0
    ranker.total_strict_parse_fallback = 0
    ranker.total_retries = 0
    ranker.total_retry_overhead_seconds = 0.0
    ranker.num_permutation = 1
    ranker.total_prompt_tokens = 0
    ranker.total_completion_tokens = 0
    ranker.scoring = "generation"
    ranker.config = SimpleNamespace(model_type="qwen3")
    ranker.tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: next(outputs))
    ranker.CHARACTERS = characters
    ranker.strict_no_parse_fallback = False
    if label_scheme is not None:
        ranker.label_scheme = label_scheme
    ranker._comparison_log_path = None
    ranker._current_qid = "q1"
    ranker._build_worst_prompt = lambda query, docs: "prompt"
    ranker._build_chat_prompt = lambda messages: messages[0]["content"]
    ranker._tokenize_inputs = lambda prompt: DummyBatch(
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }
    )
    ranker._generate = lambda inputs, max_new_tokens: torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    ranker._is_supported_causal_model = lambda: True
    return ranker


def test_dispatch_invariants():
    expect_raises(
        lambda: run_module.main(make_run_args(hits=20, k=10)),
        ValueError,
        "--hits == --k",
    )
    expect_raises(
        lambda: run_module.main(make_run_args(scoring="likelihood")),
        ValueError,
        "--scoring generation",
    )
    expect_raises(
        lambda: run_module.main(make_run_args(num_permutation=2)),
        ValueError,
        "--num_permutation 1",
    )


def test_maxcontext_init_invariants():
    common_kwargs = dict(
        model_name_or_path="Qwen/Qwen3-4B",
        tokenizer_name_or_path=None,
        device="cpu",
        cache_dir=None,
        num_child=3,
        scoring="generation",
        method="selection",
        num_permutation=1,
        k=10,
        pool_size=10,
    )

    for model_type in ("qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_moe", "llama", "mistral", "mistral3", "ministral"):
        with mock.patch.object(DualEndSetwiseLlmRanker, "__init__", new=fake_super_init_factory(model_type)):
            ranker = MaxContextDualEndSetwiseLlmRanker(
                **{**common_kwargs, "model_name_or_path": model_name_for_type(model_type)}
            )
        assert_maxcontext_numeric_attrs(ranker, 10)

    with mock.patch.object(DualEndSetwiseLlmRanker, "__init__", new=fake_super_init_factory("t5")):
        expect_raises(
            lambda: MaxContextDualEndSetwiseLlmRanker(**common_kwargs),
            ValueError,
            "Qwen3 / Qwen3.5 / Llama-3.1 / Ministral-3 model_type",
        )

    with mock.patch.object(DualEndSetwiseLlmRanker, "__init__", new=fake_super_init_factory("qwen2")):
        expect_raises(
            lambda: MaxContextDualEndSetwiseLlmRanker(**common_kwargs),
            ValueError,
            "Qwen3 / Qwen3.5 / Llama-3.1 / Ministral-3 model_type",
        )

    with mock.patch.object(DualEndSetwiseLlmRanker, "__init__", new=fake_super_init_factory("qwen3")):
        expect_raises(
            lambda: MaxContextDualEndSetwiseLlmRanker(**{**common_kwargs, "scoring": "likelihood"}),
            ValueError,
            "--scoring generation",
        )
        expect_raises(
            lambda: MaxContextDualEndSetwiseLlmRanker(**{**common_kwargs, "num_permutation": 2}),
            ValueError,
            "--num_permutation 1",
        )
        expect_raises(
            lambda: MaxContextDualEndSetwiseLlmRanker(**{**common_kwargs, "method": "bubblesort"}),
            ValueError,
            "method='selection'",
        )
        expect_raises(
            lambda: MaxContextDualEndSetwiseLlmRanker(**{**common_kwargs, "k": 9}),
            ValueError,
            "pool_size=10 but ranker.k=9",
        )


def test_parse_failure_retry_budget_invariants():
    assert not hasattr(setwise_module, "EXTENDED_PARSE_FAILURE_RETRIES")
    assert not hasattr(SetwiseLlmRanker, "_extended_retry_budget")

    for ranker_cls in (SetwiseLlmRanker, BottomUpSetwiseLlmRanker, DualEndSetwiseLlmRanker):
        ranker = object.__new__(ranker_cls)
        assert not hasattr(ranker, "_parse_failure_max_retries")
        assert ranker._parse_failure_retry_budget() == 0

    maxcontext_cases = (
        (MaxContextTopDownSetwiseLlmRanker, SetwiseLlmRanker),
        (MaxContextBottomUpSetwiseLlmRanker, BottomUpSetwiseLlmRanker),
        (MaxContextDualEndSetwiseLlmRanker, DualEndSetwiseLlmRanker),
    )
    for ranker_cls, init_owner in maxcontext_cases:
        ranker = instantiate_maxcontext_variant(ranker_cls, init_owner, pool_size=10)
        assert ranker._parse_failure_max_retries == setwise_module.MAX_PARSE_FAILURE_RETRIES
        assert ranker._parse_failure_retry_budget() == 3

    standard = object.__new__(SetwiseLlmRanker)
    standard._parse_failure_max_retries = 5
    assert standard._parse_failure_retry_budget() == 5

    maxcontext = instantiate_maxcontext_variant(
        MaxContextTopDownSetwiseLlmRanker,
        SetwiseLlmRanker,
        pool_size=10,
    )
    maxcontext._parse_failure_max_retries = 7
    assert maxcontext._parse_failure_retry_budget() == 7


def test_lenient_fallback_counter_invariants():
    for ranker_cls in (SetwiseLlmRanker, BottomUpSetwiseLlmRanker, DualEndSetwiseLlmRanker):
        ranker = instantiate_setwise_ranker(ranker_cls=ranker_cls)
        assert_fallback_counters_zero(ranker)

    maxcontext_cases = (
        (MaxContextTopDownSetwiseLlmRanker, SetwiseLlmRanker),
        (MaxContextBottomUpSetwiseLlmRanker, BottomUpSetwiseLlmRanker),
        (MaxContextDualEndSetwiseLlmRanker, DualEndSetwiseLlmRanker),
    )
    for ranker_cls, init_owner in maxcontext_cases:
        ranker = instantiate_maxcontext_variant(ranker_cls, init_owner, pool_size=10)
        assert_fallback_counters_zero(
            ranker,
            tuple(counter for counter in ALL_FALLBACK_COUNTERS if counter != "total_out_of_window_label_fallback"),
        )

    source = Path("run.py").read_text()
    assert '"total_lenient_fallback": "lenient fallbacks"' in source
    assert '"total_strict_parse_fallback": "strict parse fallbacks"' in source
    for counter in CLASSIFIER_FALLBACK_COUNTERS:
        assert f'"{counter}"' in source


def test_maxcontext_dualend_byte_identity_snapshot():
    tokenizer_sentinel = "tokenizer-sentinel"
    with mock.patch.object(
        DualEndSetwiseLlmRanker,
        "__init__",
        new=fake_super_init_factory("qwen3", tokenizer=tokenizer_sentinel),
    ):
        ranker = MaxContextDualEndSetwiseLlmRanker(
            model_name_or_path="Qwen/Qwen3-4B",
            tokenizer_name_or_path=None,
            device="cpu",
            cache_dir=None,
            num_child=3,
            scoring="generation",
            method="selection",
            num_permutation=1,
            k=10,
            pool_size=10,
        )

    expected_snapshot = {
        "config": SimpleNamespace(model_type="qwen3"),
        "scoring": "generation",
        "k": 10,
        "num_permutation": 1,
        "method": "selection",
        "max_input_tokens": 32768,
        "tokenizer": tokenizer_sentinel,
        "_comparison_log_path": None,
        "_current_qid": None,
        "CHARACTERS": [str(i + 1) for i in range(10)],
        "num_child": 9,
        "strict_no_truncation": True,
        "strict_no_parse_fallback": True,
        "_parse_failure_max_retries": 3,
        "total_parse_fallback": 0,
        "total_lexical_refusal_fallback": 0,
        "total_numeric_out_of_range_fallback": 0,
        "total_degenerate_repetition_fallback": 0,
        "total_unparseable_after_exhaustion_fallback": 0,
        "total_duplicate_label_fallback": 0,
        "total_lenient_fallback": 0,
        "total_strict_parse_fallback": 0,
        # Parse-failure hotfix 2026-05-08: new counters and BM25-fallback flag.
        "total_parse_failure_strict": 0,
        "total_parse_failure_bm25_fallback": 0,
        "total_retries": 0,
        "total_retry_overhead_seconds": 0.0,
        "_allow_parse_failure_bm25_fallback": False,
        "output_depth": 10,
        "capture_raw_responses": False,
        "label_scheme": "numeric_1_based",
        "_maxcontext_pool_size": 10,
        "shuffle": False,
        "reverse": False,
    }
    assert vars(ranker) == expected_snapshot


def test_maxcontext_dualend_allows_query_pools_below_cap():
    ranker = instantiate_maxcontext_variant(
        MaxContextDualEndSetwiseLlmRanker,
        DualEndSetwiseLlmRanker,
        pool_size=10,
    )
    ranker._assert_maxcontext_fits = lambda query, ranking: None
    calls = {"count": 0}

    def choose_edges(query, window):
        ranker.total_compare += 1
        calls["count"] += 1
        return "1", str(len(window))

    ranker.compare_both = choose_edges
    docs = make_docs(3)
    results = ranker.rerank("query", docs)
    assert calls["count"] == 1
    assert ranker.total_compare == 1
    assert_materialized_rerank(results, docs)

    one_doc_ranker = instantiate_maxcontext_variant(
        MaxContextDualEndSetwiseLlmRanker,
        DualEndSetwiseLlmRanker,
        pool_size=10,
    )
    one_doc_ranker._assert_maxcontext_fits = lambda *_args: (_ for _ in ()).throw(
        AssertionError("single-doc MaxContext should not preflight")
    )
    one_doc_ranker.compare_both = lambda *_args: (_ for _ in ()).throw(
        AssertionError("single-doc MaxContext should not call the LLM")
    )
    one_doc_results = one_doc_ranker.rerank("query", make_docs(1))
    assert one_doc_ranker.total_compare == 0
    assert_materialized_rerank(one_doc_results, make_docs(1))

    capped_ranker = instantiate_maxcontext_variant(
        MaxContextDualEndSetwiseLlmRanker,
        DualEndSetwiseLlmRanker,
        pool_size=2,
    )
    expect_raises(
        lambda: capped_ranker.rerank("query", make_docs(3)),
        ValueError,
        "at most pool_size=2",
    )


def test_generation_budget_tier():
    expected = {
        "qwen2": (256, 512),
        "qwen3": (256, 512),
        "qwen3_moe": (256, 512),
        "qwen3_5": (256, 512),
        "qwen3_5_moe": (256, 512),
        "llama": (256, 512),
        "mistral": (32, 64),
        "mistral3": (256, 512),
        "ministral": (32, 64),
        "t5": (2, 2),
    }
    for model_type, (single_budget, dual_budget) in expected.items():
        ranker = object.__new__(SetwiseLlmRanker)
        ranker.config = SimpleNamespace(model_type=model_type)
        assert ranker._generation_budget("single") == single_budget
        assert ranker._generation_budget("dual") == dual_budget

    assert THINKING_BUDGET_MODEL_TYPES == {"qwen2", "qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_moe"}


def test_chat_template_kwargs_per_family():
    expected = {
        "qwen2": {"enable_thinking": False},
        "qwen3": {"enable_thinking": False},
        "qwen3_moe": {"enable_thinking": False},
        "qwen3_5": {"enable_thinking": False},
        "qwen3_5_moe": {"enable_thinking": False},
        "llama": {},
        "mistral": {},
        "mistral3": {},
        "ministral": {},
        "t5": {},
    }
    for model_type, kwargs in expected.items():
        ranker = object.__new__(SetwiseLlmRanker)
        ranker.config = SimpleNamespace(model_type=model_type)
        assert ranker._chat_template_kwargs() == kwargs

    assert THINKING_DISABLE_MODEL_TYPES == {"qwen2", "qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_moe"}


def test_trust_remote_code_per_family():
    assert TRUST_REMOTE_CODE_MODEL_TYPES == {"qwen2", "qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_moe"}
    model_types = ("qwen2", "qwen3", "qwen3_moe", "llama", "mistral", "ministral")
    for model_type in model_types:
        config = SimpleNamespace(model_type=model_type, n_positions=4096, max_position_embeddings=4096)
        tokenizer_loader = mock.Mock(return_value=make_stub_causal_tokenizer())
        model_loader = mock.Mock(return_value=DummyCausalModel())
        with mock.patch("llmrankers.setwise.AutoConfig.from_pretrained", return_value=config), \
             mock.patch("llmrankers.setwise.AutoTokenizer.from_pretrained", tokenizer_loader), \
             mock.patch("llmrankers.setwise.AutoModelForCausalLM.from_pretrained", model_loader):
            SetwiseLlmRanker(
                model_name_or_path=model_name_for_type(model_type),
                tokenizer_name_or_path=None,
                device="cpu",
                num_child=3,
                k=10,
                scoring="generation",
                method="heapsort",
                num_permutation=1,
                cache_dir=None,
            )

        expected = model_type in TRUST_REMOTE_CODE_MODEL_TYPES
        assert tokenizer_loader.call_args.kwargs.get("trust_remote_code", False) is expected
        assert model_loader.call_args.kwargs.get("trust_remote_code", False) is expected

    for model_type in ("qwen3_5", "qwen3_5_moe", "mistral3"):
        config = SimpleNamespace(
            model_type=model_type,
            vision_config=SimpleNamespace(),
            n_positions=4096,
            max_position_embeddings=4096,
        )
        processor_loader = mock.Mock(return_value=DummyProcessor(DummyQwen35Tokenizer()))
        model_loader = mock.Mock(return_value=DummyCausalModel())
        with mock.patch("llmrankers.setwise.AutoConfig.from_pretrained", return_value=config), \
             mock.patch("llmrankers.setwise.AutoProcessor.from_pretrained", processor_loader), \
             mock.patch("llmrankers.setwise.AutoModelForImageTextToText.from_pretrained", model_loader):
            ranker = SetwiseLlmRanker(
                model_name_or_path=model_name_for_type(model_type),
                tokenizer_name_or_path=None,
                device="cpu",
                num_child=3,
                k=10,
                scoring="generation",
                method="heapsort",
                num_permutation=1,
                cache_dir=None,
            )

        assert ranker._is_multimodal_model() is True
        assert processor_loader.call_args.kwargs["trust_remote_code"] is True
        assert model_loader.call_args.kwargs["trust_remote_code"] is True


def test_multimodal_predicate_and_adapter():
    assert MULTIMODAL_MODEL_TYPES == {"mistral3", "qwen3_5", "qwen3_5_moe"}
    assert _is_multimodal_config(SimpleNamespace(model_type="mistral3", vision_config=SimpleNamespace()))
    assert _is_multimodal_config(SimpleNamespace(model_type="qwen3_5", vision_config=SimpleNamespace()))
    assert not _is_multimodal_config(SimpleNamespace(model_type="qwen3_5_text"))
    assert not _is_multimodal_config(SimpleNamespace(model_type="qwen3"))

    adapter = ProcessorTokenizerAdapter(
        DummyProcessor(DummyMistralTokenizer(), fail_apply_chat_template_once=True)
    )
    rendered = adapter.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert "hello" in rendered
    assert adapter.convert_tokens_to_string(["a", "b"]) == "0 1"
    adapter.chat_template = "ignored"
    assert adapter.chat_template is None
    adapter.use_default_system_prompt = False
    assert adapter._tok.use_default_system_prompt is False
    adapter.padding_side = "left"
    assert adapter._tok.padding_side == "left"


def test_maxcontext_multimodal_generation_gates():
    for ranker_cls, method_name, raw_output in (
        (MaxContextBottomUpSetwiseLlmRanker, "compare_worst", "2"),
        (MaxContextDualEndSetwiseLlmRanker, "compare_both", "Best: 1, Worst: 3"),
        (MaxContextDualEndSetwiseLlmRanker, "_compare_worst_single", "2"),
    ):
        ranker = object.__new__(ranker_cls)
        ranker.config = SimpleNamespace(model_type="mistral3")
        ranker._is_multimodal = True
        ranker.scoring = "generation"
        ranker.num_permutation = 1
        ranker.total_compare = 0
        ranker.total_prompt_tokens = 0
        ranker.total_completion_tokens = 0
        ranker.CHARACTERS = ["1", "2", "3"]
        ranker.label_scheme = "numeric_1_based"
        ranker.strict_no_parse_fallback = True
        ranker._comparison_log_path = None
        ranker._current_qid = None
        ranker.tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: raw_output)
        ranker._build_worst_prompt = lambda query, docs: "worst prompt"
        ranker._build_best_prompt = lambda query, docs: "best prompt"
        ranker._format_passages = lambda docs: "\n".join(doc.text for doc in docs)
        ranker._build_chat_prompt = lambda messages: messages[0]["content"] + " chat"
        ranker._tokenize_inputs = lambda prompt: DummyBatch(
            {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }
        )
        calls = []
        ranker._generate = lambda inputs, max_new_tokens: (
            calls.append(max_new_tokens) or torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        )

        assert ranker._is_supported_causal_model() is False
        result = getattr(ranker, method_name)("query", make_docs(3))
        assert calls, f"{method_name} did not enter multimodal generation branch"
        if method_name == "compare_both":
            assert result == ("1", "3")
        else:
            assert result == "2"

    assert "qwen3_5_moe" in MAXCONTEXT_ALLOWED_MODEL_TYPES


def test_early_reject_unsupported_family():
    for model_name in (
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.5-35B-A3B",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "mistralai/Ministral-3-8B-Instruct-2512",
    ):
        MaxContextDualEndSetwiseLlmRanker._early_reject_unsupported_family(model_name)

    for model_name in (
        "gpt2",
        "meta-llama/Llama-2-7b-chat-hf",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ):
        expect_raises(
            lambda name=model_name: MaxContextDualEndSetwiseLlmRanker._early_reject_unsupported_family(name),
            ValueError,
            "MaxContext supports Qwen3 / Qwen3.5 / Llama-3.1 / Ministral-3 only",
        )


def test_maxcontext_topdown_invariants():
    ranker = instantiate_maxcontext_variant(
        MaxContextTopDownSetwiseLlmRanker,
        SetwiseLlmRanker,
        pool_size=10,
    )
    assert_maxcontext_numeric_attrs(ranker, 10)
    assert ranker.total_bm25_bypass == 0

    for model_type in ("llama", "mistral", "mistral3", "ministral"):
        positive_ranker = instantiate_maxcontext_variant(
            MaxContextTopDownSetwiseLlmRanker,
            SetwiseLlmRanker,
            model_name_or_path=model_name_for_type(model_type),
            model_type=model_type,
            pool_size=10,
        )
        assert_maxcontext_numeric_attrs(positive_ranker, 10)
        assert positive_ranker.total_bm25_bypass == 0

    expect_raises(
        lambda: instantiate_maxcontext_variant(
            MaxContextTopDownSetwiseLlmRanker,
            SetwiseLlmRanker,
            model_type="t5",
        ),
        ValueError,
        "Qwen3 / Qwen3.5 / Llama-3.1 / Ministral-3 model_type",
    )
    expect_raises(
        lambda: instantiate_maxcontext_variant(
            MaxContextTopDownSetwiseLlmRanker,
            SetwiseLlmRanker,
            model_type="qwen2",
        ),
        ValueError,
        "Qwen3 / Qwen3.5 / Llama-3.1 / Ministral-3 model_type",
    )
    expect_raises(
        lambda: instantiate_maxcontext_variant(
            MaxContextTopDownSetwiseLlmRanker,
            SetwiseLlmRanker,
            scoring="likelihood",
        ),
        ValueError,
        "--scoring generation",
    )
    expect_raises(
        lambda: instantiate_maxcontext_variant(
            MaxContextTopDownSetwiseLlmRanker,
            SetwiseLlmRanker,
            method="heapsort",
        ),
        ValueError,
        "method='selection'",
    )
    expect_raises(
        lambda: run_module.main(
            make_run_args(direction="maxcontext_topdown", openai_key="sk-test")
        ),
        ValueError,
        "--direction maxcontext_topdown is not supported with --openai_key",
    )

    docs = make_docs(10)
    ranker._assert_maxcontext_fits = lambda query, ranking: None
    call_counter = {"count": 0}

    def choose_first(query, window):
        ranker.total_compare += 1
        call_counter["count"] += 1
        return "1"

    ranker.compare = choose_first
    results = ranker.rerank("query", docs)
    assert call_counter["count"] == 8
    assert ranker.total_compare == 8
    assert ranker.total_bm25_bypass == 1
    assert_materialized_rerank(results, docs)

    one_doc_ranker = instantiate_maxcontext_variant(
        MaxContextTopDownSetwiseLlmRanker,
        SetwiseLlmRanker,
        pool_size=1,
    )
    one_doc_ranker._assert_maxcontext_fits = lambda query, ranking: None
    one_doc_calls = {"count": 0}
    one_doc_ranker.compare = lambda query, window: one_doc_calls.__setitem__("count", one_doc_calls["count"] + 1) or "1"
    one_doc_results = one_doc_ranker.rerank("query", make_docs(1))
    assert one_doc_calls["count"] == 0
    assert one_doc_ranker.total_compare == 0
    assert one_doc_ranker.total_bm25_bypass == 0
    assert_materialized_rerank(one_doc_results, make_docs(1))

    two_doc_ranker = instantiate_maxcontext_variant(
        MaxContextTopDownSetwiseLlmRanker,
        SetwiseLlmRanker,
        pool_size=2,
    )
    two_doc_ranker._assert_maxcontext_fits = lambda query, ranking: None
    two_doc_ranker.compare = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("LLM should not be called at n_docs=2")
    )
    two_doc_results = two_doc_ranker.rerank("query", make_docs_with_scores([0.0, 1.0]))
    assert two_doc_ranker.total_compare == 0
    assert two_doc_ranker.total_bm25_bypass == 1
    assert [doc.docid for doc in two_doc_results] == ["d2", "d1"]

    two_doc_tie_ranker = instantiate_maxcontext_variant(
        MaxContextTopDownSetwiseLlmRanker,
        SetwiseLlmRanker,
        pool_size=2,
    )
    two_doc_tie_ranker._assert_maxcontext_fits = lambda query, ranking: None
    two_doc_tie_ranker.compare = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("LLM should not be called at n_docs=2")
    )
    two_doc_tie_results = two_doc_tie_ranker.rerank(
        "query",
        [
            SearchResult(docid="d2", score=1.0, text="doc 2"),
            SearchResult(docid="d1", score=1.0, text="doc 1"),
        ],
    )
    assert two_doc_tie_ranker.total_compare == 0
    assert two_doc_tie_ranker.total_bm25_bypass == 1
    assert [doc.docid for doc in two_doc_tie_results] == ["d2", "d1"]

    three_doc_ranker = instantiate_maxcontext_variant(
        MaxContextTopDownSetwiseLlmRanker,
        SetwiseLlmRanker,
        pool_size=3,
    )
    three_doc_ranker._assert_maxcontext_fits = lambda query, ranking: None
    three_doc_calls = {"count": 0}

    def choose_first_three(query, window):
        three_doc_ranker.total_compare += 1
        three_doc_calls["count"] += 1
        return "1"

    three_doc_ranker.compare = choose_first_three
    three_doc_results = three_doc_ranker.rerank("query", make_docs(3))
    assert three_doc_calls["count"] == 1
    assert three_doc_ranker.total_compare == 1
    assert three_doc_ranker.total_bm25_bypass == 1
    assert_materialized_rerank(three_doc_results, make_docs(3))

    short_pool_ranker = instantiate_maxcontext_variant(
        MaxContextTopDownSetwiseLlmRanker,
        SetwiseLlmRanker,
        pool_size=10,
    )
    short_pool_ranker._assert_maxcontext_fits = lambda query, ranking: None
    short_pool_calls = {"count": 0}

    def choose_first_short_pool(query, window):
        short_pool_ranker.total_compare += 1
        short_pool_calls["count"] += 1
        return "1"

    short_pool_ranker.compare = choose_first_short_pool
    short_pool_results = short_pool_ranker.rerank("query", make_docs(3))
    assert short_pool_calls["count"] == 1
    assert short_pool_ranker.total_compare == 1
    assert short_pool_ranker.total_bm25_bypass == 1
    assert_materialized_rerank(short_pool_results, make_docs(3))

    one_doc_below_cap_ranker = instantiate_maxcontext_variant(
        MaxContextTopDownSetwiseLlmRanker,
        SetwiseLlmRanker,
        pool_size=10,
    )
    one_doc_below_cap_ranker._assert_maxcontext_fits = lambda *_args: (_ for _ in ()).throw(
        AssertionError("single-doc MaxContext should not preflight")
    )
    one_doc_below_cap_ranker.compare = lambda *_args: (_ for _ in ()).throw(
        AssertionError("single-doc MaxContext should not call the LLM")
    )
    one_doc_below_cap_results = one_doc_below_cap_ranker.rerank("query", make_docs(1))
    assert one_doc_below_cap_ranker.total_compare == 0
    assert one_doc_below_cap_ranker.total_bm25_bypass == 0
    assert_materialized_rerank(one_doc_below_cap_results, make_docs(1))

    capped_ranker = instantiate_maxcontext_variant(
        MaxContextTopDownSetwiseLlmRanker,
        SetwiseLlmRanker,
        pool_size=2,
    )
    expect_raises(
        lambda: capped_ranker.rerank("query", make_docs(3)),
        ValueError,
        "at most pool_size=2",
    )

    bad_score_ranker = instantiate_maxcontext_variant(
        MaxContextTopDownSetwiseLlmRanker,
        SetwiseLlmRanker,
        pool_size=2,
    )
    bad_score_ranker._assert_maxcontext_fits = lambda query, ranking: None
    expect_raises(
        lambda: bad_score_ranker.rerank(
            "query",
            [
                SearchResult(docid="d1", score=None, text="doc 1"),
                SearchResult(docid="d2", score=1.0, text="doc 2"),
            ],
        ),
        ValueError,
        "requires finite BM25 scores",
    )

    strict_ranker = instantiate_maxcontext_variant(
        MaxContextTopDownSetwiseLlmRanker,
        SetwiseLlmRanker,
        pool_size=4,
    )
    strict_ranker._assert_maxcontext_fits = lambda query, ranking: None
    labels = iter(["1", "4"])
    strict_ranker.compare = lambda query, window: next(labels)
    strict_results = strict_ranker.rerank("query", make_docs(4))
    assert strict_ranker.total_strict_parse_fallback == 1
    assert strict_ranker.total_parse_failure_strict == 0
    assert_materialized_rerank(strict_results, make_docs(4))

    compare_ranker = build_strict_topdown_compare_stub()
    assert compare_ranker.compare("query", make_docs(2)) == "1"
    assert compare_ranker.total_retries == 3
    assert compare_ranker.total_strict_parse_fallback == 1
    assert compare_ranker.total_parse_failure_strict == 0

    prompt_ranker = SimpleNamespace(
        max_input_tokens=1000,
        tokenizer=SimpleNamespace(encode=lambda text, add_special_tokens=True: list(range(10))),
        _build_best_prompt=mock.Mock(return_value="best prompt"),
        _build_chat_prompt=mock.Mock(side_effect=lambda messages: messages[0]["content"] + " chat"),
    )
    _assert_maxcontext_topdown_fits(prompt_ranker, "query", make_docs(2))
    prompt_ranker._build_best_prompt.assert_called_once_with("query", make_docs(2))


def test_maxcontext_bottomup_invariants():
    ranker = instantiate_maxcontext_variant(
        MaxContextBottomUpSetwiseLlmRanker,
        BottomUpSetwiseLlmRanker,
        pool_size=10,
    )
    assert_maxcontext_numeric_attrs(ranker, 10)
    assert ranker.total_bm25_bypass == 0

    for model_type in ("llama", "mistral", "mistral3", "ministral"):
        positive_ranker = instantiate_maxcontext_variant(
            MaxContextBottomUpSetwiseLlmRanker,
            BottomUpSetwiseLlmRanker,
            model_name_or_path=model_name_for_type(model_type),
            model_type=model_type,
            pool_size=10,
        )
        assert_maxcontext_numeric_attrs(positive_ranker, 10)
        assert positive_ranker.total_bm25_bypass == 0

    expect_raises(
        lambda: instantiate_maxcontext_variant(
            MaxContextBottomUpSetwiseLlmRanker,
            BottomUpSetwiseLlmRanker,
            model_type="t5",
        ),
        ValueError,
        "Qwen3 / Qwen3.5 / Llama-3.1 / Ministral-3 model_type",
    )
    expect_raises(
        lambda: instantiate_maxcontext_variant(
            MaxContextBottomUpSetwiseLlmRanker,
            BottomUpSetwiseLlmRanker,
            model_type="qwen2",
        ),
        ValueError,
        "Qwen3 / Qwen3.5 / Llama-3.1 / Ministral-3 model_type",
    )
    expect_raises(
        lambda: instantiate_maxcontext_variant(
            MaxContextBottomUpSetwiseLlmRanker,
            BottomUpSetwiseLlmRanker,
            scoring="likelihood",
        ),
        ValueError,
        "--scoring generation",
    )
    expect_raises(
        lambda: instantiate_maxcontext_variant(
            MaxContextBottomUpSetwiseLlmRanker,
            BottomUpSetwiseLlmRanker,
            method="heapsort",
        ),
        ValueError,
        "method='selection'",
    )
    expect_raises(
        lambda: run_module.main(
            make_run_args(direction="maxcontext_bottomup", openai_key="sk-test")
        ),
        ValueError,
        "--direction maxcontext_bottomup is not supported with --openai_key",
    )

    docs = make_docs(10)
    ranker._assert_maxcontext_fits = lambda query, ranking: None
    call_counter = {"count": 0}

    def choose_first(query, window):
        ranker.total_compare += 1
        call_counter["count"] += 1
        return "1"

    ranker.compare_worst = choose_first
    results = ranker.rerank("query", docs)
    assert call_counter["count"] == 8
    assert ranker.total_compare == 8
    assert ranker.total_bm25_bypass == 1
    assert_materialized_rerank(results, docs)

    one_doc_ranker = instantiate_maxcontext_variant(
        MaxContextBottomUpSetwiseLlmRanker,
        BottomUpSetwiseLlmRanker,
        pool_size=1,
    )
    one_doc_ranker._assert_maxcontext_fits = lambda query, ranking: None
    one_doc_calls = {"count": 0}
    one_doc_ranker.compare_worst = lambda query, window: one_doc_calls.__setitem__("count", one_doc_calls["count"] + 1) or "1"
    one_doc_results = one_doc_ranker.rerank("query", make_docs(1))
    assert one_doc_calls["count"] == 0
    assert one_doc_ranker.total_compare == 0
    assert one_doc_ranker.total_bm25_bypass == 0
    assert_materialized_rerank(one_doc_results, make_docs(1))

    two_doc_ranker = instantiate_maxcontext_variant(
        MaxContextBottomUpSetwiseLlmRanker,
        BottomUpSetwiseLlmRanker,
        pool_size=2,
    )
    two_doc_ranker._assert_maxcontext_fits = lambda query, ranking: None
    two_doc_ranker.compare_worst = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("LLM should not be called at n_docs=2")
    )
    two_doc_results = two_doc_ranker.rerank("query", make_docs_with_scores([0.0, 1.0]))
    assert two_doc_ranker.total_compare == 0
    assert two_doc_ranker.total_bm25_bypass == 1
    assert [doc.docid for doc in two_doc_results] == ["d2", "d1"]

    two_doc_tie_ranker = instantiate_maxcontext_variant(
        MaxContextBottomUpSetwiseLlmRanker,
        BottomUpSetwiseLlmRanker,
        pool_size=2,
    )
    two_doc_tie_ranker._assert_maxcontext_fits = lambda query, ranking: None
    two_doc_tie_ranker.compare_worst = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("LLM should not be called at n_docs=2")
    )
    two_doc_tie_results = two_doc_tie_ranker.rerank(
        "query",
        [
            SearchResult(docid="d2", score=1.0, text="doc 2"),
            SearchResult(docid="d1", score=1.0, text="doc 1"),
        ],
    )
    assert two_doc_tie_ranker.total_compare == 0
    assert two_doc_tie_ranker.total_bm25_bypass == 1
    assert [doc.docid for doc in two_doc_tie_results] == ["d2", "d1"]

    three_doc_ranker = instantiate_maxcontext_variant(
        MaxContextBottomUpSetwiseLlmRanker,
        BottomUpSetwiseLlmRanker,
        pool_size=3,
    )
    three_doc_ranker._assert_maxcontext_fits = lambda query, ranking: None
    three_doc_calls = {"count": 0}

    def choose_first_three(query, window):
        three_doc_ranker.total_compare += 1
        three_doc_calls["count"] += 1
        return "1"

    three_doc_ranker.compare_worst = choose_first_three
    three_doc_results = three_doc_ranker.rerank("query", make_docs(3))
    assert three_doc_calls["count"] == 1
    assert three_doc_ranker.total_compare == 1
    assert three_doc_ranker.total_bm25_bypass == 1
    assert_materialized_rerank(three_doc_results, make_docs(3))

    short_pool_ranker = instantiate_maxcontext_variant(
        MaxContextBottomUpSetwiseLlmRanker,
        BottomUpSetwiseLlmRanker,
        pool_size=10,
    )
    short_pool_ranker._assert_maxcontext_fits = lambda query, ranking: None
    short_pool_calls = {"count": 0}

    def choose_first_short_pool(query, window):
        short_pool_ranker.total_compare += 1
        short_pool_calls["count"] += 1
        return "1"

    short_pool_ranker.compare_worst = choose_first_short_pool
    short_pool_results = short_pool_ranker.rerank("query", make_docs(3))
    assert short_pool_calls["count"] == 1
    assert short_pool_ranker.total_compare == 1
    assert short_pool_ranker.total_bm25_bypass == 1
    assert_materialized_rerank(short_pool_results, make_docs(3))

    one_doc_below_cap_ranker = instantiate_maxcontext_variant(
        MaxContextBottomUpSetwiseLlmRanker,
        BottomUpSetwiseLlmRanker,
        pool_size=10,
    )
    one_doc_below_cap_ranker._assert_maxcontext_fits = lambda *_args: (_ for _ in ()).throw(
        AssertionError("single-doc MaxContext should not preflight")
    )
    one_doc_below_cap_ranker.compare_worst = lambda *_args: (_ for _ in ()).throw(
        AssertionError("single-doc MaxContext should not call the LLM")
    )
    one_doc_below_cap_results = one_doc_below_cap_ranker.rerank("query", make_docs(1))
    assert one_doc_below_cap_ranker.total_compare == 0
    assert one_doc_below_cap_ranker.total_bm25_bypass == 0
    assert_materialized_rerank(one_doc_below_cap_results, make_docs(1))

    capped_ranker = instantiate_maxcontext_variant(
        MaxContextBottomUpSetwiseLlmRanker,
        BottomUpSetwiseLlmRanker,
        pool_size=2,
    )
    expect_raises(
        lambda: capped_ranker.rerank("query", make_docs(3)),
        ValueError,
        "at most pool_size=2",
    )

    strict_ranker = instantiate_maxcontext_variant(
        MaxContextBottomUpSetwiseLlmRanker,
        BottomUpSetwiseLlmRanker,
        pool_size=4,
    )
    strict_ranker._assert_maxcontext_fits = lambda query, ranking: None
    labels = iter(["1", "4"])
    strict_ranker.compare_worst = lambda query, window: next(labels)
    strict_results = strict_ranker.rerank("query", make_docs(4))
    assert strict_ranker.total_strict_parse_fallback == 1
    assert strict_ranker.total_parse_failure_strict == 0
    assert_materialized_rerank(strict_results, make_docs(4))

    bad_score_ranker = instantiate_maxcontext_variant(
        MaxContextBottomUpSetwiseLlmRanker,
        BottomUpSetwiseLlmRanker,
        pool_size=2,
    )
    bad_score_ranker._assert_maxcontext_fits = lambda query, ranking: None
    expect_raises(
        lambda: bad_score_ranker.rerank(
            "query",
            [
                SearchResult(docid="d1", score=None, text="doc 1"),
                SearchResult(docid="d2", score=1.0, text="doc 2"),
            ],
        ),
        ValueError,
        "requires finite BM25 scores",
    )

    compare_ranker = build_strict_bottomup_compare_stub()
    assert compare_ranker.compare_worst("query", make_docs(2)) == "2"
    assert compare_ranker.total_retries == 3
    assert compare_ranker.total_strict_parse_fallback == 1
    assert compare_ranker.total_parse_failure_strict == 0

    prompt_ranker = SimpleNamespace(
        max_input_tokens=1000,
        tokenizer=SimpleNamespace(encode=lambda text, add_special_tokens=True: list(range(10))),
        _build_worst_prompt=mock.Mock(return_value="worst prompt"),
        _build_chat_prompt=mock.Mock(side_effect=lambda messages: messages[0]["content"] + " chat"),
    )
    _assert_maxcontext_bottomup_fits(prompt_ranker, "query", make_docs(2))
    prompt_ranker._build_worst_prompt.assert_called_once_with("query", make_docs(2))


def build_dualend_stub(*, strict=False, label_scheme=None, characters=None):
    ranker = object.__new__(DualEndSetwiseLlmRanker)
    ranker.CHARACTERS = characters or [chr(ord("A") + i) for i in range(23)]
    ranker.strict_no_parse_fallback = strict
    ranker.strict_no_truncation = False
    for counter in CLASSIFIER_FALLBACK_COUNTERS:
        setattr(ranker, counter, 0)
    ranker.total_lenient_fallback = 0
    ranker.total_strict_parse_fallback = 0
    ranker.total_parse_failure_strict = 0
    if label_scheme is not None:
        ranker.label_scheme = label_scheme
    ranker._comparison_log_path = None
    ranker._current_qid = None
    return ranker


def build_numeric_dualend_stub():
    ranker = object.__new__(DualEndSetwiseLlmRanker)
    ranker.CHARACTERS = [str(i + 1) for i in range(10)]
    ranker.strict_no_parse_fallback = True
    ranker.strict_no_truncation = False
    ranker.label_scheme = "numeric_1_based"
    ranker.total_lenient_fallback = 0
    ranker.total_strict_parse_fallback = 0
    ranker._comparison_log_path = None
    ranker._current_qid = None
    return ranker


def build_dual_prompt_capture_stub(
    raw_output,
    *,
    characters,
    model_type="qwen3",
    label_scheme=None,
    is_multimodal=False,
    strict=False,
):
    ranker = object.__new__(DualEndSetwiseLlmRanker)
    captured = {}
    ranker.total_compare = 0
    ranker.total_prompt_tokens = 0
    ranker.total_completion_tokens = 0
    ranker.total_retries = 0
    ranker.total_retry_overhead_seconds = 0.0
    ranker.total_parse_fallback = 0
    ranker.total_lexical_refusal_fallback = 0
    ranker.total_numeric_out_of_range_fallback = 0
    ranker.total_degenerate_repetition_fallback = 0
    ranker.total_unparseable_after_exhaustion_fallback = 0
    ranker.total_duplicate_label_fallback = 0
    ranker.total_lenient_fallback = 0
    ranker.total_strict_parse_fallback = 0
    ranker.total_parse_failure_strict = 0
    ranker.num_permutation = 1
    ranker.scoring = "generation"
    ranker.config = SimpleNamespace(model_type=model_type)
    ranker.tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: raw_output)
    ranker.CHARACTERS = characters
    ranker.strict_no_parse_fallback = strict
    ranker.strict_no_truncation = False
    ranker._is_multimodal = is_multimodal
    if label_scheme is not None:
        ranker.label_scheme = label_scheme
    ranker._comparison_log_path = None
    ranker._current_qid = "q1"
    ranker._build_chat_prompt = lambda messages: captured.setdefault(
        "prompt",
        messages[0]["content"],
    )
    ranker._tokenize_inputs = lambda prompt: DummyBatch(
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }
    )
    ranker._generate = lambda inputs, max_new_tokens: torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    return ranker, captured


def test_parse_invariants():
    strict_ranker = build_dualend_stub(strict=True)
    expect_raises(
        lambda: strict_ranker._parse_dual_output("Best: 9, Worst: 2", 3),
        ValueError,
        "parse failed",
    )
    assert strict_ranker._parse_dual_output("Best: 1, Worst: 1", 3) == ("A", "A")

    numeric_ranker = build_numeric_dualend_stub()
    assert numeric_ranker._try_parse_dual_output("Best: Passage 3  \nWorst: Passage 4", 10) == ("3", "4")
    assert numeric_ranker._try_parse_dual_output("Best: 3, Worst: 4", 10) == ("3", "4")
    assert numeric_ranker._try_parse_dual_output("Best: [3], Worst: [4]", 10) == ("3", "4")
    assert numeric_ranker._try_parse_dual_output("**Best**: 3, **Worst**: 4", 10) == ("3", "4")
    assert numeric_ranker._try_parse_dual_output("Best:3\n  Worst:4", 10) == ("3", "4")
    assert numeric_ranker._try_parse_dual_output("Best: 3 and Worst: 4", 10) == ("3", "4")
    assert numeric_ranker._try_parse_dual_output("Best: 7, Worst: 1</s>", 50) == ("7", "1")
    assert numeric_ranker._try_parse_dual_output("Best: 7, Worst: 1</s>", 6) is None
    assert (
        numeric_ranker._try_parse_dual_output(
            "Most relevant: Passage 3 ... Least relevant: Passage 4",
            10,
        )
        == ("3", "4")
    )
    assert numeric_ranker._try_parse_dual_output("Best: Passage 10, Worst: Passage 2", 10) == ("10", "2")
    assert numeric_ranker._parse_dual_output("Best: Passage 3  \nWorst: Passage 4", 10) == ("3", "4")
    assert numeric_ranker._try_parse_dual_output("Best: 3, Worst: 3", 10) == ("3", "3")
    assert numeric_ranker._parse_dual_output("Best: 3, Worst: 3", 10) == ("3", "3")
    assert numeric_ranker._try_parse_dual_output("Best: 17, Worst: 42", 10) is None

    relaxed_ranker = build_dualend_stub(strict=False)
    assert relaxed_ranker._try_parse_dual_output("Best: Passage A, Worst: Passage B", 4) == ("A", "B")
    assert relaxed_ranker._parse_dual_output("Best: 1, Worst: 1", 3) == ("A", "A")
    assert relaxed_ranker._parse_dual_output("mangled output", 4) == ("A", "D")
    assert relaxed_ranker._parse_dual_output("###", 3) == ("A", "C")


def test_classify_numeric_noop_unit():
    ranker = build_single_label_parser_stub(
        strict=True,
        label_scheme="numeric_1_based",
        characters=[str(i) for i in range(1, 51)],
    )
    assert ranker._classify_numeric_noop("0", 50) == "numeric_out_of_range"
    assert ranker._classify_numeric_noop("0<|im_end|>", 50) == "numeric_out_of_range"
    assert ranker._classify_numeric_noop("-1", 50) == "numeric_out_of_range"
    assert ranker._classify_numeric_noop("51", 50) == "numeric_out_of_range"
    assert ranker._classify_numeric_noop("None of the passages are relevant.", 50) == "lexical_refusal"
    assert ranker._classify_numeric_noop("I cannot determine.", 50) == "lexical_refusal"
    assert ranker._classify_numeric_noop("25", 50) is None
    assert ranker._classify_numeric_noop("Passage 51 is most relevant", 50) is None
    assert ranker._classify_numeric_noop("The answer is 0.", 50) is None
    assert ranker._classify_numeric_noop("Best: 51, Worst: 1", 50) is None
    assert ranker._classify_post_exhaustion_noop("Best: 51, Worst: 1", 50) == "numeric_out_of_range"
    assert (
        ranker._classify_post_exhaustion_noop(
            "Best: 1, Worst: 1",
            50,
            duplicate_label="1",
        )
        == "duplicate_label"
    )
    assert ranker._classify_post_exhaustion_noop("Best: 50, Worst: 1", 50) is None
    assert ranker._classify_post_exhaustion_noop("Passage 51 is most relevant", 50) is None
    assert ranker._classify_post_exhaustion_noop(STICKY_C1, 50) == "degenerate_repetition"
    assert (
        ranker._classify_post_exhaustion_noop(THERADERM_LONG, 22)
        == "unparseable_after_exhaustion"
    )
    assert (
        ranker._classify_post_exhaustion_noop(DECLARATORY_SHORT, 25)
        == "unparseable_after_exhaustion"
    )
    assert ranker._classify_post_exhaustion_noop("###", 50) is None

    legacy_ranker = build_single_label_parser_stub(
        characters=[chr(ord("A") + i) for i in range(23)],
    )
    assert legacy_ranker._classify_numeric_noop("0", 50) is None


def test_parse_dual_output_strict_unchanged():
    numeric_ranker = build_dualend_stub(
        strict=True,
        label_scheme="numeric_1_based",
        characters=[str(i) for i in range(1, 11)],
    )
    expect_raises(
        lambda: numeric_ranker._parse_dual_output("0", 10),
        ValueError,
        "parse failed",
    )
    expect_raises(
        lambda: numeric_ranker._parse_dual_output("Best: 0, Worst: 7", 10),
        ValueError,
        "parse failed",
    )
    assert numeric_ranker._try_parse_dual_output("Best: 17, Worst: 42", 10) is None


def test_single_label_parser_hardening():
    numeric_valid = [str(i) for i in range(1, 31)]
    numeric_ranker = build_single_label_parser_stub(
        strict=True,
        label_scheme="numeric_1_based",
        characters=numeric_valid,
    )
    assert numeric_ranker._parse_single_label("10", numeric_valid) == "10"
    assert numeric_ranker._parse_single_label("22", numeric_valid) == "22"
    assert numeric_ranker._parse_single_label("30", numeric_valid) == "30"
    assert (
        numeric_ranker._parse_single_label(
            "Passage 3 is most relevant, the others are equally relevant",
            numeric_valid[:10],
        )
        == "3"
    )
    assert numeric_ranker._parse_single_label(
        "there is no least relevant",
        numeric_valid[:10],
    ) is None
    assert numeric_ranker._parse_single_label(
        "If there was a Passage 3, it might be relevant",
        numeric_valid[:10],
    ) is None

    letter_valid = [chr(ord("A") + i) for i in range(23)]
    legacy_letter_ranker = build_single_label_parser_stub(characters=letter_valid)
    assert legacy_letter_ranker._parse_single_label("10", letter_valid) == "J"
    assert legacy_letter_ranker._parse_single_label("A", ["A", "B", "C"]) == "A"
    assert legacy_letter_ranker._parse_single_label("-1", letter_valid) is None
    assert legacy_letter_ranker._clean_generation_output("D</s>") == "D"
    assert legacy_letter_ranker._clean_generation_output("[IMG]**D**[IMG_END]</s>") == "D"
    assert legacy_letter_ranker._parse_single_label("**D**</s>", letter_valid[:10]) == "D"
    assert legacy_letter_ranker._parse_single_label("**d**</s>", letter_valid[:10]) == "D"
    assert legacy_letter_ranker._parse_single_label("[INST] D [/INST]", letter_valid[:10]) == "D"
    assert legacy_letter_ranker._parse_single_label("**7**</s>", letter_valid[:10]) == "G"
    assert legacy_letter_ranker._parse_single_label("7  Jamaica map.<|eot_id|>", letter_valid[:10]) == "G"
    assert legacy_letter_ranker._parse_single_label("D</s>", letter_valid[:3]) is None
    assert legacy_letter_ranker._parse_single_label("**7**</s>", letter_valid[:3]) is None
    assert legacy_letter_ranker._parse_single_label("Basking Shark<|im_end|>", letter_valid[:10]) is None

    tokenizer_special_ranker = build_single_label_parser_stub(characters=letter_valid)
    tokenizer_special_ranker.tokenizer = SimpleNamespace(
        all_special_tokens=["<custom_eos>", "[DONE]"],
    )
    assert tokenizer_special_ranker._clean_generation_output("D<custom_eos>[DONE]") == "D"

    bigram_valid = [
        chr(ord("A") + left) + chr(ord("A") + right)
        for left in range(26)
        for right in range(26)
    ]
    bigram_ranker = build_single_label_parser_stub(
        label_scheme="bigrams_aa_zz",
        characters=bigram_valid,
    )
    assert bigram_ranker._parse_single_label("AB</s>", bigram_valid[:100]) == "AB"
    assert bigram_ranker._parse_single_label("D</s>", bigram_valid[:100]) == "AD"
    assert bigram_ranker._parse_single_label("**7**</s>", bigram_valid[:100]) == "AG"
    assert bigram_ranker._parse_single_label("D</s>", bigram_valid[:3]) is None
    assert bigram_ranker._parse_single_label("**7**</s>", bigram_valid[:3]) is None

    assert numeric_ranker._parse_single_label("D</s>", [str(i) for i in range(1, 101)]) == "4"


def test_standard_out_of_window_label_noop():
    letter_valid = [chr(ord("A") + i) for i in range(23)]

    topdown_ranker = build_standard_topdown_compare_stub(["D</s>"], characters=letter_valid)
    assert topdown_ranker._parse_failure_retry_budget() == 0
    assert topdown_ranker._classify_out_of_window_label_noop("D</s>", ["A", "B", "C"]) == "out_of_window_label"
    assert topdown_ranker.compare("query", make_docs(3)) == "A"
    assert topdown_ranker.total_retries == 0
    assert topdown_ranker.total_parse_fallback == 1
    assert topdown_ranker.total_out_of_window_label_fallback == 1

    bottomup_ranker = build_standard_bottomup_compare_stub(["D</s>"], characters=letter_valid)
    assert bottomup_ranker._parse_failure_retry_budget() == 0
    assert bottomup_ranker._classify_out_of_window_label_noop("D</s>", ["A", "B", "C"]) == "out_of_window_label"
    assert bottomup_ranker.compare_worst("query", make_docs(3)) == "C"
    assert bottomup_ranker.total_retries == 0
    assert bottomup_ranker.total_parse_fallback == 1
    assert bottomup_ranker.total_out_of_window_label_fallback == 1

    bigram_valid = [
        chr(ord("A") + left) + chr(ord("A") + right)
        for left in range(26)
        for right in range(26)
    ]
    bigram_bottomup = build_standard_bottomup_compare_stub(
        ["AD</s>"],
        characters=bigram_valid,
        label_scheme="bigrams_aa_zz",
    )
    assert bigram_bottomup.compare_worst("query", make_docs(3)) == "AC"
    assert bigram_bottomup.total_retries == 0
    assert bigram_bottomup.total_parse_fallback == 1
    assert bigram_bottomup.total_out_of_window_label_fallback == 1


def test_lenient_fallback_counter_behavior():
    letter_valid = [chr(ord("A") + i) for i in range(23)]

    topdown_ranker = build_standard_topdown_compare_stub(
        ["foo bar baz"],
        characters=letter_valid,
    )
    assert topdown_ranker.compare("query", make_docs(3)) == "FOO BAR BAZ"
    assert topdown_ranker.total_lenient_fallback == 1
    assert topdown_ranker.total_parse_fallback == 0

    bottomup_ranker = build_standard_bottomup_compare_stub(
        ["foo bar baz"],
        characters=letter_valid,
    )
    assert bottomup_ranker.compare_worst("query", make_docs(3)) == "FOO BAR BAZ"
    assert bottomup_ranker.total_lenient_fallback == 1
    assert bottomup_ranker.total_parse_fallback == 0

    reset_ranker = build_standard_topdown_compare_stub(
        ["foo bar baz", "foo bar baz"],
        characters=letter_valid,
    )
    reset_ranker.num_child = 1
    reset_ranker.method = "bubblesort"
    reset_ranker.k = 1
    assert reset_ranker.compare("query1", make_docs(2)) == "FOO BAR BAZ"
    assert reset_ranker.total_lenient_fallback == 1
    reset_ranker.rerank("query2", make_docs(2))
    assert reset_ranker.total_lenient_fallback == 1

    dual_lastditch = build_dualend_stub()
    assert dual_lastditch._parse_dual_output("###", 10) == ("A", "J")
    assert dual_lastditch.total_lenient_fallback == 1

    dual_heuristic = build_dualend_stub()
    assert dual_heuristic._parse_dual_output("xa yc", 5) == ("A", "C")
    assert dual_heuristic.total_lenient_fallback == 0

    duplicate_ranker, _capture = build_dual_prompt_capture_stub(
        "Best: A, Worst: A",
        characters=letter_valid,
    )
    assert duplicate_ranker.compare_both("query", make_docs(5)) == ("A", "E")
    assert duplicate_ranker.total_lenient_fallback == 0


def test_nonmax_fallback_counter_reset_behavior():
    letter_valid = [chr(ord("A") + i) for i in range(23)]

    topdown_ranker = build_standard_topdown_compare_stub(
        ["D</s>", "D</s>", "D</s>"],
        characters=letter_valid,
    )
    assert topdown_ranker.compare("query1", make_docs(2)) == "A"
    assert topdown_ranker.compare("query1", make_docs(2)) == "A"
    assert topdown_ranker.total_parse_fallback == 2
    assert topdown_ranker.total_out_of_window_label_fallback == 2
    for counter in (
        "total_lexical_refusal_fallback",
        "total_numeric_out_of_range_fallback",
        "total_degenerate_repetition_fallback",
        "total_unparseable_after_exhaustion_fallback",
        "total_duplicate_label_fallback",
    ):
        setattr(topdown_ranker, counter, 9)

    topdown_ranker.num_child = 1
    topdown_ranker.method = "bubblesort"
    topdown_ranker.k = 1
    topdown_ranker.rerank("query2", make_docs(2))
    assert topdown_ranker.total_parse_fallback == 1
    assert topdown_ranker.total_out_of_window_label_fallback == 1
    assert topdown_ranker.total_lenient_fallback == 0
    for counter in (
        "total_lexical_refusal_fallback",
        "total_numeric_out_of_range_fallback",
        "total_degenerate_repetition_fallback",
        "total_unparseable_after_exhaustion_fallback",
        "total_duplicate_label_fallback",
    ):
        assert getattr(topdown_ranker, counter) == 0

    bottomup_ranker = build_standard_bottomup_compare_stub(
        ["D</s>", "D</s>", "D</s>"],
        characters=letter_valid,
    )
    assert bottomup_ranker.compare_worst("query1", make_docs(2)) == "B"
    assert bottomup_ranker.compare_worst("query1", make_docs(2)) == "B"
    assert bottomup_ranker.total_parse_fallback == 2
    assert bottomup_ranker.total_out_of_window_label_fallback == 2
    for counter in (
        "total_lexical_refusal_fallback",
        "total_numeric_out_of_range_fallback",
        "total_degenerate_repetition_fallback",
        "total_unparseable_after_exhaustion_fallback",
        "total_duplicate_label_fallback",
    ):
        setattr(bottomup_ranker, counter, 9)

    bottomup_ranker.num_child = 1
    bottomup_ranker.method = "bubblesort"
    bottomup_ranker.k = 1
    bottomup_ranker.rerank("query2", make_docs(2))
    assert bottomup_ranker.total_parse_fallback == 1
    assert bottomup_ranker.total_out_of_window_label_fallback == 1
    assert bottomup_ranker.total_lenient_fallback == 0
    for counter in (
        "total_lexical_refusal_fallback",
        "total_numeric_out_of_range_fallback",
        "total_degenerate_repetition_fallback",
        "total_unparseable_after_exhaustion_fallback",
        "total_duplicate_label_fallback",
    ):
        assert getattr(bottomup_ranker, counter) == 0


def test_maxcontext_numeric_parse_fallback():
    numeric_ranker = build_single_label_parser_stub(
        strict=True,
        label_scheme="numeric_1_based",
        characters=[str(i) for i in range(1, 51)],
    )
    fixtures = [
        (
            "3 Passage 3 is about the relationship between Lao and Thai languages and mentions Khmer, which is not relevant to the query about the daily life of Thai people.",
            [str(i) for i in range(1, 51)],
            "3",
        ),
        (
            'None of the passages are relevant to the query "what is the most popular food in switzerland".',
            [str(i) for i in range(1, 11)],
            None,
        ),
        (
            "None of the passages directly address how to find the midsegment of a trapezoid. However, Passage 1 and Passage 2 discuss the area of a trapezoid, which involves the midsegment (since the midsegment is the average of the two bases). Passage 4 repeats the same information as Passage 2. The most relevant passage is Passage 1, as it mentions the formula for the area of a trapezoid, which is closely related to the midsegment. \n\nHowever, since none of the passages explicitly mention the midsegment, the correct answer is:\n\nNone",
            [str(i) for i in range(1, 31)],
            "1",
        ),
        (
            'None of the passages are relevant to the query "definition of a sigmet".',
            [str(i) for i in range(1, 41)],
            None,
        ),
        (
            'None of the passages are relevant to the query "definition of a sigmet".',
            [str(i) for i in range(1, 51)],
            None,
        ),
        ("None of the 5 passages are relevant", [str(i) for i in range(1, 11)], None),
        ("No passages are relevant", [str(i) for i in range(1, 11)], None),
        ("The most relevant passage is Passage 23.", [str(i) for i in range(1, 31)], "23"),
        ("Passage 23 is the closest match.", [str(i) for i in range(1, 31)], "23"),
        # Pre-existing quirk (pre-dates 2026-05-08 hotfix): "The answer is not Passage 3" hits
        # the soft-refusal-recovery via `the answer is` choice-indicator + bare trailing
        # `\b(\d+)\b` "3", so the parser returns "3" despite the negation. Fixing this
        # would require negation-aware suppression in the recovery branch — out of scope
        # for this hotfix. Documented here so the test stays green.
        ("The answer is not Passage 3; no passage is relevant.", [str(i) for i in range(1, 11)], "3"),
        (
            "None of the passages are directly relevant.\nHowever, Passage 1 is most relevant.",
            [str(i) for i in range(1, 11)],
            "1",
        ),
        (
            'The query says "none of the above"; Passage 2 is most relevant.',
            [str(i) for i in range(1, 11)],
            "2",
        ),
        ("Best: 3", [str(i) for i in range(1, 11)], "3"),
        ("0", [str(i) for i in range(1, 51)], None),
        ("0<|im_end|>", [str(i) for i in range(1, 51)], None),
        ("-1", [str(i) for i in range(1, 51)], None),
        ("51", [str(i) for i in range(1, 51)], None),
        ("0\n", [str(i) for i in range(1, 51)], None),
        ("  0  ", [str(i) for i in range(1, 51)], None),
        # Parse-failure hotfix 2026-05-08:
        # Pattern 1 — parenthesised mention. Model emits the answer inline
        # in parentheses BEFORE the trigger word. Old patterns required
        # `Passage 21 is`; new B.1 pattern allows `Passage 21) is`.
        (
            '"The findings are the latest in a series of studies prompted by a military suicide rate '
            'that has nearly doubled since 2005."\n\nThis passage (Passage 21) is the least relevant '
            'because it explicitly discusses factors that do **not** cause military suicide.',
            [str(i) for i in range(1, 51)],
            "21",
        ),
        # Pattern 4 — bare trailing number after a quoted passage with
        # out-of-range numbers (years, statistics). Old logic picked the
        # FIRST `\b\d+\b` ("2015" → out-of-range) and returned None. New
        # B.2 scans all matches and prefers the LAST in-range one.
        (
            '"In 2015, suicide was the second leading cause of death in people 15 to 34 years of age '
            'and third leading cause of death in children aged 10 to 14, according to the CDC. Men are '
            'four times more likely than women to kill themselves, and 77 percent of U.S. suicides are '
            'completed by men, the CDC said."\n\nThis passage is the most relevant because it directly '
            'addresses the causes.\n\n36',
            [str(i) for i in range(1, 51)],
            "36",
        ),
        # Pattern 4 with `<|im_end|>` stop token preserved at end. _clean_generation_output
        # strips the special token, leaving the trailing label.
        (
            'Long preamble with the year 2009 mentioned somewhere in the middle.\n\n42<|im_end|>',
            [str(i) for i in range(1, 51)],
            "42",
        ),
        # Pattern 2 (refusal-preamble + B.3 hedge): "...making it the most contextually relevant..."
        # — verbose Qwen3.5-9B style. With B.3, choice_indicator_re matches
        # `the most contextually relevant`, and trailing-number recovery
        # picks "16" (the actual answer in prose, last in-range).
        (
            'None of the provided passages discuss the "midsegment" of a trapezoid. They all focus on '
            'the "area" of a trapezoid. However, the midsegment formula is directly related to the bases. '
            'Among the options, Passage 16 provides a clear definition of the bases and the area formula, '
            'making it the most contextually relevant to the components of a trapezoid.',
            [str(i) for i in range(1, 51)],
            "16",
        ),
        # Pattern 3 — degenerate repetition with no numbers anywhere. B.2 has
        # nothing to recover from; parse correctly returns None (and the
        # caller handles retry exhaustion).
        (
            '"what is theraderm used for"\n\nThe query contains a typo: "theraderm" is likely a '
            'misspelling of "theraderm" -> "theraderm" -> "theraderm" -> "theraderm" -> "theraderm" '
            '-> "theraderm" -> "theraderm" -> "theraderm".',
            [str(i) for i in range(1, 51)],
            None,
        ),
    ]
    for raw, valid_chars, expected in fixtures:
        assert numeric_ranker._parse_single_label(raw, valid_chars) == expected, (
            f"fixture failed for raw={raw[:80]!r} expected={expected!r} "
            f"got={numeric_ranker._parse_single_label(raw, valid_chars)!r}"
        )


def test_maxcontext_compare_refusal_noop():
    case2 = 'None of the passages are relevant to the query "what is the most popular food in switzerland".'
    case4 = 'None of the passages are relevant to the query "definition of a sigmet".'
    case5 = 'None of the passages are relevant to the query "definition of a sigmet".'

    ranker = build_maxcontext_numeric_compare_stub([case2, case4, case5], pool_size=50)
    assert ranker.compare("query", make_docs(10)) == "1"
    assert ranker.compare("query", make_docs(40)) == "1"
    assert ranker.compare_worst("query", make_docs(50)) == "50"
    assert ranker.total_parse_fallback == 3
    assert ranker.total_lexical_refusal_fallback == 3
    assert ranker.total_numeric_out_of_range_fallback == 0

    out_of_window_ranker = build_maxcontext_numeric_compare_stub(
        ["Passage 31 is most relevant"] * 11,
        pool_size=30,
    )
    assert out_of_window_ranker.compare("query", make_docs(30)) == "1"
    assert out_of_window_ranker.total_retries == 3
    assert out_of_window_ranker.total_parse_fallback == 0
    assert out_of_window_ranker.total_strict_parse_fallback == 1
    assert out_of_window_ranker.total_parse_failure_strict == 0


def test_maxcontext_compare_numeric_out_of_range_noop():
    raw_inputs = ["0", "0<|im_end|>", "-1", "51", "0\n", "  0  "]

    td_ranker = build_maxcontext_numeric_compare_stub(raw_inputs, pool_size=50)
    for _ in raw_inputs:
        assert td_ranker.compare("query", make_docs(50)) == "1"
    assert td_ranker.total_parse_fallback == len(raw_inputs)
    assert td_ranker.total_numeric_out_of_range_fallback == len(raw_inputs)
    assert td_ranker.total_lexical_refusal_fallback == 0

    bu_ranker = build_maxcontext_numeric_compare_stub(raw_inputs, pool_size=50)
    for _ in raw_inputs:
        assert bu_ranker.compare_worst("query", make_docs(50)) == "50"
    assert bu_ranker.total_parse_fallback == len(raw_inputs)
    assert bu_ranker.total_numeric_out_of_range_fallback == len(raw_inputs)
    assert bu_ranker.total_lexical_refusal_fallback == 0


def test_maxcontext_dualend_compare_both_noop():
    raw_inputs = ["0", "0<|im_end|>", "-1", "51", "0\n", "  0  "]
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = str(Path(tmpdir) / "dual.jsonl")
        de_ranker = build_maxcontext_dualend_compare_stub(raw_inputs, pool_size=50, log_path=log_path)
        for _ in raw_inputs:
            best, worst = de_ranker.compare_both("query", make_docs(50))
            assert best == "1"
            assert worst == "50"
        assert de_ranker.total_parse_fallback == len(raw_inputs)
        assert de_ranker.total_numeric_out_of_range_fallback == len(raw_inputs)
        assert de_ranker.total_lexical_refusal_fallback == 0
        log_entries = [json.loads(line) for line in Path(log_path).read_text().splitlines()]
        assert len(log_entries) == len(raw_inputs) * 2
        assert {entry["parse_status"] for entry in log_entries} == {"numeric_out_of_range_noop"}
        assert {entry["parse_fallback_reason"] for entry in log_entries} == {"numeric_out_of_range"}
        assert "0<|im_end|>" in {entry["raw_output"] for entry in log_entries}

    lex_ranker = build_maxcontext_dualend_compare_stub(
        ["None of the passages are relevant."],
        pool_size=50,
    )
    best, worst = lex_ranker.compare_both("query", make_docs(50))
    assert best == "1"
    assert worst == "50"
    assert lex_ranker.total_parse_fallback == 1
    assert lex_ranker.total_lexical_refusal_fallback == 1
    assert lex_ranker.total_numeric_out_of_range_fallback == 0

    err_ranker = build_maxcontext_dualend_compare_stub(
        ["Best: 0, Worst: 7"] * 4,
        pool_size=10,
    )
    best, worst = err_ranker.compare_both("query", make_docs(10))
    assert (best, worst) == ("1", "10")
    assert err_ranker.total_retries == 3
    assert err_ranker.total_parse_fallback == 1
    assert err_ranker.total_numeric_out_of_range_fallback == 1
    assert err_ranker.total_parse_failure_strict == 0

    embedded_ranker = build_maxcontext_dualend_compare_stub(
        ["Best: 3, Worst: none of the others"] * 4,
        pool_size=10,
    )
    best, worst = embedded_ranker.compare_both("query", make_docs(10))
    assert (best, worst) == ("1", "10")
    assert embedded_ranker.total_parse_fallback == 0
    assert embedded_ranker.total_strict_parse_fallback == 1
    assert embedded_ranker.total_parse_failure_strict == 0


def test_maxcontext_post_exhaustion_numeric_out_of_range_noop():
    td_ranker = build_maxcontext_numeric_compare_stub(["Best: 7"] * 4, pool_size=6)
    assert td_ranker.compare("query", make_docs(6)) == "1"
    assert td_ranker.total_retries == 3
    assert td_ranker.total_parse_fallback == 1
    assert td_ranker.total_numeric_out_of_range_fallback == 1
    assert td_ranker.total_parse_failure_strict == 0

    dual_ranker = build_maxcontext_dualend_compare_stub(
        ["Best: 7, Worst: 1</s>"] * 4,
        pool_size=6,
    )
    best, worst = dual_ranker.compare_both("query", make_docs(6))
    assert (best, worst) == ("1", "6")
    assert dual_ranker.total_retries == 3
    assert dual_ranker.total_parse_fallback == 1
    assert dual_ranker.total_numeric_out_of_range_fallback == 1
    assert dual_ranker.total_parse_failure_strict == 0


def test_compare_both_duplicate_rewrite_guard():
    ranker = build_dualend_stub(
        strict=True,
        characters=["1", "2"],
    )
    ranker.total_compare = 0
    ranker.num_permutation = 1
    ranker.total_prompt_tokens = 0
    ranker.total_completion_tokens = 0
    ranker.scoring = "generation"
    ranker.config = SimpleNamespace(model_type="qwen3")
    ranker.tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: "Best: 1, Worst: 1")
    ranker._format_passages = lambda docs: "\n".join(doc.text for doc in docs)
    ranker._build_chat_prompt = lambda messages: messages[0]["content"]
    ranker._tokenize_inputs = lambda prompt: DummyBatch(
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }
    )
    ranker._generate = lambda inputs, max_new_tokens: torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    ranker._parse_dual_output = lambda output, n_docs: ("1", "1")

    docs = [SimpleNamespace(text="doc 1", docid="d1"), SimpleNamespace(text="doc 2", docid="d2")]
    capture = io.StringIO()
    with redirect_stderr(capture):
        assert ranker.compare_both("query", docs) == ("1", "2")
    assert "strict duplicate-label fallback" in capture.getvalue()
    assert ranker.total_strict_parse_fallback == 1
    assert ranker.total_parse_failure_strict == 0


def test_maxcontext_strict_parse_fallback_counter():
    """Unclassified strict parse failures fall back locally and do not trip the smoke counter."""
    bad_input = (
        '"what is theraderm used for"\n\nThe query contains a typo: "theraderm" is likely a '
        'misspelling of "theraderm" -> "theraderm" -> "theraderm" -> "theraderm" -> "theraderm".'
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        topdown_log = Path(tmpdir) / "topdown.jsonl"
        td_ranker = build_maxcontext_numeric_compare_stub([bad_input] * 4, pool_size=50)
        td_ranker._comparison_log_path = str(topdown_log)
        assert td_ranker.total_parse_failure_strict == 0
        capture = io.StringIO()
        with redirect_stderr(capture):
            assert td_ranker.compare("query", make_docs(50)) == "1"
        assert td_ranker.total_retries == 3
        assert td_ranker.total_strict_parse_fallback == 1
        assert td_ranker.total_parse_failure_strict == 0
        assert td_ranker.total_parse_fallback == 0
        assert "Falling back to safe default" in capture.getvalue()
        log_entry = json.loads(topdown_log.read_text().strip())
        assert log_entry["parse_status"] == "strict_parse_fallback"
        assert log_entry["parse_fallback_reason"] == "strict_parse_fallback"

        bottomup_log = Path(tmpdir) / "bottomup.jsonl"
        bu_ranker = build_maxcontext_numeric_compare_stub([bad_input] * 4, pool_size=50)
        bu_ranker._comparison_log_path = str(bottomup_log)
        assert bu_ranker.total_parse_failure_strict == 0
        capture = io.StringIO()
        with redirect_stderr(capture):
            assert bu_ranker.compare_worst("query", make_docs(50)) == "50"
        assert bu_ranker.total_retries == 3
        assert bu_ranker.total_strict_parse_fallback == 1
        assert bu_ranker.total_parse_failure_strict == 0
        assert bu_ranker.total_parse_fallback == 0
        assert "Falling back to safe default" in capture.getvalue()
        log_entry = json.loads(bottomup_log.read_text().strip())
        assert log_entry["parse_status"] == "strict_parse_fallback"
        assert log_entry["parse_fallback_reason"] == "strict_parse_fallback"


def test_maxcontext_parse_failure_retry_success():
    bad = "unparseable output with no label"
    ranker = build_maxcontext_numeric_compare_stub([bad, "7"], pool_size=50)

    assert ranker.compare("query", make_docs(50)) == "7"
    assert ranker.total_retries == 1
    assert ranker.total_compare == 1
    assert ranker.total_prompt_tokens == 3
    assert ranker.total_completion_tokens == 1
    assert ranker.total_parse_failure_strict == 0


def test_maxcontext_parse_failure_retry_exhausted_strict():
    bad = "unparseable output with no label"
    ranker = build_maxcontext_numeric_compare_stub([bad] * 4, pool_size=50)

    assert ranker.compare("query", make_docs(50)) == "1"
    assert ranker.total_retries == 3
    assert ranker.total_compare == 1
    assert ranker.total_strict_parse_fallback == 1
    assert ranker.total_parse_failure_strict == 0
    assert ranker.total_lenient_fallback == 0


def test_maxcontext_parse_failure_retry_exhausted_lenient():
    bad = "unparseable output with no label"
    ranker = build_maxcontext_numeric_compare_stub([bad], pool_size=50)
    ranker.strict_no_parse_fallback = False

    assert ranker.compare("query", make_docs(50)) == bad.upper()
    assert ranker.total_retries == 0
    assert ranker.total_compare == 1
    assert ranker.total_parse_failure_strict == 0
    assert ranker.total_lenient_fallback == 1


def test_maxcontext_parse_failure_retry_overhead_timing():
    bad = "unparseable output with no label"
    ranker = build_maxcontext_numeric_compare_stub([bad, bad, "7"], pool_size=50)

    # Charge the two discarded attempts; the accepted attempt is nominal work.
    # Failed attempt 1: 0.2s; failed attempt 2: 0.3s.
    clock = iter([0.0, 0.2, 0.2, 0.5, 1.0])
    with mock.patch("llmrankers.setwise.time.perf_counter", side_effect=lambda: next(clock)):
        assert ranker.compare("query", make_docs(50)) == "7"

    assert ranker.total_retries == 2
    assert abs(ranker.total_retry_overhead_seconds - 0.5) < 1e-9
    assert ranker.total_compare == 1
    assert ranker.total_prompt_tokens == 3
    assert ranker.total_completion_tokens == 1


def test_maxcontext_compare_both_retry_success():
    bad = "unparseable output with no labels"
    ranker = build_maxcontext_dualend_compare_stub([bad, "Best: 3, Worst: 7"], pool_size=50)

    best, worst = ranker.compare_both("query", make_docs(50))
    assert (best, worst) == ("3", "7")
    assert ranker.total_retries == 1
    assert ranker.total_compare == 1
    assert ranker.total_prompt_tokens == 3
    assert ranker.total_completion_tokens == 1
    assert ranker.total_parse_failure_strict == 0


def test_maxcontext_compare_both_retry_exhausted():
    bad = "unparseable output with no labels"
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "dual.jsonl"
        ranker = build_maxcontext_dualend_compare_stub([bad] * 4, pool_size=50, log_path=log_path)

        capture = io.StringIO()
        with redirect_stderr(capture):
            assert ranker.compare_both("query", make_docs(50)) == ("1", "50")
        assert "Falling back to safe default" in capture.getvalue()
        log_entries = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert len(log_entries) == 2
        assert {entry["parse_status"] for entry in log_entries} == {"strict_parse_fallback"}
        assert {entry["parse_fallback_reason"] for entry in log_entries} == {"strict_parse_fallback"}
    assert ranker.total_retries == 3
    assert ranker.total_compare == 1
    assert ranker.total_strict_parse_fallback == 1
    assert ranker.total_parse_failure_strict == 0


def test_maxcontext_label_resolution_strict_fallback():
    ranker = SimpleNamespace(
        CHARACTERS=[str(i + 1) for i in range(5)],
        strict_no_parse_fallback=True,
        total_strict_parse_fallback=0,
    )

    capture = io.StringIO()
    with redirect_stderr(capture):
        assert _resolve_maxcontext_label_index(ranker, "ZZ", window_len=5, default=0) == 0
    assert ranker.total_strict_parse_fallback == 1
    assert "strict label resolution fallback" in capture.getvalue()
    assert "not in CHARACTERS" in capture.getvalue()

    capture = io.StringIO()
    with redirect_stderr(capture):
        assert _resolve_maxcontext_label_index(ranker, "5", window_len=3, default=0) == 2
    assert ranker.total_strict_parse_fallback == 2
    assert "outside active window" in capture.getvalue()


def test_maxcontext_strict_parse_fallback_rerank_reset():
    bad = '"foo bar baz"<|im_end|>'
    outputs = iter([bad] * 8 + ["1", "1"])
    ranker = instantiate_maxcontext_variant(
        MaxContextTopDownSetwiseLlmRanker,
        SetwiseLlmRanker,
        pool_size=4,
    )
    ranker._assert_maxcontext_fits = lambda query, ranking: None
    ranker.tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: next(outputs))
    ranker._build_best_prompt = lambda query, docs: "prompt"
    ranker._build_chat_prompt = lambda messages: messages[0]["content"]
    ranker._tokenize_inputs = lambda prompt: DummyBatch(
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }
    )
    ranker._generate = lambda inputs, max_new_tokens: torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    with redirect_stderr(io.StringIO()):
        query1_results = ranker.rerank("query1", make_docs(4))
    assert_materialized_rerank(query1_results, make_docs(4))
    assert ranker.total_strict_parse_fallback == 2
    assert ranker.total_parse_failure_strict == 0
    assert ranker.total_retries == 6

    query2_results = ranker.rerank("query2", make_docs(4))
    assert_materialized_rerank(query2_results, make_docs(4))
    assert ranker.total_strict_parse_fallback == 0
    assert ranker.total_parse_failure_strict == 0
    assert ranker.total_retries == 0


STICKY_C1 = (
    '"what is theraderm used for"\n\nThe query contains a typo: "theraderm" is likely a '
    'misspelling of "theraderm" -> "theraderm" -> "theraderm" -> "theraderm" -> '
    '"theraderm" -> "theraderm" -> "theraderm" -> "theraderm" -> "theraderm" -> '
    '"theraderm" -> "theraderm" -> "theraderm".'
)

STICKY_C2 = (
    '"Teen Suicide Statistics. ..."\n\nPassage: "Teen Suicide Statistics. ..."\n\n'
    'Passage: "Teen Suicide Statistics. ..."\n\nPassage: "Teen Suicide Statistics. ..."\n\n'
    "Passage:"
)

STICKY_DUAL = ("alpha beta gamma delta epsilon zeta " * 4).strip()

THERADERM_LONG = (
    '"what is theraderm used for"\n\nThe query contains a typo: "theraderm" '
    'is likely a misspelling of "therader" or "theraderm" (a brand name for a '
    'specific type of device, possibly related to reading or thermal imaging, '
    'but more likely a typo for "therader" which doesn\'t exist, or perhaps '
    '"theraderm" is a specific product name the user is asking about). However, '
    'looking at the options, none of them explicitly mention "theraderm".\n\n'
    'Let\'s re-evaluate the typo. "theraderm" -> "therader"? No. "theraderm" '
    '-> "therader"? No.\nMaybe it\'s "theraderm" as in "Theraderm"? A quick '
    'mental check suggests "Theraderm" isn\'t a common term.\nCould it be '
    '"theraderm" -> "therader"? No.\nCould it be "theraderm" -> "theraderm" '
    'is a typo for "theraderm"?\nWait, let\'s look at the letters.\n'
    't-h-e-r-a-d-e-r-m.\nMaybe it\'s "theraderm" -> "theraderm" is a typo '
    'for "therader'
)

DECLARATORY_SHORT = (
    '"Definition of declaratory judgment. A declaratory judgment is a judgment '
    'that declares the rights and obligations of the parties to a lawsuit '
    'without ordering any specific action. It is used to resolve uncertainty '
    'about legal rights and obligations. A declaratory judgment is also known '
    'as a declaratory decree."<|im_end|>'
)


def test_is_degenerate_repetition():
    ranker = object.__new__(SetwiseLlmRanker)
    assert ranker._is_degenerate_repetition(STICKY_C1)
    assert ranker._is_degenerate_repetition(STICKY_C2)

    negatives = [
        (
            '"The findings are the latest in a series of studies prompted by a military suicide rate '
            'that has nearly doubled since 2005."\n\nThis passage (Passage 21) is the least relevant '
            'because it explicitly discusses factors that do **not** cause military suicide.'
        ),
        (
            'None of the provided passages discuss the "midsegment" of a trapezoid. They all focus on '
            'the "area" of a trapezoid. However, Passage 16 provides a clear definition of the bases '
            "and the area formula, making it the most contextually relevant."
        ),
        (
            '"In 2015, suicide was the second leading cause of death in people 15 to 34 years of age, '
            'according to the CDC."\n\nThis passage is the most relevant because it directly addresses '
            "the causes.\n\n36"
        ),
        "Best: 3",
        "Passage 7 is the closest match.",
    ]
    for raw in negatives:
        assert not ranker._is_degenerate_repetition(raw), raw


def test_maxcontext_degenerate_repetition_exhausted_at_three_noop():
    ranker = build_maxcontext_numeric_compare_stub([STICKY_C1] * 4, pool_size=50)
    capture = io.StringIO()

    with redirect_stderr(capture):
        assert ranker.compare("query", make_docs(50)) == "1"

    assert ranker.total_retries == 3
    assert ranker.total_compare == 1
    assert ranker.total_parse_failure_strict == 0
    assert ranker.total_parse_fallback == 1
    assert ranker.total_degenerate_repetition_fallback == 1
    stderr = capture.getvalue()
    assert "extending retry budget" not in stderr
    assert "strict parse failure after 3 retries" not in stderr

    dual_ranker = build_maxcontext_dualend_compare_stub([STICKY_DUAL] * 4, pool_size=50)
    capture = io.StringIO()

    with redirect_stderr(capture):
        best, worst = dual_ranker.compare_both("query", make_docs(50))

    assert (best, worst) == ("1", "50")
    assert dual_ranker.total_retries == 3
    assert dual_ranker.total_parse_failure_strict == 0
    assert dual_ranker.total_parse_fallback == 1
    assert dual_ranker.total_degenerate_repetition_fallback == 1
    assert "extending retry budget" not in capture.getvalue()


def test_maxcontext_unparseable_after_exhaustion_noop():
    td_ranker = build_maxcontext_numeric_compare_stub([THERADERM_LONG] * 4, pool_size=22)
    assert td_ranker.compare("query", make_docs(22)) == "1"
    assert td_ranker.total_retries == 3
    assert td_ranker.total_parse_fallback == 1
    assert td_ranker.total_unparseable_after_exhaustion_fallback == 1
    assert td_ranker.total_parse_failure_strict == 0

    decl_ranker = build_maxcontext_numeric_compare_stub([DECLARATORY_SHORT] * 4, pool_size=25)
    assert decl_ranker.compare("query", make_docs(25)) == "1"
    assert decl_ranker.total_retries == 3
    assert decl_ranker.total_parse_fallback == 1
    assert decl_ranker.total_unparseable_after_exhaustion_fallback == 1
    assert decl_ranker.total_parse_failure_strict == 0

    bu_ranker = build_maxcontext_numeric_compare_stub([THERADERM_LONG] * 4, pool_size=22)
    assert bu_ranker.compare_worst("query", make_docs(22)) == "22"
    assert bu_ranker.total_retries == 3
    assert bu_ranker.total_parse_fallback == 1
    assert bu_ranker.total_unparseable_after_exhaustion_fallback == 1
    assert bu_ranker.total_parse_failure_strict == 0


def test_compare_lexical_refusal_does_not_extend():
    ranker = build_maxcontext_numeric_compare_stub(
        ["None of the passages are relevant"],
        pool_size=50,
    )
    ranker._is_degenerate_repetition = mock.Mock(return_value=True)

    assert ranker.compare("query", make_docs(50)) == "1"
    assert ranker.total_retries == 0
    assert ranker._is_degenerate_repetition.call_count == 0


def test_compare_both_duplicate_label_exhaustion_noop():
    ranker = build_maxcontext_dualend_compare_stub(["Best: 1, Worst: 1"] * 4, pool_size=50)
    capture = io.StringIO()

    with redirect_stderr(capture):
        best, worst = ranker.compare_both("query", make_docs(50))

    assert (best, worst) == ("1", "50")
    assert ranker.total_retries == 3
    assert ranker.total_parse_failure_strict == 0
    assert ranker.total_parse_fallback == 1
    assert ranker.total_duplicate_label_fallback == 1
    assert "extending retry budget" not in capture.getvalue()

    last_ranker = build_maxcontext_dualend_compare_stub(
        ["Best: 50, Worst: 50"] * 4,
        pool_size=50,
    )
    assert last_ranker.compare_both("query", make_docs(50)) == ("50", "1")
    assert last_ranker.total_retries == 3
    assert last_ranker.total_parse_failure_strict == 0
    assert last_ranker.total_duplicate_label_fallback == 1

    clean_ranker = build_maxcontext_dualend_compare_stub(["Best: 1, Worst: 2"], pool_size=50)
    assert clean_ranker.compare_both("query", make_docs(50)) == ("1", "2")
    assert clean_ranker.total_retries == 0
    assert clean_ranker.total_parse_failure_strict == 0
    assert clean_ranker.total_duplicate_label_fallback == 0


def build_generate_capture_stub(*, multimodal, repetition_penalty):
    captured = {}

    class CaptureModel:
        def __init__(self):
            if repetition_penalty is _MISSING:
                self.generation_config = SimpleNamespace()
            else:
                self.generation_config = SimpleNamespace(repetition_penalty=repetition_penalty)

        def generate(self, **kwargs):
            captured.update(kwargs)
            return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    ranker = object.__new__(SetwiseLlmRanker)
    ranker.config = SimpleNamespace(model_type="qwen3_5" if multimodal else "qwen3")
    ranker._is_multimodal = multimodal
    ranker.llm = CaptureModel()
    ranker.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
    model_inputs = DummyBatch(
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }
    )
    return ranker, model_inputs, captured


_MISSING = object()


def test_generate_multimodal_passes_repetition_penalty_explicit():
    ranker, model_inputs, captured = build_generate_capture_stub(
        multimodal=True,
        repetition_penalty=1.05,
    )

    ranker._generate(model_inputs, max_new_tokens=2)

    assert captured["repetition_penalty"] == 1.05


def test_generate_multimodal_skips_default_repetition_penalty():
    ranker, model_inputs, captured = build_generate_capture_stub(
        multimodal=True,
        repetition_penalty=1.0,
    )

    ranker._generate(model_inputs, max_new_tokens=2)

    assert "repetition_penalty" not in captured


def test_generate_multimodal_skips_missing_repetition_penalty():
    ranker, model_inputs, captured = build_generate_capture_stub(
        multimodal=True,
        repetition_penalty=_MISSING,
    )

    ranker._generate(model_inputs, max_new_tokens=2)

    assert "repetition_penalty" not in captured


def test_generate_multimodal_skips_none_repetition_penalty():
    ranker, model_inputs, captured = build_generate_capture_stub(
        multimodal=True,
        repetition_penalty=None,
    )

    ranker._generate(model_inputs, max_new_tokens=2)

    assert "repetition_penalty" not in captured


def test_generate_non_multimodal_repetition_penalty_unchanged():
    ranker, model_inputs, captured = build_generate_capture_stub(
        multimodal=False,
        repetition_penalty=1.05,
    )

    ranker._generate(model_inputs, max_new_tokens=2)

    assert captured["generation_config"].repetition_penalty == 1.05
    assert "repetition_penalty" not in captured


def test_maxcontext_should_fallback_to_bm25_helper():
    """Direct unit test for the BM25-fallback gate."""
    from llmrankers.setwise_extended import _maxcontext_should_fallback_to_bm25

    fallback_off = SimpleNamespace(_allow_parse_failure_bm25_fallback=False)
    fallback_on = SimpleNamespace(_allow_parse_failure_bm25_fallback=True)

    parse_err = ValueError("MaxContext single-label parse failed. Raw text: 'foo'")
    dual_err = ValueError("MaxContext dual-output parse failed. Raw text: 'foo'")
    dup_err = ValueError("Duplicate best/worst label '1' under strict mode")
    contract_err = ValueError(
        "MaxContext n_docs=2 bypass requires finite BM25 scores; got None for docid 'doc-7'."
    )

    # Off → never fallback.
    assert not _maxcontext_should_fallback_to_bm25(fallback_off, parse_err)
    assert not _maxcontext_should_fallback_to_bm25(fallback_off, dual_err)
    assert not _maxcontext_should_fallback_to_bm25(fallback_off, dup_err)
    assert not _maxcontext_should_fallback_to_bm25(fallback_off, contract_err)

    # On → fallback for the three parse-failure variants only.
    assert _maxcontext_should_fallback_to_bm25(fallback_on, parse_err)
    assert _maxcontext_should_fallback_to_bm25(fallback_on, dual_err)
    assert _maxcontext_should_fallback_to_bm25(fallback_on, dup_err)
    # Contract / pre-flight errors must NOT be swallowed even with the flag on.
    assert not _maxcontext_should_fallback_to_bm25(fallback_on, contract_err)


def build_setwise_tokenizer_stub(*, strict=False):
    ranker = object.__new__(SetwiseLlmRanker)
    ranker.tokenizer = DummyTokenizer()
    ranker.max_input_tokens = 5
    ranker._warned_input_truncation = False
    ranker.strict_no_truncation = strict
    ranker.device = "cpu"
    return ranker


def build_single_label_parser_stub(*, strict=False, label_scheme=None, characters=None):
    ranker = object.__new__(SetwiseLlmRanker)
    ranker.CHARACTERS = characters or [chr(ord("A") + i) for i in range(23)]
    ranker.strict_no_parse_fallback = strict
    if label_scheme is not None:
        ranker.label_scheme = label_scheme
    return ranker


def make_stub_causal_tokenizer():
    return SimpleNamespace(pad_token=None, eos_token="</s>", model_max_length=4096)


def instantiate_setwise_ranker(
    ranker_cls=SetwiseLlmRanker,
    *,
    model_type="qwen3",
    scoring="generation",
    character_scheme="letters_a_w",
):
    config = SimpleNamespace(model_type=model_type, n_positions=4096, max_position_embeddings=4096)
    with mock.patch("llmrankers.setwise.AutoConfig.from_pretrained", return_value=config), \
         mock.patch("llmrankers.setwise.AutoTokenizer.from_pretrained", return_value=make_stub_causal_tokenizer()), \
         mock.patch("llmrankers.setwise.AutoModelForCausalLM.from_pretrained", return_value=DummyCausalModel()):
        return ranker_cls(
            model_name_or_path="Qwen/Qwen3-4B",
            tokenizer_name_or_path=None,
            device="cpu",
            num_child=3,
            k=10,
            scoring=scoring,
            character_scheme=character_scheme,
            method="heapsort",
            num_permutation=1,
            cache_dir=None,
        )


def test_tokenize_invariants():
    long_prompt = "one two three four five six seven"

    strict_ranker = build_setwise_tokenizer_stub(strict=True)
    expect_raises(
        lambda: strict_ranker._tokenize_inputs(long_prompt),
        ValueError,
        "strict_no_truncation=True",
    )

    relaxed_ranker = build_setwise_tokenizer_stub(strict=False)
    capture = io.StringIO()
    with redirect_stdout(capture):
        batch = relaxed_ranker._tokenize_inputs(long_prompt)
    assert "Warning: prompt length" in capture.getvalue()
    assert batch.input_ids.shape[1] == relaxed_ranker.max_input_tokens


def test_default_false_flags_and_logging():
    assert getattr(object.__new__(SelectiveDualEndSetwiseLlmRanker), "strict_no_truncation") is False
    assert getattr(object.__new__(SelectiveDualEndSetwiseLlmRanker), "strict_no_parse_fallback") is False
    assert getattr(object.__new__(BiasAwareDualEndSetwiseLlmRanker), "strict_no_truncation") is False
    assert getattr(object.__new__(BiasAwareDualEndSetwiseLlmRanker), "strict_no_parse_fallback") is False
    assert getattr(object.__new__(SameCallRegularizedSetwiseLlmRanker), "strict_no_truncation") is False
    assert getattr(object.__new__(SameCallRegularizedSetwiseLlmRanker), "strict_no_parse_fallback") is False

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "comparisons.jsonl"
        ranker = object.__new__(SetwiseLlmRanker)
        ranker._comparison_log_path = str(log_path)
        ranker._current_qid = "q1"
        ranker._log_comparison("best", ["A", "B"], "A")
        entry = json.loads(log_path.read_text().strip())
        assert set(entry.keys()) == {"qid", "type", "positions", "selected"}


def test_position_bias_ordering_default_off_invariants():
    ranker = object.__new__(MaxContextDualEndSetwiseLlmRanker)
    ranker.shuffle = False
    ranker.reverse = False
    ranker._current_qid = "q1"
    docs = make_docs(5)
    before_state = random.getstate()
    with mock.patch(
        "llmrankers.setwise_extended.random.Random",
        side_effect=AssertionError("default-off path must not instantiate RNG"),
    ):
        assert ranker._apply_pool_ordering(docs) is docs
    assert random.getstate() == before_state
    assert not hasattr(ranker, "_round_counter")
    assert not hasattr(ranker, "_shuffle_seed")
    assert MAXCONTEXT_SHUFFLE_SEED == 929


def test_position_bias_ordering_enabled_paths_are_stateless():
    docs = make_docs(6)

    reverse_ranker = object.__new__(MaxContextTopDownSetwiseLlmRanker)
    reverse_ranker.shuffle = False
    reverse_ranker.reverse = True
    assert [doc.docid for doc in reverse_ranker._apply_pool_ordering(docs)] == [
        doc.docid for doc in reversed(docs)
    ]
    assert [doc.docid for doc in docs] == [f"d{i}" for i in range(1, 7)]

    shuffle_ranker = object.__new__(MaxContextBottomUpSetwiseLlmRanker)
    shuffle_ranker.shuffle = True
    shuffle_ranker.reverse = False
    shuffle_ranker._current_qid = "q42"
    first = shuffle_ranker._apply_pool_ordering(docs)
    second = shuffle_ranker._apply_pool_ordering(docs)
    assert [doc.docid for doc in first] == [doc.docid for doc in second]
    assert sorted(doc.docid for doc in first) == sorted(doc.docid for doc in docs)
    assert first is not docs
    assert not hasattr(shuffle_ranker, "_round_counter")
    assert not hasattr(shuffle_ranker, "_shuffle_seed")


def test_likelihood_substitute_and_fit_helper():
    ranker = build_dualend_stub(strict=False)
    ranker.scoring = "likelihood"
    ranker.config = SimpleNamespace(model_type="qwen3")
    assert ranker._parse_dual_output("###", 3) == ("A", "C")

    helper_ranker = object.__new__(SetwiseLlmRanker)
    helper_ranker.max_input_tokens = 20
    helper_ranker.tokenizer = DummyTokenizer()
    helper_ranker._format_passages = lambda docs: "\n".join(f"Passage {i+1}: {doc.text}" for i, doc in enumerate(docs))
    helper_ranker._build_chat_prompt = lambda messages: messages[0]["content"] + " chat"
    helper_ranker._is_supported_causal_model = lambda: True
    helper_ranker._uses_chat_template = lambda: True
    docs = [SimpleNamespace(text="alpha beta"), SimpleNamespace(text="gamma delta")]
    fits, rendered_length, budget = compute_max_fit_window(helper_ranker, "query terms", docs, reserved_output_tokens=4)
    assert isinstance(fits, bool)
    assert rendered_length > 0
    assert budget == 16


def test_single_label_prompt_guardrails():
    letters = [chr(ord("A") + i) for i in range(23)]
    docs3 = make_docs(3)

    helper_ranker = object.__new__(SetwiseLlmRanker)
    helper_ranker.CHARACTERS = letters
    most_guardrail = helper_ranker._format_label_guardrail(["A", "B", "C"], "most")
    assert "one of these labels: A, B, C" in most_guardrail
    assert "Do not explain" in most_guardrail
    assert "single closest" in most_guardrail

    least_guardrail = helper_ranker._format_label_guardrail(["AA", "AB", "AC"], "least")
    assert "one of these labels: AA, AB, AC" in least_guardrail
    assert "Do not explain" in least_guardrail
    assert "single least relevant" in least_guardrail

    default_ranker = object.__new__(SetwiseLlmRanker)
    default_ranker.CHARACTERS = letters
    default_best = default_ranker._build_best_prompt("query", docs3)
    assert "one of these labels: A, B, C" in default_best
    assert "single closest" in default_best

    explicit_ranker = object.__new__(SetwiseLlmRanker)
    explicit_ranker.CHARACTERS = letters
    explicit_ranker.label_scheme = "letters_a_w"
    assert explicit_ranker._build_best_prompt("query", docs3) == default_best

    bigram_ranker = object.__new__(SetwiseLlmRanker)
    bigram_ranker.CHARACTERS = [
        chr(ord("A") + left) + chr(ord("A") + right)
        for left in range(26)
        for right in range(26)
    ]
    bigram_ranker.label_scheme = "bigrams_aa_zz"
    expected_bigram_list = "one of these labels: " + ", ".join(bigram_ranker.CHARACTERS[:25])
    assert expected_bigram_list in bigram_ranker._build_best_prompt("query", make_docs(25))

    default_worst = default_ranker._build_worst_prompt("query", docs3)
    assert "one of these labels: A, B, C" in default_worst
    assert "single least relevant" in default_worst
    assert explicit_ranker._build_worst_prompt("query", docs3) == default_worst

    numeric_ranker = object.__new__(SetwiseLlmRanker)
    numeric_ranker.CHARACTERS = [str(i) for i in range(1, 11)]
    numeric_ranker.label_scheme = "numeric_1_based"
    docs10 = make_docs(10)
    expected_best = (
        'Given a query "query", which of the following passages is the most relevant one to the query?\n\n'
        + numeric_ranker._format_passages(docs10)
        + "\n\nOutput only the passage label of the most relevant passage:"
        + "\n\nReply with exactly one passage number from 1 to 10. "
        + "Do not explain. Do not output 0 or any number outside 1 to "
        + "10. If none of the passages are clearly relevant, "
        + "still pick the single closest one."
    )
    expected_worst = (
        'Given a query "query", which of the following passages is the least relevant one to the query?\n\n'
        + numeric_ranker._format_passages(docs10)
        + "\n\nOutput only the passage label of the least relevant passage:"
        + "\n\nReply with exactly one passage number from 1 to 10. "
        + "Do not explain. Do not output 0 or any number outside 1 to "
        + "10. If none of the passages are clearly irrelevant, "
        + "still pick the single least relevant one."
    )
    numeric_best = numeric_ranker._build_best_prompt("query", docs10)
    numeric_worst = numeric_ranker._build_worst_prompt("query", docs10)
    assert numeric_best == expected_best
    assert numeric_worst == expected_worst
    assert "one of these labels" not in numeric_best
    assert "one of these labels" not in numeric_worst

    parser_ranker = build_single_label_parser_stub(characters=letters)
    assert parser_ranker._parse_single_label("A", letters[:3]) == "A"


def test_dual_prompt_guardrails_and_fit_parity():
    letters = [chr(ord("A") + i) for i in range(23)]
    helper_ranker = object.__new__(SetwiseLlmRanker)
    helper_ranker.CHARACTERS = letters
    dual_guardrail = helper_ranker._format_dual_label_guardrail(["A", "B", "C"])
    assert "two distinct passage labels" in dual_guardrail
    assert "one of these labels: A, B, C" in dual_guardrail
    assert "Best: <label>, Worst: <label>" in dual_guardrail
    assert "Do not explain" in dual_guardrail

    letter_ranker, letter_capture = build_dual_prompt_capture_stub(
        "Best: A, Worst: B",
        characters=letters,
    )
    assert letter_ranker.compare_both("query", make_docs(5)) == ("A", "B")
    assert "two distinct passage labels" in letter_capture["prompt"]
    assert "one of these labels: A, B, C, D, E" in letter_capture["prompt"]

    numeric_chars = [str(i) for i in range(1, 11)]
    multimodal_ranker, multimodal_capture = build_dual_prompt_capture_stub(
        "Best: 1, Worst: 2",
        characters=numeric_chars,
        model_type="mistral3",
        label_scheme="numeric_1_based",
        is_multimodal=True,
        strict=True,
    )
    assert multimodal_ranker.compare_both("query", make_docs(10)) == ("1", "2")
    assert "Strict format on one line: Best: <number>, Worst: <number>" in multimodal_capture["prompt"]
    assert "two distinct passage labels, each one of these labels" not in multimodal_capture["prompt"]

    llama_ranker, llama_capture = build_dual_prompt_capture_stub(
        "Best: 1, Worst: 2",
        characters=numeric_chars,
        model_type="llama",
        label_scheme="numeric_1_based",
        strict=True,
    )
    assert llama_ranker.compare_both("query", make_docs(10)) == ("1", "2")
    assert "Do not output letters, 0, 'None', or any number outside 1 to 10" in llama_capture["prompt"]
    assert "Strict format on one line: Best: <number>, Worst: <number>" in llama_capture["prompt"]
    assert "two distinct passage labels, each one of these labels" not in llama_capture["prompt"]

    fit_letters = [chr(ord("A") + i) for i in range(25)]
    fit_ranker = object.__new__(DualEndSetwiseLlmRanker)
    fit_ranker.CHARACTERS = fit_letters
    fit_ranker.config = SimpleNamespace(model_type="qwen3")
    fit_ranker.tokenizer = DummyTokenizer()
    fit_ranker.max_input_tokens = 5000
    fit_ranker._is_multimodal = False
    captured_messages = []
    fit_ranker._build_chat_prompt = lambda messages: captured_messages.append(
        messages[0]["content"]
    ) or (messages[0]["content"] + "\nassistant:")
    docs25 = make_docs(25)
    expected_prompt = fit_ranker._build_dual_prompt_text("query", docs25)
    fits, rendered_length, budget = compute_max_fit_window(
        fit_ranker,
        "query",
        docs25,
    )
    assert fits is True
    assert rendered_length > 0
    assert budget == 4872
    assert captured_messages == [expected_prompt]
    assert "one of these labels: " + ", ".join(fit_letters) in captured_messages[0]

    numeric_fit_ranker = object.__new__(MaxContextDualEndSetwiseLlmRanker)
    numeric_fit_ranker.CHARACTERS = numeric_chars
    numeric_fit_ranker.label_scheme = "numeric_1_based"
    numeric_fit_ranker.config = SimpleNamespace(model_type="llama")
    numeric_fit_ranker.tokenizer = DummyTokenizer()
    numeric_fit_ranker.max_input_tokens = 5000
    numeric_fit_ranker._is_multimodal = False
    numeric_captured_messages = []
    numeric_fit_ranker._build_chat_prompt = lambda messages: numeric_captured_messages.append(
        messages[0]["content"]
    ) or (messages[0]["content"] + "\nassistant:")
    numeric_fits, numeric_rendered_length, numeric_budget = compute_max_fit_window(
        numeric_fit_ranker,
        "query",
        make_docs(10),
    )
    assert numeric_fits is True
    assert numeric_rendered_length > 0
    assert numeric_budget == 4872
    assert "Strict format on one line: Best: <number>, Worst: <number>" in numeric_captured_messages[0]
    assert "two distinct passage labels, each one of these labels" not in numeric_captured_messages[0]


def test_topdown_bigram_scheme_invariants():
    default_ranker = instantiate_setwise_ranker()
    assert not hasattr(default_ranker, "label_scheme")
    assert default_ranker.CHARACTERS == [chr(ord("A") + i) for i in range(23)]

    bigram_ranker = instantiate_setwise_ranker(character_scheme="bigrams_aa_zz")
    assert len(bigram_ranker.CHARACTERS) == 676
    assert bigram_ranker.CHARACTERS[0] == "AA"
    assert bigram_ranker.CHARACTERS[-1] == "ZZ"
    assert bigram_ranker.label_scheme == "bigrams_aa_zz"

    bottomup_bigram_ranker = instantiate_setwise_ranker(
        BottomUpSetwiseLlmRanker,
        character_scheme="bigrams_aa_zz",
    )
    assert len(bottomup_bigram_ranker.CHARACTERS) == 676
    assert bottomup_bigram_ranker.CHARACTERS[0] == "AA"
    assert bottomup_bigram_ranker.CHARACTERS[-1] == "ZZ"
    assert bottomup_bigram_ranker.label_scheme == "bigrams_aa_zz"

    bottomup_args = make_run_args(direction="bottomup", method="heapsort")
    bottomup_args.setwise.character_scheme = "bigrams_aa_zz"
    run_module.validate_character_scheme_args(bottomup_args)

    dualend_args = make_run_args(direction="dualend", method="bubblesort")
    dualend_args.setwise.character_scheme = "bigrams_aa_zz"
    expect_raises(
        lambda: run_module.validate_character_scheme_args(dualend_args),
        ValueError,
        "--direction topdown or bottomup",
    )

    with mock.patch(
        "llmrankers.setwise.AutoConfig.from_pretrained",
        return_value=SimpleNamespace(model_type="t5"),
    ):
        expect_raises(
            lambda: SetwiseLlmRanker(
                model_name_or_path="google/flan-t5-xl",
                tokenizer_name_or_path=None,
                device="cpu",
                num_child=3,
                k=10,
                scoring="generation",
                character_scheme="bigrams_aa_zz",
                method="heapsort",
                num_permutation=1,
                cache_dir=None,
            ),
            ValueError,
            "T5",
        )

    with mock.patch(
        "llmrankers.setwise.AutoConfig.from_pretrained",
        return_value=SimpleNamespace(model_type="qwen3"),
    ):
        expect_raises(
            lambda: SetwiseLlmRanker(
                model_name_or_path="Qwen/Qwen3-4B",
                tokenizer_name_or_path=None,
                device="cpu",
                num_child=3,
                k=10,
                scoring="likelihood",
                character_scheme="bigrams_aa_zz",
                method="heapsort",
                num_permutation=1,
                cache_dir=None,
            ),
            ValueError,
            "--scoring likelihood",
        )

    with mock.patch(
        "llmrankers.setwise.AutoConfig.from_pretrained",
        return_value=SimpleNamespace(model_type="qwen3"),
    ):
        expect_raises(
            lambda: DualEndSetwiseLlmRanker(
                model_name_or_path="Qwen/Qwen3-4B",
                tokenizer_name_or_path=None,
                device="cpu",
                num_child=3,
                k=10,
                scoring="generation",
                character_scheme="bigrams_aa_zz",
                method="heapsort",
                num_permutation=1,
                cache_dir=None,
            ),
            ValueError,
            "only supported on SetwiseLlmRanker",
        )

    with mock.patch(
        "llmrankers.setwise.AutoConfig.from_pretrained",
        return_value=SimpleNamespace(model_type="qwen3"),
    ):
        expect_raises(
            lambda: MaxContextBottomUpSetwiseLlmRanker(
                model_name_or_path="Qwen/Qwen3-4B",
                tokenizer_name_or_path=None,
                device="cpu",
                num_child=3,
                k=10,
                scoring="generation",
                character_scheme="bigrams_aa_zz",
                method="selection",
                num_permutation=1,
                cache_dir=None,
                pool_size=10,
            ),
            ValueError,
            "only supported on SetwiseLlmRanker",
        )

    legacy_parser_ranker = object.__new__(SetwiseLlmRanker)
    assert not hasattr(legacy_parser_ranker, "label_scheme")
    assert legacy_parser_ranker._parse_single_label("Best: [B]", ["A", "B", "C"]) == "B"

    parser_ranker = object.__new__(SetwiseLlmRanker)
    parser_ranker.label_scheme = "bigrams_aa_zz"
    valid = ["AA", "AB", "AC"]
    assert parser_ranker._parse_single_label("Best: AB", valid) == "AB"
    assert parser_ranker._parse_single_label("[AB]", valid) == "AB"
    assert parser_ranker._parse_single_label("AB", valid) == "AB"
    assert parser_ranker._parse_single_label("I recommend AB testing", valid) is None

    bottomup_valid = [
        chr(ord("A") + left) + chr(ord("A") + right)
        for left in range(26)
        for right in range(26)
    ]
    for raw, expected in (
        ("AD", "AD"),
        ("AG", "AG"),
        ("**AD**</s>", "AD"),
        ("D", "AD"),
    ):
        compare_ranker = build_standard_bottomup_compare_stub(
            [raw],
            characters=bottomup_valid,
            label_scheme="bigrams_aa_zz",
        )
        assert compare_ranker.compare_worst("query", make_docs(24)) == expected
        assert compare_ranker.total_retries == 0
        assert compare_ranker.total_out_of_window_label_fallback == 0


def test_checkpointed_short_pool_raw_capture_and_resume():
    """Zero-call queries remain auditable and resume preserves measured costs."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first_stage = root / "input.txt"
        first_stage.write_text("q1 Q0 d1 1 1.0 BM25\n")
        args = make_run_args(direction="maxcontext_topdown", hits=10, k=10, method="selection")
        args.run.ir_dataset_name = "test/dataset"
        args.run.run_path = str(first_stage)
        args.run.save_path = str(root / "output.txt")
        args.run.log_comparisons = str(root / "comparisons.jsonl")
        args.run.checkpoint_dir = str(root / "checkpoints")
        args.run.telemetry_path = str(root / "telemetry.jsonl")
        args.run.capture_raw_responses = True
        args.run.allow_parse_failure_bm25_fallback = False
        ranker = instantiate_maxcontext_variant(
            MaxContextTopDownSetwiseLlmRanker, SetwiseLlmRanker, pool_size=10
        )
        ranker.truncate = lambda text, length: text
        ranker._assert_maxcontext_fits = lambda *_args: (_ for _ in ()).throw(
            AssertionError("single-document queries must not use inference")
        )
        dataset = SimpleNamespace(
            queries_iter=lambda: [SimpleNamespace(query_id="q1", text="query")],
            docs_store=lambda: SimpleNamespace(get=lambda docid: SimpleNamespace(text="passage")),
        )
        with mock.patch.object(run_module.AutoConfig, "from_pretrained", return_value=SimpleNamespace(model_type="qwen3")), \
             mock.patch.object(run_module, "MaxContextTopDownSetwiseLlmRanker", return_value=ranker), \
             mock.patch.object(run_module.ir_datasets, "load", return_value=dataset, create=True), \
             mock.patch.object(run_module.time, "perf_counter", side_effect=[0.0, 2.0]), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            run_module.main(args)
        assert (root / "comparisons.jsonl").read_text() == ""
        original_output = (root / "output.txt").read_text()
        telemetry = json.loads((root / "telemetry.jsonl").read_text())
        assert telemetry["llm_calls"] == 0
        assert telemetry["wall_seconds"] == 2.0
        args.run.resume = True
        ranker.rerank = lambda *_args: (_ for _ in ()).throw(AssertionError("checkpoint was not resumed"))
        stdout = io.StringIO()
        with mock.patch.object(run_module.AutoConfig, "from_pretrained", return_value=SimpleNamespace(model_type="qwen3")), \
             mock.patch.object(run_module, "MaxContextTopDownSetwiseLlmRanker", return_value=ranker), \
             mock.patch.object(run_module.ir_datasets, "load", return_value=dataset, create=True), \
             redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            run_module.main(args)
        assert (root / "output.txt").read_text() == original_output
        assert len((root / "telemetry.jsonl").read_text().splitlines()) == 1
        assert "Avg wall-clock time per query: 2.0" in stdout.getvalue()
        # A checkpoint cannot substitute for missing raw-generation evidence.
        next((root / "checkpoints" / "queries").glob("*.comparisons.jsonl")).unlink()
        with mock.patch.object(run_module.AutoConfig, "from_pretrained", return_value=SimpleNamespace(model_type="qwen3")), \
             mock.patch.object(run_module, "MaxContextTopDownSetwiseLlmRanker", return_value=ranker), \
             mock.patch.object(run_module.ir_datasets, "load", return_value=dataset, create=True), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            expect_raises(lambda: run_module.main(args), FileNotFoundError, "comparisons.jsonl")


def main():
    test_checkpointed_short_pool_raw_capture_and_resume()
    test_dispatch_invariants()
    test_maxcontext_init_invariants()
    test_parse_failure_retry_budget_invariants()
    test_lenient_fallback_counter_invariants()
    test_maxcontext_dualend_byte_identity_snapshot()
    test_maxcontext_dualend_allows_query_pools_below_cap()
    test_generation_budget_tier()
    test_chat_template_kwargs_per_family()
    test_trust_remote_code_per_family()
    test_multimodal_predicate_and_adapter()
    test_maxcontext_multimodal_generation_gates()
    test_early_reject_unsupported_family()
    test_maxcontext_topdown_invariants()
    test_maxcontext_bottomup_invariants()
    test_parse_invariants()
    test_classify_numeric_noop_unit()
    test_parse_dual_output_strict_unchanged()
    test_single_label_parser_hardening()
    test_standard_out_of_window_label_noop()
    test_lenient_fallback_counter_behavior()
    test_nonmax_fallback_counter_reset_behavior()
    test_maxcontext_numeric_parse_fallback()
    test_maxcontext_compare_refusal_noop()
    test_maxcontext_compare_numeric_out_of_range_noop()
    test_maxcontext_dualend_compare_both_noop()
    test_maxcontext_post_exhaustion_numeric_out_of_range_noop()
    test_compare_both_duplicate_rewrite_guard()
    test_maxcontext_strict_parse_fallback_counter()
    test_maxcontext_parse_failure_retry_success()
    test_maxcontext_parse_failure_retry_exhausted_strict()
    test_maxcontext_parse_failure_retry_exhausted_lenient()
    test_maxcontext_parse_failure_retry_overhead_timing()
    test_maxcontext_compare_both_retry_success()
    test_maxcontext_compare_both_retry_exhausted()
    test_maxcontext_label_resolution_strict_fallback()
    test_maxcontext_strict_parse_fallback_rerank_reset()
    test_is_degenerate_repetition()
    test_maxcontext_degenerate_repetition_exhausted_at_three_noop()
    test_maxcontext_unparseable_after_exhaustion_noop()
    test_compare_lexical_refusal_does_not_extend()
    test_compare_both_duplicate_label_exhaustion_noop()
    test_generate_multimodal_passes_repetition_penalty_explicit()
    test_generate_multimodal_skips_default_repetition_penalty()
    test_generate_multimodal_skips_missing_repetition_penalty()
    test_generate_multimodal_skips_none_repetition_penalty()
    test_generate_non_multimodal_repetition_penalty_unchanged()
    test_maxcontext_should_fallback_to_bm25_helper()
    test_tokenize_invariants()
    test_default_false_flags_and_logging()
    test_position_bias_ordering_default_off_invariants()
    test_position_bias_ordering_enabled_paths_are_stateless()
    test_likelihood_substitute_and_fit_helper()
    test_single_label_prompt_guardrails()
    test_dual_prompt_guardrails_and_fit_parity()
    test_topdown_bigram_scheme_invariants()
    print("check_maxcontext_invariants.py: all checks passed")


if __name__ == "__main__":
    main()
