from typing import List, NamedTuple, Optional, Sequence
from .rankers import LlmRanker, SearchResult
import openai
import os
import time
import re
import math
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForImageTextToText,
    AutoProcessor,
)
from ._processor_adapter import ProcessorTokenizerAdapter
import torch
import copy
from collections import Counter
import tiktoken
import random

_DEBUG = os.environ.get("LLM_RANKER_DEBUG", "").lower() in ("1", "true", "yes")
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_IMPORT_ERROR = None
except ImportError as err:
    LLM = None
    SamplingParams = None
    LoRARequest = None
    VLLM_IMPORT_ERROR = err


import logging as _stdlib_logging


class _MaxLengthWarningFilter(_stdlib_logging.Filter):
    """Suppress the cosmetic `max_new_tokens vs max_length` warning emitted
    by transformers' generation pipeline.

    When a model's `generation_config` ships an explicit `max_length` (e.g.
    Mistral 3 = 262144, Qwen 3.5 multimodal also explicit), HF emits a
    warning whenever the user passes `max_new_tokens` — even though
    `max_new_tokens` correctly takes precedence and behavior is unchanged.
    The warning fires regardless of whether we pass our own
    `generation_config` or use kwargs-only, because HF deep-copies the
    model's pristine config and merges our kwargs into it before validating.

    The warning is emitted via `logger.warning` (Python logging), not
    `warnings.warn`, so `warnings.filterwarnings` does not catch it.

    This filter is a no-op for models whose config does not trigger the
    warning (e.g. Qwen 3, Llama 3.1 — `max_length=20` sentinel), so
    Qwen 3 byte-equality is preserved.
    """

    def filter(self, record: _stdlib_logging.LogRecord) -> bool:  # type: ignore[override]
        msg = record.getMessage()
        return not (
            "max_new_tokens" in msg
            and "max_length" in msg
            and "seem to have been set" in msg
        )


# Defensive coverage across transformers versions that have moved the warning
# emission site between submodules. Idempotent (addFilter on the same instance
# is safe across re-imports).
_MAX_LENGTH_WARNING_FILTER = _MaxLengthWarningFilter()
for _logger_name in (
    "transformers.generation.utils",
    "transformers.generation",
    "transformers",
):
    _stdlib_logging.getLogger(_logger_name).addFilter(_MAX_LENGTH_WARNING_FILTER)


random.seed(929)

MAX_PARSE_FAILURE_RETRIES = 3

CAUSAL_MODEL_TYPES = frozenset({
    "llama", "qwen2", "qwen3", "qwen3_moe",
    "mistral", "ministral",
})
MULTIMODAL_MODEL_TYPES = frozenset({"mistral3", "qwen3_5", "qwen3_5_moe"})
QWEN_MODEL_TYPES = frozenset({"qwen2", "qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_moe"})
TRUST_REMOTE_CODE_MODEL_TYPES = frozenset(QWEN_MODEL_TYPES)
THINKING_DISABLE_MODEL_TYPES = frozenset(QWEN_MODEL_TYPES)
THINKING_BUDGET_MODEL_TYPES = frozenset(QWEN_MODEL_TYPES)
# Verbose chat-tuned text-only models that produce more output than the
# 32/64 default budget can hold. Llama-3.x Instruct variants are known to
# emit chain-of-thought-style preludes before the actual answer, so the
# default budget truncates the response (e.g. mid-passage echo, missing
# `Worst:` portion). 256/512 is the same tier as Qwen-thinking and
# multimodal models. NOT applied to text-only `mistral` (Mistral 7B) or
# `ministral` (Ministral 8B) since those have no current EMNLP baseline.
VERBOSE_CHAT_MODEL_TYPES = frozenset({"llama"})

_RetryCounterSnapshot = NamedTuple(
    "_RetryCounterSnapshot",
    [
        ("total_compare", int),
        ("total_prompt_tokens", int),
        ("total_completion_tokens", int),
        ("total_parse_failure_strict", int),
    ],
)


def _is_multimodal_config(config) -> bool:
    return (
        getattr(config, "model_type", None) in MULTIMODAL_MODEL_TYPES
        and hasattr(config, "vision_config")
    )


def compute_max_fit_window(
    ranker: "SetwiseLlmRanker",
    query: str,
    docs: list,
    reserved_output_tokens: int = 128,
) -> tuple[bool, int, int]:
    """Render the full MaxContext prompt via the runtime prompt path.

    Returns:
        (fits, rendered_length, budget)
    """
    if ranker.max_input_tokens is None:
        raise ValueError("Cannot compute MaxContext fit window without max_input_tokens.")

    input_text = ranker._build_dual_prompt_text(query, docs)

    rendered_prompt = input_text
    if ranker._uses_chat_template():
        rendered_prompt = ranker._build_chat_prompt([{"role": "user", "content": input_text}])

    rendered_ids = ranker.tokenizer.encode(rendered_prompt, add_special_tokens=True)
    rendered_length = len(rendered_ids)
    budget = ranker.max_input_tokens - reserved_output_tokens
    return rendered_length <= budget, rendered_length, budget


class SetwiseLlmRanker(LlmRanker):
    CHARACTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
                  "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W"]  # "Passage X" and "Passage Y" will be tokenized into 3 tokens, so we dont use for now
    strict_no_truncation: bool = False

    def __init__(self,
                 model_name_or_path,
                 tokenizer_name_or_path,
                 device,
                 num_child=3,
                 k=10,
                 scoring='generation',
                 character_scheme: str = "letters_a_w",
                 method="heapsort",
                 num_permutation=1,
                 cache_dir=None):

        self.device = device
        self.num_child = num_child
        self.num_permutation = num_permutation
        self.k = k
        self.config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir,
                                                   trust_remote_code=True)
        self.scoring = scoring
        self._apply_character_scheme(character_scheme)
        self._warned_input_truncation = False
        self._load_model_and_tokenizer(model_name_or_path, tokenizer_name_or_path, device, cache_dir)
        if self._is_multimodal_model() and self.scoring == "likelihood":
            raise NotImplementedError(
                f"--scoring likelihood is not supported for multimodal model_type="
                f"{self.config.model_type!r}. Use --scoring generation."
            )

        self.method = method
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        self.total_retries = 0
        self.total_retry_overhead_seconds = 0.0
        self.total_parse_fallback = 0
        self.total_out_of_window_label_fallback = 0
        self.total_lexical_refusal_fallback = 0
        self.total_numeric_out_of_range_fallback = 0
        self.total_degenerate_repetition_fallback = 0
        self.total_unparseable_after_exhaustion_fallback = 0
        self.total_duplicate_label_fallback = 0
        self.total_lenient_fallback = 0
        self.total_strict_parse_fallback = 0
        self.max_input_tokens = self._resolve_max_input_tokens()

    def _parse_failure_retry_budget(self) -> int:
        explicit = getattr(self, "_parse_failure_max_retries", None)
        if explicit is not None:
            return explicit
        if getattr(self, "strict_no_parse_fallback", False):
            return MAX_PARSE_FAILURE_RETRIES
        return 0

    def _is_degenerate_repetition(self, raw: str) -> bool:
        """Heuristic: classify outputs that exhibit sticky token-level repetition.

        Triggers on either:
          (1) Last 300 characters of cleaned output have >=20 whitespace tokens
              but <=10 unique tokens (heavy repetition tail).
          (2) Any 6-token n-gram appears >=4 times in the cleaned output.

        Tokenization: simple whitespace split (str.split()) on the cleaned output.
        No lowercasing, no punctuation stripping.

        NOTE: this detector does NOT alter retry budgets. It is only a
        post-budget no-op classifier signal: immediately for non-strict rankers
        with budget 0, or after the MaxContext 3-retry budget exhausts.
        """
        cleaned = self._clean_generation_output(raw)
        tail = cleaned[-300:] if len(cleaned) > 300 else cleaned
        tail_tokens = tail.split()
        if len(tail_tokens) >= 20 and len(set(tail_tokens)) <= 10:
            return True
        tokens = cleaned.split()
        if len(tokens) >= 24:
            ngrams = Counter(tuple(tokens[i:i+6]) for i in range(len(tokens) - 5))
            if ngrams and ngrams.most_common(1)[0][1] >= 4:
                return True
        return False

    def _snapshot_retry_invisible_counters(self) -> _RetryCounterSnapshot:
        return _RetryCounterSnapshot(
            total_compare=getattr(self, "total_compare", 0),
            total_prompt_tokens=getattr(self, "total_prompt_tokens", 0),
            total_completion_tokens=getattr(self, "total_completion_tokens", 0),
            total_parse_failure_strict=getattr(self, "total_parse_failure_strict", 0),
        )

    def _restore_retry_invisible_counters(self, snapshot: _RetryCounterSnapshot) -> None:
        self.total_compare = snapshot.total_compare
        self.total_prompt_tokens = snapshot.total_prompt_tokens
        self.total_completion_tokens = snapshot.total_completion_tokens
        self.total_parse_failure_strict = snapshot.total_parse_failure_strict

    def _record_parse_failure_retry(self, start_time: float) -> None:
        self.total_retries = getattr(self, "total_retries", 0) + 1
        self.total_retry_overhead_seconds = (
            getattr(self, "total_retry_overhead_seconds", 0.0)
            + (time.perf_counter() - start_time)
        )

    def _score_order_fallback_active(self) -> bool:
        """Whether parse fallbacks should use first-stage scores, not shown order."""
        return (
            getattr(self, "_maxcontext_pool_size", None) is not None
            and (getattr(self, "shuffle", False) or getattr(self, "reverse", False))
        )

    def _score_order_original_position(self, doc, fallback_index: int) -> int:
        positions = getattr(self, "_maxcontext_original_positions", None)
        if isinstance(positions, dict):
            return positions.get(getattr(doc, "docid", None), fallback_index)
        return fallback_index

    def _score_order_fallback_index(self, docs: List, polarity: str, default_index: int) -> int:
        if not self._score_order_fallback_active() or not docs:
            return default_index

        scored = []
        for idx, doc in enumerate(docs):
            score = getattr(doc, "score", None)
            try:
                score = float(score)
            except (TypeError, ValueError):
                return default_index
            if not math.isfinite(score):
                return default_index
            orig_pos = self._score_order_original_position(doc, idx)
            scored.append((idx, score, orig_pos))

        if polarity == "best":
            return max(scored, key=lambda item: (item[1], -item[2]))[0]
        if polarity == "worst":
            return min(scored, key=lambda item: (item[1], -item[2]))[0]
        raise ValueError(f"Unknown score-order fallback polarity: {polarity!r}")

    def _score_order_fallback_label(self, docs: List, polarity: str, default_index: int) -> str:
        labels = self.CHARACTERS[:len(docs)]
        if not labels:
            return self.CHARACTERS[0]
        default_index = min(max(default_index, 0), len(labels) - 1)
        fallback_index = self._score_order_fallback_index(docs, polarity, default_index)
        fallback_index = min(max(fallback_index, 0), len(labels) - 1)
        return labels[fallback_index]

    def _score_order_fallback_dual_labels(self, docs: List) -> tuple[str, str]:
        if len(docs) < 2:
            return self.CHARACTERS[0], self.CHARACTERS[0]

        best = self._score_order_fallback_label(docs, "best", 0)
        worst = self._score_order_fallback_label(docs, "worst", len(docs) - 1)
        if best != worst:
            return best, worst

        labels = self.CHARACTERS[:len(docs)]
        for label in reversed(labels):
            if label != best:
                return best, label
        return best, worst

    def _score_order_sorted_docs(self, docs: List) -> List:
        """Return docs in descending first-stage score order when scores are usable."""
        decorated = []
        for idx, doc in enumerate(docs):
            score = getattr(doc, "score", None)
            try:
                score = float(score)
            except (TypeError, ValueError):
                return list(docs)
            if not math.isfinite(score):
                return list(docs)
            orig_pos = self._score_order_original_position(doc, idx)
            decorated.append((doc, score, orig_pos))
        decorated.sort(key=lambda item: (item[1], -item[2]), reverse=True)
        return [doc for doc, _, _ in decorated]

    def _load_model_and_tokenizer(self, model_name_or_path, tokenizer_name_or_path, device, cache_dir):
        if self.config.model_type == 't5':
            self._load_t5(model_name_or_path, tokenizer_name_or_path, device, cache_dir)
        elif _is_multimodal_config(self.config):
            self._load_multimodal(model_name_or_path, tokenizer_name_or_path, device, cache_dir)
        elif self._is_supported_causal_model():
            self._load_causal(model_name_or_path, tokenizer_name_or_path, device, cache_dir)
        else:
            raise NotImplementedError(f"Model type {self.config.model_type} is not supported yet for setwise:(")

    def _load_t5(self, model_name_or_path, tokenizer_name_or_path, device, cache_dir):
        self._is_multimodal = False
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name_or_path
                                                     if tokenizer_name_or_path is not None else
                                                     model_name_or_path,
                                                     cache_dir=cache_dir)
        self.llm = T5ForConditionalGeneration.from_pretrained(model_name_or_path,
                                                              device_map='auto',
                                                              torch_dtype=torch.float16 if device == 'cuda'
                                                              else torch.float32,
                                                              cache_dir=cache_dir)
        self.decoder_input_ids = self.tokenizer.encode("<pad> Passage",
                                                       return_tensors="pt",
                                                       add_special_tokens=False).to(self.device) if self.tokenizer else None
        self.dual_decoder_input_ids = self.tokenizer.encode("<pad> Best:",
                                                            return_tensors="pt",
                                                            add_special_tokens=False).to(self.device) if self.tokenizer else None

        self.target_token_ids = self.tokenizer.batch_encode_plus([f'<pad> Passage {self.CHARACTERS[i]}'
                                                                  for i in range(len(self.CHARACTERS))],
                                                                 return_tensors="pt",
                                                                 add_special_tokens=False,
                                                                 padding=True).input_ids[:, -1]

    def _load_causal(self, model_name_or_path, tokenizer_name_or_path, device, cache_dir):
        self._is_multimodal = False
        tokenizer_kwargs = {"cache_dir": cache_dir}
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": "auto" if device == "cuda" else torch.float32,
            "cache_dir": cache_dir,
        }
        if self.config.model_type in TRUST_REMOTE_CODE_MODEL_TYPES:
            tokenizer_kwargs["trust_remote_code"] = True
            model_kwargs["trust_remote_code"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
        if hasattr(self.tokenizer, "use_default_system_prompt"):
            self.tokenizer.use_default_system_prompt = False
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if not self._is_multimodal and 'vicuna' in model_name_or_path and 'v1.5' in model_name_or_path:
            self.tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ system_message }}{% endif %}{% if message['role'] == 'user' %}{{ ' USER: ' + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ ' ASSISTANT: ' + message['content'].strip() + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ ' ASSISTANT:' }}{% endif %}"
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        ).eval()

    def _load_multimodal(self, model_name_or_path, tokenizer_name_or_path, device, cache_dir):
        processor_kwargs = {"cache_dir": cache_dir, "trust_remote_code": True}
        self.processor = AutoProcessor.from_pretrained(
            tokenizer_name_or_path or model_name_or_path,
            **processor_kwargs,
        )
        self.tokenizer = ProcessorTokenizerAdapter(self.processor)
        if hasattr(self.tokenizer, "use_default_system_prompt"):
            self.tokenizer.use_default_system_prompt = False
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            device_map="auto",
            torch_dtype="auto" if device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).eval()
        self._is_multimodal = True

    def _apply_character_scheme(self, scheme: str) -> None:
        if scheme == "letters_a_w":
            return
        if scheme == "bigrams_aa_zz":
            if self.config.model_type == 't5':
                raise ValueError("bigrams_aa_zz not supported on T5 (target_token_ids path).")
            if self.scoring == "likelihood":
                raise ValueError(
                    "bigrams_aa_zz not supported with --scoring likelihood "
                    "(bigram tokenization is not guaranteed to be single-token)."
                )
            allowed_bigram_classes = {"SetwiseLlmRanker", "BottomUpSetwiseLlmRanker"}
            if type(self).__name__ not in allowed_bigram_classes:
                raise ValueError(
                    f"bigrams_aa_zz is only supported on SetwiseLlmRanker (TopDown) "
                    f"and BottomUpSetwiseLlmRanker (BottomUp); "
                    f"got subclass {type(self).__name__}."
                )
            self.CHARACTERS = [
                chr(ord("A") + left) + chr(ord("A") + right)
                for left in range(26)
                for right in range(26)
            ]
            self.label_scheme = "bigrams_aa_zz"
            return
        raise ValueError(f"Unknown character_scheme={scheme!r}")

    def _is_supported_causal_model(self):
        return self.config.model_type in CAUSAL_MODEL_TYPES

    def _is_multimodal_model(self):
        return getattr(self, "_is_multimodal", False)

    def _uses_chat_template(self):
        return self._is_supported_causal_model() or self._is_multimodal_model()

    def _uses_causal_style_generation(self):
        return self._is_supported_causal_model() or self._is_multimodal_model()

    def _log_comparison(self, comp_type: str, positions: list, selected: str,
                        docs: list = None,
                        parse_status: str = None,
                        parse_fallback_reason: str = None,
                        raw_output: str = None):
        """Log a comparison for position bias analysis."""
        log_path = getattr(self, '_comparison_log_path', None)
        if not log_path:
            return
        import json as _json
        entry = {
            "qid": getattr(self, '_current_qid', None),
            "type": comp_type,
            "positions": positions,
            "selected": selected,
        }
        if docs is not None:
            entry["docids"] = [d.docid for d in docs]
        label_scheme = getattr(self, "label_scheme", None)
        if label_scheme:
            entry["label_scheme"] = label_scheme
        if parse_status is not None:
            entry["parse_status"] = parse_status
        if parse_fallback_reason is not None:
            entry["parse_fallback_reason"] = parse_fallback_reason
        if raw_output is not None:
            entry["raw_output"] = raw_output
        with open(log_path, 'a') as f:
            f.write(_json.dumps(entry) + "\n")

    def _chat_template_kwargs(self):
        if self.config.model_type in THINKING_DISABLE_MODEL_TYPES:
            return {"enable_thinking": False}
        return {}

    def _generation_budget(self, mode: str) -> int:
        """Return max_new_tokens for single-label or dual-label generation."""
        if self.config.model_type == "t5":
            return 2
        if self.config.model_type in THINKING_BUDGET_MODEL_TYPES:
            return 256 if mode == "single" else 512
        if self.config.model_type in MULTIMODAL_MODEL_TYPES:
            return 256 if mode == "single" else 512
        if self.config.model_type in VERBOSE_CHAT_MODEL_TYPES:
            return 256 if mode == "single" else 512
        return 32 if mode == "single" else 64

    def _format_passages(self, docs: Sequence[SearchResult]) -> str:
        return "\n\n".join(
            [f'Passage {self.CHARACTERS[i]}: "{doc.text}"' for i, doc in enumerate(docs)]
        )

    def _format_label_guardrail(
        self,
        valid_chars: Sequence[str],
        polarity: str,
    ) -> str:
        label_list = ", ".join(valid_chars)
        descriptor = "relevant" if polarity == "most" else "irrelevant"
        fallback = "single closest" if polarity == "most" else "single least relevant"
        return (
            f"\n\nReply with exactly one passage label, one of these labels: {label_list}. "
            f"Do not explain. Do not output any other label. "
            f"If none of the passages are clearly {descriptor}, "
            f"still pick the {fallback} one."
        )

    def _format_dual_label_guardrail(
        self,
        valid_chars: Sequence[str],
    ) -> str:
        label_list = ", ".join(valid_chars)
        return (
            f"\n\nReply with exactly two distinct passage labels, "
            f"each one of these labels: {label_list}. "
            f"Do not explain. "
            f"Strict format on one line: Best: <label>, Worst: <label>"
        )

    def _build_dual_prompt_text(
        self,
        query: str,
        docs: Sequence[SearchResult],
    ) -> str:
        passages = self._format_passages(docs)
        base_prompt = (
            f'Given a query "{query}", which of the following passages is the most relevant '
            f'and which is the least relevant to the query?\n\n'
            + passages
        )
        is_numeric = getattr(self, "label_scheme", None) == "numeric_1_based"
        model_type = getattr(getattr(self, "config", None), "model_type", None)

        if self._is_multimodal_model() and is_numeric:
            return base_prompt + (
                f"\n\nReply with exactly two distinct passage numbers between 1 and {len(docs)}. "
                f"Do not output 0, 'None', or any number outside the range. "
                f"Pick the closest passages even if none are clearly relevant. "
                f"Strict format on one line: Best: <number>, Worst: <number>"
            )
        if is_numeric and model_type == "llama":
            return base_prompt + (
                f"\n\nReply with exactly two distinct passage numbers between 1 and {len(docs)}. "
                f"Do not output letters, 0, 'None', or any number outside 1 to {len(docs)}. "
                f"Pick the closest passages even if none are clearly relevant. "
                f"Strict format on one line: Best: <number>, Worst: <number>"
            )
        return base_prompt + self._format_dual_label_guardrail(
            self.CHARACTERS[:len(docs)]
        )

    def _build_best_prompt(self, query: str, docs: Sequence[SearchResult]) -> str:
        passages = self._format_passages(docs)
        prompt = (
            f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n'
            + passages
            + "\n\nOutput only the passage label of the most relevant passage:"
        )
        label_scheme = getattr(self, "label_scheme", None)
        if label_scheme == "numeric_1_based":
            prompt += (
                f"\n\nReply with exactly one passage number from 1 to {len(docs)}. "
                "Do not explain. Do not output 0 or any number outside 1 to "
                f"{len(docs)}. If none of the passages are clearly relevant, "
                "still pick the single closest one."
            )
        elif label_scheme in (None, "letters_a_w", "bigrams_aa_zz"):
            prompt += self._format_label_guardrail(
                self.CHARACTERS[:len(docs)],
                polarity="most",
            )
        return prompt

    def _build_worst_prompt(self, query: str, docs: Sequence[SearchResult]) -> str:
        passages = self._format_passages(docs)
        prompt = (
            f'Given a query "{query}", which of the following passages is the least relevant one to the query?\n\n'
            + passages
            + "\n\nOutput only the passage label of the least relevant passage:"
        )
        label_scheme = getattr(self, "label_scheme", None)
        if label_scheme == "numeric_1_based":
            prompt += (
                f"\n\nReply with exactly one passage number from 1 to {len(docs)}. "
                "Do not explain. Do not output 0 or any number outside 1 to "
                f"{len(docs)}. If none of the passages are clearly irrelevant, "
                "still pick the single least relevant one."
            )
        elif label_scheme in (None, "letters_a_w", "bigrams_aa_zz"):
            prompt += self._format_label_guardrail(
                self.CHARACTERS[:len(docs)],
                polarity="least",
            )
        return prompt

    def _build_chat_prompt(self, messages):
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self._chat_template_kwargs(),
        )

    def _resolve_max_input_tokens(self):
        candidates = []
        model_max_length = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(model_max_length, int) and 0 < model_max_length < 100000:
            candidates.append(model_max_length)
        for attr in ("n_positions", "max_position_embeddings"):
            value = getattr(self.config, attr, None)
            if isinstance(value, int) and value > 0:
                candidates.append(value)
        text_config = getattr(self.config, "text_config", None)
        if text_config is not None:
            for attr in ("n_positions", "max_position_embeddings"):
                value = (
                    text_config.get(attr)
                    if isinstance(text_config, dict)
                    else getattr(text_config, attr, None)
                )
                if isinstance(value, int) and value > 0:
                    candidates.append(value)
        return min(candidates) if candidates else None

    def _raw_tokenizer_kwargs(self, padding=False, add_special_tokens=True):
        kwargs = {
            "add_special_tokens": add_special_tokens,
            "truncation": self.max_input_tokens is not None,
        }
        if self.max_input_tokens is not None:
            kwargs["max_length"] = self.max_input_tokens
        if padding:
            kwargs["padding"] = True
        return kwargs

    @staticmethod
    def _longest_common_prefix_length(lhs: Sequence[int], rhs: Sequence[int]) -> int:
        prefix_len = 0
        for left_token, right_token in zip(lhs, rhs):
            if left_token != right_token:
                break
            prefix_len += 1
        return prefix_len

    def _tokenize_inputs(self, inputs, padding=False):
        raw = self.tokenizer(inputs, add_special_tokens=True, return_attention_mask=True)
        raw_ids = raw["input_ids"]
        if raw_ids and isinstance(raw_ids[0], int):
            lengths = [len(raw_ids)]
        else:
            lengths = [len(ids) for ids in raw_ids]

        if (
            self.max_input_tokens is not None
            and lengths
            and max(lengths) > self.max_input_tokens
        ):
            if self.strict_no_truncation:
                raise ValueError(
                    f"Prompt length {max(lengths)} exceeds model input limit "
                    f"{self.max_input_tokens} and strict_no_truncation=True."
                )
            if not self._warned_input_truncation:
                print(
                    f"Warning: prompt length {max(lengths)} exceeds model limit {self.max_input_tokens}; "
                    "truncating the encoded prompt. Lower --passage_length or --query_length for cleaner comparisons."
                )
                self._warned_input_truncation = True

        kwargs = {
            "return_tensors": "pt",
            "return_attention_mask": True,
            "truncation": self.max_input_tokens is not None,
        }
        if self.max_input_tokens is not None:
            kwargs["max_length"] = self.max_input_tokens
        if padding:
            kwargs["padding"] = True
        return self.tokenizer(inputs, **kwargs).to(self.device)

    def _score_causal_label_candidates(self, input_text: str, n_docs: int) -> torch.Tensor:
        conversation = [{"role": "user", "content": input_text}]
        prompt = self._build_chat_prompt(conversation)

        prompt_inputs = self._tokenize_inputs(prompt)
        self.total_prompt_tokens += prompt_inputs.input_ids.shape[1]

        continuations = [f" Passage {label}" for label in self.CHARACTERS[:n_docs]]
        full_texts = [prompt + continuation for continuation in continuations]

        raw_kwargs = self._raw_tokenizer_kwargs()
        prompt_ids = self.tokenizer(prompt, **raw_kwargs)["input_ids"]
        full_ids = self.tokenizer(full_texts, **raw_kwargs)["input_ids"]

        prefix_lengths = [
            self._longest_common_prefix_length(prompt_ids, candidate_ids)
            for candidate_ids in full_ids
        ]
        continuation_lengths = [
            len(candidate_ids) - prefix_len
            for candidate_ids, prefix_len in zip(full_ids, prefix_lengths)
        ]

        if any(length <= 0 for length in continuation_lengths):
            raise RuntimeError(
                "Causal likelihood prompt leaves no room for the label continuation. "
                "Lower --passage_length or --query_length."
            )

        model_inputs = self._tokenize_inputs(full_texts, padding=True)
        full_lengths = [len(candidate_ids) for candidate_ids in full_ids]
        padding_side = getattr(self.tokenizer, "padding_side", "right")

        with torch.no_grad():
            logits = self.llm(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
            ).logits
            shifted_log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            shifted_targets = model_inputs.input_ids[:, 1:]
            token_log_probs = shifted_log_probs.gather(
                2, shifted_targets.unsqueeze(-1)
            ).squeeze(-1)

            scores = []
            for row, (prefix_len, continuation_len, full_length) in enumerate(
                zip(prefix_lengths, continuation_lengths, full_lengths)
            ):
                pad_offset = (
                    model_inputs.input_ids.shape[1] - full_length
                    if padding_side == "left"
                    else 0
                )
                start = pad_offset + prefix_len - 1
                end = start + continuation_len
                scores.append(token_log_probs[row, start:end].sum())

        return torch.stack(scores)

    def _score_label_candidates(self, input_text: str, n_docs: int) -> torch.Tensor:
        if self.config.model_type == 't5':
            inputs = self._tokenize_inputs(input_text)
            self.total_prompt_tokens += inputs.input_ids.shape[1]
            with torch.no_grad():
                logits = self.llm(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    decoder_input_ids=self.decoder_input_ids,
                ).logits[0][-1]
                distributions = torch.softmax(logits, dim=0)
                return distributions[self.target_token_ids[:n_docs]]

        if self._is_supported_causal_model():
            # Causal-model label scoring uses short teacher-forced answer strings
            # instead of assuming A/B/C is a single tokenizer token.
            return self._score_causal_label_candidates(input_text, n_docs)
        if self._is_multimodal_model():
            raise NotImplementedError(
                f"--scoring likelihood is not supported for multimodal model_type="
                f"{self.config.model_type!r}. Use --scoring generation."
            )

        raise NotImplementedError(
            f"Likelihood scoring is not implemented for model type {self.config.model_type}."
        )

    def _generate(self, model_inputs, max_new_tokens, decoder_input_ids=None):
        kwargs = {
            "input_ids": model_inputs.input_ids,
            "attention_mask": model_inputs.attention_mask,
            "max_new_tokens": max_new_tokens,
        }
        if decoder_input_ids is not None:
            kwargs["decoder_input_ids"] = decoder_input_ids
        if self._uses_causal_style_generation():
            if self._is_multimodal_model():
                # Multimodal models (e.g. Mistral 3) ship a 256K `max_length` in
                # their generation_config. If we deepcopy the config and pass it
                # back, HF interprets it as a user-override and warns about the
                # `max_new_tokens` vs `max_length` conflict. Instead, use
                # kwargs-only here: HF treats the model's pristine config as
                # "default" and `max_new_tokens` takes precedence cleanly.
                kwargs["do_sample"] = False
                for attr in (
                    "temperature",
                    "top_k",
                    "top_p",
                    "min_p",
                    "typical_p",
                    "epsilon_cutoff",
                    "eta_cutoff",
                ):
                    kwargs[attr] = None
                # Cosmetic (2026-05-08): make the model-shipped repetition_penalty explicit
                # in the kwargs-only path. HF already inherits this from
                # self.generation_config via its kwargs-merge logic, so this is a no-op for
                # current HF versions, but keeps the contract auditable.
                gen_config = getattr(self.llm, "generation_config", None)
                if gen_config is not None:
                    rep_pen = getattr(gen_config, "repetition_penalty", None)
                    if rep_pen is not None and rep_pen != 1.0:
                        kwargs["repetition_penalty"] = rep_pen
                if self.tokenizer.pad_token_id is not None:
                    kwargs["pad_token_id"] = self.tokenizer.pad_token_id
                if self.tokenizer.eos_token_id is not None:
                    kwargs["eos_token_id"] = self.tokenizer.eos_token_id
            else:
                generation_config = copy.deepcopy(getattr(self.llm, "generation_config", None))
                if generation_config is not None:
                    generation_config.do_sample = False
                    for attr in (
                        "temperature",
                        "top_k",
                        "top_p",
                        "min_p",
                        "typical_p",
                        "epsilon_cutoff",
                        "eta_cutoff",
                    ):
                        if hasattr(generation_config, attr):
                            setattr(generation_config, attr, None)
                    if self.tokenizer.pad_token_id is not None:
                        generation_config.pad_token_id = self.tokenizer.pad_token_id
                    if self.tokenizer.eos_token_id is not None:
                        generation_config.eos_token_id = self.tokenizer.eos_token_id
                    kwargs["generation_config"] = generation_config
                else:
                    kwargs["do_sample"] = False
                    if self.tokenizer.pad_token_id is not None:
                        kwargs["pad_token_id"] = self.tokenizer.pad_token_id
                    if self.tokenizer.eos_token_id is not None:
                        kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        return self.llm.generate(**kwargs)

    def _clean_generation_output(self, output: str) -> str:
        stripped = output.strip()

        # --- Qwen-family thinking tokens (<think>...</think>) ---
        # Strip complete blocks first, then any truncated/unclosed block. This is
        # a no-op for non-Qwen models unless they happen to emit the same tags.
        cleaned = re.sub(r"(?is)<think>.*?</think>\s*", "", stripped)
        cleaned = re.sub(r"(?is)<think>.*", "", cleaned)

        # --- Tokenizer-declared special tokens ---
        # Prefer exact token stripping when the tokenizer exposes its own special
        # token surface. This catches model-family tokens without relying on a
        # growing set of ad hoc regexes.
        tokenizer = getattr(self, "tokenizer", None)
        special_tokens = getattr(tokenizer, "all_special_tokens", None)
        if special_tokens:
            escaped_tokens = [
                re.escape(token)
                for token in sorted(set(special_tokens), key=len, reverse=True)
                if isinstance(token, str) and token
            ]
            if escaped_tokens:
                cleaned = re.sub("|".join(escaped_tokens), " ", cleaned)

        # --- Pipe-delimited special tokens (Qwen / LLaMA / general) ---
        # Matches <|im_start|>, <|im_end|>, <|endoftext|>, <|end|>, etc.
        cleaned = re.sub(r"<\|[^|]*\|>", " ", cleaned)

        # --- Bracket-style chat / multimodal tokens ---
        # Mistral-family templates and multimodal processors can surface these in
        # decoded text when we intentionally keep special tokens for parser cleanup.
        cleaned = re.sub(
            r"\[(?:/?INST|/?SYS|/?USER|/?ASSISTANT|IMG(?:_END)?|/?IMAGE)\]",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"</?(?:image|img|video|audio)\s*>", " ", cleaned, flags=re.IGNORECASE)

        # --- T5 / Flan-T5 special tokens ---
        cleaned = re.sub(r"</s>", " ", cleaned)
        cleaned = re.sub(r"<pad>", " ", cleaned)
        cleaned = re.sub(r"<unk>", " ", cleaned)
        cleaned = re.sub(r"<extra_id_\d+>", " ", cleaned)

        # --- General XML-style tags left over from any tokenizer ---
        # Strip isolated angle-bracket tokens like <s>, </s>, <bos>, <eos>, etc.
        cleaned = re.sub(r"</?(?:s|bos|eos|sep|cls|mask|pad)\s*>", " ", cleaned)

        # --- Paired markdown emphasis markers (LLM chat models often format
        # answers with markdown bold, e.g. "Best: **Passage 3**"). Only paired
        # markers are stripped; isolated `*` or `_` in passage content survive.
        cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"__(.+?)__", r"\1", cleaned)

        cleaned = cleaned.strip()
        return cleaned or stripped

    def _label_at_1_based_position(
        self,
        position: int,
        valid_chars: Sequence[str],
    ) -> Optional[str]:
        idx = position - 1
        if 0 <= idx < len(valid_chars):
            return valid_chars[idx]
        return None

    @staticmethod
    def _extract_unambiguous_letter_position(cleaned: str) -> Optional[int]:
        match = re.fullmatch(
            r"\s*(?:PASSAGE\s*)?\[?([A-Z])\]?\s*(?:[.,;:!?])?\s*",
            cleaned.upper(),
        )
        if match is None:
            return None
        return ord(match.group(1)) - ord("A") + 1

    @staticmethod
    def _extract_leading_number_position(cleaned: str) -> Optional[int]:
        match = re.match(r"^\s*(\d+)(?:\s|[.,;:!?]|$)", cleaned)
        if match is None:
            return None
        return int(match.group(1))

    @staticmethod
    def _extract_unambiguous_bigram_position(cleaned: str) -> Optional[int]:
        match = re.fullmatch(
            r"\s*(?:PASSAGE\s*)?\[?([A-Z]{2})\]?\s*(?:[.,;:!?])?\s*",
            cleaned.upper(),
        )
        if match is None:
            return None
        label = match.group(1)
        return (ord(label[0]) - ord("A")) * 26 + (ord(label[1]) - ord("A")) + 1

    def _parse_cross_scheme_single_label(
        self,
        cleaned: str,
        valid_chars: Sequence[str],
        *,
        is_bigram_scheme: bool,
        is_numeric_scheme: bool,
    ) -> Optional[str]:
        if is_bigram_scheme or is_numeric_scheme:
            letter_position = self._extract_unambiguous_letter_position(cleaned)
            if letter_position is not None:
                return self._label_at_1_based_position(letter_position, valid_chars)

        if not is_numeric_scheme:
            number_position = self._extract_leading_number_position(cleaned)
            if number_position is not None:
                return self._label_at_1_based_position(number_position, valid_chars)

        return None

    def _classify_out_of_window_label_noop(
        self,
        raw: str,
        valid_chars: Sequence[str],
    ) -> Optional[str]:
        if getattr(self, "label_scheme", None) == "numeric_1_based":
            return None

        cleaned = self._clean_generation_output(raw)
        is_bigram_scheme = getattr(self, "label_scheme", None) == "bigrams_aa_zz"
        positions = []
        if is_bigram_scheme:
            bigram_position = self._extract_unambiguous_bigram_position(cleaned)
            if bigram_position is not None:
                positions.append(bigram_position)
        letter_position = self._extract_unambiguous_letter_position(cleaned)
        if letter_position is not None:
            positions.append(letter_position)
        number_position = self._extract_leading_number_position(cleaned)
        if number_position is not None:
            positions.append(number_position)

        for position in positions:
            if self._label_at_1_based_position(position, valid_chars) is None:
                return "out_of_window_label"
        return None

    NUMERIC_REFUSAL_REGEX = (
        r"^\s*(none|no\s+passages?|neither|i\s+cannot|cannot\s+determine|"
        r"cannot\s+pick|cannot\s+decide)\b|"
        r"\bnone\s+of\s+the\s+(passages?|above)\b|"
        r"\bno\s+passages?\s+(is|are)\s+relevant\b|"
        r"\b(correct\s+answer|the\s+answer)\s+is\s*:?\s*none\b|"
        r"\bthere\s+is\s+no\s+(least|most)\s+relevant\b|"
        r"\bif\s+there\s+was\s+a\s+passage\b"
    )
    _NUMERIC_ONLY_REGEX = re.compile(r"^\s*(-?\d+)\s*$")

    def _is_numeric_refusal_output(self, raw: str) -> bool:
        cleaned = self._clean_generation_output(raw)
        return re.search(self.NUMERIC_REFUSAL_REGEX, cleaned, flags=re.IGNORECASE) is not None

    def _classify_numeric_noop(self, raw: str, n_docs: int) -> Optional[str]:
        """Return the recognized numeric-scheme no-op reason, or None."""
        if getattr(self, "label_scheme", None) != "numeric_1_based":
            return None
        cleaned = self._clean_generation_output(raw)
        if re.search(self.NUMERIC_REFUSAL_REGEX, cleaned, flags=re.IGNORECASE):
            return "lexical_refusal"
        match = self._NUMERIC_ONLY_REGEX.match(cleaned)
        if match:
            try:
                value = int(match.group(1))
            except ValueError:
                return None
            if value < 1 or value > n_docs:
                return "numeric_out_of_range"
        return None

    def _classify_post_exhaustion_noop(
        self,
        raw: str,
        n_docs: int,
        duplicate_label: Optional[str] = None,
    ) -> Optional[str]:
        """Return numeric MaxContext no-op reasons after retry exhaustion."""
        if getattr(self, "label_scheme", None) != "numeric_1_based":
            return None
        if duplicate_label is not None:
            return "duplicate_label"
        cleaned = self._clean_generation_output(raw)
        structured_nums = re.findall(
            r"\b(?:BEST|WORST)\b[:\s]*(?:PASSAGE\s*)?(-?\d+)",
            cleaned,
            flags=re.IGNORECASE,
        )
        for raw_num in structured_nums:
            try:
                value = int(raw_num)
            except ValueError:
                continue
            if value < 1 or value > n_docs:
                return "numeric_out_of_range"
        if self._is_degenerate_repetition(raw):
            return "degenerate_repetition"
        if len(cleaned.split()) >= 30:
            return "unparseable_after_exhaustion"
        return None

    def _parse_single_label(self, output: str, valid_chars: Sequence[str]) -> Optional[str]:
        cleaned = self._clean_generation_output(output)
        output_upper = cleaned.upper()
        valid = set(valid_chars)
        is_bigram_scheme = getattr(self, "label_scheme", None) == "bigrams_aa_zz"
        is_numeric_scheme = getattr(self, "label_scheme", None) == "numeric_1_based"
        refusal_regex = (
            r"(?i)\b(none of the|no passage|not relevant|none are|cannot determine|neither|"
            r"none|i cannot|no least relevant|no most relevant|both are equally relevant|"
            r"all are equally relevant|cannot pick|cannot decide|not provided|not shown|"
            r"if there was a passage|assuming passage)\b"
        )

        if is_bigram_scheme:
            patterns = (
                r"(?:BEST|WORST|MOST\s+RELEVANT|LEAST\s+RELEVANT|ANSWER|OUTPUT)\s*[:\-\s]*(?:PASSAGE\s*)?\[?([A-Z]{2})\]?",
                r"PASSAGE\s*\[?([A-Z]{2})\]?",
            )
        else:
            patterns = (
                r"(?:BEST|WORST|MOST\s+RELEVANT|LEAST\s+RELEVANT|ANSWER|OUTPUT)\s*[:\-\s]*(?:PASSAGE\s*)?\[?([A-W])\]?",
                r"PASSAGE\s*\[?([A-W])\]?",
            )

        for pattern in patterns:
            for match in re.findall(pattern, output_upper):
                if match in valid:
                    return match

        # Match standalone letter or bracketed letter like [A]
        if is_bigram_scheme:
            standalone_pattern = r"(?:\[([A-Z]{2})\]|\b([A-Z]{2})\b)"
            standalone_match = re.fullmatch(standalone_pattern, output_upper)
            if standalone_match:
                char = standalone_match.group(1) or standalone_match.group(2)
                if char in valid:
                    return char
        else:
            for match in re.findall(r"(?:\[([A-W])\]|\b([A-W])\b)", output_upper):
                char = match[0] or match[1]
                if char in valid:
                    return char

        if not is_bigram_scheme and not is_numeric_scheme:
            all_found = [char for char in output_upper if char in valid]
            if all_found and len(set(all_found)) == 1:
                return all_found[0]

        if is_numeric_scheme:
            numeric_patterns = (
                r"(?:BEST|WORST|MOST\s+RELEVANT|LEAST\s+RELEVANT|ANSWER|OUTPUT)\s*[:\-\s]*(?:PASSAGE\s*)?(\d+)",
                r"^\s*(\d+)(?:\s|[.,;:!?]|$)",
                r"(?:MOST\s+RELEVANT|LEAST\s+RELEVANT|BEST|WORST|CLOSEST(?:\s+MATCH)?)[^.\n]{0,40}?PASSAGE\s+(\d+)",
                r"PASSAGE\s+(\d+)\s+(?:IS|WAS)\s+(?:THE\s+)?(?:MOST(?:\s+RELEVANT)?|LEAST(?:\s+RELEVANT)?|BEST|WORST|CLOSEST(?:\s+MATCH)?)",
                # B.1 (parse-failure hotfix 2026-05-08): allow a closing paren or
                # comma (and surrounding whitespace) between the digit and the
                # IS/WAS trigger. Qwen3.5-9B at N=50/100 frequently emits
                # "This passage (Passage 21) is the least relevant because...",
                # which the previous pattern missed because it required digit+IS
                # adjacency. New branch — old patterns try first, so Qwen3
                # byte-equality is preserved for outputs the old patterns
                # already accepted.
                r"PASSAGE\s+(\d+)\s*[)\],]\s*(?:IS|WAS)\s+(?:THE\s+)?(?:MOST(?:\s+RELEVANT)?|LEAST(?:\s+RELEVANT)?|BEST|WORST|CLOSEST(?:\s+MATCH)?)",
            )
            for pattern in numeric_patterns:
                for match in re.findall(pattern, cleaned, flags=re.IGNORECASE):
                    idx = int(match) - 1
                    if 0 <= idx < len(valid_chars):
                        return valid_chars[idx]

        if is_numeric_scheme:
            strict = getattr(self, "strict_no_parse_fallback", False)
            if self._is_numeric_refusal_output(output):
                # Soft-refusal recovery: if the output contains both a
                # refusal phrase AND a clear choice indicator, try to extract
                # the actual chosen number from the trailing portion before
                # falling back to passage-1. Verbose chat models (Llama-3.1
                # Instruct, Mistral-3) often hedge with "None of the provided
                # passages directly address..." but then provide an answer
                # with "...if forced to choose: 16" — the strict patterns
                # above don't catch this format, but the model IS picking.
                # Qwen 3 byte-equality preserved: Qwen 3 outputs are clean
                # and don't enter this branch.
                choice_indicator_re = re.compile(
                    r"\b(choose|pick|select|forced\s+to|i\s+would|"
                    r"my\s+(?:answer|choice|pick)|the\s+answer\s+is|"
                    r"closest\s+is|would\s+be|is\s+the\s+(?:most|least)|"
                    r"final\s+answer|going\s+with|"
                    # B.3 (parse-failure hotfix 2026-05-08): hedges Qwen3.5-9B
                    # uses without an "is" anchor (e.g., "making it the most
                    # contextually relevant", "the most contextually relevant",
                    # "Among the options, Passage X provides..."). Keep narrow
                    # — must be unambiguous "the model is picking" language so
                    # we don't pull a number out of unrelated prose.
                    r"making\s+it\s+(?:the\s+)?(?:most|least)|"
                    r"the\s+(?:most|least)\s+(?:contextually\s+)?relevant|"
                    r"among\s+the\s+(?:options|passages)|"
                    r"is\s+(?:the\s+)?closest)\b",
                    flags=re.IGNORECASE,
                )
                if choice_indicator_re.search(cleaned):
                    trailing = list(re.finditer(r"\b(\d+)\b", cleaned))
                    for match in reversed(trailing):
                        try:
                            idx = int(match.group(1)) - 1
                        except ValueError:
                            continue
                        if 0 <= idx < len(valid_chars):
                            return valid_chars[idx]
                # No choice indicator OR no valid trailing number — original
                # refusal handling.
                if strict:
                    return None
                return valid_chars[0]

            if self._NUMERIC_ONLY_REGEX.match(cleaned):
                return None

            # B.2 (parse-failure hotfix 2026-05-08): scan ALL bare-number matches
            # and return the LAST in-range one. Verbose long-context models
            # (Qwen3.5-9B at N=50/100) often quote a passage with out-of-range
            # numbers (years like "2015", statistics like "44,000") at the
            # START of the output and place their actual answer at the END
            # ("...the CDC said.\n\n36"). Old behavior used `re.search`
            # (first-match-only) which would pick "44" as 44>=valid range,
            # return None, and trigger the strict raise. New behavior mirrors
            # the soft-refusal-recovery branch above (which uses
            # `reversed(trailing)`) but doesn't require refusal language.
            #
            # Qwen3 byte-equality: clean Qwen3 outputs match an EARLIER pattern
            # (numeric_patterns or single trailing digit) so they never enter
            # this branch. Snapshot regression covered by
            # scripts/check_maxcontext_invariants.py.
            all_matches = re.findall(r"\b(\d+)\b", cleaned)
            in_range_indices = []
            for raw_num in all_matches:
                try:
                    idx = int(raw_num) - 1
                except ValueError:
                    continue
                if 0 <= idx < len(valid_chars):
                    in_range_indices.append(idx)
            if in_range_indices:
                return valid_chars[in_range_indices[-1]]
            cross_scheme_label = self._parse_cross_scheme_single_label(
                cleaned,
                valid_chars,
                is_bigram_scheme=is_bigram_scheme,
                is_numeric_scheme=is_numeric_scheme,
            )
            if cross_scheme_label is not None:
                return cross_scheme_label
            return None

        cross_scheme_label = self._parse_cross_scheme_single_label(
            cleaned,
            valid_chars,
            is_bigram_scheme=is_bigram_scheme,
            is_numeric_scheme=is_numeric_scheme,
        )
        if cross_scheme_label is not None:
            return cross_scheme_label

        # Handle refusal outputs: model says none are relevant — default to first
        # passage (maintains current order, equivalent to no swap)
        if re.search(refusal_regex, cleaned):
            return valid_chars[0]

        return None

    def compare(self, query: str, docs: List):
        self.total_compare += 1 if self.num_permutation == 1 else self.num_permutation
        parse_status = "parsed"
        parse_fallback_reason = None

        input_text = self._build_best_prompt(query, docs)

        if self.scoring == 'generation':
            if self.config.model_type == 't5':

                if self.num_permutation == 1:
                    inputs = self._tokenize_inputs(input_text)
                    self.total_prompt_tokens += inputs.input_ids.shape[1]

                    output_ids = self._generate(
                        inputs,
                        max_new_tokens=2,
                        decoder_input_ids=self.decoder_input_ids,
                    )[0]

                    self.total_completion_tokens += output_ids.shape[0]

                    raw_output = self.tokenizer.decode(output_ids,
                                                       skip_special_tokens=True).strip()
                    output = self._parse_single_label(raw_output, self.CHARACTERS[:len(docs)])
                    if output is None:
                        output = self._clean_generation_output(raw_output).upper()
                else:
                    id_passage = [(i, p) for i, p in enumerate(docs)]
                    labels = [self.CHARACTERS[i] for i in range(len(docs))]
                    batch_data = []
                    for _ in range(self.num_permutation):
                        batch_data.append([random.sample(id_passage, len(id_passage)),
                                           random.sample(labels, len(labels))])

                    batch_ref = []
                    input_text = []
                    for batch in batch_data:
                        ref = []
                        passages = []
                        characters = []
                        for p, c in zip(batch[0], batch[1]):
                            ref.append(p[0])
                            passages.append(p[1].text)
                            characters.append(c)
                        batch_ref.append((ref, characters))
                        passages = "\n\n".join([f'Passage {characters[i]}: "{passages[i]}"' for i in range(len(passages))])
                        input_text.append(f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n' \
                                          + passages + '\n\nOutput only the passage label of the most relevant passage:')

                    inputs = self._tokenize_inputs(input_text, padding=True)
                    self.total_prompt_tokens += inputs.input_ids.shape[1] * inputs.input_ids.shape[0]

                    output_ids = self._generate(
                        inputs,
                        max_new_tokens=2,
                        decoder_input_ids=self.decoder_input_ids.repeat(inputs.input_ids.shape[0], 1),
                    )
                    self.total_completion_tokens += output_ids.shape[0] * (output_ids.shape[1] - self.decoder_input_ids.shape[1])

                    output = self.tokenizer.batch_decode(output_ids[:, self.decoder_input_ids.shape[1]:],
                                                         skip_special_tokens=True)

                    # vote
                    candidates = []
                    for ref, result in zip(batch_ref, output):
                        docids, characters = ref
                        result = self._parse_single_label(result, characters)
                        if result is None or result not in characters:
                            print(f"Unexpected output: {self._clean_generation_output(str(result))}")
                            continue
                        win_doc = docids[characters.index(result)]
                        candidates.append(win_doc)

                    if len(candidates) == 0:
                        print(f"Unexpected voting: {output}")
                        output = "Unexpected voting."
                    else:
                        # handle tie
                        candidate_counts = Counter(candidates)
                        max_count = max(candidate_counts.values())
                        most_common_candidates = [candidate for candidate, count in candidate_counts.items() if
                                                  count == max_count]
                        if len(most_common_candidates) == 1:
                            output = self.CHARACTERS[most_common_candidates[0]]
                        else:
                            output = self.CHARACTERS[random.choice(most_common_candidates)]

            elif self._uses_chat_template():
                conversation = [{"role": "user", "content": input_text}]

                prompt = self._build_chat_prompt(conversation)
                prompt += " Passage:"

                retry_budget = self._parse_failure_retry_budget()
                attempt = 0
                while True:
                    snapshot = self._snapshot_retry_invisible_counters()
                    retry_start = time.perf_counter() if attempt > 0 else None

                    inputs = self._tokenize_inputs(prompt)
                    self.total_prompt_tokens += inputs.input_ids.shape[1]

                    # Thinking models (e.g. Qwen3) may emit <think>...</think> before the
                    # answer, so we need enough token budget for the thinking block to
                    # complete.  Non-thinking models will hit EOS well before 256 tokens.
                    max_new = self._generation_budget("single")
                    output_ids = self._generate(inputs, max_new_tokens=max_new)[0]

                    self.total_completion_tokens += output_ids.shape[0] - inputs.input_ids.shape[1]

                    # Decode WITHOUT skipping special tokens so <think>...</think> tags
                    # are preserved for proper stripping by _clean_generation_output.
                    # With skip_special_tokens=True, <think>/<think> are removed but
                    # the thinking *content* leaks through and pollutes parsing.
                    raw_output = self.tokenizer.decode(output_ids[inputs.input_ids.shape[1]:],
                                                       skip_special_tokens=False).strip()
                    if retry_start is not None:
                        self._record_parse_failure_retry(retry_start)
                    if _DEBUG:
                        cleaned_dbg = self._clean_generation_output(raw_output)
                        print(f"[DEBUG] raw={repr(raw_output[:200])}  cleaned={repr(cleaned_dbg[:200])}")
                    output = self._parse_single_label(raw_output, self.CHARACTERS[:len(docs)])
                    if output is not None:
                        break

                    valid_chars = self.CHARACTERS[:len(docs)]
                    reason = self._classify_out_of_window_label_noop(raw_output, valid_chars)
                    if reason is not None:
                        self.total_parse_fallback = getattr(self, "total_parse_fallback", 0) + 1
                        self.total_out_of_window_label_fallback = getattr(
                            self, "total_out_of_window_label_fallback", 0
                        ) + 1
                        output = self._score_order_fallback_label(docs, "best", 0)
                        status_suffix = "score_fallback" if self._score_order_fallback_active() else "noop"
                        parse_status = f"{reason}_{status_suffix}"
                        parse_fallback_reason = reason
                        break

                    reason = self._classify_numeric_noop(raw_output, len(docs))
                    if reason is None and attempt < retry_budget:
                        self._restore_retry_invisible_counters(snapshot)
                        attempt += 1
                        continue
                    if reason is None:
                        reason = self._classify_post_exhaustion_noop(
                            raw_output,
                            len(docs),
                        )
                    if reason is not None:
                        # Deterministic fallback: for MaxContext position-bias controls,
                        # choose the first-stage-score best doc, not the shown first label.
                        self.total_parse_fallback = getattr(self, "total_parse_fallback", 0) + 1
                        counter_name = f"total_{reason}_fallback"
                        setattr(self, counter_name, getattr(self, counter_name, 0) + 1)
                        output = self._score_order_fallback_label(docs, "best", 0)
                        status_suffix = "score_fallback" if self._score_order_fallback_active() else "noop"
                        if _DEBUG or getattr(self, "strict_no_parse_fallback", False):
                            print(f"[MaxContext] {reason} {status_suffix} (best={output}). Raw: {raw_output!r}")
                        parse_status = f"{reason}_{status_suffix}"
                        parse_fallback_reason = reason
                    elif getattr(self, "strict_no_parse_fallback", False):
                        self.total_strict_parse_fallback = getattr(
                            self, "total_strict_parse_fallback", 0
                        ) + 1
                        import sys as _sys
                        safe_default = self._score_order_fallback_label(docs, "best", 0)
                        _sys.stderr.write(
                            f"[MaxContext] strict parse failure after {attempt} retries "
                            f"(budget={retry_budget}, total_retries_so_far={getattr(self, 'total_retries', 0)}, "
                            f"qid={getattr(self, '_current_qid', '?')!r}, "
                            f"model_type={getattr(self.config, 'model_type', None)!r}, "
                            f"is_multimodal={self._is_multimodal_model()!r}, "
                            f"label_scheme={getattr(self, 'label_scheme', None)!r}, "
                            f"n_docs={len(docs)}). "
                            f"Falling back to safe default (best={safe_default!r}). "
                            f"Raw text prefix: {raw_output[:200]!r}\n"
                        )
                        output = safe_default
                        parse_status = "strict_parse_fallback"
                        parse_fallback_reason = "strict_parse_fallback"
                    else:
                        self.total_lenient_fallback = getattr(self, "total_lenient_fallback", 0) + 1
                        output = self._clean_generation_output(raw_output).upper()
                        parse_status = "lenient_fallback"
                    break

        elif self.scoring == 'likelihood':
            # Completion tokens = 0: likelihood reads scores from a single forward
            # pass — no autoregressive decoding occurs.
            scores = self._score_label_candidates(input_text, len(docs))
            ranked = sorted(
                zip(self.CHARACTERS[:len(docs)], scores),
                key=lambda x: x[1],
                reverse=True,
            )
            output = ranked[0][0]

        if output in self.CHARACTERS[:len(docs)]:
            self._log_comparison(
                "best", self.CHARACTERS[:len(docs)], output, docs,
                parse_status=parse_status,
                parse_fallback_reason=parse_fallback_reason,
            )
        else:
            print(f"Unexpected output: {output}")

        return output

    def heapify(self, arr, n, i, query):
        # Find largest among root and children
        if self.num_child * i + 1 < n:  # if there are children
            docs = [arr[i]] + arr[self.num_child * i + 1: min((self.num_child * (i + 1) + 1), n)]
            inds = [i] + list(range(self.num_child * i + 1, min((self.num_child * (i + 1) + 1), n)))
            output = self.compare(query, docs)
            try:
                best_ind = self.CHARACTERS.index(output)
            except ValueError:
                best_ind = 0
            try:
                largest = inds[best_ind]
            except IndexError:
                largest = i
            # If root is not largest, swap with largest and continue heapifying
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                self.heapify(arr, n, largest, query)

    def heapSort(self, arr, query, k):
        n = len(arr)
        ranked = 0
        # Build max heap
        for i in range(n // self.num_child, -1, -1):
            self.heapify(arr, n, i, query)
        for i in range(n - 1, 0, -1):
            # Swap
            arr[i], arr[0] = arr[0], arr[i]
            ranked += 1
            if ranked == k:
                break
            # Heapify root element
            self.heapify(arr, i, 0, query)

    def rerank(self,  query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        original_ranking = copy.deepcopy(ranking)
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        self.total_retries = 0
        self.total_retry_overhead_seconds = 0.0
        self.total_parse_fallback = 0
        self.total_out_of_window_label_fallback = 0
        self.total_lexical_refusal_fallback = 0
        self.total_numeric_out_of_range_fallback = 0
        self.total_degenerate_repetition_fallback = 0
        self.total_unparseable_after_exhaustion_fallback = 0
        self.total_duplicate_label_fallback = 0
        self.total_lenient_fallback = 0
        
        if self.method == "heapsort":
            self.heapSort(ranking, query, self.k)
            ranking = list(reversed(ranking))
        elif self.method == "bubblesort":
            last_start = len(ranking) - (self.num_child + 1)
            # Disable the outer last_start clamp whenever the comparator window
            # already spans the whole live pool. The bubble window below is
            #   min(start_ind + num_child + 1, len(ranking))
            # so window-covers-pool is exactly num_child + 1 >= len(ranking).
            # Subsumes both prior clauses: (len == k == num_child) and
            # (num_child >= len), and adds the previously-missed boundary
            # num_child + 1 == len (e.g. window=k=hits=10, num_child=9).
            disable_outer_clamp = self.num_child + 1 >= len(ranking)

            for i in range(self.k):
                # Keep the local clamp for ordinary windows, but preserve the
                # upstream-style start for whole-pool runs so suffix selections
                # do not short-circuit the remaining comparisons.
                if not disable_outer_clamp and last_start < i:
                    last_start = i
                start_ind = last_start
                end_ind = min(last_start + (self.num_child + 1), len(ranking))
                is_change = False
                while True:
                    if start_ind < i:
                        start_ind = i
                    # Need at least 2 documents for a meaningful comparison
                    if end_ind - start_ind < 2:
                        break
                    output = self.compare(query, ranking[start_ind:end_ind])
                    try:
                        best_ind = self.CHARACTERS.index(output)
                    except ValueError:
                        best_ind = 0
                    if best_ind != 0:
                        ranking[start_ind], ranking[start_ind + best_ind] = ranking[start_ind + best_ind], ranking[start_ind]
                        if not is_change:
                            is_change = True
                            if last_start != len(ranking) - (self.num_child + 1) \
                                    and best_ind == len(ranking[start_ind:end_ind])-1:
                                last_start += len(ranking[start_ind:end_ind])-1

                    if start_ind == i:
                        break

                    if not is_change:
                        last_start -= self.num_child

                    start_ind -= self.num_child
                    end_ind -= self.num_child
                    
        ##  this is a bit slower but standard bobblesort implementation, keep here FYI
        # elif self.method == "bubblesort":
        #     for i in range(k):
        #         start_ind = len(ranking) - (self.num_child + 1)
        #         end_ind = len(ranking)
        #         while True:
        #             if start_ind < i:
        #                 start_ind = i
        #             output = self.compare(query, ranking[start_ind:end_ind])
        #             try:
        #                 best_ind = self.CHARACTERS.index(output)
        #             except ValueError:
        #                 best_ind = 0
        #             if best_ind != 0:
        #                 ranking[start_ind], ranking[start_ind + best_ind] = ranking[start_ind + best_ind], ranking[start_ind]
        #
        #             if start_ind == i:
        #                 break
        #
        #             start_ind -= self.num_child
        #             end_ind -= self.num_child

        else:
            raise NotImplementedError(f'Method {self.method} is not implemented.')

        results = []
        top_doc_ids = set()
        rank = 1

        for i, doc in enumerate(ranking[:self.k]):
            top_doc_ids.add(doc.docid)
            results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
            rank += 1
        for doc in original_ranking:
            if doc.docid not in top_doc_ids:
                results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
                rank += 1

        return results

    def truncate(self, text, length):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text)[:length])


class OpenAiSetwiseLlmRanker(SetwiseLlmRanker):
    def __init__(self, model_name_or_path, api_key, num_child=3, method='heapsort', k=10):
        self.llm = model_name_or_path
        self.tokenizer = tiktoken.encoding_for_model(model_name_or_path)
        self.num_child = num_child
        self.method = method
        self.k = k
        self.total_compare = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.system_prompt = "You are RankGPT, an intelligent assistant specialized in selecting the most relevant passage from a pool of passages based on their relevance to the query."
        openai.api_key = api_key

    def compare(self, query: str, docs: List):
        self.total_compare += 1
        passages = "\n\n".join([f'Passage {self.CHARACTERS[i]}: "{doc.text}"' for i, doc in enumerate(docs)])
        input_text = f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n' \
                     + passages + '\n\nOutput only the passage label of the most relevant passage.'

        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=self.llm,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": input_text},
                    ],
                    temperature=0.0,
                    request_timeout=15
                )

                self.total_completion_tokens += int(response['usage']['completion_tokens'])
                self.total_prompt_tokens += int(response['usage']['prompt_tokens'])

                output = response['choices'][0]['message']['content']
                matches = re.findall(r"(Passage [A-Z])", output, re.MULTILINE)
                if matches:
                    output = matches[0][8]
                elif output.strip() in self.CHARACTERS:
                    pass
                else:
                    print(f"Unexpected output: {output}")
                    output = "A"
                return output

            except openai.error.APIError as e:
                # Handle API error here, e.g. retry or log
                print(f"OpenAI API returned an API Error: {e}")
                time.sleep(5)
                continue
            except openai.error.APIConnectionError as e:
                # Handle connection error here
                print(f"Failed to connect to OpenAI API: {e}")
                time.sleep(5)
                continue
            except openai.error.RateLimitError as e:
                # Handle rate limit error (we recommend using exponential backoff)
                print(f"OpenAI API request exceeded rate limit: {e}")
                time.sleep(5)
                continue
            except openai.error.InvalidRequestError as e:
                # Handle invalid request error
                print(f"OpenAI API request was invalid: {e}")
                raise e
            except openai.error.AuthenticationError as e:
                # Handle authentication error
                print(f"OpenAI API request failed authentication: {e}")
                raise e
            except openai.error.Timeout as e:
                # Handle timeout error
                print(f"OpenAI API request timed out: {e}")
                time.sleep(5)
                continue
            except openai.error.ServiceUnavailableError as e:
                # Handle service unavailable error
                print(f"OpenAI API request failed with a service unavailable error: {e}")
                time.sleep(5)
                continue
            except Exception as e:
                print(f"Unknown error: {e}")
                raise e

    def truncate(self, text, length):
        return self.tokenizer.decode(self.tokenizer.encode(text)[:length])



class RankR1SetwiseLlmRanker(SetwiseLlmRanker):
    CHARACTERS = [f'[{i+1}]' for i in range(20)]

    def __init__(self,
                 model_name_or_path,
                 prompt_file,
                 lora_name_or_path=None,
                 tokenizer_name_or_path=None,
                 num_child=19,
                 k=10,
                 scoring='generation',
                 method="heapsort",
                 num_permutation=1,
                 cache_dir=None,
                 verbose=False):

        if VLLM_IMPORT_ERROR is not None:
            raise ImportError(
                "vllm is required for RankR1SetwiseLlmRanker. Install vllm to use this ranker."
            ) from VLLM_IMPORT_ERROR

        if scoring != 'generation':
            raise NotImplementedError(f"Scoring method {scoring} is not supported for RankR1SetwiseLlmRanker. RankR1SetwiseLlmRanker only supports 'generation' scoring.")
        self.verbose = verbose

        import toml
        self.prompt = toml.load(prompt_file)

        from huggingface_hub import snapshot_download
        import os
        if lora_name_or_path is not None:
            # check if the path exists
            if not os.path.exists(lora_name_or_path):
                # download the model
                lora_path = snapshot_download(lora_name_or_path)
            else:
                lora_path = lora_name_or_path
        else:
            lora_path = None

        self.lora_path = lora_path
        self.num_child = num_child
        self.num_permutation = num_permutation
        self.k = k
        self.sampling_params = SamplingParams(temperature=0.0,
                                              max_tokens=2048)
        if tokenizer_name_or_path is None:
            tokenizer_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=cache_dir)
        self.llm = LLM(model=model_name_or_path,
                       tokenizer=tokenizer_name_or_path,
                       enable_lora=True if lora_name_or_path is not None else False,
                       max_lora_rank=32,
                       )

        self.scoring = scoring
        self.method = method
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0

    def compare(self, query: str, docs: List):
        self.total_compare += 1 if self.num_permutation == 1 else self.num_permutation

        id_passage = [(i, p) for i, p in enumerate(docs)]
        labels = [self.CHARACTERS[i] for i in range(len(docs))]
        batch_data = []
        for _ in range(self.num_permutation):
            batch_data.append([random.sample(id_passage, len(id_passage)),
                               labels])

        batch_ref = []
        input_text = []
        for batch in batch_data:
            ref = []
            passages = []
            characters = []
            for p, c in zip(batch[0], batch[1]):
                ref.append(p[0])
                passages.append(p[1].text)
                characters.append(c)
            batch_ref.append((ref, characters))
            passages = "\n".join([f'{characters[i]} {passages[i]}' for i in range(len(passages))])
            system_message = self.prompt["prompt_system"]
            user_message = self.prompt['prompt_user'].format(query=query,
                                                             docs=passages)
            input_text.append([
                {'role': "system", 'content': system_message},
                {'role': "user", 'content': user_message}
            ])
        outputs = self.llm.chat(input_text,
                                sampling_params=self.sampling_params,
                                use_tqdm=False,
                                lora_request=LoRARequest("R1adapter",
                                                         1,
                                                         self.lora_path)
                                if self.lora_path is not None else None,
                                )
        results = []
        for output, input in zip(outputs, input_text):
            self.total_completion_tokens += len(output.outputs[0].token_ids)
            self.total_prompt_tokens += len(output.prompt_token_ids)

            completion = output.outputs[0].text

            if self.verbose:
                print('--------------------------------------')
                print(f'query: {query}')
                print(f'input_text:\n{self.tokenizer.apply_chat_template(input, tokenize=False)}')
                print(f'completion:\n{completion}')
                print('--------------------------------------')

            pattern = rf'{self.prompt["pattern"]}'
            match = re.search(pattern, completion.lower(), re.DOTALL)
            if match:
                results.append(match.group(1).strip())
            else:
                results.append(f'input_text:\n{input}, completion:\n{completion}')

        # vote
        candidates = []
        for ref, result in zip(batch_ref, results):
            result = result.strip()
            docids, characters = ref
            if result not in characters:
                if self.verbose:
                    print(f"Unexpected output: {result}")
                continue
            win_doc = docids[characters.index(result)]
            candidates.append(win_doc)

        if len(candidates) == 0:
            if self.verbose:
                print(f"Unexpected voting: {results}")
            output = "Unexpected voting."
        else:
            # handle tie
            candidate_counts = Counter(candidates)
            max_count = max(candidate_counts.values())
            most_common_candidates = [candidate for candidate, count in candidate_counts.items() if
                                      count == max_count]
            if len(most_common_candidates) == 1:
                output = self.CHARACTERS[most_common_candidates[0]]
            else:
                output = self.CHARACTERS[random.choice(most_common_candidates)]

        if output in self.CHARACTERS:
            pass
        else:
            if self.verbose:
                print(f"Unexpected output: {output}")

        return output
