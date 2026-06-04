"""
Extended Setwise Ranking Strategies:
1. BottomUpSetwiseLlmRanker - Selects the LEAST relevant document (reverse/bottom-up ranking)
2. DualEndSetwiseLlmRanker - Selects BOTH most and least relevant simultaneously
3. BidirectionalEnsembleRanker - Runs both top-down and bottom-up, fuses rankings

Reference: Extends the setwise approach from Zhuang et al. (SIGIR 2024)
"""

from typing import Dict, List, Optional, Sequence, Tuple
from .rankers import LlmRanker, SearchResult
from .setwise import (
    MAX_PARSE_FAILURE_RETRIES,
    QWEN_MODEL_TYPES,
    SetwiseLlmRanker,
    compute_max_fit_window,
)
import copy
import hashlib
import math
import re
import time
import torch
import random
from collections import Counter

random.seed(929)

MAXCONTEXT_SHUFFLE_SEED = 929

MAXCONTEXT_ALLOWED_MODEL_TYPES = frozenset({
    "qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_moe",
    "llama",
    "mistral", "mistral3", "ministral",
})


def _setup_maxcontext_numeric_attrs(ranker, pool_size: int) -> None:
    ranker.CHARACTERS = [str(i + 1) for i in range(pool_size)]
    ranker.num_child = pool_size - 1
    ranker.method = "selection"
    ranker.strict_no_truncation = True
    ranker.strict_no_parse_fallback = True
    ranker._parse_failure_max_retries = MAX_PARSE_FAILURE_RETRIES
    ranker.total_parse_fallback = 0
    ranker.total_lexical_refusal_fallback = 0
    ranker.total_numeric_out_of_range_fallback = 0
    ranker.total_degenerate_repetition_fallback = 0
    ranker.total_unparseable_after_exhaustion_fallback = 0
    ranker.total_duplicate_label_fallback = 0
    ranker.total_lenient_fallback = 0
    ranker.total_strict_parse_fallback = 0
    # Parse-failure counters:
    #   total_parse_failure_strict      — smoke-gated propagated strict failures
    #   total_strict_parse_fallback     — recovered local strict-mode fallbacks
    #   total_parse_failure_bm25_fallback — incremented when the BM25 fallback
    #     handler catches a parse-failure ValueError and recovers the query.
    #   total_retries / total_retry_overhead_seconds — retry-visible only in
    #     logs; hidden from comparison/token/time metrics.
    ranker.total_parse_failure_strict = 0
    ranker.total_parse_failure_bm25_fallback = 0
    ranker.total_retries = 0
    ranker.total_retry_overhead_seconds = 0.0
    ranker._allow_parse_failure_bm25_fallback = False
    ranker.label_scheme = "numeric_1_based"
    ranker._maxcontext_pool_size = pool_size


def _maxcontext_should_fallback_to_bm25(ranker, exc: BaseException) -> bool:
    """Return True iff the BM25-fallback path should swallow this ValueError.

    Gates on:
      * The ranker has the fallback flag enabled.
      * The exception message identifies a MaxContext label-parse failure
        (single-label or dual-output) — never swallows shape/contract errors
        like the "n_docs=2 bypass requires finite BM25 scores" pre-flight.
    """
    if not getattr(ranker, "_allow_parse_failure_bm25_fallback", False):
        return False
    msg = str(exc)
    return (
        "MaxContext single-label parse failed" in msg
        or "MaxContext dual-output parse failed" in msg
        or "Duplicate best/worst label" in msg
    )


def _reset_maxcontext_query_stats(ranker) -> None:
    """Reset per-query counters before MaxContext early returns/fallbacks."""
    ranker.total_compare = 0
    ranker.total_completion_tokens = 0
    ranker.total_prompt_tokens = 0
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
    ranker.total_parse_failure_bm25_fallback = 0
    if hasattr(ranker, "total_bm25_bypass"):
        ranker.total_bm25_bypass = 0


def _validate_maxcontext_pool_size(ranker, docs: List[SearchResult], name: str) -> None:
    max_docs = getattr(ranker, "_maxcontext_pool_size", None)
    if max_docs is not None and len(docs) > max_docs:
        raise ValueError(
            f"{name} expects at most pool_size={max_docs} input docs; got {len(docs)}."
        )


def _materialize_maxcontext_results(docs: Sequence[SearchResult]) -> List[SearchResult]:
    return [
        SearchResult(docid=doc.docid, score=-rank, text=None)
        for rank, doc in enumerate(docs, start=1)
    ]


def _resolve_maxcontext_label_index(
    ranker, label: str, window_len: int, default: int
) -> int:
    strict = getattr(ranker, "strict_no_parse_fallback", False)
    try:
        idx = ranker.CHARACTERS.index(label)
    except ValueError:
        if strict:
            ranker.total_strict_parse_fallback = getattr(
                ranker, "total_strict_parse_fallback", 0
            ) + 1
            import sys as _sys
            _sys.stderr.write(
                f"[MaxContext] strict label resolution fallback: label {label!r} not in CHARACTERS. "
                f"Falling back to default index {default}.\n"
            )
        return default
    if idx >= window_len:
        if strict:
            ranker.total_strict_parse_fallback = getattr(
                ranker, "total_strict_parse_fallback", 0
            ) + 1
            import sys as _sys
            _sys.stderr.write(
                f"[MaxContext] strict label resolution fallback: label {label!r} resolves to "
                f"index {idx} outside active window of size {window_len}. "
                f"Falling back to last in-window index {window_len - 1}.\n"
            )
        return min(idx, window_len - 1)
    return idx


class _MaxContextOrderingMixin:
    """MaxContext-only ordering controls for the LLM-presented pool."""

    def _apply_pool_ordering(self, docs: list) -> list:
        if not (getattr(self, "shuffle", False) or getattr(self, "reverse", False)):
            return docs
        if getattr(self, "reverse", False):
            return list(reversed(docs))
        seed = self._derive_shuffle_seed(len(docs))
        rng = random.Random(seed)
        out = list(docs)
        rng.shuffle(out)
        return out

    def _derive_shuffle_seed(self, round_pool_size: int) -> int:
        qid = str(getattr(self, "_current_qid", "") or "")
        digest = hashlib.blake2b(
            f"{MAXCONTEXT_SHUFFLE_SEED}\0{qid}\0{round_pool_size}".encode(),
            digest_size=8,
        ).digest()
        return int.from_bytes(digest, "big", signed=False)

    def _remap_label_to_original(
        self,
        label: str,
        presented: list,
        original_window: list,
        default: int,
    ) -> int:
        presented_idx = _resolve_maxcontext_label_index(
            self, label, len(presented), default
        )
        if presented is original_window:
            return presented_idx
        target_doc = presented[presented_idx]
        for i, doc in enumerate(original_window):
            if doc.docid == target_doc.docid:
                return i
        raise ValueError(
            f"Doc {target_doc.docid!r} from presented window not found in original_window"
        )


def _assert_maxcontext_topdown_fits(ranker, query, docs) -> None:
    if ranker.max_input_tokens is None:
        raise ValueError("MaxContext requires max_input_tokens to be resolvable.")

    input_text = ranker._build_best_prompt(query, docs)
    rendered = ranker._build_chat_prompt([{"role": "user", "content": input_text}])
    rendered += " Passage:"

    rendered_ids = ranker.tokenizer.encode(rendered, add_special_tokens=True)
    rendered_length = len(rendered_ids)
    budget = ranker.max_input_tokens - 256

    if rendered_length > budget:
        raise ValueError(
            f"MaxContext TopDown preflight failed: rendered prompt is "
            f"{rendered_length} tokens but the budget is {budget} "
            f"(max_input_tokens - 256). Reduce --passage_length or --k."
        )


def _assert_maxcontext_bottomup_fits(ranker, query, docs) -> None:
    if ranker.max_input_tokens is None:
        raise ValueError("MaxContext requires max_input_tokens to be resolvable.")

    input_text = ranker._build_worst_prompt(query, docs)
    rendered = ranker._build_chat_prompt([{"role": "user", "content": input_text}])
    rendered += " Passage:"

    rendered_ids = ranker.tokenizer.encode(rendered, add_special_tokens=True)
    rendered_length = len(rendered_ids)
    budget = ranker.max_input_tokens - 256

    if rendered_length > budget:
        raise ValueError(
            f"MaxContext BottomUp preflight failed: rendered prompt is "
            f"{rendered_length} tokens but the budget is {budget} "
            f"(max_input_tokens - 256). Reduce --passage_length or --k."
        )


class BottomUpSetwiseLlmRanker(SetwiseLlmRanker):
    """
    Bottom-Up Setwise Ranker: selects the LEAST relevant document from each comparison set.
    Builds the ranking from the bottom up by iteratively removing the worst documents.

    For top-k ranking (k << n), this requires n-k extractions from a min-heap plus
    k extractions to sort the survivors, totaling n comparisons. This is less efficient
    than standard top-down (k extractions from max-heap) when k << n.
    However, it may exhibit different effectiveness and position bias characteristics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compare_worst(self, query: str, docs: List):
        """Select the LEAST relevant document from the candidate set."""
        self.total_compare += 1 if self.num_permutation == 1 else self.num_permutation
        parse_status = "parsed"
        parse_fallback_reason = None

        input_text = self._build_worst_prompt(query, docs)

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
                        input_text.append(f'Given a query "{query}", which of the following passages is the least relevant one to the query?\n\n' \
                                          + passages + '\n\nOutput only the passage label of the least relevant passage:')

                    inputs = self._tokenize_inputs(input_text, padding=True)
                    self.total_prompt_tokens += inputs.input_ids.shape[1] * inputs.input_ids.shape[0]

                    output_ids = self._generate(
                        inputs,
                        max_new_tokens=2,
                        decoder_input_ids=self.decoder_input_ids.repeat(inputs.input_ids.shape[0], 1),
                    )
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
                        worst_doc = docids[characters.index(result)]
                        candidates.append(worst_doc)

                    if len(candidates) == 0:
                        print(f"Unexpected voting: {output}")
                        output = "Unexpected voting."
                    else:
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

                    max_new = self._generation_budget("single")
                    output_ids = self._generate(inputs, max_new_tokens=max_new)[0]

                    self.total_completion_tokens += output_ids.shape[0] - inputs.input_ids.shape[1]

                    raw_output = self.tokenizer.decode(output_ids[inputs.input_ids.shape[1]:],
                                                       skip_special_tokens=False).strip()
                    if retry_start is not None:
                        self._record_parse_failure_retry(retry_start)
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
                        output = self._score_order_fallback_label(docs, "worst", len(docs) - 1)
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
                        reason = self._classify_post_exhaustion_noop(raw_output, len(docs))
                    if reason is not None:
                        # Deterministic fallback: for MaxContext position-bias controls,
                        # choose the first-stage-score worst doc, not the shown tail label.
                        self.total_parse_fallback = getattr(self, "total_parse_fallback", 0) + 1
                        counter_name = f"total_{reason}_fallback"
                        setattr(self, counter_name, getattr(self, counter_name, 0) + 1)
                        output = self._score_order_fallback_label(docs, "worst", len(docs) - 1)
                        status_suffix = "score_fallback" if self._score_order_fallback_active() else "noop"
                        print(f"[MaxContext] {reason} {status_suffix} (worst={output}). Raw: {raw_output!r}")
                        parse_status = f"{reason}_{status_suffix}"
                        parse_fallback_reason = reason
                    elif getattr(self, "strict_no_parse_fallback", False):
                        self.total_strict_parse_fallback = getattr(
                            self, "total_strict_parse_fallback", 0
                        ) + 1
                        import sys as _sys
                        safe_default = self._score_order_fallback_label(docs, "worst", len(docs) - 1)
                        _sys.stderr.write(
                            f"[MaxContext] strict parse failure after {attempt} retries "
                            f"(budget={retry_budget}, total_retries_so_far={getattr(self, 'total_retries', 0)}, "
                            f"qid={getattr(self, '_current_qid', '?')!r}). "
                            f"Falling back to safe default (worst={safe_default!r}). "
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
            scores = self._score_label_candidates(input_text, len(docs))
            # For bottom-up with the explicit "least relevant" prompt, the highest-
            # scoring label is the model's predicted worst document.
            ranked = sorted(
                zip(self.CHARACTERS[:len(docs)], scores),
                key=lambda x: x[1],
                reverse=True,
            )
            output = ranked[0][0]

        if output in self.CHARACTERS[:len(docs)]:
            self._log_comparison(
                "worst", self.CHARACTERS[:len(docs)], output, docs,
                parse_status=parse_status,
                parse_fallback_reason=parse_fallback_reason,
            )
        else:
            print(f"Unexpected output: {output}")

        return output

    def heapify_min(self, arr, n, i, query):
        """Min-heapify: select the WORST (least relevant) as root."""
        if self.num_child * i + 1 < n:
            docs = [arr[i]] + arr[self.num_child * i + 1: min((self.num_child * (i + 1) + 1), n)]
            inds = [i] + list(range(self.num_child * i + 1, min((self.num_child * (i + 1) + 1), n)))
            output = self.compare_worst(query, docs)
            try:
                worst_ind = self.CHARACTERS.index(output)
            except ValueError:
                worst_ind = 0
            try:
                smallest = inds[worst_ind]
            except IndexError:
                smallest = i
            if smallest != i:
                arr[i], arr[smallest] = arr[smallest], arr[i]
                self.heapify_min(arr, n, smallest, query)

    def heapSort(self, arr, query, k):
        """Bottom-up heapsort: build min-heap, extract all minimums.

        Each extraction removes the least relevant (worst) document and places
        it at the end. After all extractions, the array is sorted best-first
        (best at position 0, worst at position n-1).

        Total work: n-1 extractions using compare_worst only.
        For k=10, n=100: 99 extractions (vs. 10 for standard top-down).
        """
        n = len(arr)

        # Build min-heap (worst at root)
        for i in range(n // self.num_child, -1, -1):
            self.heapify_min(arr, n, i, query)

        # Extract all minimums: each step removes the worst remaining document
        # After all extractions, arr is sorted best-first
        for i in range(n - 1, 0, -1):
            # Swap root (worst) with last unsorted element
            arr[i], arr[0] = arr[0], arr[i]
            # Re-heapify remaining elements
            self.heapify_min(arr, i, 0, query)

    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
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
            # heapSort produces best-first order, no reversal needed

        elif self.method == "bubblesort":
            # Bottom-up bubblesort: sink worst documents to the bottom
            # Do n-1 passes to fully sort (each pass places one more doc correctly)
            n = len(ranking)

            for i in range(n - 1):
                # Sink worst to position n-1-i
                target_pos = n - 1 - i
                start_ind = 0
                end_ind = min(self.num_child + 1, target_pos + 1)

                while end_ind <= target_pos + 1 and start_ind < target_pos:
                    window = ranking[start_ind:end_ind]
                    if len(window) < 2:
                        break
                    output = self.compare_worst(query, window)
                    try:
                        worst_ind = self.CHARACTERS.index(output)
                    except ValueError:
                        worst_ind = len(window) - 1
                    worst_ind = min(worst_ind, len(window) - 1)

                    if worst_ind != len(window) - 1:
                        # Move worst to end of window (sink down)
                        actual_worst_pos = start_ind + worst_ind
                        actual_end_pos = start_ind + len(window) - 1
                        ranking[actual_worst_pos], ranking[actual_end_pos] = ranking[actual_end_pos], ranking[actual_worst_pos]

                    start_ind += self.num_child
                    end_ind = min(start_ind + self.num_child + 1, target_pos + 1)
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



class _DualEndRoutingMixin:
    """Shared gating and accounting helpers for DualEnd refinements."""

    def _init_joint_signal_routing(
        self,
        gate_strategy: str = "hybrid",
        shortlist_size: int = 20,
        margin_threshold: float = 0.15,
        uncertainty_percentile: Optional[float] = None,
    ) -> None:
        self.gate_strategy = gate_strategy
        self.shortlist_size = shortlist_size
        # Backward compatibility: older scripts still pass --margin_threshold.
        # We now interpret that value as a query-local uncertainty percentile.
        if uncertainty_percentile is None:
            uncertainty_percentile = margin_threshold
        self.margin_threshold = margin_threshold
        self.uncertainty_percentile = self._normalize_uncertainty_percentile(
            uncertainty_percentile
        )
        self._query_uncertainty_thresholds: Dict[int, float] = {}
        self._reset_joint_signal_stats()

    def _reset_joint_signal_stats(self) -> None:
        self.total_dual_invocations = 0
        self.total_single_invocations = 0
        self.total_order_robust_windows = 0
        self.total_extra_orderings = 0
        self.total_regularized_worst_moves = 0

    def _window_relative_score_spread(self, docs: Sequence[SearchResult]) -> Optional[float]:
        scores = [doc.score for doc in docs if getattr(doc, "score", None) is not None]
        if len(scores) < 2:
            return None
        max_score = max(scores)
        min_score = min(scores)
        denom = max(abs(max_score), abs(min_score), 1e-6)
        return (max_score - min_score) / denom

    @staticmethod
    def _normalize_uncertainty_percentile(value: float) -> float:
        if value > 1.0:
            value /= 100.0
        return min(max(value, 0.0), 1.0)

    @staticmethod
    def _percentile_value(values: Sequence[float], percentile: float) -> Optional[float]:
        if not values:
            return None
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]

        position = percentile * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]

        fraction = position - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

    def _prepare_query_uncertainty_thresholds(
        self,
        ranking: Sequence[SearchResult],
    ) -> None:
        if getattr(self, "gate_strategy", "off") not in {"uncertain", "hybrid"}:
            self._query_uncertainty_thresholds = {}
            return

        thresholds: Dict[int, float] = {}
        max_window_len = min(self.num_child + 1, len(ranking))
        for window_len in range(2, max_window_len + 1):
            spreads = []
            for start in range(0, len(ranking) - window_len + 1):
                spread = self._window_relative_score_spread(
                    ranking[start:start + window_len]
                )
                if spread is not None:
                    spreads.append(spread)
            threshold = self._percentile_value(
                spreads,
                getattr(self, "uncertainty_percentile", 0.15),
            )
            if threshold is not None:
                thresholds[window_len] = threshold

        self._query_uncertainty_thresholds = thresholds

    def _should_use_joint_signal(
        self,
        start_ind: int,
        end_ind: int,
        total_docs: int,
        docs: Sequence[SearchResult],
        allow_shortlist: bool = True,
    ) -> bool:
        gate_strategy = getattr(self, "gate_strategy", "off")
        if gate_strategy == "off":
            return False

        shortlist_size = max(0, min(getattr(self, "shortlist_size", 0), total_docs))
        overlaps_shortlist = allow_shortlist and start_ind < shortlist_size

        relative_spread = self._window_relative_score_spread(docs)
        spread_threshold = getattr(self, "_query_uncertainty_thresholds", {}).get(len(docs))
        is_uncertain = (
            relative_spread is not None
            and spread_threshold is not None
            and relative_spread <= spread_threshold
        )

        if gate_strategy == "shortlist":
            return overlaps_shortlist
        if gate_strategy == "uncertain":
            return is_uncertain
        if gate_strategy == "hybrid":
            return overlaps_shortlist or is_uncertain
        raise ValueError(f"Unsupported gate strategy: {gate_strategy}")

    def _resolve_label_index(self, label: str, window_len: int, default: int) -> int:
        try:
            return min(self.CHARACTERS.index(label), window_len - 1)
        except ValueError:
            return default

    def _remap_window_label(self, original_indices: List[int], local_label: str, default: int) -> int:
        local_index = self._resolve_label_index(local_label, len(original_indices), default)
        return original_indices[local_index]


class DualEndSetwiseLlmRanker(SetwiseLlmRanker):
    """
    Dual-End Setwise Ranker: selects BOTH the most and least relevant documents
    from each comparison set in a single LLM call.

    This extracts 2x information per LLM call, potentially reducing the number
    of required comparisons for the same ranking quality.

    Supports three sorting methods:
    - 'bubblesort': Cocktail shaker sort (bidirectional bubble sort)
    - 'heapsort': Standard heapsort (uses parent compare for heap, bonus worst tracking)
    - 'selection': Double-ended selection sort
    """
    strict_no_parse_fallback: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _compare_both_window(self, query: str, ranking: List[SearchResult], start_ind: int, end_ind: int) -> Tuple[str, str]:
        """Hook for window-aware DualEnd variants."""
        return self.compare_both(query, ranking[start_ind:end_ind])

    def compare_both(self, query: str, docs: List) -> Tuple[str, str]:
        """Select both the MOST and LEAST relevant documents from the candidate set.

        Returns:
            Tuple[str, str]: (best_label, worst_label) as character labels
        """
        self.total_compare += 1 if self.num_permutation == 1 else self.num_permutation
        parse_status = "parsed"
        parse_fallback_reason = None
        comparison_raw_output = None
        latest_raw_output = None
        strict_raise_attempt = 0
        strict_raise_budget = self._parse_failure_retry_budget()
        is_numeric = getattr(self, "label_scheme", None) == "numeric_1_based"
        model_type = getattr(self.config, "model_type", None)

        if len(docs) < 2:
            return self.CHARACTERS[0], self.CHARACTERS[0]

        input_text = self._build_dual_prompt_text(query, docs)

        best = None
        worst = None

        if self.scoring == 'generation':
            if self.config.model_type == 't5':
                # T5 generation cannot reliably produce dual-format output ("Best: X, Worst: Y")
                # because: (1) T5's 512-token input limit is easily exceeded by the longer dual
                # prompt, causing truncation and garbage output; (2) T5 tends to echo the template
                # literally ("Best: [label], Worst: [label]") instead of filling in actual labels.
                #
                # Solution: use likelihood scoring internally — a single forward pass that reads
                # the full label distribution, taking max as best and min as worst. This is:
                # - Exactly ONE forward pass (satisfies the single-call constraint)
                # - No parsing needed (reads directly from logits)
                # - Uses the shorter "most relevant" prompt (fits within 512 tokens)
                likelihood_text = self._build_best_prompt(query, docs)
                scores = self._score_label_candidates(likelihood_text, len(docs))
                ranked = sorted(
                    zip(self.CHARACTERS[:len(docs)], scores),
                    key=lambda x: x[1],
                    reverse=True,
                )
                best = ranked[0][0]
                worst = ranked[-1][0]

            elif self._uses_chat_template():
                conversation = [{"role": "user", "content": input_text}]
                prompt = self._build_chat_prompt(conversation)

                retry_budget = self._parse_failure_retry_budget()
                strict_raise_budget = retry_budget
                attempt = 0
                while True:
                    strict_raise_attempt = attempt
                    snapshot = self._snapshot_retry_invisible_counters()
                    retry_start = time.perf_counter() if attempt > 0 else None

                    inputs = self._tokenize_inputs(prompt)
                    self.total_prompt_tokens += inputs.input_ids.shape[1]

                    # Thinking models (Qwen3) need a larger budget: the <think>...</think>
                    # block can easily consume 200+ tokens before the answer appears.
                    # 512 tokens gives ample room for thinking + "Best: A, Worst: C".
                    max_new = self._generation_budget("dual")
                    output_ids = self._generate(inputs, max_new_tokens=max_new)[0]

                    self.total_completion_tokens += output_ids.shape[0] - inputs.input_ids.shape[1]

                    raw_output = self.tokenizer.decode(output_ids[inputs.input_ids.shape[1]:],
                                                       skip_special_tokens=False).strip()
                    latest_raw_output = raw_output
                    if retry_start is not None:
                        self._record_parse_failure_retry(retry_start)
                    cleaned_output = self._clean_generation_output(raw_output)
                    # Always parse from the single call — never fall back to 2 separate calls
                    try:
                        best, worst = self._parse_dual_output(cleaned_output, len(docs))
                        if best == worst and self.strict_no_parse_fallback and is_numeric:
                            raise ValueError(f"duplicate_label:{best}")
                        break
                    except ValueError as exc:
                        duplicate_label = None
                        if str(exc).startswith("duplicate_label:"):
                            duplicate_label = str(exc).split(":", 1)[1]
                            reason = None
                        else:
                            reason = self._classify_numeric_noop(raw_output, len(docs))
                        if reason is None and attempt < retry_budget:
                            self._restore_retry_invisible_counters(snapshot)
                            attempt += 1
                            continue
                        post_exhaustion_reason = None
                        if reason is None:
                            post_exhaustion_reason = self._classify_post_exhaustion_noop(
                                raw_output,
                                len(docs),
                                duplicate_label=duplicate_label,
                            )
                            reason = post_exhaustion_reason
                        if not is_numeric:
                            self.total_parse_failure_strict = snapshot.total_parse_failure_strict
                            self.total_strict_parse_fallback = getattr(
                                self, "total_strict_parse_fallback", 0
                            ) + 1
                            fallback_best, fallback_worst = self._score_order_fallback_dual_labels(docs)
                            import sys as _sys
                            _sys.stderr.write(
                                f"[MaxContext] strict parse failure after {attempt} retries "
                                f"(budget={retry_budget}, total_retries_so_far={getattr(self, 'total_retries', 0)}, "
                                f"qid={getattr(self, '_current_qid', '?')!r}, "
                                f"model_type={model_type!r}, "
                                f"is_multimodal={self._is_multimodal_model()!r}, "
                                f"label_scheme={getattr(self, 'label_scheme', None)!r}, "
                                f"n_docs={len(docs)}). "
                                f"Falling back to safe default (best={fallback_best!r}, "
                                f"worst={fallback_worst!r}). "
                                f"Raw text prefix: {raw_output[:200]!r}\n"
                            )
                            best = fallback_best
                            worst = fallback_worst
                            parse_status = "strict_parse_fallback"
                            parse_fallback_reason = "strict_parse_fallback"
                            comparison_raw_output = raw_output
                            break
                        has_structured = bool(re.search(
                            r"\b(BEST|WORST|PASSAGE\s*\d+)\b",
                            cleaned_output,
                            flags=re.IGNORECASE,
                        ))
                        if reason is None or (has_structured and post_exhaustion_reason is None):
                            self.total_parse_failure_strict = snapshot.total_parse_failure_strict
                            self.total_strict_parse_fallback = getattr(
                                self, "total_strict_parse_fallback", 0
                            ) + 1
                            fallback_best, fallback_worst = self._score_order_fallback_dual_labels(docs)
                            import sys as _sys
                            _sys.stderr.write(
                                f"[MaxContext] strict parse failure after {attempt} retries "
                                f"(budget={retry_budget}, total_retries_so_far={getattr(self, 'total_retries', 0)}, "
                                f"qid={getattr(self, '_current_qid', '?')!r}, "
                                f"model_type={model_type!r}, "
                                f"is_multimodal={self._is_multimodal_model()!r}, "
                                f"label_scheme={getattr(self, 'label_scheme', None)!r}, "
                                f"n_docs={len(docs)}). "
                                f"Falling back to safe default (best={fallback_best!r}, "
                                f"worst={fallback_worst!r}). "
                                f"Raw text prefix: {raw_output[:200]!r}\n"
                            )
                            best = fallback_best
                            worst = fallback_worst
                            parse_status = "strict_parse_fallback"
                            parse_fallback_reason = "strict_parse_fallback"
                            comparison_raw_output = raw_output
                            break
                        self.total_parse_failure_strict = snapshot.total_parse_failure_strict
                        self.total_parse_fallback = getattr(self, "total_parse_fallback", 0) + 1
                        counter_name = f"total_{reason}_fallback"
                        setattr(self, counter_name, getattr(self, counter_name, 0) + 1)
                        if self._score_order_fallback_active():
                            best, worst = self._score_order_fallback_dual_labels(docs)
                        elif reason == "duplicate_label" and duplicate_label in self.CHARACTERS[:len(docs)]:
                            best = duplicate_label
                            labels = self.CHARACTERS[:len(docs)]
                            worst = labels[0] if best == labels[-1] else labels[-1]
                        else:
                            best = self.CHARACTERS[0]
                            worst = self.CHARACTERS[len(docs) - 1]
                        status_suffix = "score_fallback" if self._score_order_fallback_active() else "noop"
                        print(
                            f"[MaxContext] dual {reason} {status_suffix} "
                            f"(best={best}, worst={worst}). Raw: {raw_output!r}"
                        )
                        parse_status = f"{reason}_{status_suffix}"
                        parse_fallback_reason = reason
                        comparison_raw_output = raw_output
                        break

        elif self.scoring == 'likelihood':
            # Explicit likelihood mode uses the same single-forward-pass shortcut
            # as the T5 generation path above. We score the best-only prompt once,
            # then reuse argmax/argmin as the DualEnd heuristic.
            #
            # Important: this is a best-only proxy for DualEnd, not an exact
            # likelihood model of the full "Best: X, Worst: Y" output string.
            likelihood_text = self._build_best_prompt(query, docs)
            scores = self._score_label_candidates(likelihood_text, len(docs))
            ranked = sorted(
                zip(self.CHARACTERS[:len(docs)], scores),
                key=lambda x: x[1],
                reverse=True,
            )
            best = ranked[0][0]
            worst = ranked[-1][0]

        # Safety check: ensure best and worst are different
        if best == worst:
            if self._score_order_fallback_active():
                best, worst = self._score_order_fallback_dual_labels(docs)
            elif self.strict_no_parse_fallback:
                self.total_strict_parse_fallback = getattr(
                    self, "total_strict_parse_fallback", 0
                ) + 1
                import sys as _sys
                raw_prefix = (latest_raw_output or comparison_raw_output or "")[:200]
                _sys.stderr.write(
                    f"[MaxContext] strict duplicate-label fallback after {strict_raise_attempt} retries "
                    f"(budget={strict_raise_budget}, total_retries_so_far={getattr(self, 'total_retries', 0)}, "
                    f"qid={getattr(self, '_current_qid', '?')!r}). "
                    f"Keeping best={best!r}, salvaging worst. "
                    f"Raw text prefix: {raw_prefix!r}\n"
                )
            else:
                print(f"Warning: best and worst are the same ({best}), defaulting worst to last character")
            for c in reversed(self.CHARACTERS[:len(docs)]):
                if c != best:
                    worst = c
                    break

        self._log_comparison(
            "dual_best", self.CHARACTERS[:len(docs)], best, docs,
            parse_status=parse_status,
            parse_fallback_reason=parse_fallback_reason,
            raw_output=comparison_raw_output,
        )
        self._log_comparison(
            "dual_worst", self.CHARACTERS[:len(docs)], worst, docs,
            parse_status=parse_status,
            parse_fallback_reason=parse_fallback_reason,
            raw_output=comparison_raw_output,
        )

        return best, worst

    def _compare_worst_single(self, query: str, docs: List) -> str:
        """Fallback path for models that fail to emit both labels reliably."""
        self.total_compare += 1
        input_text = self._build_worst_prompt(query, docs)

        if self.scoring == 'likelihood':
            scores = self._score_label_candidates(input_text, len(docs))
            ranked = sorted(
                zip(self.CHARACTERS[:len(docs)], scores),
                key=lambda x: x[1],
                reverse=True,
            )
            output = ranked[0][0]
        elif self.config.model_type == 't5':
            inputs = self._tokenize_inputs(input_text)
            self.total_prompt_tokens += inputs.input_ids.shape[1]
            output_ids = self._generate(
                inputs,
                max_new_tokens=2,
                decoder_input_ids=self.decoder_input_ids,
            )[0]
            self.total_completion_tokens += output_ids.shape[0]
            raw_output = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            output = self._parse_single_label(raw_output, self.CHARACTERS[:len(docs)])
            if output is None:
                output = self._clean_generation_output(raw_output).upper()
        elif self._uses_chat_template():
            conversation = [{"role": "user", "content": input_text}]
            prompt = self._build_chat_prompt(conversation)
            prompt += " Passage:"
            inputs = self._tokenize_inputs(prompt)
            self.total_prompt_tokens += inputs.input_ids.shape[1]
            max_new = self._generation_budget("single")
            output_ids = self._generate(inputs, max_new_tokens=max_new)[0]
            self.total_completion_tokens += output_ids.shape[0] - inputs.input_ids.shape[1]
            raw_output = self.tokenizer.decode(output_ids[inputs.input_ids.shape[1]:], skip_special_tokens=False).strip()
            output = self._parse_single_label(raw_output, self.CHARACTERS[:len(docs)])
            if output is None:
                output = self._clean_generation_output(raw_output).upper()
        else:
            raise NotImplementedError

        if output:
            for char in reversed(output):
                if char in self.CHARACTERS[:len(docs)]:
                    self._log_comparison("worst", self.CHARACTERS[:len(docs)], char, docs)
                    return char
        return self.CHARACTERS[len(docs) - 1]

    def _num_to_label(self, num: int, n_docs: int) -> Optional[str]:
        """Convert a 1-based number to the corresponding passage label."""
        idx = num - 1  # 1-based → 0-based
        if 0 <= idx < n_docs:
            return self.CHARACTERS[idx]
        return None

    def _try_parse_dual_output(self, output: str, n_docs: int) -> Optional[Tuple[str, str]]:
        """Parse the dual output from the LLM.

        Handles various output formats:
        - "Best: A, Worst: C"      (letter labels)
        - "Best: [A], Worst: [C]"  (bracketed labels)
        - "Best: 1, Worst: 3"      (numeric labels)
        - "A, C"                   (bare letters)
        - "1, 3"                   (bare numbers)
        - "Passage A, Passage C"
        """
        output_upper = self._clean_generation_output(output).upper().strip()
        valid_chars = set(self.CHARACTERS[:n_docs])
        strict = getattr(self, "strict_no_parse_fallback", False)
        is_numeric_scheme = getattr(self, "label_scheme", None) == "numeric_1_based"
        is_multimodal = getattr(self, "_is_multimodal", False)
        model_type = getattr(getattr(self, "config", None), "model_type", None)

        # Label pattern: bare letter or letter in square brackets, e.g. A, [A]
        _L = r'\[?([A-W])\]?'
        invalid_pair = object()

        def validate_pair(best: Optional[str], worst: Optional[str]):
            if best in valid_chars and worst in valid_chars:
                return best, worst
            if strict and (best is not None or worst is not None):
                return invalid_pair
            return None

        # --- Pattern 0 (multimodal-only): LAST occurrence of BEST/WORST + digit ---
        # Verbose multimodal chat models (e.g. Mistral 3) often emit a
        # chain-of-thought prelude where "Best: None of the passages..." appears
        # before the actual answer. The clean answer arrives at the END of the
        # output, often with parens or other prose between BEST/WORST and the
        # digit (e.g. "Best (closest to anthropological environment): 15"). The
        # lazy `.{0,200}?` separator tolerates that, and `re.finditer` + last
        # match prefers the trailing summary. Multimodal-gated to preserve Qwen
        # 3 byte-equality.
        if is_multimodal and is_numeric_scheme:
            best_iter = list(re.finditer(r'\bBEST\b.{0,200}?(\d+)', output_upper))
            worst_iter = list(re.finditer(r'\bWORST\b.{0,200}?(\d+)', output_upper))
            if best_iter and worst_iter:
                parsed = validate_pair(
                    self._num_to_label(int(best_iter[-1].group(1)), n_docs),
                    self._num_to_label(int(worst_iter[-1].group(1)), n_docs),
                )
                if parsed is invalid_pair:
                    return None
                if parsed is not None:
                    return parsed

        if not is_numeric_scheme:
            # --- Pattern 1: "Best: X, Worst: Y" with letter labels ---
            best_match = re.search(r'BEST[:\s]*(?:PASSAGE\s*)?' + _L, output_upper)
            worst_match = re.search(r'WORST[:\s]*(?:PASSAGE\s*)?' + _L, output_upper)
            if best_match or worst_match:
                if not (best_match and worst_match):
                    if strict:
                        return None
                else:
                    parsed = validate_pair(best_match.group(1), worst_match.group(1))
                    if parsed is invalid_pair:
                        return None
                    if parsed is not None:
                        return parsed

        # --- Pattern 2: "Best: N, Worst: M" with numeric labels (1-based) ---
        best_num_match = re.search(r'BEST[:\s]*(?:PASSAGE\s*)?(\d+)', output_upper)
        worst_num_match = re.search(r'WORST[:\s]*(?:PASSAGE\s*)?(\d+)', output_upper)
        if best_num_match or worst_num_match:
            if not (best_num_match and worst_num_match):
                # Strict early-return for non-multimodal models only — verbose
                # multimodal output (e.g. Mistral 3 partial match where
                # "Best: None..." prevents BEST capture but "Worst: 12" matches)
                # should fall through to Pattern 6 instead of failing loud.
                # Qwen 3 byte-equality preserved (non-multimodal still strict).
                if strict and not is_multimodal and model_type in QWEN_MODEL_TYPES:
                    return None
            else:
                parsed = validate_pair(
                    self._num_to_label(int(best_num_match.group(1)), n_docs),
                    self._num_to_label(int(worst_num_match.group(1)), n_docs),
                )
                if parsed is invalid_pair:
                    return None
                if parsed is not None:
                    return parsed

        if not is_numeric_scheme:
            # --- Pattern 3: "most relevant: X ... least relevant: Y" ---
            most_match = re.search(r'MOST\s+RELEVANT[:\s]*(?:PASSAGE\s*)?' + _L, output_upper)
            least_match = re.search(r'LEAST\s+RELEVANT[:\s]*(?:PASSAGE\s*)?' + _L, output_upper)
            if most_match or least_match:
                if not (most_match and least_match):
                    if strict:
                        return None
                else:
                    parsed = validate_pair(most_match.group(1), least_match.group(1))
                    if parsed is invalid_pair:
                        return None
                    if parsed is not None:
                        return parsed

            # --- Pattern 4: "Passage X" patterns (at least 2) ---
            passage_matches = re.findall(r'PASSAGE\s*' + _L, output_upper)
            passage_matches = [c for c in passage_matches if c in valid_chars]
            if passage_matches:
                if len(passage_matches) < 2:
                    if strict:
                        return None
                else:
                    parsed = validate_pair(passage_matches[0], passage_matches[1])
                    if parsed is invalid_pair:
                        return None
                    if parsed is not None:
                        return parsed

            # --- Pattern 5: comma/space-separated letters or bracketed letters ---
            parts = re.split(r'[,\s]+', output_upper)
            found_chars = []
            for p in parts:
                m = re.fullmatch(r'\[?([A-W])\]?', p)
                if m and m.group(1) in valid_chars:
                    found_chars.append(m.group(1))
            if found_chars:
                if len(found_chars) < 2:
                    if strict:
                        return None
                else:
                    parsed = validate_pair(found_chars[0], found_chars[1])
                    if parsed is invalid_pair:
                        return None
                    if parsed is not None:
                        return parsed

        # --- Pattern 6: two distinct numbers (1-based) anywhere in the output ---
        all_nums = re.findall(r'\b(\d+)\b', output_upper)
        found_num_labels = []
        for n in all_nums:
            label = self._num_to_label(int(n), n_docs)
            if label is not None and (not found_num_labels or label != found_num_labels[-1]):
                found_num_labels.append(label)
            if len(found_num_labels) >= 2:
                parsed = validate_pair(found_num_labels[0], found_num_labels[1])
                if parsed is invalid_pair:
                    return None
                if parsed is not None:
                    return parsed

        if strict and all_nums:
            return None

        return None

    def _parse_dual_output(self, output: str, n_docs: int) -> Tuple[str, str]:
        """Parse dual output with guaranteed return — never returns None.

        Falls back to heuristics if _try_parse_dual_output fails:
        1. If one letter found: use it as best, default worst
        2. If one number found: map to label, default the other
        3. Last resort: default to first and last
        """
        parsed = self._try_parse_dual_output(output, n_docs)
        if parsed is not None:
            return parsed

        if getattr(self, 'strict_no_parse_fallback', False):
            self.total_parse_failure_strict = getattr(
                self, "total_parse_failure_strict", 0
            ) + 1
            raise ValueError(
                f"MaxContext dual-output parse failed. Raw text: {output!r}"
            )

        cleaned = self._clean_generation_output(output)
        output_upper = cleaned.upper().strip()
        valid_chars = set(self.CHARACTERS[:n_docs])

        # Try finding any valid letter characters
        all_found = [c for c in output_upper if c in valid_chars]
        if len(all_found) >= 2 and all_found[0] != all_found[1]:
            print(f"Warning: Partial dual parse from '{cleaned}', using {all_found[0]} as best, {all_found[1]} as worst")
            return all_found[0], all_found[1]
        if len(all_found) >= 1:
            best = all_found[0]
            worst = self.CHARACTERS[n_docs - 1] if best != self.CHARACTERS[n_docs - 1] else self.CHARACTERS[0]
            print(f"Warning: Could only parse one label from '{cleaned}', using {best} as best, {worst} as worst")
            return best, worst

        # Try numeric: find any number that maps to a valid label
        num_match = re.search(r'\b(\d+)\b', cleaned)
        if num_match:
            label = self._num_to_label(int(num_match.group(1)), n_docs)
            if label is not None:
                worst = self.CHARACTERS[n_docs - 1] if label != self.CHARACTERS[n_docs - 1] else self.CHARACTERS[0]
                print(f"Warning: Numeric-only dual parse from '{cleaned}', using {label} as best, {worst} as worst")
                return label, worst

        # Last resort: default to first and last
        self.total_lenient_fallback = getattr(self, "total_lenient_fallback", 0) + 1
        print(f"Warning: Could not parse dual output: '{cleaned}', defaulting to A and {self.CHARACTERS[n_docs-1]}")
        return self.CHARACTERS[0], self.CHARACTERS[n_docs - 1]

    def _majority_vote(self, candidates: List[int]) -> int:
        """Return the most common candidate index via majority voting."""
        candidate_counts = Counter(candidates)
        max_count = max(candidate_counts.values())
        most_common = [c for c, count in candidate_counts.items() if count == max_count]
        return most_common[0] if len(most_common) == 1 else random.choice(most_common)

    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
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
            # Use standard heapsort from parent (dual-end doesn't map cleanly to heap)
            self.heapSort(ranking, query, self.k)
            ranking = list(reversed(ranking))

        elif self.method == "bubblesort":
            # Cocktail shaker sort: properly bidirectional
            self._cocktail_shaker_sort(ranking, query, self.k)

        elif self.method == "selection":
            # Double-ended selection sort
            self._double_ended_selection(ranking, query, self.k)

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

    def _cocktail_shaker_sort(self, ranking, query, k):
        """Cocktail shaker sort (bidirectional bubblesort).

        Alternates between:
        1. Backward pass (bottom→top): Compare windows, move best to front and worst to back
        2. Forward pass (top→bottom): Compare windows, move worst to back and best to front

        Each comparison extracts both best and worst from the window, getting 2x
        information per LLM call compared to standard bubblesort.
        """
        n = len(ranking)
        top_sorted = 0  # Number of positions sorted at the top
        bottom_boundary = n  # Upper bound of unsorted region

        while top_sorted < k:
            # === Backward pass (bottom → top): bubble best to position top_sorted ===
            start_ind = bottom_boundary - (self.num_child + 1)
            if start_ind < top_sorted:
                start_ind = top_sorted
            end_ind = start_ind + self.num_child + 1
            if end_ind > bottom_boundary:
                end_ind = bottom_boundary

            worst_sunk = False
            while True:
                if start_ind < top_sorted:
                    start_ind = top_sorted
                window = ranking[start_ind:end_ind]
                if len(window) < 2:
                    break

                best_label, worst_label = self._compare_both_window(query, ranking, start_ind, end_ind)

                best_ind = min(self.CHARACTERS.index(best_label), len(window) - 1) if best_label in self.CHARACTERS else 0
                worst_ind = min(self.CHARACTERS.index(worst_label), len(window) - 1) if worst_label in self.CHARACTERS else len(window) - 1

                # Handle the case where best and worst swap with each other
                if best_ind != 0 or worst_ind != len(window) - 1:
                    # Strategy: first note both positions, then perform swaps carefully
                    actual_best = start_ind + best_ind
                    actual_worst = start_ind + worst_ind
                    front = start_ind
                    back = start_ind + len(window) - 1

                    if actual_best == back and actual_worst == front:
                        # Simple swap: best is at back, worst is at front
                        ranking[front], ranking[back] = ranking[back], ranking[front]
                    else:
                        # Move best to front first
                        if actual_best != front:
                            ranking[front], ranking[actual_best] = ranking[actual_best], ranking[front]
                            # Update worst index if it was at front (now moved to actual_best)
                            if actual_worst == front:
                                actual_worst = actual_best

                        # Move worst to back
                        if actual_worst != back and actual_worst != front:
                            ranking[back], ranking[actual_worst] = ranking[actual_worst], ranking[back]
                            if not worst_sunk:
                                worst_sunk = True

                if start_ind == top_sorted:
                    break

                start_ind -= self.num_child
                end_ind -= self.num_child
                if end_ind < start_ind + 2:
                    break

            top_sorted += 1
            if worst_sunk and bottom_boundary > top_sorted + 1:
                bottom_boundary -= 1

            if top_sorted >= k:
                break

            # === Forward pass (top → bottom): sink worst to bottom_boundary-1 ===
            start_ind = top_sorted
            end_ind = min(start_ind + self.num_child + 1, bottom_boundary)

            best_bubbled = False
            while True:
                window = ranking[start_ind:end_ind]
                if len(window) < 2:
                    break

                best_label, worst_label = self._compare_both_window(query, ranking, start_ind, end_ind)

                best_ind = min(self.CHARACTERS.index(best_label), len(window) - 1) if best_label in self.CHARACTERS else 0
                worst_ind = min(self.CHARACTERS.index(worst_label), len(window) - 1) if worst_label in self.CHARACTERS else len(window) - 1

                actual_best = start_ind + best_ind
                actual_worst = start_ind + worst_ind
                front = start_ind
                back = start_ind + len(window) - 1

                if actual_best == back and actual_worst == front:
                    ranking[front], ranking[back] = ranking[back], ranking[front]
                else:
                    # Move worst to back first (primary goal of forward pass)
                    if actual_worst != back:
                        ranking[back], ranking[actual_worst] = ranking[actual_worst], ranking[back]
                        if actual_best == back:
                            actual_best = actual_worst

                    # Move best to front (bonus)
                    if actual_best != front and actual_best != back:
                        ranking[front], ranking[actual_best] = ranking[actual_best], ranking[front]
                        if not best_bubbled:
                            best_bubbled = True

                start_ind += self.num_child
                end_ind = min(start_ind + self.num_child + 1, bottom_boundary)
                if start_ind >= bottom_boundary:
                    break

            if bottom_boundary > top_sorted + 1:
                bottom_boundary -= 1
            # If best also bubbled up during forward pass, we get a bonus top position
            # (but don't double-count — the backward pass in next iteration will verify)

    def _double_ended_selection(self, ranking, query, k):
        """Double-ended selection sort: extract best and worst per round.

        In each round:
        1. Run a tournament among groups of num_child+1 documents
        2. Compare group winners to find overall best
        3. Compare group losers to find overall worst
        4. Place best at next top position, worst at next bottom position
        5. Continue until k top positions are filled
        """
        n = len(ranking)
        top_idx = 0  # Next top position to fill
        bottom_idx = n - 1  # Next bottom position to fill

        while top_idx < k and top_idx < bottom_idx:
            unsorted_len = bottom_idx - top_idx + 1

            if unsorted_len <= 1:
                break

            if unsorted_len <= self.num_child + 1:
                # Can compare all remaining in one call
                window = ranking[top_idx:bottom_idx + 1]
                best_label, worst_label = self._compare_both_window(query, ranking, top_idx, bottom_idx + 1)
                best_pos = min(self.CHARACTERS.index(best_label), len(window) - 1) if best_label in self.CHARACTERS else 0
                worst_pos = min(self.CHARACTERS.index(worst_label), len(window) - 1) if worst_label in self.CHARACTERS else len(window) - 1

                # Convert to absolute indices
                abs_best = top_idx + best_pos
                abs_worst = top_idx + worst_pos

                # Perform swaps carefully to avoid interference
                if abs_best == bottom_idx and abs_worst == top_idx:
                    # Simple swap
                    ranking[top_idx], ranking[bottom_idx] = ranking[bottom_idx], ranking[top_idx]
                else:
                    # Move best to top_idx
                    if abs_best != top_idx:
                        ranking[top_idx], ranking[abs_best] = ranking[abs_best], ranking[top_idx]
                        # Track worst: if worst was at top_idx, it's now at abs_best
                        if abs_worst == top_idx:
                            abs_worst = abs_best

                    # Move worst to bottom_idx
                    if abs_worst != bottom_idx and abs_worst > top_idx:
                        ranking[bottom_idx], ranking[abs_worst] = ranking[abs_worst], ranking[bottom_idx]

                top_idx += 1
                bottom_idx -= 1
            else:
                # Tournament: compare in groups, find overall best and worst
                if (
                    getattr(self, "_maxcontext_pool_size", None) is not None
                    and (getattr(self, "shuffle", False) or getattr(self, "reverse", False))
                ):
                    raise RuntimeError(
                        "MaxContext shuffle/reverse requires the full-window "
                        "DualEnd branch; tournament selection is unreachable "
                        "under MaxContext invariants."
                    )
                group_bests = []  # (absolute_index, doc)
                group_worsts = []  # (absolute_index, doc)

                for g_start in range(top_idx, bottom_idx + 1, self.num_child + 1):
                    g_end = min(g_start + self.num_child + 1, bottom_idx + 1)
                    group = ranking[g_start:g_end]
                    if len(group) < 2:
                        group_bests.append((g_start, group[0]))
                        continue

                    best_label, worst_label = self._compare_both_window(query, ranking, g_start, g_end)
                    b_idx = min(self.CHARACTERS.index(best_label), len(group) - 1) if best_label in self.CHARACTERS else 0
                    w_idx = min(self.CHARACTERS.index(worst_label), len(group) - 1) if worst_label in self.CHARACTERS else len(group) - 1

                    group_bests.append((g_start + b_idx, ranking[g_start + b_idx]))
                    group_worsts.append((g_start + w_idx, ranking[g_start + w_idx]))

                # Find overall best from group winners
                if len(group_bests) > 1:
                    winner_docs_list = [item[1] for item in group_bests]
                    # Compare group winners to find the overall best
                    # May need multiple rounds if more than num_child+1 winners
                    overall_best_tournament_idx = self._tournament_select_best(query, winner_docs_list)
                    overall_best_abs = group_bests[overall_best_tournament_idx][0]
                else:
                    overall_best_abs = group_bests[0][0]

                # Find overall worst from group losers
                if len(group_worsts) > 1:
                    loser_docs_list = [item[1] for item in group_worsts]
                    overall_worst_tournament_idx = self._tournament_select_worst(query, loser_docs_list)
                    overall_worst_abs = group_worsts[overall_worst_tournament_idx][0]
                elif len(group_worsts) == 1:
                    overall_worst_abs = group_worsts[0][0]
                else:
                    overall_worst_abs = bottom_idx

                # Place best at top_idx and worst at bottom_idx
                if overall_best_abs == bottom_idx and overall_worst_abs == top_idx:
                    ranking[top_idx], ranking[bottom_idx] = ranking[bottom_idx], ranking[top_idx]
                else:
                    if overall_best_abs != top_idx:
                        ranking[top_idx], ranking[overall_best_abs] = ranking[overall_best_abs], ranking[top_idx]
                        if overall_worst_abs == top_idx:
                            overall_worst_abs = overall_best_abs

                    if overall_worst_abs != bottom_idx and overall_worst_abs > top_idx:
                        ranking[bottom_idx], ranking[overall_worst_abs] = ranking[overall_worst_abs], ranking[bottom_idx]

                top_idx += 1
                bottom_idx -= 1

    def _tournament_select_best(self, query, docs_list):
        """Run a tournament to find the best document from a list. Returns index in docs_list."""
        if len(docs_list) <= self.num_child + 1:
            best_label = self.compare(query, docs_list)
            try:
                return min(self.CHARACTERS.index(best_label), len(docs_list) - 1)
            except ValueError:
                return 0

        # Multiple rounds needed
        current_indices = list(range(len(docs_list)))
        while len(current_indices) > 1:
            next_round = []
            for g in range(0, len(current_indices), self.num_child + 1):
                group_indices = current_indices[g:g + self.num_child + 1]
                if len(group_indices) == 1:
                    next_round.append(group_indices[0])
                    continue
                group_docs = [docs_list[idx] for idx in group_indices]
                best_label = self.compare(query, group_docs)
                try:
                    winner = min(self.CHARACTERS.index(best_label), len(group_docs) - 1)
                except ValueError:
                    winner = 0
                next_round.append(group_indices[winner])
            current_indices = next_round
        return current_indices[0]

    def _tournament_select_worst(self, query, docs_list):
        """Run a tournament to find the worst document from a list. Returns index in docs_list."""
        if len(docs_list) <= self.num_child + 1:
            # Delegate to compare_worst() so tournament worst-comparisons share the
            # same parsing, accounting, and position-bias logging path as the main
            # BottomUp / DualEnd worst-selection calls.
            output = self.compare_worst(query, docs_list)
            try:
                return min(self.CHARACTERS.index(output.upper()), len(docs_list) - 1)
            except ValueError:
                return len(docs_list) - 1

        # Multiple rounds
        current_indices = list(range(len(docs_list)))
        while len(current_indices) > 1:
            next_round = []
            for g in range(0, len(current_indices), self.num_child + 1):
                group_indices = current_indices[g:g + self.num_child + 1]
                if len(group_indices) == 1:
                    next_round.append(group_indices[0])
                    continue
                group_docs = [docs_list[idx] for idx in group_indices]
                worst_idx = self._tournament_select_worst(query, group_docs)
                next_round.append(group_indices[worst_idx])
            current_indices = next_round
        return current_indices[0]


class MaxContextDualEndSetwiseLlmRanker(_MaxContextOrderingMixin, DualEndSetwiseLlmRanker):
    _MAXCONTEXT_NAME_FRAGMENTS = frozenset({
        "qwen3", "qwen3.5", "qwen3_5",
        "llama-3.1", "llama_3_1", "llama3.1", "meta-llama-3.1",
        "ministral-3", "ministral_3", "ministral3",
    })

    def __init__(self, *args, pool_size: int, shuffle: bool = False, reverse: bool = False,
                 allow_parse_failure_bm25_fallback: bool = False, **kwargs):
        self._early_reject_unsupported_family(
            kwargs.get("model_name_or_path") or (args[0] if args else None)
        )
        super().__init__(*args, **kwargs)
        if shuffle and reverse:
            raise ValueError("shuffle and reverse are mutually exclusive")
        self.shuffle = shuffle
        self.reverse = reverse
        self._assert_maxcontext_invariants(pool_size)
        _setup_maxcontext_numeric_attrs(self, pool_size)
        self._allow_parse_failure_bm25_fallback = allow_parse_failure_bm25_fallback

    @staticmethod
    def _early_reject_unsupported_family(model_name: Optional[str]) -> None:
        if not model_name:
            return
        lowered = model_name.lower()
        if any(
            fragment in lowered
            for fragment in MaxContextDualEndSetwiseLlmRanker._MAXCONTEXT_NAME_FRAGMENTS
        ):
            return
        raise ValueError(
            "MaxContext supports Qwen3 / Qwen3.5 / Llama-3.1 / Ministral-3 only; "
            f"got {model_name!r}. "
            "(Qwen2, Llama-2, Mistral-7B/Nemo/Small are explicitly not supported.)"
        )

    def _assert_maxcontext_invariants(self, pool_size: int) -> None:
        if self.config.model_type not in MAXCONTEXT_ALLOWED_MODEL_TYPES:
            raise ValueError(
                f"MaxContextDualEnd requires Qwen3 / Qwen3.5 / Llama-3.1 / Ministral-3 model_type; "
                f"got {self.config.model_type!r}."
            )
        if self.scoring != "generation":
            raise ValueError("MaxContextDualEnd requires --scoring generation.")
        if self.k != pool_size:
            raise ValueError(f"pool_size={pool_size} but ranker.k={self.k}.")
        if self.num_permutation != 1:
            raise ValueError(
                "MaxContextDualEnd requires --num_permutation 1 "
                "(compare_both does not permute)."
            )
        if self.method != "selection":
            raise ValueError(
                "MaxContextDualEnd requires method='selection' "
                "(_double_ended_selection is the only supported algorithm)."
            )

    def rerank(self, query: str, docs: List[SearchResult]) -> List[SearchResult]:
        _reset_maxcontext_query_stats(self)
        self._maxcontext_original_positions = {
            doc.docid: i for i, doc in enumerate(docs)
        }
        _validate_maxcontext_pool_size(self, docs, "MaxContextDualEnd")
        if len(docs) <= 1:
            return _materialize_maxcontext_results(docs)
        self._assert_maxcontext_fits(query, docs)
        self.total_parse_fallback = 0
        self.total_lexical_refusal_fallback = 0
        self.total_numeric_out_of_range_fallback = 0
        self.total_degenerate_repetition_fallback = 0
        self.total_unparseable_after_exhaustion_fallback = 0
        self.total_duplicate_label_fallback = 0
        self.total_lenient_fallback = 0
        self.total_strict_parse_fallback = 0
        try:
            return super().rerank(query, docs)
        except ValueError as exc:
            if not _maxcontext_should_fallback_to_bm25(self, exc):
                raise
            self.total_parse_failure_bm25_fallback = getattr(
                self, "total_parse_failure_bm25_fallback", 0
            ) + 1
            qid = getattr(self, "_current_qid", "?")
            print(
                f"[MaxContext] parse_failure_bm25_fallback for query {qid!r}; "
                f"using first-stage (BM25) score ordering. Reason: {exc}"
            )
            ordered = self._score_order_sorted_docs(docs)
            return [
                SearchResult(docid=d.docid, score=-rank, text=None)
                for rank, d in enumerate(ordered, start=1)
            ]

    def _assert_maxcontext_fits(self, query: str, docs: List[SearchResult]) -> None:
        ok, rendered_length, limit = compute_max_fit_window(
            ranker=self,
            query=query,
            docs=docs,
            reserved_output_tokens=128,
        )
        if not ok:
            raise ValueError(
                f"MaxContextDualEnd preflight failed: rendered prompt is "
                f"{rendered_length} tokens but the budget is {limit} "
                f"(max_input_tokens - reserved_output_tokens). "
                "Reduce --passage_length or --k, or pick a Qwen3.5 variant with larger context."
            )

    def _compare_both_window(
        self,
        query: str,
        ranking: List[SearchResult],
        start_ind: int,
        end_ind: int,
    ) -> Tuple[str, str]:
        original_window = ranking[start_ind:end_ind]
        presented = self._apply_pool_ordering(original_window)
        best_label, worst_label = self.compare_both(query, presented)
        if presented is original_window:
            return best_label, worst_label
        best_idx = self._remap_label_to_original(
            best_label, presented, original_window, default=0
        )
        worst_idx = self._remap_label_to_original(
            worst_label, presented, original_window, default=len(original_window) - 1
        )
        return self.CHARACTERS[best_idx], self.CHARACTERS[worst_idx]


class SelectiveDualEndSetwiseLlmRanker(_DualEndRoutingMixin, DualEndSetwiseLlmRanker):
    """Top-down setwise ranking that upgrades selected windows to joint best-worst prompts.

    The core idea is to keep the cheaper TopDown sorting procedure, but to replace the
    single best-selection prompt with a DualEnd prompt only on windows that are likely
    to be ambiguous or close to the top-k decision boundary.
    """

    def __init__(
        self,
        *args,
        gate_strategy: str = "hybrid",
        shortlist_size: int = 20,
        margin_threshold: float = 0.15,
        uncertainty_percentile: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._init_joint_signal_routing(
            gate_strategy=gate_strategy,
            shortlist_size=shortlist_size,
            margin_threshold=margin_threshold,
            uncertainty_percentile=uncertainty_percentile,
        )

    def _select_best_window(self, query: str, ranking: List[SearchResult], start_ind: int, end_ind: int) -> str:
        window = ranking[start_ind:end_ind]
        if len(window) < 2:
            return self.CHARACTERS[0]

        if self._should_use_joint_signal(start_ind, end_ind, len(ranking), window):
            self.total_dual_invocations += 1
            best_label, _ = self.compare_both(query, window)
            return best_label

        self.total_single_invocations += 1
        return self.compare(query, window)

    def _heapify_selective(self, arr: List[SearchResult], n: int, i: int, query: str) -> None:
        if self.num_child * i + 1 >= n:
            return

        child_end = min((self.num_child * (i + 1) + 1), n)
        docs = [arr[i]] + arr[self.num_child * i + 1:child_end]
        inds = [i] + list(range(self.num_child * i + 1, child_end))
        # Heap node indices are not meaningful rank positions, so shortlist
        # routing is disabled for heapsort. Uncertainty routing still applies.
        if self._should_use_joint_signal(i, child_end, n, docs, allow_shortlist=False):
            self.total_dual_invocations += 1
            label, _ = self.compare_both(query, docs)
        else:
            self.total_single_invocations += 1
            label = self.compare(query, docs)
        best_ind = self._resolve_label_index(label, len(docs), 0)
        largest = inds[best_ind]

        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self._heapify_selective(arr, n, largest, query)

    def _heap_sort_selective(self, arr: List[SearchResult], query: str) -> None:
        n = len(arr)
        ranked = 0
        for i in range(n // self.num_child, -1, -1):
            self._heapify_selective(arr, n, i, query)
        for i in range(n - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            ranked += 1
            if ranked == self.k:
                break
            self._heapify_selective(arr, i, 0, query)

    def _bubble_sort_selective(self, ranking: List[SearchResult], query: str) -> None:
        last_start = len(ranking) - (self.num_child + 1)

        for i in range(self.k):
            if last_start < i:
                last_start = i
            start_ind = last_start
            end_ind = min(last_start + (self.num_child + 1), len(ranking))
            is_change = False
            while True:
                if start_ind < i:
                    start_ind = i
                if end_ind - start_ind < 2:
                    break

                output = self._select_best_window(query, ranking, start_ind, end_ind)
                best_ind = self._resolve_label_index(output, len(ranking[start_ind:end_ind]), 0)
                if best_ind != 0:
                    ranking[start_ind], ranking[start_ind + best_ind] = ranking[start_ind + best_ind], ranking[start_ind]
                    if not is_change:
                        is_change = True
                        if last_start != len(ranking) - (self.num_child + 1) and best_ind == len(ranking[start_ind:end_ind]) - 1:
                            last_start += len(ranking[start_ind:end_ind]) - 1

                if start_ind == i:
                    break

                if not is_change:
                    last_start -= self.num_child

                start_ind -= self.num_child
                end_ind -= self.num_child

    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        original_ranking = copy.deepcopy(ranking)
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        self.total_retries = 0
        self.total_retry_overhead_seconds = 0.0
        self._reset_joint_signal_stats()
        self._prepare_query_uncertainty_thresholds(original_ranking)

        if self.method == "heapsort":
            self._heap_sort_selective(ranking, query)
            ranking = list(reversed(ranking))
        elif self.method == "bubblesort":
            self._bubble_sort_selective(ranking, query)
        else:
            raise NotImplementedError(
                f"Selective DualEnd currently supports heapsort and bubblesort, got {self.method}."
            )

        results = []
        top_doc_ids = set()
        rank = 1

        for doc in ranking[:self.k]:
            top_doc_ids.add(doc.docid)
            results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
            rank += 1
        for doc in original_ranking:
            if doc.docid not in top_doc_ids:
                results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
                rank += 1

        return results


class BiasAwareDualEndSetwiseLlmRanker(_DualEndRoutingMixin, DualEndSetwiseLlmRanker):
    """DualEnd variant that uses a small set of controlled orderings on hard windows.

    This targets the measured position-bias asymmetry directly: only windows that are
    likely to be uncertain or near the top-k boundary receive extra order-robust calls.
    """

    def __init__(
        self,
        *args,
        gate_strategy: str = "hybrid",
        shortlist_size: int = 20,
        margin_threshold: float = 0.15,
        uncertainty_percentile: Optional[float] = None,
        order_robust_orderings: int = 3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.order_robust_orderings = max(1, order_robust_orderings)
        self._init_joint_signal_routing(
            gate_strategy=gate_strategy,
            shortlist_size=shortlist_size,
            margin_threshold=margin_threshold,
            uncertainty_percentile=uncertainty_percentile,
        )

    def _controlled_orderings(self, n_docs: int) -> List[List[int]]:
        base = list(range(n_docs))
        candidates = [base, list(reversed(base))]
        for shift in range(1, n_docs):
            candidates.append(base[shift:] + base[:shift])

        unique = []
        seen = set()
        for ordering in candidates:
            key = tuple(ordering)
            if key in seen:
                continue
            seen.add(key)
            unique.append(ordering)
            if len(unique) >= self.order_robust_orderings:
                break
        return unique

    def _compare_both_window(self, query: str, ranking: List[SearchResult], start_ind: int, end_ind: int) -> Tuple[str, str]:
        window = ranking[start_ind:end_ind]
        if len(window) < 2:
            return self.CHARACTERS[0], self.CHARACTERS[0]

        if not self._should_use_joint_signal(start_ind, end_ind, len(ranking), window):
            self.total_dual_invocations += 1
            return super()._compare_both_window(query, ranking, start_ind, end_ind)

        orderings = self._controlled_orderings(len(window))
        if len(orderings) <= 1:
            self.total_dual_invocations += 1
            return super()._compare_both_window(query, ranking, start_ind, end_ind)

        self.total_order_robust_windows += 1
        self.total_dual_invocations += len(orderings)
        self.total_extra_orderings += len(orderings) - 1

        best_votes = []
        worst_votes = []
        fallback_best = 0
        fallback_worst = len(window) - 1

        for idx, ordering in enumerate(orderings):
            permuted_docs = [window[pos] for pos in ordering]
            best_label, worst_label = super().compare_both(query, permuted_docs)
            best_original = self._remap_window_label(ordering, best_label, 0)
            worst_original = self._remap_window_label(ordering, worst_label, len(ordering) - 1)
            if idx == 0:
                fallback_best = best_original
                fallback_worst = worst_original
            best_votes.append(best_original)
            worst_votes.append(worst_original)

        best_index = self._majority_vote(best_votes)
        worst_index = self._majority_vote(worst_votes)
        if best_index == worst_index:
            best_index = fallback_best
            worst_index = fallback_worst
            if best_index == worst_index:
                worst_index = len(window) - 1 if best_index != len(window) - 1 else 0

        return self.CHARACTERS[best_index], self.CHARACTERS[worst_index]

    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        if self.method not in {"bubblesort", "selection"}:
            raise NotImplementedError(
                "Bias-aware DualEnd currently supports bubblesort and selection only; "
                "heapsort bypasses the order-robust joint prompting path."
            )

        self._reset_joint_signal_stats()
        self._prepare_query_uncertainty_thresholds(ranking)
        return super().rerank(query, ranking)


class SameCallRegularizedSetwiseLlmRanker(_DualEndRoutingMixin, DualEndSetwiseLlmRanker):
    """Top-down bubblesort regularized by the same-call worst signal.

    Unlike DualEnd-Cocktail, this method keeps a head-focused TopDown pass structure.
    The best output still drives promotions, while the worst output only acts as a
    local negative constraint by pushing a clearly bad candidate to the end of the
    current window.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_joint_signal_routing(
            gate_strategy="off",
            shortlist_size=0,
            margin_threshold=0.0,
            uncertainty_percentile=0.0,
        )

    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        if self.method != "bubblesort":
            raise NotImplementedError("Same-call regularization currently supports bubblesort only.")

        original_ranking = copy.deepcopy(ranking)
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        self.total_retries = 0
        self.total_retry_overhead_seconds = 0.0
        self._reset_joint_signal_stats()
        # Keep the current ranking head protected from the extra worst-signal
        # demotion. We protect the top-k plus one full active window because
        # candidates just beyond k can still be promoted into the final answer
        # by the next few TopDown comparisons.
        protected_frontier = min(len(ranking), self.k + self.num_child + 1)

        last_start = len(ranking) - (self.num_child + 1)

        for i in range(self.k):
            if last_start < i:
                last_start = i
            start_ind = last_start
            end_ind = min(last_start + (self.num_child + 1), len(ranking))
            is_change = False
            while True:
                if start_ind < i:
                    start_ind = i
                if end_ind - start_ind < 2:
                    break

                window = ranking[start_ind:end_ind]
                self.total_dual_invocations += 1
                best_label, worst_label = self.compare_both(query, window)

                best_ind = self._resolve_label_index(best_label, len(window), 0)
                worst_ind = self._resolve_label_index(worst_label, len(window), len(window) - 1)

                actual_best = start_ind + best_ind
                actual_worst = start_ind + worst_ind
                original_worst = actual_worst
                front = start_ind
                back = start_ind + len(window) - 1

                if actual_best == back and actual_worst == front:
                    ranking[front], ranking[back] = ranking[back], ranking[front]
                    self.total_regularized_worst_moves += 1
                    if not is_change:
                        is_change = True
                else:
                    if actual_best != front:
                        ranking[front], ranking[actual_best] = ranking[actual_best], ranking[front]
                        if actual_worst == front:
                            actual_worst = actual_best
                        if not is_change:
                            is_change = True
                            if last_start != len(ranking) - (self.num_child + 1) and best_ind == len(window) - 1:
                                last_start += len(window) - 1

                    if original_worst >= protected_frontier and actual_worst != back and actual_worst != front:
                        ranking[back], ranking[actual_worst] = ranking[actual_worst], ranking[back]
                        self.total_regularized_worst_moves += 1
                        is_change = True

                if start_ind == i:
                    break

                if not is_change:
                    last_start -= self.num_child

                start_ind -= self.num_child
                end_ind -= self.num_child

        results = []
        top_doc_ids = set()
        rank = 1

        for doc in ranking[:self.k]:
            top_doc_ids.add(doc.docid)
            results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
            rank += 1
        for doc in original_ranking:
            if doc.docid not in top_doc_ids:
                results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
                rank += 1

        return results


class BidirectionalEnsembleRanker(LlmRanker):
    """
    Bidirectional Ensemble Ranker: runs both standard (top-down) and bottom-up
    setwise ranking independently, then fuses the rankings.

    Fusion methods:
    - 'rrf': Reciprocal Rank Fusion
    - 'combsum': Normalized score summation
    - 'weighted': Weighted combination (alpha * topdown + (1-alpha) * bottomup)
    """

    def __init__(self,
                 model_name_or_path,
                 tokenizer_name_or_path,
                 device,
                 num_child=3,
                 k=10,
                 scoring='generation',
                 method='heapsort',
                 num_permutation=1,
                 fusion='rrf',
                 alpha=0.5,
                 cache_dir=None):

        self.k = k
        self.fusion = fusion
        self.alpha = alpha
        self.__comparison_log_path = None
        self.__current_qid = None

        # Create top-down ranker first
        self.topdown_ranker = SetwiseLlmRanker(
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            device=device,
            num_child=num_child,
            k=k,
            scoring=scoring,
            method=method,
            num_permutation=num_permutation,
            cache_dir=cache_dir
        )

        # Create bottom-up ranker that shares model weights with top-down
        # We avoid double model loading by creating a minimal instance and
        # copying the model reference
        self.bottomup_ranker = BottomUpSetwiseLlmRanker.__new__(BottomUpSetwiseLlmRanker)
        # Copy all attributes from topdown_ranker
        self.bottomup_ranker.device = self.topdown_ranker.device
        self.bottomup_ranker.num_child = self.topdown_ranker.num_child
        self.bottomup_ranker.num_permutation = self.topdown_ranker.num_permutation
        self.bottomup_ranker.k = self.topdown_ranker.k
        self.bottomup_ranker.config = self.topdown_ranker.config
        self.bottomup_ranker.tokenizer = self.topdown_ranker.tokenizer
        self.bottomup_ranker.llm = self.topdown_ranker.llm
        self.bottomup_ranker.scoring = self.topdown_ranker.scoring
        self.bottomup_ranker.method = self.topdown_ranker.method
        self.bottomup_ranker.total_compare = 0
        self.bottomup_ranker.total_completion_tokens = 0
        self.bottomup_ranker.total_prompt_tokens = 0
        self.bottomup_ranker.total_retries = 0
        self.bottomup_ranker.total_retry_overhead_seconds = 0.0
        if hasattr(self.topdown_ranker, 'decoder_input_ids'):
            self.bottomup_ranker.decoder_input_ids = self.topdown_ranker.decoder_input_ids
        if hasattr(self.topdown_ranker, 'dual_decoder_input_ids'):
            self.bottomup_ranker.dual_decoder_input_ids = self.topdown_ranker.dual_decoder_input_ids
        if hasattr(self.topdown_ranker, 'target_token_ids'):
            self.bottomup_ranker.target_token_ids = self.topdown_ranker.target_token_ids
        self.bottomup_ranker.max_input_tokens = self.topdown_ranker.max_input_tokens
        self.bottomup_ranker._warned_input_truncation = False

        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        self.total_retries = 0
        self.total_retry_overhead_seconds = 0.0

    @property
    def _comparison_log_path(self):
        return self.__comparison_log_path

    @_comparison_log_path.setter
    def _comparison_log_path(self, value):
        self.__comparison_log_path = value
        if hasattr(self, "topdown_ranker"):
            self.topdown_ranker._comparison_log_path = value
        if hasattr(self, "bottomup_ranker"):
            self.bottomup_ranker._comparison_log_path = value

    @property
    def _current_qid(self):
        return self.__current_qid

    @_current_qid.setter
    def _current_qid(self, value):
        self.__current_qid = value
        if hasattr(self, "topdown_ranker"):
            self.topdown_ranker._current_qid = value
        if hasattr(self, "bottomup_ranker"):
            self.bottomup_ranker._current_qid = value

    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        ranking_copy1 = copy.deepcopy(ranking)
        ranking_copy2 = copy.deepcopy(ranking)

        # Run top-down
        topdown_results = self.topdown_ranker.rerank(query, ranking_copy1)

        # Run bottom-up
        bottomup_results = self.bottomup_ranker.rerank(query, ranking_copy2)

        # Aggregate stats
        self.total_compare = self.topdown_ranker.total_compare + self.bottomup_ranker.total_compare
        self.total_prompt_tokens = self.topdown_ranker.total_prompt_tokens + self.bottomup_ranker.total_prompt_tokens
        self.total_completion_tokens = self.topdown_ranker.total_completion_tokens + self.bottomup_ranker.total_completion_tokens
        self.total_retries = (
            getattr(self.topdown_ranker, "total_retries", 0)
            + getattr(self.bottomup_ranker, "total_retries", 0)
        )
        self.total_retry_overhead_seconds = (
            getattr(self.topdown_ranker, "total_retry_overhead_seconds", 0.0)
            + getattr(self.bottomup_ranker, "total_retry_overhead_seconds", 0.0)
        )

        # Fuse rankings
        if self.fusion == 'rrf':
            return self._rrf_fusion(topdown_results, bottomup_results)
        elif self.fusion == 'combsum':
            return self._combsum_fusion(topdown_results, bottomup_results)
        elif self.fusion == 'weighted':
            return self._weighted_fusion(topdown_results, bottomup_results)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion}")

    def _rrf_fusion(self, results1: List[SearchResult], results2: List[SearchResult],
                    rrf_k: int = 60) -> List[SearchResult]:
        """Reciprocal Rank Fusion: score(d) = sum(1 / (k + rank_i(d)))"""
        scores = {}

        for rank, doc in enumerate(results1, 1):
            scores[doc.docid] = scores.get(doc.docid, 0) + 1.0 / (rrf_k + rank)

        for rank, doc in enumerate(results2, 1):
            scores[doc.docid] = scores.get(doc.docid, 0) + 1.0 / (rrf_k + rank)

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for rank, (docid, score) in enumerate(sorted_docs, 1):
            results.append(SearchResult(docid=docid, score=-rank, text=None))

        return results

    def _combsum_fusion(self, results1: List[SearchResult], results2: List[SearchResult]) -> List[SearchResult]:
        """CombSUM: normalize scores per run, then sum."""
        scores = {}

        n1 = len(results1)
        for rank, doc in enumerate(results1, 1):
            scores[doc.docid] = scores.get(doc.docid, 0) + (n1 - rank + 1) / n1

        n2 = len(results2)
        for rank, doc in enumerate(results2, 1):
            scores[doc.docid] = scores.get(doc.docid, 0) + (n2 - rank + 1) / n2

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for rank, (docid, score) in enumerate(sorted_docs, 1):
            results.append(SearchResult(docid=docid, score=-rank, text=None))

        return results

    def _weighted_fusion(self, results1: List[SearchResult], results2: List[SearchResult]) -> List[SearchResult]:
        """Weighted fusion: alpha * topdown_score + (1-alpha) * bottomup_score."""
        scores = {}

        n1 = len(results1)
        for rank, doc in enumerate(results1, 1):
            scores[doc.docid] = scores.get(doc.docid, 0) + self.alpha * (n1 - rank + 1) / n1

        n2 = len(results2)
        for rank, doc in enumerate(results2, 1):
            scores[doc.docid] = scores.get(doc.docid, 0) + (1 - self.alpha) * (n2 - rank + 1) / n2

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for rank, (docid, score) in enumerate(sorted_docs, 1):
            results.append(SearchResult(docid=docid, score=-rank, text=None))

        return results

    def truncate(self, text, length):
        return self.topdown_ranker.truncate(text, length)


class MaxContextTopDownSetwiseLlmRanker(_MaxContextOrderingMixin, SetwiseLlmRanker):
    """MaxContext-style best-only selection over the full pool."""

    total_bm25_bypass: int = 0

    def __init__(self, *args, pool_size: int, shuffle: bool = False, reverse: bool = False,
                 allow_parse_failure_bm25_fallback: bool = False, **kwargs):
        MaxContextDualEndSetwiseLlmRanker._early_reject_unsupported_family(
            kwargs.get("model_name_or_path") or (args[0] if args else None)
        )
        super().__init__(*args, **kwargs)
        if shuffle and reverse:
            raise ValueError("shuffle and reverse are mutually exclusive")
        self.shuffle = shuffle
        self.reverse = reverse
        self._assert_maxcontext_invariants(pool_size)
        _setup_maxcontext_numeric_attrs(self, pool_size)
        self._allow_parse_failure_bm25_fallback = allow_parse_failure_bm25_fallback

    def _assert_maxcontext_invariants(self, pool_size: int) -> None:
        if self.config.model_type not in MAXCONTEXT_ALLOWED_MODEL_TYPES:
            raise ValueError(
                f"MaxContextTopDown requires Qwen3 / Qwen3.5 / Llama-3.1 / Ministral-3 model_type; "
                f"got {self.config.model_type!r}."
            )
        if self.scoring != "generation":
            raise ValueError("MaxContextTopDown requires --scoring generation.")
        if self.k != pool_size:
            raise ValueError(f"pool_size={pool_size} but ranker.k={self.k}.")
        if self.num_permutation != 1:
            raise ValueError(
                "MaxContextTopDown requires --num_permutation 1 "
                "(compare does not permute)."
            )
        if self.method != "selection":
            raise ValueError(
                "MaxContextTopDown requires method='selection' "
                "(full-pool best-only selection is the only supported algorithm)."
            )

    def _assert_maxcontext_fits(self, query: str, docs: List[SearchResult]) -> None:
        _assert_maxcontext_topdown_fits(self, query, docs)

    def _maxcontext_topdown_select(
        self, query: str, docs: List[SearchResult]
    ) -> List[SearchResult]:
        ranking = list(docs)
        self.total_compare = 0
        self.total_parse_fallback = 0
        self.total_lexical_refusal_fallback = 0
        self.total_numeric_out_of_range_fallback = 0
        self.total_degenerate_repetition_fallback = 0
        self.total_unparseable_after_exhaustion_fallback = 0
        self.total_duplicate_label_fallback = 0
        self.total_lenient_fallback = 0
        self.total_strict_parse_fallback = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        self.total_retries = 0
        self.total_retry_overhead_seconds = 0.0
        self.total_bm25_bypass = 0
        orig_pos = {doc.docid: i for i, doc in enumerate(docs)}
        self._maxcontext_original_positions = orig_pos
        for doc in docs:
            if doc.score is None or not math.isfinite(doc.score):
                raise ValueError(
                    "MaxContext n_docs=2 bypass requires finite BM25 scores; "
                    f"got {doc.score!r} for docid {doc.docid!r}."
                )

        n = len(ranking)
        top_idx = 0
        while n - top_idx > 1:
            window = ranking[top_idx:]
            window_len = len(window)
            if window_len == 2:
                s0, s1 = window[0].score, window[1].score
                if s0 > s1:
                    best_window_pos = 0
                elif s1 > s0:
                    best_window_pos = 1
                else:
                    best_window_pos = (
                        0
                        if orig_pos[window[0].docid] < orig_pos[window[1].docid]
                        else 1
                    )
                self.total_bm25_bypass += 1
            else:
                presented = self._apply_pool_ordering(window)
                best_label = self.compare(query, presented)
                best_window_pos = self._remap_label_to_original(
                    best_label, presented, window, default=0
                )
            if best_window_pos != 0:
                ranking[top_idx], ranking[top_idx + best_window_pos] = (
                    ranking[top_idx + best_window_pos],
                    ranking[top_idx],
                )
            top_idx += 1
        return ranking

    def rerank(self, query: str, docs: List[SearchResult]) -> List[SearchResult]:
        _reset_maxcontext_query_stats(self)
        self._maxcontext_original_positions = {
            doc.docid: i for i, doc in enumerate(docs)
        }
        _validate_maxcontext_pool_size(self, docs, "MaxContextTopDown")
        if len(docs) <= 1:
            return _materialize_maxcontext_results(docs)
        self._assert_maxcontext_fits(query, docs)
        try:
            ordered = self._maxcontext_topdown_select(query, docs)
        except ValueError as exc:
            if not _maxcontext_should_fallback_to_bm25(self, exc):
                raise
            self.total_parse_failure_bm25_fallback = getattr(
                self, "total_parse_failure_bm25_fallback", 0
            ) + 1
            qid = getattr(self, "_current_qid", "?")
            print(
                f"[MaxContext] parse_failure_bm25_fallback for query {qid!r}; "
                f"using first-stage (BM25) score ordering. Reason: {exc}"
            )
            ordered = self._score_order_sorted_docs(docs)
        return [
            SearchResult(docid=d.docid, score=-rank, text=None)
            for rank, d in enumerate(ordered, start=1)
        ]


class MaxContextBottomUpSetwiseLlmRanker(_MaxContextOrderingMixin, BottomUpSetwiseLlmRanker):
    """MaxContext-style worst-only selection over the full pool."""

    total_bm25_bypass: int = 0

    def __init__(self, *args, pool_size: int, shuffle: bool = False, reverse: bool = False,
                 allow_parse_failure_bm25_fallback: bool = False, **kwargs):
        MaxContextDualEndSetwiseLlmRanker._early_reject_unsupported_family(
            kwargs.get("model_name_or_path") or (args[0] if args else None)
        )
        super().__init__(*args, **kwargs)
        if shuffle and reverse:
            raise ValueError("shuffle and reverse are mutually exclusive")
        self.shuffle = shuffle
        self.reverse = reverse
        self._assert_maxcontext_invariants(pool_size)
        _setup_maxcontext_numeric_attrs(self, pool_size)
        self._allow_parse_failure_bm25_fallback = allow_parse_failure_bm25_fallback

    def _assert_maxcontext_invariants(self, pool_size: int) -> None:
        if self.config.model_type not in MAXCONTEXT_ALLOWED_MODEL_TYPES:
            raise ValueError(
                f"MaxContextBottomUp requires Qwen3 / Qwen3.5 / Llama-3.1 / Ministral-3 model_type; "
                f"got {self.config.model_type!r}."
            )
        if self.scoring != "generation":
            raise ValueError("MaxContextBottomUp requires --scoring generation.")
        if self.k != pool_size:
            raise ValueError(f"pool_size={pool_size} but ranker.k={self.k}.")
        if self.num_permutation != 1:
            raise ValueError(
                "MaxContextBottomUp requires --num_permutation 1 "
                "(compare_worst does not permute)."
            )
        if self.method != "selection":
            raise ValueError(
                "MaxContextBottomUp requires method='selection' "
                "(full-pool worst-only selection is the only supported algorithm)."
            )

    def _assert_maxcontext_fits(self, query: str, docs: List[SearchResult]) -> None:
        _assert_maxcontext_bottomup_fits(self, query, docs)

    def _maxcontext_bottomup_select(
        self, query: str, docs: List[SearchResult]
    ) -> List[SearchResult]:
        ranking = list(docs)
        self.total_compare = 0
        self.total_parse_fallback = 0
        self.total_lexical_refusal_fallback = 0
        self.total_numeric_out_of_range_fallback = 0
        self.total_degenerate_repetition_fallback = 0
        self.total_unparseable_after_exhaustion_fallback = 0
        self.total_duplicate_label_fallback = 0
        self.total_lenient_fallback = 0
        self.total_strict_parse_fallback = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        self.total_retries = 0
        self.total_retry_overhead_seconds = 0.0
        self.total_bm25_bypass = 0
        orig_pos = {doc.docid: i for i, doc in enumerate(docs)}
        self._maxcontext_original_positions = orig_pos
        for doc in docs:
            if doc.score is None or not math.isfinite(doc.score):
                raise ValueError(
                    "MaxContext n_docs=2 bypass requires finite BM25 scores; "
                    f"got {doc.score!r} for docid {doc.docid!r}."
                )

        bottom_idx = len(ranking) - 1
        while bottom_idx > 0:
            window = ranking[: bottom_idx + 1]
            window_len = len(window)
            if window_len == 2:
                s0, s1 = window[0].score, window[1].score
                if s0 < s1:
                    worst_window_pos = 0
                elif s1 < s0:
                    worst_window_pos = 1
                else:
                    worst_window_pos = (
                        0
                        if orig_pos[window[0].docid] > orig_pos[window[1].docid]
                        else 1
                    )
                self.total_bm25_bypass += 1
            else:
                presented = self._apply_pool_ordering(window)
                worst_label = self.compare_worst(query, presented)
                worst_window_pos = self._remap_label_to_original(
                    worst_label, presented, window, default=window_len - 1
                )
            if worst_window_pos != bottom_idx:
                ranking[bottom_idx], ranking[worst_window_pos] = (
                    ranking[worst_window_pos],
                    ranking[bottom_idx],
                )
            bottom_idx -= 1
        return ranking

    def rerank(self, query: str, docs: List[SearchResult]) -> List[SearchResult]:
        _reset_maxcontext_query_stats(self)
        self._maxcontext_original_positions = {
            doc.docid: i for i, doc in enumerate(docs)
        }
        _validate_maxcontext_pool_size(self, docs, "MaxContextBottomUp")
        if len(docs) <= 1:
            return _materialize_maxcontext_results(docs)
        self._assert_maxcontext_fits(query, docs)
        try:
            ordered = self._maxcontext_bottomup_select(query, docs)
        except ValueError as exc:
            if not _maxcontext_should_fallback_to_bm25(self, exc):
                raise
            self.total_parse_failure_bm25_fallback = getattr(
                self, "total_parse_failure_bm25_fallback", 0
            ) + 1
            qid = getattr(self, "_current_qid", "?")
            print(
                f"[MaxContext] parse_failure_bm25_fallback for query {qid!r}; "
                f"using first-stage (BM25) score ordering. Reason: {exc}"
            )
            ordered = self._score_order_sorted_docs(docs)
        return [
            SearchResult(docid=d.docid, score=-rank, text=None)
            for rank, d in enumerate(ordered, start=1)
        ]
