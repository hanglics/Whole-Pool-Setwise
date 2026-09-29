from types import MethodType, SimpleNamespace

from llmrankers.rankers import SearchResult
from llmrankers.setwise_extended import (
    MaxContextDualEndSetwiseLlmRanker,
    MaxContextTopDownSetwiseLlmRanker,
)


def docs(n=6):
    return [SearchResult(docid=f"d{i}", score=float(i), text=f"text {i}") for i in range(n)]


def configure(ranker, n, output_depth):
    ranker._maxcontext_pool_size = n
    ranker.output_depth = output_depth
    ranker.CHARACTERS = [str(i + 1) for i in range(n)]
    ranker.shuffle = False
    ranker.reverse = False
    ranker.num_permutation = 1
    ranker.num_child = n - 1
    ranker.k = n
    ranker.strict_no_parse_fallback = True
    ranker._allow_parse_failure_bm25_fallback = False
    ranker._current_qid = "q1"
    ranker.capture_raw_responses = False
    ranker._comparison_log_path = None


def scripted_best(self, query, window):
    self.total_compare += 1
    index = max(range(len(window)), key=lambda i: window[i].score)
    return self.CHARACTERS[index]


def scripted_both(self, query, window):
    self.total_compare += 1
    best = max(range(len(window)), key=lambda i: window[i].score)
    worst = min(range(len(window)), key=lambda i: window[i].score)
    return self.CHARACTERS[best], self.CHARACTERS[worst]


def test_topdown_early_stop_matches_full_prefix_and_counts_calls():
    early = MaxContextTopDownSetwiseLlmRanker.__new__(MaxContextTopDownSetwiseLlmRanker)
    configure(early, 6, 2)
    early.compare = MethodType(scripted_best, early)
    early_order = early._maxcontext_topdown_select("q", docs())
    assert [doc.docid for doc in early_order[:2]] == ["d5", "d4"]
    assert early.total_compare == 2
    assert early.total_selection_steps == 2
    assert early.total_non_llm_bypasses == 0

    full = MaxContextTopDownSetwiseLlmRanker.__new__(MaxContextTopDownSetwiseLlmRanker)
    configure(full, 6, 6)
    full.compare = MethodType(scripted_best, full)
    full_order = full._maxcontext_topdown_select("q", docs())
    assert [doc.docid for doc in full_order[:2]] == [doc.docid for doc in early_order[:2]]
    assert full.total_compare == 4
    assert full.total_selection_steps == 5
    assert full.total_non_llm_bypasses == 1


def test_dual_early_stop_matches_full_prefix():
    early = MaxContextDualEndSetwiseLlmRanker.__new__(MaxContextDualEndSetwiseLlmRanker)
    configure(early, 6, 2)
    early.compare_both = MethodType(scripted_both, early)
    early.total_compare = 0
    early_order = docs()
    early._double_ended_selection(early_order, "q", 2)

    full = MaxContextDualEndSetwiseLlmRanker.__new__(MaxContextDualEndSetwiseLlmRanker)
    configure(full, 6, 6)
    full.compare_both = MethodType(scripted_both, full)
    full.total_compare = 0
    full_order = docs()
    full._double_ended_selection(full_order, "q", 6)
    assert [doc.docid for doc in early_order[:2]] == [doc.docid for doc in full_order[:2]] == ["d5", "d4"]
    assert early.total_compare == 2
    assert full.total_compare == 3


