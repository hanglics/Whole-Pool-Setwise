from __future__ import annotations

import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest


def _missing_dependency(*_args, **_kwargs):
    raise RuntimeError("Optional runtime dependency is not available in this test environment.")


for name in ("ir_datasets", "openai", "tiktoken"):
    if name not in sys.modules:
        sys.modules[name] = ModuleType(name)

if "pyserini" not in sys.modules:
    pyserini_stub = ModuleType("pyserini")
    pyserini_search_stub = ModuleType("pyserini.search")
    pyserini_lucene_stub = ModuleType("pyserini.search.lucene")
    pyserini_base_stub = ModuleType("pyserini.search._base")
    pyserini_lucene_stub.LuceneSearcher = SimpleNamespace(from_prebuilt_index=_missing_dependency)
    pyserini_base_stub.get_topics = _missing_dependency
    sys.modules["pyserini"] = pyserini_stub
    sys.modules["pyserini.search"] = pyserini_search_stub
    sys.modules["pyserini.search.lucene"] = pyserini_lucene_stub
    sys.modules["pyserini.search._base"] = pyserini_base_stub

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
from llmrankers.rankers import SearchResult
from llmrankers.setwise_extended import (
    DualEndSetwiseLlmRanker,
    MaxContextBottomUpSetwiseLlmRanker,
    MaxContextDualEndSetwiseLlmRanker,
    MaxContextTopDownSetwiseLlmRanker,
)


def make_docs(n: int) -> list[SearchResult]:
    return [SearchResult(docid=f"d{i}", score=float(n - i), text=f"doc {i}") for i in range(1, n + 1)]


def bare_ranker(cls, *, shuffle=False, reverse=False):
    ranker = object.__new__(cls)
    ranker.shuffle = shuffle
    ranker.reverse = reverse
    ranker.CHARACTERS = ["1", "2", "3", "4", "5"]
    ranker._current_qid = "q1"
    return ranker


def fake_dualend_init(self, *args, **kwargs):
    self.config = SimpleNamespace(model_type="qwen3")
    self.scoring = kwargs["scoring"]
    self.k = kwargs["k"]
    self.num_permutation = kwargs["num_permutation"]
    self.method = kwargs["method"]


def make_run_args(*, shuffle=False, reverse=False, direction="maxcontext_dualend", setwise=True):
    setwise_args = None
    if setwise:
        setwise_args = SimpleNamespace(
            direction=direction,
            num_child=3,
            method="selection",
            k=10,
            num_permutation=1,
            character_scheme="letters_a_w",
            fusion="rrf",
            alpha=0.5,
            gate_strategy="hybrid",
            shortlist_size=20,
            margin_threshold=0.15,
            uncertainty_percentile=None,
            order_robust_orderings=3,
        )
    return SimpleNamespace(
        run=SimpleNamespace(
            model_name_or_path="Qwen/Qwen3-4B",
            tokenizer_name_or_path=None,
            device="cpu",
            cache_dir=None,
            openai_key=None,
            scoring="generation",
            hits=10,
            query_length=128,
            passage_length=512,
            log_comparisons=None,
            ir_dataset_name=None,
            pyserini_index=None,
            run_path="unused",
            save_path="unused",
            shuffle_ranking=None,
            shuffle=shuffle,
            reverse=reverse,
        ),
        setwise=setwise_args,
        pointwise=None if setwise else SimpleNamespace(method="yes_no", batch_size=2),
        pairwise=None,
        listwise=None,
    )


def test_ranker_constructor_rejects_mutual_exclusion():
    with mock.patch.object(DualEndSetwiseLlmRanker, "__init__", new=fake_dualend_init):
        with pytest.raises(ValueError, match="mutually exclusive"):
            MaxContextDualEndSetwiseLlmRanker(
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
                shuffle=True,
                reverse=True,
            )


def test_apply_pool_ordering_default_reverse_and_shuffle():
    docs = make_docs(5)

    default_ranker = bare_ranker(MaxContextDualEndSetwiseLlmRanker)
    assert default_ranker._apply_pool_ordering(docs) is docs

    reverse_ranker = bare_ranker(MaxContextTopDownSetwiseLlmRanker, reverse=True)
    assert [doc.docid for doc in reverse_ranker._apply_pool_ordering(docs)] == [
        "d5",
        "d4",
        "d3",
        "d2",
        "d1",
    ]

    shuffle_ranker = bare_ranker(MaxContextBottomUpSetwiseLlmRanker, shuffle=True)
    first = shuffle_ranker._apply_pool_ordering(docs)
    second = shuffle_ranker._apply_pool_ordering(docs)
    assert [doc.docid for doc in first] == [doc.docid for doc in second]
    assert sorted(doc.docid for doc in first) == sorted(doc.docid for doc in docs)
    assert first is not docs


def test_shuffle_hash_is_cross_process_deterministic():
    code = (
        "import hashlib, random\n"
        "docs=list(range(10))\n"
        "payload=b'929' + b'\\0' + b'q1' + b'\\0' + b'10'\n"
        "digest=hashlib.blake2b(payload, digest_size=8).digest()\n"
        "rng=random.Random(int.from_bytes(digest, 'big', signed=False))\n"
        "rng.shuffle(docs)\n"
        "print(','.join(map(str, docs)))\n"
    )
    first = subprocess.check_output([sys.executable, "-c", code], text=True)
    second = subprocess.check_output([sys.executable, "-c", code], text=True)
    assert first == second


@pytest.mark.parametrize(
    "ranker_cls,label,default,expected_idx",
    [
        (MaxContextTopDownSetwiseLlmRanker, "1", 0, 2),
        (MaxContextBottomUpSetwiseLlmRanker, "3", 2, 0),
        (MaxContextDualEndSetwiseLlmRanker, "2", 0, 1),
    ],
)
def test_label_remaps_from_presented_to_original_window(ranker_cls, label, default, expected_idx):
    ranker = bare_ranker(ranker_cls, reverse=True)
    original = make_docs(3)
    presented = list(reversed(original))
    assert ranker._remap_label_to_original(label, presented, original, default) == expected_idx


def test_dualend_window_override_remaps_both_labels():
    ranker = bare_ranker(MaxContextDualEndSetwiseLlmRanker, reverse=True)
    docs = make_docs(3)
    seen = {}

    def compare_both(_query, presented):
        seen["presented"] = [doc.docid for doc in presented]
        return "1", "3"

    ranker.compare_both = compare_both
    assert ranker._compare_both_window("query", docs, 0, 3) == ("3", "1")
    assert seen["presented"] == ["d3", "d2", "d1"]


def test_dualend_default_off_passes_labels_through():
    ranker = bare_ranker(MaxContextDualEndSetwiseLlmRanker)
    docs = make_docs(3)
    seen = {}

    def compare_both(_query, presented):
        seen["presented"] = presented
        return "1", "3"

    ranker.compare_both = compare_both
    assert ranker._compare_both_window("query", docs, 0, 3) == ("1", "3")
    assert seen["presented"] == docs


def test_dualend_tournament_branch_rejects_condition_flags_for_maxcontext():
    ranker = bare_ranker(MaxContextDualEndSetwiseLlmRanker, shuffle=True)
    ranker.num_child = 1
    ranker._maxcontext_pool_size = 5
    with pytest.raises(RuntimeError, match="tournament selection is unreachable"):
        DualEndSetwiseLlmRanker._double_ended_selection(ranker, make_docs(5), "query", 1)


def test_run_py_rejects_shuffle_reverse_together():
    with pytest.raises(SystemExit, match="mutually exclusive"):
        run_module.main(make_run_args(shuffle=True, reverse=True))


def test_run_py_rejects_condition_flags_for_non_maxcontext_direction():
    with pytest.raises(SystemExit, match="only supported for MaxContext"):
        run_module.main(make_run_args(shuffle=True, direction="topdown"))


def test_run_py_rejects_condition_flags_without_setwise_subparser():
    with pytest.raises(SystemExit, match="only supported for MaxContext"):
        run_module.main(make_run_args(shuffle=True, setwise=False))
