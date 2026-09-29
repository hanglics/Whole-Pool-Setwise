import hashlib
import importlib
import importlib.util
import json
import py_compile
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "experiments" / "baselines"


def require_upstream(name):
    if not (BASELINES / ".upstreams" / name).is_dir():
        pytest.skip("Fetch the pinned upstreams to run parser integration checks.")


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, BASELINES / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


official = load("official_baselines_for_test", "official_baselines.py")
fetcher = load("fetch_official_baselines_for_test", "fetch_official_baselines.py")
sys.path.insert(0, str(BASELINES))
try:
    tour_runner = load("tourrank_runner_for_test", "run_tourrank_model_matched.py")
    liu_runner = load("liu_runner_for_test", "run_liu_zero_shot_model_matched.py")
    official_finalizer = load(
        "official_finalizer_for_test", "finalize_official_attempt.py"
    )
finally:
    sys.path.pop(0)


def load_liu_upstream_with_test_stubs(monkeypatch):
    """Load pinned CPU-only prompt/parser code without importing model weights."""
    require_upstream("liu_fullrank")
    for name in ("data", "utils", "rerank.rankllm", "rerank.reranker"):
        sys.modules.pop(name, None)
    ftfy = types.ModuleType("ftfy")
    ftfy.fix_text = lambda value: value.replace("FranÃ§ais", "Français")
    dacite = types.ModuleType("dacite")
    dacite.from_dict = lambda **_kwargs: None
    # The pinned Liu prompt/parser closure only needs this type for annotations.
    # Importing real Pyserini here initializes Pyjnius and requires a JDK, which
    # makes the CPU-only unit tests depend on javac and test execution order.
    pyserini = types.ModuleType("pyserini")
    pyserini_search = types.ModuleType("pyserini.search")
    pyserini_search.JLuceneSearcherResult = type("JLuceneSearcherResult", (), {})
    pyserini.search = pyserini_search
    monkeypatch.setitem(sys.modules, "ftfy", ftfy)
    monkeypatch.setitem(sys.modules, "dacite", dacite)
    monkeypatch.setitem(sys.modules, "pyserini", pyserini)
    monkeypatch.setitem(sys.modules, "pyserini.search", pyserini_search)
    monkeypatch.syspath_prepend(str(BASELINES / ".upstreams" / "liu_fullrank"))
    with official.isolated_upstream_imports():
        rankllm = importlib.import_module("rerank.rankllm")
        data = importlib.import_module("data")
        utils = importlib.import_module("utils")
        reranker = importlib.import_module("rerank.reranker")
    return rankllm, data, utils, reranker


def make_test_liu_agent(monkeypatch):
    rankllm, data, utils, _ = load_liu_upstream_with_test_stubs(monkeypatch)

    class FakeTokenizer:
        @staticmethod
        def encode(text, *args, **kwargs):
            return list(text.encode("utf-8"))

    class FakeRanker:
        max_input_tokens = 32768
        tokenizer = FakeTokenizer()

    class FakeBackend:
        model = "Qwen/Qwen3.5-9B"
        ranker = FakeRanker()

        def __init__(self):
            self.last_messages = None

        def count_prompt_tokens(self, messages):
            self.last_messages = messages
            rendered = "CHAT:" + "|".join(
                f"{message['role']}:{message['content']}" for message in messages
            )
            return len(rendered.encode("utf-8"))

    Agent = liu_runner.make_model_matched_agent_class(rankllm.RankLLM, utils)
    return Agent(FakeBackend()), data, utils


def test_locked_upstreams_have_real_hashes_and_zero_shot_scope():
    lock = json.loads((BASELINES / "upstreams.lock.json").read_text())
    for upstream in lock["upstreams"].values():
        assert len(upstream["commit"]) == 40
        assert all(len(digest) == 64 for digest in upstream["files"].values())
    executable_baselines = "\n".join(
        path.read_text()
        for path in (
            BASELINES / "run_liu_zero_shot_model_matched.py",
            BASELINES / "run_tourrank_model_matched.py",
            BASELINES / "run_liu.sh",
            BASELINES / "run_tourrank.sh",
            BASELINES / "submit_jobs.sh",
        )
    )
    assert "RankMistral100" not in executable_baselines
    assert "OPENAI_API_KEY" not in executable_baselines
    assert "gpt-3.5" not in executable_baselines
    assert "Mistral-7B-Instruct-v0.3" not in executable_baselines
    model_locks = json.loads((BASELINES / "model_revisions.lock.json").read_text())
    assert set(model_locks) == liu_runner.PAPER_MODELS
    assert set(model_locks) == tour_runner.PAPER_MODELS
    for model, revision in model_locks.items():
        official.require_paper_model_revision(model, revision)
    with pytest.raises(ValueError, match="Revision mismatch"):
        official.require_paper_model_revision(
            "Qwen/Qwen3.5-9B", "0" * 40
        )
    liu_files = lock["upstreams"]["liu_fullrank"]["files"]
    assert {
        "config.py", "index_and_topics.py", "utils.py", "rerank/__init__.py",
        "data.py", "rerank/rankllm.py", "rerank/reranker.py",
    }.issubset(liu_files)
    requirements = (ROOT / "requirements.txt").read_text()
    assert "ftfy==6.3.1" in requirements and "dacite==1.8.1" in requirements


def test_upstream_status_ignores_only_generated_python_bytecode():
    meaningful, ignored = fetcher.split_checkout_status(
        "\n".join(
            [
                "?? __pycache__/config.cpython-310.pyc",
                "?? rerank/__pycache__/rankllm.cpython-310.pyo",
                " M rerank/rankllm.py",
                "?? generated-result.json",
            ]
        )
    )
    assert ignored == [
        "__pycache__/config.cpython-310.pyc",
        "rerank/__pycache__/rankllm.cpython-310.pyo",
    ]
    assert meaningful == [" M rerank/rankllm.py", "?? generated-result.json"]


def test_liu_upstream_bytecode_suppression_restores_interpreter_state():
    previous_write_policy = sys.dont_write_bytecode
    previous_cache_prefix = sys.pycache_prefix
    with liu_runner.suppress_upstream_bytecode():
        assert sys.dont_write_bytecode is True
        assert sys.pycache_prefix != previous_cache_prefix
    assert sys.dont_write_bytecode is previous_write_policy
    assert sys.pycache_prefix == previous_cache_prefix


def test_isolated_upstream_import_does_not_execute_existing_unchecked_pyc(
    tmp_path,
):
    source = tmp_path / "upstream_cache_probe.py"
    source.write_text("VALUE = 'cached'\n")
    adjacent_cache = Path(importlib.util.cache_from_source(str(source)))
    adjacent_cache.parent.mkdir()
    py_compile.compile(
        str(source),
        cfile=str(adjacent_cache),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )
    source.write_text("VALUE = 'source'\n")
    assert adjacent_cache.is_file()

    spec = importlib.util.spec_from_file_location(
        "isolated_upstream_cache_probe", source
    )
    module = importlib.util.module_from_spec(spec)
    with official.isolated_upstream_imports():
        spec.loader.exec_module(module)

    assert module.VALUE == "source"


def test_ordered_pool_validation_detects_ranked_doc_mismatch():
    official_pool = {"q": [{"docid": "a", "score": 1.0}, {"docid": "b", "score": 0.5}]}
    local_pool = {"q": [{"docid": "b", "score": 1.0}, {"docid": "a", "score": 0.5}]}
    discrepancies = official.validate_ordered_pools(official_pool, local_pool)
    assert [row["field"] for row in discrepancies].count("docid") == 2


def test_ordered_pool_validation_detects_explicit_rank_mismatch():
    official_pool = {"q": [{"docid": "a", "rank": 1, "score": 1.0}]}
    local_pool = {"q": [{"docid": "a", "rank": 2, "score": 1.0}]}
    discrepancies = official.validate_ordered_pools(official_pool, local_pool)
    assert discrepancies == [
        {
            "qid": "q",
            "position": 1,
            "field": "rank",
            "expected": 1,
            "official": 1,
            "local": 2,
        }
    ]


def test_ordered_pool_validation_rejects_equal_but_nonpositional_ranks():
    official_pool = {"q": [{"docid": "a", "rank": 2, "score": 1.0}]}
    local_pool = {"q": [{"docid": "a", "rank": 2, "score": 1.0}]}
    discrepancies = official.validate_ordered_pools(official_pool, local_pool)
    assert discrepancies == [
        {
            "qid": "q",
            "position": 1,
            "field": "rank",
            "expected": 1,
            "official": 2,
            "local": 2,
        }
    ]


def test_tourrank_uses_local_passage_text_after_structural_validation(
    monkeypatch, tmp_path
):
    official_records = {
        "q1": {
            "query": "query",
            "hits": [
                {
                    "qid": "q1",
                    "docid": "d1",
                    "rank": 1,
                    "score": 2.0,
                    "content": "official passage version",
                }
            ],
        }
    }
    local_pools = {
        "q1": [
            {
                "docid": "d1",
                "rank": 1,
                "score": 2.0,
                "content": "our ir_datasets passage version",
            }
        ]
    }
    monkeypatch.setattr(
        tour_runner,
        "local_tourrank_records",
        lambda _dataset, _run: ({"q1": "query"}, local_pools),
    )

    validation, inference_records, discrepancies = tour_runner.verify_input(
        "dl19", official_records, tmp_path / "run.txt", tmp_path / "diffs.json"
    )

    assert validation["structural_discrepancies"] == 0
    assert validation["content_discrepancies"] == 1
    assert validation["content_discrepancy_rate"] == 1.0
    assert validation["official_json_used_for_inference"] is False
    assert validation["local_ir_datasets_content_used_for_inference"] is True
    assert len(validation["local_untruncated_input_sha256"]) == 64
    assert inference_records["q1"]["hits"][0]["content"] == (
        "our ir_datasets passage version"
    )
    assert inference_records["q1"]["hits"][0]["docid"] == "d1"
    assert not (tmp_path / "diffs.json").exists()
    assert discrepancies == [
        {
            "field": "content",
            "qid": "q1",
            "rank": 1,
            "official_hash": official.normalized_text_hash(
                "official passage version"
            ),
            "local_hash": official.normalized_text_hash(
                "our ir_datasets passage version"
            ),
        }
    ]
    tour_runner.persist_discrepancy_report(tmp_path / "diffs.json", discrepancies)
    assert json.loads((tmp_path / "diffs.json").read_text()) == discrepancies


def test_tourrank_local_passage_hash_binds_model_facing_content():
    records = {
        "q1": {
            "query": "query",
            "hits": [{"qid": "q1", "docid": "d1", "content": "first"}],
        }
    }
    first = tour_runner.inference_records_sha256(records)
    records["q1"]["hits"][0]["content"] = "second"
    assert tour_runner.inference_records_sha256(records) != first


def test_tourrank_query_limit_uses_local_first_stage_run_order_before_hash():
    records = {
        "q1": {"query": "one", "hits": [{"docid": "d1", "content": "a"}]},
        "q2": {"query": "two", "hits": [{"docid": "d2", "content": "b"}]},
    }
    selected = tour_runner.select_query_limit(records, 1, ["q2", "q1"])
    assert list(selected) == ["q2"]
    assert tour_runner.inference_records_sha256(selected) != (
        tour_runner.inference_records_sha256(records)
    )


def test_tourrank_smoke_qid_matches_finalizer_order_on_locked_dl19_inputs():
    require_upstream("tourrank")
    official_records = official.load_tourrank_jsonl(
        BASELINES / ".upstreams" / "tourrank" / "data" / "bm25_dl19_top100.jsonl"
    )
    local_run = official.load_trec(
        ROOT / "runs" / "bm25" / "run.msmarco-v1-passage.bm25-default.dl19.txt",
        100,
    )

    assert next(iter(official_records)) != next(iter(local_run))
    selected = tour_runner.select_query_limit(
        official_records, 1, list(local_run)
    )
    assert list(selected) == list(local_run)[:1] == ["19335"]


def test_tourrank_local_passages_use_native_active_tokenizer_cap():
    records = {
        "q1": {
            "query": "query",
            "hits": [{"qid": "q1", "docid": "d1", "content": "full passage"}],
        }
    }

    class FakeTokenizer:
        @staticmethod
        def tokenize(text):
            return text.split()

    class FakeRanker:
        tokenizer = FakeTokenizer()

        @staticmethod
        def truncate(text, length):
            return f"truncated:{length}:{text}"

    truncated, stats = tour_runner.truncate_local_passages(
        records, FakeRanker(), 1
    )
    assert truncated["q1"]["hits"][0]["content"] == (
        "truncated:1:full passage"
    )
    assert records["q1"]["hits"][0]["content"] == "full passage"
    assert stats == {
        "passage_count": 1,
        "truncated_passage_count": 1,
        "max_untruncated_passage_tokens": 2,
        "passage_length": 1,
        "truncation_method": (
            "active_tokenizer_tokenize_then_convert_tokens_to_string"
        ),
    }


def test_tourrank_discrepancy_report_refuses_resume_drift(tmp_path):
    path = tmp_path / "input_discrepancies.json"
    original = [{"qid": "q1", "rank": 1, "field": "content"}]
    tour_runner.persist_discrepancy_report(path, original)
    with pytest.raises(ValueError, match="discrepancy report drift"):
        tour_runner.persist_discrepancy_report(
            path, [{"qid": "q1", "rank": 2, "field": "content"}]
        )
    assert json.loads(path.read_text()) == original


def test_tourrank_validate_only_refuses_a_production_attempt(tmp_path):
    protocol = tmp_path / "checkpoints" / "protocol.json"
    protocol.parent.mkdir()
    protocol.write_text("{}")
    with pytest.raises(ValueError, match="experiment attempt directory"):
        tour_runner.assert_validate_only_output_safe(tmp_path)


def test_tourrank_finalizer_enforces_local_passage_provenance(tmp_path):
    discrepancy_path = tmp_path / "input_discrepancies.json"
    discrepancy_path.write_text(
        json.dumps(
            [
                {
                    "qid": "q1",
                    "rank": 1,
                    "field": "content",
                    "official_hash": "1" * 64,
                    "local_hash": "2" * 64,
                }
            ]
        )
    )
    manifest = {
        "passage_source_policy": official_finalizer.TOURRANK_PASSAGE_POLICY,
        "passage_length": 512,
        "query_limit_policy": official.TOURRANK_QUERY_LIMIT_POLICY,
        "inference_input_sha256": "a" * 64,
        "inference_qids": ["q1"],
        "inference_query_count": 1,
        "inference_candidate_count": 100,
        "input_validation": {
            "discrepancies": 1,
            "content_discrepancies": 1,
            "structural_discrepancies": 0,
            "official_json_used_for_inference": False,
            "local_ir_datasets_content_used_for_inference": True,
        },
        "passage_truncation": {
            "passage_count": 100,
            "truncated_passage_count": 4,
            "passage_length": 512,
            "truncation_method": (
                "active_tokenizer_tokenize_then_convert_tokens_to_string"
            ),
        },
    }
    official_finalizer.validate_tourrank_provenance(
        manifest, query_ids={"q1"}, discrepancy_path=discrepancy_path
    )
    manifest["inference_qids"] = ["q2"]
    with pytest.raises(ValueError, match="inference qids"):
        official_finalizer.validate_tourrank_provenance(
            manifest, query_ids={"q1"}, discrepancy_path=discrepancy_path
        )
    manifest["inference_qids"] = ["q1"]
    manifest["input_validation"]["official_json_used_for_inference"] = True
    with pytest.raises(ValueError, match="released passage text"):
        official_finalizer.validate_tourrank_provenance(
            manifest, query_ids={"q1"}, discrepancy_path=discrepancy_path
        )


def test_liu_uses_local_passage_text_after_structural_validation():
    request = types.SimpleNamespace(
        query=types.SimpleNamespace(qid="q1", text="query"),
        candidates=[
            types.SimpleNamespace(
                docid="d1",
                score=2.0,
                doc={"id": "d1", "contents": "official mojibake Â°F"},
            )
        ],
    )
    official_docs = {
        "q1": [
            {
                "docid": "d1",
                "rank": 1,
                "score": 2.0,
                "content": "official mojibake Â°F",
            }
        ]
    }
    local_pools = {
        "q1": [
            {
                "docid": "d1",
                "rank": 1,
                "score": 2.0,
                "content": "our clean local °F",
                "_local_text": "our clean local °F",
                "_local_title": None,
            }
        ]
    }

    validation, adapted, discrepancies = liu_runner.prepare_local_liu_input(
        [request],
        {"q1": "query"},
        official_docs,
        {"q1": "query"},
        local_pools,
    )

    assert validation["structural_discrepancies"] == 0
    assert validation["content_discrepancies"] == 1
    assert validation["official_index_content_used_for_inference"] is False
    assert validation["local_ir_datasets_content_used_for_inference"] is True
    assert len(validation["local_untruncated_input_sha256"]) == 64
    assert adapted[0].candidates[0].doc == {
        "id": "d1",
        "contents": "our clean local °F",
    }
    assert request.candidates[0].doc["contents"] == "official mojibake Â°F"
    assert discrepancies == [
        {
            "qid": "q1",
            "rank": 1,
            "field": "content",
            "official_hash": official.normalized_text_hash(
                "official mojibake Â°F"
            ),
            "local_hash": official.normalized_text_hash("our clean local °F"),
        }
    ]


def test_liu_local_passage_hash_binds_candidate_documents():
    request = types.SimpleNamespace(
        query=types.SimpleNamespace(qid="q1", text="query"),
        candidates=[
            types.SimpleNamespace(
                docid="d1", score=1.0, doc={"contents": "first"}
            )
        ],
    )
    first = liu_runner.inference_requests_sha256([request])
    request.candidates[0].doc["contents"] = "second"
    assert liu_runner.inference_requests_sha256([request]) != first


def test_liu_discrepancy_report_refuses_resume_drift(tmp_path):
    path = tmp_path / "input_discrepancies.json"
    original = [{"qid": "q1", "rank": 1, "field": "content"}]
    liu_runner.persist_discrepancy_report(path, original)
    with pytest.raises(ValueError, match="discrepancy report drift"):
        liu_runner.persist_discrepancy_report(
            path, [{"qid": "q1", "rank": 2, "field": "content"}]
        )
    assert json.loads(path.read_text()) == original


def test_liu_validate_only_refuses_a_production_attempt(tmp_path):
    protocol = tmp_path / "checkpoints" / "protocol.json"
    protocol.parent.mkdir()
    protocol.write_text("{}")
    with pytest.raises(ValueError, match="experiment attempt directory"):
        liu_runner.assert_validate_only_output_safe(tmp_path)


def test_liu_finalizer_enforces_local_passage_provenance(tmp_path):
    discrepancy_path = tmp_path / "input_discrepancies.json"
    discrepancy_path.write_text(
        json.dumps(
            [
                {
                    "qid": "q1",
                    "rank": 1,
                    "field": "content",
                    "official_hash": "1" * 64,
                    "local_hash": "2" * 64,
                }
            ]
        )
    )
    manifest = {
        "passage_source_policy": official_finalizer.LIU_PASSAGE_POLICY,
        "max_passage_length": 100,
        "inference_input_sha256": "a" * 64,
        "inference_query_count": 1,
        "inference_candidate_count": 100,
        "official_index_content_used_for_inference": False,
        "local_ir_datasets_content_used_for_inference": True,
        "input_validation": {
            "discrepancies": 1,
            "content_discrepancies": 1,
            "structural_discrepancies": 0,
            "official_index_content_used_for_inference": False,
            "local_ir_datasets_content_used_for_inference": True,
        },
    }
    official_finalizer.validate_liu_provenance(
        manifest, query_count=1, discrepancy_path=discrepancy_path
    )
    manifest["official_index_content_used_for_inference"] = True
    with pytest.raises(ValueError, match="released-index passage text"):
        official_finalizer.validate_liu_provenance(
            manifest, query_count=1, discrepancy_path=discrepancy_path
        )


def test_protocol_argv_drops_only_execution_control_flags():
    argv = ["runner.py", "--dataset", "dl19", "--resume", "--validate-only"]
    assert official.sanitized_protocol_argv(argv) == ["runner.py", "--dataset", "dl19"]


def test_tourrank_schedule_has_thirteen_successes_per_tournament():
    shapes = [(20, 10)] * 5 + [(10, 4)] * 5 + [(20, 10), (10, 5), (5, 2)]
    events = [
        {
            "tournament": 0,
            "request_index": index,
            "status": "ok",
            "n": n,
            "m": m,
            "started_at": float(index),
            "ended_at": float(index) + 0.5,
            "duration_seconds": 0.5,
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "generation_budget": 512,
            "special_tokens_removed": False,
            "format_valid": True,
        }
        for index, (n, m) in enumerate(shapes)
    ]
    events[0].update(
        {
            "duration_seconds": 0.7,
            "prompt_tokens": 20,
            "completion_tokens": 4,
            "retry_calls": 1,
            "retry_seconds": 0.2,
            "initial_format_valid": False,
            "format_valid": True,
            "format_repair_succeeded": True,
            "generation_attempts": [
                {"special_tokens_removed": True},
                {"special_tokens_removed": True},
            ],
        }
    )
    tour_runner.annotate_stages(events)
    row = tour_runner.build_telemetry("q", "paper-model", 1, events, 10.0)
    assert row["total_calls"] == 13
    assert row["retry_calls"] == 1
    assert row["retry_inclusive_llm_calls"] == 14
    assert row["generation_attempt_count"] == 14
    assert row["prompt_tokens"] == 140
    assert row["completion_tokens"] == 28
    assert row["initial_format_failure_calls"] == 1
    assert row["format_failure_calls"] == 0
    assert row["format_repair_calls"] == 1
    assert row["format_repair_successes"] == 1
    assert row["retry_seconds"] == pytest.approx(0.2)
    assert row["retry_excluded_wall_seconds"] == pytest.approx(9.8)
    assert row["special_token_stripped_generations"] == 2
    assert all(event["generation_budget"] == 512 for event in events)
    assert [event["stage"] for event in events].count("100_to_50") == 5


def test_open_backend_passes_liu_equal_min_max_generation_budget():
    import torch

    captured = {}

    class FakeInputs:
        input_ids = torch.tensor([[10, 11]])
        attention_mask = torch.tensor([[1, 1]])

    class FakeTokenizer:
        @staticmethod
        def decode(token_ids, skip_special_tokens):
            captured.setdefault("skip_special_tokens", []).append(
                skip_special_tokens
            )
            return "rank" if skip_special_tokens else "rank</s>"

    class FakeRanker:
        device = "cpu"
        max_input_tokens = 100
        tokenizer = FakeTokenizer()

        @staticmethod
        def _build_chat_prompt(messages):
            return str(messages)

        @staticmethod
        def _tokenize_inputs(_prompt):
            return FakeInputs()

        @staticmethod
        def _generate(_inputs, max_new_tokens, min_new_tokens):
            captured["max_new_tokens"] = max_new_tokens
            captured["min_new_tokens"] = min_new_tokens
            return torch.tensor([[10, 11, 12, 13]])

    backend = tour_runner.OpenModelBackend.__new__(tour_runner.OpenModelBackend)
    backend.ranker = FakeRanker()
    generated = backend.generate(
        [{"role": "user", "content": "rank"}],
        max_new_tokens=17,
        min_new_tokens=17,
    )
    assert captured == {
        "max_new_tokens": 17,
        "min_new_tokens": 17,
        "skip_special_tokens": [False, True],
    }
    assert generated["min_new_tokens"] == generated["max_new_tokens"] == 17
    assert generated["raw_output"] == "rank</s>"
    assert generated["content_output"] == "rank"


def test_liu_parser_keeps_raw_completion_transport(monkeypatch):
    agent, _, _ = make_test_liu_agent(monkeypatch)

    def generate(_messages, **_kwargs):
        return {
            "raw_output": "[2] > [1]</s>",
            "content_output": "[2] > [1]",
            "completion_tokens": 7,
        }

    agent.backend.generate = generate
    output, token_count = agent.run_llm("prompt", output_passages_num=2)

    assert output == "[2] > [1]</s>"
    assert token_count == 7
    assert agent.events[0]["raw_output"] != agent.events[0]["content_output"]


def test_tourrank_passes_api_equivalent_content_to_official_parser():
    class Backend:
        @staticmethod
        def generate(*_args, **_kwargs):
            return {
                "raw_output": "Document 3, Document 1<|im_end|>",
                "content_output": "Document 3, Document 1",
                "wall_seconds": 0.5,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "rendered_prompt_sha256": "a" * 64,
            }

    responder = tour_runner.InstrumentedResponder(Backend())
    responder.begin_tournament(0)
    parser_input = responder(
        [
            {"role": "system", "content": "rank"},
            {
                "role": "user",
                "content": "Compare the query and 4 documents; select the 2 documents.",
            },
        ]
    )

    assert parser_input == "Document 3, Document 1"
    assert responder.events[0]["raw_output"].endswith("<|im_end|>")
    assert responder.events[0]["content_output"] == parser_input
    assert responder.events[0]["parser_input"] == parser_input
    assert responder.events[0]["special_tokens_removed"] is True
    assert responder.events[0]["format_valid"] is True
    assert responder.events[0]["no_document_policy"] == (
        official.TOURRANK_NO_DOCUMENT_POLICY
    )
    assert responder.events[0]["generation_attempt_count"] == 1
    assert responder.events[0]["retry_calls"] == 0
    assert responder.events[0]["format_repair_triggered"] is False
    assert responder.events[0]["format_repair_succeeded"] is False
    assert responder.events[0]["format_repair_policy"] == (
        official.TOURRANK_FORMAT_REPAIR_POLICY
    )


def test_tourrank_upstream_workdir_is_attempt_local_and_restored(
    monkeypatch, tmp_path
):
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    original = Path.cwd()
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    with tour_runner.isolated_upstream_workdir(attempt):
        assert Path.cwd() == attempt.resolve()
        Path("debug.txt").write_text("isolated\n")
    assert Path.cwd() == original
    assert (attempt / "debug.txt").read_text() == "isolated\n"
    assert not (original / "debug.txt").exists()


def test_tourrank_no_document_response_gets_one_charged_format_repair():
    class Backend:
        calls = []
        responses = iter(
            (
                {
                    "raw_output": "No listed passage answers the query.</s>",
                    "content_output": "No listed passage answers the query.",
                    "wall_seconds": 0.5,
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "rendered_prompt_sha256": "a" * 64,
                },
                {
                    "raw_output": "Document 4, Document 2</s>",
                    "content_output": "Document 4, Document 2",
                    "wall_seconds": 0.7,
                    "prompt_tokens": 14,
                    "completion_tokens": 6,
                    "rendered_prompt_sha256": "b" * 64,
                },
            )
        )

        @classmethod
        def generate(cls, messages, **_kwargs):
            cls.calls.append(messages)
            return next(cls.responses)

    responder = tour_runner.InstrumentedResponder(Backend())
    responder.begin_tournament(0)
    messages = [
        {"role": "system", "content": "rank"},
        {
            "role": "user",
            "content": "Compare the query and 5 documents; select the 2 documents.",
        },
    ]
    parser_input = responder(messages)

    assert parser_input == "Document 4, Document 2"
    assert len(Backend.calls) == 2
    assert Backend.calls[0] == messages
    assert Backend.calls[1][-2] == {
        "role": "assistant",
        "content": "No listed passage answers the query.",
    }
    assert "select the relatively best 2 documents" in Backend.calls[1][-1]["content"]
    event = responder.events[0]
    assert event["status"] == "ok"
    assert event["prompt_tokens"] == 24
    assert event["completion_tokens"] == 14
    assert event["duration_seconds"] == pytest.approx(1.2)
    assert event["retry_seconds"] == pytest.approx(0.7)
    assert event["generation_attempt_count"] == 2
    assert event["retry_calls"] == 1
    assert event["initial_format_valid"] is False
    assert event["format_valid"] is True
    assert event["format_repair_triggered"] is True
    assert event["format_repair_succeeded"] is True
    assert event["generation_attempts"][0]["content_output"].startswith(
        "No listed passage"
    )
    assert event["generation_attempts"][1]["content_output"] == parser_input


def test_tourrank_second_no_document_response_fails_without_synthetic_ranking():
    class Backend:
        calls = 0

        @classmethod
        def generate(cls, *_args, **_kwargs):
            cls.calls += 1
            return {
                "raw_output": "No listed passage answers the query.</s>",
                "content_output": "No listed passage answers the query.",
                "wall_seconds": 0.5,
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "rendered_prompt_sha256": f"{cls.calls}" * 64,
            }

    responder = tour_runner.InstrumentedResponder(Backend())
    responder.begin_tournament(0)
    with pytest.raises(tour_runner.TourRankFormatError, match="after one bounded"):
        responder(
            [
                {"role": "system", "content": "rank"},
                {
                    "role": "user",
                    "content": (
                        "Compare the query and 5 documents; select the 2 documents."
                    ),
                },
            ]
        )

    assert Backend.calls == 2
    event = responder.events[0]
    assert event["status"] == "failed"
    assert event["generation_attempt_count"] == 2
    assert event["retry_calls"] == 1
    assert event["format_repair_triggered"] is True
    assert event["format_repair_succeeded"] is False
    assert event["initial_format_valid"] is False
    assert event["format_valid"] is False
    assert event["parser_input"] == event["content_output"]
    assert event["special_tokens_removed"] is True
    assert event["no_document_policy"] == official.TOURRANK_NO_DOCUMENT_POLICY
    assert "parser_status" not in event


def test_tourrank_official_parser_valueerror_fallback_is_preserved(
    capsys, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)

    require_upstream("tourrank")
    module = official.load_tourrank_module()
    responder = tour_runner.InstrumentedResponder(object())
    responder.events = [{"n": 5, "m": 2}]
    instrumented = tour_runner.InstrumentedOfficialParser(
        module.get_top_M, responder, tmp_path / "debug.txt"
    )
    selected = instrumented(
        "Document INVALID, Document INVALID",
        N=5,
        M=2,
        groups_docid=["d1", "d2", "d3", "d4", "d5"],
    )
    assert selected == ["d2", "d3"]
    assert capsys.readouterr().out.count("ValueError occured in score") == 2
    assert responder.events[0]["parser_status"] == "ok"
    assert responder.events[0]["parser_value_error_fallbacks"] == 2
    assert responder.events[0]["parser_index_error_fallbacks"] == 0
    assert responder.events[0]["parser_selected_count"] == 2
    assert responder.events[0]["parser_unique_selected_count"] == 2
    assert responder.events[0]["parser_debug_delta_bytes"] > 0
    assert responder.events[0]["parser_contract_valid"] is True
    assert responder.events[0]["malformed_item_policy"] == (
        official.TOURRANK_MALFORMED_ITEM_POLICY
    )


def test_tourrank_official_parser_indexerror_ignore_is_preserved(
    capsys, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)

    require_upstream("tourrank")
    module = official.load_tourrank_module()
    responder = tour_runner.InstrumentedResponder(object())
    responder.events = [{"n": 5, "m": 2}]
    instrumented = tour_runner.InstrumentedOfficialParser(
        module.get_top_M, responder, tmp_path / "debug.txt"
    )
    selected = instrumented(
        "Document 1, ",
        N=5,
        M=2,
        groups_docid=["d1", "d2", "d3", "d4", "d5"],
    )

    assert selected == ["d1"]
    assert capsys.readouterr().out.count("IndexError occured in doc") == 1
    event = responder.events[0]
    assert event["parser_status"] == "ok"
    assert event["parser_value_error_fallbacks"] == 0
    assert event["parser_index_error_fallbacks"] == 1
    assert event["parser_selected_count"] == 1
    assert event["parser_under_selection_slots"] == 1
    assert event["parser_contract_valid"] is True
    assert event["malformed_item_policy"] == official.TOURRANK_MALFORMED_ITEM_POLICY


def test_tourrank_parser_preserves_official_duplicate_selection_semantics(tmp_path):
    responder = tour_runner.InstrumentedResponder(object())
    responder.events = [{"n": 5, "m": 2}]
    instrumented = tour_runner.InstrumentedOfficialParser(
        lambda *_args, **_kwargs: ["d1", "d1"],
        responder,
        tmp_path / "debug.txt",
    )

    selected = instrumented(
        "Document 1, Document 1",
        N=5,
        M=2,
        groups_docid=["d1", "d2", "d3", "d4", "d5"],
    )
    assert selected == ["d1", "d1"]
    assert responder.events[0]["parser_selected_count"] == 2
    assert responder.events[0]["parser_unique_selected_count"] == 1
    assert responder.events[0]["parser_duplicate_selection_slots"] == 1
    assert responder.events[0]["parser_selection_unique"] is False
    assert responder.events[0]["duplicate_selection_policy"] == (
        official.TOURRANK_DUPLICATE_SELECTION_POLICY
    )
    assert responder.events[0]["parser_selection_count_delta"] == 0
    assert responder.events[0]["parser_under_selection_slots"] == 0
    assert responder.events[0]["parser_over_selection_slots"] == 0
    assert responder.events[0]["parser_selection_count_matches_requested"] is True
    assert responder.events[0]["selection_cardinality_policy"] == (
        official.TOURRANK_SELECTION_CARDINALITY_POLICY
    )
    assert responder.events[0]["parser_contract_valid"] is True


@pytest.mark.parametrize(
    ("selected_count", "count_delta", "under_slots", "over_slots"),
    [(9, -1, 1, 0), (11, 1, 0, 1)],
)
def test_tourrank_parser_preserves_official_cardinality_semantics(
    tmp_path, selected_count, count_delta, under_slots, over_slots
):
    responder = tour_runner.InstrumentedResponder(object())
    responder.events = [{"n": 20, "m": 10}]
    selected_ids = [f"d{index}" for index in range(1, selected_count + 1)]
    instrumented = tour_runner.InstrumentedOfficialParser(
        lambda *_args, **_kwargs: selected_ids,
        responder,
        tmp_path / "debug.txt",
    )

    selected = instrumented(
        ", ".join(f"Document {index}" for index in range(1, selected_count + 1)),
        N=20,
        M=10,
        groups_docid=[f"d{index}" for index in range(1, 21)],
    )

    assert selected == selected_ids
    event = responder.events[0]
    assert event["parser_selected_count"] == selected_count
    assert event["parser_unique_selected_count"] == selected_count
    assert event["parser_selection_count_delta"] == count_delta
    assert event["parser_under_selection_slots"] == under_slots
    assert event["parser_over_selection_slots"] == over_slots
    assert event["parser_selection_count_matches_requested"] is False
    assert event["parser_contract_valid"] is True


def test_tourrank_finalizer_validates_charged_format_repair(tmp_path):
    class Backend:
        responses = iter(
            (
                {
                    "raw_output": "None are relevant.</s>",
                    "content_output": "None are relevant.",
                    "wall_seconds": 0.5,
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "rendered_prompt_sha256": "a" * 64,
                },
                {
                    "raw_output": "Document 4, Document 2</s>",
                    "content_output": "Document 4, Document 2",
                    "wall_seconds": 0.7,
                    "prompt_tokens": 14,
                    "completion_tokens": 6,
                    "rendered_prompt_sha256": "b" * 64,
                },
            )
        )

        @classmethod
        def generate(cls, *_args, **_kwargs):
            return next(cls.responses)

    responder = tour_runner.InstrumentedResponder(Backend())
    responder.begin_tournament(0)
    parser_input = responder(
        [
            {"role": "system", "content": "rank"},
            {
                "role": "user",
                "content": "Compare the query and 5 documents; select the 2 documents.",
            },
        ]
    )
    parser = tour_runner.InstrumentedOfficialParser(
        lambda *_args, **_kwargs: ["d4", "d2"],
        responder,
        tmp_path / "debug.txt",
    )
    assert parser(
        parser_input,
        N=5,
        M=2,
        groups_docid=["d1", "d2", "d3", "d4", "d5"],
    ) == ["d4", "d2"]
    event = responder.events[0]
    query_wall_seconds = 2.0
    row = {
        "nominal_calls": 1,
        "total_calls": 1,
        "retry_calls": 1,
        "retry_inclusive_llm_calls": 2,
        "generation_attempt_count": 2,
        "events": [event],
        "prompt_tokens": event["prompt_tokens"],
        "completion_tokens": event["completion_tokens"],
        "request_seconds_sum": event["duration_seconds"],
        "query_wall_seconds": query_wall_seconds,
        "retry_seconds": event["retry_seconds"],
        "retry_excluded_wall_seconds": query_wall_seconds - event["retry_seconds"],
        "special_token_stripped_calls": 1,
        "special_token_stripped_generations": 2,
        "format_failure_calls": 0,
        "initial_format_failure_calls": 1,
        "format_repair_calls": 1,
        "format_repair_successes": 1,
        "parser_value_error_fallbacks": 0,
        "parser_index_error_fallbacks": 0,
        "parser_duplicate_selection_calls": 0,
        "parser_duplicate_selection_slots": 0,
        "parser_under_selection_calls": 0,
        "parser_under_selection_slots": 0,
        "parser_over_selection_calls": 0,
        "parser_over_selection_slots": 0,
        "parser_input_policy": official.TOURRANK_PARSER_INPUT_POLICY,
        "no_document_policy": official.TOURRANK_NO_DOCUMENT_POLICY,
        "debug_output_policy": official.TOURRANK_DEBUG_OUTPUT_POLICY,
        "duplicate_selection_policy": official.TOURRANK_DUPLICATE_SELECTION_POLICY,
        "selection_cardinality_policy": (
            official.TOURRANK_SELECTION_CARDINALITY_POLICY
        ),
        "malformed_item_policy": official.TOURRANK_MALFORMED_ITEM_POLICY,
        "format_repair_policy": official.TOURRANK_FORMAT_REPAIR_POLICY,
    }
    prompt_template = official.TOURRANK_FORMAT_REPAIR_PROMPT_TEMPLATE
    manifest = {
        "generation_budget": 512,
        "parser_input_policy": official.TOURRANK_PARSER_INPUT_POLICY,
        "no_document_policy": official.TOURRANK_NO_DOCUMENT_POLICY,
        "debug_output_policy": official.TOURRANK_DEBUG_OUTPUT_POLICY,
        "duplicate_selection_policy": official.TOURRANK_DUPLICATE_SELECTION_POLICY,
        "selection_cardinality_policy": (
            official.TOURRANK_SELECTION_CARDINALITY_POLICY
        ),
        "malformed_item_policy": official.TOURRANK_MALFORMED_ITEM_POLICY,
        "format_repair_policy": official.TOURRANK_FORMAT_REPAIR_POLICY,
        "max_format_retries": 1,
        "format_repair_prompt_template": prompt_template,
        "format_repair_prompt_sha256": hashlib.sha256(
            prompt_template.encode("utf-8")
        ).hexdigest(),
        "parser_debug_artifact": {
            "sha256": hashlib.sha256(b"").hexdigest(),
            "bytes": 0,
            "entries": 0,
        },
    }
    official_finalizer.validate_tourrank_generation_contract(
        manifest, {"q1": row}, tmp_path / "debug.txt"
    )

    row["retry_calls"] = 0
    with pytest.raises(ValueError, match="retry-call aggregate mismatch"):
        official_finalizer.validate_tourrank_generation_contract(
            manifest, {"q1": row}, tmp_path / "debug.txt"
        )


def test_tourrank_finalizer_rejects_hidden_parser_transport_drift(tmp_path):
    event = {
        "status": "ok",
        "n": 4,
        "m": 2,
        "raw_output": "Document 3, Document 1</s>",
        "content_output": "Document 3, Document 1",
        "parser_input": "Document 3, Document 1",
        "special_tokens_removed": True,
        "format_valid": True,
        "parser_input_policy": official.TOURRANK_PARSER_INPUT_POLICY,
        "no_document_policy": official.TOURRANK_NO_DOCUMENT_POLICY,
        "parser_status": "ok",
        "parser_value_error_fallbacks": 0,
        "parser_index_error_fallbacks": 0,
        "parser_debug_delta_sha256": hashlib.sha256(b"").hexdigest(),
        "parser_debug_delta_bytes": 0,
        "parser_selected_count": 2,
        "parser_unique_selected_count": 2,
        "parser_duplicate_selection_slots": 0,
        "parser_selection_unique": True,
        "duplicate_selection_policy": official.TOURRANK_DUPLICATE_SELECTION_POLICY,
        "parser_selection_count_delta": 0,
        "parser_under_selection_slots": 0,
        "parser_over_selection_slots": 0,
        "parser_selection_count_matches_requested": True,
        "selection_cardinality_policy": (
            official.TOURRANK_SELECTION_CARDINALITY_POLICY
        ),
        "malformed_item_policy": official.TOURRANK_MALFORMED_ITEM_POLICY,
        "parser_contract_valid": True,
    }
    row = {
        "total_calls": 1,
        "events": [event],
        "special_token_stripped_calls": 1,
        "format_failure_calls": 0,
        "parser_value_error_fallbacks": 0,
        "parser_index_error_fallbacks": 0,
        "parser_duplicate_selection_calls": 0,
        "parser_duplicate_selection_slots": 0,
        "parser_under_selection_calls": 0,
        "parser_under_selection_slots": 0,
        "parser_over_selection_calls": 0,
        "parser_over_selection_slots": 0,
        "parser_input_policy": official.TOURRANK_PARSER_INPUT_POLICY,
        "no_document_policy": official.TOURRANK_NO_DOCUMENT_POLICY,
        "debug_output_policy": official.TOURRANK_DEBUG_OUTPUT_POLICY,
        "duplicate_selection_policy": official.TOURRANK_DUPLICATE_SELECTION_POLICY,
        "selection_cardinality_policy": (
            official.TOURRANK_SELECTION_CARDINALITY_POLICY
        ),
        "malformed_item_policy": official.TOURRANK_MALFORMED_ITEM_POLICY,
    }
    manifest = {
        "parser_input_policy": official.TOURRANK_PARSER_INPUT_POLICY,
        "no_document_policy": official.TOURRANK_NO_DOCUMENT_POLICY,
        "debug_output_policy": official.TOURRANK_DEBUG_OUTPUT_POLICY,
        "duplicate_selection_policy": official.TOURRANK_DUPLICATE_SELECTION_POLICY,
        "selection_cardinality_policy": (
            official.TOURRANK_SELECTION_CARDINALITY_POLICY
        ),
        "malformed_item_policy": official.TOURRANK_MALFORMED_ITEM_POLICY,
        "parser_debug_artifact": {
            "sha256": hashlib.sha256(b"").hexdigest(),
            "bytes": 0,
            "entries": 0,
        },
    }
    official_finalizer.validate_tourrank_generation_contract(
        manifest, {"q1": row}, tmp_path / "debug.txt"
    )

    event["parser_selected_count"] = 1
    event["parser_unique_selected_count"] = 1
    event["parser_duplicate_selection_slots"] = 0
    event["parser_selection_unique"] = True
    event["parser_selection_count_delta"] = -1
    event["parser_under_selection_slots"] = 1
    event["parser_selection_count_matches_requested"] = False
    row["parser_duplicate_selection_calls"] = 0
    row["parser_duplicate_selection_slots"] = 0
    row["parser_under_selection_calls"] = 1
    row["parser_under_selection_slots"] = 1
    official_finalizer.validate_tourrank_generation_contract(
        manifest, {"q1": row}, tmp_path / "debug.txt"
    )

    event["parser_selected_count"] = 2
    event["parser_unique_selected_count"] = 1
    event["parser_duplicate_selection_slots"] = 1
    event["parser_selection_unique"] = False
    event["parser_selection_count_delta"] = 0
    event["parser_under_selection_slots"] = 0
    event["parser_selection_count_matches_requested"] = True
    row["parser_duplicate_selection_calls"] = 1
    row["parser_duplicate_selection_slots"] = 1
    row["parser_under_selection_calls"] = 0
    row["parser_under_selection_slots"] = 0
    official_finalizer.validate_tourrank_generation_contract(
        manifest, {"q1": row}, tmp_path / "debug.txt"
    )

    event["parser_debug_delta_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="empty TourRank debug delta hash mismatch"):
        official_finalizer.validate_tourrank_generation_contract(
            manifest, {"q1": row}, tmp_path / "debug.txt"
        )
    event["parser_debug_delta_sha256"] = hashlib.sha256(b"").hexdigest()

    event["parser_input"] = event["raw_output"]
    with pytest.raises(ValueError, match="parser input was altered"):
        official_finalizer.validate_tourrank_generation_contract(
            manifest, {"q1": row}, tmp_path / "debug.txt"
        )


def test_tourrank_finalizer_accepts_accounted_official_parser_fallback(tmp_path):
    debug_text = (
        "New Error: \n[1]\nDocument X\n"
        "ValueError, 4, 2, Document X, Document 1\nend\n\n"
    )
    debug_path = tmp_path / "debug.txt"
    debug_path.write_text(debug_text)
    event = {
        "status": "ok",
        "n": 4,
        "m": 2,
        "raw_output": "Document X, Document 1</s>",
        "content_output": "Document X, Document 1",
        "parser_input": "Document X, Document 1",
        "special_tokens_removed": True,
        "format_valid": True,
        "parser_input_policy": official.TOURRANK_PARSER_INPUT_POLICY,
        "no_document_policy": official.TOURRANK_NO_DOCUMENT_POLICY,
        "parser_status": "ok",
        "parser_value_error_fallbacks": 1,
        "parser_index_error_fallbacks": 0,
        "parser_debug_delta_sha256": hashlib.sha256(debug_text.encode()).hexdigest(),
        "parser_debug_delta_bytes": len(debug_text.encode()),
        "parser_selected_count": 2,
        "parser_unique_selected_count": 2,
        "parser_duplicate_selection_slots": 0,
        "parser_selection_unique": True,
        "duplicate_selection_policy": official.TOURRANK_DUPLICATE_SELECTION_POLICY,
        "parser_selection_count_delta": 0,
        "parser_under_selection_slots": 0,
        "parser_over_selection_slots": 0,
        "parser_selection_count_matches_requested": True,
        "selection_cardinality_policy": (
            official.TOURRANK_SELECTION_CARDINALITY_POLICY
        ),
        "malformed_item_policy": official.TOURRANK_MALFORMED_ITEM_POLICY,
        "parser_contract_valid": True,
    }
    row = {
        "total_calls": 1,
        "events": [event],
        "special_token_stripped_calls": 1,
        "format_failure_calls": 0,
        "parser_value_error_fallbacks": 1,
        "parser_index_error_fallbacks": 0,
        "parser_duplicate_selection_calls": 0,
        "parser_duplicate_selection_slots": 0,
        "parser_under_selection_calls": 0,
        "parser_under_selection_slots": 0,
        "parser_over_selection_calls": 0,
        "parser_over_selection_slots": 0,
        "parser_input_policy": official.TOURRANK_PARSER_INPUT_POLICY,
        "no_document_policy": official.TOURRANK_NO_DOCUMENT_POLICY,
        "debug_output_policy": official.TOURRANK_DEBUG_OUTPUT_POLICY,
        "duplicate_selection_policy": official.TOURRANK_DUPLICATE_SELECTION_POLICY,
        "selection_cardinality_policy": (
            official.TOURRANK_SELECTION_CARDINALITY_POLICY
        ),
        "malformed_item_policy": official.TOURRANK_MALFORMED_ITEM_POLICY,
    }
    payload = debug_text.encode()
    manifest = {
        "parser_input_policy": official.TOURRANK_PARSER_INPUT_POLICY,
        "no_document_policy": official.TOURRANK_NO_DOCUMENT_POLICY,
        "debug_output_policy": official.TOURRANK_DEBUG_OUTPUT_POLICY,
        "duplicate_selection_policy": official.TOURRANK_DUPLICATE_SELECTION_POLICY,
        "selection_cardinality_policy": (
            official.TOURRANK_SELECTION_CARDINALITY_POLICY
        ),
        "malformed_item_policy": official.TOURRANK_MALFORMED_ITEM_POLICY,
        "parser_debug_artifact": {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "entries": 1,
        },
    }
    official_finalizer.validate_tourrank_generation_contract(
        manifest, {"q1": row}, debug_path
    )

    row["parser_value_error_fallbacks"] = 0
    with pytest.raises(ValueError, match="ValueError aggregate mismatch"):
        official_finalizer.validate_tourrank_generation_contract(
            manifest, {"q1": row}, debug_path
        )








def test_baseline_failures_preserve_partial_events_without_success_checkpoint(
    monkeypatch, tmp_path
):
    class FailingTourBackend:
        @staticmethod
        def generate(*_args, **_kwargs):
            raise RuntimeError("tour generation failed")

    responder = tour_runner.InstrumentedResponder(FailingTourBackend())
    responder.begin_tournament(0)
    with pytest.raises(RuntimeError, match="tour generation failed"):
        responder([{"role": "user", "content": "rank"}])
    assert responder.events[0]["status"] == "failed"
    tour_failures = tmp_path / "tourrank_failures.jsonl"
    tour_runner.record_tourrank_failure(
        tour_failures,
        qid="q1",
        model="Qwen/Qwen3.5-9B",
        protocol_hash_value="protocol",
        events=responder.events,
        wall_seconds=1.0,
        error=RuntimeError("tour generation failed"),
    )
    assert json.loads(tour_failures.read_text())["status"] == "failed"

    agent, _, _ = make_test_liu_agent(monkeypatch)

    def fail_liu(*_args, **_kwargs):
        raise RuntimeError("liu generation failed")

    agent.backend.generate = fail_liu
    with pytest.raises(RuntimeError, match="liu generation failed"):
        agent.run_llm("prompt", output_passages_num=2)
    assert agent.events[0]["status"] == "failed"
    liu_failures = tmp_path / "liu_failures.jsonl"
    liu_runner.record_liu_failure(
        liu_failures,
        qid="q1",
        model="Qwen/Qwen3.5-9B",
        protocol_hash_value="protocol",
        generation_events=agent.events,
        prompt_builds=agent.prompt_builds,
        wall_seconds=1.0,
        error=RuntimeError("liu generation failed"),
    )
    assert json.loads(liu_failures.read_text())["status"] == "failed"


def test_tourrank_mocked_responses_preserve_upstream_points_and_stable_ties(monkeypatch):
    require_upstream("tourrank")
    module = official.load_tourrank_module()
    monkeypatch.setattr(module.random, "shuffle", lambda values: None)
    responses = iter(("Document 3, Document 1", "Document 2, Document 1"))
    monkeypatch.setattr(module, "get_response", lambda _messages: next(responses))
    docids = ["d1", "d2", "d3", "d4"]
    contents = {docid: f"content {docid}" for docid in docids}
    point_rows = []
    module.group_processing(docids.copy(), "query", 4, 2, contents, point_rows)
    module.group_processing(docids.copy(), "query", 4, 2, contents, point_rows)
    assert point_rows == [{"d3": 1, "d1": 1}, {"d2": 1, "d1": 1}]

    actual = official.ordered_score_ranking(module, docids, point_rows)
    aggregate = {
        docid: sum(row.get(docid, 0) for row in point_rows)
        for docid in docids
    }
    expected = module.sort_docs_by_relevance(docids, [aggregate[docid] for docid in docids])
    assert actual == expected == ["d1", "d2", "d3", "d4"]


def test_liu_prompt_is_byte_identical_to_pinned_upstream_composition(monkeypatch):
    agent, data, utils = make_test_liu_agent(monkeypatch)
    result = data.Result(
        query=data.Query(qid="q1", text="FranÃ§ais query [7]"),
        candidates=[
            data.Candidate("d1", 2.0, {"title": "T1", "text": "alpha [12]"}),
            data.Candidate("d2", 1.0, {"text": "beta"}),
        ],
        ranking_exec_summary=[],
    )

    prompt, token_count = agent.create_prompt(result, 0, 2)
    query = agent._replace_number(result.query.text).strip()
    prefix = utils.add_prefix_prompt(agent.prompt_mode, query, 2)
    context = f"{prefix}\n"
    for rank, candidate in enumerate(result.candidates, start=1):
        content = utils.convert_doc_to_prompt_content(
            agent.backend.ranker.tokenizer, candidate.doc, 100, truncate_by_word=True
        )
        context += f"[{rank}] {content}\n"
    context += utils.add_post_prompt(agent.prompt_mode, True, query, 2)
    context = sys.modules["ftfy"].fix_text(context)
    assert prompt.encode("utf-8") == context.encode("utf-8")
    expected_messages = [
        {"role": "system", "content": liu_runner.SYSTEM_MESSAGE},
        {"role": "user", "content": context},
    ]
    assert agent.backend.last_messages == expected_messages
    expected_render = "CHAT:" + "|".join(
        f"{message['role']}:{message['content']}" for message in expected_messages
    )
    assert token_count == len(expected_render.encode("utf-8"))
    batched = agent.create_prompt_batched([result], 0, 2, batch_size=1)
    assert batched == [(prompt, token_count)]


@pytest.mark.parametrize(
    ("response", "expected_docids"),
    [
        ("[3] > [1] > [2]", ["d3", "d1", "d2"]),
        ("[2] > [2] > [1]", ["d2", "d1", "d3"]),
        ("[2]", ["d2", "d1", "d3"]),
        ("malformed output", ["d1", "d2", "d3"]),
        ("[9] > [2]", ["d2", "d1", "d3"]),
    ],
)
def test_liu_receive_permutation_matches_pinned_upstream_edge_cases(
    monkeypatch, response, expected_docids
):
    agent, data, _ = make_test_liu_agent(monkeypatch)
    result = data.Result(
        query=data.Query(qid="q1", text="query"),
        candidates=[
            data.Candidate("d1", 3.0, {"text": "one"}),
            data.Candidate("d2", 2.0, {"text": "two"}),
            data.Candidate("d3", 1.0, {"text": "three"}),
        ],
        ranking_exec_summary=[],
    )
    reranked = agent.receive_permutation(result, response, 0, 3)
    assert [candidate.docid for candidate in reranked.candidates] == expected_docids
    assert [candidate.score for candidate in reranked.candidates] == [3.0, 2.0, 1.0]
