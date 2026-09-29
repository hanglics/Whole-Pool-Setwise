#!/usr/bin/env python3
"""Run pinned Liu zero-shot ranking logic with paper open models."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
import types
from contextlib import contextmanager
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
    add_fullrank_to_path,
    hash_ir_dataset_qrels,
    isolated_upstream_imports,
    load_trec,
    paper_model_revisions,
    require_paper_model_revision,
    sanitized_protocol_argv,
    validate_ordered_pools,
    write_trec,
)
from open_model_backend import OpenModelBackend


UPSTREAM_COMMIT = "5e74c7906d7f15c31bfc10c44c4a028318d0ebe7"
CONTEXT_SIZE = 32768
SYSTEM_MESSAGE = (
    "You are RankLLM, an intelligent assistant that can rank passages based on "
    "their relevancy to the query."
)
PAPER_MODELS = frozenset(paper_model_revisions())
DATASETS = {
    "dl19": "msmarco-passage/trec-dl-2019/judged",
    "dl20": "msmarco-passage/trec-dl-2020/judged",
}
PASSAGE_SOURCE_POLICY = (
    "local_ir_datasets_official_liu_100_word_prompt_pipeline_for_"
    "cross_method_comparability"
)


@contextmanager
def suppress_upstream_bytecode():
    """Use source-only imports without dirtying the pinned upstream checkout."""
    with isolated_upstream_imports():
        yield


def load_upstream_components():
    """Import only the pinned Liu instruction/parser/reranker closure we execute."""
    with suppress_upstream_bytecode():
        add_fullrank_to_path()
        import importlib

        pyserini_search = importlib.import_module("pyserini.search")
        if not hasattr(pyserini_search, "JLuceneSearcherResult"):
            pyserini_search.JLuceneSearcherResult = types.SimpleNamespace
        from data import Candidate, Query, Request
        from rerank.rankllm import RankLLM
        from rerank.reranker import Reranker
        import utils

    return Candidate, Query, Request, RankLLM, Reranker, utils


def build_official_requests(dataset_tag: str):
    with suppress_upstream_bytecode():
        Candidate, Query, Request, *_ = load_upstream_components()
        from index_and_topics import THE_INDEX, THE_TOPICS
    from pyserini.search._base import get_topics
    from pyserini.search.lucene import LuceneSearcher

    official_run = UPSTREAM_ROOT / "liu_fullrank" / "runs" / dataset_tag / "bm25_top100.txt"
    pools = load_trec(official_run, 100)
    topics = {str(qid): row["title"] for qid, row in get_topics(THE_TOPICS[dataset_tag]).items()}
    searcher = LuceneSearcher.from_prebuilt_index(THE_INDEX[dataset_tag])
    requests = []
    official_docs = {}
    for qid, rows in pools.items():
        candidates = []
        official_docs[qid] = []
        for row in rows:
            document = json.loads(searcher.doc(row["docid"]).raw())
            candidates.append(Candidate(docid=row["docid"], score=row["score"], doc=document))
            text = document.get("text") or document.get("contents") or document.get("passage")
            if document.get("title"):
                text = f"{document['title']} {text}"
            official_docs[qid].append({**row, "content": text})
        requests.append(Request(query=Query(qid=qid, text=topics[qid]), candidates=candidates))
    return requests, topics, official_docs, official_run


def local_liu_records(dataset_name: str, local_run: Path) -> tuple[dict, dict]:
    import ir_datasets

    dataset = ir_datasets.load(dataset_name)
    queries = {str(query.query_id): query.text for query in dataset.queries_iter()}
    docstore = dataset.docs_store()
    pools = load_trec(local_run, 100)
    local = {}
    for qid, rows in pools.items():
        output = []
        for row in rows:
            doc = docstore.get(row["docid"])
            title = doc.title if hasattr(doc, "title") and doc.title else None
            text = doc.text
            content = f"{title} {text}" if title else text
            output.append(
                {
                    **row,
                    "content": content,
                    "_local_text": text,
                    "_local_title": title,
                }
            )
        local[qid] = output
    return queries, local


def _replace_candidate_content(document: dict, local_row: dict) -> dict:
    """Preserve the released document schema but replace its passage payload."""
    updated = copy.deepcopy(document)
    content_fields = [
        field
        for field in ("text", "segment", "contents", "body", "passage")
        if field in updated
    ]
    if not content_fields:
        raise ValueError("Liu candidate document has no supported passage field.")
    for field in content_fields:
        updated[field] = local_row["_local_text"]
    if "title" in updated:
        updated["title"] = local_row["_local_title"] or ""
    return updated


def use_local_liu_passages(requests: list, local_pools: dict) -> list:
    adapted = copy.deepcopy(requests)
    for request in adapted:
        qid = str(request.query.qid)
        local_rows = local_pools[qid]
        if len(request.candidates) != len(local_rows):
            raise ValueError(f"Liu qid={qid}: candidate depth changed during adaptation.")
        for candidate, local_row in zip(request.candidates, local_rows):
            if str(candidate.docid) != str(local_row["docid"]):
                raise ValueError(f"Liu qid={qid}: candidate order changed during adaptation.")
            candidate.doc = _replace_candidate_content(candidate.doc, local_row)
    return adapted


def inference_requests_sha256(requests: list) -> str:
    digest = hashlib.sha256()
    for request in sorted(requests, key=lambda row: str(row.query.qid)):
        payload = {
            "qid": str(request.query.qid),
            "query": request.query.text,
            "candidates": [
                {
                    "docid": str(candidate.docid),
                    "score": float(candidate.score),
                    "doc": candidate.doc,
                }
                for candidate in request.candidates
            ],
        }
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


def prepare_local_liu_input(
    requests: list,
    topics: dict,
    official_docs: dict,
    local_queries: dict,
    local_pools: dict,
) -> tuple[dict, list | None, list[dict]]:
    discrepancies = validate_ordered_pools(
        official_docs, local_pools, score_tolerance=1e-6
    )
    for qid in sorted(set(topics) | set(local_queries)):
        if qid not in official_docs and qid not in local_pools:
            continue
        if qid not in topics or topics[qid] != local_queries.get(qid):
            discrepancies.append({"qid": qid, "field": "query"})
    fatal = [row for row in discrepancies if row["field"] != "content"]
    candidate_count = sum(len(rows) for rows in official_docs.values())
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
        "preflight_query_count": len(official_docs),
        "preflight_candidate_count": candidate_count,
        "official_index_used_for_candidate_structure": True,
        "official_index_content_used_for_inference": False,
        "local_ir_datasets_content_used_for_inference": True,
    }
    if fatal:
        return validation, None, discrepancies
    adapted = use_local_liu_passages(requests, local_pools)
    validation["local_untruncated_input_sha256"] = inference_requests_sha256(
        adapted
    )
    return validation, adapted, discrepancies


def persist_discrepancy_report(path: Path, discrepancies: list[dict]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != discrepancies:
            raise ValueError(
                f"Liu discrepancy report drift at {path}; use a new attempt."
            )
        return
    atomic_write_json(path, discrepancies)


def assert_validate_only_output_safe(output_dir: Path) -> None:
    if (output_dir / "checkpoints" / "protocol.json").exists():
        raise ValueError(
            "--validate-only refuses an experiment attempt directory; "
            "use a dedicated preflight directory."
        )


def make_model_matched_agent_class(rank_llm_class, upstream_utils):
    class ModelMatchedLiuAgent(rank_llm_class):
        def __init__(self, backend: OpenModelBackend):
            model_limit = int(backend.ranker.max_input_tokens or 0)
            if model_limit < CONTEXT_SIZE:
                raise ValueError(
                    f"Liu zero-shot requires the official {CONTEXT_SIZE}-token context; "
                    f"the model advertises {model_limit}."
                )
            super().__init__(
                backend.model,
                CONTEXT_SIZE,
                "rank_GPT",
                0,
            )
            self.backend = backend
            self.max_passage_length = 100
            self.prompt_mode = "rank_GPT"
            self.variable_passages = True
            self.window_size = 100
            self.system_message = SYSTEM_MESSAGE
            self.events: list[dict] = []
            self.prompt_builds: list[dict] = []

        def num_output_tokens(self, current_window_size=None):
            count = current_window_size or self.window_size
            permutation = " > ".join(f"[{index + 1}]" for index in range(count))
            return len(self.backend.ranker.tokenizer.encode(permutation))

        def get_num_tokens(self, prompt):
            return self.backend.count_prompt_tokens(
                [
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt},
                ]
            )

        def create_prompt(self, result, rank_start, rank_end):
            from ftfy import fix_text

            query = self._replace_number(result.query.text).strip()
            count = len(result.candidates[rank_start:rank_end])
            max_length = self.max_passage_length
            while True:
                prefix = upstream_utils.add_prefix_prompt(self.prompt_mode, query, count)
                context = f"{prefix}\n"
                for rank, candidate in enumerate(
                    result.candidates[rank_start:rank_end], start=1
                ):
                    content = upstream_utils.convert_doc_to_prompt_content(
                        self.backend.ranker.tokenizer,
                        candidate.doc,
                        max_length,
                        truncate_by_word=True,
                    )
                    context += f"[{rank}] {content}\n"
                context += upstream_utils.add_post_prompt(
                    self.prompt_mode, self.variable_passages, query, count
                )
                # The official open-model wrapper normalizes the composed prompt
                # with ftfy after adding the conversation envelope. The envelope
                # is intentionally model-native here, so normalize the complete
                # official inner instruction before native rendering.
                context = fix_text(context)
                tokens = self.get_num_tokens(context)
                if tokens <= self.max_tokens() - self.num_output_tokens(count):
                    self.prompt_builds.append(
                        {
                            "passage_word_cap": max_length,
                            "rendered_prompt_tokens": tokens,
                            "reserved_output_tokens": self.num_output_tokens(count),
                            "passages": count,
                        }
                    )
                    return context, tokens
                max_length -= max(
                    1,
                    (tokens - self.max_tokens() + self.num_output_tokens(count))
                    // max(1, count * 4),
                )
                if max_length < 1:
                    raise ValueError("Liu prompt does not fit even at one word per passage.")

        def create_prompt_batched(self, results, rank_start, rank_end, batch_size=32):
            return [self.create_prompt(result, rank_start, rank_end) for result in results]

        def run_llm(self, prompt, output_passages_num=None):
            generation_budget = self.num_output_tokens(output_passages_num)
            started = time.perf_counter()
            try:
                generated = self.backend.generate(
                    [
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": prompt},
                    ],
                    max_new_tokens=generation_budget,
                    min_new_tokens=generation_budget,
                )
            except Exception as exc:
                self.events.append(
                    {
                        "status": "failed",
                        "generation_budget": generation_budget,
                        "max_new_tokens": generation_budget,
                        "min_new_tokens": generation_budget,
                        "wall_seconds": time.perf_counter() - started,
                        "exception_class": type(exc).__name__,
                        "exception_message": str(exc),
                    }
                )
                raise
            generated["generation_budget"] = generation_budget
            self.events.append(generated)
            return generated["raw_output"], generated["completion_tokens"]

        def run_llm_batched(self, prompts, output_passages_num=None):
            return [self.run_llm(prompt, output_passages_num) for prompt in prompts]

        def cost_per_1k_token(self, input_token):
            return 0.0

    return ModelMatchedLiuAgent


def checkpoint_path(root: Path, qid: str) -> Path:
    digest = hashlib.sha256(str(qid).encode("utf-8")).hexdigest()[:16]
    return root / f"query-{digest}.json"


def record_liu_failure(
    path: Path,
    *,
    qid: str,
    model: str,
    protocol_hash_value: str,
    generation_events: list[dict],
    prompt_builds: list[dict],
    wall_seconds: float,
    error: Exception,
) -> None:
    append_jsonl(
        path,
        {
            "schema_version": 1,
            "qid": qid,
            "method": "Liu et al. zero-shot model-matched",
            "model": model,
            "protocol_hash": protocol_hash_value,
            "status": "failed",
            "query_wall_seconds": wall_seconds,
            "generation_events": generation_events,
            "prompt_builds": prompt_builds,
            "exception_class": type(error).__name__,
            "exception_message": str(error),
            "occurred_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--local-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=sorted(PAPER_MODELS))
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()
    if args.max_queries is not None and args.max_queries < 1:
        raise ValueError("--max-queries must be positive when provided.")
    require_paper_model_revision(args.model, args.model_revision)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.validate_only:
        assert_validate_only_output_safe(args.output_dir)

    subprocess.run(
        [sys.executable, str(HERE / "fetch_official_baselines.py"), "--baseline", "liu_fullrank", "--verify-only"],
        check=True,
    )
    official_requests, topics, official_docs, official_run = build_official_requests(
        args.dataset
    )
    discrepancy_path = args.output_dir / "input_discrepancies.json"
    local_queries, local_pools = local_liu_records(
        DATASETS[args.dataset],
        args.local_run,
    )
    input_validation, requests, discrepancies = prepare_local_liu_input(
        official_requests,
        topics,
        official_docs,
        local_queries,
        local_pools,
    )
    if input_validation["structural_discrepancies"]:
        if (args.output_dir / "checkpoints" / "protocol.json").exists():
            raise ValueError(
                "Liu input fidelity failed for an existing attempt; its "
                "artifacts were not modified. Use a new preflight directory."
            )
        persist_discrepancy_report(discrepancy_path, discrepancies)
        raise ValueError(f"Liu input preflight failed; see {discrepancy_path}")
    assert requests is not None
    if args.validate_only:
        persist_discrepancy_report(discrepancy_path, discrepancies)
        print(
            f"Validated {len(requests)} Liu zero-shot input queries; "
            f"content discrepancies={input_validation['content_discrepancies']}; "
            "local ir_datasets passage text selected for the official Liu "
            "prompt pipeline."
        )
        return
    if args.max_queries is not None:
        requests = requests[: args.max_queries]

    backend = OpenModelBackend(
        args.model, args.model_revision, cache_dir=args.cache_dir, device="cuda"
    )
    _, _, _, RankLLM, Reranker, upstream_utils = load_upstream_components()
    Agent = make_model_matched_agent_class(RankLLM, upstream_utils)
    agent = Agent(backend)
    reranker = Reranker(agent)
    checkpoint_root = args.output_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    protocol_path = checkpoint_root / "protocol.json"
    protocol = {
        "schema_version": 1,
        "baseline": "Liu et al. zero-shot model-matched adaptation",
        "upstream_commit": UPSTREAM_COMMIT,
        "dataset": args.dataset,
        "model": args.model,
        "model_revision": args.model_revision,
        "tokenizer_revision": args.model_revision,
        "backend_contract": backend.contract(),
        "output_token_reserve_rule": "tokenizer length of '[1] > ... > [N]'",
        "generation_min_equals_max": True,
        "prompt_mode": "rank_GPT",
        "system_message": SYSTEM_MESSAGE,
        "context_size": CONTEXT_SIZE,
        "window_size": 100,
        "retrieval_num": 100,
        "num_passes": 1,
        "inference_engine": "transformers_generate_via_setwise_loader",
        "batching": "per_query_batch_size_1",
        "gpus": 1,
        "official_batched_latency_comparable": False,
        "max_passage_length": 100,
        "variable_passages": True,
        "official_run_sha256": sha256_file(official_run),
        "local_run_sha256": sha256_file(args.local_run),
        "inference_input_sha256": inference_requests_sha256(requests),
        "inference_query_count": len(requests),
        "inference_candidate_count": sum(
            len(request.candidates) for request in requests
        ),
        "passage_source": DATASETS[args.dataset],
        "passage_source_policy": PASSAGE_SOURCE_POLICY,
        "official_index_content_used_for_inference": False,
        "local_ir_datasets_content_used_for_inference": True,
        "output_objective": "full",
        "output_depth": 100,
        "query_limit": args.max_queries,
        "command_argv": sanitized_protocol_argv(sys.argv),
        "source_snapshot_sha256": source_snapshot()["sha256"],
        "qrels_id": "dl19-passage" if args.dataset == "dl19" else "dl20-passage",
        "qrels_sha256": hash_ir_dataset_qrels(DATASETS[args.dataset]),
    }
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["started_at_utc"] = existing.get("started_at_utc")
        protocol["protocol_hash"] = protocol_hash(protocol)
        if existing != protocol:
            raise ValueError("Liu checkpoint protocol mismatch; use a new attempt.")
        if not args.resume:
            raise FileExistsError("Liu checkpoints exist; pass --resume.")
    else:
        protocol["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        protocol["protocol_hash"] = protocol_hash(protocol)
        atomic_write_json(protocol_path, protocol)
    persist_discrepancy_report(discrepancy_path, discrepancies)

    checkpoint_paths = {}
    failure_path = args.output_dir / "liu_zero_shot_failures.jsonl"
    for request in requests:
        qid = str(request.query.qid)
        path = checkpoint_path(checkpoint_root, qid)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("protocol_hash") != protocol["protocol_hash"]:
                raise ValueError(f"Liu checkpoint hash mismatch for qid={qid}")
            checkpoint_paths[qid] = path
            continue
        agent.events = []
        agent.prompt_builds = []
        started = time.perf_counter()
        try:
            results, upstream_seconds = reranker.rerank_batch(
                [request],
                rank_start=0,
                rank_end=100,
                window_size=100,
                step=10,
                shuffle_candidates=False,
                vllm_batched=False,
            )
            wall = time.perf_counter() - started
            if len(results) != 1 or len(agent.events) != 1:
                raise RuntimeError(f"Liu qid={qid}: expected exactly one full-list generation")
        except Exception as exc:
            record_liu_failure(
                failure_path,
                qid=qid,
                model=args.model,
                protocol_hash_value=protocol["protocol_hash"],
                generation_events=list(agent.events),
                prompt_builds=list(agent.prompt_builds),
                wall_seconds=time.perf_counter() - started,
                error=exc,
            )
            raise
        result = results[0]
        executions = [
            {
                "prompt": info.prompt,
                "response": info.response,
                "input_token_count": info.input_token_count,
                "output_token_count": info.output_token_count,
            }
            for info in result.ranking_exec_summary
        ]
        atomic_write_json(
            path,
            {
                "schema_version": 1,
                "qid": qid,
                "protocol_hash": protocol["protocol_hash"],
                "docids": [candidate.docid for candidate in result.candidates],
                "executions": executions,
                "generation_events": agent.events,
                "prompt_builds": agent.prompt_builds,
                "query_wall_seconds": wall,
                "upstream_pipeline_seconds": upstream_seconds,
            },
        )
        checkpoint_paths[qid] = path

    payloads = [
        json.loads(checkpoint_paths[str(request.query.qid)].read_text(encoding="utf-8"))
        for request in requests
    ]
    rankings = {payload["qid"]: payload["docids"] for payload in payloads}
    write_trec(args.output_dir / "liu_zero_shot.txt", rankings, "Liu-zero-shot-matched")
    raw_path = args.output_dir / "liu_raw_permutations.jsonl"
    telemetry_path = args.output_dir / "liu_zero_shot_telemetry.jsonl"
    raw_path.write_text(
        "".join(
            json.dumps(
                {
                    "qid": payload["qid"],
                    "executions": payload["executions"],
                    "generation_events": payload["generation_events"],
                    "prompt_builds": payload["prompt_builds"],
                },
                sort_keys=True,
            )
            + "\n"
            for payload in payloads
        ),
        encoding="utf-8",
    )
    telemetry_path.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": 1,
                    "qid": payload["qid"],
                    "method": "Liu et al. zero-shot model-matched",
                    "model": args.model,
                    "output_objective": "full",
                    "output_depth": 100,
                    "llm_calls": len(payload["executions"]),
                    "prompt_tokens": sum(
                        row["input_token_count"] for row in payload["executions"]
                    ),
                    "completion_tokens": sum(
                        row["output_token_count"] for row in payload["executions"]
                    ),
                    "query_wall_seconds": payload["query_wall_seconds"],
                    "batch_id": None,
                    "batch_size": 1,
                    "batch_wall_seconds": None,
                    "amortized_seconds_per_query": payload["query_wall_seconds"],
                    "status": "ok",
                },
                sort_keys=True,
            )
            + "\n"
            for payload in payloads
        ),
        encoding="utf-8",
    )
    manifest = adaptation_manifest(
        "Liu et al. zero-shot model-matched adaptation",
        UPSTREAM_COMMIT,
        [
            {"field": "ranking_logic", "changed": False, "reason": "official Reranker and sliding-window logic executed"},
            {"field": "prompt_text", "changed": False, "reason": "official rank_GPT prefix/postfix and document conversion executed"},
            {"field": "system_message", "changed": False, "reason": "official RankLLM system instruction preserved"},
            {"field": "parsing", "changed": False, "reason": "official receive_permutation executed"},
            {"field": "generation", "changed": True, "reason": "paper open model with native chat template"},
            {"field": "chat_template", "changed": True, "reason": "model-native template replaces Mistral FastChat envelope"},
            {"field": "inference_engine", "changed": True, "reason": "Transformers generation replaces released vLLM effectiveness path"},
            {"field": "batching_parallelism", "changed": True, "reason": "per-query single-GPU execution replaces released four-GPU batching"},
            {
                "field": "input_plumbing",
                "changed": True,
                "reason": (
                    "official candidate structure validated; local ir_datasets "
                    "passage text passed through the official Liu document "
                    "conversion pipeline"
                ),
            },
        ],
    )
    manifest.update(
        {
            "algorithm_faithful": True,
            "official_model_reproduction": False,
            "external_api": False,
            "training_required": False,
            "setting": "zero_shot",
            "dataset": args.dataset,
            "model": args.model,
            "model_revision": args.model_revision,
            "tokenizer_revision": args.model_revision,
            "backend_contract": backend.contract(),
            "output_token_reserve_rule": "tokenizer length of '[1] > ... > [N]'",
            "generation_min_equals_max": True,
            "prompt_mode": "rank_GPT",
            "system_message": SYSTEM_MESSAGE,
            "context_size": CONTEXT_SIZE,
            "window_size": 100,
            "inference_engine": "transformers_generate_via_setwise_loader",
            "batching": "per_query_batch_size_1",
            "gpus": 1,
            "official_batched_latency_comparable": False,
            "output_objective": "full",
            "output_depth": 100,
            "query_limit": args.max_queries,
            "baseline_category": "model_matched_adapted",
            "protocol_hash": protocol["protocol_hash"],
            "official_run_sha256": protocol["official_run_sha256"],
            "local_run_sha256": protocol["local_run_sha256"],
            "inference_input_sha256": protocol["inference_input_sha256"],
            "inference_query_count": protocol["inference_query_count"],
            "inference_candidate_count": protocol["inference_candidate_count"],
            "passage_source": protocol["passage_source"],
            "passage_source_policy": protocol["passage_source_policy"],
            "official_index_content_used_for_inference": False,
            "local_ir_datasets_content_used_for_inference": True,
            "max_passage_length": protocol["max_passage_length"],
            "source_snapshot_sha256": protocol["source_snapshot_sha256"],
            "qrels_id": protocol["qrels_id"],
            "qrels_sha256": protocol["qrels_sha256"],
            "command_argv": protocol["command_argv"],
            "started_at_utc": protocol["started_at_utc"],
            "input_validation": input_validation,
        }
    )
    atomic_write_json(args.output_dir / "protocol_manifest.json", manifest)


if __name__ == "__main__":
    main()
