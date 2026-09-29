#!/usr/bin/env python3
"""Tokenizer-only BEIR pool=100 context-fit probe for the paper experiments."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import ir_datasets
except ModuleNotFoundError as exc:
    raise SystemExit("Missing dependency: install ir_datasets before running this probe.") from exc
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from llmrankers._processor_adapter import ProcessorTokenizerAdapter
from llmrankers.setwise import _is_multimodal_config

FAMILIES = {
    "qwen3.5": "Qwen/Qwen3.5-9B",
    "llama3.1": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "ministral3": "mistralai/Ministral-3-8B-Instruct-2512",
}
DATASETS = {
    "beir-dbpedia": "beir/dbpedia-entity/test",
    "beir-nfcorpus": "beir/nfcorpus/test",
    "beir-scifact": "beir/scifact/test",
    "beir-trec-covid": "beir/trec-covid",
    "beir-touche2020": "beir/webis-touche2020/v2",
    "beir-fiqa": "beir/fiqa/test",
}
FIELDS = "family,dataset,p95_passage_tokens,prompt_tokens,max_pos_emb,headroom,verdict".split(",")
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(FAMILIES), default=None)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default=None)
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--passage-length", type=int, default=512)
    parser.add_argument("--query-length", type=int, default=128)
    parser.add_argument("--reserved-output-tokens", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=Path("results/probe_beir_pool100_fit.csv"))
    return parser.parse_args()
def text_of(obj, attrs):
    values = [str(getattr(obj, attr, "")).strip() for attr in attrs]
    return " ".join(value for value in values if value) or str(obj)
def max_pos(config, tokenizer):
    for attr in ("max_position_embeddings", "n_positions"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        for attr in ("max_position_embeddings", "n_positions"):
            value = text_config.get(attr) if isinstance(text_config, dict) else getattr(text_config, attr, None)
            if isinstance(value, int) and value > 0:
                return value
    value = getattr(tokenizer, "model_max_length", None)
    if isinstance(value, int) and 0 < value < 10**8:
        return value
    raise ValueError("could not resolve max_position_embeddings")
def chat_kwargs(config):
    qwen_types = {"qwen2", "qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_moe"}
    return {"enable_thinking": False} if getattr(config, "model_type", "") in qwen_types else {}
def enc(tokenizer, text, max_length=None):
    kwargs = {"add_special_tokens": False}
    if max_length is not None:
        kwargs.update({"truncation": True, "max_length": max_length})
    return tokenizer.encode(text, **kwargs)
def p95_passage(dataset, tokenizer, sample_size, passage_length):
    rows = []
    for i, doc in enumerate(dataset.docs_iter()):
        if i >= sample_size:
            break
        ids = enc(tokenizer, text_of(doc, ("title", "text", "body", "abstract")), passage_length)
        rows.append((len(ids), ids))
    if not rows:
        raise ValueError("dataset yielded no passages")
    rows.sort(key=lambda item: item[0])
    length, ids = rows[max(0, math.ceil(0.95 * len(rows)) - 1)]
    return length, tokenizer.decode(ids, skip_special_tokens=True)
def sample_query(dataset, tokenizer, query_length):
    for query in dataset.queries_iter():
        ids = enc(tokenizer, text_of(query, ("title", "text", "query")), query_length)
        return tokenizer.decode(ids, skip_special_tokens=True)
    return "sample query"
def dual_prompt_suffix(config, pool_size):
    if _is_multimodal_config(config):
        return (
            f"\n\nReply with exactly two distinct passage numbers between 1 and {pool_size}. "
            f"Do not output 0, 'None', or any number outside the range. "
            f"Pick the closest passages even if none are clearly relevant. "
            f"Strict format on one line: Best: <number>, Worst: <number>"
        )
    if getattr(config, "model_type", None) == "llama":
        return (
            f"\n\nReply with exactly two distinct passage numbers between 1 and {pool_size}. "
            f"Do not output letters, 0, 'None', or any number outside 1 to {pool_size}. "
            f"Pick the closest passages even if none are clearly relevant. "
            f"Strict format on one line: Best: <number>, Worst: <number>"
        )
    labels = ", ".join(str(i + 1) for i in range(pool_size))
    return (
        f"\n\nReply with exactly two distinct passage labels, "
        f"each one of these labels: {labels}. "
        f"Do not explain. "
        f"Strict format on one line: Best: <label>, Worst: <label>"
    )
def render_prompt(tokenizer, config, query, passage, pool_size):
    passages = "\n\n".join(f'Passage {i + 1}: "{passage}"' for i in range(pool_size))
    content = (
        f'Given a query "{query}", which of the following passages is the most relevant '
        f"and which is the least relevant to the query?\n\n{passages}"
        + dual_prompt_suffix(config, pool_size)
    )
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **chat_kwargs(config))
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
def probe(family, dataset_name, args):
    model_id = FAMILIES[family]
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if _is_multimodal_config(config):
        tokenizer = ProcessorTokenizerAdapter(
            AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    dataset = ir_datasets.load(DATASETS[dataset_name])
    p95_tokens, passage = p95_passage(dataset, tokenizer, args.sample_size, args.passage_length)
    query = sample_query(dataset, tokenizer, args.query_length)
    prompt = render_prompt(tokenizer, config, query, passage, args.pool_size)
    prompt_tokens = len(enc(tokenizer, prompt))
    max_emb = max_pos(config, tokenizer)
    headroom = max_emb - args.reserved_output_tokens - prompt_tokens
    return {"family": family, "dataset": dataset_name, "p95_passage_tokens": p95_tokens,
            "prompt_tokens": prompt_tokens, "max_pos_emb": max_emb, "headroom": headroom,
            "verdict": "PASS" if headroom > 0 else "FAIL"}
def main():
    args = parse_args()
    families = [args.family] if args.family else list(FAMILIES)
    datasets = [args.dataset] if args.dataset else list(DATASETS)
    rows = [probe(family, dataset, args) for family in families for dataset in datasets]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print("{family} {dataset}: {verdict} prompt={prompt_tokens} max={max_pos_emb} headroom={headroom} p95={p95_passage_tokens}".format(**row))

if __name__ == "__main__":
    main()
