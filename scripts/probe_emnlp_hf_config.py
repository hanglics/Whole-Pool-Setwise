#!/usr/bin/env python3
"""HF config/context gate for EMNLP v8 model IDs."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODELS = [
    ("qwen3.5", "Qwen/Qwen3.5-0.8B"),
    ("qwen3.5", "Qwen/Qwen3.5-2B"),
    ("qwen3.5", "Qwen/Qwen3.5-4B"),
    ("qwen3.5", "Qwen/Qwen3.5-9B"),
    ("qwen3.5", "Qwen/Qwen3.5-27B"),
    ("llama3.1", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    ("ministral3", "mistralai/Ministral-3-3B-Instruct-2512"),
    ("ministral3", "mistralai/Ministral-3-8B-Instruct-2512"),
    ("ministral3", "mistralai/Ministral-3-14B-Instruct-2512"),
]
FAMILY_ALIASES = {"q3.5": "qwen3.5", "llama3.1": "llama3.1", "ministral3": "ministral3"}
FIELDS = "family,model_id,model_type,model_type_allowed,max_pos_emb,prompt_tokens,reserved_output,headroom,verdict,fail_reason".split(",")
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(FAMILY_ALIASES), default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--passage-length", type=int, default=512)
    parser.add_argument("--query-length", type=int, default=128)
    parser.add_argument("--reserved-output-tokens", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=Path("results/emnlp/probe_emnlp_hf_config.csv"))
    return parser.parse_args()
def load_runtime():
    global AutoConfig, AutoProcessor, AutoTokenizer, ProcessorTokenizerAdapter, _is_multimodal_config
    global MAXCONTEXT_ALLOWED_MODEL_TYPES
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer
    from llmrankers._processor_adapter import ProcessorTokenizerAdapter
    from llmrankers.setwise import _is_multimodal_config
    from llmrankers.setwise_extended import MAXCONTEXT_ALLOWED_MODEL_TYPES
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
def synthetic_text(tokenizer, max_length, token):
    text = " ".join([token] * max(600, max_length * 2))
    return tokenizer.decode(enc(tokenizer, text, max_length), skip_special_tokens=True)
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
def fail(row, reason):
    row.update({"verdict": "FAIL", "fail_reason": " ".join(str(reason).split())})
    return row
def probe(family, model_id, args):
    row = dict.fromkeys(FIELDS, "")
    row.update({
        "family": family, "model_id": model_id, "model_type_allowed": False,
        "reserved_output": args.reserved_output_tokens, "verdict": "FAIL",
    })
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:
        return fail(row, f"config_load: {exc}")
    model_type = getattr(config, "model_type", "")
    row["model_type"] = model_type
    row["model_type_allowed"] = model_type in MAXCONTEXT_ALLOWED_MODEL_TYPES
    if not row["model_type_allowed"]:
        return fail(row, f"model_type_not_allowed: {model_type or 'missing'}")
    try:
        if _is_multimodal_config(config):
            tokenizer = ProcessorTokenizerAdapter(
                AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        passage = synthetic_text(tokenizer, args.passage_length, "placeholder")
        query = synthetic_text(tokenizer, args.query_length, "query")
        prompt = render_prompt(tokenizer, config, query, passage, args.pool_size)
        prompt_tokens = len(enc(tokenizer, prompt))
        max_emb = max_pos(config, tokenizer)
    except Exception as exc:
        return fail(row, f"probe: {exc}")
    headroom = max_emb - args.reserved_output_tokens - prompt_tokens
    row.update({"max_pos_emb": max_emb, "prompt_tokens": prompt_tokens, "headroom": headroom})
    if headroom <= 0:
        return fail(row, "context_length_exceeded: headroom <= 0")
    row["verdict"] = "PASS"
    return row
def selected_models(args):
    family = FAMILY_ALIASES.get(args.family)
    models = [
        (row_family, model_id) for row_family, model_id in MODELS
        if (family is None or row_family == family) and (args.model is None or model_id == args.model)
    ]
    if not models:
        raise SystemExit("No required model IDs matched the requested subset.")
    return models
def main():
    args = parse_args()
    models = selected_models(args)
    load_runtime()
    rows = [probe(family, model_id, args) for family, model_id in models]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            "{family} {model_id}: {verdict} model_type={model_type} max={max_pos_emb} "
            "prompt={prompt_tokens} headroom={headroom}".format(**row)
        )
    return 0 if all(row["verdict"] == "PASS" for row in rows) else 1
if __name__ == "__main__":
    raise SystemExit(main())
