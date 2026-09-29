#!/usr/bin/env python3
"""Deterministic local generation backend for model-matched baseline adaptations."""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

class OpenModelBackend:
    """Load one paper model and expose its native chat template for local inference."""

    def __init__(
        self,
        model: str,
        revision: str,
        *,
        cache_dir: str | None = None,
        device: str = "cuda",
    ) -> None:
        # Keep the heavyweight ranker import lazy so lightweight entrypoint
        # validation (including ``--help`` checks) does not require optional
        # legacy API dependencies that this local-only backend never uses.
        from llmrankers.setwise import SetwiseLlmRanker

        self.model = model
        self.revision = revision
        self.ranker = SetwiseLlmRanker(
            model_name_or_path=model,
            tokenizer_name_or_path=model,
            device=device,
            cache_dir=cache_dir,
            num_child=2,
            k=2,
            scoring="generation",
            method="selection",
            num_permutation=1,
            model_revision=revision,
            tokenizer_revision=revision,
        )
        self.ranker.strict_no_truncation = True

    def render(self, messages: Sequence[dict[str, str]]) -> str:
        return self.ranker._build_chat_prompt(list(messages))

    def count_prompt_tokens(self, messages: Sequence[dict[str, str]]) -> int:
        prompt = self.render(messages)
        inputs = self.ranker._tokenize_inputs(prompt)
        return int(inputs.input_ids.shape[1])

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_new_tokens: int,
        min_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        if min_new_tokens is not None and not 0 <= min_new_tokens <= max_new_tokens:
            raise ValueError("min_new_tokens must be between zero and max_new_tokens.")
        prompt = self.render(messages)
        inputs = self.ranker._tokenize_inputs(prompt)
        prompt_tokens = int(inputs.input_ids.shape[1])
        context_limit = self.ranker.max_input_tokens
        if context_limit is not None and prompt_tokens + max_new_tokens > context_limit:
            raise ValueError(
                f"Rendered prompt plus generation budget exceeds context: "
                f"{prompt_tokens}+{max_new_tokens}>{context_limit}"
            )
        cuda_active = self.ranker.device.startswith("cuda") and torch.cuda.is_available()
        if cuda_active:
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            output_ids = self.ranker._generate(
                inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
            )[0]
        if cuda_active:
            torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - started
        completion_ids = output_ids[prompt_tokens:]
        # Preserve the exact tokenizer rendering for telemetry and Liu's pinned
        # open-model parser, while also exposing the API-equivalent message
        # content needed by TourRank. Hosted chat-completion APIs do not include
        # tokenizer control tokens in ``message.content``.
        raw_output = self.ranker.tokenizer.decode(
            completion_ids, skip_special_tokens=False
        )
        content_output = self.ranker.tokenizer.decode(
            completion_ids, skip_special_tokens=True
        )
        return {
            "raw_output": raw_output,
            "content_output": content_output,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": int(completion_ids.shape[0]),
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "wall_seconds": wall_seconds,
            "rendered_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        }

    def contract(self) -> dict[str, Any]:
        processor = getattr(self.ranker, "processor", None)
        processor_template = getattr(processor, "chat_template", None)
        tokenizer_template = getattr(self.ranker.tokenizer, "chat_template", None)
        if processor_template is not None:
            chat_template = processor_template
            chat_template_source = "processor"
        else:
            chat_template = tokenizer_template
            chat_template_source = "tokenizer"
        if chat_template is not None:
            chat_template = str(chat_template)
        template_kwargs = self.ranker._chat_template_kwargs()
        rendered_probe = self.render(
            [{"role": "user", "content": "__experiment_chat_template_probe__"}]
        )
        generation_config = getattr(self.ranker.llm, "generation_config", None)
        return {
            "model": self.model,
            "model_revision": self.revision,
            "tokenizer_revision": self.revision,
            "chat_template": chat_template,
            "chat_template_source": chat_template_source,
            "chat_template_kwargs": template_kwargs,
            "chat_template_sha256": (
                hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
                if chat_template is not None
                else None
            ),
            "rendered_chat_probe_sha256": hashlib.sha256(
                rendered_probe.encode("utf-8")
            ).hexdigest(),
            "context_limit": self.ranker.max_input_tokens,
            "generation_config_class": (
                type(generation_config).__name__ if generation_config is not None else None
            ),
            "do_sample": False,
        }
