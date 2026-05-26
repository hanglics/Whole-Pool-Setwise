from types import ModuleType, SimpleNamespace
from unittest import mock
import sys

import pytest
import torch


def _missing_dependency(*_args, **_kwargs):
    raise RuntimeError("Optional runtime dependency is not available in this test environment.")


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

for name in ("openai", "tiktoken", "ir_datasets"):
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

import run as run_module
from llmrankers._processor_adapter import ProcessorTokenizerAdapter
from llmrankers.setwise import SetwiseLlmRanker, _is_multimodal_config


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

    def __call__(self, inputs, return_tensors=None, **_kwargs):
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        return DummyBatch({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    def encode(self, text, add_special_tokens=True, return_tensors=None, **_kwargs):
        ids = [1, 2, 3]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(item) for item in ids)

    def batch_decode(self, rows, skip_special_tokens=True):
        return [self.decode(row, skip_special_tokens=skip_special_tokens) for row in rows]

    def batch_encode_plus(self, texts, return_tensors=None, add_special_tokens=True, padding=False):
        input_ids = [[1, 2, 3] for _ in texts]
        return SimpleNamespace(input_ids=torch.tensor(input_ids, dtype=torch.long))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **_kwargs):
        rendered = "\n".join(message["content"] for message in messages)
        if add_generation_prompt:
            rendered += "\nassistant:"
        return self.encode(rendered) if tokenize else rendered

    def tokenize(self, text):
        return str(text).split()

    def convert_tokens_to_ids(self, tokens):
        return list(range(len(tokens)))

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)


class DummyMistralTokenizer(DummyTokenizer):
    def __init__(self):
        self.use_default_system_prompt = True

    @property
    def chat_template(self):
        return None

    @chat_template.setter
    def chat_template(self, _value):
        raise AttributeError("chat_template assignment rejected")

    def convert_tokens_to_string(self, tokens):
        raise NotImplementedError("missing on MistralCommonBackend")


class DummyProcessor:
    def __init__(self, tokenizer=None, *, fail_once=False):
        self.tokenizer = tokenizer or DummyTokenizer()
        self.fail_once = fail_once

    def apply_chat_template(self, *args, **kwargs):
        if self.fail_once:
            self.fail_once = False
            raise TypeError("processor rejected plain strings")
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def __call__(self, text=None, images=None, return_tensors=None):
        return self.tokenizer(text or "", return_tensors=return_tensors)


class DummyModel:
    def eval(self):
        return self


def config(model_type, *, multimodal=False):
    kwargs = {"model_type": model_type, "max_position_embeddings": 4096}
    if multimodal:
        kwargs["vision_config"] = SimpleNamespace()
    return SimpleNamespace(**kwargs)


def instantiate_setwise(model_type, *, multimodal=False, scoring="generation"):
    cfg = config(model_type, multimodal=multimodal)
    tokenizer_loader = mock.Mock(return_value=DummyTokenizer())
    processor_loader = mock.Mock(return_value=DummyProcessor(DummyMistralTokenizer()))
    causal_loader = mock.Mock(return_value=DummyModel())
    multimodal_loader = mock.Mock(return_value=DummyModel())
    with mock.patch("llmrankers.setwise.AutoConfig.from_pretrained", return_value=cfg), \
         mock.patch("llmrankers.setwise.AutoTokenizer.from_pretrained", tokenizer_loader), \
         mock.patch("llmrankers.setwise.AutoProcessor.from_pretrained", processor_loader), \
         mock.patch("llmrankers.setwise.AutoModelForCausalLM.from_pretrained", causal_loader), \
         mock.patch("llmrankers.setwise.AutoModelForImageTextToText.from_pretrained", multimodal_loader):
        ranker = SetwiseLlmRanker(
            model_name_or_path="model",
            tokenizer_name_or_path=None,
            device="cpu",
            num_child=3,
            k=10,
            scoring=scoring,
            method="heapsort",
            num_permutation=1,
            cache_dir=None,
        )
    return ranker, tokenizer_loader, processor_loader, causal_loader, multimodal_loader


def test_multimodal_config_predicate():
    assert _is_multimodal_config(config("mistral3", multimodal=True))
    assert _is_multimodal_config(config("qwen3_5", multimodal=True))
    assert _is_multimodal_config(config("qwen3_5_moe", multimodal=True))
    assert not _is_multimodal_config(config("qwen3_5_text"))
    assert not _is_multimodal_config(config("qwen3"))


def test_setwise_dispatch_routes_to_expected_loader():
    ranker, tokenizer_loader, processor_loader, causal_loader, multimodal_loader = instantiate_setwise("qwen3")
    assert ranker._is_multimodal_model() is False
    assert tokenizer_loader.called
    assert causal_loader.called
    assert not processor_loader.called
    assert not multimodal_loader.called

    ranker, tokenizer_loader, processor_loader, causal_loader, multimodal_loader = instantiate_setwise(
        "mistral3", multimodal=True
    )
    assert ranker._is_multimodal_model() is True
    assert processor_loader.called
    assert multimodal_loader.called
    assert not tokenizer_loader.called
    assert not causal_loader.called


def test_qwen3_moe_stays_on_causal_path():
    ranker, tokenizer_loader, processor_loader, causal_loader, multimodal_loader = instantiate_setwise("qwen3_moe")
    assert ranker._is_supported_causal_model() is True
    assert ranker._is_multimodal_model() is False
    assert tokenizer_loader.called
    assert causal_loader.called
    assert not processor_loader.called
    assert not multimodal_loader.called


def test_processor_tokenizer_adapter_fallbacks_and_properties():
    adapter = ProcessorTokenizerAdapter(DummyProcessor(DummyMistralTokenizer(), fail_once=True))
    assert "hello" in adapter.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert adapter.convert_tokens_to_string(["aa", "bb"]) == "0 1"
    adapter.chat_template = "ignored"
    assert adapter.chat_template is None
    assert adapter.pad_token is None
    adapter.pad_token = "</s>"
    assert adapter.pad_token == "</s>"
    assert adapter.eos_token == "</s>"
    assert adapter.pad_token_id == 0
    assert adapter.eos_token_id == 2
    assert adapter.batch_decode([[1, 2]]) == ["1 2"]
    assert adapter.tokenize("a b") == ["a", "b"]
    adapter.use_default_system_prompt = False
    assert adapter._tok.use_default_system_prompt is False
    adapter.padding_side = "left"
    assert adapter._tok.padding_side == "left"


def test_likelihood_rejected_in_init_for_multimodal():
    with pytest.raises(NotImplementedError, match="likelihood is not supported"):
        instantiate_setwise("qwen3_5", multimodal=True, scoring="likelihood")


def test_run_py_likelihood_guard_rejects_before_ranker_construction():
    args = SimpleNamespace(
        run=SimpleNamespace(
            model_name_or_path="Qwen/Qwen3.5-9B",
            tokenizer_name_or_path=None,
            device="cpu",
            cache_dir=None,
            openai_key=None,
            scoring="likelihood",
            hits=10,
            query_length=128,
            passage_length=512,
            log_comparisons=None,
            ir_dataset_name=None,
            pyserini_index=None,
            run_path="unused",
            save_path="unused",
            shuffle_ranking=None,
            shuffle=False,
            reverse=False,
        ),
        setwise=SimpleNamespace(
            direction="topdown",
            num_child=3,
            method="heapsort",
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
        ),
        pointwise=None,
        pairwise=None,
        listwise=None,
    )
    with mock.patch("run.AutoConfig.from_pretrained", return_value=config("qwen3_5", multimodal=True)), \
         mock.patch("run.SetwiseLlmRanker") as ranker_cls:
        with pytest.raises(SystemExit, match="likelihood is not supported"):
            run_module.main(args)
    assert not ranker_cls.called
