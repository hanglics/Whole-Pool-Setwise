class ProcessorTokenizerAdapter:
    """Expose a processor tokenizer through the tokenizer surface rankers expect."""

    def __init__(self, processor):
        self._processor = processor
        self._tok = processor.tokenizer

    def __getattr__(self, name):
        return getattr(self._tok, name)

    def __call__(self, *args, **kwargs):
        return self._tok(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        if hasattr(self._processor, "apply_chat_template"):
            try:
                return self._processor.apply_chat_template(*args, **kwargs)
            except (TypeError, ValueError, NotImplementedError):
                pass
        return self._tok.apply_chat_template(*args, **kwargs)

    def convert_tokens_to_string(self, tokens):
        if hasattr(self._tok, "convert_tokens_to_string"):
            try:
                return self._tok.convert_tokens_to_string(tokens)
            except (AttributeError, NotImplementedError):
                pass
        ids = self._tok.convert_tokens_to_ids(tokens)
        return self._tok.decode(ids, skip_special_tokens=False)

    @property
    def pad_token(self):
        return self._tok.pad_token

    @pad_token.setter
    def pad_token(self, value):
        self._tok.pad_token = value

    @property
    def eos_token(self):
        return self._tok.eos_token

    @property
    def pad_token_id(self):
        return self._tok.pad_token_id

    @property
    def eos_token_id(self):
        return self._tok.eos_token_id

    @property
    def chat_template(self):
        return getattr(self._tok, "chat_template", None)

    @chat_template.setter
    def chat_template(self, value):
        try:
            self._tok.chat_template = value
        except (AttributeError, NotImplementedError):
            pass

    @property
    def use_default_system_prompt(self):
        return getattr(self._tok, "use_default_system_prompt", None)

    @use_default_system_prompt.setter
    def use_default_system_prompt(self, value):
        if hasattr(self._tok, "use_default_system_prompt"):
            self._tok.use_default_system_prompt = value

    @property
    def padding_side(self):
        return getattr(self._tok, "padding_side", None)

    @padding_side.setter
    def padding_side(self, value):
        self._tok.padding_side = value
