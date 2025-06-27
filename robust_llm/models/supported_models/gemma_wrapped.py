from __future__ import annotations

from transformers.models.gemma.tokenization_gemma_fast import GemmaTokenizerFast
from typing_extensions import override

from robust_llm.config.model_configs import ModelConfig
from robust_llm.models.wrapped_model import WrappedModel


@WrappedModel.register_subclass("gemma")
class GemmaModel(WrappedModel):
    # Context length from:
    # https://huggingface.co/google/gemma-1.1-2b/blob/main/config.json
    # https://huggingface.co/google/gemma-2-9b/blob/main/config.json
    CONTEXT_LENGTH = 8192

    def post_init(self):
        super().post_init()
        assert self.family in ["gemma"]

    @override
    def forward(self, **inputs):
        # Gemma is idiosyncratic in that it requires use_cache=True to use
        # provided past_key_values. This is not the default behavior for other
        # models (except Qwen), where use_cache indicates whether to return
        # past_key_values.
        if "past_key_values" in inputs:
            inputs["use_cache"] = True
        return super().forward(**inputs)

    @classmethod
    def load_tokenizer(
        cls,
        model_config: ModelConfig,
    ) -> GemmaTokenizerFast:
        """Load the tokenizer."""
        tokenizer = GemmaTokenizerFast.from_pretrained(
            model_config.name_or_path,
            revision=model_config.revision,
            padding_side="right",  # Left padding is handled separately
            model_max_length=cls.CONTEXT_LENGTH,
            clean_up_tokenization_spaces=False,
        )
        assert isinstance(tokenizer, GemmaTokenizerFast)  # for type-checking

        return tokenizer
