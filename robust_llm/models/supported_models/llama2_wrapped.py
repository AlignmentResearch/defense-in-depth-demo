from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from typing_extensions import override

from robust_llm.config.model_configs import ModelConfig
from robust_llm.models.wrapped_model import WrappedModel


@WrappedModel.register_subclass("llama2")
class Llama2Model(WrappedModel):
    CONTEXT_LENGTH = 4096

    def post_init(self):
        super().post_init()
        assert self.family in ["llama2"]

    @override
    def forward(self, **inputs):
        if "past_key_values" in inputs:
            inputs["use_cache"] = True
        return super().forward(**inputs)

    @override
    @classmethod
    def load_tokenizer(
        cls,
        model_config: ModelConfig,
    ) -> LlamaTokenizerFast:
        """Load the tokenizer."""
        tokenizer = LlamaTokenizerFast.from_pretrained(
            model_config.name_or_path,
            revision=model_config.revision,
            padding_side="right",  # Left padding is handled separately
            model_max_length=cls.CONTEXT_LENGTH,
            clean_up_tokenization_spaces=False,
        )

        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
