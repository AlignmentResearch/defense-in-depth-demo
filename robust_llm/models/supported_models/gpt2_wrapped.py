from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from typing_extensions import override

from robust_llm.config.model_configs import ModelConfig
from robust_llm.models.wrapped_model import WrappedModel


@WrappedModel.register_subclass("gpt2")
class GPT2Model(WrappedModel):
    CONTEXT_LENGTH = 1024

    def post_init(self):
        super().post_init()
        # Special setup needed for gpt2.
        assert self.family in ["gpt2"]

    @override
    @classmethod
    def load_tokenizer(
        cls,
        model_config: ModelConfig,
    ) -> GPT2TokenizerFast:
        """Load the tokenizer."""
        tokenizer = GPT2TokenizerFast.from_pretrained(
            model_config.name_or_path,
            revision=model_config.revision,
            padding_side="right",  # Left padding is handled separately
            model_max_length=cls.CONTEXT_LENGTH,
            clean_up_tokenization_spaces=False,
        )
        assert isinstance(tokenizer, GPT2TokenizerFast)  # for type-checking

        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
