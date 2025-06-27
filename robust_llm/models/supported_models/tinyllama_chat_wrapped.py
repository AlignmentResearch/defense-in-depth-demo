from __future__ import annotations

from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from typing_extensions import override

from robust_llm.config.model_configs import ModelConfig
from robust_llm.models.prompt_templates import Conversation
from robust_llm.models.wrapped_chat_model import WrappedChatModel
from robust_llm.models.wrapped_model import WrappedModel


@WrappedModel.register_subclass("tinyllama")
class TinyLlamaChatModel(WrappedChatModel):
    CONTEXT_LENGTH = 8192

    def post_init(self) -> None:
        super().post_init()
        assert self.family in ["tinyllama"]

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
        assert isinstance(tokenizer, LlamaTokenizerFast)  # for type-checking

        return tokenizer

    @override
    def init_conversation(self) -> Conversation:
        """TinyLlama-specific conversation template."""
        return Conversation(
            prompt_prefix="",
            system_prefix="<|system|>\n",
            system_suffix="</s>\n",
            user_prefix="<|user|>\n",
            user_suffix="</s>\n",
            assistant_prefix="<|assistant|>\n",
            assistant_suffix="</s>\n",
            system_prompt=self.system_prompt,
        )
