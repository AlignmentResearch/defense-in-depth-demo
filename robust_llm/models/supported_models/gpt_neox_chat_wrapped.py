from __future__ import annotations

from transformers.models.gpt_neox.tokenization_gpt_neox_fast import GPTNeoXTokenizerFast
from typing_extensions import override

from robust_llm.config.model_configs import ModelConfig
from robust_llm.models.prompt_templates import Conversation
from robust_llm.models.wrapped_chat_model import WrappedChatModel
from robust_llm.models.wrapped_model import WrappedModel


@WrappedModel.register_subclass("pythia-chat")
class GPTNeoXChatModel(WrappedChatModel):
    # NOTE: Pythia models are based on GPTNeoX
    CONTEXT_LENGTH = 2048

    def post_init(self) -> None:
        super().post_init()
        assert self.family in ["pythia-chat"]

    @classmethod
    def load_tokenizer(
        cls,
        model_config: ModelConfig,
    ) -> GPTNeoXTokenizerFast:
        """Load the tokenizer."""
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(
            model_config.name_or_path,
            revision=model_config.revision,
            padding_side="right",  # Left padding is handled separately
            model_max_length=cls.CONTEXT_LENGTH,
            clean_up_tokenization_spaces=False,
        )
        assert isinstance(tokenizer, GPTNeoXTokenizerFast)  # for type-checking

        # Special setup needed for pythia.
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @override
    def init_conversation(self) -> Conversation:
        """Pythia/NeoX-specific conversation template."""
        return Conversation(
            prompt_prefix="",
            system_prefix="<|im_start|>system\n",
            system_suffix="<|im_end|>\n",
            user_prefix="<|im_start|>user\n",
            user_suffix="<|im_end|>\n",
            assistant_prefix="<|im_start|>assistant\n",
            assistant_suffix="<|im_end|>\n",
            system_prompt=self.system_prompt,
        )
