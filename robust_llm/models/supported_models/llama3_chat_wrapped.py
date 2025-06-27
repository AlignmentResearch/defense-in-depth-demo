from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from typing_extensions import override

from robust_llm.config.model_configs import ModelConfig
from robust_llm.models.prompt_templates import Conversation
from robust_llm.models.wrapped_chat_model import WrappedChatModel
from robust_llm.models.wrapped_model import WrappedModel


@WrappedModel.register_subclass("llama3-chat")
class Llama3ChatModel(WrappedChatModel):
    CONTEXT_LENGTH = 4096

    def post_init(self) -> None:
        super().post_init()
        assert self.family in ["llama3-chat", "llama3-chat-vllm"]

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

    @override
    def init_conversation(self) -> Conversation:
        """Llama3-specific conversation template."""
        return Conversation(
            prompt_prefix="<|begin_of_text|>",
            system_prefix="<|start_header_id|>system<|end_header_id|>\n\n",
            system_suffix="<|eot_id|>",
            user_prefix="<|start_header_id|>user<|end_header_id|>\n\n",
            user_suffix="<|eot_id|>",
            assistant_prefix="<|start_header_id|>assistant<|end_header_id|>\n\n",
            assistant_suffix="<|eot_id|>",
            system_prompt=self.system_prompt,
        )
