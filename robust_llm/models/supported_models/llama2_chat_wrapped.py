from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from typing_extensions import override

from robust_llm.config.model_configs import ModelConfig
from robust_llm.models.prompt_templates import Conversation
from robust_llm.models.wrapped_chat_model import WrappedChatModel
from robust_llm.models.wrapped_model import WrappedModel


@WrappedModel.register_subclass("llama2-chat")
@WrappedModel.register_subclass("vicuna")
@WrappedModel.register_subclass("mistralai-chat")
class Llama2ChatModel(WrappedChatModel):
    CONTEXT_LENGTH = 4096

    def post_init(self) -> None:
        super().post_init()
        assert self.family in ["llama2-chat", "vicuna", "mistralai-chat"]

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
            add_prefix_space=False,  # This is handled by the prompt template
        )

        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @override
    def init_conversation(self) -> Conversation:
        """Llama2-specific conversation template."""
        return Conversation(
            prompt_prefix="<s>[INST] ",
            system_prefix="<<SYS>>\n",
            system_suffix="\n<</SYS>>\n\n",
            user_prefix="",
            user_suffix=" [/INST]",
            assistant_prefix="",
            assistant_suffix=" </s>",
            system_prompt=self.system_prompt,
            repeat_prompt_prefix=True,
            require_leading_whitespace=True,
        )

    @override
    def forward(self, **inputs):
        # Like Gemma, Llama2 requires use_cache=True to use
        # provided past_key_values.
        if "past_key_values" in inputs:
            inputs["use_cache"] = True
        return super().forward(**inputs)
