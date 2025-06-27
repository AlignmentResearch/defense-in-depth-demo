from __future__ import annotations

from typing import Optional

from accelerate import Accelerator
from typing_extensions import override

from robust_llm.config.model_configs import ModelConfig
from robust_llm.models.prompt_templates import Conversation
from robust_llm.models.wrapped_model import Prompt, WrappedModel
from robust_llm.utils import print_time


class WrappedChatModel(WrappedModel):
    """Wrapper for Chat/IFT models.

    Chat and instruction-fine-tuned models require special handling for the
    input. Instead of regular text, they expect the input to be chat-formatted.
    The specific format varies from model to model, but this should abstract that away.
    """

    def post_init(self) -> None:
        super().post_init()
        assert self.init_conversation() != Conversation(
            prompt_prefix="",
            system_prefix="",
            system_suffix="",
            user_prefix="",
            user_suffix="",
            assistant_prefix="",
            assistant_suffix="",
        )

    def get_user_prompt(self, text: str) -> str:
        """Get the user prompt for a chat model.

        Args:
            text: The user input.

        Returns:
            The chat-formatted user prompt.
        """

        conv = self.init_conversation()
        conv.append_user_message(text)
        conv.append_assistant_message("")
        return conv.get_prompt(skip_last_suffix=True)

    @override
    def maybe_apply_user_template(self, text: Prompt) -> Prompt:
        """If working with a chat model, return text with chat template applied.

        Since this is the base class for chat models, we apply the chat template.

        Args:
            text: The user prompt(s) to apply the chat template to.

        Returns:
            The text with the chat template applied.
        """

        if isinstance(text, str):
            return self.get_user_prompt(text)

        else:
            return [self.get_user_prompt(u) for u in text]

    @print_time()
    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        accelerator: Accelerator | None,
        num_classes: Optional[int] = None,
        overwrite_cache: bool = False,
        **kwargs,
    ) -> WrappedChatModel:
        model = super().from_config(
            config,
            accelerator,
            num_classes=num_classes,
            overwrite_cache=overwrite_cache,
            **kwargs,
        )
        assert isinstance(model, WrappedChatModel)
        return model
