from typing import Optional

import torch
from accelerate import Accelerator
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from robust_llm.layered_defense.components import (
    ComponentRegistry,
    ComponentResult,
    InputComponent,
    InputPipelineData,
)
from robust_llm.message_utils import MessageFilter, MessageList
from robust_llm.models.model_utils import InferenceType, load_hf_model
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.utils import restrict_to_last_nonpad_token


@ComponentRegistry.register("LlamaGuard4InputFilter")
class LlamaGuard4InputFilter(InputComponent):
    """Input filter that flags if the text is a jailbreak using LlamaGuard4.

    NOTE: We use LlamaGuard4 as text-only, ignoring its multi-modal capabilities.
    """

    SAFE_TOKEN_ID: int = 60411
    UNSAFE_TOKEN_ID: int = 72110

    def __init__(
        self,
        victim: WrappedModel,
        accelerator: Accelerator,
        model_name: str,
        dtype: str = "bfloat16",
        threshold: float = 0.5,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = load_hf_model(
            accelerator=accelerator,
            name_or_path=model_name,
            revision="main",
            inference_type=InferenceType.GENERATION,
            torch_dtype=getattr(torch, dtype),
            strict_load=False,  # We're throwing away the vision weights.
        )
        # LlamaGuard4 has a strange discrepancy in the pad token id.
        self.tokenizer.pad_token_id = self.model.config.pad_token_id

        assert self.tokenizer.decode(self.SAFE_TOKEN_ID) == "safe"
        assert self.tokenizer.decode(self.UNSAFE_TOKEN_ID) == "unsafe"

    def __call__(
        self,
        data: InputPipelineData,
        victim: Optional[WrappedModel],
    ) -> tuple[InputPipelineData, list[ComponentResult]]:
        """Input filter that flags if the text is a jailbreak.

        LlamaGuard4 responds with either safe or unsafe.
        In tokens, that is: either 60411 or 72110.
        """
        message_lists = [
            message_list.filter(MessageFilter.INPUT)
            for message_list in data.message_lists
        ]
        texts = [
            format_input_messages(message_list, self.tokenizer)
            for message_list in message_lists
        ]
        batch = self.tokenizer(texts, return_tensors="pt", padding=True).to(
            self.model.device
        )

        with torch.inference_mode():
            # We set use_cache=False because Llama4 has a bug where .forward
            # errors if use_cache=True. Additionally, you can't set use_cache=False
            # in AutoModelForCausalLM.from_pretrained() for Llama4, but you *can*
            # for Llama4ForCausalLM.from_pretrained().
            output = self.model(**batch, use_cache=False)
            logits = (
                restrict_to_last_nonpad_token(output.logits, batch.attention_mask)
                .detach()
                .cpu()
            )
        safe_and_unsafe_logits = logits[:, [self.SAFE_TOKEN_ID, self.UNSAFE_TOKEN_ID]]
        probs = torch.softmax(safe_and_unsafe_logits.to(dtype=torch.float32), dim=1)

        results = []
        for i in range(len(texts)):
            result = ComponentResult(
                # Output is 1 for jailbreak, 0 for benign.
                flag=probs[i, 1].item() > self.threshold,
                name="LlamaGuard4InputFilter",
                score=probs[i, 1],
            )
            results.append(result)

        return data, results


def format_input_messages(
    message_list: MessageList, tokenizer: PreTrainedTokenizerBase
) -> str:
    message_dicts = [
        {"role": message.role, "content": [{"type": "text", "text": message.content}]}
        for message in message_list.messages
    ]
    templated = tokenizer.apply_chat_template(
        message_dicts,  # type: ignore
        tokenize=False,
    )
    assert isinstance(templated, str)
    return templated + "\n\n"  # Append redundantly generated tokens
