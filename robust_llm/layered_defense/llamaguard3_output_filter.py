from typing import Optional

import torch
from accelerate import Accelerator
from transformers.models.auto.tokenization_auto import AutoTokenizer

from robust_llm.layered_defense.components import (
    ComponentRegistry,
    ComponentResult,
    OutputComponent,
    OutputPipelineData,
)
from robust_llm.message_utils import Message, MessageFilter, MessageList
from robust_llm.models.model_utils import InferenceType, load_hf_model
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.utils import restrict_to_last_nonpad_token


@ComponentRegistry.register("LlamaGuard3OutputFilter")
class LlamaGuard3OutputFilter(OutputComponent):
    UNSAFE_TOKEN_ID: int = 39257
    SAFE_TOKEN_ID: int = 19193

    def __init__(
        self,
        victim: WrappedModel,
        accelerator: Accelerator,
        model_name: str = "meta-llama/Llama-Guard-3-8B",
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
            strict_load=True,
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = (
            self.model.config.eos_token_id[0]
        )
        assert self.tokenizer.decode(self.UNSAFE_TOKEN_ID) == "unsafe"
        assert self.tokenizer.decode(self.SAFE_TOKEN_ID) == "safe"

    def __call__(
        self,
        data: OutputPipelineData,
        victim: Optional[WrappedModel],
    ) -> tuple[OutputPipelineData, list[ComponentResult]]:
        """Output filter that flags if the text is a jailbreak.

        LlamaGuard3 responds with double newlines, then either safe or unsafe.
        """
        assert all(
            message_list[-1].role == "assistant" for message_list in data.message_lists
        )
        # We have to add empty user messages because tokenizer expects
        # alternating user/assistant messages.
        message_lists = [
            MessageList([Message(role="user", content=""), message_list[-1]])
            for message_list in data.message_lists
        ]
        texts = [
            message_list.format(
                message_filter=MessageFilter.OUTPUT,
                tokenizer=self.tokenizer,
            )
            + "\n\n"  # Append redundantly generated tokens
            for message_list in message_lists
        ]
        batch = self.tokenizer(texts, return_tensors="pt", padding=True).to(
            self.model.device
        )
        with torch.inference_mode():
            output = self.model(**batch)
            logits = (
                restrict_to_last_nonpad_token(output.logits, batch.attention_mask)
                .detach()
                .cpu()
            )
        predictions = logits.argmax(dim=-1)
        assert (
            (predictions == self.UNSAFE_TOKEN_ID) | (predictions == self.SAFE_TOKEN_ID)
        ).all(), f"Unexpected predictions: {predictions}"
        safe_and_unsafe_logits = logits[:, [self.SAFE_TOKEN_ID, self.UNSAFE_TOKEN_ID]]
        probs = torch.softmax(safe_and_unsafe_logits.to(dtype=torch.float32), dim=1)

        results = []
        for i in range(len(texts)):
            result = ComponentResult(
                # Output is 1 for jailbreak, 0 for benign.
                flag=probs[i, 1].item() > self.threshold,
                name="LlamaGuard3OutputFilter",
                score=probs[i, 1],
            )
            results.append(result)

        return data, results
