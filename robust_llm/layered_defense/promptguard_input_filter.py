from typing import Optional

import torch
from accelerate import Accelerator
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer

from robust_llm.layered_defense.components import (
    ComponentRegistry,
    ComponentResult,
    InputComponent,
    InputPipelineData,
)
from robust_llm.models.wrapped_model import WrappedModel


@ComponentRegistry.register("PromptGuardInputFilter")
class PromptGuardInputFilter(InputComponent):
    def __init__(
        self,
        victim: Optional[WrappedModel],
        accelerator: Accelerator,
        model_name: str = "meta-llama/Prompt-Guard-86M",
        threshold: float = 0.5,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def __call__(
        self, data: InputPipelineData, victim: Optional[WrappedModel]
    ) -> tuple[InputPipelineData, list[ComponentResult]]:
        """Input filter that flags if the text is a jailbreak.

        PromptGuard has three classes: BENIGN, INJECTION, and JAILBREAK.
        We ignore INJECTION since it is designed for classifying third-party
        data, e.g. as returned by a tool.
        """
        texts = [message_list.to_text() for message_list in data.message_lists]
        batch = self.tokenizer(texts, return_tensors="pt", padding=True)

        output = self.model(**batch)
        # Drop the INJECTION class.
        benign_and_jailbreak_logits = output.logits[:, [0, 2]]
        probs = torch.softmax(benign_and_jailbreak_logits, dim=1)

        results = []
        for i in range(len(texts)):
            result = ComponentResult(
                # Output is 1 for jailbreak, 0 for benign.
                flag=probs[i, 1].item() > self.threshold,
                name="PromptGuardInputFilter",
                score=probs[i, 1],
            )
            results.append(result)

        return data, results
