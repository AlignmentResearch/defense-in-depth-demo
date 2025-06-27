from typing import Optional

import torch
from accelerate import Accelerator

from robust_llm.config.model_configs import ModelConfig
from robust_llm.layered_defense.components import (
    ComponentRegistry,
    ComponentResult,
    InputComponent,
    InputPipelineData,
)
from robust_llm.message_utils import MessageFilter
from robust_llm.models.model_outputs import ClassificationLogits
from robust_llm.models.wrapped_model import WrappedModel


@ComponentRegistry.register("TrainedInputFilter")
class TrainedInputFilter(InputComponent):
    def __init__(
        self,
        victim: Optional[WrappedModel],
        accelerator: Accelerator,
        filter_config: ModelConfig,
        threshold: float = 0.5,
    ):
        """Initialize an input filter trained using our pipeline."""
        if victim is None:
            accelerator = Accelerator()
        else:
            assert victim.accelerator is not None
            accelerator = victim.accelerator
        self.filter = WrappedModel.from_config(filter_config, accelerator)
        self.threshold = threshold

    def __call__(
        self, data: InputPipelineData, victim: Optional[WrappedModel]
    ) -> tuple[InputPipelineData, list[ComponentResult]]:
        """Input filter that flags if the text is a jailbreak."""
        message_lists = data.message_lists
        inputs = self.filter.message_lists_to_tokens(
            message_lists,
            message_filter=MessageFilter.INPUT,
            padding_side="right",
        )
        output = ClassificationLogits.from_tokens(
            self.filter,
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
        # Combine logits into a single batch.
        logits = torch.cat(list(output), dim=0)
        probs = torch.softmax(logits.to(torch.float32), dim=1)

        results = []
        for i in range(len(message_lists)):
            result = ComponentResult(
                # Output is 0 for benign, 1 for jailbreak.
                flag=probs[i, 1].item() > self.threshold,
                name="TrainedInputFilter",
                score=probs[i, 1].detach().cpu(),
            )
            results.append(result)

        return data, results
