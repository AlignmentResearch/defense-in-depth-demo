from typing import Optional

import torch
from accelerate import Accelerator
from transformers.generation.utils import GenerationMixin
from transformers.models.auto.tokenization_auto import AutoTokenizer

from robust_llm.layered_defense.components import (
    ComponentRegistry,
    ComponentResult,
    OutputComponent,
    OutputPipelineData,
)
from robust_llm.layered_defense.wildguard_filters.wildguard_utils import (
    wildguard_scores,
)
from robust_llm.models.model_utils import InferenceType, load_hf_model
from robust_llm.models.wrapped_model import WrappedModel


@ComponentRegistry.register("WildGuardOutputFilter")
class WildGuardOutputFilter(OutputComponent):
    def __init__(
        self,
        victim: Optional[WrappedModel],
        accelerator: Accelerator,
        threshold: float = 0.5,
    ):
        """Initialize an input filter trained using our pipeline."""
        self.filter = load_hf_model(
            accelerator=accelerator,
            name_or_path="allenai/wildguard",
            revision="main",
            inference_type=InferenceType.GENERATION,
            torch_dtype=torch.float16,
            strict_load=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/wildguard")
        self.filter.pad_token_id = self.tokenizer.pad_token_id
        self.threshold = threshold

    def __call__(
        self, data: OutputPipelineData, victim: Optional[WrappedModel]
    ) -> tuple[OutputPipelineData, list[ComponentResult]]:
        """Input filter that flags if the text is a jailbreak."""
        message_lists = data.message_lists
        prompts = ["" for _ in range(len(message_lists))]
        responses = [message_list[-1].content for message_list in data.message_lists]
        assert isinstance(self.filter, GenerationMixin)
        scores = wildguard_scores(self.filter, self.tokenizer, prompts, responses)

        # Get probability of 'yes' for harmful response.
        response_harm_probs = scores[:, 2, 0].detach().cpu()

        results = []
        for i in range(len(message_lists)):
            result = ComponentResult(
                # Output is 0 for benign, 1 for jailbreak.
                flag=response_harm_probs[i].item() > self.threshold,
                name="WildGuardOutputFilter",
                score=response_harm_probs[i],
            )
            results.append(result)

        return data, results
