import uuid
from typing import Optional

import torch
from accelerate import Accelerator
from transformers.tokenization_utils_base import BatchEncoding

from robust_llm.config.dataset_configs import MessageFilter
from robust_llm.config.layered_defense_configs import PromptedFilterTemplateConfig
from robust_llm.config.model_configs import ModelConfig
from robust_llm.layered_defense.components import (
    ComponentRegistry,
    ComponentResult,
    InputComponent,
    InputPipelineData,
    OutputComponent,
    OutputPipelineData,
)
from robust_llm.message_utils import Message, MessageList
from robust_llm.models.model_outputs import GenerationLogits
from robust_llm.models.wrapped_model import WrappedModel


class PromptedFilter:
    def __init__(
        self,
        victim: Optional[WrappedModel],
        accelerator: Accelerator,
        filter_config: ModelConfig,
        template_config: PromptedFilterTemplateConfig,
        threshold: float = 0.5,
    ):
        if victim is None:
            accelerator = Accelerator()
        else:
            assert victim.accelerator is not None
            accelerator = victim.accelerator
        self.filter = WrappedModel.from_config(filter_config, accelerator)
        self.threshold = threshold
        YES_TOKEN = "yes"
        NO_TOKEN = "no"
        encoded_yes = self.filter.tokenize(YES_TOKEN, add_special_tokens=False)
        self.yes_token_id = int(encoded_yes.input_ids[0])
        encoded_no = self.filter.tokenize(NO_TOKEN, add_special_tokens=False)
        self.no_token_id = int(encoded_no.input_ids[0])
        self.template = template_config.template

    def prompts_to_component_results(
        self, inputs: BatchEncoding, component_name: str
    ) -> list[ComponentResult]:
        output = GenerationLogits.from_tokens(
            self.filter,
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
        # Grab last token logits, move to cpu, and combine logits into a single batch.
        # (This is slightly inefficient because we do multiple
        # moves to the CPU, but it avoids a cuda OOM.)
        all_logits = []
        for batch_logits in output:
            all_logits.append(batch_logits[:, -1, :].detach().cpu())
        logits = torch.cat(all_logits, dim=0)
        predictions = torch.argmax(logits, dim=1)
        if not (
            (predictions == self.yes_token_id) | (predictions == self.no_token_id)
        ).all():
            print(
                f"Unexpected predictions: {predictions}\n"
                f"Yes token id: {self.yes_token_id}\n"
                f"No token id: {self.no_token_id}\n",
            )
        yes_no_logits = logits[:, [self.yes_token_id, self.no_token_id]].to(
            torch.float32
        )
        probs = torch.softmax(yes_no_logits, dim=1)
        probs_yes = probs[:, 0]

        results = []
        for i in range(len(inputs.input_ids)):
            result = ComponentResult(
                flag=probs_yes[i].item() > self.threshold,
                name=component_name,
                score=probs_yes[i],
            )
            results.append(result)

        return results


@ComponentRegistry.register("PromptedInputFilter")
class PromptedInputFilter(PromptedFilter, InputComponent):
    def __call__(
        self, data: InputPipelineData, victim: Optional[WrappedModel]
    ) -> tuple[InputPipelineData, list[ComponentResult]]:
        inputs = self._prepare_tokenized_input(data)
        results = self.prompts_to_component_results(inputs, "PromptedInputFilter")
        return data, results

    def _prepare_tokenized_input(self, data: InputPipelineData) -> BatchEncoding:
        content_id = str(uuid.uuid4())[:8]
        message_lists = data.message_lists
        # Assuming that the final message in each message list is the input.
        input_messages = [message_list.messages[-1] for message_list in message_lists]
        if not all(
            [input_message.role == "user" for input_message in input_messages],
        ):
            raise ValueError(
                "Input messages must be user messages, " f"but got {input_messages}",
            )
        formatted_messages = [
            self.template.format(content=message.content, content_id=content_id)
            for message in input_messages
        ]
        input_message_lists = [
            MessageList([Message(role="user", content=message)])
            for message in formatted_messages
        ]
        inputs = self.filter.message_lists_to_tokens(
            input_message_lists,
            message_filter=MessageFilter.INPUT,
            padding_side="left",
        )
        return inputs


@ComponentRegistry.register("PromptedOutputFilter")
class PromptedOutputFilter(PromptedFilter, OutputComponent):
    def __call__(
        self, data: OutputPipelineData, victim: Optional[WrappedModel]
    ) -> tuple[OutputPipelineData, list[ComponentResult]]:
        # uuid to avoid prompt injection.
        inputs = self._prepare_tokenized_input(data)
        results = self.prompts_to_component_results(inputs, "PromptedOutputFilter")
        return data, results

    def _prepare_tokenized_input(self, data: OutputPipelineData) -> BatchEncoding:
        content_id = str(uuid.uuid4())[:8]
        message_lists = data.message_lists
        # Assuming that the final message in each message list is the output.
        output_messages = [message_list.messages[-1] for message_list in message_lists]
        if not all(
            [output_message.role == "assistant" for output_message in output_messages],
        ):
            raise ValueError(
                "Output messages must be assistant messages,"
                f" but got {output_messages}",
            )

        formatted_messages = [
            self.template.format(content=message.content, content_id=content_id)
            for message in output_messages
        ]
        output_message_lists = [
            MessageList([Message(role="user", content=message)])
            for message in formatted_messages
        ]

        inputs = self.filter.message_lists_to_tokens(
            output_message_lists,
            # Even though this is an output filter, we pass the prompt in as a
            # user message, since we are prompting an instruct model.
            message_filter=MessageFilter.INPUT,
            padding_side="left",
        )
        return inputs
