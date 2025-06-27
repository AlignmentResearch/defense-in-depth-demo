import copy
from collections.abc import Sequence
from functools import partial
from typing import Callable, Literal, Optional, overload

import torch
from accelerate import Accelerator

from robust_llm.config.layered_defense_configs import LayeredDefenseConfig
from robust_llm.layered_defense.components import (
    AutoregressiveLayeredDefenseOutput,
    BinaryLayeredDefenseOutput,
    ComponentResult,
    InputComponent,
    InputPipelineData,
    LayeredDefenseOutput,
    LossLayeredDefenseOutput,
    OutputComponent,
    OutputPipelineData,
    PipelineData,
)
from robust_llm.layered_defense.factory import (
    create_input_component,
    create_output_component,
)
from robust_llm.message_utils import ConversationList, Message, MessageList
from robust_llm.models.model_output_metrics import GenerationMetric, Metric
from robust_llm.models.wrapped_model import WrappedModel


class LayeredDefense:
    def __init__(
        self,
        victim: Optional[WrappedModel],
        input_components: tuple[InputComponent, ...],
        output_components: tuple[OutputComponent, ...],
    ) -> None:
        self.victim = victim
        self.input_components = input_components
        self.output_components = output_components

    @classmethod
    def from_config(cls, config: LayeredDefenseConfig, victim: Optional[WrappedModel]):
        if victim is None:
            accelerator = Accelerator()
        else:
            assert victim.accelerator is not None
            accelerator = victim.accelerator
        input_components = tuple(
            create_input_component(
                input_component,
                victim,
                accelerator,
            )
            for input_component in config.input_components.values()
        )
        output_components = tuple(
            create_output_component(
                output_component,
                victim,
                accelerator,
            )
            for output_component in config.output_components.values()
        )
        return cls(victim, input_components, output_components)

    def prepare_pipeline_data(
        self,
        input_data: ConversationList,
    ) -> PipelineData:
        """Prepares the inputs to the defense."""
        if self.victim is None:
            raise ValueError("Victim is required for prepare_pipeline_data")
        # We use right-padding here because we are not actually generating
        tokenized = self.victim.message_lists_to_tokens(
            input_data.message_lists,
            message_filter=input_data.message_filter,
            padding_side="right",
        )
        assert self.victim.accelerator is not None
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        # Only compute latents if there are trained input/output probes.
        attention_masks: list[torch.Tensor] | None
        latents_list = None
        attention_masks = None

        return PipelineData(
            message_lists=list(input_data.message_lists),
            attention_masks=attention_masks,
            latents=latents_list,
        )

    @property
    def probes_in_pipeline(self) -> bool:
        return False

    @overload
    def run_with_targets(
        self,
        input_data: ConversationList,
        raw_targets: list[str],
        metric: Literal[Metric.LOSS],
    ) -> Sequence[LossLayeredDefenseOutput]: ...

    @overload
    def run_with_targets(
        self,
        input_data: ConversationList,
        raw_targets: list[str],
        metric: Literal[Metric.SUCCESS],
    ) -> Sequence[BinaryLayeredDefenseOutput]: ...

    def run_with_targets(
        self,
        input_data: ConversationList,
        raw_targets: list[str],
        metric: Literal[Metric.LOSS, Metric.SUCCESS],
    ) -> Sequence[LayeredDefenseOutput]:
        """Run the defense pipeline on a list of texts with targets.

        Instead of generating text, we are getting loss/accuracy on the target.

        Args:
            input_data:
                The input messages to run the pipeline on.
            raw_targets:
                The target generations to get loss/accuracy on.
            metric:
                The metric to compute - either loss or success.
        """
        if self.victim is None:
            raise ValueError("Victim is required for run_with_targets")
        n_examples = len(input_data)
        minibatch_size = self.victim.eval_minibatch_size
        pipeline_outputs: list[LayeredDefenseOutput] = []

        metric_callback: Callable[
            [WrappedModel, ConversationList, list[str]],
            list[torch.Tensor] | list[bool],
        ] = partial(GenerationMetric.from_messages, metric=metric)

        # Process in batches to reduce memory usage ONLY if we have probes
        if not self.probes_in_pipeline:
            minibatch_size = n_examples

        for batch_start in range(0, n_examples, minibatch_size):
            batch_end = min(batch_start + minibatch_size, n_examples)

            # Create batch slices of inputs and targets
            batch_input_data = ConversationList(
                message_lists=input_data.message_lists[batch_start:batch_end],
                message_filter=input_data.message_filter,
                add_generation_prompt=input_data.add_generation_prompt,
            )
            batch_raw_targets = raw_targets[batch_start:batch_end]

            # Process this batch through input components
            batch_in_data = self.prepare_pipeline_data(batch_input_data)
            batch_in_data = InputPipelineData(
                message_lists=batch_in_data.message_lists,
                attention_masks=batch_in_data.attention_masks,
                latents=batch_in_data.latents,
            )

            # Shape: n_components list of minibatch_size ComponentResults
            batch_input_results: list[list[ComponentResult]] = []

            for input_component in self.input_components:
                batch_in_data, results = input_component(batch_in_data, self.victim)
                batch_input_results.append(results)

            # Get metric output for this batch
            batch_metric_output: list[torch.Tensor] | list[bool] = metric_callback(  # type: ignore # noqa: E501
                self.victim,
                batch_input_data,
                batch_raw_targets,
            )

            # Create outputs for this batch
            for i in range(len(batch_input_data.message_lists)):
                input_result_i = []
                for input_result in batch_input_results:
                    input_result_i.append(input_result[i])

                if metric == Metric.LOSS:
                    metric_output_i = batch_metric_output[i]
                    assert isinstance(metric_output_i, torch.Tensor)
                    pipeline_outputs.append(
                        LossLayeredDefenseOutput(
                            loss=metric_output_i,
                            input_results=tuple(input_result_i),
                            # Output results don't make sense for loss method.
                            output_results=tuple([]),
                        )
                    )
                elif metric == Metric.SUCCESS:
                    metric_output_i = batch_metric_output[i]
                    assert isinstance(metric_output_i, bool)
                    pipeline_outputs.append(
                        BinaryLayeredDefenseOutput(
                            success=metric_output_i,
                            input_results=tuple(input_result_i),
                            # Output results don't make sense for success method.
                            output_results=tuple([]),
                        )
                    )

        return pipeline_outputs

    def __call__(
        self,
        input_data: ConversationList,
    ) -> list[AutoregressiveLayeredDefenseOutput]:
        """Run the defense pipeline on the input text.

        Args:
            input_data: The input messages to run the pipeline on.

        Returns:
            A tuple containing the output text, the input results, and the
            output results.
        """
        if self.victim is None:
            raise ValueError("Victim is required for __call__")
        n_examples = len(input_data)
        minibatch_size = self.victim.eval_minibatch_size
        pipeline_outputs = []

        # If there are no probes, we can use one big batch since there are no
        # latents to store in memory, and the underlying models for the victim
        # and filters handle minibatching. If there are probes, we minibatch the
        # whole pipeline to reduce memory usage by avoiding carrying large
        # latents around.
        if not self.probes_in_pipeline:
            minibatch_size = n_examples

        for batch_start in range(0, n_examples, minibatch_size):
            batch_end = min(batch_start + minibatch_size, n_examples)
            batch_input_data = ConversationList(
                message_lists=input_data.message_lists[batch_start:batch_end],
                message_filter=input_data.message_filter,
                add_generation_prompt=input_data.add_generation_prompt,
            )

            # Process this batch through input components
            batch_in_data = self.prepare_pipeline_data(batch_input_data)
            batch_in_data = InputPipelineData(
                message_lists=batch_in_data.message_lists,
                attention_masks=batch_in_data.attention_masks,
                latents=batch_in_data.latents,
            )

            batch_input_results = []
            for input_component in self.input_components:
                batch_in_data, result = input_component(batch_in_data, self.victim)
                batch_input_results.append(result)

            # Generate outputs for this batch
            batch_outs = []
            for gen_batch in self.victim.autoregressive_generation_from_messages(
                batch_in_data.message_lists,
            ):
                batch_outs.extend(gen_batch)

            # Process generated outputs through output components
            batch_out_data = self.prepare_pipeline_data(
                ConversationList(
                    message_lists=batch_outs,
                    message_filter=input_data.message_filter,
                ),
            )

            batch_out_data = OutputPipelineData(
                message_lists=batch_out_data.message_lists,
                attention_masks=batch_out_data.attention_masks,
                latents=batch_out_data.latents,
            )

            batch_output_results = []
            for output_component in self.output_components:
                batch_out_data, result = output_component(batch_out_data, self.victim)
                batch_output_results.append(result)

            # Create outputs for this batch
            for i in range(len(batch_outs)):
                input_result_i = []
                output_result_i = []
                for input_result in batch_input_results:
                    input_result_i.append(input_result[i])
                for output_result in batch_output_results:
                    output_result_i.append(output_result[i])

                pipeline_outputs.append(
                    AutoregressiveLayeredDefenseOutput(
                        message_list=batch_outs[i],
                        input_results=tuple(input_result_i),
                        output_results=tuple(output_result_i),
                    )
                )

        return pipeline_outputs

    def run_with_retrieved_generations(
        self,
        input_data: ConversationList,
        generations: list[str],
    ) -> Sequence[LayeredDefenseOutput]:
        """Run the defense pipeline on a list of texts with retrieved generations.

        Instead of generating text, we are reusing generations retrieved from,
        for example, a previous wandb run or a HF dataset.

        Args:
            input_data:
                The input messages to run the pipeline on.
            generations:
                The generations to run the pipeline on.
        """
        n_examples = len(input_data)
        pipeline_outputs: list[LayeredDefenseOutput] = []

        in_data = InputPipelineData(
            message_lists=list(input_data.message_lists),
            attention_masks=None,
            latents=None,
        )

        # Shape: n_components list of minibatch_size ComponentResults
        in_results: list[list[ComponentResult]] = []

        for input_component in self.input_components:
            in_data, results = input_component(in_data, self.victim)
            in_results.append(results)

        # Create outputs for this batch
        # We need to add the existing generations to the conversation lists.
        initial_out_data = copy.deepcopy(input_data)
        for i in range(n_examples):
            initial_out_data.message_lists[i].append(
                Message(role="assistant", content=generations[i])
            )
        out_data = OutputPipelineData(
            message_lists=list(initial_out_data.message_lists),
            attention_masks=None,
            latents=None,
        )

        out_results = []
        for output_component in self.output_components:
            out_data, result = output_component(out_data, self.victim)
            out_results.append(result)

        # Create return outputs
        for i in range(len(input_data.message_lists)):
            input_result_i = []
            output_result_i = []
            for input_result in in_results:
                input_result_i.append(input_result[i])

            for output_result in out_results:
                output_result_i.append(output_result[i])

            pipeline_outputs.append(
                AutoregressiveLayeredDefenseOutput(
                    message_list=out_data.message_lists[i],
                    input_results=tuple(input_result_i),
                    output_results=tuple(output_result_i),
                )
            )

        return pipeline_outputs

    def run_input_components(
        self,
        input_data: ConversationList,
    ) -> Sequence[LayeredDefenseOutput]:
        """Run only the input components of the defense pipeline.

        Args:
            input_data:
                The input messages to run the pipeline on.
        """
        pipeline_outputs: list[LayeredDefenseOutput] = []

        in_data = InputPipelineData(
            message_lists=list(input_data.message_lists),
            attention_masks=None,
            latents=None,
        )

        # Shape: n_components lists of minibatch_size ComponentResults
        in_results: list[list[ComponentResult]] = []

        for input_component in self.input_components:
            in_data, results = input_component(in_data, self.victim)
            in_results.append(results)

        for i in range(len(input_data.message_lists)):
            input_result_i = []
            for input_result in in_results:
                input_result_i.append(input_result[i])

            pipeline_outputs.append(
                LayeredDefenseOutput(
                    input_results=tuple(input_result_i),
                    output_results=tuple([]),
                )
            )

        return pipeline_outputs

    def run_output_components(
        self,
        input_data: ConversationList,
    ) -> Sequence[LayeredDefenseOutput]:
        """Run only the output components of the defense pipeline.

        Args:
            input_data:
                The input messages to run the pipeline on.
        """
        pipeline_outputs: list[LayeredDefenseOutput] = []
        # Make sure messages are assistant messages.
        new_message_lists = []
        for message_list in input_data.message_lists:
            assert len(message_list) == 1
            new_message_list = MessageList(
                messages=[Message(role="assistant", content=message_list[0].content)]
            )
            new_message_lists.append(new_message_list)
        out_data = OutputPipelineData(
            message_lists=new_message_lists,
            attention_masks=None,
            latents=None,
        )

        # Shape: n_components list of minibatch_size ComponentResults
        out_results: list[list[ComponentResult]] = []

        for output_component in self.output_components:
            out_data, results = output_component(out_data, self.victim)
            out_results.append(results)

        for i in range(len(input_data.message_lists)):
            output_result_i = []
            for output_result in out_results:
                output_result_i.append(output_result[i])

            pipeline_outputs.append(
                LayeredDefenseOutput(
                    input_results=tuple([]),
                    output_results=tuple(output_result_i),
                )
            )

        return pipeline_outputs
