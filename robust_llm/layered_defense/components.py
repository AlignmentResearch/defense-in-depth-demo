from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Type

import torch
from accelerate import Accelerator

from robust_llm.message_utils import MessageList
from robust_llm.models.wrapped_model import WrappedModel


@dataclass
class PipelineData(ABC):
    """Data to pass between components.

    Args:
        message_lists:
            [dataset_size] list of MessageList for each example.
        attention_masks:
            [batches] list of [batch_size, seq_len] tensors
        latents:
            [batches] list of dicts with layer index as key and
            [batch_size, seq_len, d_model] tensors of victim model
            latent states as values.
    """

    message_lists: list[MessageList]
    # logits: list[torch.Tensor]  # Don't need logits for filters/probes.
    attention_masks: list[torch.Tensor] | None
    latents: list[dict[int, torch.Tensor]] | None


@dataclass
class InputPipelineData(PipelineData):
    pass


@dataclass
class OutputPipelineData(PipelineData):
    pass


@dataclass(frozen=True)
class ComponentResult:
    """A result from a component.

    Attributes:
        flag:
            Whether the component flagged the input or output as harmful.
        name:
            The name of the component.
        score:
            The score of the component (between 0 and 1).
            If given, should be a single-item tensor rather than a float, to allow for
            backpropagation.
    """

    flag: bool
    name: str
    score: torch.Tensor | None

    def __post_init__(self) -> None:
        # If score is given, it should be a single-item tensor.
        if self.score is not None and (
            not isinstance(self.score, torch.Tensor) or self.score.ndim != 0
        ):
            raise ValueError(f"Score should be a single-item tensor, got {self.score}.")

    def to_dict(self, prefix: str = "") -> dict[str, Any]:
        """Convert the component result to a dictionary.

        Args:
            prefix: Optional prefix to add to the dictionary keys.

        Returns:
            Dictionary with keys {prefix}name, {prefix}flag, {prefix}score
        """
        return {
            f"{prefix}name": self.name,
            f"{prefix}flag": self.flag,
            f"{prefix}score": self.score.item() if self.score is not None else None,
        }


class DefenseComponent(ABC):
    @abstractmethod
    def __init__(
        self, victim: Optional[WrappedModel], accelerator: Accelerator, **kwargs
    ):
        """Initialize the component."""


class InputComponent(DefenseComponent):
    @abstractmethod
    def __call__(
        self, data: InputPipelineData, victim: Optional[WrappedModel]
    ) -> tuple[InputPipelineData, list[ComponentResult]]:
        """Runs an input component on the text.

        Args:
            data: The input data to work with.
            victim: The victim model to use for the input component.

        Returns:
            A tuple containing the input text (optionally modified) and a
            ComponentResult with flag info.
        """  # noqa: DCO031  # (we want to document the return value)


class OutputComponent(DefenseComponent):
    @abstractmethod
    def __call__(
        self, data: OutputPipelineData, victim: Optional[WrappedModel]
    ) -> tuple[OutputPipelineData, list[ComponentResult]]:
        """Runs an output component on the text.

        Args:
            data: The output data to work with.
            victim: The victim model to use for the output component.

        Returns:
            A tuple containing the output text (optionally modified) and a
            ComponentResult with flag info.

        """  # noqa: DCO031  # (we want to document the return value)


@dataclass
class LayeredDefenseOutput(ABC):
    input_results: tuple[ComponentResult, ...]
    output_results: tuple[ComponentResult, ...]


@dataclass
class AutoregressiveLayeredDefenseOutput(LayeredDefenseOutput):
    message_list: MessageList
    clean_message_list: MessageList | None = None

    def with_clean_input_text(
        self, clean_input_text: str
    ) -> "AutoregressiveLayeredDefenseOutput":
        """Returns a new AutoregressiveLayeredDefenseOutput with clean_input_text.

        Args:
            clean_input_text: The clean input text to use.
        """
        clean_message_list = self.message_list.replace_first_user_message(
            clean_input_text
        )
        return AutoregressiveLayeredDefenseOutput(
            message_list=self.message_list,
            clean_message_list=clean_message_list,
            input_results=self.input_results,
            output_results=self.output_results,
        )

    def with_clean_message_list(
        self, clean_message_list: MessageList
    ) -> "AutoregressiveLayeredDefenseOutput":
        return AutoregressiveLayeredDefenseOutput(
            message_list=self.message_list,
            clean_message_list=clean_message_list,
            input_results=self.input_results,
            output_results=self.output_results,
        )


@dataclass
class BinaryLayeredDefenseOutput(LayeredDefenseOutput):
    success: bool


@dataclass
class LossLayeredDefenseOutput(LayeredDefenseOutput):
    loss: torch.Tensor


class ComponentRegistry:
    _components: dict[str, Type[DefenseComponent]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(component_cls: Type[DefenseComponent]) -> Type[DefenseComponent]:
            cls._components[name] = component_cls
            return component_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[DefenseComponent]:
        if name not in cls._components:
            raise ValueError(
                f"Unknown component: {name}\n"
                f"Available components: {cls._components.keys()}"
            )
        return cls._components[name]
