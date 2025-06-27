from __future__ import annotations

from argparse import Namespace
from collections.abc import Sequence
from typing import Any

import torch
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def equal_ignore_padding(x: torch.Tensor, y: torch.Tensor, pad_token_id: int) -> bool:
    """Checks if two 1D tensors are equal, ignoring padding at the end."""
    while len(x) > 0 and x[-1] == pad_token_id:
        x = x[:-1]
    while len(y) > 0 and y[-1] == pad_token_id:
        y = y[:-1]
    return x.equal(y.to(x.device))


class FakeModelForSequenceClassification:
    """Fake model class used in tests."""

    @property
    def name_or_path(self) -> str:
        return "fake-model/fake-model-for-sequence-classification"

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def num_labels(self) -> int:
        return 2

    def num_parameters(self, exclude_embeddings: bool = True) -> int:
        return 0

    @property
    def config(self):
        return Namespace(
            pad_token_id=1,
            eos_token_id=2,
            task_specific_params={},
            id2label={0: "LABEL_0", 1: "LABEL_1"},
        )

    def can_generate(self) -> bool:
        return False

    def eval(self) -> FakeModelForSequenceClassification:
        return self

    def to(self, *args, **kwargs) -> FakeModelForSequenceClassification:
        return self

    def forward(
        self, input_ids: torch.Tensor, *args, **kwargs
    ) -> SequenceClassifierOutput:
        """Mimics a sequence classification model.

        Returns logits of the shape [batch_size, num_labels].
        """
        return SequenceClassifierOutput(
            logits=torch.rand(input_ids.shape[0], self.num_labels),  # type: ignore
        )

    def __call__(
        self, input_ids: torch.Tensor, *args, **kwargs
    ) -> dict[str, torch.Tensor]:
        return self.forward(input_ids, *args, **kwargs)

    def estimate_tokens(self, input_dict: dict[str, Any]) -> int:
        return 1

    def register_forward_hook(self, hook: Any) -> None:
        pass

    def register_full_backward_hook(self, hook: Any) -> None:
        pass

    def modules(self):
        return [self, self]

    @property
    def num_processes(self):
        return 1


class FakeClassifierWithPositiveList(FakeModelForSequenceClassification):
    """Fake classification model with a pre-defined list of positive examples."""

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, positives: Sequence[torch.Tensor]
    ):
        self.tokenizer = tokenizer
        self.positives = positives

    def forward(
        self, input_ids: torch.Tensor, *args, **kwargs
    ) -> SequenceClassifierOutput:
        logits = []
        pad_token_id = self.tokenizer.pad_token_id
        assert isinstance(pad_token_id, int)
        for x in input_ids:
            logits.append(
                [0.0, 1.0]
                if any(
                    [equal_ignore_padding(x, y, pad_token_id) for y in self.positives]
                )
                else [1.0, 0.0]
            )
        return SequenceClassifierOutput(logits=torch.tensor(logits))  # type: ignore
