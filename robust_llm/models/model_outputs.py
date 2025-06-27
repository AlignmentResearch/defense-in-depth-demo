"""Methods for computing logits/logprobs."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from robust_llm.message_utils import ConversationList
from robust_llm.models.model_utils import (
    SuppressPadTokenWarning,
    build_dataloader,
    maybe_no_grad,
)
from robust_llm.models.vllm_adapter import VLLMModelAdapter
from robust_llm.models.wrapped_model import WrappedModel


class ClassificationLogits:
    """Utility class providing classification logits computation."""

    def __new__(cls, *args, **kwargs):
        raise TypeError(
            f"{cls.__name__} should not be instantiated. Use class methods instead."
        )

    @staticmethod
    def from_tokens(
        victim: WrappedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_no_grad: bool = True,
        minibatch_size: int | None = None,
    ) -> Iterator[torch.Tensor]:
        """Returns the classification logits from the token ids.

        Args:
            victim: The model to calculate the logits on.
            input_ids: The token ids to calculate the logits on.
            attention_mask: The attention mask for the input_ids. Only needed
                if the input_ids are actually padded.
            use_no_grad: Whether to use torch.no_grad(). Defaults to True
                because usually we do not need gradient information, but it is
                needed for gradient-based attacks like GCG. Additionally, this will
                fail loudly if we try to backpropagate through it, whereas False
                will just be silently inefficient.
            minibatch_size: The minibatch size to use. If None, we use
                self.eval_minibatch_size.

        Yields:
            A tensor of logits.
        """

        minibatch_size = victim.get_minibatch_size(input_ids, minibatch_size)

        dataloader = build_dataloader(
            input_ids=input_ids,
            attention_mask=attention_mask,
            minibatch_size=minibatch_size,
        )
        assert victim.accelerator is not None
        dataloader = victim.accelerator.prepare(dataloader)

        with maybe_no_grad(use_no_grad):
            for minibatch in dataloader:
                minibatch_out = victim(
                    input_ids=minibatch["input_ids"],
                    attention_mask=minibatch["attention_mask"],
                )
                logits = minibatch_out.logits
                gathered_logits = victim.accelerator.gather_for_metrics(logits)
                assert isinstance(gathered_logits, torch.Tensor)
                yield gathered_logits

    @staticmethod
    def from_embeddings(
        victim: WrappedModel,
        input_ids: torch.Tensor,
        embeddings: torch.Tensor,
        use_no_grad: bool = True,
    ) -> torch.Tensor:
        """Returns the classification logits from the embeddings.

        Args:
            victim: The model to calculate the logits on.
            input_ids: The token ids to calculate the logits on.
                These are needed to check for cache hits.
            embeddings: The embeddings to calculate the logits on.
            use_no_grad: Whether to use torch.no_grad(). Defaults to True
                because usually we do not need gradient information, but it is
                needed for gradient-based attacks like GCG. Additionally, this will
                fail loudly if we try to backpropagate through it, whereas False
                will just be silently inefficient.

        Returns:
            A tensor of logits.
        """

        if embeddings.shape[0] != 1:
            raise ValueError("This method currently only works for batch size 1.")
        assert embeddings.shape[0] == input_ids.shape[0]
        assert embeddings.shape[1] == input_ids.shape[1]

        with maybe_no_grad(use_no_grad):
            with SuppressPadTokenWarning(victim.model):
                out = victim(
                    input_ids=input_ids,
                    inputs_embeds=embeddings,
                )
        return out.logits

    @classmethod
    def from_messages(
        cls,
        victim: WrappedModel,
        conversation_list: ConversationList,
        use_no_grad: bool = True,
        minibatch_size: int | None = None,
    ) -> Iterator[torch.Tensor]:
        inputs = victim.message_lists_to_tokens(
            conversation_list.message_lists,
            conversation_list.message_filter,
            add_generation_prompt=conversation_list.add_generation_prompt,
            padding_side="right",
        )
        return cls.from_tokens(
            victim=victim,
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            use_no_grad=use_no_grad,
            minibatch_size=minibatch_size,
        )


class GenerationLogits:
    """Utility class providing generation logits computation."""

    def __new__(cls, *args, **kwargs):
        raise TypeError(
            f"{cls.__name__} should not be instantiated. Use class methods instead."
        )

    @staticmethod
    def from_embeddings(
        victim: WrappedModel,
        input_ids: torch.Tensor,
        embeddings: torch.Tensor,
        use_no_grad: bool = True,
    ) -> Iterator[torch.Tensor]:
        """Returns the classification logits from the embeddings.

        Args:
            input_ids: The token ids to calculate the logits on.
                These are needed to check for cache hits.
            embeddings: The embeddings to calculate the logits on.
            use_no_grad: Whether to use torch.no_grad(). Defaults to True
                because usually we do not need gradient information, but it is
                needed for gradient-based attacks like GCG. Additionally, this will
                fail loudly if we try to backpropagate through it, whereas False
                will just be silently inefficient.

        Yields:
            A tensor of logits.
        """
        if isinstance(victim, VLLMModelAdapter):
            raise NotImplementedError(
                "vLLM does not support generation logits from embeddings"
            )

        if embeddings.shape[0] != 1:
            raise ValueError("This method currently only works for batch size 1.")
        assert embeddings.shape[0] == input_ids.shape[0]
        # We don't need these to be equal because the input_ids are just for cache.
        assert embeddings.shape[1] >= input_ids.shape[1]

        with maybe_no_grad(use_no_grad):
            with SuppressPadTokenWarning(victim.model):
                out = victim(
                    input_ids=input_ids,
                    inputs_embeds=embeddings,
                )
                yield out.logits

    @staticmethod
    def from_tokens(
        victim: WrappedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_no_grad: bool = True,
        minibatch_size: int | None = None,
    ) -> Iterator[torch.Tensor]:
        return victim.generation_logits_from_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_no_grad=use_no_grad,
            minibatch_size=minibatch_size,
        )

    @classmethod
    def from_messages(
        cls,
        victim: WrappedModel,
        conversation_list: ConversationList,
        use_no_grad: bool = True,
        minibatch_size: int | None = None,
    ) -> Iterator[torch.Tensor]:
        inputs = victim.message_lists_to_tokens(
            conversation_list.message_lists,
            conversation_list.message_filter,
            padding_side="right",
        )
        return cls.from_tokens(
            victim=victim,
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            use_no_grad=use_no_grad,
            minibatch_size=minibatch_size,
        )
