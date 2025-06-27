import warnings
from contextlib import contextmanager

import torch
from transformers.cache_utils import DynamicCache
from transformers.tokenization_utils_base import BatchEncoding
from typing_extensions import override

from robust_llm.dist_utils import is_main_process
from robust_llm.models.model_utils import PastKeyValues
from robust_llm.models.wrapped_model import Prompt, WrappedModel


class CachingWrappedModel(WrappedModel):
    """Thin wrapper around a WrappedModel for KV caching.

    We instantiate this class with a WrappedModel and an empty cache.
    The cache maps sequences of token_ids to their KV values.

    The cache is currently populated manually by calling add_to_cache.

    Attributes:
    _wrapped_model: The original model that this class wraps.
    cache: The dictionary keys are tuples of
        token IDs representing input sequences, and the dictionary values are
        PastKeyValues objects that contain the cached key-value pairs for these
        sequences.
    """

    def __init__(self, wrapped_model: WrappedModel, max_cache_size: int = 10):
        self._wrapped_model = wrapped_model
        self.cache: dict[tuple[int], PastKeyValues] = {}
        self.max_cache_size = max_cache_size

    def __getattr__(self, name):
        """If an attribute is not found in this class, look in the WrappedModel."""
        return getattr(self._wrapped_model, name)

    def __setattr__(self, name, value):
        """Set all attributes except _wrapped_model and cache in the WrappedModel."""
        if name in ["_wrapped_model", "cache"]:
            super().__setattr__(name, value)
        else:
            setattr(self._wrapped_model, name, value)

    def __call__(self, **inputs):
        return self.forward(**inputs)

    def _apply_cache_hit(self, inputs: dict) -> bool:
        """Return the cache hit for the given inputs, if it exists.

        WARNING: This modifies 'inputs' in place.

        Returns:
            True if a cache hit was found, False otherwise.
        """
        input_ids = inputs["input_ids"]
        cache_result = self.return_cache_overlap(input_ids)
        if cache_result is not None:
            prefix, overlapping_cached_kv = cache_result
            if len(prefix) == input_ids.shape[1]:
                warnings.warn(
                    "Cache hit is the entire input. Unfortunately this throws an"
                    " error inside PreTrainedModel. Luckily we shouldn't need this"
                    " use-case often since we'll always be appending an attack."
                )
                # We still got a cache hit, we just can't apply it.
                return True

            inputs["input_ids"] = input_ids[:, len(prefix) :]
            # If we have inputs_embeds, we need to truncate them as well.
            if "inputs_embeds" in inputs:
                inputs["inputs_embeds"] = inputs["inputs_embeds"][:, len(prefix) :]

            inputs["past_key_values"] = DynamicCache.from_legacy_cache(
                overlapping_cached_kv,  # type: ignore
            )
            return True
        return False

    def forward(self, **inputs):
        if "input_ids" not in inputs:
            raise ValueError("CachingWrappedModel requires input_ids.")
        input_ids = inputs["input_ids"]
        cache_hit = self._apply_cache_hit(inputs)
        if not cache_hit:
            if len(self.cache) < self.max_cache_size:
                # On a cache miss, if the cache is not full, add a representative of the
                # input to the cache. This involves an extra forward pass.
                # We take the first row of input_ids, keeping the dimension.
                if "attention_mask" in inputs:
                    attention_mask = inputs["attention_mask"][0:1, :]
                else:
                    attention_mask = None
                self.add_to_cache(input_ids[0:1, :], attention_mask)
                # We then try the cache again.
                cache_hit = self._apply_cache_hit(inputs)
                if not cache_hit:
                    warnings.warn(
                        "Cache missed even when directly populated."
                        " This is expected if using BoN."
                    )
        # If there is no attention mask, we add one, since this affects behavior as of
        # 4.42.0 (see https://github.com/huggingface/transformers/issues/31943)
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = self._compute_attn_mask(inputs)

        with self._wrapped_model.dont_count_flops():
            out = self._wrapped_model.forward(**inputs)

        if self.accelerator is not None:
            input_ids = self.accelerator.gather_for_metrics(input_ids)
        if is_main_process():
            self._wrapped_model.update_flop_count({"input_ids": input_ids})

        return out

    def _compute_attn_mask(self, inputs: dict) -> torch.Tensor:
        """Return an attention mask for the given inputs.

        Assumes that the input is not padded (if it's padded, a mask should have
        been passed in).

        Getting the shape right is a little tricky.
        """
        pad_token_id = self._wrapped_model.config.pad_token_id
        if pad_token_id is not None:
            assert "input_ids" in inputs, "Expected input_ids in inputs."
            assert pad_token_id not in inputs["input_ids"], "Input should not be padded"

        batch_size = inputs["input_ids"].shape[0]
        if "past_key_values" in inputs:
            # seq_len is dim 2 of the component tensors
            cache_len = inputs["past_key_values"][0][0].shape[2]
        else:
            cache_len = 0
        # inputs_embeds take priority; input_ids will be dropped if they are present.
        if "inputs_embeds" in inputs:
            seq_len = inputs["inputs_embeds"].shape[1]
        elif "input_ids" in inputs:
            seq_len = inputs["input_ids"].shape[1]
        else:
            raise ValueError("Expected input_ids or inputs_embeds in inputs.")

        return torch.ones((batch_size, seq_len + cache_len), device=self.device)

    @torch.no_grad()
    def add_to_cache(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> None:
        """Add to the KV cache for the given token_ids.

        The input should contain no padding tokens.

        We use torch.no_grad because having gradients in the cache is
        unnecessary and causes issues with trying to backprop twice.
        """
        # Preconditions.
        assert input_ids.shape[0] == 1, "Only one input sequence at a time (for now)."
        if attention_mask is None:
            if self._wrapped_model.right_tokenizer.pad_token_id in input_ids:
                raise ValueError(
                    "Input should not be padded if no attention mask is provided."
                )
        else:
            # Since we are doing a single sequence, we can drop the padding tokens
            # Drop padding tokens since we are doing a single sequence
            input_ids = input_ids[attention_mask.to(torch.bool)].unsqueeze(0)

        with self._wrapped_model.dont_count_flops():
            outputs = self._wrapped_model(input_ids=input_ids, use_cache=True)
        kv_cache = outputs.past_key_values
        # We access the first element of the batch, which is the only one.
        self.cache[tuple(input_ids[0].tolist())] = kv_cache

    def return_cache_overlap(
        self, input_ids: torch.Tensor
    ) -> tuple[tuple[int, ...], PastKeyValues] | None:
        """Return as much of the cache as possible for the given token_ids.

        We look for the longest prefix of the input_ids that is also a prefix
        of something in the cache. We return the corresponding KV values.

        Because ultimately the KV cache consists of tensors, we need to return
        the same number of tokens for each tensor in the cache. This means we
        find a common prefix of *all* the sequences in input_ids, and then
        check for a cache hit with that.

        Note: In the body of this function, we use "query" to refer to the
        input_ids used to search the cache, and "key" to refer to the token_ids
        in the cache; they are not related to attention.
        """

        query = get_common_prefix_for_batch(input_ids)

        # Find the key with the longest common prefix with the query, and
        # its length.
        longest_common_prefix_key = None
        longest_common_prefix_length = -1
        for key in self.cache.keys():
            common_prefix = get_common_prefix(query, key)
            if len(common_prefix) > longest_common_prefix_length:
                longest_common_prefix_key = key
                longest_common_prefix_length = len(common_prefix)

        # If there is no cache hit, return None.
        if longest_common_prefix_length <= 0:
            return None
        assert longest_common_prefix_key is not None

        # Return the KV values corresponding to the longest common prefix.
        cached_kv = self.cache[longest_common_prefix_key]
        overlapping_cached_kv = get_overlapping_kv(
            cached_kv, longest_common_prefix_length
        )
        final_kv = repeat_kv_for_batch(overlapping_cached_kv, input_ids.shape[0])

        longest_common_prefix = longest_common_prefix_key[:longest_common_prefix_length]
        return longest_common_prefix, final_kv

    @override
    def tokenize(
        self,
        text: str | list[str],
        return_tensors: str | None = None,
        padding_side: str | None = None,
        add_special_tokens: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        """Tokenize the input text using the WrappedModel's tokenizer."""
        return self._wrapped_model.tokenize(
            text,
            return_tensors=return_tensors,
            padding_side=padding_side,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )

    @override
    def maybe_apply_user_template(self, text: Prompt) -> Prompt:
        """If working with a chat model, return text with chat template applied.

        Args:
            text: The user prompt(s) to apply the chat template to.

        Returns:
            The text with the chat template applied.
        """
        return self._wrapped_model.maybe_apply_user_template(text=text)

    @classmethod
    def load_tokenizer(cls, model_config):
        raise NotImplementedError("Does not need load_tokenizer.")


def get_common_prefix(input_ids: torch.Tensor, key: tuple[int]) -> torch.Tensor:
    """Return the longest common prefix of the input_ids and the key.

    NOTE: input_ids and key could be different lengths.
    """
    # Preconditions.
    assert len(input_ids.shape) == 1, "Input should be a single sequence."

    prefix = []
    for i, token_id in enumerate(key):
        if i >= len(input_ids) or input_ids[i] != token_id:
            break
        prefix.append(token_id)
    return torch.tensor(prefix, dtype=torch.long)


def get_common_prefix_for_batch(input_ids: torch.Tensor) -> torch.Tensor:
    """Return the longest common prefix of all sequences in the batch of input_ids.

    Args:
        input_ids: Tensor of shape (batch_size, sequence_length) with token ids.

    Returns:
        prefix: Tensor of shape (prefix_length,) containing the common prefix.
    """

    # Preconditions.
    assert len(input_ids.shape) == 2, "Input should be a batch of sequences."
    assert input_ids.shape[0] > 0, "Input should not be empty."

    # Tensor of booleans indicating where each sequence agrees and disagrees
    # with input_ids[0].
    boolean_mask = input_ids != input_ids[0].unsqueeze(0)
    # Look across the sequences and check if any of them diverge from the first.
    first_divergence_points = boolean_mask.any(dim=0)
    # Find the indices of the divergence points.
    divergence_indices = torch.nonzero(first_divergence_points).squeeze(dim=1)
    # If there are no divergence points then all sequences are the same; return
    # the first one arbitrarily.
    if len(divergence_indices) == 0:
        return input_ids[0]
    divergence_ix = divergence_indices[0].item()
    assert isinstance(divergence_ix, int)
    prefix = input_ids[0, :divergence_ix]

    # Postconditions.
    assert len(prefix.shape) == 1, "Prefix should be a 1D tensor."
    assert len(prefix) <= input_ids.shape[1], "Prefix should be shorter than inputs."

    return prefix


def get_overlapping_kv(kv: PastKeyValues, length: int) -> PastKeyValues:
    """Return the KV values corresponding to the first 'length' tokens.

    Because PastKeyValues is a tuple of tuples of tensors, it's a little
    tricky to shorten.
    """

    def shorten(tensor: torch.Tensor) -> torch.Tensor:
        """Shorten a tensor to the first 'length' tokens."""
        # Sequence length is dimension 2.
        return tensor[:, :, :length, :]

    return tuple(tuple(shorten(tensor) for tensor in layer) for layer in kv)


def repeat_kv_for_batch(kv: PastKeyValues, batch_size: int) -> PastKeyValues:
    """Repeat the KV values for a batch of sequences.

    This is useful when we want to use the same KV values for multiple sequences.
    """
    # Preconditions.
    assert batch_size > 0, "Batch size should be positive."
    assert len(kv[0][0].shape) == 4, "KV values should have 4 dimensions."
    assert kv[0][0].shape[0] == 1, "Initial KV values should have batch size 1."

    def repeat(tensor: torch.Tensor) -> torch.Tensor:
        """Repeat a tensor for a batch of sequences."""
        # Dimension 0 is the batch size.
        return tensor.repeat(batch_size, 1, 1, 1)

    return tuple(tuple(repeat(tensor) for tensor in layer) for layer in kv)


@contextmanager
def get_caching_model(model: WrappedModel, max_cache_size: int = 10):
    """Get an empty CachingWrappedModel.

    It'll be populated with examples when called.
    """
    if "vllm" in model.family:
        warnings.warn(
            "CachingWrappedModel is not supported for vLLM models. "
            "get_caching_model will return the original model."
        )
        try:
            yield model
        finally:
            pass  # No cleanup needed for the original model
    else:
        caching_model = CachingWrappedModel(model, max_cache_size=max_cache_size)
        try:
            yield caching_model
        finally:
            del caching_model
