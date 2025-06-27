from __future__ import annotations

import warnings
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional, TypeVar

import datasets
import torch
import torch.nn.functional as F
import vllm
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from transformers.models.llama4.modeling_llama4 import Llama4ForCausalLM
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from robust_llm import logger
from robust_llm.config.model_configs import ModelConfig

if TYPE_CHECKING:
    from robust_llm.models import WrappedModel

# The tensors are (batch_size, n_heads, seq_len, head_dim).
PastKeyValues = tuple[tuple[torch.Tensor, ...], ...]

Tdict = TypeVar("Tdict", bound=dict)
MASK_LABEL = -100


class InferenceType(Enum):
    """The type of inference the model is used for.

    This is used to determine the type of model to load from HuggingFace.
    """

    CLASSIFICATION = "classification"
    GENERATION = "generation"


@dataclass(frozen=True)
class AutoregressiveOutput:
    """The output of an autoregressive model.

    Attributes:
        input_text:
            The input text given to the model.
        output_text:
            The output text from the model.
        clean_input_text:
            The unattacked version of the input_text, if available.
    """

    input_text: str
    output_text: str
    clean_input_text: str | None = None

    def with_clean_input_text(self, clean_input_text: str) -> AutoregressiveOutput:
        """Returns a new AutoregressiveOutput with the clean_input_text set."""
        return AutoregressiveOutput(
            input_text=self.input_text,
            output_text=self.output_text,
            clean_input_text=clean_input_text,
        )

    def get_full_text(self, delimiter: str = "\n-----\n") -> str:
        """Get the full text (input + output) for logging to wandb."""
        return self.input_text + delimiter + self.output_text


def load_vllm_model(
    name_or_path: str,
    tokenizer_name: str,
    revision: str,
    inference_type: InferenceType,
    strict_load: bool,
    dtype: str,
    attention_implementation: Optional[str] = None,
    num_classes: Optional[int] = None,
    gpu_memory_utilization: float = 0.9,
    n_logprobs: int = 128,
    enforce_eager: bool = True,
    enable_chunked_prefill: bool = True,
) -> vllm.LLM:
    """Loads a model with vLLM.

    Args:
        name_or_path: The name or path of the model.
        tokenizer_name: The name or path of the tokenizer.
        revision: The revision of the model.
        inference_type: The type of inference the model is used for.
        strict_load: Whether to enforce that no weights are ignored or randomly
            initialized while loading.
        dtype: Data type of the model.
        attention_implementation: The implementation with which to compute
            attention (ignored).
        num_classes: The number of classes for a classification model (ignored).
        gpu_memory_utilization: The fraction of GPU memory to use for the model.
        n_logprobs: The number of logprobs to return for each prompt token.
        enforce_eager: Whether to enforce eager execution of the model. Eager
            leads to faster loading, but slower inference.
        enable_chunked_prefill: Whether to enable chunked prefill for the model.
            This is a workaround for a bug in vllm:
            github.com/vllm-project/vllm/issues/5907.
    """
    if strict_load:
        raise ValueError("strict_load is not supported for vLLM models")

    match inference_type:
        case InferenceType.CLASSIFICATION:
            raise ValueError("Classification is not supported for vLLM models")
        case InferenceType.GENERATION:
            model = vllm.LLM(
                model=name_or_path,
                tokenizer=tokenizer_name,
                revision=revision,
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                max_logprobs=n_logprobs,
                # DEBUG(ian): Using eager so tests are faster
                enforce_eager=enforce_eager,
                # This is less than the default max len, which means we should get
                # better parallelism from vllm.
                max_model_len=2048,
                # enable_chunked_prefill and max_num_seqs/max_num_batched_tokens
                # are a workaround for a bug in vllm:
                # github.com/vllm-project/vllm/issues/5907
                enable_chunked_prefill=enable_chunked_prefill,
                max_num_seqs=256 if enable_chunked_prefill else None,
                max_num_batched_tokens=256 if enable_chunked_prefill else None,
                hf_overrides={"attn_implementation": attention_implementation},
            )

    return model


@dataclass(frozen=True)
class ModelKey:
    name_or_path: str
    revision: str
    inference_type: InferenceType
    torch_dtype: torch.dtype
    attention_implementation: Optional[str]
    num_classes: int


class ModelLoader:
    """Loads a model from given args with caching.

    This class is a singleton that loads a model from given args, and
    caches the model so that it is not loaded multiple times.
    """

    _model_dict: dict[ModelKey, tuple[PreTrainedModel, dict[str, str]]] = {}

    def __init__(self) -> None:
        raise ValueError("ModelLoader should not be instantiated directly")

    @staticmethod
    def _load_model(
        name_or_path: str,
        revision: str,
        inference_type: InferenceType,
        torch_dtype: torch.dtype,
        attention_implementation: Optional[str] = None,
        num_classes: int = 2,
    ) -> tuple[PreTrainedModel, dict[str, str]]:
        match inference_type:
            case InferenceType.CLASSIFICATION:
                return AutoModelForSequenceClassification.from_pretrained(
                    name_or_path,
                    revision=revision,
                    output_loading_info=True,
                    num_labels=num_classes,
                    use_cache=False,  # By default we don't want to output cache values.
                    torch_dtype=torch_dtype,
                    attn_implementation=attention_implementation,
                )
            case InferenceType.GENERATION:
                with warnings.catch_warnings():
                    # We set the generation config later so don't show this warning.
                    warnings.filterwarnings(
                        "ignore",
                        category=UserWarning,
                        message=".*do_sample.*False.*",
                    )
                    # Special case for Llama4: We can only set use_cache=False if
                    # we use Llama4ForCausalLM rather than AutoModelForCausalLM.
                    model_cls: Any
                    if name_or_path.startswith(
                        "meta-llama/Llama-Guard-4",
                    ) or name_or_path.startswith(
                        "meta-llama/Llama-4",
                    ):
                        model_cls = Llama4ForCausalLM
                    else:
                        model_cls = AutoModelForCausalLM
                    return model_cls.from_pretrained(
                        name_or_path,
                        revision=revision,
                        output_loading_info=True,
                        use_cache=False,  # We don't want to output cache values.
                        torch_dtype=torch_dtype,
                        attn_implementation=attention_implementation,
                    )  # pyright: ignore[reportReturnType]  # Both return a tuple.

    @classmethod
    def load_model(
        cls,
        accelerator: Accelerator | None,
        name_or_path: str,
        revision: str,
        inference_type: InferenceType,
        torch_dtype: torch.dtype,
        attention_implementation: Optional[str] = None,
        num_classes: int = 2,
        overwrite_cache: bool = False,
    ) -> tuple[PreTrainedModel, dict[str, str]]:
        if torch.distributed.is_initialized() and not overwrite_cache:
            logger.warning(
                "Overwriting cache because we are running in distributed mode."
            )
            overwrite_cache = True
        key = ModelKey(
            name_or_path=name_or_path,
            revision=revision,
            inference_type=inference_type,
            torch_dtype=torch_dtype,
            attention_implementation=attention_implementation,
            num_classes=num_classes,
        )
        if key in cls._model_dict and not overwrite_cache:
            return cls._model_dict[key]

        model, loading_info = cls._load_model(
            **asdict(key),
        )
        cls._model_dict[key] = (
            prepare_model_with_accelerate(accelerator, model) if accelerator else model,
            loading_info,
        )
        return cls._model_dict[key]


def load_hf_model(
    accelerator: Accelerator | None,
    name_or_path: str,
    revision: str,
    inference_type: InferenceType,
    strict_load: bool,
    torch_dtype: torch.dtype,
    attention_implementation: Optional[str] = None,
    num_classes: Optional[int] = None,
    overwrite_cache: bool = False,
) -> PreTrainedModel:
    """Loads a model from HuggingFace.

    NOTE: We have to suppress a type error because the from_pretrained method
    returns a tuple when output_loading_info is set to True but this is not reflected
    in the type hints.

    Args:
        accelerator: The accelerator to prepare the model with.
        name_or_path: The name or path of the model.
        revision: The revision of the model.
        inference_type: The type of inference the model is used for.
        strict_load: Whether to enforce that no weights are ignored or randomly
            initialized while loading.
        torch_dtype: Data type of the model.
        attention_implementation: The implementation with which to compute
            attention.
        num_classes: The number of classes for a classification model.
        overwrite_cache: Whether to force overwrite of the cache.
    """
    # Even though num_labels is optional, passing None to it will cause an error
    # because if a value is passed, it must be an int. Two is the default.
    if num_classes is None:
        num_classes = 2

    model, loading_info = ModelLoader.load_model(
        accelerator=accelerator,
        name_or_path=name_or_path,
        revision=revision,
        inference_type=inference_type,
        torch_dtype=torch_dtype,
        attention_implementation=attention_implementation,
        num_classes=num_classes,
        overwrite_cache=overwrite_cache,
    )
    # Optionally, check that there are no weights skipped or randomly initialized.
    if strict_load:
        assert loading_info_is_empty(loading_info), (  # type: ignore
            f"Loading info is not empty: {loading_info}"
        )
    return model


def loading_info_is_empty(loading_info: dict[str, str]) -> bool:
    """Checks whether there is any loading info.

    This is useful for checking whether there are any weights of the loaded
    model that are unused, or new weights that are randomly initialized.

    Args:
        loading_info: a dictionary mapping potential events to lists of weight names.
            If the loaded model uses exactly all of the downloaded weights, then
            all these lists should be empty.

    Returns:
        True if all the lists in loading_info are empty, False otherwise.
    """
    return all(len(v) == 0 for v in loading_info.values())


def prepare_model_with_accelerate(
    accelerator: Accelerator, model: PreTrainedModel
) -> PreTrainedModel:
    model = accelerator.prepare(model)
    # When using FSDP, there is some lazy initialization that happens. Enforce it here
    # to avoid issues from lack of proper initialization (e.g. when accessing embedding
    # layer in GCG).
    _ = model(
        input_ids=torch.tensor([[0]], device=accelerator.device),
        attention_mask=torch.tensor([[1]], device=accelerator.device),
    )

    return model


class SuppressPadTokenWarning:
    """Context manager to suppress pad token warnings.

    These warnings occur when you call a model with inputs_embeds rather than
    tokens. We get the embeddings by running the input tokens through the
    embedding layer. When we run the model on embeddings rather than tokens,
    information about whether some of the input tokens were padding tokens is lost,
    so padding tokens (if present) can't be masked out and huggingface
    (reasonably) gives a warning: it's important to mask out padding tokens
    since otherwise they are interpreted as normal input tokens and
    they affect the output of the model.

    The problem is the warning is repeated for every single call to the model,
    which can be annoying and make the logs unreadable. Additionally, since the
    warning is not from the 'warnings' module, it is not easy to suppress.

    This context manager suppresses the warning by disabling the padding token
    for the duration of the model call. Since we shouldn't have any padding
    tokens in the input sequence due to the issues mentioned above, and since
    the padding token is not used when calling the model with inputs_embeds,
    this should be safe.
    """

    def __init__(self, model: PreTrainedModel | WrappedModel):
        self.model = model
        self.saved_pad_token = model.config.pad_token_id

    def __enter__(self):
        self.model.config.pad_token_id = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.config.pad_token_id = self.saved_pad_token


def combine_output_dicts(dicts: Sequence[Tdict]) -> Tdict:
    """Combine outputs into a single object."""
    target_keys = dicts[0].keys()
    target_type = type(dicts[0])
    target_types = {k: type(dicts[0][k]) for k in target_keys}

    assert all(d.keys() == target_keys for d in dicts)
    assert all(isinstance(d, target_type) for d in dicts)
    assert all(isinstance(d[k], target_types[k]) for d in dicts for k in target_keys)

    combined = target_type()
    for key in target_keys:
        nones = [d[key] is None for d in dicts]
        if any(nones):
            assert all(nones)
            combined[key] = None
        else:
            combined[key] = torch.cat([d[key] for d in dicts])
    return combined


def build_dataloader(minibatch_size: int, **kwargs):
    """Build a DataLoader from arbitrary keyword arguments.

    This saves us having to manually specify the inputs multiple times.

    Args:
        minibatch_size: The size of the minibatches.
        kwargs: The keyword arguments with the actual data we want in the
            DataLoader. The keys should be the names of the fields in the dataset
            (e.g. input_ids, attention_mask, ...).
    """
    dataset = datasets.Dataset.from_dict(
        {
            **kwargs,
        }
    ).with_format("torch")

    dataloader = DataLoader(
        dataset=dataset,  # type: ignore  # Typehint is wrong in DataLoader.
        batch_size=minibatch_size,
    )
    return dataloader


def dict_to_device(d: Tdict, device: str | torch.device) -> Tdict:
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.to(device=device)
    return d


@contextmanager
def maybe_no_grad(use_no_grad: bool):
    if use_no_grad:
        with torch.no_grad():
            yield
    else:
        yield


def remove_padding_tokens(
    tokenizer: PreTrainedTokenizerBase, texts: list[str]
) -> list[str]:
    """Remove padding tokens from a list of output texts.

    Args:
        tokenizer: The tokenizer used to tokenize the text.
        texts: The list of texts to remove padding tokens from.

    Returns:
        The list of texts with padding tokens removed.
    """
    assert isinstance(tokenizer.pad_token, str)
    return [text.replace(tokenizer.pad_token, "") for text in texts]


def compute_batch_sizes_from_config(config: ModelConfig):
    """Computes the train and eval minibatch sizes from the config."""

    max_mb_size = max(
        1, int(config.max_minibatch_size * config.env_minibatch_multiplier)
    )
    train_mb_size = max(
        1,
        min(
            int(max_mb_size * config.train_minibatch_multiplier),
            config.effective_batch_size,
        ),
    )
    eval_mb_size = max(1, int(max_mb_size * config.eval_minibatch_multiplier))
    return train_mb_size, eval_mb_size


def get_hidden_size(config: PretrainedConfig) -> int:
    """Get the residual stream dimension from a model config."""
    for attr in ["hidden_size", "d_model", "n_embd"]:
        if hasattr(config, attr):
            return getattr(config, attr)
    raise ValueError(f"Could not find hidden size in config: {config}")


def get_num_layers(config: PretrainedConfig) -> int:
    for attr in ["num_hidden_layers"]:
        if hasattr(config, attr):
            return getattr(config, attr)
    raise ValueError(f"Could not find number of layers in config: {config}")


def lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute the loss for a language model.

    Based on GPT2LMHeadModel.forward()

    Args:
        logits: Shape (batch, seq_len, vocab_size). The logits from the model.
        labels: Shape (batch, seq_len). The ground truth labels.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    logits_reshaped = shift_logits.reshape(
        -1, shift_logits.shape[-1]
    )  # [batch_size * (seq_len - 1), vocab_size]
    labels_reshaped = shift_labels.reshape(-1)  # [batch_size * (seq_len - 1)]
    token_losses = F.cross_entropy(
        logits_reshaped, labels_reshaped, reduction="none", ignore_index=MASK_LABEL
    )  # [batch_size * (seq_len - 1)]
    losses = token_losses.view(shift_labels.shape[0], -1)  # [batch_size, seq_len - 1]
    # We take the mean across the token positions, excluding the padding tokens.
    mask = (shift_labels != MASK_LABEL).float()  # [batch_size, seq_len - 1]
    masked_losses = losses * mask  # [batch_size, seq_len - 1]
    return masked_losses.sum(dim=1) / mask.sum(dim=1)  # [batch_size]


def mean_of_batch_losses(losses: torch.Tensor, mask: torch.Tensor) -> float:
    """Compute the mean of the losses for each example in the batch.

    This is necessary because each batch may have a different number of
    non-padding tokens.

    Args:
        losses: Shape (batch,). The LM losses.
        mask: Shape (batch, seq_len). The attentionmask.
    """
    n_non_pad_per_example = mask[:, 1:].sum(dim=1)
    n_non_pad = n_non_pad_per_example.sum()
    loss_all_tokens = losses.to(device=mask.device) * n_non_pad_per_example
    loss_per_token = loss_all_tokens / n_non_pad
    return loss_per_token.sum().item()


@lru_cache(maxsize=1000)
def tokenize_single(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    single_token_str: str,
) -> int:
    """Tokenize a string that results in a single token.

    Args:
        tokenizer: The tokenizer to use.
        single_token_str: The string to tokenize.

    Returns:
        The token ID
    """
    tokenized = tokenizer.encode(single_token_str, add_special_tokens=False)
    if len(tokenized) != 1:
        raise ValueError(
            f"Expected a single token for {single_token_str},"
            f" got {len(tokenized)} tokens: {tokenized}."
        )
    return tokenized[0]
