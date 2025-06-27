from __future__ import annotations

from itertools import islice
from typing import Any, Iterable, Iterator, Optional, TypeVar

import torch
import vllm
from accelerate import Accelerator
from transformers.generation.configuration_utils import (
    GenerationConfig as TransformersGenerationConfig,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing_extensions import override

from robust_llm.config.dataset_configs import MessageFilter
from robust_llm.config.model_configs import GenerationConfig, ModelConfig
from robust_llm.message_utils import Message, MessageList
from robust_llm.models.model_utils import (
    InferenceType,
    compute_batch_sizes_from_config,
    load_vllm_model,
)
from robust_llm.models.supported_models.gemma_chat_wrapped import GemmaChatModel
from robust_llm.models.supported_models.gemma_wrapped import GemmaModel
from robust_llm.models.supported_models.gpt_neox_wrapped import GPTNeoXModel
from robust_llm.models.supported_models.llama2_chat_wrapped import Llama2ChatModel
from robust_llm.models.supported_models.llama3_chat_wrapped import Llama3ChatModel
from robust_llm.models.supported_models.llama3_wrapped import Llama3Model
from robust_llm.models.supported_models.qwen_chat_wrapped import QwenChatModel
from robust_llm.models.supported_models.qwen_wrapped import QwenModel
from robust_llm.models.wrapped_model import WrappedModel

T = TypeVar("T")


class VLLMModelAdapter(WrappedModel):
    """Adapter for vLLM models that provides vLLM-specific implementations.

    This class is intended to be used as a mixin with model classes derived from
    WrappedModel. When used with multiple inheritance, this class should come
    BEFORE the WrappedModel-derived class to ensure its methods take precedence.
    """

    def __init__(
        self,
        model: vllm.LLM,
        right_tokenizer: PreTrainedTokenizerBase,
        accelerator: Accelerator | None,
        inference_type: InferenceType,
        train_minibatch_size: int,
        eval_minibatch_size: int,
        family: str,
        effective_batch_size: int = 8,
        generation_config: GenerationConfig | None = None,
        system_prompt: str | None = None,
        seed: int = 0,
        n_logprobs: int = 128,
    ) -> None:
        """Initialize a vLLM model adapter.

        This replaces the standard WrappedModel initialization for vLLM models.

        Args:
            model: The vLLM model to wrap.
            right_tokenizer: The tokenizer to use.
            accelerator: The accelerator to use.
            inference_type: The type of inference this model is for (must be
                'generation' for vLLM models)
            train_minibatch_size: The minibatch size to use for training.
            eval_minibatch_size: The minibatch size to use for evaluation.
            effective_batch_size: The product of the train batch size and gradient
                accumulation steps.
            family: The family of the model.
            generation_config: The generation config to use for generation.
            system_prompt: The system prompt to use for chat models.
            seed: The seed to use for text generation.
            n_logprobs: The number of logprobs to return for each prompt token.
        """
        # We don't call super().__init__() here because this is a mixin
        # The actual class hierarchy will be determined when this is mixed with
        # a concrete model
        self._n_params = 0  # Number of parameters for vLLM model
        self.model = model  # type: ignore
        self.family = family
        self.accelerator = accelerator

        self.right_tokenizer = right_tokenizer
        self.inference_type = inference_type
        self.train_minibatch_size = train_minibatch_size
        self.eval_minibatch_size = eval_minibatch_size
        self.effective_batch_size = effective_batch_size
        self.generation_config = generation_config
        self.system_prompt = system_prompt
        self.seed = seed
        self.n_logprobs = n_logprobs

        # Initialize FLOP tracking variables but don't use them
        self._flop_count = 0
        self._n_forward_calls = 0
        self._n_backward_calls = 0
        self._input_shapes: list[tuple] = []
        self._skip_hooks = True  # Disable FLOP counting for vLLM models

    def forward(self, **inputs: Any) -> Any:
        """Forward pass is not supported for vLLM models."""
        raise NotImplementedError("vLLM models do not support forward()")

    def __call__(self, **inputs: Any) -> Any:
        """Call operator is not supported for vLLM models."""
        raise NotImplementedError("vLLM models do not support __call__()")

    @override
    def generation_logits_from_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_no_grad: bool = True,
        minibatch_size: int | None = None,
    ) -> Iterator[torch.Tensor]:
        """Compute the log probabilities of the tokens in the generation."""
        raise NotImplementedError(
            "vLLM does not support generation logprobs from tokens"
        )

    @override
    def autoregressive_generation_from_messages(
        self,
        message_lists: list[MessageList],
        message_filter: MessageFilter = MessageFilter.IDENTITY,
        minibatch_size: int | None = None,
    ) -> Iterator[list[MessageList]]:
        """Returns the autoregressive generation from message lists using vLLM.

        Args:
            message_lists: The message lists to generate from.
            message_filter: The message filter to use.
            minibatch_size: The minibatch size to use.

        Yields:
            A list of message lists with the assistant's responses added.
        """
        assert self.inference_type == InferenceType.GENERATION

        # vLLM handles padding internally, so we don't need to add it
        tokenized = self.message_lists_to_tokens(
            message_lists,
            message_filter,
            padding_side=None,
            return_tensors=None,
        )
        input_ids = tokenized.input_ids

        prompts = [
            vllm.TokensPrompt(prompt_token_ids=prompt_input_ids)
            for prompt_input_ids in input_ids
        ]
        assert isinstance(self.model, vllm.LLM)
        outs = self.model.generate(
            prompts,
            sampling_params=self.vllm_sampling_params(),
            use_tqdm=False,
        )
        out_tokens = [list(out.outputs[0].token_ids) for out in outs]
        out_texts = self.batch_decode(out_tokens, skip_special_tokens=True)
        new_message_lists = [
            message_list + Message(role="assistant", content=out_text)
            for message_list, out_text in zip(message_lists, out_texts, strict=True)
        ]
        yield new_message_lists

    def vllm_sampling_params(
        self,
        generation_config: (
            GenerationConfig | TransformersGenerationConfig | None
        ) = None,
    ) -> vllm.SamplingParams:
        """The sampling parameters to use for vLLM models."""
        if self.generation_config is None and generation_config is None:
            raise ValueError("Generation config is required for vLLM models")
        else:
            generation_config = generation_config or self.generation_config
        assert generation_config is not None

        temperature = generation_config.temperature
        if generation_config.do_sample is False:
            temperature = 0.0

        return vllm.SamplingParams(
            max_tokens=generation_config.max_new_tokens,
            temperature=temperature,
            top_k=generation_config.top_k,
            top_p=generation_config.top_p,
            stop=generation_config.stop_strings,
        )

    def eval(self):
        """Sets the model to evaluation mode (no-op for vLLM)."""
        # This is a no-op for vLLM models
        return self

    def train(self):
        """Training is not supported for vLLM models."""
        raise NotImplementedError("vLLM models do not support training")

    @property
    def device(self) -> torch.device:
        """Returns the device of the model (always CUDA for vLLM)."""
        # vLLM only supports CUDA
        return torch.device("cuda")

    @override
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_config: TransformersGenerationConfig | None = None,
    ) -> torch.Tensor:
        """Generate text using vLLM's generate method.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask (not used by vLLM).
            generation_config: Optional generation configuration.

        Returns:
            Generated token IDs
        """
        # Convert to list format for vLLM
        if isinstance(input_ids, torch.Tensor):
            input_ids_list = input_ids.cpu().tolist()
        else:
            input_ids_list = input_ids

        prompts = [
            vllm.TokensPrompt(prompt_token_ids=input_ids)
            for input_ids in input_ids_list
        ]
        assert isinstance(self.model, vllm.LLM)
        outs = self.model.generate(
            prompts,
            sampling_params=self.vllm_sampling_params(generation_config),
        )

        # Extract output tokens including input
        full_token_ids = []
        for i, out in enumerate(outs):
            # Concatenate input and output tokens
            tokens = input_ids_list[i] + list(out.outputs[0].token_ids)
            full_token_ids.append(tokens)

        # Convert back to tensor and ensure same format as original function
        max_length = max(len(tokens) for tokens in full_token_ids)

        pad_token = self.right_tokenizer.pad_token_id
        assert isinstance(pad_token, int)
        result = torch.full(
            (len(full_token_ids), max_length),
            pad_token,
            dtype=torch.long,
        )
        for i, tokens in enumerate(full_token_ids):
            result[i, : len(tokens)] = torch.tensor(tokens)

        return result

    @classmethod
    @override
    def _from_config(
        cls,
        config: ModelConfig,
        accelerator: Accelerator | None,
        num_classes: Optional[int] = None,
        overwrite_cache: bool = False,
        **kwargs,
    ) -> WrappedModel:
        """Creates a vLLM model from a ModelConfig.

        This method should be called from the concrete model class's _from_config method
        when creating vLLM models.

        Returns:
            A vLLM-adapted model instance
        """
        inference_type = InferenceType(config.inference_type)
        if inference_type != InferenceType.GENERATION:
            raise ValueError("vLLM models only support generation inference type")

        model = load_vllm_model(
            name_or_path=config.name_or_path,
            tokenizer_name=config.tokenizer_name,
            revision=config.revision,
            inference_type=inference_type,
            strict_load=config.strict_load,
            dtype=config.dtype,
            attention_implementation=config.attention_implementation,
            num_classes=num_classes,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enforce_eager=config.enforce_eager,
            enable_chunked_prefill=config.enable_chunked_prefill,
        )

        # Load the tokenizer with right padding
        right_tokenizer = cls.load_tokenizer(config)
        train_mb_size, eval_mb_size = compute_batch_sizes_from_config(config)

        return cls(
            model=model,
            right_tokenizer=right_tokenizer,
            accelerator=accelerator,
            inference_type=inference_type,
            train_minibatch_size=train_mb_size,
            eval_minibatch_size=eval_mb_size,
            effective_batch_size=config.effective_batch_size,
            generation_config=config.generation_config,
            family=config.family,
            system_prompt=config.system_prompt,
            seed=config.seed,
        )


@WrappedModel.register_subclass("qwen1.5-vllm")
@WrappedModel.register_subclass("qwen2-vllm")
@WrappedModel.register_subclass("qwen2.5-vllm")
class VLLMQwenModel(VLLMModelAdapter, QwenModel):
    """Qwen model using vLLM for inference.

    This class extends QwenModel and adds vLLM support using the VLLMModelAdapter.
    """


@WrappedModel.register_subclass("qwen1.5-chat-vllm")
@WrappedModel.register_subclass("qwen2-chat-vllm")
@WrappedModel.register_subclass("qwen2.5-chat-vllm")
@WrappedModel.register_subclass("qwen3-chat-vllm")
class VLLMQwenChatModel(VLLMModelAdapter, QwenChatModel):
    """Qwen chat model using vLLM for inference.

    This class extends QwenChatModel and adds vLLM support using the VLLMModelAdapter.
    """


@WrappedModel.register_subclass("llama3-vllm")
class VLLMLlama3Model(VLLMModelAdapter, Llama3Model):
    """Llama3 model using vLLM for inference.

    This class extends Llama3Model and adds vLLM support using the VLLMModelAdapter.
    """


@WrappedModel.register_subclass("llama3-chat-vllm")
class VLLMLlama3ChatModel(VLLMModelAdapter, Llama3ChatModel):
    """Llama3 chat model using vLLM for inference.

    This class extends Llama3ChatModel and adds vLLM support using the VLLMModelAdapter.
    """


@WrappedModel.register_subclass("llama2-chat-vllm")
class VLLMLlama2ChatModel(VLLMModelAdapter, Llama2ChatModel):
    """Llama2 chat model using vLLM for inference.

    This class extends Llama2ChatModel and adds vLLM support using the VLLMModelAdapter.
    """


@WrappedModel.register_subclass("gpt_neox-vllm")
@WrappedModel.register_subclass("pythia-vllm")
class VLLMGPTNeoXModel(VLLMModelAdapter, GPTNeoXModel):
    """GPTNeoX model using vLLM for inference.

    This class extends GPTNeoXModel and adds vLLM support using the VLLMModelAdapter.
    (Pythia models are GPTNeoX models.)
    """


@WrappedModel.register_subclass("gemma-vllm")
class VLLMGemmaModel(VLLMModelAdapter, GemmaModel):
    """Gemma model using vLLM for inference.

    This class extends GemmaModel and adds vLLM support using the VLLMModelAdapter.
    """


@WrappedModel.register_subclass("gemma-chat-vllm")
class VLLMGemmaChatModel(VLLMModelAdapter, GemmaChatModel):
    """Gemma chat model using vLLM for inference.

    This class extends GemmaChatModel and adds vLLM support using the VLLMModelAdapter.
    """


def batched(iterable: Iterable[T], n: int) -> Iterator[tuple[T, ...]]:
    """Batch data into lists of length n.

    The last batch may be shorter.
    """
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = tuple(islice(it, n))
        if not batch:
            return
        yield batch
