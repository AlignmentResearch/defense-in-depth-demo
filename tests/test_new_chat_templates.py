import textwrap
from unittest.mock import patch

import pytest
from accelerate import Accelerator

from robust_llm.config import ModelConfig
from robust_llm.config.model_configs import GenerationConfig
from robust_llm.message_utils import Message, MessageFilter, MessageList
from robust_llm.models.mock_models import mock_load_hf_model
from robust_llm.models.wrapped_model import WrappedModel


@pytest.fixture
def gemma_2_model():
    config = ModelConfig(
        name_or_path="google/gemma-2-9b-it",
        family="gemma-chat",
        inference_type="generation",
        effective_batch_size=1,
        train_minibatch_multiplier=1,
        eval_minibatch_multiplier=1,
        env_minibatch_multiplier=1,
        max_minibatch_size=1,
        strict_load=True,
        generation_config=GenerationConfig(),
        tokenizer_name="google/gemma-2-9b-it",
    )
    accelerator = Accelerator()
    with patch("robust_llm.models.wrapped_model.load_hf_model", mock_load_hf_model):
        return WrappedModel.from_config(config, accelerator=accelerator)


@pytest.fixture
def gemma_1p1_model():
    config = ModelConfig(
        name_or_path="google/gemma-1.1-2b-it",
        family="gemma-chat",
        inference_type="generation",
        effective_batch_size=1,
        train_minibatch_multiplier=1,
        eval_minibatch_multiplier=1,
        env_minibatch_multiplier=1,
        max_minibatch_size=1,
        strict_load=True,
        generation_config=GenerationConfig(),
        tokenizer_name="google/gemma-1.1-2b-it",
    )
    accelerator = Accelerator()
    with patch("robust_llm.models.wrapped_model.load_hf_model", mock_load_hf_model):
        return WrappedModel.from_config(config, accelerator=accelerator)


@pytest.fixture
def qwen_2p5_model():
    config = ModelConfig(
        name_or_path="Qwen/Qwen2.5-7B-Instruct",
        family="qwen2.5-chat",
        inference_type="generation",
        effective_batch_size=1,
        train_minibatch_multiplier=1,
        eval_minibatch_multiplier=1,
        env_minibatch_multiplier=1,
        max_minibatch_size=1,
        strict_load=True,
        generation_config=GenerationConfig(),
        tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
    )
    accelerator = Accelerator()
    with patch("robust_llm.models.wrapped_model.load_hf_model", mock_load_hf_model):
        return WrappedModel.from_config(config, accelerator=accelerator)


@pytest.fixture
def pythia_model():
    config = ModelConfig(
        name_or_path="EleutherAI/pythia-14m",
        family="pythia",
        inference_type="generation",
        effective_batch_size=1,
        train_minibatch_multiplier=1,
        eval_minibatch_multiplier=1,
        env_minibatch_multiplier=1,
        max_minibatch_size=1,
        strict_load=True,
        generation_config=GenerationConfig(),
        tokenizer_name="EleutherAI/pythia-14m",
    )
    accelerator = Accelerator()
    return WrappedModel.from_config(config, accelerator=accelerator)


@pytest.fixture
def messages():
    messages = MessageList([Message(role="user", content="[This is a user message!]")])
    return messages


def test_message_list_to_text_gemma2(
    gemma_2_model: WrappedModel, messages: MessageList
):
    prompt = gemma_2_model.message_list_to_text(
        messages, message_filter=MessageFilter.INPUT, add_generation_prompt=False
    )
    assert prompt == textwrap.dedent(
        """\
        <bos><start_of_turn>user
        [This is a user message!]<end_of_turn>
        """
    )

    gen_prompt = gemma_2_model.message_list_to_text(
        messages, message_filter=MessageFilter.INPUT, add_generation_prompt=True
    )
    assert gen_prompt == textwrap.dedent(
        """\
        <bos><start_of_turn>user
        [This is a user message!]<end_of_turn>
        <start_of_turn>model
        """
    )


def test_message_list_to_text_qwen2p5(
    qwen_2p5_model: WrappedModel, messages: MessageList
):
    prompt = qwen_2p5_model.message_list_to_text(
        messages, message_filter=MessageFilter.INPUT, add_generation_prompt=False
    )
    assert prompt == textwrap.dedent(
        """\
        <|im_start|>system
        You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
        <|im_start|>user
        [This is a user message!]<|im_end|>
        """
    )

    gen_prompt = qwen_2p5_model.message_list_to_text(
        messages, message_filter=MessageFilter.INPUT, add_generation_prompt=True
    )
    assert gen_prompt == textwrap.dedent(
        """\
        <|im_start|>system
        You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
        <|im_start|>user
        [This is a user message!]<|im_end|>
        <|im_start|>assistant
        """
    )


def test_message_list_to_text_pythia(pythia_model: WrappedModel, messages: MessageList):
    prompt = pythia_model.message_list_to_text(
        messages, message_filter=MessageFilter.INPUT, add_generation_prompt=False
    )
    assert prompt == "[This is a user message!]"

    gen_prompt = pythia_model.message_list_to_text(
        messages, message_filter=MessageFilter.INPUT, add_generation_prompt=True
    )
    assert gen_prompt == "[This is a user message!]"
