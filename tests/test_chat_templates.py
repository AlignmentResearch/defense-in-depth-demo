from unittest.mock import MagicMock

import pytest
from transformers.models.auto.tokenization_auto import AutoTokenizer

from robust_llm.dist_utils import DistributedRNG
from robust_llm.models.model_utils import InferenceType
from robust_llm.models.prompt_templates import AttackChunks
from robust_llm.models.wrapped_chat_model import WrappedChatModel

NAME_AND_TEMPLATE = [
    ("meta-llama/Llama-2-7b-chat-hf", "llama2-chat"),
    ("meta-llama/Meta-Llama-3-8B-Instruct", "llama3-chat"),
    ("Qwen/Qwen1.5-1.8B-Chat", "qwen1.5-chat"),
    ("Qwen/Qwen2-7B-Instruct", "qwen2-chat"),
    ("Qwen/Qwen2.5-7B-Instruct", "qwen2.5-chat"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tinyllama"),
    ("Felladrin/Pythia-31M-Chat-v1", "pythia-chat"),
    pytest.param(
        "lmsys/vicuna-7b-v1.5",
        "vicuna",
        marks=pytest.mark.xfail(
            reason=(
                "Vicuna does not supply a chat template"
                " which is required since transformers==4.4.0"
                " (see GH#1104)"
            )
        ),
    ),
]
# Gemma models do not support system prompts.
NAME_AND_TEMPLATE_NO_SYSTEM_PROMPT = NAME_AND_TEMPLATE + [
    ("google/gemma-1.1-2b-it", "gemma-chat"),
    ("google/gemma-2-9b-it", "gemma-chat"),
]


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.modules.return_value = [MagicMock()]
    model.num_parameters.return_value = 0
    return model


@pytest.mark.parametrize("model_name, model_family", NAME_AND_TEMPLATE)
def test_wrap_prompt_template(mock_model, model_name: str, model_family: str):
    chunks = AttackChunks(
        unmodifiable_prefix="Unmodifiable prefix. ",
        modifiable_infix="Modifiable infix. ",
        unmodifiable_suffix="Unmodifiable suffix.",
    )
    model_constructor = WrappedChatModel._registry[model_family]
    model = model_constructor(
        model=mock_model,
        right_tokenizer=MagicMock(),
        accelerator=None,
        inference_type=InferenceType.GENERATION,
        train_minibatch_size=2,
        eval_minibatch_size=3,
        generation_config=None,
        family=model_family,
        system_prompt="System prompt.",
    )
    assert isinstance(model, WrappedChatModel)
    conv = model.init_conversation()
    conv.append_user_message(
        "Unmodifiable prefix. Modifiable infix. Unmodifiable suffix."
    )
    conv.append_assistant_message("")
    base_template = chunks.get_prompt_template(
        perturb_min=1.0, perturb_max=1.0, rng=DistributedRNG(0, None)
    )
    prompt_template = conv.wrap_prompt_template(base_template)
    assert prompt_template.before_attack.endswith(
        "Unmodifiable prefix. Modifiable infix. "
    )
    assert prompt_template.after_attack.startswith(
        "!ATTACK_STRING_PLACEHOLDER!Unmodifiable suffix."
    )
    prompt_template_prompt = prompt_template.build_prompt()
    conv_prompt = conv.get_prompt()
    assert (
        prompt_template_prompt == conv_prompt
    ), f"{prompt_template_prompt}\n!=\n{conv_prompt}"


@pytest.mark.parametrize("model_name, model_family", NAME_AND_TEMPLATE)
def test_build_prompt(mock_model, model_name: str, model_family: str):
    chunks = AttackChunks(
        unmodifiable_prefix="Unmodifiable prefix. ",
        modifiable_infix="Modifiable infix. ",
        unmodifiable_suffix="Unmodifiable suffix.",
    )
    model_constructor = WrappedChatModel._registry[model_family]
    model = model_constructor(
        model=mock_model,
        right_tokenizer=MagicMock(),
        accelerator=None,
        inference_type=InferenceType.GENERATION,
        train_minibatch_size=2,
        eval_minibatch_size=3,
        generation_config=None,
        family=model_family,
        system_prompt="System prompt.",
    )
    assert isinstance(model, WrappedChatModel)
    conv = model.init_conversation()
    base_template = chunks.get_prompt_template(
        perturb_min=1.0, perturb_max=1.0, rng=DistributedRNG(0, None)
    )
    prompt_template = conv.wrap_prompt_template(base_template)
    conv.append_user_message(
        "Unmodifiable prefix. Modifiable infix. Attack text. Unmodifiable suffix."
    )
    conv.append_assistant_message(" Target.")
    built = prompt_template.build_prompt(attack_text="Attack text. ", target=" Target.")
    assert built == conv.get_prompt()


@pytest.mark.parametrize("model_name, model_family", NAME_AND_TEMPLATE)
def test_repeated_role_raises_error(mock_model, model_name: str, model_family: str):
    model_constructor = WrappedChatModel._registry[model_family]
    model = model_constructor(
        model=mock_model,
        right_tokenizer=MagicMock(),
        accelerator=None,
        inference_type=InferenceType.GENERATION,
        train_minibatch_size=2,
        eval_minibatch_size=3,
        generation_config=None,
        family=model_family,
        system_prompt="System prompt.",
    )
    assert isinstance(model, WrappedChatModel)
    conv = model.init_conversation()
    with pytest.raises(AssertionError):
        conv.append_assistant_message("Assistant prompt 1.")
    conv.append_user_message("User prompt 1.")
    with pytest.raises(AssertionError):
        conv.append_user_message("User prompt 2.")
    conv.append_assistant_message("Assistant prompt 1.")
    with pytest.raises(AssertionError):
        conv.append_assistant_message("Assistant prompt 2.")
    conv.append_user_message("User prompt 2.")


@pytest.mark.parametrize("model_name, model_family", NAME_AND_TEMPLATE)
def test_multi_round_chat_template(mock_model, model_name: str, model_family: str):
    model_constructor = WrappedChatModel._registry[model_family]
    model = model_constructor(
        model=mock_model,
        right_tokenizer=MagicMock(),
        accelerator=None,
        inference_type=InferenceType.GENERATION,
        train_minibatch_size=2,
        eval_minibatch_size=3,
        generation_config=None,
        family=model_family,
        system_prompt="System prompt.",
    )
    assert isinstance(model, WrappedChatModel)
    conv = model.init_conversation()
    conv.append_user_message("User prompt 1.")
    conv.append_assistant_message("Assistant prompt 1.")
    conv.append_user_message("User prompt 2.")
    conv.append_assistant_message("Assistant prompt 2.")
    prompt = conv.get_prompt(skip_last_suffix=False)
    assert "User prompt 2." in prompt
    assert "System prompt." in prompt
    assert "Assistant prompt 2." in prompt
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_out = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "System prompt."},
            {
                "role": "user",
                "content": ("User prompt 1."),
            },
            {"role": "assistant", "content": "Assistant prompt 1."},
            {
                "role": "user",
                "content": ("User prompt 2."),
            },
            {"role": "assistant", "content": "Assistant prompt 2."},
        ],
        tokenize=False,
    )
    assert tokenizer_out == prompt


@pytest.mark.parametrize("model_name, model_family", NAME_AND_TEMPLATE)
def test_full_chat_template(mock_model, model_name: str, model_family: str):
    model_constructor = WrappedChatModel._registry[model_family]
    model = model_constructor(
        model=mock_model,
        right_tokenizer=MagicMock(),
        accelerator=None,
        inference_type=InferenceType.GENERATION,
        train_minibatch_size=2,
        eval_minibatch_size=3,
        generation_config=None,
        family=model_family,
        system_prompt="System prompt.",
    )
    assert isinstance(model, WrappedChatModel)
    conv = model.init_conversation()
    conv.append_user_message("User prompt.")
    conv.append_assistant_message("Assistant prompt.")
    prompt = conv.get_prompt(skip_last_suffix=False)
    assert "User prompt." in prompt
    assert "System prompt." in prompt
    assert "Assistant prompt." in prompt
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_out = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "System prompt."},
            {
                "role": "user",
                "content": ("User prompt."),
            },
            {"role": "assistant", "content": "Assistant prompt."},
        ],
        tokenize=False,
    )
    assert tokenizer_out == prompt


@pytest.mark.parametrize("model_name, model_family", NAME_AND_TEMPLATE)
def test_user_template_with_system_prompt(
    mock_model, model_name: str, model_family: str
):
    model_constructor = WrappedChatModel._registry[model_family]
    model = model_constructor(
        model=mock_model,
        right_tokenizer=MagicMock(),
        accelerator=None,
        inference_type=InferenceType.GENERATION,
        train_minibatch_size=2,
        eval_minibatch_size=3,
        generation_config=None,
        family=model_family,
        system_prompt="System prompt.",
    )
    assert isinstance(model, WrappedChatModel)
    prompt = model.maybe_apply_user_template(
        "This is the user prompt.",
    )
    assert "This is the user prompt." in prompt
    assert "System prompt." in prompt
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_out = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "System prompt."},
            {
                "role": "user",
                "content": ("This is the user prompt."),
            },
            {"role": "assistant", "content": "Assistant prompt."},
        ],
        tokenize=False,
    )
    assert tokenizer_out[: tokenizer_out.find("Assistant prompt")].rstrip(" ") == prompt


@pytest.mark.parametrize("model_name, model_family", NAME_AND_TEMPLATE_NO_SYSTEM_PROMPT)
def test_user_template_without_system_prompt(
    mock_model, model_name: str, model_family: str
):
    model_constructor = WrappedChatModel._registry[model_family]
    model = model_constructor(
        model=mock_model,
        right_tokenizer=MagicMock(),
        accelerator=None,
        inference_type=InferenceType.GENERATION,
        train_minibatch_size=2,
        eval_minibatch_size=3,
        generation_config=None,
        family=model_family,
    )
    assert isinstance(model, WrappedChatModel)
    prompt = model.maybe_apply_user_template(
        "User prompt.",
    )
    assert "User prompt." in prompt
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_out = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": ("User prompt."),
            },
            {"role": "assistant", "content": "Assistant prompt"},
        ],
        tokenize=False,
    )
    assert tokenizer_out[: tokenizer_out.find("Assistant prompt")].rstrip(" ") == prompt
