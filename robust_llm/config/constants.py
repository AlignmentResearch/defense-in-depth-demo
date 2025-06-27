from enum import Enum

MODEL_FAMILIES = [
    "gpt2",
    "llama2",
    "llama2-vllm",
    "llama2-chat",
    "llama2-chat-vllm",
    "llama3",
    "llama3-vllm",
    "llama3-chat",
    "llama3-chat-vllm",
    "vicuna",
    "pythia",
    "pythia-vllm",
    "qwen1.5",
    "qwen1.5-vllm",
    "qwen1.5-chat",
    "qwen1.5-chat-vllm",
    "qwen2",
    "qwen2-vllm",
    "qwen2-chat",
    "qwen2-chat-vllm",
    "qwen2.5",
    "qwen2.5-vllm",
    "qwen2.5-chat",
    "qwen2.5-chat-vllm",
    "qwen3-chat",
    "qwen3-chat-vllm",
    "tinyllama",
    "mistralai",
    "mistralai-chat",
    "gpt_neox",
    "gpt_neox-vllm",
    "pythia-chat",
    "gemma",
    "gemma-chat",
    "gemma-chat-vllm",
]


class ProbeType(Enum):
    LINEAR = "linear"
    MLP = "mlp"
    ATTENTION = "attention"
    TRANSFORMER = "transformer"
