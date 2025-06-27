import logging
import warnings
from dataclasses import dataclass
from typing import Any

import torch
from robust_llm.file_utils import get_save_root


def loudly_update(kwargs: dict[str, Any], key: str, value: Any):
    if key in kwargs and kwargs[key] != value:
        warnings.warn(f"Overriding {key} from {kwargs[key]} to {value}")
    kwargs[key] = value


@dataclass
class EnvironmentConfig:
    """
    Configs used in environment setup.

    Attributes:
        device: Device to use for models.
        test_mode: Whether or not we're currently testing
        save_root: Prefix to use for the local artifacts directory.
        minibatch_multiplier: Multiplier for the minibatch size.
            This is useful if we want to set default batch sizes for models in the
            ModelConfig but then adjust all of these values based on the GPU memory
            available or the dataset we're attacking.
        logging_level:
            Logging level to use for console handler.
            Choose among logging.DEBUG, logging.INFO,
            logging.WARNING, logging.ERROR, logging.CRITICAL.
        logging_filename: If set, logs will be saved to this file.
        wandb_info_filename: Log the W&B run name + ID to this file. Use this if
          you need to programatically get the run name and ID after running
          a job.
        allow_checkpointing: Whether to allow checkpointing during training and also
            attacks that support it.
        resume_mode: How often to resume from checkpoint during training.
            - "once": Resume from checkpoint and then run as normal
            - "always": Resume from checkpoint at the beginning of each epoch.
                The "always" mode is useful for debugging and ensuring determinism.
        deterministic: Whether to force use of deterministic CUDA algorithms at the
            cost of performance.
            N.B. When doing generation with sampling rather than greedy decoding,
            determinism leads to a huge performance hit because we generate one input
            at a time, rather than batched.
        cublas_config: The configuration string for cuBLAS, only used if
            deterministic is True.
            See https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
            for why this is necessary.

    """

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    test_mode: bool = False
    save_root: str = get_save_root()
    minibatch_multiplier: float = 1.0
    logging_level: int = logging.INFO
    logging_filename: str = "robust_llm.log"
    wandb_info_filename: str | None = None
    allow_checkpointing: bool = True
    resume_mode: str = "once"
    deterministic: bool = True
    cublas_config: str = ":4096:8"
