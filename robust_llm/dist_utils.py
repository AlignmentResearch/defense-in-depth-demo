"""Utility functions for working with torch.distributed/accelerate."""

import functools
import shutil
import time
import traceback
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from robust_llm import logger

INT_TO_DTYPE = {
    0: torch.float32,
    1: torch.float64,
    2: torch.float16,
    3: torch.int8,
    4: torch.uint8,
    5: torch.int16,
    6: torch.int32,
    7: torch.int64,
    8: torch.bool,
}
DTYPE_TO_INT = {v: k for k, v in INT_TO_DTYPE.items()}
BIT_GENERATOR = "PCG64"


# We can cache this because it will not vary once called on a given process.
@functools.lru_cache(maxsize=1)
def is_main_process() -> bool:
    """Find out if we are the main process without passing in an Accelerator object."""
    accelerator = Accelerator()
    return accelerator.is_main_process


# We can cache this because it will not vary once called on a given process.
@functools.lru_cache(maxsize=1)
def process_index() -> int:
    """Find out the process index without passing in an Accelerator object."""
    accelerator = Accelerator()
    return accelerator.process_index


def broadcast_list_of_bools(
    data: list[bool] | None, accelerator: Accelerator
) -> list[bool]:
    # If we're not using distributed training, we don't need to do anything.
    if not dist.is_initialized():
        assert isinstance(data, list)
        return data
    if is_main_process():
        assert data is not None
        assert all(isinstance(x, bool) for x in data)
        tensor_input_data = torch.tensor(
            data, dtype=torch.bool, device=accelerator.device
        )
    else:
        tensor_input_data = None
    tensor_data = broadcast_tensor(tensor_input_data, accelerator)
    return tensor_data.tolist()


def broadcast_list_of_floats(
    data: list[float] | None, accelerator: Accelerator
) -> list[float]:
    """Broadcasts floats, rounding to 32-bit precision."""
    # If we're not using distributed training, we don't need to do anything.
    if not dist.is_initialized():
        assert isinstance(data, list)
        return data
    if is_main_process():
        assert data is not None
        assert all(isinstance(x, float) for x in data)
        tensor_input_data = torch.tensor(
            data, dtype=torch.float32, device=accelerator.device
        )
    else:
        tensor_input_data = None
    tensor_data = broadcast_tensor(tensor_input_data, accelerator)
    return tensor_data.tolist()


def broadcast_list_of_ints(
    data: list[int] | None,
    accelerator: Accelerator,
) -> list[int]:
    """Broadcasts ints, asserting 32-bit precision."""
    # If we're not using distributed training, we don't need to do anything.
    if not dist.is_initialized():
        assert isinstance(data, list)
        return data
    if is_main_process():
        assert data is not None
        assert all(isinstance(x, int) and x.bit_length() <= 32 for x in data)
        tensor_input_data = torch.tensor(
            data, dtype=torch.int32, device=accelerator.device
        )
    else:
        tensor_input_data = None
    tensor_data = broadcast_tensor(tensor_input_data, accelerator)
    return tensor_data.tolist()


def broadcast_list_of_longs(
    data: list[int] | None,
    accelerator: Accelerator,
) -> list[int]:
    """Broadcasts ints, asserting 64-bit precision."""
    # If we're not using distributed training, we don't need to do anything.
    if not dist.is_initialized():
        assert isinstance(data, list)
        return data
    if is_main_process():
        assert data is not None
        assert all(isinstance(x, int) and x.bit_length() <= 64 for x in data)
        tensor_input_data = torch.tensor(
            data, dtype=torch.int64, device=accelerator.device
        )
    else:
        tensor_input_data = None
    tensor_data = broadcast_tensor(tensor_input_data, accelerator)
    return tensor_data.tolist()


def broadcast_int(
    data: int | None,
    accelerator: Accelerator,
) -> int:
    """Broadcasts int, rounding to 32-bit precision."""
    return broadcast_list_of_ints([data] if data is not None else None, accelerator)[0]


def broadcast_long(
    data: int | None,
    accelerator: Accelerator,
) -> int:
    """Broadcasts int, rounding to 64-bit precision."""
    return broadcast_list_of_longs([data] if data is not None else None, accelerator)[0]


def split_int128(n: int) -> list[int]:
    """Split a 128-bit integer into a list of four int32 values."""
    mask = (1 << 32) - 1  # Mask for 32 bits
    return [n & mask, (n >> 32) & mask, (n >> 64) & mask, (n >> 96) & mask]


def reconstruct_int128(int_list: list[int]) -> int:
    """Reconstruct a 128-bit integer from a list of four int32 values."""
    assert len(int_list) == 4
    return int_list[0] | (int_list[1] << 32) | (int_list[2] << 64) | (int_list[3] << 96)


def broadcast_int128(
    data: int | None,
    accelerator: Accelerator,
) -> int:
    """Broadcasts a 128-bit integer by splitting into list[int32]."""
    if not dist.is_initialized():
        assert isinstance(data, int)
        return data

    if is_main_process():
        assert data is not None
        assert isinstance(data, int) and data.bit_length() <= 128
        split_data = split_int128(data)
    else:
        split_data = None

    broadcasted_data = broadcast_list_of_longs(split_data, accelerator)
    return reconstruct_int128(broadcasted_data)


def broadcast_float(
    data: float | None,
    accelerator: Accelerator,
) -> float:
    """Broadcasts float, rounding to 32-bit precision."""
    return broadcast_list_of_floats([data] if data is not None else None, accelerator)[
        0
    ]


def broadcast_str(data: str | None, accelerator: Accelerator) -> str:
    if is_main_process():
        assert data is not None, "Must have str data on main process"
        data_ord = [ord(d) for d in data]
    else:
        data_ord = None
    broadcast_ord = broadcast_list_of_ints(data_ord, accelerator=accelerator)
    broadcast_str = "".join([chr(o) for o in broadcast_ord])
    return broadcast_str


def broadcast_list_of_str(
    data: list[str] | None, accelerator: Accelerator
) -> list[str]:
    """Broadcasts a list of strings from the main process."""
    # If we're not using distributed training, we don't need to do anything.
    if not dist.is_initialized():
        assert isinstance(data, list)
        return data

    # Broadcast number of strings
    num_strings: int | None
    if is_main_process():
        assert data is not None
        assert all(isinstance(s, str) for s in data)
        num_strings = len(data)
    else:
        num_strings = None
    num_strings = broadcast_int(num_strings, accelerator)
    assert isinstance(num_strings, int)  # Should be int on all processes now

    # Broadcast lengths of each string
    lengths: list[int] | None
    if is_main_process():
        assert data is not None
        lengths = [len(s) for s in data]
    else:
        lengths = None
    broadcasted_lengths = broadcast_list_of_ints(lengths, accelerator)

    # Broadcast the concatenated string
    concatenated_str: str | None
    if is_main_process():
        assert data is not None
        concatenated_str = "".join(data)
    else:
        concatenated_str = None
    broadcasted_concatenated_str = broadcast_str(concatenated_str, accelerator)

    # Reconstruct the list of strings
    result: list[str] = []
    start_index = 0
    for length in broadcasted_lengths:
        end_index = start_index + length
        result.append(broadcasted_concatenated_str[start_index:end_index])
        start_index = end_index

    # Sanity check
    assert len(result) == num_strings
    assert "".join(result) == broadcasted_concatenated_str

    return result


def broadcast_rng_state(
    data: dict[str, Any] | None,
    accelerator: Accelerator,
) -> dict[str, Any]:
    """Broadcasts the state of the RNG from the main process"""
    # If we're not using distributed training, we don't need to do anything.
    if not dist.is_initialized():
        assert isinstance(data, dict)
        return data
    return {
        "bit_generator": BIT_GENERATOR,
        "state": {
            "state": broadcast_int128(
                data["state"]["state"] if data is not None else None, accelerator
            ),
            "inc": broadcast_int128(
                data["state"]["inc"] if data is not None else None, accelerator
            ),
        },
        "has_uint32": broadcast_int(
            data["has_uint32"] if data is not None else None, accelerator
        ),
        "uinteger": broadcast_long(
            data["uinteger"] if data is not None else None, accelerator
        ),
    }


def broadcast_tensor(
    data: torch.Tensor | None, accelerator: Accelerator
) -> torch.Tensor:
    """Broadcast the data in 'data' to all processes in the group.

    We can't just broadcast the data because we need an empty tensor of the
    right shape on the other devices to broadcast into.
    """
    # If we're not using distributed training, we don't need to do anything.
    if not dist.is_initialized():
        assert isinstance(data, torch.Tensor)
        return data

    # First we need to broadcast the dimension of the tensor.
    if is_main_process():
        assert isinstance(data, torch.Tensor)
        ndims = torch.tensor(data.ndim, device=accelerator.device)
    else:
        ndims = torch.tensor(-1, device=accelerator.device)
    dist.broadcast(ndims, src=0)

    # Now we need to broadcast the actual shape.
    if is_main_process():
        assert isinstance(data, torch.Tensor)
        shape = torch.tensor(data.shape, device=accelerator.device, dtype=torch.int32)
    else:
        shape = torch.empty(
            int(ndims.item()), device=accelerator.device, dtype=torch.int32
        )
    dist.broadcast(shape, src=0)

    # Then we need to broadcast the datatype.
    data_dtype = broadcast_dtype(data, accelerator)

    # Finally, broadcast the data itself.
    if is_main_process():
        assert data is not None
        data = data.to(accelerator.device)
    else:
        data = torch.empty(*shape.tolist(), dtype=data_dtype, device=accelerator.device)
    dist.broadcast(data, src=0)
    assert isinstance(data, torch.Tensor)
    return data


def broadcast_dtype(data: torch.Tensor | None, accelerator: Accelerator):
    """Broadcast the datatype of a tensor between ranks"""
    if is_main_process():
        assert isinstance(data, torch.Tensor)
        dtype = torch.tensor(DTYPE_TO_INT[data.dtype], device=accelerator.device)
    else:
        dtype = torch.tensor(-1, device=accelerator.device)
    dist.broadcast(dtype, src=0)
    return INT_TO_DTYPE[int(dtype.item())]


class DistributedRNG:

    def __init__(self, seed: int | None, accelerator: Accelerator | None) -> None:
        self._rng = np.random.default_rng(seed) if is_main_process() else None
        self.accelerator = accelerator if accelerator is not None else Accelerator()
        self.skip_broadcast = self.accelerator.device == "cpu"
        assert (
            self._rng is None
            or self._rng.bit_generator.state["bit_generator"] == BIT_GENERATOR
        )

    def randint(self, a: int, b: int) -> int:
        i = (
            int(self._rng.integers(a, b, endpoint=True))
            if self._rng is not None
            else None
        )
        if self.skip_broadcast:
            assert isinstance(i, int)
            return i
        return broadcast_int(i, self.accelerator)

    def random(self) -> float:
        f = float(self._rng.random()) if self._rng is not None else None
        if self.skip_broadcast:
            assert isinstance(f, float)
            return f
        return broadcast_float(f, self.accelerator)

    def choice(
        self,
        seq: int | list[int],
        size: int,
        replace: bool = False,
        p: list[float] | np.ndarray | None = None,
    ) -> list[int]:
        if p is not None:
            probs = np.array(p)
            probs_sum = probs.sum()
            if not 0 < probs_sum <= 1.1:
                raise ValueError(
                    "Sum of probs must be between 0 and 1."
                    f" They actually sum to {probs_sum}."
                    " Did you pass logits?"
                )
            if not np.isclose(probs_sum, 1, atol=1e-3):
                warnings.warn(
                    "Probabilities should usually (approximately) sum to 1."
                    f" They actually sum to {probs_sum}."
                    " Did you pass logits, or are you using only topk logits?"
                )
            probs = probs / probs_sum
        else:
            probs = None
        array = (
            self._rng.choice(seq, size=size, replace=replace, p=probs).tolist()
            if self._rng is not None
            else None
        )
        if self.skip_broadcast:
            assert isinstance(array, list)
            return array
        return broadcast_list_of_ints(array, self.accelerator)

    def sample(self, seq: list[Any], size: int) -> list[Any]:
        indices = self.choice(len(seq), size=size, replace=False)
        return [seq[i] for i in indices]

    def getstate(self) -> dict[str, Any]:
        state = self._rng.bit_generator.state if self._rng is not None else None
        assert state is None or isinstance(state, dict)
        if self.skip_broadcast:
            assert isinstance(state, dict)
            return state
        return broadcast_rng_state(state, self.accelerator)

    def setstate(self, state: dict[str, Any] | None) -> None:
        if self._rng is not None and state is not None:
            self._rng.bit_generator.state = state


def pad_batch_across_processes(
    tokenizer: PreTrainedTokenizerBase, accelerator: Accelerator, batch: dict
) -> dict:
    """Pad a batch across processes.

    We have to do this because otherwise we can end up with different length sequences
    across processes, which will cause a hang.
    """
    pad_token_id = tokenizer.pad_token_id
    assert isinstance(pad_token_id, int)
    input_ids = accelerator.pad_across_processes(
        batch["input_ids"], pad_index=pad_token_id, dim=1
    )
    attention_mask = accelerator.pad_across_processes(
        batch["attention_mask"], pad_index=0, dim=1
    )
    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if "labels" in batch:
        out["labels"] = batch["labels"]
    return out


def assert_same_data_between_processes(
    accelerator: Accelerator, data: Sequence[Any]
) -> None:
    length = len(data)
    # We use 'gather_object' rather than 'gather_for_metrics' because we want to see
    # all the data gathered, especially repeats. (In theory 'gather_for_metrics'
    # should also work here, but we were having issues with flaky tests on CircleCI.)
    data_gathered = gather_object(data)
    for i in range(accelerator.num_processes):
        start = i * length
        end = (i + 1) * length
        assert data_gathered[start:end] == data, (
            f"Data from process {i} does not match original.\n"
            f"Original (len {length}): {data}\n"
            f"Process {i} ({start=}, {end=}): {data_gathered[start:end]}\n"
        )


def rmtree_if_exists(path: Path):
    if path.exists():
        shutil.rmtree(path)


def dist_rmtree(
    path: str | Path,
    retries: int = 5,
    cooldown_seconds: float = 5,
    accelerator: Accelerator | None = None,
):
    if isinstance(path, str):
        path = Path(path)
    return try_except_main_process_loop(
        retries=retries,
        cooldown_seconds=cooldown_seconds,
        accelerator=accelerator,
        func=rmtree_if_exists,
        path=path,
    )


def try_except_main_process_loop(
    retries: int,
    cooldown_seconds: float,
    func: Callable,
    accelerator: Accelerator | None = None,
    *args,
    **kwargs,
):
    """Run a main process only function with retries and error handling.

    N.B. `func` must not gather or wait for other processes!
    """
    if accelerator is None:
        accelerator = Accelerator()
    assert retries >= 0
    attempts = retries + 1
    logger.debug(f"Running {func.__name__} with retries and error handling.")
    device = accelerator.device
    is_main_process = accelerator.is_main_process
    last_error = None
    for attempt in range(attempts):
        logger.debug(f"{func.__name__} attempt {attempt + 1}/{attempts}")
        error_occurred = torch.tensor(0, device=device)
        if dist.is_initialized():
            dist.barrier()
        try:
            if is_main_process:
                func(*args, **kwargs)
        except Exception as e:
            last_error = e
            warnings.warn(
                f"Failed {func.__name__} on attempt {attempt + 1} of {attempts}: "
                f"{e}\n{traceback.format_exc()}",
                stacklevel=2,
            )
            error_occurred = torch.tensor(1, device=device)
        logger.debug(f"Local error code: {error_occurred.item()}")
        if dist.is_initialized():
            logger.debug("Waiting for all error codes...")
            dist.barrier()
            logger.debug("Broadcasting error codes...")
            dist.broadcast(error_occurred, src=0)
            logger.debug(f"Error code: {error_occurred.item()}")
        if error_occurred.item() == 0:
            logger.debug(f"{func.__name__} successful.")
            return
        if attempt + 1 < attempts:
            time.sleep(cooldown_seconds * 2**attempt)
    if last_error is not None:
        raise last_error
    else:
        raise RuntimeError(f"Failed {func.__name__} after {attempts} attempts.")


def get_device0() -> torch.device:
    """Get the device corresponding to the main process."""
    accelerator = Accelerator()
    if accelerator.device.type == "cpu":
        return torch.device("cpu")
    elif accelerator.device.type == "cuda":
        return torch.device("cuda:0")
    elif accelerator.device.type == "mps":
        return torch.device("mps")
    else:
        return torch.device("cpu")


def safe_clip_grad_norm(
    parameters: Iterator[torch.nn.Parameter],
    max_norm: float,
    accelerator: Accelerator,
) -> torch.Tensor:
    """Clip gradients, treating tensor mismatch error as a CUDA OOM error."""
    try:
        out = accelerator.clip_grad_norm_(parameters, max_norm)
        assert isinstance(out, torch.Tensor)
        return out
    except RuntimeError as e:
        # The error message that we want to catch is:
        # RuntimeError: Boolean value of Tensor with more than one value is ambiguous
        # Example run: https://wandb.ai/farai/robust-llm/runs/hpujhq55/logs
        if "Boolean value of Tensor with more than one value is ambiguous" in str(e):
            # We need to ensure that accelerate.utils.memory.should_reduce_batch_size
            # evaluates to True. This requires raising a RuntimeError with the
            # message "CUDA out of memory".
            # https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/memory.py#L96
            raise RuntimeError("CUDA out of memory. " + str(e))
        else:
            raise e
