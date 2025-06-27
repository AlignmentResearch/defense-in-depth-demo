from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch.distributed as dist

from robust_llm.dist_utils import (
    DistributedRNG,
    broadcast_int128,
    broadcast_rng_state,
    reconstruct_int128,
    split_int128,
)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_np_rng_consistency(seed):
    # We use the numpy method __setstate__ when resuming training from a checkpoint
    # so here we test that the state of the bit generator is correctly restored.
    # cf. AdversarialTrainingState.load()
    rng = np.random.default_rng(seed)
    rng.random()
    alt_rng = np.random.default_rng()
    alt_rng.__setstate__(rng.__getstate__())
    assert rng.random() == alt_rng.random()
    assert rng.random() == alt_rng.random()


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_dist_rng_consistency(seed):
    # We use the numpy method __setstate__ when resuming training from a checkpoint
    # so here we test that the state of the bit generator is correctly restored.
    # cf. AdversarialTrainingState.load()
    rng = DistributedRNG(seed, None)
    rng.random()
    alt_rng = DistributedRNG(None, None)
    alt_rng.setstate(rng.getstate())
    assert rng.random() == alt_rng.random()
    assert rng.random() == alt_rng.random()


@pytest.mark.parametrize("i", [2**127 - 1, 2**127, 2**127 + 1])
def test_reconstruct_int128(i: int):
    split = split_int128(i)
    reconstructed = reconstruct_int128(split)
    assert i == reconstructed


@pytest.mark.parametrize(
    "i",
    [
        0,
        1,
        2**32 - 1,
        2**32,
        2**64 - 1,
        2**64,
        2**96 - 1,
        2**96,
        2**128 - 1,
        12345678901234567890,  # A large number
        0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,  # Maximum 128-bit unsigned integer
    ],
)
def test_broadcast_int128(i: int):
    accelerator = MagicMock()
    accelerator.is_main_process = True
    accelerator.device = "cpu"

    with (
        patch.object(dist, "is_initialized", return_value=True),
        patch.object(dist, "get_rank", return_value=0),
        patch("robust_llm.dist_utils.broadcast_tensor", lambda data, accelerator: data),
    ):
        result = broadcast_int128(i, accelerator)
    assert result == i


def test_broadcast_rng_state():
    accelerator = MagicMock()
    accelerator.is_main_process = True
    accelerator.device = "cpu"

    rng = DistributedRNG(0, accelerator)
    state = rng.getstate()

    with (
        patch.object(dist, "is_initialized", return_value=True),
        patch.object(dist, "get_rank", return_value=0),
        patch("robust_llm.dist_utils.broadcast_tensor", lambda data, accelerator: data),
    ):
        result = broadcast_rng_state(state, accelerator)
    assert result == state
