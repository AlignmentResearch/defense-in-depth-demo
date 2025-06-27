import pytest

from robust_llm.rllm_datasets.modifiable_chunk_spec import (
    ChunkType,
    ModifiableChunkSpec,
)


def test_ModifiableChunkSpec():
    spec = ModifiableChunkSpec(
        ChunkType.IMMUTABLE, ChunkType.PERTURBABLE, ChunkType.OVERWRITABLE
    )
    assert spec.n_modifiable_chunks == 2
    assert spec.n_overwritable_chunks == 1
    with pytest.raises(ValueError) as ve:
        spec.get_modifiable_chunk_index()
    assert str(ve.value) == "There must be exactly one modifiable chunk"


def test_modifiable_index():
    """Check that the methods for a single modifiable chunk work correctly."""
    spec = ModifiableChunkSpec(
        ChunkType.IMMUTABLE, ChunkType.PERTURBABLE, ChunkType.IMMUTABLE
    )
    assert spec.n_modifiable_chunks == 1
    assert spec.n_overwritable_chunks == 0
    index = spec.get_modifiable_chunk_index()
    assert index == 1
    value = spec.get_modifiable_chunk()
    assert value == ChunkType.PERTURBABLE
