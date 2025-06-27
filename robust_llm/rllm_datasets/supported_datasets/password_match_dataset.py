from __future__ import annotations

from typing_extensions import override

from robust_llm.rllm_datasets.modifiable_chunk_spec import (
    ChunkType,
    ModifiableChunkSpec,
)
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset

OLD_RESPONSE_SEPARATOR = "\n---\n"


@RLLMDataset.register_dataset("AlignmentResearch/PasswordMatch")
class PasswordMatchDataset(RLLMDataset):
    @property
    @override
    def num_classes(self) -> int:
        return 2

    @property
    @override
    def modifiable_chunk_spec(self) -> ModifiableChunkSpec:
        """
        The PasswordMatch dataset has four chunks:

        1. The instructions (IMMUTABLE).
        2. The system and user passwords (IMMUTABLE).
        3. The irrelevant text to ignore (OVERWRITABLE).
        4. The answer prompt (IMMUTABLE).
        """
        return ModifiableChunkSpec(
            ChunkType.IMMUTABLE,
            ChunkType.IMMUTABLE,
            ChunkType.OVERWRITABLE,
            ChunkType.IMMUTABLE,
        )
