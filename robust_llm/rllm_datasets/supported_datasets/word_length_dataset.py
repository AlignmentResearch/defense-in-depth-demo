from typing_extensions import override

from robust_llm.rllm_datasets.modifiable_chunk_spec import (
    ChunkType,
    ModifiableChunkSpec,
)
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset


@RLLMDataset.register_dataset("AlignmentResearch/WordLength")
class WordLengthDataset(RLLMDataset):
    @property
    @override
    def num_classes(self) -> int:
        """In WordLength there are two classes: first word longer (0) or second (1)."""
        return 2

    @property
    @override
    def modifiable_chunk_spec(self) -> ModifiableChunkSpec:
        """
        The WordLength dataset has four chunks:

        1. The instructions (IMMUTABLE).
        2. The two words (IMMUTABLE).
        3. The empty chunk after the two words (OVERWRITABLE).
        4. The answer prompt (IMMUTABLE).
        """
        return ModifiableChunkSpec(
            ChunkType.IMMUTABLE,
            ChunkType.IMMUTABLE,
            ChunkType.OVERWRITABLE,
            ChunkType.IMMUTABLE,
        )
