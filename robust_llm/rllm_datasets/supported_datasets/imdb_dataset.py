from typing_extensions import override

from robust_llm.rllm_datasets.modifiable_chunk_spec import (
    ChunkType,
    ModifiableChunkSpec,
)
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset


@RLLMDataset.register_dataset("AlignmentResearch/IMDB")
class IMDBDataset(RLLMDataset):
    """IMDB dataset for robust LLM experiments."""

    @property
    @override
    def num_classes(self) -> int:
        """IMDB has two labels (neg and pos)."""
        return 2

    @property
    @override
    def modifiable_chunk_spec(self) -> ModifiableChunkSpec:
        """IMDB consists of three chunks:

        1. The instructions (IMMUTABLE)
        2. The review text (PERTURBABLE)
        3. The answer prompt (IMMUTABLE)
        """
        return ModifiableChunkSpec(
            ChunkType.IMMUTABLE,
            ChunkType.PERTURBABLE,
            ChunkType.IMMUTABLE,
        )
