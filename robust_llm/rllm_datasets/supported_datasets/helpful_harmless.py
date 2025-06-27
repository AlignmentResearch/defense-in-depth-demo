from typing_extensions import override

from robust_llm.rllm_datasets.modifiable_chunk_spec import (
    ChunkType,
    ModifiableChunkSpec,
)
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset


@RLLMDataset.register_dataset("AlignmentResearch/Helpful")
@RLLMDataset.register_dataset("AlignmentResearch/Harmless")
class HelpfulHarmlessDataset(RLLMDataset):
    """Helpfulness and Harmlessness datasets for robust LLM experiments."""

    @property
    @override
    def num_classes(self) -> int:
        """Helpful and Harmless have two labels (0: FIRST and 1: SECOND)."""
        return 2

    @property
    @override
    def modifiable_chunk_spec(self) -> ModifiableChunkSpec:
        """Helpful and Harmless consists of three chunks:

        1. The instructions (IMMUTABLE)
        2. The two conversations (PERTURBABLE)
        3. The answer prompt (IMMUTABLE)
        """
        return ModifiableChunkSpec(
            ChunkType.IMMUTABLE,
            ChunkType.PERTURBABLE,
            ChunkType.IMMUTABLE,
        )
