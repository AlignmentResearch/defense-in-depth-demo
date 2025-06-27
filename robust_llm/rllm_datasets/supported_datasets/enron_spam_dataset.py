from typing_extensions import override

from robust_llm.rllm_datasets.modifiable_chunk_spec import (
    ChunkType,
    ModifiableChunkSpec,
)
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset


@RLLMDataset.register_dataset("AlignmentResearch/EnronSpam")
class EnronSpamDataset(RLLMDataset):
    """enron_spam dataset for robust LLM experiments."""

    @property
    @override
    def num_classes(self) -> int:
        """The EnronSpam dataset has two labels (0: ham and 1: spam)."""
        return 2

    @property
    @override
    def modifiable_chunk_spec(self) -> ModifiableChunkSpec:
        """The EnronSpam dataset consists of three chunks:

        1. The instructions (IMMUTABLE)
        2. The email text (PERTURBABLE)
        3. The answer prompt (IMMUTABLE)
        """
        return ModifiableChunkSpec(
            ChunkType.IMMUTABLE,
            ChunkType.PERTURBABLE,
            ChunkType.IMMUTABLE,
        )
