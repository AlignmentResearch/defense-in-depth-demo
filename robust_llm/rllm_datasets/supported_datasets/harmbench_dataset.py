from typing_extensions import override

from robust_llm.rllm_datasets.modifiable_chunk_spec import (
    ChunkType,
    ModifiableChunkSpec,
)
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset


@RLLMDataset.register_dataset("AlignmentResearch/HarmBench")
class HarmBenchDataset(RLLMDataset):
    """HarmBench dataset for robust LLM experiments."""

    @property
    @override
    def num_classes(self) -> int:
        """The HarmBench dataset has two classes:

        0 for the 'Benign' class
        1 for the 'Harmful' class
        """
        return 2

    @property
    @override
    def modifiable_chunk_spec(self) -> ModifiableChunkSpec:
        """The HarmBench dataset consists of three chunks:

        1. The instructions, which are empty (IMMUTABLE)
        2. The prompt (PERTURBABLE)
        3. The answer prompt, also empty (IMMUTABLE)
        """
        return ModifiableChunkSpec(
            ChunkType.IMMUTABLE,
            ChunkType.PERTURBABLE,
            ChunkType.IMMUTABLE,
        )
