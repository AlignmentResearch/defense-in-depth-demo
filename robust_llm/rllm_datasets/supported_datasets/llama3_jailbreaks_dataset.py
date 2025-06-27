from typing_extensions import override

from robust_llm.rllm_datasets.modifiable_chunk_spec import (
    ChunkType,
    ModifiableChunkSpec,
)
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset


@RLLMDataset.register_dataset("AlignmentResearch/Llama3Jailbreaks")
@RLLMDataset.register_dataset("AlignmentResearch/JailbreakInputs")
@RLLMDataset.register_dataset("AlignmentResearch/JailbreakCompletions")
@RLLMDataset.register_dataset("AlignmentResearch/JailbreakCompletionsCurriculum")
class Llama3JailbreaksDataset(RLLMDataset):
    """Llama3Jailbreaks dataset for robust LLM experiments."""

    @property
    @override
    def num_classes(self) -> int:
        return 2

    @property
    @override
    def modifiable_chunk_spec(self) -> ModifiableChunkSpec:
        """The Llama3Jailbreaks dataset consists of three chunks:

        1. The instructions, which are empty (IMMUTABLE)
        2. The prompt (PERTURBABLE)
        3. The answer prompt, also empty (IMMUTABLE)
        """
        return ModifiableChunkSpec(
            ChunkType.IMMUTABLE,
            ChunkType.PERTURBABLE,
            ChunkType.IMMUTABLE,
        )
