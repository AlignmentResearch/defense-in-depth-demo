from typing_extensions import override

from robust_llm.rllm_datasets.modifiable_chunk_spec import (
    ChunkType,
    ModifiableChunkSpec,
)
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset


@RLLMDataset.register_dataset("AlignmentResearch/StrongREJECT")
@RLLMDataset.register_dataset("AlignmentResearch/BoNStrongREJECT")
@RLLMDataset.register_dataset("AlignmentResearch/ReNeLLMStrongREJECT")
@RLLMDataset.register_dataset("AlignmentResearch/PAPStrongREJECT")
@RLLMDataset.register_dataset("ianmckenzie/IanBoNStrongREJECT")
class StrongREJECTDataset(RLLMDataset):
    """StrongREJECT dataset for robust LLM experiments."""

    @property
    @override
    def num_classes(self) -> int:
        return 2

    @property
    @override
    def modifiable_chunk_spec(self) -> ModifiableChunkSpec:
        """The StrongREJECT dataset consists of three chunks:

        1. The instructions, which are empty (IMMUTABLE)
        2. The forbidden_prompt (PERTURBABLE)
        3. The answer prompt (IMMUTABLE)
        """
        return ModifiableChunkSpec(
            ChunkType.IMMUTABLE,
            ChunkType.PERTURBABLE,
            ChunkType.IMMUTABLE,
        )
