"""Defines which chunks of the text can be changed.

Our attacks operate by modifying certain chunks of the input text.
However, not all chunks of the input text can be modified while preserving
the overall meaning of the example.

There are three main things we can do to a chunk of text:
- We can overwrite it, wiping it out entirely and replacing it with something else.
- We can perturb it, changing small details while trying to preserve the meaning.
- We can leave it alone.
"""

from enum import Enum
from typing import Iterable


class ChunkType(Enum):
    """Defines the types of chunks that can be modified in the input text.

    The three types are:
    - IMMUTABLE: The chunk must not be changed.
    - PERTURBABLE: The chunk can be changed, but should be kept similar to the
        original.
    - OVERWRITABLE: The chunk can be changed, and can be replaced with something
        entirely different. Note that an overwritable chunk is also perturbable.
    """

    IMMUTABLE = 0
    PERTURBABLE = 1
    OVERWRITABLE = 2

    def __repr__(self):
        return f"{self.name}"


class ModifiableChunkSpec(tuple):
    """A specification for which chunks of the text can be changed.

    This class is a tuple of ChunkType enums, which specifies which chunks of
    the input text can be changed.
    """

    def __new__(cls, *args: ChunkType | Iterable[ChunkType]):
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            elements = args[0]
        else:
            elements = args

        if not all(isinstance(elem, ChunkType) for elem in elements):
            raise ValueError("All elements must be instances of ChunkType Enum")

        return super().__new__(cls, elements)

    @property
    def n_modifiable_chunks(self) -> int:
        """Returns the number of chunks that can be modified."""
        return sum(1 for chunk_type in self if chunk_type != ChunkType.IMMUTABLE)

    @property
    def n_perturbable_chunks(self) -> int:
        """Returns the number of PERTURBABLE chunks."""
        return sum(1 for chunk_type in self if chunk_type == ChunkType.PERTURBABLE)

    @property
    def n_overwritable_chunks(self) -> int:
        """Returns the number of OVERWRITABLE chunks."""
        return sum(1 for chunk_type in self if chunk_type == ChunkType.OVERWRITABLE)

    def get_modifiable_chunk_index(self) -> int:
        """If there is only one modifiable chunk, returns its index."""
        if self.n_modifiable_chunks != 1:
            raise ValueError("There must be exactly one modifiable chunk")

        for i, chunk_type in enumerate(self):
            if chunk_type != ChunkType.IMMUTABLE:
                return i

        raise ValueError("There must be exactly one modifiable chunk")

    def get_modifiable_chunk(self) -> ChunkType:
        """If there is only one modifiable chunk, returns its type."""
        index = self.get_modifiable_chunk_index()
        return self[index]

    def __repr__(self):
        return f"ModifiableChunkSpec{super().__repr__()}"
