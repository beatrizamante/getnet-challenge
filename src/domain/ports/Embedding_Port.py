from abc import ABC, abstractmethod


class EmbeddingPort(ABC):
    """Contract for any embedding provider (OpenAI, local model, etc.)."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Convert a single text into its vector representation.

        Args:
            text: Text to embed.

        Returns:
            Vector representation of the supplied text.
        """

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Convert multiple texts into vector representations.

        Args:
            texts: Texts to embed.

        Returns:
            Vector representations in the same order as the supplied texts.
        """
