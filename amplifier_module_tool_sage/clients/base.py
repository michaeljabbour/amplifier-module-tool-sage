"""Base client protocol for AI providers."""

from typing import Protocol


class AIClient(Protocol):
    """Protocol for AI provider clients."""

    async def complete(
        self,
        prompt: str,
        model: str,
        system_prompt: str,
        max_tokens: int,
    ) -> str:
        """
        Send a completion request to the AI provider.

        Args:
            prompt: The user prompt to send
            model: Model identifier to use
            system_prompt: System instructions for the model
            max_tokens: Maximum tokens in response

        Returns:
            The model's response text
        """
        ...
