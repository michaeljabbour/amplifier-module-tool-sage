"""OpenAI client for Sage."""

import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Client for OpenAI API."""

    def __init__(self, api_key: str):
        """Initialize the OpenAI client."""
        self._client = AsyncOpenAI(api_key=api_key)

    async def complete(
        self,
        prompt: str,
        model: str = "gpt-4o",
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 4096,
    ) -> str:
        """
        Send a completion request to OpenAI.

        Args:
            prompt: The user prompt
            model: OpenAI model name (default: gpt-4o)
            system_prompt: System instructions
            max_tokens: Maximum output tokens

        Returns:
            The model's response text
        """
        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
