"""Google Gemini client for Sage using the new google.genai SDK."""

import asyncio
import logging
from typing import Any

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiClient:
    """Client for Google Gemini API using the new google.genai SDK."""

    def __init__(self, api_key: str):
        """Initialize the Gemini client."""
        self._client = genai.Client(api_key=api_key)

    async def complete(
        self,
        prompt: str,
        model: str = "gemini-2.0-flash",
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 4096,
    ) -> str:
        """
        Send a completion request to Gemini.

        Args:
            prompt: The user prompt
            model: Gemini model name (default: gemini-2.0-flash)
            system_prompt: System instructions
            max_tokens: Maximum output tokens

        Returns:
            The model's response text
        """
        loop = asyncio.get_event_loop()

        def sync_generate() -> str:
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        max_output_tokens=max_tokens,
                    ),
                )
                return response.text
            except Exception as e:
                logger.error(f"Gemini generation failed: {e}")
                raise

        return await loop.run_in_executor(None, sync_generate)
