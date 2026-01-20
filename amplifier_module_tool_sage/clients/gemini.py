"""Google Gemini client for Sage."""

import asyncio
import logging
from typing import Any

import google.generativeai as genai

logger = logging.getLogger(__name__)


class GeminiClient:
    """Client for Google Gemini API."""

    def __init__(self, api_key: str):
        """Initialize the Gemini client."""
        genai.configure(api_key=api_key)
        self._models: dict[str, Any] = {}

    def _get_model(self, model_name: str, system_prompt: str) -> Any:
        """Get or create a model instance with the given system prompt."""
        cache_key = f"{model_name}:{hash(system_prompt)}"
        if cache_key not in self._models:
            self._models[cache_key] = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_prompt,
            )
        return self._models[cache_key]

    async def complete(
        self,
        prompt: str,
        model: str = "gemini-3-pro",
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 4096,
    ) -> str:
        """
        Send a completion request to Gemini.

        Args:
            prompt: The user prompt
            model: Gemini model name (default: gemini-3-pro)
            system_prompt: System instructions
            max_tokens: Maximum output tokens

        Returns:
            The model's response text
        """
        model_instance = self._get_model(model, system_prompt)

        # Gemini SDK is synchronous, wrap for async
        loop = asyncio.get_event_loop()

        def sync_generate() -> str:
            try:
                response = model_instance.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                    ),
                )
                return response.text
            except Exception as e:
                logger.error(f"Gemini generation failed: {e}")
                raise

        return await loop.run_in_executor(None, sync_generate)
