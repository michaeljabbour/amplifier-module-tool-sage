"""
Sage - Strategic advisor tool for Amplifier.

Provides direct, outcome-focused guidance on architecture, design,
product, and implementation decisions via Gemini (primary) and OpenAI (secondary).
"""

import logging
import os
from typing import TYPE_CHECKING, Any

from amplifier_core import ModuleCoordinator, ToolResult

from .clients.gemini import GeminiClient
from .clients.openai import OpenAIClient

if TYPE_CHECKING:
    from .clients.base import AIClient

logger = logging.getLogger(__name__)

# The system prompt that shapes Sage's responses
SAGE_SYSTEM_PROMPT = """You are Sage, a direct strategic advisor for software architecture, design, and product decisions.

## Your Core Principles

1. **ZERO FLUFF** - No marketing speak, no hedging, no "it depends" without specifics. Be direct.

2. **OUTCOMES FIRST** - Every recommendation ties to a measurable outcome. Ask: "What does success look like?"

3. **CONFIDENT GUIDANCE** - Give clear recommendations. If you need more info, ask specific questions.

4. **TRADEOFFS ARE EXPLICIT** - When presenting options, state the actual tradeoffs with specifics, not vague pros/cons.

5. **REDIRECT TO OUTCOMES** - If a question is getting into weeds, pull back to: "What outcome are we trying to achieve here?"

## Response Format

For architecture/design questions:
- State your recommendation clearly upfront
- Explain WHY in 2-3 sentences
- List specific tradeoffs if relevant
- End with the outcome this achieves

For product questions:
- Clarify the user outcome first
- Recommend the simplest path to that outcome
- Flag complexity that doesn't serve the outcome

## What NOT to Do

- Don't say "it depends" without immediately following with specifics
- Don't list options without a recommendation
- Don't use phrases like "consider", "might want to", "could potentially"
- Don't pad responses with caveats and disclaimers
- Don't repeat the question back
- Don't use marketing language or buzzwords

## Session Context

You may receive conversation history. Use it to:
- Understand what's already been discussed
- Avoid retreading covered ground
- Build on existing decisions
- Identify if the discussion is drifting from outcomes
"""


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """Mount the Sage strategic advisor tool."""
    config = config or {}
    tool = SageTool(config, coordinator)
    await coordinator.mount("tools", tool, name=tool.name)
    logger.info("Mounted Sage strategic advisor tool")


class SageTool:
    """Strategic advisor via Gemini/OpenAI - outcome-focused, zero fluff."""

    name = "sage"
    description = """Consult Sage for direct, outcome-focused guidance on architecture, design, product, and implementation questions.

Sage provides:
- Clear recommendations (not wishy-washy options)
- Explicit tradeoffs with specifics
- Guidance tied to measurable outcomes
- Redirection when discussions drift from goals

**For best results, provide structured context:**

- **goal**: What outcome are you trying to achieve?
- **constraints**: Time, resources, technical limitations
- **current_approach**: What's being considered
- **concerns**: Specific uncertainties

**Example:**
```json
{
  "question": "Should we use a microservices or monolith architecture?",
  "domain": "architecture",
  "context": {
    "goal": "Ship MVP in 6 weeks with 2 engineers",
    "constraints": ["small team", "tight timeline", "uncertain requirements"],
    "concerns": ["scaling later", "team velocity"]
  }
}
```

**Domains:** architecture, design, product, implementation, outcomes

Sage accesses session history by default for additional context."""

    def __init__(self, config: dict[str, Any], coordinator: ModuleCoordinator):
        self.config = config
        self.coordinator = coordinator
        self.default_provider = config.get("default_provider", "gemini")
        self.default_model = config.get("default_model", "gemini-2.0-flash")
        self.max_tokens = config.get("max_tokens", 4096)
        self.max_session_messages = config.get("max_session_messages", 20)
        self._clients: dict[str, "AIClient"] = {}

    @property
    def input_schema(self) -> dict:
        """JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The specific question to answer",
                },
                "domain": {
                    "type": "string",
                    "enum": ["architecture", "design", "product", "implementation", "outcomes"],
                    "description": "Primary domain of the question (default: inferred from question)",
                },
                "context": {
                    "type": "object",
                    "description": "Structured context for the consultation",
                    "properties": {
                        "goal": {
                            "type": "string",
                            "description": "What outcome are we trying to achieve?",
                        },
                        "constraints": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Known constraints (time, resources, technical)",
                        },
                        "current_approach": {
                            "type": "string",
                            "description": "What's being considered or already decided",
                        },
                        "concerns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific worries or uncertainties",
                        },
                    },
                },
                "session_context": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include recent session history for context (default: true)",
                },
                "provider": {
                    "type": "string",
                    "enum": ["gemini", "openai"],
                    "description": "Which AI provider to consult (default: gemini)",
                },
                "model": {
                    "type": "string",
                    "description": "Specific model to use (default: gemini-2.0-flash)",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens in response (default: 4096)",
                },
            },
            "required": ["question"],
        }

    def _get_client(self, provider: str) -> "AIClient":
        """Get or create a client for the specified provider."""
        if provider not in self._clients:
            if provider == "gemini":
                api_key = self.config.get("gemini_api_key") or os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError(
                        "GOOGLE_API_KEY environment variable not set. "
                        "Set it or provide gemini_api_key in config."
                    )
                self._clients[provider] = GeminiClient(api_key)

            elif provider == "openai":
                api_key = self.config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError(
                        "OPENAI_API_KEY environment variable not set. "
                        "Set it or provide openai_api_key in config."
                    )
                self._clients[provider] = OpenAIClient(api_key)

            else:
                raise ValueError(f"Unknown provider: {provider}. Supported: gemini, openai")

        return self._clients[provider]

    async def _get_session_context(self) -> str:
        """Extract recent session history for context."""
        try:
            context_mgr = self.coordinator.get("context")
            if not context_mgr:
                return ""

            messages = await context_mgr.get_messages()
            if not messages:
                return ""

            # Get recent messages
            max_messages = self.max_session_messages
            recent = messages[-max_messages:] if len(messages) > max_messages else messages

            # Format for Sage
            formatted = []
            for msg in recent:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                # Handle content blocks (list of dicts with type/text)
                if isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                    content = " ".join(text_parts)

                # Truncate very long messages
                if len(content) > 1000:
                    content = content[:1000] + "... [truncated]"

                if content.strip():
                    formatted.append(f"**{role.upper()}**: {content}")

            return "\n\n---\n\n".join(formatted)

        except Exception as e:
            logger.warning(f"Failed to get session context: {e}")
            return ""

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        """Execute the Sage consultation."""
        question = input.get("question")
        if not question:
            return ToolResult(success=False, error={"message": "Question is required"})

        provider = input.get("provider", self.default_provider)
        model = input.get("model", self.default_model)
        context = input.get("context", {})
        domain = input.get("domain", "general")
        include_session = input.get("session_context", True)
        max_tokens = input.get("max_tokens", self.max_tokens)

        # Build the full prompt
        prompt_parts = []

        # Add domain focus
        prompt_parts.append(f"**DOMAIN**: {domain.upper()}")

        # Add structured context if provided
        if context:
            prompt_parts.append("\n**CONTEXT**:")
            if context.get("goal"):
                prompt_parts.append(f"- **Goal**: {context['goal']}")
            if context.get("constraints"):
                constraints = context["constraints"]
                if isinstance(constraints, list):
                    prompt_parts.append(f"- **Constraints**: {', '.join(constraints)}")
                else:
                    prompt_parts.append(f"- **Constraints**: {constraints}")
            if context.get("current_approach"):
                prompt_parts.append(f"- **Current approach**: {context['current_approach']}")
            if context.get("concerns"):
                concerns = context["concerns"]
                if isinstance(concerns, list):
                    prompt_parts.append(f"- **Concerns**: {', '.join(concerns)}")
                else:
                    prompt_parts.append(f"- **Concerns**: {concerns}")

        # Add session history if requested
        if include_session:
            session_context = await self._get_session_context()
            if session_context:
                prompt_parts.append(f"\n**SESSION HISTORY** (for context):\n\n{session_context}")

        # Add the actual question
        prompt_parts.append(f"\n**QUESTION**:\n{question}")

        full_prompt = "\n".join(prompt_parts)

        try:
            client = self._get_client(provider)
            response = await client.complete(
                prompt=full_prompt,
                model=model,
                system_prompt=SAGE_SYSTEM_PROMPT,
                max_tokens=max_tokens,
            )

            return ToolResult(
                success=True,
                output={
                    "provider": provider,
                    "model": model,
                    "domain": domain,
                    "response": response,
                },
            )

        except Exception as e:
            logger.error(f"Sage consultation failed: {e}")

            # Try fallback to other provider if primary fails
            fallback_provider = "openai" if provider == "gemini" else "gemini"
            try:
                logger.info(f"Attempting fallback to {fallback_provider}")
                client = self._get_client(fallback_provider)
                response = await client.complete(
                    prompt=full_prompt,
                    model=self._get_fallback_model(fallback_provider),
                    system_prompt=SAGE_SYSTEM_PROMPT,
                    max_tokens=max_tokens,
                )

                return ToolResult(
                    success=True,
                    output={
                        "provider": fallback_provider,
                        "model": self._get_fallback_model(fallback_provider),
                        "domain": domain,
                        "response": response,
                        "note": f"Used fallback provider ({fallback_provider}) due to {provider} error",
                    },
                )
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                return ToolResult(
                    success=False,
                    error={
                        "message": f"Sage consultation failed: {str(e)}. Fallback also failed: {str(fallback_error)}"
                    },
                )

    def _get_fallback_model(self, provider: str) -> str:
        """Get default model for fallback provider."""
        if provider == "gemini":
            return "gemini-2.0-flash"
        elif provider == "openai":
            return "gpt-4o"
        return "unknown"
