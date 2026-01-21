"""
Sage - Strategic advisor tool for Amplifier.

Provides direct, outcome-focused guidance on architecture, design,
product, and implementation decisions.

Supports two execution modes:
- **native**: Uses Amplifier's mounted providers (provider-gemini, provider-openai)
- **direct**: Uses SDK clients directly (google-genai, openai)
- **auto**: Tries native first, falls back to direct if provider not mounted
"""

import logging
import os
from typing import TYPE_CHECKING, Any, Literal

from amplifier_core import ModuleCoordinator, ToolResult

if TYPE_CHECKING:
    from amplifier_core.interfaces import Provider

    from .clients.base import AIClient

logger = logging.getLogger(__name__)

# Execution modes
ExecutionMode = Literal["native", "direct", "auto"]

# The system prompt that shapes Sage's responses
SAGE_SYSTEM_PROMPT = """You are Sage, a direct strategic advisor for software architecture, \
design, and product decisions.

## Your Core Principles

1. **ZERO FLUFF** - No marketing speak, no hedging, no "it depends" without specifics. Be direct.

2. **OUTCOMES FIRST** - Every recommendation ties to a measurable outcome. Ask: "What does success look like?"

3. **CONFIDENT GUIDANCE** - Give clear recommendations. If you need more info, ask specific questions.

4. **TRADEOFFS ARE EXPLICIT** - When presenting options, state the actual tradeoffs with specifics, \
not vague pros/cons.

5. **REDIRECT TO OUTCOMES** - If a question is getting into weeds, pull back to: \
"What outcome are we trying to achieve here?"

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

# Default models per provider
DEFAULT_MODELS = {
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o",
}


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """Mount the Sage strategic advisor tool."""
    config = config or {}
    tool = SageTool(config, coordinator)
    await coordinator.mount("tools", tool, name=tool.name)
    logger.info(f"Mounted Sage strategic advisor tool (mode: {tool.default_mode})")


class SageTool:
    """Strategic advisor via Gemini/OpenAI - outcome-focused, zero fluff.

    Supports dual execution modes:
    - native: Uses Amplifier's provider system (requires providers to be mounted)
    - direct: Uses SDK clients directly (self-contained, requires API keys)
    - auto: Intelligent selection - tries native first, falls back to direct
    """

    name = "sage"
    description = """Consult Sage for direct, outcome-focused guidance on architecture, design, \
product, and implementation questions.

Sage provides:
- Clear recommendations (not wishy-washy options)
- Explicit tradeoffs with specifics
- Guidance tied to measurable outcomes
- Redirection when discussions drift from goals

**Execution Modes:**
- `native`: Use Amplifier's mounted providers (unified observability)
- `direct`: Use SDK clients directly (self-contained, works without provider config)
- `auto`: Try native first, fall back to direct (default)

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
  "mode": "auto",
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

        # Mode configuration
        self.default_mode: ExecutionMode = config.get("mode", "auto")

        # Provider configuration
        self.default_provider = config.get("default_provider", "gemini")
        self.fallback_provider = config.get("fallback_provider", "openai")
        self.default_model = config.get("default_model", DEFAULT_MODELS.get(self.default_provider, "gemini-2.0-flash"))

        # Token limits
        self.max_tokens = config.get("max_tokens", 4096)
        self.max_session_messages = config.get("max_session_messages", 20)

        # Direct mode clients (lazy initialized)
        self._direct_clients: dict[str, "AIClient"] = {}

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
                "mode": {
                    "type": "string",
                    "enum": ["native", "direct", "auto"],
                    "description": (
                        "Execution mode: 'native' (Amplifier providers), 'direct' (SDK clients), "
                        "'auto' (try native, fall back to direct). Default: auto"
                    ),
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

    # =========================================================================
    # Mode Detection & Selection
    # =========================================================================

    def _is_native_provider_available(self, provider_name: str) -> bool:
        """Check if a provider is mounted in the Amplifier session."""
        providers = self.coordinator.get("providers")
        if not providers:
            return False
        return provider_name in providers

    def _get_native_provider(self, provider_name: str) -> "Provider | None":
        """Get a mounted Amplifier provider by name."""
        return self.coordinator.get("providers", provider_name)

    def _resolve_mode(self, requested_mode: ExecutionMode, provider_name: str) -> ExecutionMode:
        """Resolve the actual execution mode based on request and availability.

        Args:
            requested_mode: The mode requested (native/direct/auto)
            provider_name: The provider to use

        Returns:
            The resolved mode to use
        """
        if requested_mode == "native":
            # User explicitly wants native - will fail if not available
            return "native"
        elif requested_mode == "direct":
            # User explicitly wants direct SDK
            return "direct"
        else:  # auto
            # Try native first, fall back to direct
            if self._is_native_provider_available(provider_name):
                logger.debug(f"Auto mode: using native provider '{provider_name}'")
                return "native"
            else:
                logger.debug(f"Auto mode: provider '{provider_name}' not mounted, using direct SDK")
                return "direct"

    # =========================================================================
    # Native Mode Execution (Amplifier Providers)
    # =========================================================================

    async def _execute_native(
        self,
        provider_name: str,
        model: str,
        full_prompt: str,
        max_tokens: int,
        domain: str,
    ) -> ToolResult:
        """Execute consultation using Amplifier's native provider system.

        Uses ChatRequest/ChatResponse models for unified observability.
        """
        from amplifier_core.message_models import ChatRequest, Message

        provider = self._get_native_provider(provider_name)
        if not provider:
            return ToolResult(
                success=False,
                error={
                    "message": (
                        f"Provider '{provider_name}' not mounted. Either mount it in your bundle or use mode='direct'."
                    )
                },
            )

        try:
            # Build ChatRequest using Amplifier's message models
            request = ChatRequest(
                messages=[
                    Message(role="system", content=SAGE_SYSTEM_PROMPT),
                    Message(role="user", content=full_prompt),
                ],
                max_output_tokens=max_tokens,
            )

            # Call the provider
            response = await provider.complete(request, model=model)

            # Extract text from response content blocks
            response_text = self._extract_text_from_response(response)

            return ToolResult(
                success=True,
                output={
                    "mode": "native",
                    "provider": provider_name,
                    "model": model,
                    "domain": domain,
                    "response": response_text,
                    "usage": response.usage.model_dump() if response.usage else None,
                },
            )

        except Exception as e:
            logger.error(f"Native provider execution failed: {e}")
            raise

    def _extract_text_from_response(self, response) -> str:
        """Extract text content from a ChatResponse."""
        text_parts = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
            elif hasattr(block, "thinking"):
                # Include thinking blocks if present
                pass  # Skip thinking for now, just return main response
        return "".join(text_parts)

    # =========================================================================
    # Direct Mode Execution (SDK Clients)
    # =========================================================================

    def _get_direct_client(self, provider: str) -> "AIClient":
        """Get or create a direct SDK client for the specified provider."""
        if provider not in self._direct_clients:
            if provider == "gemini":
                from .clients.gemini import GeminiClient

                api_key = self.config.get("gemini_api_key") or os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError(
                        "GOOGLE_API_KEY environment variable not set. Set it or provide gemini_api_key in config."
                    )
                self._direct_clients[provider] = GeminiClient(api_key)

            elif provider == "openai":
                from .clients.openai import OpenAIClient

                api_key = self.config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError(
                        "OPENAI_API_KEY environment variable not set. Set it or provide openai_api_key in config."
                    )
                self._direct_clients[provider] = OpenAIClient(api_key)

            else:
                raise ValueError(f"Unknown provider: {provider}. Supported: gemini, openai")

        return self._direct_clients[provider]

    async def _execute_direct(
        self,
        provider_name: str,
        model: str,
        full_prompt: str,
        max_tokens: int,
        domain: str,
    ) -> ToolResult:
        """Execute consultation using direct SDK clients."""
        try:
            client = self._get_direct_client(provider_name)
            response = await client.complete(
                prompt=full_prompt,
                model=model,
                system_prompt=SAGE_SYSTEM_PROMPT,
                max_tokens=max_tokens,
            )

            return ToolResult(
                success=True,
                output={
                    "mode": "direct",
                    "provider": provider_name,
                    "model": model,
                    "domain": domain,
                    "response": response,
                },
            )

        except Exception as e:
            logger.error(f"Direct SDK execution failed: {e}")
            raise

    # =========================================================================
    # Session Context
    # =========================================================================

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

    # =========================================================================
    # Prompt Building
    # =========================================================================

    def _build_prompt(self, question: str, domain: str, context: dict, session_context: str) -> str:
        """Build the full prompt for Sage consultation."""
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

        # Add session history if available
        if session_context:
            prompt_parts.append(f"\n**SESSION HISTORY** (for context):\n\n{session_context}")

        # Add the actual question
        prompt_parts.append(f"\n**QUESTION**:\n{question}")

        return "\n".join(prompt_parts)

    # =========================================================================
    # Main Execute
    # =========================================================================

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        """Execute the Sage consultation.

        Supports three modes:
        - native: Uses Amplifier's mounted providers
        - direct: Uses SDK clients directly
        - auto: Tries native first, falls back to direct
        """
        question = input.get("question")
        if not question:
            return ToolResult(success=False, error={"message": "Question is required"})

        # Get parameters
        requested_mode: ExecutionMode = input.get("mode", self.default_mode)
        provider_name = input.get("provider", self.default_provider)
        model = (
            input.get("model")
            or self.config.get("default_model")
            or DEFAULT_MODELS.get(provider_name, "gemini-2.0-flash")
        )
        context = input.get("context", {})
        domain = input.get("domain", "general")
        include_session = input.get("session_context", True)
        max_tokens = input.get("max_tokens", self.max_tokens)

        # Get session context if requested
        session_context = ""
        if include_session:
            session_context = await self._get_session_context()

        # Build the full prompt
        full_prompt = self._build_prompt(question, domain, context, session_context)

        # Resolve execution mode
        resolved_mode = self._resolve_mode(requested_mode, provider_name)

        logger.info(f"Sage consultation: mode={resolved_mode}, provider={provider_name}, model={model}")

        # Execute based on mode
        try:
            if resolved_mode == "native":
                return await self._execute_native(provider_name, model, full_prompt, max_tokens, domain)
            else:  # direct
                return await self._execute_direct(provider_name, model, full_prompt, max_tokens, domain)

        except Exception as primary_error:
            logger.error(f"Primary execution failed ({resolved_mode}/{provider_name}): {primary_error}")

            # Try fallback provider
            fallback_provider = (
                self.fallback_provider if provider_name != self.fallback_provider else self.default_provider
            )
            if fallback_provider == provider_name:
                # No different fallback available
                return ToolResult(success=False, error={"message": f"Sage consultation failed: {str(primary_error)}"})

            logger.info(f"Attempting fallback to {fallback_provider}")
            fallback_model = DEFAULT_MODELS.get(fallback_provider, "gpt-4o")
            fallback_mode = self._resolve_mode(requested_mode, fallback_provider)

            try:
                if fallback_mode == "native":
                    result = await self._execute_native(
                        fallback_provider, fallback_model, full_prompt, max_tokens, domain
                    )
                else:
                    result = await self._execute_direct(
                        fallback_provider, fallback_model, full_prompt, max_tokens, domain
                    )

                # Add fallback note
                if result.success and result.output:
                    result.output["note"] = (
                        f"Used fallback provider ({fallback_provider}) "
                        f"due to {provider_name} error: {str(primary_error)}"
                    )

                return result

            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                return ToolResult(
                    success=False,
                    error={
                        "message": (
                            f"Sage consultation failed: {str(primary_error)}. "
                            f"Fallback ({fallback_provider}) also failed: {str(fallback_error)}"
                        )
                    },
                )
