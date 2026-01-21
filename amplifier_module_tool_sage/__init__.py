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

# Response formats
ResponseFormat = Literal["text", "json", "markdown", "bullets", "mermaid", "tradeoff_matrix"]

# =========================================================================
# Domain-Specific System Prompts
# =========================================================================

SAGE_BASE_PROMPT = """You are Sage, a direct strategic advisor. Your core principles:

1. **ZERO FLUFF** - No marketing speak, no hedging, no "it depends" without specifics.
2. **OUTCOMES FIRST** - Every recommendation ties to a measurable outcome.
3. **CONFIDENT GUIDANCE** - Give clear recommendations. Ask specific questions if needed.
4. **TRADEOFFS ARE EXPLICIT** - State actual tradeoffs with specifics, not vague pros/cons.
5. **REDIRECT TO OUTCOMES** - Pull back to outcomes when discussions drift.

What NOT to do:
- Don't say "it depends" without immediately following with specifics
- Don't list options without a recommendation
- Don't use phrases like "consider", "might want to", "could potentially"
- Don't pad responses with caveats and disclaimers
- Don't repeat the question back
"""

DOMAIN_PROMPTS = {
    "architecture": """## Architecture Domain

You're advising on system architecture decisions. Focus on:
- Scalability patterns and their concrete tradeoffs
- Component boundaries and coupling
- Data flow and consistency models
- Infrastructure choices and their operational costs

When recommending:
1. State the architecture pattern clearly
2. Explain WHY it fits THIS situation (not generic benefits)
3. List concrete tradeoffs (latency vs consistency, complexity vs flexibility)
4. Specify what changes if requirements shift
""",
    "design": """## Design Domain

You're advising on software design decisions. Focus on:
- API design and contracts
- Data modeling and schema decisions
- Design patterns and when they apply
- Code organization and module boundaries

When recommending:
1. State the design approach
2. Show how it simplifies the problem
3. Identify complexity it introduces
4. Provide concrete implementation guidance
""",
    "product": """## Product Domain

You're advising on product decisions. Focus on:
- User outcomes, not features
- Simplest path to validate assumptions
- Scope that matches resources
- Metrics that indicate success

When recommending:
1. Clarify the user problem being solved
2. Recommend minimum viable scope
3. Flag complexity that doesn't serve the outcome
4. Suggest how to measure success
""",
    "implementation": """## Implementation Domain

You're advising on implementation decisions. Focus on:
- Concrete technical approaches
- Library/framework selection
- Performance considerations
- Testing and reliability strategies

When recommending:
1. Recommend specific technologies
2. Explain integration complexity
3. Identify maintenance burden
4. Suggest incremental implementation steps
""",
    "outcomes": """## Outcomes Domain

You're helping clarify and define outcomes. Focus on:
- What success looks like concretely
- How to measure progress
- Alignment between stated goals and proposed solutions
- Identifying unstated assumptions

When advising:
1. Help articulate concrete success criteria
2. Identify metrics that would indicate progress
3. Surface conflicts between goals
4. Recommend prioritization when goals compete
""",
    "general": """## General Consultation

Provide direct strategic guidance. Identify the core decision needed and give a clear recommendation.
""",
}

# =========================================================================
# Response Format Instructions
# =========================================================================

FORMAT_INSTRUCTIONS = {
    "text": "",  # Default, no special instructions
    "json": """
**OUTPUT FORMAT**: Respond with valid JSON only. Structure:
```json
{
  "recommendation": "Your clear recommendation",
  "reasoning": "Why this recommendation",
  "tradeoffs": ["tradeoff 1", "tradeoff 2"],
  "outcome": "What this achieves",
  "follow_up_questions": ["question 1", "question 2"]
}
```
""",
    "markdown": """
**OUTPUT FORMAT**: Use structured markdown with clear headers:
## Recommendation
[Your recommendation]

## Reasoning
[Why]

## Tradeoffs
- [Tradeoff 1]
- [Tradeoff 2]

## Outcome
[What this achieves]
""",
    "bullets": """
**OUTPUT FORMAT**: Use concise bullet points:
• **Recommendation**: [one sentence]
• **Why**: [one sentence]
• **Tradeoffs**: [comma-separated list]
• **Outcome**: [one sentence]
""",
    "mermaid": """
**OUTPUT FORMAT**: Include a Mermaid diagram that visualizes the recommendation.
For architecture questions, use flowchart or C4 diagrams.
For process questions, use sequence or flowchart diagrams.
For data questions, use ER diagrams.

Wrap diagrams in ```mermaid fences. Follow the diagram with a brief explanation.

Example:
```mermaid
flowchart TD
    A[Client] --> B[API Gateway]
    B --> C[Service A]
    B --> D[Service B]
```

[Brief explanation of the diagram and recommendation]
""",
    "tradeoff_matrix": """
**OUTPUT FORMAT**: Present your analysis as a tradeoff matrix table.

First state your recommendation, then provide the matrix:

| Option | Pros | Cons | Best When | Risk Level |
|--------|------|------|-----------|------------|
| Option A | ... | ... | ... | Low/Med/High |
| Option B | ... | ... | ... | Low/Med/High |

**Recommendation**: [Which option and why]
**Key Differentiator**: [The deciding factor]
""",
}

# =========================================================================
# Follow-up Suggestions Template
# =========================================================================

FOLLOW_UP_INSTRUCTION = """
After your main response, suggest 2-3 follow-up questions the user might want to explore.
Format as:

**Want to go deeper?**
- [Follow-up question 1]
- [Follow-up question 2]
- [Follow-up question 3]
"""

# Default models per provider
DEFAULT_MODELS = {
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o",
}


def _build_system_prompt(domain: str, response_format: ResponseFormat, include_follow_up: bool) -> str:
    """Build the complete system prompt based on domain and format."""
    parts = [SAGE_BASE_PROMPT]

    # Add domain-specific guidance
    domain_prompt = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["general"])
    parts.append(domain_prompt)

    # Add format instructions
    format_instruction = FORMAT_INSTRUCTIONS.get(response_format, "")
    if format_instruction:
        parts.append(format_instruction)

    # Add follow-up instruction if requested
    if include_follow_up and response_format != "json":  # JSON has its own follow_up field
        parts.append(FOLLOW_UP_INSTRUCTION)

    return "\n".join(parts)


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
- Multiple response formats for different needs

**Response Formats:**
- `text`: Natural prose (default)
- `json`: Structured JSON for programmatic use
- `markdown`: Structured markdown with headers
- `bullets`: Concise bullet points
- `mermaid`: Visual diagrams (architecture, flow, data)
- `tradeoff_matrix`: Comparison tables for options

**Execution Modes:**
- `native`: Use Amplifier's mounted providers
- `direct`: Use SDK clients directly
- `auto`: Try native first, fall back to direct (default)

**Example:**
```json
{
  "question": "Should we use a microservices or monolith architecture?",
  "domain": "architecture",
  "format": "tradeoff_matrix",
  "context": {
    "goal": "Ship MVP in 6 weeks with 2 engineers",
    "constraints": ["small team", "tight timeline"],
    "concerns": ["scaling later", "team velocity"]
  }
}
```

**Domains:** architecture, design, product, implementation, outcomes"""

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
                "format": {
                    "type": "string",
                    "enum": ["text", "json", "markdown", "bullets", "mermaid", "tradeoff_matrix"],
                    "description": (
                        "Response format: 'text' (prose), 'json' (structured), 'markdown' (headers), "
                        "'bullets' (concise), 'mermaid' (diagrams), 'tradeoff_matrix' (comparison table). "
                        "Default: text"
                    ),
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
                        "codebase_path": {
                            "type": "string",
                            "description": "Path to codebase for code-aware consultation (requires RLM)",
                        },
                    },
                },
                "include_follow_up": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include follow-up question suggestions (default: true)",
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
        """Resolve the actual execution mode based on request and availability."""
        if requested_mode == "native":
            return "native"
        elif requested_mode == "direct":
            return "direct"
        else:  # auto
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
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        domain: str,
        response_format: ResponseFormat,
    ) -> ToolResult:
        """Execute consultation using Amplifier's native provider system."""
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
            request = ChatRequest(
                messages=[
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=user_prompt),
                ],
                max_output_tokens=max_tokens,
            )

            response = await provider.complete(request, model=model)
            response_text = self._extract_text_from_response(response)

            return ToolResult(
                success=True,
                output={
                    "mode": "native",
                    "provider": provider_name,
                    "model": model,
                    "domain": domain,
                    "format": response_format,
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
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        domain: str,
        response_format: ResponseFormat,
    ) -> ToolResult:
        """Execute consultation using direct SDK clients."""
        try:
            client = self._get_direct_client(provider_name)
            response = await client.complete(
                prompt=user_prompt,
                model=model,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )

            return ToolResult(
                success=True,
                output={
                    "mode": "direct",
                    "provider": provider_name,
                    "model": model,
                    "domain": domain,
                    "format": response_format,
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

            max_messages = self.max_session_messages
            recent = messages[-max_messages:] if len(messages) > max_messages else messages

            formatted = []
            for msg in recent:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                if isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                    content = " ".join(text_parts)

                if len(content) > 1000:
                    content = content[:1000] + "... [truncated]"

                if content.strip():
                    formatted.append(f"**{role.upper()}**: {content}")

            return "\n\n---\n\n".join(formatted)

        except Exception as e:
            logger.warning(f"Failed to get session context: {e}")
            return ""

    # =========================================================================
    # Codebase Context (RLM Integration)
    # =========================================================================

    async def _get_codebase_context(self, codebase_path: str, question: str) -> str:
        """Use RLM to extract relevant codebase context for the question."""
        try:
            # Check if RLM tool is available
            tools = self.coordinator.get("tools")
            if not tools or "rlm" not in tools:
                logger.warning("RLM tool not available - codebase context skipped")
                return ""

            rlm_tool = tools["rlm"]

            # Query RLM for relevant facts
            result = await rlm_tool.execute(
                {
                    "file_path": codebase_path,
                    "query": f"Extract architecture and design facts relevant to answering: {question}",
                    "content_type": "code",
                }
            )

            if result.success and result.output:
                return result.output.get("answer", "")
            else:
                logger.warning(f"RLM query failed: {result.error}")
                return ""

        except Exception as e:
            logger.warning(f"Failed to get codebase context: {e}")
            return ""

    # =========================================================================
    # Prompt Building
    # =========================================================================

    def _build_user_prompt(
        self, question: str, domain: str, context: dict, session_context: str, codebase_context: str
    ) -> str:
        """Build the user prompt for Sage consultation."""
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

        # Add codebase context if available
        if codebase_context:
            prompt_parts.append(f"\n**CODEBASE FACTS** (from analysis):\n{codebase_context}")

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
        """Execute the Sage consultation."""
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
        response_format: ResponseFormat = input.get("format", "text")
        include_follow_up = input.get("include_follow_up", True)
        include_session = input.get("session_context", True)
        max_tokens = input.get("max_tokens", self.max_tokens)

        # Get session context if requested
        session_context = ""
        if include_session:
            session_context = await self._get_session_context()

        # Get codebase context if path provided
        codebase_context = ""
        codebase_path = context.get("codebase_path")
        if codebase_path:
            codebase_context = await self._get_codebase_context(codebase_path, question)

        # Build prompts
        system_prompt = _build_system_prompt(domain, response_format, include_follow_up)
        user_prompt = self._build_user_prompt(question, domain, context, session_context, codebase_context)

        # Resolve execution mode
        resolved_mode = self._resolve_mode(requested_mode, provider_name)

        logger.info(
            f"Sage consultation: mode={resolved_mode}, provider={provider_name}, "
            f"model={model}, format={response_format}"
        )

        # Execute based on mode
        try:
            if resolved_mode == "native":
                return await self._execute_native(
                    provider_name, model, system_prompt, user_prompt, max_tokens, domain, response_format
                )
            else:  # direct
                return await self._execute_direct(
                    provider_name, model, system_prompt, user_prompt, max_tokens, domain, response_format
                )

        except Exception as primary_error:
            logger.error(f"Primary execution failed ({resolved_mode}/{provider_name}): {primary_error}")

            # Try fallback provider
            fallback_provider = (
                self.fallback_provider if provider_name != self.fallback_provider else self.default_provider
            )
            if fallback_provider == provider_name:
                return ToolResult(success=False, error={"message": f"Sage consultation failed: {str(primary_error)}"})

            logger.info(f"Attempting fallback to {fallback_provider}")
            fallback_model = DEFAULT_MODELS.get(fallback_provider, "gpt-4o")
            fallback_mode = self._resolve_mode(requested_mode, fallback_provider)

            try:
                if fallback_mode == "native":
                    result = await self._execute_native(
                        fallback_provider,
                        fallback_model,
                        system_prompt,
                        user_prompt,
                        max_tokens,
                        domain,
                        response_format,
                    )
                else:
                    result = await self._execute_direct(
                        fallback_provider,
                        fallback_model,
                        system_prompt,
                        user_prompt,
                        max_tokens,
                        domain,
                        response_format,
                    )

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
