# amplifier-module-tool-sage

Strategic advisor tool for Amplifier that provides direct, outcome-focused guidance on architecture, design, product, and implementation decisions.

## Overview

Sage consults AI providers (Gemini primary, OpenAI secondary) to provide:

- **Direct recommendations** - No hedging, no "it depends" without specifics
- **Outcome-focused guidance** - Every response ties to measurable outcomes
- **Session-aware context** - Accesses conversation history for informed responses
- **Zero fluff** - No marketing speak, no vague promises

## Execution Modes

Sage supports **dual execution modes** - use Amplifier's native provider system OR direct SDK access:

| Mode | Description | When to Use |
|------|-------------|-------------|
| `native` | Uses Amplifier's mounted providers (`provider-gemini`, `provider-openai`) | Unified observability, centralized config |
| `direct` | Uses SDK clients directly (`google-genai`, `openai`) | Self-contained, works without provider config |
| `auto` | Tries native first, falls back to direct | **Default** - best of both worlds |

### Native Mode

Uses Amplifier's provider system. Requires providers to be mounted in your bundle:

```yaml
providers:
  - module: provider-gemini
    source: git+https://github.com/microsoft/amplifier-module-provider-gemini@main
  - module: provider-openai
    source: git+https://github.com/microsoft/amplifier-module-provider-openai@main

tools:
  - module: tool-sage
    source: git+https://github.com/michaeljabbour/amplifier-module-tool-sage@main
    config:
      mode: native  # Explicitly use native providers
```

**Benefits:**
- Unified observability through Amplifier's event system
- Centralized provider configuration
- Usage tracking alongside main session

### Direct Mode

Uses SDK clients directly. Self-contained, works without provider configuration:

```yaml
tools:
  - module: tool-sage
    source: git+https://github.com/michaeljabbour/amplifier-module-tool-sage@main
    config:
      mode: direct  # Explicitly use SDK clients
```

**Benefits:**
- Works out of the box with just API keys
- Independent of session's provider configuration
- Guaranteed access to Gemini/OpenAI regardless of bundle setup

### Auto Mode (Default)

Intelligently selects the best available option:

1. Checks if requested provider is mounted in Amplifier
2. If mounted → uses native mode (unified observability)
3. If not mounted → falls back to direct SDK

```yaml
tools:
  - module: tool-sage
    source: git+https://github.com/michaeljabbour/amplifier-module-tool-sage@main
    config:
      mode: auto  # Default - tries native, falls back to direct
```

## Installation

### Basic (Auto Mode)

```yaml
tools:
  - module: tool-sage
    source: git+https://github.com/michaeljabbour/amplifier-module-tool-sage@main
```

### With Native Providers

```yaml
providers:
  - module: provider-gemini
    source: git+https://github.com/microsoft/amplifier-module-provider-gemini@main
  - module: provider-openai
    source: git+https://github.com/microsoft/amplifier-module-provider-openai@main

tools:
  - module: tool-sage
    source: git+https://github.com/michaeljabbour/amplifier-module-tool-sage@main
    config:
      mode: native
      default_provider: gemini
      fallback_provider: openai
```

Or use the full bundle: [amplifier-bundle-sage](https://github.com/michaeljabbour/amplifier-bundle-sage)

## Environment Variables

Required for **direct mode** or when native providers aren't configured:

```bash
# Required for Gemini (primary)
export GOOGLE_API_KEY=your-gemini-api-key

# Optional for OpenAI (secondary/fallback)
export OPENAI_API_KEY=your-openai-api-key
```

**Note:** In native mode, providers handle their own API key management.

---

## Tool Usage

### Basic

```json
{
  "question": "Should we use PostgreSQL or MongoDB?"
}
```

### With Mode Selection

```json
{
  "question": "Should we use PostgreSQL or MongoDB?",
  "mode": "native",
  "provider": "gemini"
}
```

### With Full Context (Recommended)

```json
{
  "question": "Should we use PostgreSQL or MongoDB?",
  "domain": "architecture",
  "mode": "auto",
  "context": {
    "goal": "Support 10k DAU with flexible schema evolution",
    "constraints": ["2 engineers", "3 month timeline", "Django app"],
    "current_approach": "Leaning toward MongoDB",
    "concerns": ["query complexity", "ORM compatibility"]
  }
}
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `question` | string | **Required.** The question to answer |
| `mode` | string | `native`, `direct`, or `auto` (default: auto) |
| `domain` | string | architecture, design, product, implementation, outcomes |
| `context` | object | goal, constraints, current_approach, concerns |
| `session_context` | boolean | Include session history (default: true) |
| `provider` | string | `gemini` (default) or `openai` |
| `model` | string | Model to use (default: gemini-2.0-flash) |
| `max_tokens` | integer | Maximum tokens in response (default: 4096) |

---

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `mode` | string | `auto` | Execution mode: `native`, `direct`, `auto` |
| `default_provider` | string | `gemini` | Primary provider to consult |
| `fallback_provider` | string | `openai` | Fallback if primary fails |
| `default_model` | string | `gemini-2.0-flash` | Default model |
| `max_tokens` | integer | `4096` | Default max tokens |
| `max_session_messages` | integer | `20` | Max session messages for context |

---

## Enabling Long-Running Autonomous Work

Sage is designed to be the **strategic counsel** for extended autonomous sessions - work that runs for hours or days until complete.

### The Core Pattern

```
[Clear Outcome] + [Explicit Constraints] + [When to Consult Sage] + [Permission to Run Until Done]
```

### Sample Prompts

#### Build Until Done

```
Build the user authentication system described in the spec.

Outcome: Registration, login, logout, password reset - all working with tests.
Constraints: Use existing database, follow project patterns, test coverage > 80%.

Work autonomously. Consult sage for:
- Architecture decisions before implementing
- Tradeoffs when multiple approaches seem viable  
- Validation that you're still on track for the outcome

Don't stop until it's complete and tests pass.
```

#### Fix Everything

```
Fix all issues in this codebase until it's clean.

Outcome: Zero lint errors, all tests passing, no security issues.

For each issue:
1. Understand the root cause
2. If fix involves tradeoffs, consult sage
3. Implement and verify
4. Move to next issue

Work until the codebase is clean. Take as long as needed.
```

#### Implement a Spec

```
Implement @project:specs/feature.md completely.

Outcome: Feature working as specified, tested, documented.

Approach:
1. Read and understand the spec
2. Consult sage to validate your plan
3. Build incrementally, testing as you go
4. When spec is ambiguous, consult sage for the pragmatic choice

Continue until done. This may take hours - that's expected.
```

#### Debug Until Resolved

```
The API is returning 500 errors intermittently. Find and fix the root cause.

Outcome: API runs reliably with no intermittent errors.

Process:
1. Gather evidence
2. Form hypotheses
3. Consult sage if multiple hypotheses seem equally likely
4. Test fix thoroughly

Continue until the issue is resolved. Don't accept workarounds 
without consulting sage on whether that's the right tradeoff.
```

### Keys to Effective Autonomous Work

#### 1. Clear Outcomes (Not Tasks)

```
# Bad - task-focused
"Add authentication to the app"

# Good - outcome-focused  
"Users can register, login, logout, and reset passwords. 
All flows tested. No known security issues."
```

#### 2. Explicit Constraints

```
Constraints:
- Use existing PostgreSQL database
- Follow project coding standards
- No new dependencies without justification
- Test coverage > 80% for new code
```

#### 3. When to Consult Sage

```
Consult sage when:
- Making architecture decisions
- Facing tradeoffs with no clear winner
- Unsure if current approach serves the outcome
- Blocked and considering workarounds
- Completing major milestones (for validation)
```

#### 4. Permission to Run

```
# Enable true autonomy
"Work until complete"
"Take as long as needed"  
"Continue until the outcome is achieved"
"Don't stop until all tests pass"
```

---

## Domains

- **architecture** - System design, component boundaries, technology choices
- **design** - Patterns, interfaces, data modeling, API design
- **product** - Feature prioritization, MVP scope, user outcomes
- **implementation** - Approach, sequencing, tradeoffs
- **outcomes** - Measuring success, KPIs, validation

## Philosophy

1. **Zero fluff** - Direct answers, no corporate speak
2. **Outcomes first** - Tie every recommendation to measurable results
3. **Confident guidance** - Clear recommendations, not endless options
4. **Explicit tradeoffs** - Specifics, not vague pros/cons
5. **Redirect to outcomes** - Pull discussions back to what matters

## License

MIT
