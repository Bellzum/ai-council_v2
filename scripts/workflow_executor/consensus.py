"""
Multi-Agent Consensus Framework

Implements the k-phase commit agreement pattern per workflow-flow.md specification:
"The consensus process implements a simple k-phase commit agreement with a leader,
where k specifies the number of iteration rounds (default to 2)."

This module provides a generic, reusable consensus orchestrator that can be
instantiated for different use cases:

1. Epic-to-Feature breakdown (backlog-grooming)
   - Leader: senior-engineer
   - Evaluators: architect, security-specialist, tester

2. Feature-to-Task creation (backlog-grooming)
   - Leader: senior-engineer
   - Evaluators: architect, security-specialist, tester

3. Task/Bug implementation (sprint-execution)
   - Leader: engineer
   - Evaluators: tester

Key Design Principles:
- Parallel evaluation using asyncio.gather() for evaluator agents
- Early termination when all evaluators accept
- Efficient feedback relay (reference files by path, don't copy content)
- No truncation of context (per workflow-flow.md spec)
"""

import asyncio
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union

# Try to import anthropic for API calls
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AgentSettings:
    """Per-agent configuration for model, thinking, and context window."""
    extended_thinking: bool = False  # Enable extended thinking (budget_tokens)
    thinking_budget: int = 10000  # Budget tokens for extended thinking
    max_context: bool = False  # Use maximum context window (200k for claude-sonnet-4)
    model_override: Optional[str] = None  # Override default model for this agent


@dataclass
class ConsensusConfig:
    """
    Configuration for a consensus process instance.

    Defines the leader agent, evaluator agents, max rounds, and prompts
    for a specific consensus use case.
    """
    leader_agent: str  # e.g., "senior-engineer" or "engineer"
    evaluator_agents: List[str]  # e.g., ["architect", "security-specialist", "tester"]
    max_rounds: int = 2  # k=2 default for design, k=3 for implementation
    leader_proposal_prompt: str = ""  # What leader should create/implement
    evaluator_prompts: Dict[str, str] = field(default_factory=dict)  # Per-evaluator focus
    output_format: str = "json"  # "json" or "markdown"
    artifact_dir: Optional[str] = None  # Directory to save artifacts
    # Per-agent settings for thinking and context
    agent_settings: Dict[str, AgentSettings] = field(default_factory=dict)


@dataclass
class AgentEvaluation:
    """Single agent's evaluation of a proposal."""
    agent_name: str
    accepted: bool
    concerns: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    raw_response: str = ""


@dataclass
class ConsensusRound:
    """Results from one round of consensus."""
    round_number: int
    leader_proposal: Dict[str, Any]
    evaluations: List[AgentEvaluation]
    all_accepted: bool
    duration_seconds: float = 0.0


@dataclass
class ConsensusResult:
    """Final result of consensus process."""
    final_proposal: Dict[str, Any]
    rounds: List[ConsensusRound]
    consensus_reached: bool
    artifacts: List[str] = field(default_factory=list)  # Paths to generated files
    total_duration_seconds: float = 0.0
    total_tokens_used: int = 0


# =============================================================================
# Agent Interface
# =============================================================================

class AgentInterface:
    """
    Interface for invoking AI agents using Claude Agent SDK.

    Provides:
    - Tool access (Read, Edit, Write, Bash, Grep, Glob)
    - Session continuity (resume previous conversations)
    - Per-agent session management
    - Token usage tracking

    Supports backends:
    - sdk: Claude Agent SDK (default - provides tools and session continuity)
    - anthropic: Direct Anthropic API (no tools, for simple prompts)
    - mock: For testing
    """

    def __init__(
        self,
        backend: str = "sdk",
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 8000,
        api_key: Optional[str] = None,
        tool_preset: str = "implementation"
    ):
        """
        Initialize agent interface.

        Args:
            backend: "sdk" (default), "anthropic", or "mock"
            model: Model to use
            max_tokens: Max tokens for response (used by anthropic backend)
            api_key: Anthropic API key (reads from env if not provided)
            tool_preset: Tool preset for SDK ("implementation", "read_only", "testing")
        """
        self.backend = backend
        self.model = model
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.tool_preset = tool_preset
        self._client = None
        self._last_token_usage = {"input_tokens": 0, "output_tokens": 0}
        self.total_tokens = {"input_tokens": 0, "output_tokens": 0}

        # Per-agent SDK wrappers for session continuity
        self._agent_wrappers: Dict[str, Any] = {}

    def _get_client(self):
        """Get or create Anthropic client (for anthropic backend)."""
        if self._client is None and HAS_ANTHROPIC and self.api_key:
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def _get_agent_wrapper(self, agent_name: str, model_override: Optional[str] = None):
        """Get or create AgentSDKWrapper for an agent (enables session continuity)."""
        # Use model override if provided, otherwise use default
        model = model_override or self.model

        # Create unique key including model to support different models per agent
        cache_key = f"{agent_name}:{model}"

        if cache_key not in self._agent_wrappers:
            from scripts.workflow_executor.agent_sdk import AgentSDKWrapper
            self._agent_wrappers[cache_key] = AgentSDKWrapper(
                workflow_name=f"consensus-{agent_name}",
                tool_preset=self.tool_preset,
                max_turns=30,
                model=model,
            )
        return self._agent_wrappers[cache_key]

    # Retry delays tuned for Claude's per-minute rate limits
    # 15s + 20s + 30s = 65s total wait, giving ~3 opportunities per minute window
    RETRY_DELAYS = [15, 20, 30]

    async def invoke(
        self,
        agent_name: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        agent_settings: Optional[AgentSettings] = None,
        continue_session: bool = False,
        retry_delays: Optional[List[float]] = None
    ) -> str:
        """
        Invoke an agent with the given prompt, with retry logic for transient errors.

        Args:
            agent_name: Name of the agent (for logging/agent definition loading)
            prompt: The prompt to send to the agent
            system_prompt: Optional system prompt (agent definition)
            agent_settings: Optional per-agent settings (thinking, context window)
            continue_session: If True, continue previous session for this agent
            retry_delays: List of delays in seconds between retries (default: [15, 20, 30])
                          Tuned for Claude's per-minute rate limits.

        Returns:
            Agent's response text

        Raises:
            RuntimeError: If all retries exhausted or non-retryable error occurs

        Timing:
            With default retry_delays [15, 20, 30]:
            - Max retries: 3
            - Total wait time: 65 seconds
            - Worst case with 60s API timeout per attempt: ~305 seconds
        """
        delays = retry_delays if retry_delays is not None else self.RETRY_DELAYS
        max_retries = len(delays)
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                # Get model override from agent settings if provided
                model_override = agent_settings.model_override if agent_settings else None

                if self.backend == "mock":
                    return await self._invoke_mock(agent_name, prompt)
                elif self.backend == "sdk":
                    return await self._invoke_sdk(agent_name, prompt, system_prompt, continue_session, model_override)
                elif self.backend == "anthropic":
                    return await self._invoke_anthropic(agent_name, prompt, system_prompt, agent_settings)
                else:
                    raise ValueError(f"Unknown backend: {self.backend}")

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check if error is retryable (rate limit, overload, timeout)
                is_retryable = any(keyword in error_str for keyword in [
                    "rate_limit", "rate limit", "ratelimit",
                    "overloaded", "overload",
                    "timeout", "timed out",
                    "529",  # Overloaded status code
                    "429",  # Rate limit status code
                    "503",  # Service unavailable
                    "502",  # Bad gateway
                ])

                if not is_retryable:
                    # Non-retryable error - raise immediately
                    raise

                if attempt < max_retries:
                    # Use configured delay with small jitter to prevent thundering herd
                    delay = delays[attempt] + (time.time() % 2)  # Add 0-2s jitter
                    print(f"  [Retry {attempt + 1}/{max_retries}] {agent_name}: {type(e).__name__}, waiting {delay:.0f}s...", flush=True)
                    await asyncio.sleep(delay)
                else:
                    # All retries exhausted
                    raise RuntimeError(f"Agent {agent_name} failed after {max_retries} retries: {last_error}")

    async def _invoke_mock(self, agent_name: str, prompt: str) -> str:
        """Mock invocation for testing."""
        await asyncio.sleep(0.1)  # Simulate latency
        return json.dumps({
            "accepted": True,
            "concerns": [],
            "suggestions": [],
            "proposal": {"mock": True, "agent": agent_name}
        })

    async def _invoke_sdk(
        self,
        agent_name: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        continue_session: bool = False,
        model_override: Optional[str] = None
    ) -> str:
        """
        Invoke via Claude Agent SDK (provides tools and session continuity).

        This is the preferred backend as it gives agents actual tool access:
        - Read/Edit/Write for file operations
        - Bash for running commands
        - Grep/Glob for searching

        Each agent maintains its own session, enabling conversation continuity
        within a consensus round (leader can continue from previous turn).

        Args:
            model_override: Optional model to use instead of default (e.g., for tiered pricing)
        """
        wrapper = self._get_agent_wrapper(agent_name, model_override)

        # Build full prompt with system context if provided
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n---\n\n{prompt}"

        # Query the agent
        result = await wrapper.query(
            prompt=full_prompt,
            agent_type=agent_name,
            continue_session=continue_session
        )

        # Track token usage
        if result.token_usage:
            self._last_token_usage = {
                "input_tokens": result.token_usage.input_tokens,
                "output_tokens": result.token_usage.output_tokens,
            }
            self.total_tokens["input_tokens"] += result.token_usage.input_tokens
            self.total_tokens["output_tokens"] += result.token_usage.output_tokens

        if not result.success:
            raise RuntimeError(f"Agent SDK invocation failed: {result.error}")

        return result.response

    async def _invoke_anthropic(
        self,
        agent_name: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        agent_settings: Optional[AgentSettings] = None
    ) -> str:
        """Invoke via Anthropic API with optional extended thinking and max context."""
        client = self._get_client()
        if not client:
            raise RuntimeError("Anthropic client not available. Install anthropic package and set API key.")

        messages = [{"role": "user", "content": prompt}]

        # Determine model - allow per-agent override
        model = self.model
        if agent_settings and agent_settings.model_override:
            model = agent_settings.model_override

        # Determine max_tokens - use larger value for max context
        max_tokens = self.max_tokens
        if agent_settings and agent_settings.max_context:
            # For max context, allow larger output
            max_tokens = 16000

        # Build API call kwargs
        api_kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt or f"You are the {agent_name} agent.",
            "messages": messages
        }

        # Add extended thinking if enabled
        if agent_settings and agent_settings.extended_thinking:
            api_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": agent_settings.thinking_budget
            }
            # Extended thinking requires higher max_tokens
            if max_tokens < agent_settings.thinking_budget + 4000:
                api_kwargs["max_tokens"] = agent_settings.thinking_budget + 8000

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.messages.create(**api_kwargs)
        )

        # Track token usage
        if hasattr(response, 'usage') and response.usage:
            self._last_token_usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            # Accumulate totals
            self.total_tokens["input_tokens"] += response.usage.input_tokens
            self.total_tokens["output_tokens"] += response.usage.output_tokens
        else:
            self._last_token_usage = {"input_tokens": 0, "output_tokens": 0}

        # Extract text from response - handle thinking blocks
        result_text = ""
        if response.content:
            for block in response.content:
                if hasattr(block, 'text'):
                    result_text += block.text
        return result_text

    def get_token_usage(self) -> Dict[str, int]:
        """Get total token usage across all invocations."""
        return self.total_tokens.copy()

    async def _invoke_claude_code(self, agent_name: str, prompt: str) -> str:
        """Invoke via Claude Code subprocess."""
        import subprocess

        # Write prompt to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(prompt)
            prompt_file = f.name

        try:
            result = subprocess.run(
                ["claude", "-p", prompt_file, "--output-format", "json"],
                capture_output=True,
                text=True,
                timeout=300
            )
            os.unlink(prompt_file)

            if result.returncode != 0:
                raise RuntimeError(f"Claude Code failed: {result.stderr}")

            return result.stdout
        except subprocess.TimeoutExpired:
            os.unlink(prompt_file)
            raise RuntimeError("Claude Code timed out")


# =============================================================================
# Consensus Orchestrator
# =============================================================================

class ConsensusOrchestrator:
    """
    Generic k-phase commit agreement with parallel evaluation.

    Used by:
    - backlog_grooming.py: Epic→Feature (4 agents), Feature→Task (4 agents)
    - sprint_execution.py: Task implementation (2 agents)

    The orchestrator coordinates:
    1. Leader creates initial proposal
    2. Evaluators critique IN PARALLEL
    3. If all accept → done; else → leader revises with feedback
    4. Repeat for k rounds
    """

    def __init__(
        self,
        config: ConsensusConfig,
        adapter: Any = None,
        agent_interface: Optional[AgentInterface] = None,
        verbose: bool = True
    ):
        """
        Initialize consensus orchestrator.

        Args:
            config: Consensus configuration
            adapter: Work tracking adapter (for attachment operations)
            agent_interface: Interface for invoking agents
            verbose: Whether to print progress messages
        """
        self.config = config
        self.adapter = adapter
        self.agent_interface = agent_interface or AgentInterface()
        self.verbose = verbose

    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message, flush=True)

    async def run_consensus(
        self,
        context: Dict[str, Any],
        leader_system_prompt: Optional[str] = None,
        evaluator_system_prompts: Optional[Dict[str, str]] = None
    ) -> ConsensusResult:
        """
        Execute k-phase consensus.

        Args:
            context: Full context (Epic, Feature, Task info) - NO truncation
            leader_system_prompt: Optional system prompt for leader agent
            evaluator_system_prompts: Optional dict of agent_name -> system prompt

        Returns:
            ConsensusResult with final proposal and round history
        """
        start_time = datetime.now()
        rounds: List[ConsensusRound] = []
        current_proposal: Dict[str, Any] = {}
        artifacts: List[str] = []
        total_tokens = 0

        self._log(f"\n{'='*60}")
        self._log(f"Starting {self.config.max_rounds}-phase consensus")
        self._log(f"Leader: {self.config.leader_agent}")
        self._log(f"Evaluators: {', '.join(self.config.evaluator_agents)}")
        self._log(f"{'='*60}\n")

        for round_num in range(1, self.config.max_rounds + 1):
            round_start = datetime.now()
            self._log(f"\n--- Round {round_num}/{self.config.max_rounds} ---\n")

            # Phase 1: Leader creates (round 1) or revises (round 2+)
            if round_num == 1:
                self._log(f"[{self.config.leader_agent}] Creating initial proposal...")
                leader_prompt = self._build_leader_proposal_prompt(context)
            else:
                self._log(f"[{self.config.leader_agent}] Revising based on feedback...")
                feedback = self._aggregate_feedback(rounds[-1].evaluations)
                leader_prompt = self._build_leader_revision_prompt(
                    context, current_proposal, feedback
                )

            # Get leader agent settings (thinking, max_context)
            leader_settings = self.config.agent_settings.get(self.config.leader_agent)

            leader_response = await self.agent_interface.invoke(
                self.config.leader_agent,
                leader_prompt,
                leader_system_prompt,
                leader_settings
            )
            current_proposal = self._parse_proposal(leader_response)

            # EXTERNAL ENFORCEMENT: Verify claimed artifacts exist before invoking evaluators
            # This prevents wasting tokens on tester evaluation of non-existent files
            verification = self._verify_leader_claims(current_proposal)
            if not verification["verified"]:
                self._log(f"\n  EXTERNAL VERIFICATION FAILED:")
                for issue in verification["issues"]:
                    self._log(f"    - {issue}")

                # Create a synthetic rejection - don't invoke evaluators
                evaluations = [AgentEvaluation(
                    agent_name="external-verification",
                    accepted=False,
                    concerns=verification["issues"],
                    suggestions=["Engineer must actually create the claimed files before tester can verify"],
                    raw_response=""
                )]

                round_duration = (datetime.now() - round_start).total_seconds()
                rounds.append(ConsensusRound(
                    round_number=round_num,
                    leader_proposal=current_proposal,
                    evaluations=evaluations,
                    all_accepted=False,
                    duration_seconds=round_duration
                ))

                if round_num < self.config.max_rounds:
                    self._log(f"\n→ Proceeding to round {round_num + 1} - engineer must create missing files...")
                continue  # Skip to next round without invoking evaluators

            self._log(f"  External verification passed: {verification['summary']}")

            # Save proposal artifact if configured
            if self.config.artifact_dir:
                artifact_path = self._save_artifact(
                    f"round-{round_num}-proposal.json",
                    current_proposal
                )
                artifacts.append(artifact_path)

            # Phase 2: Evaluators critique IN PARALLEL
            self._log(f"\nRunning parallel evaluation with {len(self.config.evaluator_agents)} agents...")
            evaluations = await self._evaluate_parallel(
                current_proposal,
                context,
                evaluator_system_prompts or {}
            )

            # Check if all accepted
            all_accepted = all(e.accepted for e in evaluations)
            round_duration = (datetime.now() - round_start).total_seconds()

            rounds.append(ConsensusRound(
                round_number=round_num,
                leader_proposal=current_proposal,
                evaluations=evaluations,
                all_accepted=all_accepted,
                duration_seconds=round_duration
            ))

            # Log evaluation results
            for eval in evaluations:
                status = "✓ ACCEPTED" if eval.accepted else "✗ CONCERNS"
                self._log(f"  [{eval.agent_name}] {status}")
                if eval.concerns:
                    for concern in eval.concerns[:3]:  # Show first 3
                        self._log(f"    - {concern[:100]}")

            if all_accepted:
                self._log(f"\n✓ Consensus reached in round {round_num}!")
                break
            elif round_num < self.config.max_rounds:
                self._log(f"\n→ Proceeding to round {round_num + 1} with feedback...")

        total_duration = (datetime.now() - start_time).total_seconds()

        # Get total token usage from agent interface
        token_usage = self.agent_interface.get_token_usage()
        total_tokens = token_usage.get("input_tokens", 0) + token_usage.get("output_tokens", 0)

        result = ConsensusResult(
            final_proposal=current_proposal,
            rounds=rounds,
            consensus_reached=rounds[-1].all_accepted if rounds else False,
            artifacts=artifacts,
            total_duration_seconds=total_duration,
            total_tokens_used=total_tokens
        )

        self._log(f"\n{'='*60}")
        self._log(f"Consensus {'REACHED' if result.consensus_reached else 'NOT REACHED'}")
        self._log(f"Total rounds: {len(rounds)}")
        self._log(f"Duration: {total_duration:.1f}s")
        self._log(f"Tokens used: {total_tokens:,} (input: {token_usage.get('input_tokens', 0):,}, output: {token_usage.get('output_tokens', 0):,})")
        self._log(f"{'='*60}\n")

        return result

    def _build_leader_proposal_prompt(self, context: Dict[str, Any]) -> str:
        """Build the initial proposal prompt for the leader agent."""
        context_str = self._format_context(context)

        return f"""{self.config.leader_proposal_prompt}

## Context (FULL - No Truncation)

{context_str}

## Output Format

Respond with a JSON object containing your proposal. Include all required fields
based on the task type (Feature breakdown, Task creation, or Implementation).

IMPORTANT: Your proposal will be reviewed by {len(self.config.evaluator_agents)} specialist agents:
{', '.join(self.config.evaluator_agents)}

They will evaluate for:
{self._format_evaluator_focuses()}

Make your proposal comprehensive and address potential concerns proactively.
"""

    def _build_leader_revision_prompt(
        self,
        context: Dict[str, Any],
        current_proposal: Dict[str, Any],
        feedback: str
    ) -> str:
        """Build the revision prompt for the leader agent."""
        return f"""{self.config.leader_proposal_prompt}

## Your Previous Proposal

```json
{json.dumps(current_proposal, indent=2)}
```

## Evaluator Feedback

The following concerns were raised by specialist agents. You MUST address each one:

{feedback}

## Instructions

Revise your proposal to address ALL concerns raised. For each concern:
1. Explain how you're addressing it (briefly)
2. Update the relevant sections of your proposal

Respond with the COMPLETE revised proposal as JSON (not just the changes).
"""

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context dict as readable string."""
        parts = []

        # Epic context
        if "epic" in context:
            epic = context["epic"]
            parts.append(f"### Epic #{epic.get('id')}: {epic.get('title')}")
            if epic.get('description'):
                parts.append(f"\n{epic['description']}")
            if epic.get('acceptance_criteria'):
                parts.append(f"\n**Acceptance Criteria:**\n{epic['acceptance_criteria']}")

        # Feature context
        if "feature" in context:
            feature = context["feature"]
            parts.append(f"\n### Feature #{feature.get('id')}: {feature.get('title')}")
            if feature.get('description'):
                parts.append(f"\n{feature['description']}")
            if feature.get('acceptance_criteria'):
                parts.append(f"\n**Acceptance Criteria:**\n{feature['acceptance_criteria']}")

        # Work item context (for implementation)
        if "work_item" in context:
            wi = context["work_item"]
            parts.append(f"\n### Work Item #{wi.get('id')}: {wi.get('title')}")
            if wi.get('description'):
                parts.append(f"\n{wi['description']}")

        # Parent chain (for sprint execution)
        if "parent_chain" in context:
            parts.append("\n### Parent Requirements")
            for parent in context["parent_chain"]:
                parts.append(f"\n#### {parent.get('type')} #{parent.get('id')}: {parent.get('title')}")
                if parent.get('description'):
                    parts.append(f"\n{parent['description']}")

        # Design docs
        if "design_docs" in context:
            parts.append("\n### Attached Design Documents")
            for doc in context["design_docs"]:
                parts.append(f"\n#### {doc.get('name')}")
                if doc.get('content'):
                    parts.append(f"\n{doc['content']}")

        return "\n".join(parts)

    def _format_evaluator_focuses(self) -> str:
        """Format evaluator focus areas for display."""
        lines = []
        for agent, focus in self.config.evaluator_prompts.items():
            lines.append(f"- **{agent}**: {focus}")
        return "\n".join(lines)

    async def _evaluate_parallel(
        self,
        proposal: Dict[str, Any],
        context: Dict[str, Any],
        system_prompts: Dict[str, str]
    ) -> List[AgentEvaluation]:
        """Run evaluator agents in parallel using asyncio.gather()."""
        tasks = [
            self._single_evaluation(
                agent,
                proposal,
                context,
                system_prompts.get(agent)
            )
            for agent in self.config.evaluator_agents
        ]
        return await asyncio.gather(*tasks)

    async def _single_evaluation(
        self,
        agent_name: str,
        proposal: Dict[str, Any],
        context: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> AgentEvaluation:
        """
        Run a single evaluator agent.

        Each evaluator reviews the proposal and either:
        - Accepts (no concerns)
        - Raises concerns with specific suggestions
        """
        focus = self.config.evaluator_prompts.get(agent_name, "")

        prompt = f"""## Evaluation Task

You are the {agent_name} agent reviewing a proposal from {self.config.leader_agent}.

### Your Focus Area
{focus}

### Proposal to Review

```json
{json.dumps(proposal, indent=2)}
```

### Context

{self._format_context(context)}

### Instructions

Evaluate the proposal from your perspective ({agent_name}). You MUST respond with JSON:

```json
{{
    "accepted": true/false,
    "concerns": ["list of specific concerns if not accepted"],
    "suggestions": ["list of specific improvements if not accepted"]
}}
```

**Acceptance Criteria:**
- Accept ONLY if the proposal adequately addresses your focus area
- If you have concerns, be SPECIFIC about what needs to change
- Reference concrete parts of the proposal in your feedback
- Do NOT accept proposals with obvious gaps in your area of expertise

**IMPORTANT:** Respond ONLY with the JSON object, no additional text.
"""

        try:
            # Get evaluator agent settings (thinking)
            agent_settings = self.config.agent_settings.get(agent_name)

            response = await self.agent_interface.invoke(
                agent_name,
                prompt,
                system_prompt,
                agent_settings
            )

            # Parse evaluation response
            eval_data = self._parse_evaluation(response)

            return AgentEvaluation(
                agent_name=agent_name,
                accepted=eval_data.get("accepted", False),
                concerns=eval_data.get("concerns", []),
                suggestions=eval_data.get("suggestions", []),
                raw_response=response
            )

        except Exception as e:
            # On error, treat as non-acceptance with error concern
            return AgentEvaluation(
                agent_name=agent_name,
                accepted=False,
                concerns=[f"Evaluation failed: {str(e)}"],
                suggestions=[],
                raw_response=""
            )

    def _parse_proposal(self, response: str) -> Dict[str, Any]:
        """Parse leader's proposal from response text."""
        import re

        # Try multiple strategies to extract JSON from response
        try:
            # Strategy 1: Look for ```json ... ``` block
            if "```json" in response:
                start = response.index("```json") + 7
                end = response.index("```", start)
                json_str = response[start:end].strip()
                return json.loads(json_str)

            # Strategy 2: Look for ``` ... ``` block (might be JSON without annotation)
            elif "```" in response:
                start = response.index("```") + 3
                # Skip any language identifier on the same line
                newline_pos = response.find("\n", start)
                if newline_pos != -1 and newline_pos < start + 20:
                    start = newline_pos + 1
                end = response.index("```", start)
                json_str = response[start:end].strip()
                return json.loads(json_str)

            # Strategy 3: Try parsing entire response as JSON
            else:
                return json.loads(response)

        except (json.JSONDecodeError, ValueError) as first_error:
            # Strategy 4: Try to find any JSON object in the response using regex
            try:
                # Look for the outermost {...} pattern
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    return json.loads(json_str)
            except (json.JSONDecodeError, ValueError):
                pass

            # Strategy 5: Look for nested JSON objects more aggressively
            try:
                # Find positions of first { and last }
                first_brace = response.find('{')
                last_brace = response.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_str = response[first_brace:last_brace + 1]
                    return json.loads(json_str)
            except (json.JSONDecodeError, ValueError):
                pass

            # All strategies failed - return with parse error
            self._log(f"Warning: Failed to parse JSON from leader response")
            self._log(f"Response length: {len(response)} chars")
            return {"raw_proposal": response[:2000], "parse_error": True}

    def _parse_evaluation(self, response: str) -> Dict[str, Any]:
        """Parse evaluator's response into structured evaluation."""
        try:
            # Look for JSON block in markdown
            if "```json" in response:
                start = response.index("```json") + 7
                end = response.index("```", start)
                json_str = response[start:end].strip()
                return json.loads(json_str)
            elif "```" in response:
                start = response.index("```") + 3
                end = response.index("```", start)
                json_str = response[start:end].strip()
                return json.loads(json_str)
            else:
                # Try parsing entire response as JSON
                return json.loads(response)
        except (json.JSONDecodeError, ValueError):
            # Default to non-acceptance if parsing fails
            return {
                "accepted": False,
                "concerns": ["Could not parse evaluation response"],
                "suggestions": []
            }

    def _aggregate_feedback(self, evaluations: List[AgentEvaluation]) -> str:
        """
        Aggregate feedback from evaluators.

        References files by path (efficient), doesn't copy full content.
        """
        parts = []

        for eval in evaluations:
            if not eval.accepted:
                parts.append(f"## {eval.agent_name} Concerns\n")

                if eval.concerns:
                    parts.append("**Issues to Address:**")
                    for i, concern in enumerate(eval.concerns, 1):
                        parts.append(f"{i}. {concern}")

                if eval.suggestions:
                    parts.append("\n**Suggestions:**")
                    for suggestion in eval.suggestions:
                        parts.append(f"- {suggestion}")

                parts.append("")  # Blank line between agents

        return "\n".join(parts)

    def _verify_leader_claims(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        EXTERNAL ENFORCEMENT: Verify claimed artifacts exist before invoking evaluators.

        This prevents wasting tokens on tester evaluation when engineer claims
        to have created files that don't actually exist. The script (not the AI)
        verifies claims against the filesystem.

        Checks:
        1. Files in files_created actually exist
        2. Files in files_modified actually exist
        3. Test files in test_files_created actually exist

        Returns:
            Dict with:
            - verified: bool - True if all claims are valid
            - issues: List[str] - List of verification failures
            - summary: str - Human-readable summary
            - files_checked: int - Number of files checked
        """
        result = {
            "verified": True,
            "issues": [],
            "summary": "",
            "files_checked": 0,
            "files_found": [],
            "files_missing": []
        }

        # Extract file claims from proposal
        impl = proposal.get("implementation", {})
        tests = proposal.get("tests", {})

        files_created = impl.get("files_created", [])
        files_modified = impl.get("files_modified", [])
        test_files = tests.get("test_files_created", [])

        # Combine all claimed files
        all_claimed_files = []
        if isinstance(files_created, list):
            all_claimed_files.extend(files_created)
        if isinstance(files_modified, list):
            all_claimed_files.extend(files_modified)
        if isinstance(test_files, list):
            all_claimed_files.extend(test_files)

        # Filter out empty strings and None values
        all_claimed_files = [f for f in all_claimed_files if f and isinstance(f, str)]

        if not all_claimed_files:
            # No file claims to verify - this might be a design proposal (backlog grooming)
            # or the engineer didn't claim any files yet
            result["summary"] = "No file claims to verify (design proposal or no files claimed)"
            return result

        result["files_checked"] = len(all_claimed_files)

        # Verify each claimed file exists
        for file_path in all_claimed_files:
            try:
                path = Path(file_path)
                # Handle relative paths by checking from cwd
                if not path.is_absolute():
                    path = Path.cwd() / path

                if path.exists():
                    result["files_found"].append(file_path)
                else:
                    result["files_missing"].append(file_path)
                    result["issues"].append(f"File does not exist: {file_path}")
                    result["verified"] = False
            except Exception as e:
                result["issues"].append(f"Error checking file {file_path}: {str(e)}")
                result["verified"] = False

        # Build summary
        found_count = len(result["files_found"])
        missing_count = len(result["files_missing"])

        if result["verified"]:
            result["summary"] = f"All {found_count} claimed files exist"
        else:
            result["summary"] = f"{missing_count}/{result['files_checked']} claimed files missing"

        return result

    def _save_artifact(self, filename: str, content: Any) -> str:
        """Save an artifact to the configured directory."""
        if not self.config.artifact_dir:
            return ""

        artifact_dir = Path(self.config.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        filepath = artifact_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            if isinstance(content, dict):
                json.dump(content, f, indent=2)
            else:
                f.write(str(content))

        return str(filepath)


# =============================================================================
# Synchronous Wrapper
# =============================================================================

def run_consensus_sync(
    config: ConsensusConfig,
    context: Dict[str, Any],
    adapter: Any = None,
    agent_interface: Optional[AgentInterface] = None,
    verbose: bool = True
) -> ConsensusResult:
    """
    Synchronous wrapper for running consensus.

    Use this when not in an async context.

    Args:
        config: Consensus configuration
        context: Full context (Epic, Feature, Task info)
        adapter: Work tracking adapter
        agent_interface: Interface for invoking agents
        verbose: Whether to print progress messages

    Returns:
        ConsensusResult with final proposal and round history
    """
    orchestrator = ConsensusOrchestrator(
        config=config,
        adapter=adapter,
        agent_interface=agent_interface,
        verbose=verbose
    )

    return asyncio.run(orchestrator.run_consensus(context))
