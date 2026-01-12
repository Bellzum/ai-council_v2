"""
Claude Agent SDK Wrapper

Provides a unified interface for invoking Claude Code agents with:
- Session continuity between calls (resume previous conversations)
- Token usage tracking and cost monitoring
- Automatic prompt caching for efficiency
- Multi-turn session support for complex workflows

This replaces direct subprocess calls to `claude --print` with the
programmatic Claude Agent SDK, enabling better context management
and significant token savings through session reuse.

Usage:
    # Single query with tool access
    wrapper = AgentSDKWrapper(
        workflow_name="sprint-execution",
        allowed_tools=["Read", "Edit", "Bash", "Grep", "Glob"]
    )
    result = await wrapper.query("Implement feature X")

    # Multi-turn session (engineer → tester handoff)
    async with wrapper.multi_turn_session() as session:
        eng_result = await session.query(engineer_prompt)
        test_result = await session.query(tester_prompt)  # Has full context!

See: /home/sundance/.claude/plans/claude-agent-sdk-migration.md
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

# Import Claude Agent SDK
from claude_code_sdk import (
    query as sdk_query,
    ClaudeCodeOptions,
    ClaudeSDKClient,
    ResultMessage,
    SystemMessage,
    AssistantMessage,
    UserMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
)


@dataclass
class TokenUsage:
    """Token usage metrics from an agent call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def total_input(self) -> int:
        """Total input tokens including cache."""
        return self.input_tokens + self.cache_read_tokens + self.cache_creation_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Percentage of input tokens from cache (0.0 to 1.0)."""
        total_cached = self.cache_read_tokens + self.cache_creation_tokens
        if total_cached == 0:
            return 0.0
        return self.cache_read_tokens / total_cached

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "total_input": self.total_input,
        }


@dataclass
class AgentResult:
    """Result from an agent execution."""

    success: bool
    response: str
    session_id: Optional[str] = None
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    duration_ms: int = 0
    num_turns: int = 0
    issues: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "response": self.response,
            "session_id": self.session_id,
            "token_usage": self.token_usage.to_dict(),
            "cost_usd": self.cost_usd,
            "duration_ms": self.duration_ms,
            "num_turns": self.num_turns,
            "issues": self.issues,
            "error": self.error,
        }


class AgentSDKWrapper:
    """
    Wrapper for Claude Agent SDK with session management and metrics.

    Provides:
    - Unified interface for agent invocation
    - Session continuity (resume previous conversations)
    - Cumulative token usage tracking
    - Automatic API key handling
    - Configurable tool access

    Example:
        wrapper = AgentSDKWrapper(
            workflow_name="sprint-execution",
            allowed_tools=["Read", "Edit", "Bash"]
        )

        # First call - creates new session
        result1 = await wrapper.query("Implement login feature")
        print(f"Session: {result1.session_id}")

        # Second call - continues same session
        result2 = await wrapper.query(
            "Now test the implementation",
            continue_session=True
        )
    """

    # Default tools for different workflow types
    TOOL_PRESETS = {
        "read_only": ["Read", "Grep", "Glob"],
        "implementation": ["Read", "Edit", "Write", "Bash", "Grep", "Glob"],
        "testing": ["Read", "Edit", "Write", "Bash", "Grep", "Glob"],
        "analysis": ["Read", "Grep", "Glob", "WebSearch"],
    }

    def __init__(
        self,
        workflow_name: str,
        allowed_tools: Optional[List[str]] = None,
        tool_preset: Optional[str] = None,
        max_turns: int = 10,
        model: str = "claude-sonnet-4-5",
        cwd: Optional[Path] = None,
        api_key_env: str = "KEYCHAIN_ANTHROPIC_API_KEY",
    ):
        """
        Initialize the SDK wrapper.

        Args:
            workflow_name: Name of the workflow (for tracking)
            allowed_tools: List of allowed tools, or use tool_preset
            tool_preset: One of 'read_only', 'implementation', 'testing', 'analysis'
            max_turns: Maximum conversation turns per query
            model: Model to use (default: claude-sonnet-4-5)
            cwd: Working directory for file operations
            api_key_env: Environment variable for API key
        """
        self.workflow_name = workflow_name
        self.max_turns = max_turns
        self.model = model
        self.cwd = cwd or Path.cwd()
        self.api_key_env = api_key_env

        # Set allowed tools from preset or explicit list
        if allowed_tools:
            self.allowed_tools = allowed_tools
        elif tool_preset and tool_preset in self.TOOL_PRESETS:
            self.allowed_tools = self.TOOL_PRESETS[tool_preset]
        else:
            self.allowed_tools = self.TOOL_PRESETS["read_only"]

        # Session tracking
        self.session_id: Optional[str] = None
        self.session_history: List[Dict[str, Any]] = []

        # Cumulative token usage for the workflow
        self.cumulative_usage = TokenUsage()
        self.cumulative_cost_usd: float = 0.0
        self.call_count: int = 0

        # Tracking by agent type (if used)
        self.usage_by_agent: Dict[str, Dict[str, Any]] = {}

    def _get_env(self) -> Dict[str, str]:
        """Get environment with API key set."""
        env = os.environ.copy()
        api_key = os.environ.get(self.api_key_env)
        if api_key:
            env["ANTHROPIC_API_KEY"] = api_key
        return env

    async def query(
        self,
        prompt: str,
        agent_type: Optional[str] = None,
        continue_session: bool = False,
        fork_session: bool = False,
        system_prompt: Optional[str] = None,
        max_turns: Optional[int] = None,
        on_message: Optional[Callable[[Any], None]] = None,
    ) -> AgentResult:
        """
        Execute an agent query.

        Args:
            prompt: The task/question for the agent
            agent_type: Optional agent type for tracking (e.g., "engineer", "tester")
            continue_session: If True, resume previous session
            fork_session: If True, fork from previous session (new branch)
            system_prompt: Optional system prompt override
            max_turns: Override max_turns for this query
            on_message: Optional callback for each message (for progress display)

        Returns:
            AgentResult with response, metrics, and session ID
        """
        start_time = datetime.now()

        # Build options
        options = ClaudeCodeOptions(
            model=self.model,
            max_turns=max_turns or self.max_turns,
            allowed_tools=self.allowed_tools,
            permission_mode="acceptEdits",
            cwd=str(self.cwd),
            env=self._get_env(),
        )

        # Session management
        if continue_session and self.session_id:
            options.resume = self.session_id
        if fork_session and self.session_id:
            options.resume = self.session_id
            # Note: fork behavior is handled by SDK

        # System prompt
        if system_prompt:
            options.system_prompt = system_prompt

        # Collect response
        response_text = ""
        new_session_id = None
        result_message: Optional[ResultMessage] = None

        try:
            async for message in sdk_query(prompt=prompt, options=options):
                # Call progress callback if provided
                if on_message:
                    on_message(message)

                # Extract session ID from system message
                if isinstance(message, SystemMessage):
                    if hasattr(message, "session_id"):
                        new_session_id = message.session_id

                # Collect text content from assistant messages
                elif isinstance(message, AssistantMessage):
                    if hasattr(message, "content"):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                response_text += block.text + "\n"

                # Capture result message with metrics
                elif isinstance(message, ResultMessage):
                    result_message = message
                    if hasattr(message, "session_id"):
                        new_session_id = message.session_id
                    if hasattr(message, "result") and message.result:
                        response_text = message.result

        except Exception as e:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            return AgentResult(
                success=False,
                response="",
                error=str(e),
                duration_ms=duration_ms,
            )

        # Update session ID
        if new_session_id:
            self.session_id = new_session_id

        # Extract metrics from result message
        token_usage = TokenUsage()
        cost_usd = 0.0
        duration_ms = 0
        num_turns = 0

        if result_message:
            duration_ms = getattr(result_message, "duration_ms", 0)
            num_turns = getattr(result_message, "num_turns", 0)
            cost_usd = getattr(result_message, "total_cost_usd", 0.0) or 0.0

            usage_dict = getattr(result_message, "usage", {}) or {}
            token_usage = TokenUsage(
                input_tokens=usage_dict.get("input_tokens", 0),
                output_tokens=usage_dict.get("output_tokens", 0),
                cache_read_tokens=usage_dict.get("cache_read_input_tokens", 0),
                cache_creation_tokens=usage_dict.get("cache_creation_input_tokens", 0),
            )

        # Update cumulative metrics
        self._update_cumulative_metrics(token_usage, cost_usd, agent_type)

        # Record in session history
        self.session_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "agent_type": agent_type,
                "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "token_usage": token_usage.to_dict(),
                "cost_usd": cost_usd,
                "session_id": self.session_id,
            }
        )

        # Extract issues from response
        issues = self._extract_issues(response_text)

        return AgentResult(
            success=True,
            response=response_text,
            session_id=self.session_id,
            token_usage=token_usage,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
            num_turns=num_turns,
            issues=issues,
        )

    def _update_cumulative_metrics(
        self, usage: TokenUsage, cost: float, agent_type: Optional[str]
    ):
        """Update cumulative token usage metrics."""
        self.cumulative_usage.input_tokens += usage.input_tokens
        self.cumulative_usage.output_tokens += usage.output_tokens
        self.cumulative_usage.cache_read_tokens += usage.cache_read_tokens
        self.cumulative_usage.cache_creation_tokens += usage.cache_creation_tokens
        self.cumulative_cost_usd += cost
        self.call_count += 1

        # Track by agent type
        if agent_type:
            if agent_type not in self.usage_by_agent:
                self.usage_by_agent[agent_type] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                }
            self.usage_by_agent[agent_type]["calls"] += 1
            self.usage_by_agent[agent_type]["input_tokens"] += usage.input_tokens
            self.usage_by_agent[agent_type]["output_tokens"] += usage.output_tokens
            self.usage_by_agent[agent_type]["cost_usd"] += cost

    def _extract_issues(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract issues from agent response by parsing JSON blocks.

        Agents are prompted to include a JSON block with an "issues" array.
        """
        issues = []

        # Look for JSON blocks in the response
        json_pattern = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
        matches = re.findall(json_pattern, response)

        for match in matches:
            try:
                data = json.loads(match)
                if isinstance(data, dict) and "issues" in data:
                    issues.extend(data["issues"])
            except json.JSONDecodeError:
                continue

        return issues

    async def multi_turn_session(self) -> "MultiTurnSession":
        """
        Create a multi-turn session for continuous conversation.

        Use this when multiple agents need shared context:
        - Engineer → Tester handoff
        - Analyst → Architect → Engineer pipeline

        Example:
            async with wrapper.multi_turn_session() as session:
                eng_result = await session.query("Implement feature")
                test_result = await session.query("Test the implementation")
        """
        return MultiTurnSession(self)

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get summary of token usage for the workflow."""
        return {
            "workflow_name": self.workflow_name,
            "call_count": self.call_count,
            "total_tokens": {
                "input": self.cumulative_usage.input_tokens,
                "output": self.cumulative_usage.output_tokens,
                "cache_read": self.cumulative_usage.cache_read_tokens,
                "cache_creation": self.cumulative_usage.cache_creation_tokens,
            },
            "total_cost_usd": self.cumulative_cost_usd,
            "cache_hit_rate": self.cumulative_usage.cache_hit_rate,
            "by_agent": self.usage_by_agent,
            "session_history": self.session_history,
        }

    def print_usage_summary(self):
        """Print formatted token usage summary."""
        summary = self.get_usage_summary()

        print("\n" + "=" * 70)
        print(" 📊 TOKEN USAGE SUMMARY")
        print("=" * 70)

        print(f"\n Workflow: {summary['workflow_name']}")
        print(f" Total Calls: {summary['call_count']}")

        totals = summary["total_tokens"]
        print(f"\n Total Tokens:")
        print(f"   Input:  {totals['input']:,}")
        print(f"   Output: {totals['output']:,}")
        print(f"   Cache Read: {totals['cache_read']:,}")
        print(f"   Cache Created: {totals['cache_creation']:,}")

        print(f"\n💰 Total Cost: ${summary['total_cost_usd']:.4f}")
        print(f" Cache Hit Rate: {summary['cache_hit_rate']:.1%}")

        if summary["by_agent"]:
            print(f"\n By Agent:")
            for agent, stats in summary["by_agent"].items():
                print(
                    f"   {agent.title()}: {stats['calls']} calls, "
                    f"{stats['input_tokens']:,} in / {stats['output_tokens']:,} out, "
                    f"${stats['cost_usd']:.4f}"
                )


class MultiTurnSession:
    """
    Manages multi-turn conversations with full context preservation.

    This enables workflows where multiple agents share context:
    - Engineer implements, then Tester tests (tester sees engineer's work)
    - Analyst prioritizes, Architect designs, Engineer implements

    The session maintains full conversation history, so each subsequent
    query has access to all previous context without re-sending.
    """

    def __init__(self, wrapper: AgentSDKWrapper):
        self.wrapper = wrapper
        self.turn_count = 0
        self.messages: List[Dict[str, Any]] = []

    async def __aenter__(self):
        """Enter the session context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the session context."""
        # Nothing to clean up - session persists via session_id
        pass

    async def query(
        self,
        prompt: str,
        agent_type: Optional[str] = None,
        on_message: Optional[Callable[[Any], None]] = None,
    ) -> AgentResult:
        """
        Send a message in the multi-turn session.

        Each query continues the previous conversation, maintaining
        full context from prior turns.

        Args:
            prompt: The task/question
            agent_type: Optional agent type for tracking
            on_message: Optional callback for progress display

        Returns:
            AgentResult with response and metrics
        """
        # First turn creates session, subsequent turns continue it
        continue_session = self.turn_count > 0

        result = await self.wrapper.query(
            prompt=prompt,
            agent_type=agent_type,
            continue_session=continue_session,
            on_message=on_message,
        )

        self.turn_count += 1
        self.messages.append(
            {
                "turn": self.turn_count,
                "agent_type": agent_type,
                "prompt_preview": prompt[:100],
                "success": result.success,
            }
        )

        return result

    @property
    def session_id(self) -> Optional[str]:
        """Get the current session ID."""
        return self.wrapper.session_id


# Convenience function for simple one-off queries
async def invoke_agent(
    prompt: str,
    allowed_tools: Optional[List[str]] = None,
    model: str = "claude-sonnet-4-5",
    max_turns: int = 10,
) -> AgentResult:
    """
    Simple function to invoke an agent for a one-off task.

    For more complex workflows with session management, use AgentSDKWrapper.

    Args:
        prompt: The task/question
        allowed_tools: Tools the agent can use
        model: Model to use
        max_turns: Maximum conversation turns

    Returns:
        AgentResult with response and metrics
    """
    wrapper = AgentSDKWrapper(
        workflow_name="one-off",
        allowed_tools=allowed_tools,
        model=model,
        max_turns=max_turns,
    )
    return await wrapper.query(prompt)


def run_agent_sync(
    prompt: str,
    allowed_tools: Optional[List[str]] = None,
    model: str = "claude-sonnet-4-5",
    max_turns: int = 10,
) -> AgentResult:
    """
    Synchronous wrapper for invoke_agent.

    Use this when you need to call from synchronous code.

    Args:
        prompt: The task/question
        allowed_tools: Tools the agent can use
        model: Model to use
        max_turns: Maximum conversation turns

    Returns:
        AgentResult with response and metrics
    """
    return asyncio.run(
        invoke_agent(
            prompt=prompt,
            allowed_tools=allowed_tools,
            model=model,
            max_turns=max_turns,
        )
    )
