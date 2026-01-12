#!/usr/bin/env python3
"""
Interactive Session Helper for Mode 3 AI Collaboration

Provides reusable helper class for multi-turn AI conversations using either:
- Anthropic SDK: For simple text queries (no tool access needed)
- Claude Agent SDK: For queries requiring tool access (Read, Edit, Bash, etc.)

All workflows can use this for interactive analysis, planning, and collaboration.

Usage:
    from workflow_executor.interactive_session import InteractiveSession

    session = InteractiveSession(
        workflow_name="sprint-planning",
        session_id="sprint-8"
    )

    # Single-turn query (uses Anthropic SDK by default - faster, no tools)
    response = session.ask("Analyze these EPICs for prioritization", context={"epics": [...]})

    # Query with tool access (uses Claude Agent SDK)
    response = session.ask_with_tools("Read the CLAUDE.md file and summarize", tools=["Read", "Glob"])

    # Multi-turn discussion until user approves
    result = session.discuss(
        initial_prompt="Help me prioritize these EPICs",
        context={"epics": [...]},
        max_iterations=5
    )
"""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable


class InteractiveSession:
    """
    Helper for Mode 3 interactive AI sessions using Claude Agent SDK.

    Provides:
    - Single-turn queries with context (Anthropic SDK - fast, no tools)
    - Tool-enabled queries (Claude Agent SDK - Read, Edit, Bash, etc.)
    - Multi-turn discussions with approval gates
    - Session state persistence
    - Conversation history tracking
    - Graceful fallback if SDK unavailable
    """

    def __init__(
        self,
        workflow_name: str,
        session_id: str,
        model: str = "claude-sonnet-4-5",
        max_tokens: int = 4000,
        enable_tools: bool = False,
        allowed_tools: Optional[List[str]] = None
    ):
        """
        Initialize interactive session.

        Args:
            workflow_name: Name of workflow (e.g., "sprint-planning")
            session_id: Unique session ID (e.g., "sprint-8")
            model: Claude model to use
            max_tokens: Maximum tokens per response
            enable_tools: If True, initialize Claude Agent SDK for tool access
            allowed_tools: List of allowed tools (default: read-only)
        """
        self.workflow_name = workflow_name
        self.session_id = session_id
        self.model = model
        self.max_tokens = max_tokens
        self.enable_tools = enable_tools
        self.allowed_tools = allowed_tools or ["Read", "Grep", "Glob"]
        self.conversation_history: List[Dict[str, Any]] = []
        self.start_time = datetime.now()

        # Lazy-load SDK (avoid caching import failures in bytecode)
        self._sdk_checked = False
        self._sdk_available = False
        self._client = None  # Anthropic client, initialized lazily

        # Claude Agent SDK wrapper (for tool-enabled queries)
        self._agent_wrapper = None
        self._agent_sdk_checked = False
        self._agent_sdk_available = False

    def _ensure_sdk_loaded(self) -> bool:
        """
        Lazy-load the SDK on first use.

        Uses importlib.util.find_spec() for a fresh check that isn't affected
        by stale bytecode cache, then imports if available.
        """
        if self._sdk_checked:
            return self._sdk_available

        self._sdk_checked = True

        # Check API key - use KEYCHAIN_ANTHROPIC_API_KEY for automated scripts
        # This keeps ANTHROPIC_API_KEY unset in interactive sessions (uses Max subscription)
        # while automated scripts use the API key (uses API credits)
        api_key = os.environ.get("KEYCHAIN_ANTHROPIC_API_KEY")
        if not api_key:
            print("⚠️  KEYCHAIN_ANTHROPIC_API_KEY not set - interactive mode unavailable")
            self._sdk_available = False
            return False

        # Try to import anthropic SDK
        import importlib.util
        if importlib.util.find_spec("anthropic") is None:
            print("⚠️  anthropic SDK not installed - interactive mode unavailable")
            print("    Install with: pip install anthropic")
            self._sdk_available = False
            return False

        # SDK found, now import and create client
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
            self._sdk_available = True
        except ImportError as e:
            print(f"⚠️  anthropic SDK import failed: {e}")
            self._sdk_available = False
            return False
        except Exception as e:
            print(f"⚠️  Failed to initialize Anthropic client: {e}")
            self._sdk_available = False
            return False

        return True

    def _ensure_agent_sdk_loaded(self) -> bool:
        """
        Lazy-load the Claude Agent SDK for tool-enabled queries.

        This is separate from the Anthropic SDK - the Agent SDK provides
        full tool access (Read, Edit, Bash, etc.) while Anthropic SDK
        only provides text completion.
        """
        if self._agent_sdk_checked:
            return self._agent_sdk_available

        self._agent_sdk_checked = True

        # Check API key
        api_key = os.environ.get("KEYCHAIN_ANTHROPIC_API_KEY")
        if not api_key:
            print("⚠️  KEYCHAIN_ANTHROPIC_API_KEY not set - agent SDK unavailable")
            self._agent_sdk_available = False
            return False

        # Try to import Agent SDK wrapper
        try:
            from scripts.workflow_executor.agent_sdk import AgentSDKWrapper
            self._agent_wrapper = AgentSDKWrapper(
                workflow_name=self.workflow_name,
                allowed_tools=self.allowed_tools,
                model=self.model,
            )
            self._agent_sdk_available = True
        except ImportError as e:
            print(f"⚠️  Claude Agent SDK import failed: {e}")
            print("    Install with: pip install claude-code-sdk")
            self._agent_sdk_available = False
            return False
        except Exception as e:
            print(f"⚠️  Failed to initialize Agent SDK wrapper: {e}")
            self._agent_sdk_available = False
            return False

        return True

    def is_available(self) -> bool:
        """Check if interactive mode is available (SDK + API key)."""
        return self._ensure_sdk_loaded()

    def is_agent_sdk_available(self) -> bool:
        """Check if Agent SDK is available for tool-enabled queries."""
        return self._ensure_agent_sdk_loaded()

    def _run_async(self, coro):
        """Run async coroutine in sync context."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    def ask_with_tools(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        tools: Optional[List[str]] = None,
        continue_session: bool = False,
        on_progress: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """
        Query with tool access using Claude Agent SDK.

        Use this when the AI needs to read files, explore the codebase,
        or perform other actions that require tools.

        Args:
            prompt: User prompt/question
            context: Optional context data (converted to markdown)
            tools: Override allowed tools for this query
            continue_session: If True, continue previous session
            on_progress: Optional callback for progress updates

        Returns:
            {
                "response": str,
                "success": bool,
                "token_usage": Dict,
                "cost_usd": float,
                "session_id": str
            }

        Raises:
            RuntimeError: If Agent SDK unavailable
        """
        if not self._ensure_agent_sdk_loaded():
            raise RuntimeError(
                "Agent SDK unavailable - claude-code-sdk not installed or API key missing"
            )

        # Override tools if specified
        if tools:
            self._agent_wrapper.allowed_tools = tools

        # Build full prompt with context
        full_prompt = self._build_prompt(prompt, context)

        # Run async query
        async def _query():
            return await self._agent_wrapper.query(
                prompt=full_prompt,
                continue_session=continue_session,
            )

        result = self._run_async(_query())

        # Record in conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "ask_with_tools",
            "user_prompt": prompt,
            "context": context,
            "ai_response": result.response if result.success else None,
            "error": result.error if not result.success else None,
            "token_usage": result.token_usage.to_dict() if result.token_usage else {},
            "cost_usd": result.cost_usd,
            "session_id": result.session_id,
        })

        return {
            "response": result.response,
            "success": result.success,
            "token_usage": result.token_usage.to_dict() if result.token_usage else {},
            "cost_usd": result.cost_usd,
            "session_id": result.session_id,
            "issues": result.issues,
            "error": result.error,
        }

    def ask(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: int = 60
    ) -> str:
        """
        Single-turn AI query with context.

        Args:
            prompt: User prompt/question
            context: Optional context data (converted to markdown)
            timeout: Timeout in seconds

        Returns:
            AI response text

        Raises:
            RuntimeError: If SDK unavailable
            TimeoutError: If query times out
        """
        if not self._sdk_available:
            raise RuntimeError(
                "Interactive mode unavailable - anthropic SDK not installed or API key missing"
            )

        # Build full prompt with context
        full_prompt = self._build_prompt(prompt, context)

        # Rate limit retry configuration with Fibonacci backoff
        # Delays: 60s, 80s, 130s, 210s (max)
        fibonacci_delays = [60, 80, 130, 210]
        max_retries = len(fibonacci_delays)

        last_error = None
        for attempt in range(max_retries):
            try:
                # Query using Anthropic SDK
                message = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "user", "content": full_prompt}
                    ]
                )

                # Extract response text
                response = message.content[0].text if message.content else ""

                # Record in conversation history
                self.conversation_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "ask",
                    "user_prompt": prompt,
                    "context": context,
                    "ai_response": response,
                })

                return response

            except Exception as e:
                error_str = str(e)
                last_error = e

                # Check if this is a rate limit error (429)
                if "429" in error_str or "rate_limit" in error_str.lower():
                    if attempt < max_retries - 1:
                        # Use Fibonacci backoff delays
                        delay = fibonacci_delays[attempt]
                        print(f"  ⏳ Rate limit hit. Waiting {delay}s before retry ({attempt + 1}/{max_retries})...")
                        time.sleep(delay)
                        continue
                    else:
                        error_msg = f"AI query failed after {max_retries} retries due to rate limiting: {e}"
                else:
                    # Non-rate-limit error, don't retry
                    error_msg = f"AI query failed: {e}"
                    break

        # Record error in history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "ask",
            "user_prompt": prompt,
            "context": context,
            "error": error_msg,
        })
        raise RuntimeError(error_msg) from last_error

    def discuss(
        self,
        initial_prompt: str,
        context: Optional[Dict[str, Any]] = None,
        max_iterations: int = 5,
        approval_keywords: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Multi-turn discussion until user approves.

        User can refine AI analysis through multiple rounds of feedback.
        Conversation continues until user types approval keyword or max iterations reached.

        Args:
            initial_prompt: Initial prompt to AI
            context: Optional context data
            max_iterations: Maximum conversation turns (prevents infinite loops)
            approval_keywords: Keywords that signal approval (default: ["done", "approve", "approved"])

        Returns:
            {
                "final_response": str,
                "turns": int,
                "approved": bool,
                "conversation": List[Dict]
            }

        Raises:
            RuntimeError: If SDK unavailable
            KeyboardInterrupt: If user cancels (Ctrl+C)
        """
        if not self._sdk_available:
            raise RuntimeError(
                "Interactive mode unavailable - anthropic SDK not installed or API key missing"
            )

        if approval_keywords is None:
            approval_keywords = ["done", "approve", "approved", "finish", "complete"]

        print(f"\n💬 Interactive Discussion: {initial_prompt}\n")
        print(f"Max iterations: {max_iterations}")
        print(f"Type any of {approval_keywords} to finish, or provide feedback to refine.\n")
        print("─" * 70)

        conversation_turns = []
        ai_response = None

        for iteration in range(1, max_iterations + 1):
            print(f"\n🔄 Iteration {iteration}/{max_iterations}")

            # First iteration: use initial prompt
            if iteration == 1:
                prompt = initial_prompt
            else:
                # Subsequent iterations: get user feedback
                print("\nOptions:")
                print(f"  - Type feedback to refine AI response")
                print(f"  - Type {'/'.join(approval_keywords[:3])} to approve and continue")
                print(f"  - Press Ctrl+C to cancel")

                try:
                    user_input = input("\nYour response: ").strip()
                except KeyboardInterrupt:
                    print("\n\n❌ Discussion cancelled by user")
                    raise

                # Check for approval
                if user_input.lower() in approval_keywords:
                    print(f"\n✅ User approved after {iteration - 1} refinement(s)")
                    return {
                        "final_response": ai_response,
                        "turns": len(conversation_turns),
                        "approved": True,
                        "conversation": conversation_turns,
                        "iterations": iteration - 1
                    }

                # Build refinement prompt
                prompt = f"User feedback: {user_input}\n\nPlease refine your previous response based on this feedback."

            # Query AI
            print(f"\n🤖 AI ({self.model}):")
            try:
                ai_response = self.ask(prompt, context=context if iteration == 1 else None)
                print(ai_response)
            except Exception as e:
                print(f"\n❌ AI query failed: {e}")
                raise

            # Record turn
            conversation_turns.append({
                "iteration": iteration,
                "user_input": prompt,
                "ai_response": ai_response,
                "timestamp": datetime.now().isoformat()
            })

            print("\n" + "─" * 70)

        # Max iterations reached
        print(f"\n⚠️  Max iterations ({max_iterations}) reached")
        print("Using AI's final response...")

        return {
            "final_response": ai_response,
            "turns": len(conversation_turns),
            "approved": False,  # User didn't explicitly approve
            "conversation": conversation_turns,
            "iterations": max_iterations,
            "max_iterations_reached": True
        }

    def save_session_state(self, output_dir: Optional[Path] = None) -> Path:
        """
        Save session state to disk for audit/resume.

        Args:
            output_dir: Directory to save state (default: .claude/workflow-state/)

        Returns:
            Path to saved state file
        """
        if output_dir is None:
            output_dir = Path(".claude/workflow-state")

        output_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "workflow_name": self.workflow_name,
            "session_id": self.session_id,
            "model": self.model,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "conversation_history": self.conversation_history,
            "total_turns": len(self.conversation_history)
        }

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{self.workflow_name}-session-{self.session_id}-{timestamp}.json"
        filepath = output_dir / filename

        # Save with UTF-8 encoding (cross-platform compatibility)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        print(f"✓ Session state saved: {filepath}")
        return filepath

    def _build_prompt(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build full prompt with context.

        Args:
            prompt: User prompt
            context: Optional context data

        Returns:
            Full prompt with context section
        """
        parts = []

        if context:
            parts.append("## Context\n")
            for key, value in context.items():
                # Format value based on type
                if isinstance(value, (list, dict)):
                    value_str = json.dumps(value, indent=2)
                else:
                    value_str = str(value)

                parts.append(f"**{key}**:\n```\n{value_str}\n```\n")

            parts.append("")  # Blank line

        parts.append(prompt)

        return "\n".join(parts)

    def _repair_truncated_json(self, json_str: str) -> str:
        """
        Attempt to repair truncated JSON by closing open braces/brackets.

        This handles cases where the AI response was cut off mid-JSON due to
        token limits, leaving unclosed structures or unterminated strings.

        Uses a state-machine approach to properly track string boundaries
        and only count structural characters outside of strings.

        Args:
            json_str: Potentially truncated JSON string

        Returns:
            Repaired JSON string (may still be invalid if too corrupted)
        """
        # State machine to parse JSON structure
        # Track: in_string, escape_next, brace_stack, last_complete_pos
        in_string = False
        escape_next = False
        structure_stack = []  # Stack of '{' and '['
        last_complete_pos = 0  # Last position where we had a complete value

        i = 0
        while i < len(json_str):
            char = json_str[i]

            if escape_next:
                escape_next = False
                i += 1
                continue

            if char == '\\' and in_string:
                escape_next = True
                i += 1
                continue

            if char == '"':
                if in_string:
                    # End of string
                    in_string = False
                    # Check if this is a VALUE string (complete) or KEY string (incomplete)
                    # A value string is followed by , or } or ]
                    # A key string is followed by :
                    rest = json_str[i+1:].lstrip()
                    if not rest or rest[0] in ',}]':
                        # This is a complete value - mark as safe truncation point
                        last_complete_pos = i
                    # If followed by ':', this is a key - not a complete position
                else:
                    in_string = True
                i += 1
                continue

            # Only process structural chars when not in a string
            if not in_string:
                if char == '{':
                    structure_stack.append('{')
                elif char == '[':
                    structure_stack.append('[')
                elif char == '}':
                    if structure_stack and structure_stack[-1] == '{':
                        structure_stack.pop()
                        last_complete_pos = i
                elif char == ']':
                    if structure_stack and structure_stack[-1] == '[':
                        structure_stack.pop()
                        last_complete_pos = i
                elif char in ',':
                    # After a comma, the previous value was complete
                    pass

            i += 1

        # If we ended inside a string or have unclosed structures, truncate to last complete position
        if (in_string or structure_stack) and last_complete_pos > 0:
            json_str = json_str[:last_complete_pos + 1]

        # Recalculate structure stack for (potentially truncated) string
        structure_stack = []
        in_string = False
        escape_next = False
        for char in json_str:
            if escape_next:
                escape_next = False
                continue
            if char == '\\' and in_string:
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if not in_string:
                if char == '{':
                    structure_stack.append('{')
                elif char == '[':
                    structure_stack.append('[')
                elif char == '}' and structure_stack and structure_stack[-1] == '{':
                    structure_stack.pop()
                elif char == ']' and structure_stack and structure_stack[-1] == '[':
                    structure_stack.pop()

        # Remove trailing comma if present (outside strings)
        # Work backwards to find and remove trailing comma
        temp = json_str.rstrip()
        while temp and temp[-1] == ',':
            temp = temp[:-1].rstrip()
        json_str = temp

        # Close any remaining open structures in reverse order
        for struct in reversed(structure_stack):
            if struct == '{':
                json_str += '}'
            elif struct == '[':
                json_str += ']'

        return json_str

    def extract_json_from_response(
        self,
        response: str,
        schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract JSON from AI response (handles markdown code blocks).

        Args:
            response: AI response text
            schema: Optional JSON schema for validation

        Returns:
            Extracted JSON data

        Raises:
            ValueError: If no valid JSON found
            jsonschema.ValidationError: If schema validation fails
        """
        import re

        json_str = None

        # Strategy 1: Try to extract JSON from markdown code blocks (complete or truncated)
        # First try complete block
        json_block_pattern = r'```(?:json)?\s*\n(.*?)\n```'
        matches = re.findall(json_block_pattern, response, re.DOTALL)
        if matches:
            json_str = matches[0]

        # If no complete block, try truncated (no closing ```)
        if json_str is None:
            truncated_pattern = r'```(?:json)?\s*\n(\{.*)'
            truncated_match = re.search(truncated_pattern, response, re.DOTALL)
            if truncated_match:
                json_str = truncated_match.group(1)

        # Strategy 2: Find any JSON object in the response
        if json_str is None:
            json_obj_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            obj_matches = re.findall(json_obj_pattern, response, re.DOTALL)
            if obj_matches:
                # Try the largest match first
                for match in sorted(obj_matches, key=len, reverse=True):
                    try:
                        json.loads(match)
                        json_str = match
                        break
                    except json.JSONDecodeError:
                        continue

        # Strategy 3: No code block, try parsing entire response
        if json_str is None:
            json_str = response.strip()

        # Try to parse JSON, with repair on failure
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as first_error:
            # Parsing failed - try to repair truncated JSON
            repaired = self._repair_truncated_json(json_str)
            try:
                data = json.loads(repaired)
            except json.JSONDecodeError as repair_error:
                # Repair didn't work - raise original error with context
                raise ValueError(f"Failed to parse JSON from response: {first_error}\n\nResponse:\n{response[:500]}...")

        # Validate against schema if provided
        if schema:
            try:
                from jsonschema import validate
                validate(instance=data, schema=schema)
            except ImportError:
                print("⚠️  jsonschema not installed - skipping schema validation")
            except Exception as e:
                raise ValueError(f"JSON schema validation failed: {e}\n\nData:\n{json.dumps(data, indent=2)}")

        return data

    def get_token_usage_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get token usage summary from Agent SDK wrapper.

        Returns None if Agent SDK was not used.
        """
        if self._agent_wrapper:
            return self._agent_wrapper.get_usage_summary()
        return None

    def print_token_usage_summary(self):
        """Print formatted token usage summary if Agent SDK was used."""
        if self._agent_wrapper:
            self._agent_wrapper.print_usage_summary()
        else:
            print("(No Agent SDK usage to report)")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get session summary statistics.

        Returns:
            {
                "total_turns": int,
                "duration_seconds": float,
                "conversation_history": List[Dict],
                "agent_sdk_usage": Optional[Dict]
            }
        """
        duration = (datetime.now() - self.start_time).total_seconds()

        summary = {
            "workflow_name": self.workflow_name,
            "session_id": self.session_id,
            "total_turns": len(self.conversation_history),
            "duration_seconds": duration,
            "conversation_history": self.conversation_history,
        }

        # Include Agent SDK usage if available
        if self._agent_wrapper:
            summary["agent_sdk_usage"] = self._agent_wrapper.get_usage_summary()

        return summary
