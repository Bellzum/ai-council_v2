"""
Prompt Caching Strategy for Claude Agent SDK

Implements efficient prompt caching to minimize token costs:
- System prompts cached for 5 minutes
- Agent definitions cached between calls
- Project context (CLAUDE.md) cached
- Cache breakpoint strategy for optimal reuse

Caching significantly reduces costs:
- First call: 1.25x cost (cache creation)
- Subsequent calls: 0.10x cost (cache reads)
- ~90% savings on stable content

Usage:
    builder = PromptCacheBuilder(workflow_name="sprint-execution")

    # Add cacheable content with breakpoints
    builder.add_system_prompt("You are a software engineer...")
    builder.add_agent_definition(agent_def_content)
    builder.add_project_context(claude_md_content)

    # Add variable content (not cached)
    builder.add_work_item_context(work_item)
    builder.add_instructions(specific_instructions)

    # Get the optimized prompt
    prompt = builder.build()

See: /home/sundance/.claude/plans/claude-agent-sdk-migration.md
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CacheableBlock:
    """A block of content that can be cached."""

    content: str
    cache_key: str
    block_type: str  # system_prompt, agent_def, project_context, etc.
    priority: int = 0  # Higher priority = cached first
    min_tokens: int = 1024  # Minimum tokens for caching to be effective

    def estimated_tokens(self) -> int:
        """Estimate token count (rough: ~4 chars per token)."""
        return len(self.content) // 4


@dataclass
class CacheMetrics:
    """Metrics for cache effectiveness."""

    cache_hits: int = 0
    cache_misses: int = 0
    tokens_saved: int = 0
    cost_saved_usd: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total


class PromptCacheBuilder:
    """
    Builds prompts with optimal cache breakpoint strategy.

    The builder organizes content in cacheable order:
    1. System prompt (stable, cached)
    2. Agent definition (stable per agent type, cached)
    3. Project context (stable, cached)
    4. Work item context (variable, not cached)
    5. Instructions (variable, not cached)

    This ordering maximizes cache hits since stable content
    comes first and can be reused across calls.
    """

    # Token costs for pricing estimates
    COST_PER_1K_TOKENS = {
        "input": 0.003,  # $3 per million
        "cache_write": 0.00375,  # 1.25x input
        "cache_read": 0.0003,  # 0.1x input
    }

    def __init__(self, workflow_name: str):
        """
        Initialize the prompt builder.

        Args:
            workflow_name: Name of the workflow (for cache key namespacing)
        """
        self.workflow_name = workflow_name
        self.blocks: List[CacheableBlock] = []
        self.variable_content: List[str] = []
        self.metrics = CacheMetrics()

    def add_system_prompt(self, content: str) -> "PromptCacheBuilder":
        """
        Add system prompt (highest cache priority).

        System prompts are stable across all calls and should be
        cached for maximum efficiency.

        Args:
            content: System prompt content

        Returns:
            self for chaining
        """
        self.blocks.append(
            CacheableBlock(
                content=content,
                cache_key=self._generate_cache_key("system", content),
                block_type="system_prompt",
                priority=100,
            )
        )
        return self

    def add_agent_definition(
        self, content: str, agent_type: str = "default"
    ) -> "PromptCacheBuilder":
        """
        Add agent definition (high cache priority).

        Agent definitions are stable per agent type and can be
        cached for reuse within the same workflow.

        Args:
            content: Agent definition content
            agent_type: Type of agent (for cache key)

        Returns:
            self for chaining
        """
        self.blocks.append(
            CacheableBlock(
                content=content,
                cache_key=self._generate_cache_key(f"agent_{agent_type}", content),
                block_type="agent_definition",
                priority=90,
            )
        )
        return self

    def add_project_context(self, content: str) -> "PromptCacheBuilder":
        """
        Add project context like CLAUDE.md (medium cache priority).

        Project context is stable within a project and can be
        cached across many calls.

        Args:
            content: Project context content

        Returns:
            self for chaining
        """
        self.blocks.append(
            CacheableBlock(
                content=content,
                cache_key=self._generate_cache_key("project", content),
                block_type="project_context",
                priority=80,
            )
        )
        return self

    def add_tool_definitions(self, tools: List[Dict[str, Any]]) -> "PromptCacheBuilder":
        """
        Add tool definitions (medium cache priority).

        Tool definitions are stable and can be cached.

        Args:
            tools: List of tool definitions

        Returns:
            self for chaining
        """
        content = json.dumps(tools, indent=2)
        self.blocks.append(
            CacheableBlock(
                content=content,
                cache_key=self._generate_cache_key("tools", content),
                block_type="tool_definitions",
                priority=70,
            )
        )
        return self

    def add_work_item_context(self, context: Dict[str, Any]) -> "PromptCacheBuilder":
        """
        Add work item context (variable, not cached).

        Work item context changes per call and is not cached.

        Args:
            context: Work item data

        Returns:
            self for chaining
        """
        self.variable_content.append(self._format_work_item(context))
        return self

    def add_instructions(self, instructions: str) -> "PromptCacheBuilder":
        """
        Add specific instructions (variable, not cached).

        Instructions are specific to each call and not cached.

        Args:
            instructions: Instruction text

        Returns:
            self for chaining
        """
        self.variable_content.append(instructions)
        return self

    def add_variable_content(self, content: str) -> "PromptCacheBuilder":
        """
        Add any variable content that should not be cached.

        Args:
            content: Variable content

        Returns:
            self for chaining
        """
        self.variable_content.append(content)
        return self

    def build(self) -> str:
        """
        Build the final prompt with optimal ordering.

        Returns:
            Complete prompt with cacheable content first
        """
        # Sort blocks by priority (highest first)
        sorted_blocks = sorted(self.blocks, key=lambda b: b.priority, reverse=True)

        # Build prompt sections
        sections = []

        # Add cacheable blocks first (for maximum cache effectiveness)
        for block in sorted_blocks:
            sections.append(f"<!-- cache:{block.block_type} -->\n{block.content}")

        # Add separator before variable content
        if self.variable_content:
            sections.append("<!-- variable content below -->")

        # Add variable content
        sections.extend(self.variable_content)

        return "\n\n".join(sections)

    def build_with_markers(self) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Build prompt with cache breakpoint markers.

        Returns tuple of (prompt, breakpoints) where breakpoints
        indicate where cache boundaries should be placed.

        Returns:
            (prompt, breakpoints) tuple
        """
        sorted_blocks = sorted(self.blocks, key=lambda b: b.priority, reverse=True)

        sections = []
        breakpoints = []
        current_pos = 0

        for i, block in enumerate(sorted_blocks):
            section = f"{block.content}\n"
            sections.append(section)

            # Add breakpoint after each cacheable block
            breakpoints.append(
                {
                    "position": current_pos + len(section),
                    "type": block.block_type,
                    "cache_key": block.cache_key,
                    "estimated_tokens": block.estimated_tokens(),
                }
            )
            current_pos += len(section)

        # Add variable content
        for content in self.variable_content:
            sections.append(content)

        return "\n".join(sections), breakpoints

    def estimate_savings(
        self, is_first_call: bool = False
    ) -> Dict[str, Any]:
        """
        Estimate token savings from caching.

        Args:
            is_first_call: True if this is the first call (cache creation)

        Returns:
            Dictionary with estimated savings
        """
        cacheable_tokens = sum(b.estimated_tokens() for b in self.blocks)
        variable_tokens = sum(len(c) // 4 for c in self.variable_content)
        total_tokens = cacheable_tokens + variable_tokens

        if is_first_call:
            # First call: pay cache creation cost
            cost = (
                cacheable_tokens * self.COST_PER_1K_TOKENS["cache_write"] / 1000
                + variable_tokens * self.COST_PER_1K_TOKENS["input"] / 1000
            )
            savings = 0
        else:
            # Subsequent calls: pay cache read cost
            cost = (
                cacheable_tokens * self.COST_PER_1K_TOKENS["cache_read"] / 1000
                + variable_tokens * self.COST_PER_1K_TOKENS["input"] / 1000
            )
            # Compare to no-cache cost
            no_cache_cost = total_tokens * self.COST_PER_1K_TOKENS["input"] / 1000
            savings = no_cache_cost - cost

        return {
            "cacheable_tokens": cacheable_tokens,
            "variable_tokens": variable_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": cost,
            "estimated_savings_usd": savings,
            "is_first_call": is_first_call,
        }

    def _generate_cache_key(self, prefix: str, content: str) -> str:
        """Generate a cache key from content hash."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        return f"{self.workflow_name}:{prefix}:{content_hash}"

    def _format_work_item(self, context: Dict[str, Any]) -> str:
        """Format work item context as markdown."""
        lines = [f"## Work Item #{context.get('id', 'N/A')}"]

        if "title" in context:
            lines.append(f"**Title**: {context['title']}")

        if "type" in context:
            lines.append(f"**Type**: {context['type']}")

        if "description" in context:
            lines.append(f"\n### Description\n{context['description']}")

        if "acceptance_criteria" in context:
            lines.append(f"\n### Acceptance Criteria\n{context['acceptance_criteria']}")

        if "attachments" in context and context["attachments"]:
            lines.append("\n### Attachments")
            for att in context["attachments"]:
                lines.append(f"- {att.get('name', 'Unknown')}: {att.get('url', '')}")

        return "\n".join(lines)


class CacheRegistry:
    """
    Tracks cache usage across workflow executions.

    Provides insights into cache effectiveness and helps
    optimize caching strategy.
    """

    REGISTRY_FILE = Path(".claude/workflow-state/cache-registry.json")

    def __init__(self):
        """Initialize the cache registry."""
        self.REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.entries: Dict[str, Dict[str, Any]] = {}
        self._load()

    def record_cache_event(
        self,
        cache_key: str,
        event_type: str,  # "create", "hit", "miss"
        tokens: int,
        cost_usd: float,
    ) -> None:
        """
        Record a cache event.

        Args:
            cache_key: The cache key
            event_type: Type of event (create, hit, miss)
            tokens: Number of tokens involved
            cost_usd: Cost of this operation
        """
        if cache_key not in self.entries:
            self.entries[cache_key] = {
                "created_at": datetime.now().isoformat(),
                "hits": 0,
                "misses": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
            }

        entry = self.entries[cache_key]

        if event_type == "create":
            entry["created_at"] = datetime.now().isoformat()
        elif event_type == "hit":
            entry["hits"] += 1
        elif event_type == "miss":
            entry["misses"] += 1

        entry["total_tokens"] += tokens
        entry["total_cost_usd"] += cost_usd
        entry["last_used"] = datetime.now().isoformat()

        self._save()

    def get_stats(self) -> Dict[str, Any]:
        """Get overall cache statistics."""
        total_hits = sum(e["hits"] for e in self.entries.values())
        total_misses = sum(e["misses"] for e in self.entries.values())
        total_cost = sum(e["total_cost_usd"] for e in self.entries.values())

        hit_rate = total_hits / max(total_hits + total_misses, 1)

        return {
            "cache_key_count": len(self.entries),
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate": hit_rate,
            "total_cost_usd": total_cost,
        }

    def get_top_cached(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top cached content by hit count."""
        sorted_entries = sorted(
            [
                {"cache_key": k, **v}
                for k, v in self.entries.items()
            ],
            key=lambda x: x["hits"],
            reverse=True,
        )
        return sorted_entries[:limit]

    def cleanup_expired(self, max_age_hours: int = 24) -> int:
        """
        Remove cache entries older than max_age_hours.

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of entries removed
        """
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        removed = 0

        keys_to_remove = []
        for key, entry in self.entries.items():
            last_used = entry.get("last_used") or entry.get("created_at")
            if last_used:
                try:
                    last_time = datetime.fromisoformat(last_used)
                    if last_time < cutoff:
                        keys_to_remove.append(key)
                except ValueError:
                    pass

        for key in keys_to_remove:
            del self.entries[key]
            removed += 1

        if removed > 0:
            self._save()

        return removed

    def _load(self) -> None:
        """Load registry from file."""
        if self.REGISTRY_FILE.exists():
            try:
                self.entries = json.loads(
                    self.REGISTRY_FILE.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError:
                self.entries = {}

    def _save(self) -> None:
        """Save registry to file."""
        self.REGISTRY_FILE.write_text(
            json.dumps(self.entries, indent=2), encoding="utf-8"
        )


def load_agent_definition(agent_type: str) -> str:
    """
    Load agent definition from rendered agents directory.

    Args:
        agent_type: Type of agent (e.g., "engineer", "tester")

    Returns:
        Agent definition content
    """
    agent_path = Path(f".claude/agents/{agent_type}.md")
    if agent_path.exists():
        return agent_path.read_text(encoding="utf-8")

    # Fallback to templates
    template_path = Path(f"agents/templates/{agent_type}.j2")
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")

    raise FileNotFoundError(f"Agent definition not found: {agent_type}")


def load_project_context() -> str:
    """
    Load project context (CLAUDE.md files).

    Returns:
        Combined project context
    """
    context_parts = []

    # Load main CLAUDE.md
    main_claude = Path("CLAUDE.md")
    if main_claude.exists():
        context_parts.append(main_claude.read_text(encoding="utf-8"))

    # Load .claude/CLAUDE.md if exists
    dot_claude = Path(".claude/CLAUDE.md")
    if dot_claude.exists():
        context_parts.append(dot_claude.read_text(encoding="utf-8"))

    return "\n\n---\n\n".join(context_parts)
