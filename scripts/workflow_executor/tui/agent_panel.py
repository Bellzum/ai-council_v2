"""
Agent activity panel for multi-agent workflow visualization.

Shows:
- Active agents with spinners and status messages
- Agent states (active, waiting, queued, complete)
- Consensus visualization when multiple agents collaborate
- Token usage per agent
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any

from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class AgentStatus(Enum):
    """Agent execution status."""

    QUEUED = "queued"
    ACTIVE = "active"
    WAITING = "waiting"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class AgentState:
    """State of an individual agent."""

    agent_type: str
    status: AgentStatus = AgentStatus.QUEUED
    message: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    tokens_input: int = 0
    tokens_output: int = 0
    result_summary: Optional[str] = None
    confidence: Optional[str] = None  # HIGH, MEDIUM, LOW

    @property
    def elapsed(self) -> Optional[timedelta]:
        """Get elapsed time for active/complete agents."""
        if not self.started_at:
            return None
        end_time = self.completed_at or datetime.now()
        return end_time - self.started_at

    @property
    def elapsed_str(self) -> str:
        """Format elapsed time as string."""
        if not self.elapsed:
            return ""
        total_seconds = self.elapsed.total_seconds()
        if total_seconds < 60:
            return f"{total_seconds:.1f}s"
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        return f"{minutes}m {seconds}s"


@dataclass
class ConsensusState:
    """State of multi-agent consensus."""

    topic: str
    agents: Dict[str, Any] = field(default_factory=dict)  # agent_type -> value
    confidences: Dict[str, str] = field(default_factory=dict)  # agent_type -> HIGH/MEDIUM/LOW
    reached: bool = False
    final_value: Optional[Any] = None
    variance: Optional[float] = None


class AgentPanel:
    """
    Renders agent activity panel with friendly voice.

    Shows which agents are working, their status, and consensus
    when multiple agents collaborate on decisions.
    """

    # Status symbols
    SYMBOLS = {
        AgentStatus.QUEUED: "[dim]○[/dim]",
        AgentStatus.ACTIVE: "[accent3]●[/accent3]",
        AgentStatus.WAITING: "[warning]◐[/warning]",
        AgentStatus.COMPLETE: "[success]✔[/success]",
        AgentStatus.FAILED: "[error]✖[/error]",
    }

    # Friendly status messages for agents
    FRIENDLY_MESSAGES = {
        AgentStatus.QUEUED: "I'll help once it's my turn",
        AgentStatus.WAITING: "Waiting for the others",
        AgentStatus.COMPLETE: "All done!",
        AgentStatus.FAILED: "I ran into an issue",
    }

    def __init__(self):
        """Initialize agent panel."""
        self.agents: List[AgentState] = []
        self.consensus: Optional[ConsensusState] = None
        self.show_tokens: bool = True

    def add_agent(self, agent_type: str) -> AgentState:
        """
        Add an agent to the panel.

        Args:
            agent_type: Type of agent (e.g., "senior-engineer")

        Returns:
            The created AgentState
        """
        state = AgentState(
            agent_type=agent_type,
            message=self.FRIENDLY_MESSAGES[AgentStatus.QUEUED]
        )
        self.agents.append(state)
        return state

    def update_agent(
        self,
        agent_type: str,
        status: Optional[AgentStatus] = None,
        message: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Update an agent's state.

        Args:
            agent_type: Type of agent to update
            status: New status
            message: New message
            **kwargs: Other AgentState fields to update
        """
        for agent in self.agents:
            if agent.agent_type == agent_type:
                if status:
                    agent.status = status
                    if status == AgentStatus.ACTIVE and not agent.started_at:
                        agent.started_at = datetime.now()
                    elif status in (AgentStatus.COMPLETE, AgentStatus.FAILED):
                        agent.completed_at = datetime.now()
                if message:
                    agent.message = message
                for key, value in kwargs.items():
                    if hasattr(agent, key):
                        setattr(agent, key, value)
                break

    def start_consensus(self, topic: str, agent_types: List[str]) -> None:
        """
        Start tracking consensus on a topic.

        Args:
            topic: What agents are deciding on
            agent_types: List of agent types participating
        """
        self.consensus = ConsensusState(topic=topic)
        for agent_type in agent_types:
            self.consensus.agents[agent_type] = None
            self.consensus.confidences[agent_type] = ""

    def record_consensus_vote(
        self,
        agent_type: str,
        value: Any,
        confidence: str = "HIGH"
    ) -> None:
        """
        Record an agent's contribution to consensus.

        Args:
            agent_type: Agent providing the value
            value: The agent's value/vote
            confidence: Confidence level (HIGH, MEDIUM, LOW)
        """
        if self.consensus:
            self.consensus.agents[agent_type] = value
            self.consensus.confidences[agent_type] = confidence

    def finalize_consensus(self, final_value: Any, variance: Optional[float] = None) -> None:
        """
        Mark consensus as reached.

        Args:
            final_value: The agreed-upon value
            variance: Optional variance/spread in votes
        """
        if self.consensus:
            self.consensus.reached = True
            self.consensus.final_value = final_value
            self.consensus.variance = variance

    def render(self) -> Panel:
        """
        Render the agent panel.

        Returns:
            Rich Panel with agent activity
        """
        content = []

        # Render each agent
        for agent in self.agents:
            content.append(self._render_agent_row(agent))

        # Add spacing
        if content:
            content.append(Text())

        # Token summary if any agents have used tokens
        total_input = sum(a.tokens_input for a in self.agents)
        total_output = sum(a.tokens_output for a in self.agents)
        if total_input > 0 or total_output > 0:
            total_cost = self._estimate_cost(total_input, total_output)
            token_line = Text()
            token_line.append("Tokens: ", style="dim")
            token_line.append(f"{total_input:,} input", style="secondary")
            token_line.append(" │ ", style="dim")
            token_line.append(f"{total_output:,} output", style="secondary")
            token_line.append(" │ ", style="dim")
            token_line.append(f"Est. cost: ${total_cost:.2f}", style="accent2")
            content.append(token_line)

        # Combine into panel
        panel_content = Text("\n").join(content) if content else Text("No agents active")

        return Panel(
            panel_content,
            title="[bold accent3]Agent Activity[/bold accent3]",
            border_style="accent3",
            padding=(0, 1)
        )

    def render_consensus(self) -> Optional[Panel]:
        """
        Render consensus visualization if active.

        Returns:
            Rich Panel with consensus display, or None if no consensus
        """
        if not self.consensus:
            return None

        content = []

        # Render each agent's contribution
        for agent_type, value in self.consensus.agents.items():
            confidence = self.consensus.confidences.get(agent_type, "")
            line = Text()

            if value is not None:
                line.append(f"  {agent_type:<20}", style="primary")
                line.append(" →  ", style="dim")
                line.append(f"{value}", style="bold accent1")
                if confidence:
                    conf_style = {
                        "HIGH": "success",
                        "MEDIUM": "warning",
                        "LOW": "error"
                    }.get(confidence, "dim")
                    line.append(f"   {confidence} confidence", style=conf_style)
            else:
                line.append(f"  {agent_type:<20}", style="dim")
                line.append(" →  ", style="dim")
                line.append("thinking...", style="dim italic")

            content.append(line)

        # Add consensus result if reached
        if self.consensus.reached:
            content.append(Text())
            content.append(Text("─" * 50, style="dim"))

            result_line = Text()
            result_line.append("  CONSENSUS REACHED: ", style="bold success")
            result_line.append(f"{self.consensus.final_value}", style="bold accent1")
            if self.consensus.variance is not None:
                result_line.append(f" (σ = {self.consensus.variance:.1f})", style="dim")
            content.append(result_line)

        panel_content = Text("\n").join(content)

        return Panel(
            panel_content,
            title=f"[bold accent1]Consensus: {self.consensus.topic}[/bold accent1]",
            border_style="accent1",
            padding=(0, 1)
        )

    def _render_agent_row(self, agent: AgentState) -> Text:
        """Render a single agent row."""
        line = Text()

        # Status symbol
        symbol = self.SYMBOLS.get(agent.status, "○")
        line.append(f"  {symbol} ")

        # Agent name
        line.append(f"{agent.agent_type:<18}", style="primary")

        # Message (friendly voice)
        message = agent.message or self.FRIENDLY_MESSAGES.get(agent.status, "")
        if agent.status == AgentStatus.ACTIVE:
            line.append(message, style="accent3")
        elif agent.status == AgentStatus.COMPLETE:
            line.append(message, style="success")
        elif agent.status == AgentStatus.FAILED:
            line.append(message, style="error")
        else:
            line.append(message, style="dim")

        # Elapsed time for active/complete agents
        if agent.elapsed_str:
            # Pad to right-align elapsed time
            padding = 50 - len(line.plain)
            if padding > 0:
                line.append(" " * padding)
            line.append(agent.elapsed_str, style="tertiary")

        return line

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost based on Claude Sonnet pricing."""
        # Claude Sonnet pricing (approximate)
        input_cost = (input_tokens / 1_000_000) * 3.0
        output_cost = (output_tokens / 1_000_000) * 15.0
        return input_cost + output_cost
