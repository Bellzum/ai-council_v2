"""
Live display manager for in-place terminal updates.

Uses Rich's Live display to update content in-place without scrolling,
providing a clean, professional experience.

Layout Structure:
    ┌─────────────────────────────────────────┐
    │  WORKFLOW HEADER (static)               │
    ├─────────────────────────────────────────┤
    │  Step Timeline (updates in-place)       │
    ├─────────────────────────────────────────┤
    │  Agent Activity Panel (dynamic)         │
    ├─────────────────────────────────────────┤
    │  Current Activity (spinner/message)     │
    ├─────────────────────────────────────────┤
    │  Status Line (persistent)               │
    └─────────────────────────────────────────┘
"""

import sys
from datetime import datetime, timedelta
from typing import Optional, List

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

# Import themed console from CLI
from cli.console import console as themed_console, get_console

from scripts.workflow_executor.tui.status_line import StatusLine, StatusMetrics
from scripts.workflow_executor.tui.step_timeline import StepTimeline, StepState
from scripts.workflow_executor.tui.agent_panel import AgentPanel, AgentState, AgentStatus


class LiveDisplay:
    """
    Manages Rich Live display for workflow execution.

    Provides in-place updates for:
    - Workflow header
    - Step timeline
    - Agent activity panel
    - Current activity message
    - Status line

    For non-TTY environments (CI/CD), falls back to simple text output.
    """

    def __init__(
        self,
        workflow_name: str,
        sprint_name: Optional[str] = None,
        total_steps: int = 0,
        show_agent_panel: bool = True,
        console: Optional[Console] = None
    ):
        """
        Initialize live display.

        Args:
            workflow_name: Name of the workflow
            sprint_name: Optional sprint context
            total_steps: Total number of workflow steps
            show_agent_panel: Whether to show agent activity panel
            console: Rich Console instance (creates new if None)
        """
        self.workflow_name = workflow_name
        self.sprint_name = sprint_name
        self.total_steps = total_steps
        self.show_agent_panel = show_agent_panel

        self.console = console or get_console()  # Use themed console
        self.is_tty = self.console.is_terminal

        # Components
        self.status_line = StatusLine()
        self.step_timeline = StepTimeline()
        self.agent_panel = AgentPanel() if show_agent_panel else None

        # State
        self.started_at: Optional[datetime] = None
        self.current_activity: str = ""
        self.live: Optional[Live] = None

        # Initialize status metrics
        self.status_line.update(
            workflow_name=workflow_name,
            sprint_name=sprint_name,
            total_steps=total_steps
        )

    def start(self) -> None:
        """Start the live display."""
        self.started_at = datetime.now()

        if self.is_tty:
            self._print_header()
            self.live = Live(
                self._render_layout(),
                console=self.console,
                refresh_per_second=4,
                transient=False  # Keep content when done
            )
            self.live.start()
        else:
            # Non-TTY: Simple text output
            self._print_header()

    def stop(self) -> None:
        """Stop the live display."""
        if self.live:
            self.live.stop()
            self.live = None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False

    def _print_header(self) -> None:
        """Print the workflow header (static, not updated)."""
        header_text = Text()
        header_text.append(f"  {self.workflow_name.upper()}", style="bold primary")

        if self.sprint_name:
            header_text.append(f"  │  {self.sprint_name}", style="accent1")

        # Mode indicator
        mode = "AI" if self.show_agent_panel else "Fast"
        header_text.append(f"  │  Mode: {mode}", style="tertiary")

        self.console.print(Panel(
            header_text,
            border_style="accent1",
            padding=(0, 1)
        ))
        self.console.print()

    def _render_layout(self) -> Group:
        """Render the complete live layout."""
        components = []

        # Step timeline
        if self.step_timeline.steps:
            timeline_content = self.step_timeline.render_current()
            components.append(timeline_content)
            components.append(Text())

        # Agent panel
        if self.agent_panel and self.agent_panel.agents:
            components.append(self.agent_panel.render())

            # Consensus panel if active
            consensus = self.agent_panel.render_consensus()
            if consensus:
                components.append(consensus)

        # Current activity
        if self.current_activity:
            activity = Text()
            activity.append("◐ ", style="accent3")
            activity.append(self.current_activity, style="accent3")
            components.append(activity)
            components.append(Text())

        # Update elapsed time
        if self.started_at:
            self.status_line.update(elapsed=datetime.now() - self.started_at)

        # Status line
        components.append(Text("─" * 70, style="dim"))
        components.append(self.status_line.render())

        return Group(*components)

    def refresh(self) -> None:
        """Refresh the live display."""
        if self.live:
            self.live.update(self._render_layout())

    # ─── Step Management ─────────────────────────────────────────────────────

    def add_step(self, name: str) -> None:
        """Add a step to the timeline."""
        self.step_timeline.add_step(name)
        self.status_line.update(total_steps=len(self.step_timeline.steps))

    def start_step(self, step_number: int, message: Optional[str] = None) -> None:
        """
        Start a workflow step.

        Args:
            step_number: Step number (1-indexed)
            message: Optional activity message
        """
        self.step_timeline.start_step(step_number)
        self.status_line.update(current_step=step_number)

        if message:
            self.current_activity = message
        else:
            step = self.step_timeline.steps[step_number - 1]
            self.current_activity = f"I'm working on: {step.name}..."

        if self.is_tty:
            self.refresh()
        else:
            step = self.step_timeline.steps[step_number - 1]
            self.console.print(f"\n[bold accent2]Step {step_number}:[/bold accent2] {step.name}")

    def complete_step(
        self,
        step_number: Optional[int] = None,
        summary: Optional[str] = None,
        success: bool = True
    ) -> None:
        """
        Complete a workflow step.

        Args:
            step_number: Step to complete (defaults to current)
            summary: Brief result summary
            success: Whether step succeeded
        """
        step_num = step_number or self.step_timeline.current_step or 1

        if success:
            self.step_timeline.complete_step(step_num, summary)
            self.current_activity = ""
        else:
            self.step_timeline.fail_step(step_num, summary)
            self.current_activity = ""

        if self.is_tty:
            self.refresh()
        else:
            status = "[success]✔[/success]" if success else "[error]✖[/error]"
            msg = f"  {status} {summary}" if summary else f"  {status} Complete"
            self.console.print(msg)

    # ─── Agent Management ────────────────────────────────────────────────────

    def add_agent(self, agent_type: str) -> AgentState:
        """Add an agent to the activity panel."""
        if not self.agent_panel:
            self.agent_panel = AgentPanel()

        state = self.agent_panel.add_agent(agent_type)
        self.refresh()
        return state

    def start_agent(self, agent_type: str, message: str = "") -> None:
        """Start an agent's activity."""
        if self.agent_panel:
            friendly_msg = message or f"I'm analyzing the data..."
            self.agent_panel.update_agent(
                agent_type,
                status=AgentStatus.ACTIVE,
                message=friendly_msg
            )

            if self.is_tty:
                self.refresh()
            else:
                self.console.print(f"  [accent3]● {agent_type}:[/accent3] {friendly_msg}")

    def complete_agent(
        self,
        agent_type: str,
        result_summary: str = "",
        tokens_input: int = 0,
        tokens_output: int = 0,
        success: bool = True
    ) -> None:
        """Complete an agent's activity."""
        if self.agent_panel:
            status = AgentStatus.COMPLETE if success else AgentStatus.FAILED
            message = result_summary or ("All done!" if success else "I ran into an issue")

            self.agent_panel.update_agent(
                agent_type,
                status=status,
                message=message,
                result_summary=result_summary,
                tokens_input=tokens_input,
                tokens_output=tokens_output
            )

            # Update total token count in status line
            total_tokens = sum(
                a.tokens_input + a.tokens_output
                for a in self.agent_panel.agents
            )
            total_cost = self.agent_panel._estimate_cost(
                sum(a.tokens_input for a in self.agent_panel.agents),
                sum(a.tokens_output for a in self.agent_panel.agents)
            )
            self.status_line.update(tokens_used=total_tokens, cost_usd=total_cost)

            if self.is_tty:
                self.refresh()
            else:
                status_sym = "[success]✔[/success]" if success else "[error]✖[/error]"
                self.console.print(f"  {status_sym} {agent_type}: {message}")

    # ─── Activity Messages ───────────────────────────────────────────────────

    def update_activity(self, message: str) -> None:
        """Update the current activity message."""
        self.current_activity = message
        if self.is_tty:
            self.refresh()
        else:
            self.console.print(f"  ◐ {message}")

    def clear_activity(self) -> None:
        """Clear the current activity message."""
        self.current_activity = ""
        self.refresh()

    # ─── Output Methods ──────────────────────────────────────────────────────

    def print(self, *args, **kwargs) -> None:
        """
        Print content (pauses live display momentarily).

        Use for permanent output that should persist in terminal history.
        """
        if self.live:
            self.live.stop()
            self.console.print(*args, **kwargs)
            self.live.start()
        else:
            self.console.print(*args, **kwargs)

    def print_panel(self, panel: Panel) -> None:
        """Print a Rich Panel."""
        self.print(panel)

    def print_table(self, table) -> None:
        """Print a Rich Table."""
        self.print(table)

    # ─── Metrics ─────────────────────────────────────────────────────────────

    def update_metrics(self, **kwargs) -> None:
        """Update status line metrics."""
        self.status_line.update(**kwargs)
        self.refresh()
