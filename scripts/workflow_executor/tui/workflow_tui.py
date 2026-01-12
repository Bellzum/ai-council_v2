"""
Main WorkflowTUI API - the primary entry point for workflow terminal UI.

Provides a high-level, easy-to-use API that orchestrates all TUI components:
- Live display with in-place updates
- Step timeline progression
- Agent activity visualization
- Approval gates with friendly prompts
- Output formatting

Usage:
    from scripts.workflow_executor.tui import WorkflowTUI

    with WorkflowTUI("Sprint Planning", sprint_name="Sprint 9", total_steps=8) as tui:
        tui.start_step(1, "I'm extracting work items...")
        items = get_work_items()
        tui.complete_step(summary=f"Found {len(items)} items!")

        with tui.agent_activity("senior-engineer") as agent:
            agent.update("I'm estimating complexity...")
            result = call_ai(...)
            agent.complete(f"Estimated {len(result)} items")

        tui.print_work_items(items, title="Sprint Backlog")

        choice = tui.approval_gate(
            title="Ready to Assign",
            summary_lines=["12 items selected", "47 story points"],
            options=[("y", "Approve"), ("n", "Cancel"), ("e", "Edit")]
        )
"""

import sys
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel

# Import themed console from CLI
from cli.console import get_console

from scripts.workflow_executor.tui.live_display import LiveDisplay
from scripts.workflow_executor.tui.agent_panel import AgentPanel, AgentState, AgentStatus
from scripts.workflow_executor.tui.step_timeline import StepTimeline, StepState
from scripts.workflow_executor.tui.output_formatter import (
    OutputFormatter,
    WorkItem,
    WorkItemType,
    ApprovalGate,
)


class AgentActivityContext:
    """Context manager for tracking agent activity."""

    def __init__(self, tui: "WorkflowTUI", agent_type: str):
        """
        Initialize agent activity context.

        Args:
            tui: Parent WorkflowTUI instance
            agent_type: Type of agent
        """
        self.tui = tui
        self.agent_type = agent_type
        self.tokens_input = 0
        self.tokens_output = 0
        self._success = True
        self._result_summary = ""

    def __enter__(self):
        """Start agent activity."""
        self.tui.display.start_agent(self.agent_type)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Complete agent activity."""
        if exc_type is not None:
            self._success = False
            self._result_summary = str(exc_val) if exc_val else "An error occurred"

        self.tui.display.complete_agent(
            self.agent_type,
            result_summary=self._result_summary,
            tokens_input=self.tokens_input,
            tokens_output=self.tokens_output,
            success=self._success
        )
        return False  # Don't suppress exceptions

    def update(self, message: str) -> None:
        """Update agent's activity message."""
        if self.tui.display.agent_panel:
            self.tui.display.agent_panel.update_agent(
                self.agent_type,
                message=message
            )
            self.tui.display.refresh()

    def complete(
        self,
        summary: str = "",
        tokens_input: int = 0,
        tokens_output: int = 0
    ) -> None:
        """Mark agent activity as complete with results."""
        self._result_summary = summary
        self.tokens_input = tokens_input
        self.tokens_output = tokens_output
        self._success = True


class WorkflowTUI:
    """
    Main entry point for workflow terminal UI.

    Provides a friendly, high-level API for creating beautiful
    terminal interfaces with live updates, agent visualization,
    and consistent styling.

    Example:
        with WorkflowTUI("Sprint Planning", total_steps=5) as tui:
            tui.say("Hi! I'm ready to help plan your sprint.")

            tui.start_step(1, "Gathering work items")
            items = fetch_items()
            tui.complete_step(f"Found {len(items)} items!")

            with tui.agent("senior-engineer") as agent:
                agent.update("Analyzing complexity...")
                result = estimate(items)
                agent.complete(f"Estimated all {len(items)} items")

            tui.print_items(items)
    """

    def __init__(
        self,
        workflow_name: str,
        sprint_name: Optional[str] = None,
        total_steps: int = 0,
        step_names: Optional[List[str]] = None,
        show_agent_panel: bool = True,
        enabled: bool = True,
        console: Optional[Console] = None
    ):
        """
        Initialize WorkflowTUI.

        Args:
            workflow_name: Name of the workflow
            sprint_name: Optional sprint context
            total_steps: Total number of steps (or len(step_names))
            step_names: Optional list of step names
            show_agent_panel: Whether to show agent activity panel
            enabled: If False, use minimal output (for CI/CD)
            console: Rich Console instance
        """
        self.workflow_name = workflow_name
        self.sprint_name = sprint_name
        self.enabled = enabled and sys.stdout.isatty()

        # Calculate total steps
        if step_names:
            total_steps = len(step_names)

        self.console = console or get_console()  # Use themed console

        # Create display
        self.display = LiveDisplay(
            workflow_name=workflow_name,
            sprint_name=sprint_name,
            total_steps=total_steps,
            show_agent_panel=show_agent_panel,
            console=self.console
        )

        # Initialize steps if names provided
        if step_names:
            for name in step_names:
                self.display.add_step(name)

        # Track state
        self._started = False
        self._current_step = 0

    def __enter__(self):
        """Start the TUI."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the TUI."""
        if exc_type is not None:
            self.say(f"Oops! Something went wrong: {exc_val}", style="error")
        self.stop()
        return False

    def start(self) -> None:
        """Start the TUI display."""
        if not self._started:
            self.display.start()
            self._started = True

    def stop(self) -> None:
        """Stop the TUI display."""
        if self._started:
            self.display.stop()
            self._started = False

    # ─── Friendly Communication ──────────────────────────────────────────────

    def say(self, message: str, style: str = "primary") -> None:
        """
        Say something to the user (friendly voice).

        Args:
            message: Message to display
            style: Rich style (primary, success, warning, error)
        """
        self.display.print(f"[{style}]{message}[/{style}]")

    def celebrate(self, message: str) -> None:
        """Celebrate a success."""
        self.display.print(f"[success]✔ {message}[/success]")

    def warn(self, message: str) -> None:
        """Show a warning."""
        self.display.print(f"[warning]⚠ {message}[/warning]")

    def error(self, message: str, suggestion: Optional[str] = None) -> None:
        """Show an error with optional suggestion."""
        self.display.print(f"[error]✖ {message}[/error]")
        if suggestion:
            self.display.print(f"[accent1]  → {suggestion}[/accent1]")

    # ─── Step Management ─────────────────────────────────────────────────────

    def add_step(self, name: str) -> None:
        """Add a step to the workflow."""
        self.display.add_step(name)

    def start_step(self, step_number: int, message: Optional[str] = None) -> None:
        """
        Start a workflow step.

        Args:
            step_number: Step number (1-indexed)
            message: Optional friendly message (e.g., "I'm checking the backlog...")
        """
        self._current_step = step_number
        self.display.start_step(step_number, message)

    def complete_step(
        self,
        summary: Optional[str] = None,
        success: bool = True
    ) -> None:
        """
        Complete the current step.

        Args:
            summary: Brief result summary (e.g., "Found 12 items!")
            success: Whether step succeeded
        """
        self.display.complete_step(self._current_step, summary, success)

    def skip_step(self, reason: str = "Skipped") -> None:
        """Skip the current step."""
        self.display.step_timeline.skip_step(self._current_step, reason)
        self.display.refresh()

    # ─── Agent Activity ──────────────────────────────────────────────────────

    @contextmanager
    def agent(self, agent_type: str):
        """
        Context manager for agent activity.

        Usage:
            with tui.agent("senior-engineer") as agent:
                agent.update("I'm analyzing...")
                result = do_work()
                agent.complete("Done!", tokens_input=100)
        """
        # Ensure agent exists in panel
        self.display.add_agent(agent_type)

        ctx = AgentActivityContext(self, agent_type)
        yield ctx

    def agent_activity(self, agent_type: str):
        """Alias for agent() context manager."""
        return self.agent(agent_type)

    def show_consensus(
        self,
        topic: str,
        votes: Dict[str, Any],
        confidences: Optional[Dict[str, str]] = None,
        final_value: Optional[Any] = None
    ) -> None:
        """
        Show multi-agent consensus.

        Args:
            topic: What agents are deciding
            votes: Dict of agent_type -> value
            confidences: Optional dict of agent_type -> confidence level
            final_value: The agreed-upon value if consensus reached
        """
        if self.display.agent_panel:
            panel = self.display.agent_panel
            panel.start_consensus(topic, list(votes.keys()))

            for agent_type, value in votes.items():
                confidence = (confidences or {}).get(agent_type, "HIGH")
                panel.record_consensus_vote(agent_type, value, confidence)

            if final_value is not None:
                # Calculate variance if numeric
                variance = None
                try:
                    values = [v for v in votes.values() if isinstance(v, (int, float))]
                    if len(values) > 1:
                        mean = sum(values) / len(values)
                        variance = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
                except (TypeError, ValueError):
                    pass

                panel.finalize_consensus(final_value, variance)

            self.display.print(panel.render_consensus())

    # ─── Output ──────────────────────────────────────────────────────────────

    def print(self, *args, **kwargs) -> None:
        """Print to console (for permanent output)."""
        self.display.print(*args, **kwargs)

    def print_panel(self, panel: Panel) -> None:
        """Print a Rich Panel."""
        self.display.print_panel(panel)

    def print_work_items(
        self,
        items: List[Dict[str, Any]],
        title: str = "Work Items",
        compact: bool = False
    ) -> None:
        """
        Print work items as a formatted table.

        Args:
            items: List of work item dicts (from adapter)
            title: Table title
            compact: Use compact layout
        """
        # Convert to WorkItem objects
        work_items = []
        for item in items:
            # Handle both normalized and raw Azure DevOps formats
            item_id = item.get("id", 0)
            item_title = item.get("title") or item.get("fields", {}).get("System.Title", "Untitled")
            item_type_str = item.get("type") or item.get("fields", {}).get("System.WorkItemType", "Task")
            item_state = item.get("state") or item.get("fields", {}).get("System.State", "New")
            item_points = item.get("story_points") or item.get("fields", {}).get(
                "Microsoft.VSTS.Scheduling.StoryPoints"
            )
            item_assigned = item.get("assigned_to")
            if not item_assigned:
                assigned_obj = item.get("fields", {}).get("System.AssignedTo")
                if isinstance(assigned_obj, dict):
                    item_assigned = assigned_obj.get("displayName")

            # Map type string to enum
            type_map = {
                "Epic": WorkItemType.EPIC,
                "Feature": WorkItemType.FEATURE,
                "Task": WorkItemType.TASK,
                "Bug": WorkItemType.BUG,
            }
            work_item_type = type_map.get(item_type_str, WorkItemType.TASK)

            work_items.append(WorkItem(
                id=item_id,
                title=item_title,
                type=work_item_type,
                state=item_state,
                story_points=int(item_points) if item_points else None,
                assigned_to=item_assigned
            ))

        panel = OutputFormatter.work_item_table(work_items, title=title, compact=compact)
        self.display.print_panel(panel)

    def print_summary(
        self,
        title: str,
        metrics: Dict[str, Any],
        style: str = "success"
    ) -> None:
        """Print a summary card with metrics."""
        panel = OutputFormatter.summary_card(title, metrics, style)
        self.display.print_panel(panel)

    # ─── Approval Gates ──────────────────────────────────────────────────────

    def approval_gate(
        self,
        title: str,
        summary_lines: List[str],
        options: List[Tuple[str, str]],
        default: Optional[str] = None
    ) -> str:
        """
        Show an approval gate and wait for user input.

        Args:
            title: Gate title
            summary_lines: Summary bullet points
            options: List of (key, description) tuples
            default: Default option key

        Returns:
            The key of the selected option
        """
        # Stop live display to show approval gate
        was_live = self.display.live is not None
        if was_live:
            self.display.stop()

        # Create and print approval gate
        gate = ApprovalGate(
            title=title,
            summary_lines=summary_lines,
            options=options,
            default=default
        )
        self.console.print(OutputFormatter.approval_gate(gate))

        # Build prompt
        valid_keys = [opt[0].lower() for opt in options]
        prompt = f"\n{gate.prompt}"
        if default:
            prompt += f" [{default}]"
        prompt += ": "

        # Get user input
        while True:
            try:
                response = input(prompt).strip().lower()
                if not response and default:
                    response = default.lower()

                if response in valid_keys:
                    break
                else:
                    self.console.print(
                        f"[warning]Please enter one of: {', '.join(valid_keys)}[/warning]"
                    )
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[warning]Cancelled[/warning]")
                response = "n"  # Default to no/cancel
                break

        # Restart live display
        if was_live:
            self.display.start()

        return response

    def confirm(self, message: str, default: bool = False) -> bool:
        """
        Simple yes/no confirmation.

        Args:
            message: Question to ask
            default: Default value if user presses Enter

        Returns:
            True for yes, False for no
        """
        response = self.approval_gate(
            title="Confirmation",
            summary_lines=[message],
            options=[("y", "Yes"), ("n", "No")],
            default="y" if default else "n"
        )
        return response == "y"

    # ─── Utility ─────────────────────────────────────────────────────────────

    def update_activity(self, message: str) -> None:
        """Update the current activity message."""
        self.display.update_activity(message)

    def update_metrics(self, **kwargs) -> None:
        """Update status line metrics."""
        self.display.update_metrics(**kwargs)
