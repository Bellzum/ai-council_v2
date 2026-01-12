"""
Step timeline visualization for workflow progress.

Displays workflow steps as a horizontal timeline showing:
- Completed steps with checkmarks
- Current step with activity indicator
- Pending steps
- Brief status under each step
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from rich.text import Text


class StepState(Enum):
    """Step execution state."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class Step:
    """A workflow step with state and optional summary."""

    number: int
    name: str
    state: StepState = StepState.PENDING
    summary: Optional[str] = None  # Brief result summary (e.g., "12 items")


class StepTimeline:
    """
    Horizontal step timeline renderer.

    Renders workflow steps as a visual timeline:

        ● Extract ─── ● Prioritize ─── ◐ Estimate ─── ○ Assemble ─── ○ Approve
          ✓ 12 items     ✓ ranked        3/8 items      pending       pending
    """

    # Step state symbols
    SYMBOLS = {
        StepState.PENDING: "[dim]○[/dim]",
        StepState.IN_PROGRESS: "[accent3]◐[/accent3]",
        StepState.COMPLETE: "[success]●[/success]",
        StepState.SKIPPED: "[warning]⊘[/warning]",
        StepState.FAILED: "[error]✖[/error]",
    }

    # Connector styles
    CONNECTOR_COMPLETE = "[success]───[/success]"
    CONNECTOR_PENDING = "[dim]───[/dim]"

    def __init__(self, steps: Optional[List[str]] = None):
        """
        Initialize step timeline.

        Args:
            steps: Optional list of step names to initialize
        """
        self.steps: List[Step] = []
        if steps:
            for i, name in enumerate(steps, 1):
                self.steps.append(Step(number=i, name=name))

    def add_step(self, name: str) -> Step:
        """
        Add a step to the timeline.

        Args:
            name: Step name

        Returns:
            The created Step
        """
        step = Step(number=len(self.steps) + 1, name=name)
        self.steps.append(step)
        return step

    def update_step(
        self,
        step_number: int,
        state: Optional[StepState] = None,
        summary: Optional[str] = None
    ) -> None:
        """
        Update a step's state and/or summary.

        Args:
            step_number: Step number (1-indexed)
            state: New state
            summary: Brief result summary
        """
        if 1 <= step_number <= len(self.steps):
            step = self.steps[step_number - 1]
            if state:
                step.state = state
            if summary is not None:
                step.summary = summary

    def start_step(self, step_number: int) -> None:
        """Mark a step as in progress."""
        self.update_step(step_number, state=StepState.IN_PROGRESS)

    def complete_step(self, step_number: int, summary: Optional[str] = None) -> None:
        """Mark a step as complete with optional summary."""
        self.update_step(step_number, state=StepState.COMPLETE, summary=summary)

    def fail_step(self, step_number: int, summary: Optional[str] = None) -> None:
        """Mark a step as failed."""
        self.update_step(step_number, state=StepState.FAILED, summary=summary)

    def skip_step(self, step_number: int, reason: Optional[str] = None) -> None:
        """Mark a step as skipped."""
        self.update_step(step_number, state=StepState.SKIPPED, summary=reason or "skipped")

    @property
    def current_step(self) -> Optional[int]:
        """Get the current (in-progress) step number."""
        for step in self.steps:
            if step.state == StepState.IN_PROGRESS:
                return step.number
        return None

    def render(self, compact: bool = False) -> Text:
        """
        Render the step timeline.

        Args:
            compact: If True, use abbreviated step names

        Returns:
            Rich Text with the rendered timeline
        """
        if not self.steps:
            return Text("No steps defined", style="dim")

        # Render in two lines: symbols/names and summaries
        line1_parts = []  # Symbols and names
        line2_parts = []  # Summaries

        for i, step in enumerate(self.steps):
            # Get symbol
            symbol = self.SYMBOLS.get(step.state, "○")

            # Get name (possibly truncated)
            name = step.name
            if compact and len(name) > 10:
                name = name[:9] + "…"

            # Build step display
            step_display = f"{symbol} {name}"
            line1_parts.append(step_display)

            # Build summary display
            if step.summary:
                if step.state == StepState.COMPLETE:
                    summary = f"[success]✓ {step.summary}[/success]"
                elif step.state == StepState.FAILED:
                    summary = f"[error]✖ {step.summary}[/error]"
                elif step.state == StepState.IN_PROGRESS:
                    summary = f"[accent3]{step.summary}[/accent3]"
                else:
                    summary = f"[dim]{step.summary}[/dim]"
            else:
                if step.state == StepState.PENDING:
                    summary = "[dim]pending[/dim]"
                elif step.state == StepState.IN_PROGRESS:
                    summary = "[accent3]working...[/accent3]"
                else:
                    summary = ""

            line2_parts.append(summary)

            # Add connector (except after last step)
            if i < len(self.steps) - 1:
                # Use complete connector if this and next are complete/in-progress
                if step.state in (StepState.COMPLETE,) and self.steps[i + 1].state != StepState.PENDING:
                    line1_parts.append(self.CONNECTOR_COMPLETE)
                else:
                    line1_parts.append(self.CONNECTOR_PENDING)
                line2_parts.append("   ")  # Spacing under connector

        # Combine into final text
        line1 = Text.from_markup(" ".join(line1_parts))
        line2 = Text.from_markup("  " + "   ".join(line2_parts))  # Extra indent for alignment

        result = Text()
        result.append(line1)
        result.append("\n")
        result.append(line2)

        return result

    def render_vertical(self) -> Text:
        """
        Render the timeline vertically for narrow terminals.

        Returns:
            Rich Text with vertical timeline
        """
        lines = []

        for step in self.steps:
            symbol = self.SYMBOLS.get(step.state, "○")

            # Step line
            line = Text()
            line.append(f"{symbol} ", style="bold")
            line.append(f"Step {step.number}: ", style="bold accent2")
            line.append(step.name, style="primary")

            lines.append(line)

            # Summary line (indented)
            if step.summary:
                summary_line = Text()
                summary_line.append("  └─ ")
                if step.state == StepState.COMPLETE:
                    summary_line.append(step.summary, style="success")
                elif step.state == StepState.FAILED:
                    summary_line.append(step.summary, style="error")
                elif step.state == StepState.IN_PROGRESS:
                    summary_line.append(step.summary, style="accent3")
                else:
                    summary_line.append(step.summary, style="dim")
                lines.append(summary_line)

        return Text("\n").join(lines)

    def render_current(self) -> Text:
        """
        Render just the current step with details.

        Returns:
            Rich Text showing current step
        """
        current = self.current_step
        if not current or current > len(self.steps):
            return Text("No step in progress", style="dim")

        step = self.steps[current - 1]

        line = Text()
        line.append(f"Step {current} of {len(self.steps)}: ", style="bold accent2")
        line.append(step.name, style="bold primary")

        if step.summary:
            line.append("\n├─ ", style="dim")
            line.append(step.summary, style="accent3")

        return line
