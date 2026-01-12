"""
Persistent status line for workflow execution.

Renders a compact status bar showing:
- Workflow/sprint context
- Current step progress
- Elapsed time
- Token usage and cost
- Connection status
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

from rich.text import Text


@dataclass
class StatusMetrics:
    """Metrics displayed in the status line."""

    workflow_name: str
    sprint_name: Optional[str] = None
    current_step: int = 0
    total_steps: int = 0
    elapsed: Optional[timedelta] = None
    tokens_used: int = 0
    cost_usd: float = 0.0
    work_item_count: int = 0
    connected: bool = True


class StatusLine:
    """
    Persistent status line renderer.

    Displays compact metrics at the bottom of the terminal.

    Example output:
        Sprint 9 | Step 3/8 | 2m 45s | 12.5k tokens | $0.04 | 12 items | ● Connected
    """

    def __init__(self):
        """Initialize status line."""
        self.metrics = StatusMetrics(workflow_name="Workflow")

    def update(self, **kwargs) -> None:
        """
        Update status metrics.

        Args:
            **kwargs: Any StatusMetrics field to update
        """
        for key, value in kwargs.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)

    def render(self) -> Text:
        """
        Render the status line as Rich Text.

        Returns:
            Rich Text object with formatted status line
        """
        parts = []

        # Workflow/Sprint name
        if self.metrics.sprint_name:
            parts.append(f"[bold accent1]{self.metrics.sprint_name}[/bold accent1]")
        else:
            parts.append(f"[bold accent1]{self.metrics.workflow_name}[/bold accent1]")

        # Step progress
        if self.metrics.total_steps > 0:
            parts.append(
                f"[primary]Step {self.metrics.current_step}/{self.metrics.total_steps}[/primary]"
            )

        # Elapsed time
        if self.metrics.elapsed:
            elapsed_str = self._format_elapsed(self.metrics.elapsed)
            parts.append(f"[tertiary]{elapsed_str}[/tertiary]")

        # Token usage
        if self.metrics.tokens_used > 0:
            tokens_str = self._format_tokens(self.metrics.tokens_used)
            parts.append(f"[secondary]{tokens_str} tokens[/secondary]")

        # Cost
        if self.metrics.cost_usd > 0:
            parts.append(f"[accent2]${self.metrics.cost_usd:.2f}[/accent2]")

        # Work item count
        if self.metrics.work_item_count > 0:
            parts.append(f"[primary]{self.metrics.work_item_count} items[/primary]")

        # Connection status
        if self.metrics.connected:
            parts.append("[success]● Connected[/success]")
        else:
            parts.append("[error]○ Disconnected[/error]")

        # Join with separator
        separator = " [dim]│[/dim] "
        return Text.from_markup(separator.join(parts))

    def _format_elapsed(self, elapsed: timedelta) -> str:
        """Format elapsed time as human-readable string."""
        total_seconds = int(elapsed.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def _format_tokens(self, tokens: int) -> str:
        """Format token count with K suffix for large numbers."""
        if tokens >= 1000:
            return f"{tokens / 1000:.1f}k"
        return str(tokens)
