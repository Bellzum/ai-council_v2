"""
Output formatting utilities for consistent, beautiful workflow output.

Provides standardized formatters for:
- Work item tables (compact and full)
- Approval gates with friendly prompts
- Summary cards with metrics
- Agent result attribution
- Error messages with empathy
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Keychain brand colors - using hex values for theme-independent rendering
COLORS = {
    "primary": "#D9EAFC",      # Light blue - primary text
    "secondary": "#758B9B",    # Muted blue-gray
    "tertiary": "#94A5CC",     # Medium blue
    "accent1": "#71E4D1",      # Cyan - highlights
    "accent2": "#67CFEE",      # Light cyan
    "accent3": "#BB93DD",      # Purple - AI agents
    "success": "#71E4D1",      # Cyan
    "warning": "#FFA500",      # Orange
    "error": "#FF6B6B",        # Red
    "info": "#67CFEE",         # Light cyan - informational
}


class WorkItemType(Enum):
    """Work item types with associated styling."""

    EPIC = ("Epic", "magenta", "📦")
    FEATURE = ("Feature", "blue", "🎯")
    TASK = ("Task", "green", "📋")
    BUG = ("Bug", "red", "🐛")

    def __init__(self, label: str, color: str, emoji: str):
        self.label = label
        self.color = color
        self.emoji = emoji

    @property
    def badge(self) -> str:
        """Get Rich markup badge for this type."""
        return f"[bold {self.color}]{self.emoji} {self.label}[/bold {self.color}]"


@dataclass
class WorkItem:
    """Work item data for display."""

    id: int
    title: str
    type: WorkItemType
    state: str
    story_points: Optional[int] = None
    assigned_to: Optional[str] = None
    tags: Optional[str] = None


@dataclass
class ApprovalGate:
    """Approval gate configuration."""

    title: str
    summary_lines: List[str]
    options: List[Tuple[str, str]]  # (key, description)
    prompt: str = "Your choice"
    default: Optional[str] = None


class OutputFormatter:
    """
    Standardized output formatting with friendly voice.

    All formatters use consistent styling and warm language.
    """

    @staticmethod
    def work_item_table(
        items: List[WorkItem],
        title: str = "Work Items",
        compact: bool = False,
        show_total_points: bool = True
    ) -> Panel:
        """
        Format work items as a Rich table.

        Args:
            items: List of work items
            title: Table title
            compact: If True, use compact layout
            show_total_points: If True, show total story points

        Returns:
            Rich Panel containing the table
        """
        if not items:
            return Panel(
                Text("No work items to display", style="dim"),
                title=f"[bold {COLORS['accent1']}]{title}[/bold {COLORS['accent1']}]",
                border_style=COLORS["accent1"]
            )

        table = Table(
            show_header=True,
            header_style=f"bold {COLORS['primary']}",
            border_style=COLORS["accent1"],
            show_lines=False,
            padding=(0, 1)
        )

        # Add columns based on compact mode
        table.add_column("ID", style=COLORS["accent2"], width=8)
        table.add_column("Type", width=12 if not compact else 8)
        table.add_column("Title", style=COLORS["primary"], no_wrap=False, ratio=2)

        if not compact:
            table.add_column("State", style=COLORS["tertiary"], width=12)

        table.add_column("Pts", justify="right", width=5)

        if not compact:
            table.add_column("Assigned", style=COLORS["secondary"], width=15)

        # Add rows
        for item in items:
            row = [
                f"#{item.id}",
                item.type.badge if not compact else f"[{item.type.color}]{item.type.emoji}[/{item.type.color}]",
                item.title[:60] + "..." if len(item.title) > 60 else item.title,
            ]

            if not compact:
                row.append(item.state)

            row.append(str(item.story_points) if item.story_points else "-")

            if not compact:
                row.append(item.assigned_to or "-")

            table.add_row(*row)

        # Calculate total points
        total_points = sum(item.story_points or 0 for item in items)

        # Build panel content
        content = table

        if show_total_points:
            footer = Text()
            footer.append("\n")
            footer.append(f"Total: {len(items)} items", style=COLORS["tertiary"])
            footer.append(" | ", style="dim")
            footer.append(f"{total_points} story points", style=f"bold {COLORS['accent1']}")

            from rich.console import Group
            content = Group(table, footer)

        return Panel(
            content,
            title=f"[bold {COLORS['accent1']}]{title} ({len(items)})[/bold {COLORS['accent1']}]",
            border_style=COLORS["accent1"],
            padding=(0, 1)
        )

    @staticmethod
    def approval_gate(gate: ApprovalGate) -> Panel:
        """
        Format an approval gate with friendly prompts.

        Args:
            gate: Approval gate configuration

        Returns:
            Rich Panel for the approval gate
        """
        content = []

        # Friendly intro
        content.append(Text("I need your approval to continue.", style=COLORS["primary"]))
        content.append(Text())

        # Summary with bullet points
        for line in gate.summary_lines:
            summary_line = Text()
            summary_line.append("  * ", style=COLORS["accent1"])
            summary_line.append(line, style=COLORS["primary"])
            content.append(summary_line)

        content.append(Text())
        content.append(Text("─" * 50, style="dim"))
        content.append(Text())

        # Options
        for key, description in gate.options:
            option_line = Text()
            option_line.append(f"  [{key}] ", style=f"bold {COLORS['accent2']}")
            option_line.append(description, style=COLORS["primary"])
            content.append(option_line)

        panel_content = Text("\n").join(content)

        return Panel(
            panel_content,
            title=f"[bold {COLORS['warning']}]Pause {gate.title}[/bold {COLORS['warning']}]",
            border_style=COLORS["warning"],
            padding=(1, 2)
        )

    @staticmethod
    def summary_card(
        title: str,
        metrics: Dict[str, Any],
        style: str = "success"
    ) -> Panel:
        """
        Format a summary card with metrics.

        Args:
            title: Card title
            metrics: Dictionary of metric name to value
            style: Border style name (success, warning, error, info)

        Returns:
            Rich Panel with metrics
        """
        # Map style name to color
        style_color = COLORS.get(style, COLORS["success"])

        table = Table.grid(padding=(0, 2))
        table.add_column(style=f"bold {COLORS['primary']}")
        table.add_column(style=COLORS["accent1"])

        for key, value in metrics.items():
            # Format value based on type
            if isinstance(value, float):
                formatted = f"{value:.1f}"
            elif isinstance(value, bool):
                formatted = "Yes" if value else "No"
            else:
                formatted = str(value)

            table.add_row(f"{key}:", formatted)

        return Panel(
            table,
            title=f"[bold {style_color}]{title}[/bold {style_color}]",
            border_style=style_color,
            padding=(0, 1)
        )

    @staticmethod
    def agent_result(
        agent_type: str,
        result: str,
        confidence: Optional[str] = None
    ) -> Text:
        """
        Format an agent's result with attribution.

        Args:
            agent_type: Type of agent
            result: The agent's result/recommendation
            confidence: Optional confidence level

        Returns:
            Rich Text with formatted result
        """
        text = Text()
        text.append(f"The {agent_type} ", style="dim")
        text.append("says: ", style="dim")
        text.append(f'"{result}"', style=f"{COLORS['accent3']} italic")

        if confidence:
            conf_color = {
                "HIGH": COLORS["success"],
                "MEDIUM": COLORS["warning"],
                "LOW": COLORS["error"]
            }.get(confidence, "dim")
            text.append(f" ({confidence} confidence)", style=conf_color)

        return text

    @staticmethod
    def friendly_error(
        error_type: str,
        message: str,
        suggestion: Optional[str] = None,
        options: Optional[List[Tuple[str, str]]] = None
    ) -> Panel:
        """
        Format an error message with empathy and helpful suggestions.

        Args:
            error_type: Type of error (e.g., "Connection Error")
            message: What went wrong
            suggestion: How to fix it
            options: Recovery options [(key, description), ...]

        Returns:
            Rich Panel with friendly error
        """
        content = []

        # Empathetic error message
        error_line = Text()
        error_line.append("X ", style=COLORS["error"])
        error_line.append(message, style=COLORS["error"])
        content.append(error_line)

        content.append(Text())

        # Suggestion with friendly tone
        if suggestion:
            suggestion_line = Text()
            suggestion_line.append("Let's fix that: ", style=COLORS["accent1"])
            suggestion_line.append(suggestion, style=COLORS["primary"])
            content.append(suggestion_line)

        # Recovery options
        if options:
            content.append(Text())
            for key, description in options:
                option_line = Text()
                option_line.append(f"  [{key}] ", style=f"bold {COLORS['accent2']}")
                option_line.append(description, style=COLORS["primary"])
                content.append(option_line)

        panel_content = Text("\n").join(content)

        return Panel(
            panel_content,
            title=f"[bold {COLORS['error']}]Oops! {error_type}[/bold {COLORS['error']}]",
            border_style=COLORS["error"],
            padding=(0, 1)
        )

    @staticmethod
    def success_message(message: str, details: Optional[List[str]] = None) -> Text:
        """
        Format a success message.

        Args:
            message: Main success message
            details: Optional list of detail lines

        Returns:
            Rich Text with success formatting
        """
        text = Text()
        text.append("+ ", style=COLORS["success"])
        text.append(message, style=COLORS["success"])

        if details:
            for detail in details:
                text.append("\n  ")
                text.append(detail, style="dim")

        return text

    @staticmethod
    def progress_message(message: str, current: int = 0, total: int = 0) -> Text:
        """
        Format a progress message.

        Args:
            message: What's happening
            current: Current item number
            total: Total items

        Returns:
            Rich Text with progress formatting
        """
        text = Text()
        text.append("* ", style=COLORS["accent3"])
        text.append(message, style=COLORS["accent3"])

        if total > 0:
            text.append(f" ({current}/{total})", style=COLORS["tertiary"])

        return text
