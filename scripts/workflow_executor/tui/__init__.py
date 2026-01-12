"""
Trustable AI Terminal UI System.

A comprehensive, delightful terminal UI for workflow scripts featuring:
- Live in-place updates (no scroll clutter)
- Agent activity visualization with consensus
- Step timeline progression
- Persistent status line
- Friendly, personable voice

Usage:
    from scripts.workflow_executor.tui import WorkflowTUI

    with WorkflowTUI("Sprint Planning", total_steps=8) as tui:
        tui.start_step(1, "Extract Work Items")
        # ... do work ...
        tui.complete_step()
"""

from scripts.workflow_executor.tui.workflow_tui import WorkflowTUI
from scripts.workflow_executor.tui.live_display import LiveDisplay
from scripts.workflow_executor.tui.status_line import StatusLine
from scripts.workflow_executor.tui.agent_panel import AgentPanel, AgentState, AgentStatus
from scripts.workflow_executor.tui.step_timeline import StepTimeline, StepState
from scripts.workflow_executor.tui.output_formatter import (
    OutputFormatter,
    WorkItemType,
    WorkItem,
    ApprovalGate,
)

__all__ = [
    "WorkflowTUI",
    "LiveDisplay",
    "StatusLine",
    "AgentPanel",
    "AgentState",
    "AgentStatus",
    "StepTimeline",
    "StepState",
    "OutputFormatter",
    "WorkItemType",
    "WorkItem",
    "ApprovalGate",
]
