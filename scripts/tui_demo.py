#!/usr/bin/env python3
"""
TUI Demonstration Script

Showcases the trustable-ai terminal UI components with friendly voice.

Usage:
    python scripts/tui_demo.py
    python scripts/tui_demo.py --fast  # Skip delays
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.workflow_executor.tui import (
    WorkflowTUI,
    OutputFormatter,
    WorkItemType,
    WorkItem,
)


def demo_work_items():
    """Sample work items for demonstration."""
    return [
        {"id": 1234, "title": "Implement user authentication middleware", "type": "Task", "state": "New", "story_points": 5},
        {"id": 1235, "title": "Fix login crash on network timeout", "type": "Bug", "state": "New", "story_points": 3},
        {"id": 1236, "title": "Add unit tests for input validator", "type": "Task", "state": "In Progress", "story_points": 2},
        {"id": 1237, "title": "Update API documentation", "type": "Task", "state": "New", "story_points": 1},
        {"id": 1238, "title": "Refactor database connection pooling", "type": "Task", "state": "New", "story_points": 8},
    ]


def main():
    parser = argparse.ArgumentParser(description="TUI Demonstration")
    parser.add_argument("--fast", action="store_true", help="Skip delays")
    args = parser.parse_args()

    delay = 0 if args.fast else 1.5

    print("\n" + "─" * 60)
    print("  TRUSTABLE AI - TUI DEMONSTRATION")
    print("─" * 60 + "\n")

    # Demo 1: Basic WorkflowTUI
    print("[Demo 1: Basic Workflow with Steps]\n")

    with WorkflowTUI(
        "Sprint Planning",
        sprint_name="Sprint 9",
        step_names=["Extract Items", "Prioritize", "Estimate", "Assign"]
    ) as tui:
        tui.say("Hi! I'm going to demonstrate the TUI components.")
        time.sleep(delay)

        # Step 1
        tui.start_step(1, "I'm extracting work items from the backlog...")
        time.sleep(delay)
        tui.complete_step("Found 5 items!")

        # Step 2
        tui.start_step(2, "Let me prioritize these by business value...")
        time.sleep(delay)
        tui.complete_step("Ranked by priority!")

        # Step 3 with agent
        tui.start_step(3, "Time for estimation...")
        time.sleep(delay / 2)

        with tui.agent("senior-engineer") as agent:
            agent.update("I'm analyzing the complexity of each item...")
            time.sleep(delay)
            agent.update("Almost done with estimations...")
            time.sleep(delay / 2)
            agent.complete("Estimated all 5 items!", tokens_input=1200, tokens_output=350)

        tui.complete_step("19 total story points")

        # Print work items
        tui.print_work_items(demo_work_items(), title="Sprint Backlog")

        # Step 4 with approval
        tui.start_step(4, "Ready to assign work items to the sprint")
        time.sleep(delay / 2)

        if not args.fast:
            choice = tui.approval_gate(
                title="Sprint Assignment",
                summary_lines=[
                    "5 work items selected (19 story points)",
                    "Utilization: 95% of 20 point capacity",
                    "All items have estimates"
                ],
                options=[
                    ("y", "Approve and assign to sprint"),
                    ("n", "Cancel (no changes)"),
                    ("e", "Edit selection")
                ]
            )
            if choice == "y":
                tui.complete_step("Assigned to Sprint 9!")
                tui.celebrate("Sprint planning complete!")
            else:
                tui.skip_step("User cancelled")
        else:
            tui.complete_step("Assigned to Sprint 9!")
            tui.celebrate("Sprint planning complete!")

    print("\n" + "─" * 60 + "\n")

    # Demo 2: Multi-agent consensus
    print("[Demo 2: Multi-Agent Consensus]\n")

    with WorkflowTUI("Capacity Planning", sprint_name="Sprint 9") as tui:
        tui.say("Let me get consensus from the team on sprint capacity...")
        time.sleep(delay)

        # Simulate agents providing estimates
        votes = {
            "business-analyst": 22,
            "senior-engineer": 18,
            "scrum-master": 20,
        }
        confidences = {
            "business-analyst": "HIGH",
            "senior-engineer": "HIGH",
            "scrum-master": "MEDIUM",
        }

        tui.show_consensus(
            topic="Sprint Capacity",
            votes=votes,
            confidences=confidences,
            final_value=20
        )

        tui.celebrate("Team agreed on 20 story points!")

    print("\n" + "─" * 60 + "\n")

    # Demo 3: Error handling with empathy
    print("[Demo 3: Friendly Error Messages]\n")

    from scripts.workflow_executor.tui.output_formatter import OutputFormatter
    from cli.console import console

    error_panel = OutputFormatter.friendly_error(
        error_type="Connection Error",
        message="I couldn't connect to Azure DevOps",
        suggestion="Run 'az login' to refresh your credentials",
        options=[
            ("r", "Retry connection"),
            ("s", "Skip this step"),
            ("q", "Quit")
        ]
    )
    console.print(error_panel)

    print("\n" + "─" * 60 + "\n")

    # Demo 4: Summary cards
    print("[Demo 4: Summary Cards]\n")

    summary = OutputFormatter.summary_card(
        title="Sprint Complete!",
        metrics={
            "Work Items": "12 completed",
            "Story Points": "47 delivered",
            "Velocity": "23.5 pts/week",
            "Completion Rate": "94.2%",
            "Bugs Fixed": 3,
        },
        style="success"
    )
    console.print(summary)

    print("\n" + "─" * 60)
    print("  DEMONSTRATION COMPLETE")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
