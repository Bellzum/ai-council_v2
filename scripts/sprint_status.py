#!/usr/bin/env python3
"""
Sprint Status - List open tickets with full parent hierarchy.

Displays all work items in a sprint with their complete parent chain
(Epic -> Feature -> Task/Bug), even if parents are not in the sprint.

Usage:
    python scripts/sprint_status.py --sprint "Sprint 9"
    python scripts/sprint_status.py --sprint "Sprint 9" --include-done
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add skills to path
skills_path = project_root / ".claude" / "skills"
if skills_path.exists():
    sys.path.insert(0, str(skills_path))

from work_tracking import get_adapter
from scripts.workflow_executor.tui import WorkflowTUI, OutputFormatter
from cli.console import console


def get_parent_hierarchy(adapter, work_item_id: int, visited: Set[int] = None) -> List[Dict]:
    """
    Get full parent hierarchy for a work item.

    Args:
        adapter: Work tracking adapter
        work_item_id: Work item ID to get hierarchy for
        visited: Set of visited IDs to prevent infinite loops

    Returns:
        List of work items from top-level parent down to this item
    """
    if visited is None:
        visited = set()

    # Prevent infinite loops
    if work_item_id in visited:
        return []
    visited.add(work_item_id)

    # Get work item
    work_item = adapter.get_work_item(work_item_id)
    if not work_item:
        return []

    # Extract fields
    fields = work_item.get("fields", {})
    parent_id = None

    # Try to get parent from relations
    relations = work_item.get("relations", [])
    for relation in relations:
        if relation.get("rel") == "System.LinkTypes.Hierarchy-Reverse":
            # Parent link
            url = relation.get("url", "")
            if url:
                # Extract ID from URL (last segment)
                parent_id = int(url.rstrip("/").split("/")[-1])
                break

    # Build hierarchy
    hierarchy = []

    # Recurse up parent chain
    if parent_id:
        hierarchy = get_parent_hierarchy(adapter, parent_id, visited)

    # Add current item
    hierarchy.append({
        "id": work_item.get("id"),
        "type": fields.get("System.WorkItemType", "Unknown"),
        "title": fields.get("System.Title", "Untitled"),
        "state": fields.get("System.State", "Unknown"),
        "parent_id": parent_id
    })

    return hierarchy


def display_hierarchy_tree(items_by_parent: Dict[Optional[int], List[Dict]],
                           parent_id: Optional[int] = None,
                           indent: int = 0,
                           displayed: Set[int] = None) -> None:
    """
    Display work items in a tree structure.

    Args:
        items_by_parent: Dict mapping parent ID to list of children
        parent_id: Current parent ID (None for root level)
        indent: Current indentation level
        displayed: Set of already displayed item IDs
    """
    if displayed is None:
        displayed = set()

    children = items_by_parent.get(parent_id, [])

    for item in children:
        item_id = item["id"]

        # Skip if already displayed (prevent duplicates)
        if item_id in displayed:
            continue
        displayed.add(item_id)

        # Format indentation with tree characters
        if indent == 0:
            prefix = ""
        else:
            prefix = "  " * (indent - 1) + "├─ "

        # Type badges with colors
        type_styles = {
            "Epic": ("magenta", "E"),
            "Feature": ("blue", "F"),
            "Task": ("green", "T"),
            "Bug": ("red", "B"),
            "User Story": ("cyan", "U")
        }
        color, symbol = type_styles.get(item["type"], ("white", "*"))

        # State styling
        state = item["state"]
        state_style = "dim"
        if state in ("Done", "Closed", "Resolved"):
            state_style = "#71E4D1"  # Success color
        elif state in ("In Progress", "Active"):
            state_style = "#BB93DD"  # Accent3 color
        elif state == "New":
            state_style = "#94A5CC"  # Tertiary

        # Format output
        console.print(
            f"{prefix}[bold {color}][{symbol}][/bold {color}] "
            f"[{color}]{item['type']}[/{color}] "
            f"[#67CFEE]#{item['id']}[/#67CFEE]  "
            f"[{state_style}]{state}[/{state_style}]  "
            f"[#D9EAFC]{item['title']}[/#D9EAFC]"
        )

        # Recurse to children
        display_hierarchy_tree(items_by_parent, item_id, indent + 1, displayed)


def main():
    parser = argparse.ArgumentParser(
        description="List open tickets in a sprint with full parent hierarchy"
    )
    parser.add_argument(
        "--sprint",
        required=True,
        help="Sprint name (e.g., 'Sprint 9')"
    )
    parser.add_argument(
        "--include-done",
        action="store_true",
        help="Include Done/Closed items (default: only open items)"
    )

    args = parser.parse_args()

    with WorkflowTUI(
        "Sprint Status",
        sprint_name=args.sprint,
        step_names=["Initialize", "Query Sprint", "Build Hierarchy", "Display"],
        show_agent_panel=False
    ) as tui:

        # Step 1: Initialize adapter
        tui.start_step(1, "I'm connecting to your work tracking system...")
        try:
            adapter = get_adapter()
            tui.complete_step("Connected!")
        except Exception as e:
            tui.error(
                f"I couldn't connect to the work tracking system: {e}",
                suggestion="Make sure you're in a project with .claude/config.yaml"
            )
            return 1

        # Step 2: Query sprint work items
        tui.start_step(2, f"I'm fetching work items from {args.sprint}...")
        try:
            sprint_items = adapter.query_sprint_work_items(args.sprint)
        except Exception as e:
            tui.error(f"I had trouble querying the sprint: {e}")
            return 1

        if not sprint_items:
            tui.complete_step("No items found")
            tui.say(f"I didn't find any work items in {args.sprint}.")
            return 0

        tui.complete_step(f"Found {len(sprint_items)} items!")

        # Step 3: Build hierarchy
        tui.start_step(3, "I'm building the work item hierarchy...")
        all_items = {}  # id -> item
        sprint_item_ids = set()

        for sprint_item in sprint_items:
            item_id = sprint_item.get("id")
            sprint_item_ids.add(item_id)

            # Get full hierarchy for this item
            hierarchy = get_parent_hierarchy(adapter, item_id)

            # Add all items in hierarchy to our collection
            for item in hierarchy:
                item_id = item["id"]
                if item_id not in all_items:
                    all_items[item_id] = item

        # Filter by state if needed
        if not args.include_done:
            done_states = {"Done", "Closed", "Resolved", "Removed"}
            filtered_items = {}
            for item_id, item in all_items.items():
                # Always include parents even if done (for hierarchy)
                # Only filter out sprint items that are done
                if item_id in sprint_item_ids and item["state"] in done_states:
                    continue
                filtered_items[item_id] = item
            all_items = filtered_items

        # Group items by parent for tree display
        items_by_parent = {}
        for item in all_items.values():
            parent_id = item.get("parent_id")
            if parent_id not in items_by_parent:
                items_by_parent[parent_id] = []
            items_by_parent[parent_id].append(item)

        # Sort by type (Epic, Feature, then others) and ID
        type_order = {"Epic": 0, "Feature": 1, "User Story": 2, "Task": 3, "Bug": 4}
        for parent_id in items_by_parent:
            items_by_parent[parent_id].sort(
                key=lambda x: (type_order.get(x["type"], 99), x["id"])
            )

        tui.complete_step(f"Built hierarchy with {len(all_items)} total items")

        # Step 4: Display
        tui.start_step(4, "Here's what I found...")

        # Display header
        tui.print()
        console.print(f"[bold #71E4D1]{args.sprint} - Work Item Hierarchy[/bold #71E4D1]")
        console.print("─" * 70)
        tui.print()

        # Display tree starting from root (items with no parent)
        display_hierarchy_tree(items_by_parent)

        # Summary
        tui.print()
        console.print("─" * 70)

        # Count by type and state
        type_counts = {}
        state_counts = {}
        sprint_open = 0

        for item_id, item in all_items.items():
            item_type = item["type"]
            item_state = item["state"]

            type_counts[item_type] = type_counts.get(item_type, 0) + 1
            state_counts[item_state] = state_counts.get(item_state, 0) + 1

            # Count open items that are actually in the sprint
            if item_id in sprint_item_ids and item_state not in {"Done", "Closed", "Resolved"}:
                sprint_open += 1

        # Print summary card
        tui.print_summary(
            "Summary",
            {
                "Total Items": len(all_items),
                "Open in Sprint": sprint_open,
                "By Type": ", ".join(f"{t}: {c}" for t, c in sorted(type_counts.items())),
            },
            style="info"
        )

        tui.complete_step("All done!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
