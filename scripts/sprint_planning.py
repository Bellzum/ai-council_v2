#!/usr/bin/env python3
"""
Sprint Planning Workflow with External Enforcement

8-Step workflow for sprint planning that focuses on groomed Tasks and Bugs
from the backlog. Test plans, architecture review, and security review are
now handled during backlog-grooming, so this workflow focuses on:

1. Extracting groomed Tasks and Bugs from backlog
2. Business Analyst prioritization
3. Senior Engineer estimation
4. Scrum Master assembly
5. Approval gate (BLOCKING)
6. Create sprint iteration
7. Assign work items to sprint
8. Validate quality

Usage:
    python3 scripts/sprint_planning.py --sprint-number 8 --capacity 80
    python3 scripts/sprint_planning.py --sprint-number 8 --capacity 80 --use-ai
    python3 scripts/sprint_planning.py --sprint-number 8 --capacity 80 --interactive
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import workflow executor base
from scripts.workflow_executor.base import WorkflowOrchestrator, ExecutionMode

# Import standardized argument parser
from scripts.workflow_executor.argument_config import (
    WorkflowArgumentParser,
    StandardArguments
)

# Import work tracking adapter
sys.path.insert(0, '.claude/skills')
from work_tracking import get_adapter

# Import unified console functions
from core.console_workflow import (
    print_workflow_header,
    print_step_header,
    print_work_items_table,
    print_approval_gate,
    print_metrics_table,
    print_summary_panel,
    WorkItem,
    WorkItemType,
    CapacityMetrics,
    ApprovalGateData
)

# Import JSON schema validation
try:
    from jsonschema import validate, ValidationError
except ImportError:
    from core.console_workflow import print_warning
    print_warning("jsonschema package not installed - install with: pip install jsonschema")
    ValidationError = Exception  # Fallback


class SprintPlanningWorkflow(WorkflowOrchestrator):
    """
    Sprint Planning workflow with external enforcement.

    This workflow implements 8 steps focusing on groomed Tasks and Bugs:
    - Multi-agent orchestration (3 agents: BA, Engineer, Scrum Master)
    - Falsifiability checks (description >= 500 chars, AC >= 3)
    - Blocking approval gate
    - External verification of work item assignment

    Note: Test plans, architecture review, and security review are now
    handled during backlog-grooming, not during sprint planning.
    """

    def __init__(
        self,
        sprint_number: int,
        team_capacity: int,
        use_ai: bool = True,  # Default to AI mode (non-interactive)
        interactive: bool = False,
        args: Optional[argparse.Namespace] = None
    ):
        """
        Initialize sprint planning workflow.

        Args:
            sprint_number: Sprint number (e.g., 8)
            team_capacity: Team capacity in story points
            use_ai: If True, spawn AI agents (Mode 2) - default is True for non-interactive AI
            interactive: If True, use Mode 3 interactive collaboration with Claude Agent SDK
            args: Command-line arguments for targeted planning
        """
        self.sprint_number = sprint_number
        self.sprint_name = f"Sprint {sprint_number}"
        self.team_capacity = team_capacity
        self.use_ai = use_ai
        self.interactive = interactive
        self.args = args  # Store for use in workflow steps

        # Interactive mode overrides use_ai (Mode 3 vs Mode 2)
        if interactive and use_ai:
            from cli.console import print_warning
            print_warning("Both --use-ai and --interactive specified - using interactive mode (Mode 3)")
            self.use_ai = False  # Interactive mode takes precedence

        # Initialize work tracking adapter
        try:
            self.adapter = get_adapter()
        except Exception as e:
            from cli.console import print_warning
            print_warning(f"Warning: Could not initialize work tracking adapter: {e}")
            print_warning("    Continuing with limited functionality...")
            self.adapter = None

        # Initialize Claude API client if using AI
        # Use KEYCHAIN_ANTHROPIC_API_KEY for automated scripts
        # This keeps ANTHROPIC_API_KEY unset in interactive sessions (uses Max subscription)
        # while automated scripts use the API key (uses API credits)
        self.claude_client = None
        self.token_usage: Dict[str, Dict[str, Any]] = {}
        if use_ai:
            try:
                from anthropic import Anthropic  # type: ignore
                api_key = os.environ.get("KEYCHAIN_ANTHROPIC_API_KEY")
                if api_key:
                    self.claude_client = Anthropic(api_key=api_key)
                    from cli.console import print_success
                    print_success("✓ Claude API initialized")
                else:
                    from cli.console import print_warning
                    print_warning("KEYCHAIN_ANTHROPIC_API_KEY not set - falling back to mock data")
                    self.use_ai = False  # Disable AI if no API key
            except ImportError:
                from cli.console import print_warning
                print_warning("anthropic package not installed - falling back to mock data")
                self.use_ai = False  # Disable AI if anthropic not installed

        # Initialize interactive session if interactive mode
        self.interactive_session = None
        if interactive:
            try:
                from scripts.workflow_executor.interactive_session import InteractiveSession
                self.interactive_session = InteractiveSession(
                    workflow_name="sprint-planning",
                    session_id=self.sprint_name.replace(' ', '-'),
                    model="claude-sonnet-4-5",
                    max_tokens=4000
                )
                if self.interactive_session.is_available():
                    from cli.console import print_success
                    print_success("✓ Interactive mode initialized (Mode 3)")
                else:
                    from cli.console import print_warning
                    print_warning("Interactive mode unavailable - falling back to AI mode")
                    self.interactive = False
                    self.use_ai = True
            except ImportError as e:
                from cli.console import print_warning
                print_warning(f"Interactive mode unavailable: {e}")
                print_warning("    Falling back to AI mode")
                self.interactive = False
                self.use_ai = True

        # Determine execution mode
        if interactive:
            mode = ExecutionMode.INTERACTIVE_AI  # Mode 3
        elif use_ai:
            mode = ExecutionMode.AI_JSON_VALIDATION  # Mode 2
        else:
            mode = ExecutionMode.PURE_PYTHON  # Mode 1

        # Include timestamp in workflow ID to ensure fresh data on each run
        # (prevents caching external state like work item queries across runs)
        workflow_id = f"{self.sprint_name.replace(' ', '-')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        super().__init__(
            workflow_name="sprint-planning",
            workflow_id=workflow_id,
            mode=mode,
            enable_checkpoints=True
        )

    def _define_steps(self) -> List[Dict[str, Any]]:
        """Define all 8 workflow steps."""
        return [
            {"id": "1-extract-work-items", "name": "Extract Groomed Tasks and Bugs"},
            {"id": "2-prioritize", "name": "Business Analyst Prioritization"},
            {"id": "3-estimate", "name": "Senior Engineer Estimation"},
            {"id": "4-assemble", "name": "Scrum Master Assembly"},
            {"id": "5-approval", "name": "Approval Gate (BLOCKING)"},
            {"id": "6-create-iteration", "name": "Verify/Create Sprint Iteration"},
            {"id": "7-assign-work-items", "name": "Assign Work Items to Sprint"},
            {"id": "8-validate-quality", "name": "Validate Quality Standards"},
        ]

    def _execute_step(
        self,
        step: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single workflow step."""
        step_id = step["id"]

        # Route to appropriate step handler
        if step_id == "1-extract-work-items":
            return self._step_1_extract_work_items()
        elif step_id == "2-prioritize":
            return self._step_2_prioritize()
        elif step_id == "3-estimate":
            return self._step_3_estimate()
        elif step_id == "4-assemble":
            return self._step_4_assemble()
        elif step_id == "5-approval":
            return self._step_5_approval()
        elif step_id == "6-create-iteration":
            return self._step_6_create_iteration()
        elif step_id == "7-assign-work-items":
            return self._step_7_assign_work_items()
        elif step_id == "8-validate-quality":
            return self._step_8_validate_quality()
        else:
            raise ValueError(f"Unknown step: {step_id}")

    # ========================================================================
    # Step Implementations
    # ========================================================================

    def _step_1_extract_work_items(self) -> Dict[str, Any]:
        """Step 1: Extract groomed Tasks and Bugs from backlog."""
        if self.interactive:
            mode_str = "Interactive AI (Mode 3)"
        elif self.use_ai:
            mode_str = "AI-Assisted (Mode 2)"
        else:
            mode_str = "Pure Python (Mode 1)"

        # Print workflow header using unified console
        print_workflow_header("Sprint Planning", sprint_name=self.sprint_name, mode=mode_str)

        # Print capacity metrics
        print_metrics_table(
            CapacityMetrics(
                total_capacity=self.team_capacity,
                used_capacity=0,
                available_capacity=self.team_capacity,
                work_item_count=0,
                utilization_percent=0.0
            ),
            title="Initial Capacity"
        )

        print_step_header(1, "Extract Groomed Tasks and Bugs", "Querying groomed Tasks and Bugs from backlog")

        # Check for argument-based filtering
        include_item_ids = []
        exclude_item_ids = []
        include_ungroomed = False

        if self.args:
            include_item_ids = getattr(self.args, 'include_item_ids', []) or []
            exclude_item_ids = getattr(self.args, 'exclude_item_ids', []) or []
            include_ungroomed = getattr(self.args, 'include_ungroomed', False)

            if include_item_ids:
                from cli.console import print_info
                print_info(f"📌 Including specific items: {', '.join(map(str, include_item_ids))} (argument provided)")
            if exclude_item_ids:
                print_info(f"📌 Excluding items: {', '.join(map(str, exclude_item_ids))} (argument provided)")
            if include_ungroomed:
                print_info(f"📌 Including ungroomed items (--include-ungroomed)")

        if not self.adapter:
            from cli.console import print_warning
            print_warning("No adapter - using mock work items")
            work_items = [
                {
                    "id": 1001,
                    "fields": {
                        "System.Title": "Implement login page",
                        "System.Description": "Create the user login page with email/password authentication. ## Detailed Design\nThis task implements the login functionality with unit tests, integration tests, and acceptance criteria.",
                        "System.WorkItemType": "Task",
                        "System.State": "New",
                        "System.Tags": "groomed; implementation",
                        "Microsoft.VSTS.Scheduling.StoryPoints": 3
                    }
                },
                {
                    "id": 1002,
                    "fields": {
                        "System.Title": "Fix crash on logout",
                        "System.Description": "Application crashes when user logs out from settings page. ## Root Cause Analysis\nThe issue is caused by a null reference. ## Solution\nFix by checking reference before use.",
                        "System.WorkItemType": "Bug",
                        "System.State": "New",
                        "System.Tags": "groomed",
                        "Microsoft.VSTS.Common.Severity": "2 - High"
                    }
                }
            ]
        else:
            try:
                # Query for Tasks and Bugs in backlog (not assigned to a sprint yet)
                all_items = self.adapter.query_work_items()
                candidate_items = [
                    item for item in all_items
                    if item.get("fields", {}).get("System.WorkItemType") in ["Task", "Bug"]
                    and item.get("fields", {}).get("System.State") not in ["Done", "Closed", "Removed"]
                    # Only include items not yet assigned to a sprint iteration
                    and not self._is_assigned_to_sprint(item)
                ]

                # Apply exclusions first
                if exclude_item_ids:
                    before_count = len(candidate_items)
                    candidate_items = [
                        item for item in candidate_items
                        if item.get("id") not in exclude_item_ids
                    ]
                    from cli.console import print_success
                    print_success(f"  ✓ Excluded {before_count - len(candidate_items)} item(s) (filtered by --exclude-item-ids)")

                # Handle explicitly included items separately (they bypass grooming check)
                explicitly_included = []
                explicitly_included_ungroomed = []

                if include_item_ids:
                    # Find explicitly included items from all_items (not just candidate_items)
                    # because we want to include them even if they don't meet basic filters
                    for item in all_items:
                        item_id = item.get("id")
                        if item_id in include_item_ids:
                            is_groomed, reason = self._is_groomed(item)
                            explicitly_included.append(item)
                            if not is_groomed:
                                explicitly_included_ungroomed.append({
                                    "item": item,
                                    "reason": reason
                                })

                    # Warn about ungroomed explicitly included items
                    if explicitly_included_ungroomed:
                        from cli.console import print_warning, console
                        print_warning(f"\nWARNING: {len(explicitly_included_ungroomed)} explicitly included item(s) are NOT groomed:")
                        for entry in explicitly_included_ungroomed:
                            item = entry["item"]
                            item_id = item.get("id", "Unknown")
                            item_title = item.get("fields", {}).get("System.Title", "Untitled")[:50]
                            console.print(f"    - #{item_id}: {item_title}... ({entry['reason']})")
                        print_warning(f"\n    Best practice: Run /backlog-grooming --work-item-ids {' '.join(str(e['item'].get('id')) for e in explicitly_included_ungroomed)}")
                        console.print(f"    → Continuing anyway since items were explicitly specified...")

                    from cli.console import print_success
                    print_success(f"  ✓ Including {len(explicitly_included)} explicitly specified item(s) (--include-item-ids)")

                    # Use explicitly included items as the work items
                    work_items = explicitly_included
                    ungroomed_items = []  # Already handled above
                else:
                    # No explicit inclusion - filter by grooming status
                    work_items = []
                    ungroomed_items = []

                    for item in candidate_items:
                        is_groomed, reason = self._is_groomed(item)
                        if is_groomed:
                            work_items.append(item)
                        else:
                            ungroomed_items.append({"item": item, "reason": reason})

                    # Report groomed vs ungroomed
                    from cli.console import print_success, print_warning, console
                    print_success(f"\n✓ Found {len(work_items)} groomed work item(s) ready for sprint")

                    if ungroomed_items:
                        print_warning(f"{len(ungroomed_items)} ungroomed item(s) excluded:")
                        for entry in ungroomed_items[:5]:  # Show first 5
                            item = entry["item"]
                            item_id = item.get("id", "Unknown")
                            item_title = item.get("fields", {}).get("System.Title", "Untitled")[:50]
                            console.print(f"    - #{item_id}: {item_title}... ({entry['reason']})")
                        if len(ungroomed_items) > 5:
                            console.print(f"    ... and {len(ungroomed_items) - 5} more")
                        console.print(f"\n    → Run /backlog-grooming to prepare these items")
                        console.print(f"    → Or use --include-ungroomed to include them anyway")

                    # Include ungroomed items if flag is set
                    if include_ungroomed and ungroomed_items:
                        from cli.console import print_info
                        print_info(f"\n📌 Including {len(ungroomed_items)} ungroomed item(s) per --include-ungroomed flag")
                        work_items.extend([entry["item"] for entry in ungroomed_items])

                # Display work items using unified console function
                if work_items:
                    formatted_items = []
                    for item in work_items:
                        fields = item.get("fields", {})
                        item_type_str = fields.get("System.WorkItemType", "Task")
                        item_type = WorkItemType.TASK if item_type_str == "Task" else WorkItemType.BUG

                        formatted_items.append(WorkItem(
                            id=item.get("id", 0),
                            title=fields.get("System.Title", "Untitled"),
                            type=item_type,
                            state=fields.get("System.State", "New"),
                            story_points=fields.get("Microsoft.VSTS.Scheduling.StoryPoints"),
                            tags=fields.get("System.Tags", "")
                        ))

                    print_work_items_table(formatted_items, title="Groomed Work Items")

            except Exception as e:
                from cli.console import print_error
                print_error(f"Error querying work items: {e}")
                work_items = []

        return {
            "work_item_count": len(work_items),
            "work_items": work_items,
            "task_count": len([i for i in work_items if i.get("fields", {}).get("System.WorkItemType") == "Task"]),
            "bug_count": len([i for i in work_items if i.get("fields", {}).get("System.WorkItemType") == "Bug"])
        }

    def _is_groomed(self, item: Dict[str, Any]) -> tuple:
        """
        Check if a work item has been groomed (ready for sprint).

        A work item is considered groomed if:
        1. It has the 'groomed' tag (set by backlog-grooming workflow), OR
        2. It passes the grooming validation criteria for its type

        Returns:
            tuple: (is_groomed: bool, reason: str)
        """
        fields = item.get("fields", {})
        item_type = fields.get("System.WorkItemType", "")
        tags = (fields.get("System.Tags", "") or "").lower()

        # Check for 'groomed' tag first (fastest check)
        if "groomed" in tags:
            return True, "tagged"

        # Fallback: Validate against grooming criteria
        if item_type == "Task":
            return self._validate_task_grooming(item)
        elif item_type == "Bug":
            return self._validate_bug_grooming(item)
        else:
            return False, f"unknown type: {item_type}"

    def _validate_task_grooming(self, task: Dict[str, Any]) -> tuple:
        """
        Validate a Task against grooming criteria.

        Requirements (from backlog-grooming):
        - Detailed design (implementation approach)
        - Unit test design
        - Integration test design
        - Acceptance test design

        Returns:
            tuple: (is_groomed: bool, reason: str)
        """
        fields = task.get("fields", {})
        description = (fields.get("System.Description", "") or "").lower()

        missing = []

        # Check for detailed design (description > 200 chars with design keywords)
        has_detailed_design = (
            len(description) > 200 and
            any(kw in description for kw in [
                'design', 'approach', 'implementation', 'algorithm',
                'component', 'class', 'function', 'method', 'module',
                'step', 'process', 'flow', 'logic'
            ])
        )
        if not has_detailed_design:
            missing.append("detailed_design")

        # Check for unit test design
        has_unit_tests = any(kw in description for kw in [
            'unit test', 'unit-test', 'unittest', 'test case',
            'mock', 'stub', 'assert', 'expect', 'test coverage'
        ])
        if not has_unit_tests:
            missing.append("unit_tests")

        # Check for integration test design
        has_integration_tests = any(kw in description for kw in [
            'integration test', 'integration-test', 'e2e', 'end-to-end',
            'api test', 'system test', 'component test'
        ])
        if not has_integration_tests:
            missing.append("integration_tests")

        # Check for acceptance test design
        has_acceptance_tests = any(kw in description for kw in [
            'acceptance test', 'acceptance-test', 'acceptance criteria',
            'user acceptance', 'uat', 'bdd', 'given', 'when', 'then'
        ])
        if not has_acceptance_tests:
            missing.append("acceptance_tests")

        if missing:
            return False, f"missing: {', '.join(missing)}"
        return True, "validated"

    def _validate_bug_grooming(self, bug: Dict[str, Any]) -> tuple:
        """
        Validate a Bug against grooming criteria.

        Requirements (from backlog-grooming):
        - Reproduction steps
        - Root cause analysis
        - Solution design
        - Acceptance test design

        Returns:
            tuple: (is_groomed: bool, reason: str)
        """
        fields = bug.get("fields", {})
        description = (fields.get("System.Description", "") or "").lower()
        repro_steps = (fields.get("Microsoft.VSTS.TCM.ReproSteps", "") or "").lower()
        combined = description + " " + repro_steps

        missing = []

        # Check for reproduction steps
        has_repro_steps = (
            len(repro_steps) > 50 or
            any(kw in combined for kw in [
                'steps to reproduce', 'reproduction steps', 'repro steps',
                'step 1', '1.', '1)', 'to reproduce', 'how to reproduce'
            ])
        )
        if not has_repro_steps:
            missing.append("repro_steps")

        # Check for root cause analysis
        has_root_cause = any(kw in combined for kw in [
            'root cause', 'cause', 'reason', 'because', 'due to',
            'investigation', 'analysis', 'the issue is', 'the problem is'
        ])
        if not has_root_cause:
            missing.append("root_cause")

        # Check for solution design
        has_solution = any(kw in combined for kw in [
            'solution', 'fix', 'resolution', 'approach', 'to resolve',
            'will fix', 'should fix', 'proposed fix', 'remediation'
        ])
        if not has_solution:
            missing.append("solution")

        # Check for acceptance test design
        has_acceptance = any(kw in combined for kw in [
            'acceptance test', 'verify', 'validate', 'test case',
            'acceptance criteria', 'how to test', 'verification steps'
        ])
        if not has_acceptance:
            missing.append("acceptance")

        if missing:
            return False, f"missing: {', '.join(missing)}"
        return True, "validated"

    def _is_assigned_to_sprint(self, item: Dict[str, Any]) -> bool:
        """Check if a work item is already assigned to a sprint iteration."""
        fields = item.get("fields", {})
        iteration_path = fields.get("System.IterationPath", "")

        # Check if iteration path contains "Sprint" (indicating sprint assignment)
        # This is a simple heuristic - adjust based on your iteration path structure
        if "Sprint" in iteration_path:
            return True

        # Check if iteration level is > 1 (i.e., assigned to a child iteration)
        iteration_level = fields.get("System.IterationLevel", 0)
        if iteration_level > 1:
            return True

        return False

    def _step_2_prioritize(self) -> Dict[str, Any]:
        """Step 2: Business analyst prioritization of Tasks and Bugs."""
        work_items = self.step_evidence.get("1-extract-work-items", {}).get("work_items", [])

        print_step_header(2, "Business Analyst Prioritization", "Analyzing and prioritizing work items")

        if not work_items:
            from cli.console import print_warning
            print_warning("No work items to prioritize")
            return {"priorities": []}

        if self.interactive:
            priorities = self._prioritize_interactive(work_items)
        elif self.use_ai:
            priorities = self._prioritize_ai(work_items)
        else:
            priorities = self._prioritize_simple(work_items)

        # Sort by priority score descending
        priorities.sort(key=lambda x: x['priority_score'], reverse=True)

        from core.console_workflow import print_section_divider
        from cli.console import console
        print_section_divider("Prioritized Work Items")
        for priority in priorities:
            item_type = priority.get('work_item_type', 'Item')
            console.print(f"  {item_type} [accent2]#{priority['work_item_id']}[/accent2]: Priority [bold]{priority['priority_score']}/10[/bold]")
            console.print(f"    [dim]Rationale: {priority['rationale'][:100]}...[/dim]")

        return {
            "priorities": priorities
        }

    def _prioritize_simple(self, work_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Simple prioritization using real work item data (no AI).

        Uses work item fields to calculate priority:
        - Bugs get higher priority than Tasks
        - Severity (for Bugs) affects priority
        - Backlog order (earlier in list = higher priority)
        - Existing story points (if present)

        Args:
            work_items: Real Tasks and Bugs from adapter

        Returns:
            Priority scores based on real data, simple logic
        """
        priorities = []
        for idx, item in enumerate(work_items):
            item_id = item.get("id", 0)
            fields = item.get("fields", {})
            item_type = fields.get("System.WorkItemType", "Task")

            # Base priority from backlog position
            backlog_position_score = max(1, 10 - idx)

            # Type bonus: Bugs get +2
            type_bonus = 2 if item_type == "Bug" else 0

            # Severity bonus for Bugs
            severity_bonus = 0
            if item_type == "Bug":
                severity = fields.get("Microsoft.VSTS.Common.Severity", "3 - Medium")
                if "1 - Critical" in severity:
                    severity_bonus = 4
                elif "2 - High" in severity:
                    severity_bonus = 2
                elif "3 - Medium" in severity:
                    severity_bonus = 1

            # Priority field bonus (if set)
            priority_field = fields.get("Microsoft.VSTS.Common.Priority", 3)
            priority_bonus = max(0, 4 - priority_field)  # Priority 1 = +3, Priority 4 = 0

            # Combine factors
            priority_score = min(10, max(1, backlog_position_score + type_bonus + severity_bonus + priority_bonus))

            priorities.append({
                "work_item_id": item_id,
                "work_item_type": item_type,
                "priority_score": priority_score,
                "rationale": f"Based on backlog position (#{idx+1}), type ({item_type}), severity/priority fields"
            })
        return priorities

    def _prioritize_ai(self, work_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        AI-based prioritization using business-analyst agent (Mode 2).

        Uses Claude Agent SDK with tool access so the agent can:
        - Read CLAUDE.md for project context
        - Explore the codebase to understand technical dependencies

        Returns:
            List of dicts with prioritized work items and scores
        """
        # Define JSON schema for AI response
        schema = {
            "type": "object",
            "properties": {
                "prioritized_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "work_item_id": {"type": "integer"},
                            "work_item_type": {"type": "string"},
                            "priority_score": {"type": "integer", "minimum": 1, "maximum": 10},
                            "rationale": {"type": "string", "minLength": 50}
                        },
                        "required": ["work_item_id", "work_item_type", "priority_score", "rationale"]
                    }
                }
            },
            "required": ["prioritized_items"]
        }

        # Build prompt with work item details
        item_descriptions = "\n\n".join([
            f"{item.get('fields', {}).get('System.WorkItemType', 'Item')} #{item.get('id')}:\n"
            f"Title: {item.get('fields', {}).get('System.Title', 'Untitled')}\n"
            f"Description: {item.get('fields', {}).get('System.Description', 'No description')[:500]}\n"
            f"State: {item.get('fields', {}).get('System.State', 'New')}\n"
            f"Story Points: {item.get('fields', {}).get('Microsoft.VSTS.Scheduling.StoryPoints', 'Not set')}\n"
            f"Severity: {item.get('fields', {}).get('Microsoft.VSTS.Common.Severity', 'N/A')}"
            for item in work_items
        ])

        prompt = f"""You are a business analyst. Prioritize these Tasks and Bugs for sprint planning.

**IMPORTANT: First read the project context**

Before prioritizing, use your Read tool to read `CLAUDE.md` in the project root to understand:
- Project purpose and architecture
- Current priorities and constraints
- Technical stack and dependencies

This will help you make accurate prioritization decisions.

Work items to prioritize:
{item_descriptions}

Team Capacity: {self.team_capacity} story points

Analyze each work item and assign a priority score (1-10) based on:
- Business value and user impact
- Bug severity (critical bugs should be highest priority)
- Dependencies and blockers
- Risk of delay
- Strategic alignment with project goals (from CLAUDE.md)

Return ONLY valid JSON matching this exact schema:
{json.dumps(schema, indent=2)}

Ensure each rationale is at least 50 characters explaining the priority score."""

        # Use Agent SDK with tool access for codebase exploration
        try:
            from scripts.workflow_executor.agent_sdk import AgentSDKWrapper
            import asyncio

            print("   Using Agent SDK with tool access...")

            wrapper = AgentSDKWrapper(
                workflow_name="sprint-planning-prioritize",
                tool_preset="read_only",  # Read, Grep, Glob
                max_turns=15,
                model="claude-sonnet-4-5",
            )

            async def _run_prioritize():
                return await wrapper.query(
                    prompt=prompt,
                    agent_type="business-analyst",
                )

            # Execute async
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, _run_prioritize())
                        result = future.result()
                else:
                    result = loop.run_until_complete(_run_prioritize())
            except RuntimeError:
                result = asyncio.run(_run_prioritize())

            if not result.success:
                from cli.console import print_warning
                print_warning(f"Agent SDK query failed: {result.error}")
                return self._prioritize_simple(work_items)

            # Extract JSON from response
            response_text = result.response

            json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

            parsed_result = json.loads(response_text)
            validate(parsed_result, schema)

            # Track token usage
            self.token_usage["prioritize"] = {
                "input_tokens": result.token_usage.input_tokens,
                "output_tokens": result.token_usage.output_tokens,
                "cost_usd": result.cost_usd
            }

            return parsed_result["prioritized_items"]

        except ImportError as e:
            # ALWAYS report import errors - no silent fallbacks
            from cli.console import print_error, print_info
            print_error(f"AI Prioritization Failed: Import error: {e}")
            print_error("The Claude Agent SDK is not available. AI prioritization requires tool access.")
            print_info("Install with: pip install claude-code-sdk")
            print_info("Workflow will continue with simple priority ordering (by type and ID).")
            return self._prioritize_simple(work_items)

        except (json.JSONDecodeError, ValidationError) as e:
            # Report validation errors clearly
            from cli.console import print_error, print_info
            print_error(f"AI Prioritization Failed: {type(e).__name__}: {e}")
            print_info("Workflow will continue with simple priority ordering (by type and ID).")
            return self._prioritize_simple(work_items)

        except Exception as e:
            # ALWAYS report errors - no silent fallbacks
            from cli.console import print_error, print_info
            print_error(f"AI Prioritization Failed: {type(e).__name__}: {e}")
            print_info("Workflow will continue with simple priority ordering (by type and ID).")
            return self._prioritize_simple(work_items)

    def _prioritize_anthropic_fallback(
        self,
        work_items: List[Dict[str, Any]],
        schema: Dict[str, Any],
        prompt: str
    ) -> List[Dict[str, Any]]:
        """Fallback to Anthropic API when Agent SDK is unavailable."""
        if not self.claude_client:
            return self._prioritize_simple(work_items)

        for attempt in range(3):
            try:
                response = self.claude_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}]
                )

                response_text = response.content[0].text

                json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(1)

                result = json.loads(response_text)
                validate(result, schema)

                self.token_usage["prioritize"] = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cost_usd": self._calculate_cost(response.usage)
                }

                return result["prioritized_items"]

            except (json.JSONDecodeError, ValidationError) as e:
                from cli.console import print_warning, print_error
                print_warning(f"Attempt {attempt + 1}/3 failed: {type(e).__name__}: {e}")
                if attempt == 2:
                    print_error("All retries exhausted, falling back to simple logic")
                    return self._prioritize_simple(work_items)
            except Exception as e:
                from cli.console import print_error
                print_error(f"API error: {type(e).__name__}: {e}, falling back to simple logic")
                return self._prioritize_simple(work_items)

        return self._prioritize_simple(work_items)

    def _prioritize_interactive(self, work_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Interactive prioritization with multi-turn collaboration (Mode 3).

        User can refine AI's priority scores through feedback until satisfied.

        Args:
            work_items: List of Task and Bug work items

        Returns:
            List of priority scores with rationale
        """
        if not self.interactive_session or not self.interactive_session.is_available():
            from cli.console import print_warning
            print_warning("Interactive mode unavailable - falling back to AI prioritization")
            return self._prioritize_ai(work_items)

        # Build context for AI
        item_summaries = []
        for item in work_items:
            item_id = item.get("id", 0)
            fields = item.get("fields", {})
            title = fields.get("System.Title", "Untitled")
            description = fields.get("System.Description", "No description")[:300]
            item_type = fields.get("System.WorkItemType", "Task")

            item_summaries.append({
                "id": item_id,
                "type": item_type,
                "title": title,
                "description": description
            })

        context = {
            "item_count": len(work_items),
            "work_items": item_summaries,
            "team_capacity": self.team_capacity
        }

        initial_prompt = f"""You are a business analyst helping prioritize {len(work_items)} Tasks and Bugs for sprint planning.

Analyze these work items and provide initial priority scores (1-10, where 10 is highest priority).

Consider:
- Business value (customer impact, revenue potential)
- Bug severity (critical bugs need immediate attention)
- Dependencies (blocking other work)
- Risk and urgency

Provide your analysis with priority scores and rationale for each item."""

        # Multi-turn discussion
        result = self.interactive_session.discuss(
            initial_prompt=initial_prompt,
            context=context,
            max_iterations=5
        )

        # Extract structured JSON from final response
        schema = {
            "type": "object",
            "properties": {
                "prioritized_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "work_item_id": {"type": "integer"},
                            "work_item_type": {"type": "string"},
                            "priority_score": {"type": "integer", "minimum": 1, "maximum": 10},
                            "rationale": {"type": "string", "minLength": 50}
                        },
                        "required": ["work_item_id", "work_item_type", "priority_score", "rationale"]
                    }
                }
            },
            "required": ["prioritized_items"]
        }

        # CRITICAL: Include the final AI response as context for JSON extraction.
        # The ask() method creates a fresh API call with no conversation history,
        # so we must explicitly provide the analysis to extract from.
        final_ai_analysis = result.get("final_response", "")

        extraction_prompt = f"""Extract priority scores from the following prioritization analysis.

## AI PRIORITIZATION ANALYSIS TO EXTRACT FROM:
{final_ai_analysis}

## JSON SCHEMA:
{json.dumps(schema, indent=2)}

Provide ONLY the JSON matching the schema above, no other text. Extract values directly from the analysis provided."""

        try:
            json_response = self.interactive_session.ask(extraction_prompt)
            priorities_data = self.interactive_session.extract_json_from_response(json_response, schema)

            return priorities_data["prioritized_items"]

        except Exception as e:
            from cli.console import print_warning
            print_warning(f"Failed to extract JSON: {e}")
            print_warning("    Falling back to AI prioritization")
            return self._prioritize_ai(work_items)

    def _step_3_estimate(self) -> Dict[str, Any]:
        """Step 3: Senior engineer estimation (validates existing estimates or provides new ones)."""
        work_items = self.step_evidence.get("1-extract-work-items", {}).get("work_items", [])
        priorities = self.step_evidence.get("2-prioritize", {}).get("priorities", [])

        print_step_header(3, "Senior Engineer Estimation", "Validating and estimating story points")

        if not work_items:
            from cli.console import print_warning
            print_warning("No work items to estimate")
            return {"estimates": [], "total_points": 0}

        if self.interactive:
            estimates = self._estimate_interactive(work_items, priorities)
        elif self.use_ai:
            estimates = self._estimate_ai(work_items, priorities)
        else:
            estimates = self._estimate_simple(work_items, priorities)

        # Calculate total points
        total_points = sum(e.get("story_points", 0) for e in estimates)

        # Check capacity fit using unified console
        available = self.team_capacity - total_points if total_points <= self.team_capacity else 0
        utilization = (total_points / self.team_capacity * 100) if self.team_capacity > 0 else 0

        print_metrics_table(
            CapacityMetrics(
                total_capacity=self.team_capacity,
                used_capacity=total_points,
                available_capacity=available,
                work_item_count=len(estimates),
                utilization_percent=utilization
            ),
            title="Estimation Summary"
        )

        if total_points > self.team_capacity:
            from cli.console import print_warning, console
            print_warning(f"Over capacity by {total_points - self.team_capacity} points")
            # Sort by priority and recommend what fits
            priority_map = {p["work_item_id"]: p["priority_score"] for p in priorities}
            estimates.sort(key=lambda x: priority_map.get(x["work_item_id"], 0), reverse=True)

            running_total = 0
            recommended = []
            deferred = []
            for est in estimates:
                if running_total + est.get("story_points", 0) <= self.team_capacity:
                    running_total += est.get("story_points", 0)
                    recommended.append(est)
                else:
                    deferred.append(est)

            console.print(f"\n[success]  Recommended for sprint ({running_total} pts):[/success]")
            for est in recommended:
                console.print(f"    - [accent2]#{est['work_item_id']}[/accent2]: {est['title']} ([bold]{est['story_points']} pts[/bold])")

            if deferred:
                console.print(f"\n[warning]  Deferred to next sprint:[/warning]")
                for est in deferred:
                    console.print(f"    - [accent2]#{est['work_item_id']}[/accent2]: {est['title']} ([bold]{est['story_points']} pts[/bold])")
        else:
            from cli.console import print_success
            print_success(f"  ✓ All items fit within capacity")

        return {
            "estimates": estimates,
            "total_points": total_points,
            "fits_capacity": total_points <= self.team_capacity
        }

    def _estimate_simple(
        self,
        work_items: List[Dict[str, Any]],
        priorities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Simple estimation using existing story points (no AI).

        Uses existing story points if set, otherwise assigns defaults based on type.

        Args:
            work_items: Real Tasks and Bugs from adapter
            priorities: Priority scores from previous step

        Returns:
            Estimates based on existing story points or defaults
        """
        estimates = []
        for item in work_items:
            item_id = item.get("id", 0)
            fields = item.get("fields", {})
            title = fields.get("System.Title", "Untitled")
            item_type = fields.get("System.WorkItemType", "Task")

            # Use existing story points if available
            story_points = fields.get("Microsoft.VSTS.Scheduling.StoryPoints")

            if story_points is None:
                # Default estimation based on type
                if item_type == "Bug":
                    # Bugs default to 2 points (investigation + fix)
                    story_points = 2
                else:
                    # Tasks default to 3 points
                    story_points = 3
                estimation_source = "default"
            else:
                estimation_source = "existing"

            estimates.append({
                "work_item_id": item_id,
                "work_item_type": item_type,
                "title": title,
                "story_points": int(story_points),
                "estimation_source": estimation_source,
                "notes": f"{'Using existing estimate' if estimation_source == 'existing' else 'Default estimate based on work item type'}"
            })

        return estimates

    def _estimate_ai(
        self,
        work_items: List[Dict[str, Any]],
        priorities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        AI-based estimation using senior-engineer agent (Mode 2).

        Uses Claude Agent SDK with tool access so the agent can:
        - Read CLAUDE.md for project context
        - Explore the codebase to assess technical complexity

        Returns:
            List of dicts with story point estimates
        """
        # Define JSON schema
        schema = {
            "type": "object",
            "properties": {
                "estimates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "work_item_id": {"type": "integer"},
                            "work_item_type": {"type": "string"},
                            "title": {"type": "string"},
                            "story_points": {"type": "integer", "minimum": 1, "maximum": 13},
                            "estimation_source": {"type": "string"},
                            "notes": {"type": "string"}
                        },
                        "required": ["work_item_id", "work_item_type", "title", "story_points", "notes"]
                    }
                }
            },
            "required": ["estimates"]
        }

        # Build comprehensive prompt
        priority_map = {p["work_item_id"]: p["priority_score"] for p in priorities}

        item_context = "\n\n".join([
            f"{item.get('fields', {}).get('System.WorkItemType', 'Item')} #{item.get('id')}:\n"
            f"Title: {item.get('fields', {}).get('System.Title', 'Untitled')}\n"
            f"Description: {item.get('fields', {}).get('System.Description', 'No description')[:500]}\n"
            f"Current Story Points: {item.get('fields', {}).get('Microsoft.VSTS.Scheduling.StoryPoints', 'Not set')}\n"
            f"Priority Score: {priority_map.get(item.get('id'), 'N/A')}/10"
            for item in work_items
        ])

        prompt = f"""You are a senior engineer. Review and estimate these Tasks and Bugs for sprint planning.

**IMPORTANT: First read the project context**

Before estimating, use your Read tool to read `CLAUDE.md` in the project root to understand:
- Project architecture and tech stack
- Code organization and patterns
- Testing requirements

If work items reference specific modules or features, use Grep/Glob to explore those areas
and assess technical complexity accurately.

Work items to estimate:
{item_context}

Team Capacity: {self.team_capacity} story points

For each work item:
1. If story points are already set, validate they're reasonable
2. If story points are not set, provide an estimate
3. Use Fibonacci scale: 1, 2, 3, 5, 8, 13

Consider:
- Technical complexity (explore the code if needed)
- Testing requirements
- Risk and uncertainty
- Dependencies

Return ONLY valid JSON matching this exact schema:
{json.dumps(schema, indent=2)}"""

        # Use Agent SDK with tool access for codebase exploration
        try:
            from scripts.workflow_executor.agent_sdk import AgentSDKWrapper
            import asyncio

            print("   Using Agent SDK with tool access...")

            wrapper = AgentSDKWrapper(
                workflow_name="sprint-planning-estimate",
                tool_preset="read_only",  # Read, Grep, Glob
                max_turns=15,
                model="claude-sonnet-4-5",
            )

            async def _run_estimate():
                return await wrapper.query(
                    prompt=prompt,
                    agent_type="senior-engineer",
                )

            # Execute async
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, _run_estimate())
                        result = future.result()
                else:
                    result = loop.run_until_complete(_run_estimate())
            except RuntimeError:
                result = asyncio.run(_run_estimate())

            if not result.success:
                from cli.console import print_warning
                print_warning(f"Agent SDK query failed: {result.error}")
                return self._estimate_simple(work_items, priorities)

            # Extract JSON from response
            response_text = result.response

            json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

            parsed_result = json.loads(response_text)
            validate(parsed_result, schema)

            # Track token usage
            self.token_usage["estimate"] = {
                "input_tokens": result.token_usage.input_tokens,
                "output_tokens": result.token_usage.output_tokens,
                "cost_usd": result.cost_usd
            }

            return parsed_result["estimates"]

        except ImportError as e:
            # ALWAYS report import errors - no silent fallbacks
            from cli.console import print_error, print_info
            print_error(f"AI Estimation Failed: Import error: {e}")
            print_error("The Claude Agent SDK is not available. AI estimation requires tool access.")
            print_info("Install with: pip install claude-code-sdk")
            print_info("Workflow will continue with simple estimation (based on item type).")
            return self._estimate_simple(work_items, priorities)

        except (json.JSONDecodeError, ValidationError) as e:
            # Report validation errors clearly
            from cli.console import print_error, print_info
            print_error(f"AI Estimation Failed: {type(e).__name__}: {e}")
            print_info("Workflow will continue with simple estimation (based on item type).")
            return self._estimate_simple(work_items, priorities)

        except Exception as e:
            # ALWAYS report errors - no silent fallbacks
            from cli.console import print_error, print_info
            print_error(f"AI Estimation Failed: {type(e).__name__}: {e}")
            print_info("Workflow will continue with simple estimation (based on item type).")
            return self._estimate_simple(work_items, priorities)

    def _estimate_anthropic_fallback(
        self,
        work_items: List[Dict[str, Any]],
        priorities: List[Dict[str, Any]],
        schema: Dict[str, Any],
        prompt: str
    ) -> List[Dict[str, Any]]:
        """Fallback to Anthropic API when Agent SDK is unavailable."""
        if not self.claude_client:
            return self._estimate_simple(work_items, priorities)

        for attempt in range(3):
            try:
                response = self.claude_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}]
                )

                response_text = response.content[0].text

                json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(1)

                result = json.loads(response_text)
                validate(result, schema)

                self.token_usage["estimate"] = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cost_usd": self._calculate_cost(response.usage)
                }

                return result["estimates"]

            except (json.JSONDecodeError, ValidationError) as e:
                from cli.console import print_warning, print_error
                print_warning(f"Attempt {attempt + 1}/3 failed: {type(e).__name__}: {e}")
                if attempt == 2:
                    print_error("All retries exhausted, falling back to simple logic")
                    return self._estimate_simple(work_items, priorities)
            except Exception as e:
                from cli.console import print_error
                print_error(f"API error: {type(e).__name__}: {e}, falling back to simple logic")
                return self._estimate_simple(work_items, priorities)

        return self._estimate_simple(work_items, priorities)

    def _estimate_interactive(
        self,
        work_items: List[Dict[str, Any]],
        priorities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Interactive estimation with user review (Mode 3).

        User can challenge estimates and request adjustments.

        Args:
            work_items: List of work items
            priorities: Priority scores from previous step

        Returns:
            Estimates with user-approved story points
        """
        if not self.interactive_session or not self.interactive_session.is_available():
            from cli.console import print_warning
            print_warning("Interactive mode unavailable - falling back to AI estimation")
            return self._estimate_ai(work_items, priorities)

        # Build context
        priority_map = {p["work_item_id"]: p for p in priorities}

        item_context = []
        for item in work_items:
            item_id = item.get("id", 0)
            fields = item.get("fields", {})
            title = fields.get("System.Title", "Untitled")
            description = fields.get("System.Description", "No description")[:300]
            item_type = fields.get("System.WorkItemType", "Task")
            existing_points = fields.get("Microsoft.VSTS.Scheduling.StoryPoints")

            priority_info = priority_map.get(item_id, {})

            item_context.append({
                "id": item_id,
                "type": item_type,
                "title": title,
                "description": description,
                "existing_points": existing_points,
                "priority": priority_info.get("priority_score", 5)
            })

        context = {
            "work_items": item_context,
            "team_capacity": self.team_capacity
        }

        initial_prompt = f"""You are a senior engineer reviewing {len(work_items)} Tasks and Bugs for sprint planning.

For each work item, validate existing estimates or provide new ones using Fibonacci scale (1, 2, 3, 5, 8, 13).

Consider:
- Technical complexity
- Testing requirements
- Risk and uncertainty
- Dependencies

Total team capacity: {self.team_capacity} points"""

        # Multi-turn discussion
        result = self.interactive_session.discuss(
            initial_prompt=initial_prompt,
            context=context,
            max_iterations=5
        )

        # Extract structured JSON
        schema = {
            "type": "object",
            "properties": {
                "estimates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "work_item_id": {"type": "integer"},
                            "work_item_type": {"type": "string"},
                            "title": {"type": "string"},
                            "story_points": {"type": "integer", "minimum": 1, "maximum": 13},
                            "notes": {"type": "string"}
                        },
                        "required": ["work_item_id", "story_points"]
                    }
                }
            },
            "required": ["estimates"]
        }

        # CRITICAL: Include the final AI response as context for JSON extraction.
        # The ask() method creates a fresh API call with no conversation history,
        # so we must explicitly provide the analysis to extract from.
        final_ai_analysis = result.get("final_response", "")

        extraction_prompt = f"""Extract story point estimates from the following estimation analysis.

## AI ESTIMATION ANALYSIS TO EXTRACT FROM:
{final_ai_analysis}

## JSON SCHEMA:
{json.dumps(schema, indent=2)}

Provide ONLY the JSON matching the schema above, no other text. Extract values directly from the analysis provided."""

        try:
            json_response = self.interactive_session.ask(extraction_prompt)
            estimates_data = self.interactive_session.extract_json_from_response(json_response, schema)

            return estimates_data["estimates"]

        except Exception as e:
            from cli.console import print_warning
            print_warning(f"Failed to extract JSON: {e}")
            return self._estimate_ai(work_items, priorities)

    def _step_4_assemble(self) -> Dict[str, Any]:
        """Step 4: Scrum master assembly of sprint plan."""
        work_items = self.step_evidence.get("1-extract-work-items", {}).get("work_items", [])
        priorities = self.step_evidence.get("2-prioritize", {}).get("priorities", [])
        estimates = self.step_evidence.get("3-estimate", {}).get("estimates", [])
        fits_capacity = self.step_evidence.get("3-estimate", {}).get("fits_capacity", True)

        print_step_header(4, "Scrum Master Assembly", "Assembling final sprint plan")

        if self.interactive:
            sprint_plan = self._assemble_interactive(work_items, priorities, estimates)
        elif self.use_ai:
            sprint_plan = self._assemble_ai(work_items, priorities, estimates)
        else:
            sprint_plan = self._assemble_simple(work_items, priorities, estimates)

        total_points = sprint_plan.get('total_points', sum(
            e.get("story_points", 0) for e in estimates
        ))

        # Display sprint plan summary using unified console
        summary_content = {
            "Sprint": self.sprint_name,
            "Total Points": f"{total_points}",
            "Team Capacity": f"{self.team_capacity}",
            "Utilization": f"{total_points / self.team_capacity * 100:.1f}%",
            "Work Items": f"{len(sprint_plan.get('selected_items', []))}",
            "Deferred Items": f"{len(sprint_plan.get('deferred_items', []))}"
        }

        print_summary_panel("Sprint Plan Summary", summary_content, style="success")

        from core.console_workflow import print_bullet_list
        from cli.console import console
        console.print("[bold]Sprint Goals:[/bold]")
        print_bullet_list(sprint_plan.get("sprint_goals", []))

        # Display selected work items in table
        if sprint_plan.get('selected_items'):
            selected_work_items = []
            for item in sprint_plan.get('selected_items', []):
                item_type_str = item.get('work_item_type', 'Task')
                item_type = WorkItemType.TASK if item_type_str == "Task" else WorkItemType.BUG

                selected_work_items.append(WorkItem(
                    id=item['work_item_id'],
                    title=item.get('title', 'Untitled'),
                    type=item_type,
                    state="New",
                    story_points=item.get('story_points')
                ))

            print_work_items_table(selected_work_items, title="Selected Work Items")

        if sprint_plan.get("deferred_items"):
            from cli.console import console
            console.print("[bold]Deferred to Next Sprint:[/bold]")
            for item in sprint_plan.get("deferred_items", []):
                console.print(f"  • [accent2]#{item['work_item_id']}[/accent2]: {item.get('title', 'Untitled')} - [bold]{item['story_points']} pts[/bold]")
            console.print()

        return {
            "sprint_plan": sprint_plan
        }

    def _assemble_simple(
        self,
        work_items: List[Dict[str, Any]],
        priorities: List[Dict[str, Any]],
        estimates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Simple sprint plan assembly (no AI).

        Selects items that fit within capacity based on priority.

        Args:
            work_items: Work items from step 1
            priorities: Priority scores from step 2
            estimates: Estimates from step 3

        Returns:
            Sprint plan with selected items
        """
        # Build maps for quick lookup
        priority_map = {p["work_item_id"]: p["priority_score"] for p in priorities}
        estimate_map = {e["work_item_id"]: e for e in estimates}
        work_item_map = {
            item.get("id"): item.get("fields", {}).get("System.Title", "Untitled")
            for item in work_items
        }

        # Sort estimates by priority (descending)
        sorted_estimates = sorted(
            estimates,
            key=lambda x: priority_map.get(x["work_item_id"], 0),
            reverse=True
        )

        # Select items that fit within capacity
        selected_items = []
        deferred_items = []
        running_total = 0

        for est in sorted_estimates:
            item_points = est.get("story_points", 0)
            if running_total + item_points <= self.team_capacity:
                running_total += item_points
                selected_items.append({
                    **est,
                    "title": work_item_map.get(est["work_item_id"], est.get("title", "Untitled")),
                    "priority_score": priority_map.get(est["work_item_id"], 5)
                })
            else:
                deferred_items.append({
                    **est,
                    "title": work_item_map.get(est["work_item_id"], est.get("title", "Untitled")),
                    "priority_score": priority_map.get(est["work_item_id"], 5)
                })

        # Generate sprint goals
        bug_count = sum(1 for item in selected_items if item.get("work_item_type") == "Bug")
        task_count = sum(1 for item in selected_items if item.get("work_item_type") == "Task")

        sprint_goals = []
        if bug_count > 0:
            sprint_goals.append(f"Fix {bug_count} bug(s) to improve system stability")
        if task_count > 0:
            sprint_goals.append(f"Complete {task_count} task(s) for feature development")
        sprint_goals.append("Maintain quality standards and test coverage")

        return {
            "sprint_name": self.sprint_name,
            "sprint_goals": sprint_goals,
            "selected_items": selected_items,
            "deferred_items": deferred_items,
            "total_points": running_total,
            "team_capacity": self.team_capacity,
            "utilization_percent": (running_total / self.team_capacity * 100) if self.team_capacity > 0 else 0
        }

    def _assemble_ai(
        self,
        work_items: List[Dict[str, Any]],
        priorities: List[Dict[str, Any]],
        estimates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        AI-based sprint plan assembly using scrum-master agent (Mode 2).

        Uses Claude Agent SDK with tool access so the agent can:
        - Read CLAUDE.md for project vision and goals
        - Align sprint goals with project strategy

        Returns:
            Dict with assembled sprint plan
        """
        # Get simple assembly as baseline
        simple_plan = self._assemble_simple(work_items, priorities, estimates)

        # Define JSON schema
        schema = {
            "type": "object",
            "properties": {
                "sprint_goals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 4
                },
                "recommendations": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "risk_summary": {"type": "string"}
            },
            "required": ["sprint_goals", "recommendations"]
        }

        # Build prompt
        item_summary = "\n".join([
            f"- #{item['work_item_id']} ({item['work_item_type']}): {item.get('title', 'Untitled')} - {item['story_points']} pts (Priority: {item.get('priority_score', 'N/A')}/10)"
            for item in simple_plan["selected_items"]
        ])

        prompt = f"""You are a scrum master. Review this sprint plan and provide goals and recommendations.

**IMPORTANT: First read the project context**

Before defining sprint goals, use your Read tool to read `CLAUDE.md` in the project root to understand:
- Project vision and strategic goals
- Current priorities and constraints
- Quality standards

Sprint goals should align with the project's strategic direction.

Sprint: {self.sprint_name}
Team Capacity: {self.team_capacity} points
Total Planned: {simple_plan['total_points']} points
Utilization: {simple_plan['utilization_percent']:.1f}%

Selected Work Items:
{item_summary}

{"Deferred Items: " + str(len(simple_plan['deferred_items'])) if simple_plan['deferred_items'] else "All items fit within capacity."}

Provide:
1. 2-4 specific, measurable sprint goals (aligned with project vision)
2. Recommendations for the team
3. Brief risk summary

Return ONLY valid JSON matching this schema:
{json.dumps(schema, indent=2)}"""

        # Use Agent SDK with tool access
        try:
            from scripts.workflow_executor.agent_sdk import AgentSDKWrapper
            import asyncio

            print("   Using Agent SDK with tool access...")

            wrapper = AgentSDKWrapper(
                workflow_name="sprint-planning-assemble",
                tool_preset="read_only",
                max_turns=10,
                model="claude-sonnet-4-5",
            )

            async def _run_assemble():
                return await wrapper.query(
                    prompt=prompt,
                    agent_type="scrum-master",
                )

            # Execute async
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, _run_assemble())
                        result = future.result()
                else:
                    result = loop.run_until_complete(_run_assemble())
            except RuntimeError:
                result = asyncio.run(_run_assemble())

            if not result.success:
                from cli.console import print_warning
                print_warning(f"Agent SDK query failed: {result.error}")
                return simple_plan

            # Extract JSON from response
            response_text = result.response

            json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

            parsed_result = json.loads(response_text)
            validate(parsed_result, schema)

            # Track token usage
            self.token_usage["assemble"] = {
                "input_tokens": result.token_usage.input_tokens,
                "output_tokens": result.token_usage.output_tokens,
                "cost_usd": result.cost_usd
            }

            # Merge AI results with simple plan
            simple_plan["sprint_goals"] = parsed_result["sprint_goals"]
            simple_plan["recommendations"] = parsed_result.get("recommendations", [])
            simple_plan["risk_summary"] = parsed_result.get("risk_summary", "")

            return simple_plan

        except ImportError as e:
            # ALWAYS report import errors - no silent fallbacks
            from cli.console import print_error, print_info
            print_error(f"AI Sprint Assembly Failed: Import error: {e}")
            print_error("The Claude Agent SDK is not available. AI assembly requires tool access.")
            print_info("Install with: pip install claude-code-sdk")
            print_info("Workflow will continue with basic sprint plan (no AI-generated goals/recommendations).")
            return simple_plan

        except Exception as e:
            # ALWAYS report errors - no silent fallbacks
            from cli.console import print_error, print_info
            print_error(f"AI Sprint Assembly Failed: {type(e).__name__}: {e}")
            print_info("Workflow will continue with basic sprint plan (no AI-generated goals/recommendations).")
            return simple_plan

    def _assemble_anthropic_fallback(
        self,
        simple_plan: Dict[str, Any],
        schema: Dict[str, Any],
        prompt: str
    ) -> Dict[str, Any]:
        """Fallback to Anthropic API when Agent SDK is unavailable."""
        if not self.claude_client:
            return simple_plan

        try:
            response = self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text

            json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

            result = json.loads(response_text)
            validate(result, schema)

            self.token_usage["assemble"] = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cost_usd": self._calculate_cost(response.usage)
            }

            simple_plan["sprint_goals"] = result["sprint_goals"]
            simple_plan["recommendations"] = result.get("recommendations", [])
            simple_plan["risk_summary"] = result.get("risk_summary", "")

            return simple_plan

        except Exception as e:
            from cli.console import print_warning
            print_warning(f"Anthropic fallback failed: {e}, using simple assembly")
            return simple_plan

    def _assemble_interactive(
        self,
        work_items: List[Dict[str, Any]],
        priorities: List[Dict[str, Any]],
        estimates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Interactive sprint plan assembly with final review (Mode 3).

        Args:
            work_items: Work items from step 1
            priorities: Priority scores from step 2
            estimates: Estimates from step 3

        Returns:
            Final sprint plan with user input
        """
        if not self.interactive_session or not self.interactive_session.is_available():
            from cli.console import print_warning
            print_warning("Interactive mode unavailable - falling back to AI assembly")
            return self._assemble_ai(work_items, priorities, estimates)

        # Get baseline plan
        simple_plan = self._assemble_simple(work_items, priorities, estimates)

        context = {
            "sprint_name": self.sprint_name,
            "team_capacity": self.team_capacity,
            "total_points": simple_plan["total_points"],
            "selected_count": len(simple_plan["selected_items"]),
            "deferred_count": len(simple_plan["deferred_items"])
        }

        initial_prompt = f"""You are a scrum master reviewing the sprint plan for {self.sprint_name}.

Selected items: {len(simple_plan['selected_items'])} ({simple_plan['total_points']} points)
Deferred items: {len(simple_plan['deferred_items'])}
Utilization: {simple_plan['utilization_percent']:.1f}%

Help define sprint goals and identify any risks or concerns."""

        # Multi-turn discussion
        result = self.interactive_session.discuss(
            initial_prompt=initial_prompt,
            context=context,
            max_iterations=3
        )

        # Return plan with any modifications from discussion
        return simple_plan

    def _step_5_approval(self) -> Dict[str, Any]:
        """Step 5: Approval gate (BLOCKING)."""
        sprint_plan = self.step_evidence.get("4-assemble", {}).get("sprint_plan", {})

        print_step_header(5, "Approval Gate", "Review and approve sprint plan")

        # Build approval gate data
        summary_lines = [
            f"Sprint: {sprint_plan.get('sprint_name', self.sprint_name)}",
            f"Total Points: {sprint_plan.get('total_points', 0)}",
            f"Team Capacity: {self.team_capacity}",
            f"Work Items: {len(sprint_plan.get('selected_items', []))}",
            "",
            "Goals:"
        ]

        for goal in sprint_plan.get("sprint_goals", []):
            summary_lines.append(f"  • {goal}")

        if sprint_plan.get("recommendations"):
            summary_lines.append("")
            summary_lines.append("Recommendations:")
            for rec in sprint_plan.get("recommendations", []):
                summary_lines.append(f"  • {rec}")

        approval_gate = ApprovalGateData(
            title="Approval Gate - Blocking Checkpoint",
            summary=summary_lines,
            options=[
                ("yes", "Approve sprint plan and assign work items to sprint"),
                ("no", "Cancel sprint planning (no changes made)")
            ],
            question="Approve sprint plan? (yes/no)"
        )

        print_approval_gate(approval_gate)

        # BLOCKING CALL - Execution halts here
        try:
            response = input("Approve sprint plan? (yes/no): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            from cli.console import print_error, console
            console.print("\n")
            print_error("Approval cancelled by user")
            response = "no"

        approved = response == "yes"

        if approved:
            from cli.console import print_success, console
            console.print()
            print_success("✅ Sprint plan APPROVED")
        else:
            from cli.console import print_error, console
            console.print()
            print_error("Sprint plan REJECTED")
            raise ValueError("Sprint plan rejected by user - workflow cancelled")

        return {
            "approved": approved,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }

    def _step_6_create_iteration(self) -> Dict[str, Any]:
        """Step 6: Create or verify sprint iteration exists."""
        print_step_header(6, "Verify/Create Sprint Iteration", f"Checking sprint iteration: {self.sprint_name}")

        if not self.adapter:
            from cli.console import print_warning
            print_warning("No adapter - cannot check/create iteration")
            return {"created": False, "exists": False, "reason": "No adapter"}

        try:
            # Check if iteration already exists
            sprints = self.adapter.list_sprints()
            existing_sprint = None
            for s in sprints:
                if s.get("name") == self.sprint_name:
                    existing_sprint = s
                    break

            if existing_sprint:
                # Sprint already exists - this is the expected case for active sprints
                from cli.console import print_success
                print_success(f"✓ Sprint iteration exists: {self.sprint_name}")

                # Get current items in this sprint for context
                current_points = 0
                remaining_points = 0
                try:
                    sprint_items = self.adapter.query_sprint_work_items(self.sprint_name)
                    current_tasks = [i for i in sprint_items if i.get("fields", {}).get("System.WorkItemType") == "Task"]
                    current_bugs = [i for i in sprint_items if i.get("fields", {}).get("System.WorkItemType") == "Bug"]

                    # Separate completed vs remaining items
                    completed_states = {"Done", "Closed", "Resolved", "Removed"}
                    completed_items = []
                    remaining_items = []

                    for item in sprint_items:
                        state = item.get("fields", {}).get("System.State", "")
                        if state in completed_states:
                            completed_items.append(item)
                        else:
                            remaining_items.append(item)

                    # Calculate story points separately
                    completed_points = sum(
                        i.get("fields", {}).get("Microsoft.VSTS.Scheduling.StoryPoints", 0) or 0
                        for i in completed_items
                    )
                    remaining_points = sum(
                        i.get("fields", {}).get("Microsoft.VSTS.Scheduling.StoryPoints", 0) or 0
                        for i in remaining_items
                    )
                    current_points = completed_points + remaining_points

                    from cli.console import console
                    console.print(f"[primary]  Current sprint contents:[/primary]")
                    console.print(f"    - Tasks: [accent1]{len(current_tasks)}[/accent1]")
                    console.print(f"    - Bugs: [accent1]{len(current_bugs)}[/accent1]")
                    console.print(f"    - Total story points: [bold]{current_points}[/bold]")
                    console.print(f"      └─ Completed: [success]{completed_points} pts[/success] ([dim]{len(completed_items)} items[/dim])")
                    console.print(f"      └─ Remaining: [warning]{remaining_points} pts[/warning] ([dim]{len(remaining_items)} items[/dim])")

                    # Get sprint plan to see what we're adding
                    sprint_plan = self.step_evidence.get("4-assemble", {}).get("sprint_plan", {})
                    adding_points = sprint_plan.get("total_points", 0)

                    # Capacity is measured against REMAINING work (not completed)
                    new_remaining = remaining_points + adding_points

                    from cli.console import console, print_warning, print_success
                    console.print(f"\n[primary]  After adding selected items:[/primary]")
                    console.print(f"    - Adding: [accent1]{adding_points}[/accent1] story points")
                    console.print(f"    - New remaining work: [bold]{new_remaining}[/bold] story points")

                    if new_remaining > self.team_capacity:
                        print_warning(f"    Warning: Remaining work ({new_remaining} pts) exceeds capacity ({self.team_capacity} pts) by {new_remaining - self.team_capacity} pts")
                    else:
                        available = self.team_capacity - new_remaining
                        print_success(f"    ✓ Within capacity ({self.team_capacity} pts) - {available} pts available")

                except Exception as e:
                    from cli.console import print_warning
                    print_warning(f"  Could not query current sprint items: {e}")

                return {
                    "created": False,
                    "exists": True,
                    "sprint_name": self.sprint_name,
                    "current_points": current_points,
                    "remaining_points": remaining_points,
                    "reason": "Already exists - will add items to existing sprint"
                }
            else:
                # Sprint doesn't exist - create it
                from cli.console import print_info, console
                print_info(f"Sprint iteration '{self.sprint_name}' does not exist")
                console.print(f"  → Creating sprint iteration...")

                try:
                    # Calculate default dates (2-week sprint starting today)
                    from datetime import datetime, timedelta
                    start_date = datetime.now().strftime("%Y-%m-%d")
                    finish_date = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")

                    # Create the sprint iteration
                    result = self.adapter.create_sprint(
                        name=self.sprint_name,
                        start_date=start_date,
                        end_date=finish_date
                    )

                    if result:
                        from cli.console import print_success, console
                        print_success(f"  ✓ Created sprint iteration: {self.sprint_name}")
                        console.print(f"    Start: [accent1]{start_date}[/accent1]")
                        console.print(f"    Finish: [accent1]{finish_date}[/accent1]")

                        # Activate the sprint by adding it to the team's backlog
                        iteration_id = result.get("identifier")
                        if iteration_id:
                            try:
                                activate_result = self.adapter.add_iteration_to_team(iteration_id)
                                from cli.console import print_success, print_warning, console
                                if activate_result.get("status") == "already_added":
                                    print_success(f"  ✓ Sprint already active in team backlog")
                                else:
                                    print_success(f"  ✓ Activated sprint in team backlog")
                            except Exception as activate_error:
                                from cli.console import print_warning
                                print_warning(f"  Could not activate sprint: {activate_error}")
                                print_warning(f"      You may need to manually add the sprint to your team's backlog")

                        return {
                            "created": True,
                            "exists": True,
                            "sprint_name": self.sprint_name,
                            "start_date": start_date,
                            "finish_date": finish_date,
                            "activated": bool(iteration_id),
                            "reason": "Created new sprint iteration"
                        }
                    else:
                        from cli.console import print_warning
                        print_warning(f"  Failed to create sprint iteration")
                        return {
                            "created": False,
                            "exists": False,
                            "sprint_name": self.sprint_name,
                            "reason": "Failed to create sprint iteration"
                        }
                except Exception as create_error:
                    from cli.console import print_error
                    print_error(f"  Error creating sprint: {create_error}")
                    return {
                        "created": False,
                        "exists": False,
                        "sprint_name": self.sprint_name,
                        "reason": f"Error creating sprint: {create_error}"
                    }
        except Exception as e:
            from cli.console import print_warning, console
            print_warning(f"Error checking iteration: {e}")
            console.print(f"  → Continuing anyway - items will be assigned to: {self.sprint_name}")
            return {"created": False, "exists": False, "reason": str(e), "sprint_name": self.sprint_name}

    def _step_7_assign_work_items(self) -> Dict[str, Any]:
        """Step 7: Assign approved work items to sprint iteration."""
        sprint_plan = self.step_evidence.get("4-assemble", {}).get("sprint_plan", {})
        selected_items = sprint_plan.get("selected_items", [])

        print_step_header(7, "Assign Work Items to Sprint", f"Assigning {len(selected_items)} work items to {self.sprint_name}")

        if not self.adapter:
            from cli.console import print_warning
            print_warning("No adapter - cannot assign work items")
            return {"assigned_count": 0, "reason": "No adapter"}

        # Check if sprint iteration exists (from step 6)
        iteration_step = self.step_evidence.get("6-create-iteration", {})
        if not iteration_step.get("exists", False):
            from cli.console import print_error
            print_error(f"Cannot assign items - sprint iteration '{self.sprint_name}' does not exist")
            print_error(f"   Reason: {iteration_step.get('reason', 'Unknown')}")
            return {
                "assigned_count": 0,
                "failed_count": len(selected_items),
                "reason": f"Sprint iteration does not exist: {iteration_step.get('reason', 'Unknown')}"
            }

        assigned_items = []
        failed_items = []

        for item in selected_items:
            work_item_id = item.get("work_item_id")
            item_title = item.get("title", "Untitled")

            try:
                # Get the full iteration path using adapter helper
                iteration_path = self.adapter.get_iteration_path(self.sprint_name)

                # Update work item to assign to sprint iteration
                result = self.adapter.update_work_item(
                    work_item_id=work_item_id,
                    fields={
                        "System.IterationPath": iteration_path
                    }
                )

                # Also update story points if they were estimated
                story_points = item.get("story_points")
                if story_points:
                    self.adapter.update_work_item(
                        work_item_id=work_item_id,
                        fields={
                            "Microsoft.VSTS.Scheduling.StoryPoints": story_points
                        }
                    )

                from cli.console import print_success
                print_success(f"  ✓ Assigned #{work_item_id}: {item_title}")

                # EXTERNAL VERIFICATION: Re-query to confirm assignment
                verified = self.adapter.get_work_item(work_item_id)
                if verified:
                    verified_iteration = verified.get("fields", {}).get("System.IterationPath", "")
                    if self.sprint_name in verified_iteration:
                        from cli.console import print_success
                        print_success(f"    ✓ Verified: #{work_item_id} assigned to {self.sprint_name}")
                        assigned_items.append({
                            "work_item_id": work_item_id,
                            "title": item_title,
                            "verified": True
                        })
                    else:
                        from cli.console import print_warning
                        print_warning(f"    Verification failed: Iteration is '{verified_iteration}'")
                        assigned_items.append({
                            "work_item_id": work_item_id,
                            "title": item_title,
                            "verified": False
                        })
                else:
                    from cli.console import print_warning
                    print_warning(f"    Could not verify #{work_item_id}")
                    assigned_items.append({
                        "work_item_id": work_item_id,
                        "title": item_title,
                        "verified": False
                    })

            except Exception as e:
                from cli.console import print_error
                print_error(f"  Failed to assign #{work_item_id}: {e}")
                failed_items.append({
                    "work_item_id": work_item_id,
                    "title": item_title,
                    "error": str(e)
                })

        verified_count = sum(1 for item in assigned_items if item.get("verified"))
        from cli.console import print_success, print_error, console
        console.print()
        print_success(f"✓ Assigned {len(assigned_items)} work items ({verified_count} verified)")

        if failed_items:
            print_error(f"Failed to assign {len(failed_items)} work items")

        return {
            "assigned_count": len(assigned_items),
            "verified_count": verified_count,
            "assigned_items": assigned_items,
            "failed_items": failed_items
        }

    def _step_8_validate_quality(self) -> Dict[str, Any]:
        """Step 8: Validate quality standards for assigned work items."""
        assigned_items = self.step_evidence.get("7-assign-work-items", {}).get("assigned_items", [])

        print_step_header(8, "Validate Quality Standards", f"Validating quality standards for {len(assigned_items)} work items")

        if not self.adapter:
            from cli.console import print_warning
            print_warning("No adapter - cannot validate quality")
            return {"validated": False, "reason": "No adapter"}

        validation_results = []
        validation_failures = []

        for item in assigned_items:
            work_item_id = item.get("work_item_id")

            try:
                # Re-query work item for validation
                work_item = self.adapter.get_work_item(work_item_id)
                if not work_item:
                    validation_failures.append({
                        "work_item_id": work_item_id,
                        "error": "Work item not found"
                    })
                    continue

                fields = work_item.get("fields", {})
                title = fields.get("System.Title", "Untitled")
                description = fields.get("System.Description", "")
                iteration = fields.get("System.IterationPath", "")

                issues = []

                # Check 1: Work item is assigned to correct sprint
                if self.sprint_name not in iteration:
                    issues.append(f"Not assigned to {self.sprint_name}")

                # Check 2: Has description (basic quality check)
                if len(description) < 50:
                    issues.append(f"Description too short ({len(description)} chars)")

                # Check 3: Has title
                if len(title) < 10:
                    issues.append("Title too short")

                if issues:
                    validation_failures.append({
                        "work_item_id": work_item_id,
                        "title": title,
                        "issues": issues
                    })
                    from cli.console import print_warning
                    print_warning(f"  #{work_item_id}: {', '.join(issues)}")
                else:
                    validation_results.append({
                        "work_item_id": work_item_id,
                        "title": title,
                        "passed": True
                    })
                    from cli.console import print_success
                    print_success(f"  ✓ #{work_item_id}: Quality check passed")

            except Exception as e:
                validation_failures.append({
                    "work_item_id": work_item_id,
                    "error": str(e)
                })

        passed_count = len(validation_results)
        failed_count = len(validation_failures)

        from cli.console import print_warning, print_success, console
        console.print()
        if validation_failures:
            print_warning(f"Quality validation: {passed_count} passed, {failed_count} with issues")
            # Note: We don't fail the workflow for quality issues, just report them
        else:
            print_success(f"✅ All {passed_count} work items passed quality checks")

        return {
            "validated": True,
            "passed_count": passed_count,
            "failed_count": failed_count,
            "validation_results": validation_results,
            "validation_failures": validation_failures
        }

    def _calculate_cost(self, usage) -> float:
        """
        Calculate estimated cost in USD for Claude API usage.

        Pricing (as of 2024):
        - Input: $3 per million tokens
        - Output: $15 per million tokens
        """
        input_cost = (usage.input_tokens / 1_000_000) * 3.0
        output_cost = (usage.output_tokens / 1_000_000) * 15.0
        return input_cost + output_cost


def main():
    """Main entry point."""
    # Use standardized argument parser
    parser = WorkflowArgumentParser(
        description="Sprint Planning with External Enforcement - 8-step workflow for groomed Tasks and Bugs",
        standard_args=[
            StandardArguments.SPRINT_NUMBER,
            StandardArguments.CAPACITY,
            StandardArguments.NO_AI,
            StandardArguments.NO_INTERACTIVE,
            StandardArguments.CONFIG
        ]
    )

    # Add sprint planning-specific arguments
    parser.add_custom_argument(
        "--include-item-ids",
        nargs='+',
        type=int,
        help="Include specific work item IDs in sprint planning (includes even if not groomed, with warning)"
    )
    parser.add_custom_argument(
        "--exclude-item-ids",
        nargs='+',
        type=int,
        help="Exclude specific work item IDs from sprint planning"
    )
    parser.add_custom_argument(
        "--include-ungroomed",
        action="store_true",
        help="Include ungroomed work items (not recommended - run /backlog-grooming first)"
    )

    # Parse and validate arguments
    args = parser.parse()

    # Determine AI and interactive modes
    use_ai = not args.no_ai  # Default to AI mode unless --no-ai is specified
    interactive = not args.no_interactive  # Default to interactive unless --no-interactive is specified

    # Create workflow and execute
    workflow = SprintPlanningWorkflow(
        sprint_number=args.sprint_number,
        team_capacity=args.capacity,
        use_ai=use_ai,
        interactive=interactive,
        args=args
    )

    from cli.console import console

    try:
        success = workflow.execute()

        # Print token usage summary if AI was used
        if workflow.token_usage:
            from core.console_workflow import print_section_divider
            console.print()
            print_section_divider("TOKEN USAGE SUMMARY")
            total_cost = 0
            for step, usage in workflow.token_usage.items():
                console.print(f"  [bold]{step}:[/bold]")
                console.print(f"    Input tokens: [accent1]{usage.get('input_tokens', 0):,}[/accent1]")
                console.print(f"    Output tokens: [accent1]{usage.get('output_tokens', 0):,}[/accent1]")
                console.print(f"    Cost: [accent2]${usage.get('cost_usd', 0):.4f}[/accent2]")
                total_cost += usage.get('cost_usd', 0)
            console.print(f"\n  [bold]Total estimated cost:[/bold] [accent2]${total_cost:.4f}[/accent2]")

        if success:
            console.print()
            console.print("─" * 80)
            console.print("[bold #71E4D1]  Sprint planning complete![/bold #71E4D1]")
            console.print("─" * 80)
            console.print()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        console.print()
        console.print("[#758B9B]Sprint planning cancelled by user.[/#758B9B]")
        sys.exit(130)
    except Exception as e:
        console.print()
        console.print("─" * 80)
        console.print("[bold #FF6B6B]  Oops! Something went wrong[/bold #FF6B6B]")
        console.print("─" * 80)
        console.print()
        console.print(f"[#FF6B6B]{e}[/#FF6B6B]")
        console.print()
        console.print("[#758B9B]You can resume from where you left off by running the same command again.[/#758B9B]")
        sys.exit(1)


if __name__ == "__main__":
    main()
