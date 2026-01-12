#!/usr/bin/env python3
"""
Sprint Execution Workflow with External Enforcement

Implementation-focused workflow that executes sprint work items by spawning
specialized agents (engineer, tester) for actual implementation work.

For monitoring/standup functionality, use daily-standup.py instead.

Loop-Until-Done Behavior:
- By default, execution loops until no incomplete work items remain
- Each cycle re-queries the sprint to find remaining or newly created items
- New bugs created by tester agents will be picked up in subsequent cycles
- Use --max-iteration-cycles to limit the number of cycles (default: 10)

Workflow Per Cycle:
1. Query sprint work items sorted by priority
2. For each work item:
   a. Mark as "In Progress" (or "Committed" for Bugs)
   b. Fetch full context (fields + attachments)
   c. Spawn engineer agent for implementation
   d. Spawn tester agent for validation
   e. Attach reports to work item
   f. Handle issues (create conformant bugs if needed)
   g. Auto-commit if successful (local only)
   h. Mark as complete (only if no child bugs open)

Usage:
    # Execute all sprint work items (loops until done, max 10 cycles)
    python3 scripts/sprint-execution.py --sprint "Sprint 8"

    # With initial confirmation prompt
    python3 scripts/sprint-execution.py --sprint "Sprint 8" --confirm-start

    # Limit to 3 execution cycles
    python3 scripts/sprint-execution.py --sprint "Sprint 8" --max-iteration-cycles 3

    # Unlimited cycles (loop until all work complete)
    python3 scripts/sprint-execution.py --sprint "Sprint 8" --max-iteration-cycles 0

    # Execute specific work items only
    python3 scripts/sprint-execution.py --sprint "Sprint 8" --work-items 1234 1235

    # Dry run (show what would be executed)
    python3 scripts/sprint-execution.py --sprint "Sprint 8" --dry-run
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import select
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from difflib import SequenceMatcher
from dataclasses import dataclass, asdict, field

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import workflow executor base
from scripts.workflow_executor.base import WorkflowOrchestrator, ExecutionMode

# Import Claude Agent SDK wrapper (new!)
from scripts.workflow_executor.agent_sdk import (
    AgentSDKWrapper,
    AgentResult,
    TokenUsage,
    MultiTurnSession,
)
from scripts.workflow_executor.session_manager import SessionManager

# Import consensus framework (Feature #1344)
from scripts.workflow_executor.consensus import (
    ConsensusOrchestrator,
    AgentInterface,
    ConsensusResult,
)
from scripts.workflow_executor.consensus_configs import TASK_IMPLEMENTATION_CONFIG

# Import work tracking adapter
sys.path.insert(0, '.claude/skills')
from work_tracking import get_adapter

# Import workflow utilities
from workflows.utilities import normalize_azure_timestamp


# ============================================================================
# Ping-Pong Flow Dataclasses
# ============================================================================

@dataclass
class TesterIssue:
    """Single issue found by tester during ping-pong flow."""
    issue_id: str           # Unique ID for tracking (e.g., "ISS-001")
    severity: str           # "critical", "major", "minor"
    category: str           # "test_failure", "missing_test", "code_issue", "requirement_gap"
    title: str              # Short description
    description: str        # Detailed description
    file_path: Optional[str] = None       # File where issue was found
    line_number: Optional[int] = None     # Line number if applicable
    reproduction_steps: List[str] = field(default_factory=list)  # Steps to reproduce
    expected_behavior: str = ""
    actual_behavior: str = ""


@dataclass
class TesterIssueReport:
    """Tester's issue report - returned instead of creating bugs immediately."""
    work_item_id: int
    iteration: int                 # Which ping-pong iteration
    issues_found: List[TesterIssue] = field(default_factory=list)
    tests_executed: List[str] = field(default_factory=list)  # Test files/commands run
    tests_passed: int = 0
    tests_failed: int = 0
    overall_assessment: str = "pass"  # "pass", "fail", "partial"
    recommendation: str = ""          # Summary of what needs fixing


@dataclass
class EngineerFix:
    """Single fix applied by engineer in response to tester issue."""
    issue_id: str           # References TesterIssue.issue_id
    fix_description: str    # What was done to fix
    files_modified: List[str] = field(default_factory=list)
    tests_added: List[str] = field(default_factory=list)   # New tests added
    tests_run: List[str] = field(default_factory=list)     # Tests executed to verify
    test_result: str = "pass"        # "pass", "fail", "partial"


@dataclass
class EngineerFixReport:
    """Engineer's response after fixing issues from tester."""
    work_item_id: int
    iteration: int
    fixes_applied: List[EngineerFix] = field(default_factory=list)
    issues_not_fixed: List[str] = field(default_factory=list)    # issue_ids that couldn't be fixed
    not_fixed_reasons: Dict[str, str] = field(default_factory=dict)  # issue_id -> reason
    overall_test_result: str = "pass"  # "pass", "fail", "partial"
    summary: str = ""


@dataclass
class TesterValidationReport:
    """Tester's validation after engineer applies fixes."""
    work_item_id: int
    iteration: int
    issues_resolved: List[str] = field(default_factory=list)     # issue_ids now fixed
    issues_remaining: List[TesterIssue] = field(default_factory=list)  # Still broken
    new_issues_found: List[TesterIssue] = field(default_factory=list)  # Regression or new problems
    overall_status: str = "all_resolved"  # "all_resolved", "some_remaining", "new_issues"
    ready_for_next_iteration: bool = True


@dataclass
class BugReportContent:
    """Content for creating a bug work item from unresolved issues."""
    title: str
    description: str
    severity: str = "major"
    reproduction_steps: str = ""
    expected_behavior: str = ""
    actual_behavior: str = ""
    related_files: List[str] = field(default_factory=list)
    parent_work_item_id: int = 0


def flush_stdin():
    """Flush any buffered stdin input to prevent it from being consumed by subsequent prompts."""
    try:
        # Use select with zero timeout to check for available input (Unix only)
        if hasattr(select, 'select'):
            while select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
    except Exception:
        # Fallback: just continue without flushing
        pass


def normalize_sprint_name(sprint_name: str) -> str:
    """
    Normalize sprint name to standard format.

    Handles various input formats:
    - "9" → "Sprint 9"
    - "sprint 9" → "Sprint 9"
    - "Sprint 9" → "Sprint 9" (unchanged)
    - "SPRINT 9" → "Sprint 9"

    Args:
        sprint_name: Sprint name in any format

    Returns:
        str: Normalized sprint name in "Sprint N" format
    """
    if not sprint_name:
        return sprint_name

    # Remove leading/trailing whitespace
    sprint_name = sprint_name.strip()

    # If it's just a number, prepend "Sprint "
    if sprint_name.isdigit():
        return f"Sprint {sprint_name}"

    # If it starts with "sprint" (case-insensitive), normalize capitalization
    if sprint_name.lower().startswith("sprint"):
        # Split on whitespace
        parts = sprint_name.split()
        if len(parts) >= 2:
            # Return "Sprint" + rest of parts
            return "Sprint " + " ".join(parts[1:])
        else:
            # Just "sprint" with no number
            return sprint_name

    # Return as-is if doesn't match patterns
    return sprint_name


class SprintExecutionWorkflow(WorkflowOrchestrator):
    """
    Sprint Execution workflow with external enforcement.

    Implements work item execution by spawning specialized agents:
    - Engineer agent for implementation tasks and bug fixes
    - Tester agent for test tasks with adversarial critique
    """

    # Completed work item states (for checking if items are done)
    # Includes states from various Azure DevOps process templates
    COMPLETED_STATES = {"Done", "Closed", "Resolved", "Removed"}

    # Active/in-progress work item states (for checking if items are being worked on)
    # - Task: "In Progress"
    # - Feature/Epic: "In Progress"
    # - Bug: "Committed" (Scrum template), "Approved" (intermediate)
    ACTIVE_STATES = {"In Progress", "Committed", "Approved", "Active", "Doing"}

    # Test taxonomy categories - aligned with SDLC stage filtering
    # See tests/CLAUDE.md for complete classification documentation
    TEST_TAXONOMY = {
        # Test Types (What is being tested)
        "functional": "Functional requirement verification",
        "security": "Security vulnerability and compliance testing",
        "performance": "Performance, load, and stress testing",
        "usability": "UX and accessibility testing",

        # Test Levels (Scope/granularity)
        "unit": "Individual component testing (fast, isolated)",
        "integration_whitebox": "Component interaction with internal knowledge",
        "integration_blackbox": "Component interaction via public APIs only",
        "system": "End-to-end blackbox system testing",
        "acceptance": "User acceptance criteria validation",
        "validation": "Production environment validation",

        # Quality Attributes
        "regression": "Prevent bug recurrence",
        "adversarial": "Red team attack vectors",
        "falsifiable": "Tests that can definitively fail"
    }

    # SDLC Stage → Test Markers mapping for pytest filtering
    SDLC_TEST_FILTERS = {
        "feature_testing": "unit or integration_whitebox or acceptance or system",
        "ci_pipeline": "unit or integration_whitebox or integration_blackbox or functional",
        "sprint_review": "acceptance or system",
        "production_validation": "validation",
        "security_audit": "security"
    }

    # Agent timeout: 23 minutes for complex implementations
    AGENT_TIMEOUT = 1380

    # Maximum fix-rerun iterations per agent (optimization #4)
    MAX_FIX_ITERATIONS = 3

    def __init__(
        self,
        sprint_name: str,
        work_item_ids: Optional[List[int]] = None,
        dry_run: bool = False,
        confirm_start: bool = False,
        max_retries: int = 3,
        skip_tests: bool = False,
        use_sdk: bool = True,
        max_fix_iterations: int = 3,
        use_consensus: bool = False,
        args: Optional[argparse.Namespace] = None
    ):
        """
        Initialize sprint execution workflow.

        Args:
            sprint_name: Sprint name (e.g., "Sprint 8")
            work_item_ids: Optional list of specific work item IDs to execute
            dry_run: If True, show what would be executed without making changes
            confirm_start: If True, prompt for confirmation before starting
            max_retries: Maximum retry attempts per work item
            skip_tests: If True, skip adversarial testing
            use_sdk: If True, use Claude Agent SDK (session continuity); else use CLI subprocess
            max_fix_iterations: Maximum ping-pong iterations between engineer and tester
            use_consensus: If True, use multi-agent consensus for implementation (Feature #1344)
            args: Command-line arguments
        """
        # Normalize sprint name (e.g., "9" → "Sprint 9")
        self.sprint_name = normalize_sprint_name(sprint_name)
        self.work_item_ids = work_item_ids
        self.dry_run = dry_run
        self.confirm_start = confirm_start
        self.max_retries = max_retries
        self.skip_tests = skip_tests
        self.max_fix_iterations = max_fix_iterations
        self.use_consensus = use_consensus
        self.args = args

        # Track retry attempts per work item
        self.retry_counts: Dict[int, int] = {}
        # Track issues per work item for divergence detection
        self.issue_history: Dict[int, List[int]] = {}

        # Token usage tracking for cost monitoring
        self.token_usage: Dict[str, Dict[str, int]] = {
            "totals": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_creation_tokens": 0,
                "total_cost_usd": 0.0
            },
            "by_agent": {},
            "by_work_item": {}
        }

        # Initialize work tracking adapter
        try:
            self.adapter = get_adapter()
        except Exception as e:
            print(f"Could not initialize the work tracking adapter: {e}")
            print("Sprint execution requires work tracking to be configured.")
            sys.exit(1)

        # Get current user for work item assignment
        self.current_user = None
        try:
            user_info = self.adapter.get_current_user()
            if user_info:
                self.current_user = user_info.get('display_name') or user_info.get('email')
        except Exception as e:
            print(f"⚠ Could not get current user: {e}")
            print(" Work items will be created without assignment")

        # Determine execution mode - always use AI with JSON validation
        mode = ExecutionMode.AI_JSON_VALIDATION

        # Include timestamp in workflow ID
        workflow_id = f"{sprint_name.replace(' ', '-')}-exec-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        super().__init__(
            workflow_name="sprint-execution",
            workflow_id=workflow_id,
            mode=mode,
            enable_checkpoints=True
        )

        # Initialize Claude Agent SDK wrapper for agent invocations
        # This replaces subprocess calls to `claude --print` with the SDK
        self.sdk_wrapper = AgentSDKWrapper(
            workflow_name="sprint-execution",
            tool_preset="implementation",  # Full access: Read, Edit, Write, Bash, Grep, Glob
            max_turns=15,  # Allow more turns for complex implementations
            model="claude-sonnet-4-5",  # Use Sonnet for speed, Opus for complex tasks
        )

        # Session manager for crash recovery
        self.session_manager = SessionManager(
            workflow_name="sprint-execution",
            workflow_id=workflow_id
        )

        # Track whether to use SDK (controlled via --use-sdk/--use-cli flags)
        self.use_sdk = use_sdk

    def _define_steps(self) -> List[Dict[str, Any]]:
        """Define workflow steps."""
        return [
            {"id": "1-query-items", "name": "Query Sprint Work Items"},
            {"id": "2-sort-priority", "name": "Sort by Priority"},
            {"id": "3-execute-items", "name": "Execute Work Items"},
            {"id": "4-summary", "name": "Generate Execution Summary"},
        ]

    def _execute_step(
        self,
        step: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single workflow step."""
        step_id = step["id"]

        if step_id == "1-query-items":
            return self._step_1_query_items()
        elif step_id == "2-sort-priority":
            return self._step_2_sort_priority()
        elif step_id == "3-execute-items":
            return self._step_3_execute_items()
        elif step_id == "4-summary":
            return self._step_4_summary()
        else:
            raise ValueError(f"Unknown step: {step_id}")

    # ========================================================================
    # Step 1: Query Sprint Work Items
    # ========================================================================

    def _step_1_query_items(self) -> Dict[str, Any]:
        """Query sprint work items from work tracking."""
        print(f"\n Let me check what work items are in {self.sprint_name}...")

        try:
            items = self.adapter.query_sprint_work_items(self.sprint_name)

            # Filter to specific work item IDs if provided
            if self.work_item_ids:
                before_count = len(items)
                items = [item for item in items if item.get("id") in self.work_item_ids]
                print(f" I filtered to {len(items)} specific work item(s) (from {before_count} total)")
            else:
                print(f" I found {len(items)} work items")

            # Filter out completed items
            active_items = []
            completed_count = 0
            for item in items:
                state = item.get("fields", {}).get("System.State", item.get("state", ""))
                if state in self.COMPLETED_STATES:
                    completed_count += 1
                else:
                    active_items.append(item)

            if completed_count > 0:
                print(f" Skipping {completed_count} already completed items")

            return {
                "items": active_items,
                "total_count": len(items),
                "active_count": len(active_items),
                "completed_count": completed_count
            }

        except Exception as e:
            print(f" Error querying sprint: {e}")
            return {
                "items": [],
                "error": str(e)
            }

    # ========================================================================
    # Step 2: Sort by Priority
    # ========================================================================

    def _step_2_sort_priority(self) -> Dict[str, Any]:
        """Sort work items by priority, with children before parents."""
        items = self.step_evidence.get("1-query-items", {}).get("items", [])

        if not items:
            print("\n⚠ No active work items to execute")
            return {"sorted_items": [], "order": []}

        print(f"\n Now sorting {len(items)} work items by priority...")

        # Build parent-child relationships
        # Relation type "System.LinkTypes.Hierarchy-Reverse" means this item is a child of target
        # Relation type "System.LinkTypes.Hierarchy-Forward" means this item is a parent of target
        parent_map = {} # child_id -> parent_id
        child_map = {} # parent_id -> [child_ids]
        item_ids = {item.get("id") for item in items}

        for item in items:
            item_id = item.get("id")
            relations = item.get("relations", [])

            for relation in relations:
                rel_type = relation.get("rel", "")
                url = relation.get("url", "")

                # Extract target work item ID from URL
                # URL format: https://.../workitems/{id}
                target_id = None
                if "/workitems/" in url:
                    try:
                        target_id = int(url.split("/workitems/")[-1].split("?")[0])
                    except (ValueError, IndexError):
                        pass

                if target_id:
                    if rel_type == "System.LinkTypes.Hierarchy-Reverse":
                        # This item is a child of target_id
                        parent_map[item_id] = target_id
                    elif rel_type == "System.LinkTypes.Hierarchy-Forward":
                        # This item is a parent of target_id
                        if item_id not in child_map:
                            child_map[item_id] = []
                        child_map[item_id].append(target_id)

        # Calculate depth for each item (children have higher depth, parents have lower)
        # Items with no parent in the sprint have depth 0
        # Child items have depth = parent_depth + 1
        depth_map = {}

        def get_depth(item_id: int, visited: set = None) -> int:
            if visited is None:
                visited = set()
            if item_id in visited:
                return 0 # Cycle detected, stop
            visited.add(item_id)

            if item_id in depth_map:
                return depth_map[item_id]

            parent_id = parent_map.get(item_id)
            if parent_id and parent_id in item_ids:
                # Parent is in the sprint, calculate depth relative to parent
                depth_map[item_id] = get_depth(parent_id, visited) + 1
            else:
                # No parent in sprint (or no parent), this is a root item
                depth_map[item_id] = 0

            return depth_map[item_id]

        for item in items:
            get_depth(item.get("id"))

        # Sort by: depth (descending - children first), priority, type
        # Higher depth = deeper child = process first
        def get_priority(item):
            fields = item.get("fields", {})
            item_id = item.get("id")
            priority = fields.get("Microsoft.VSTS.Common.Priority", 3)
            item_type = fields.get("System.WorkItemType", "Task")
            type_order = {"Bug": 0, "Task": 1, "User Story": 2, "Feature": 3, "Epic": 4}
            depth = depth_map.get(item_id, 0)
            # Negative depth so higher depth sorts first
            return (-depth, priority, type_order.get(item_type, 99))

        sorted_items = sorted(items, key=get_priority)

        # Log hierarchy info
        has_children = any(d > 0 for d in depth_map.values())
        if has_children:
            print(f" ℹ Found parent-child relationships - children will be processed first")

        print(f" ✓ Execution order:")
        for i, item in enumerate(sorted_items[:10], 1):
            fields = item.get("fields", {})
            item_id = item.get("id")
            title = fields.get("System.Title", "Untitled")[:50]
            item_type = fields.get("System.WorkItemType", "Task")
            priority = fields.get("Microsoft.VSTS.Common.Priority", 3)
            depth = depth_map.get(item_id, 0)
            depth_indicator = " └─" * depth if depth > 0 else ""
            print(f" {i}. {depth_indicator}[{item_type}] #{item_id}: {title} (P{priority})")

        if len(sorted_items) > 10:
            print(f" ... and {len(sorted_items) - 10} more")

        return {
            "sorted_items": sorted_items,
            "order": [item.get("id") for item in sorted_items],
            "parent_child_map": parent_map,
            "depth_map": depth_map
        }

    # ========================================================================
    # Step 3: Execute Work Items
    # ========================================================================

    def _step_3_execute_items(self) -> Dict[str, Any]:
        """Execute each work item by spawning appropriate agents."""
        sorted_items = self.step_evidence.get("2-sort-priority", {}).get("sorted_items", [])

        if not sorted_items:
            return {"executed": [], "skipped": [], "failed": []}

        print(f"\n Executing {len(sorted_items)} work items in priority order...")

        if self.dry_run:
            print(" [DRY RUN - No changes will be made]")

        # Optional: One-time confirmation at start
        if self.confirm_start and not self.dry_run:
            flush_stdin()
            print("\n Now ready to begin execution. This will:")
            print(" - Spawn engineer agents for implementation")
            print(" - Spawn tester agents for validation")
            print(" - Auto-commit successful work (local only)")
            print(" - Create bugs for any issues found")
            response = input("\n Proceed? (y/n): ").strip().lower()
            if response != 'y':
                print(" Execution cancelled by user")
                return {"executed": [], "skipped": [], "failed": []}

        executed = []
        skipped = []
        failed = []

        for i, item in enumerate(sorted_items, 1):
            item_id = item.get("id")
            fields = item.get("fields", {})
            title = fields.get("System.Title", "Untitled")
            item_type = fields.get("System.WorkItemType", "Task")

            print(f"\n{'=' * 70}")
            print(f" [{i}/{len(sorted_items)}] {item_type} #{item_id}: {title}")
            print(f"{'=' * 70}")

            # Execute the work item (no per-item prompts)
            try:
                result = self._execute_single_item(item)

                if result.get("success"):
                    executed.append({
                        "id": item_id,
                        "title": title,
                        "type": item_type,
                        "result": result
                    })
                else:
                    failed.append({
                        "id": item_id,
                        "title": title,
                        "type": item_type,
                        "error": result.get("error", "Unknown error")
                    })

            except Exception as e:
                print(f" Error executing #{item_id}: {e}")
                failed.append({
                    "id": item_id,
                    "title": title,
                    "error": str(e)
                })

        return {
            "executed": executed,
            "skipped": skipped,
            "failed": failed,
            "total": len(sorted_items)
        }

    def _run_async(self, coro):
        """Run async coroutine in sync context."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context - use create_task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            # No event loop - create one
            return asyncio.run(coro)

    def _execute_single_item(self, item: Dict) -> Dict[str, Any]:
        """Execute a single work item with engineer → tester flow."""
        # Route to SDK (ping-pong flow) or CLI based on self.use_sdk flag
        if self.use_sdk:
            # Use the new ping-pong flow with separate sessions
            return self._run_async(self._execute_single_item_pingpong(item))
        return self._execute_single_item_cli(item)

    def _execute_single_item_cli(self, item: Dict) -> Dict[str, Any]:
        """Execute a single work item using CLI subprocess (legacy)."""
        item_id = item.get("id")
        fields = item.get("fields", {})
        item_type = fields.get("System.WorkItemType", "Task")
        title = fields.get("System.Title", "Untitled")

        # Step 1: Mark as in-progress
        # Work item state transitions for this project:
        # - Task: To Do → In Progress → Done
        # - Feature/Epic: New → In Progress → Done
        # - Bug: New → Approved → Committed → Done (special case!)
        is_bug = item_type == "Bug"
        active_state = "Committed" if is_bug else "In Progress"
        print(f"\n Marking #{item_id} as '{active_state}'...")
        if not self.dry_run:
            state_result = self._update_work_item_state(item_id, active_state)
            if not state_result.get("success"):
                return {"success": False, "error": f"Failed to update state: {state_result.get('error')}"}

        # Step 2: Fetch full context with attachments
        print(f" Fetching full context for the work item...")
        context = self._get_work_item_context(item_id)
        if context.get("error"):
            return {"success": False, "error": f"Failed to get context: {context.get('error')}"}

        # Step 3: Execute implementation - either via consensus or traditional agents
        if self.use_consensus:
            # Consensus-based implementation (Feature #1344)
            print(f"\n CONSENSUS Running multi-agent consensus for implementation...")
            result = self._run_implementation_consensus(context, is_bug)

            # Attach consensus report
            if result.get("response") and not self.dry_run:
                self._attach_report(item_id, result["response"], "consensus-implementation")

            if not result.get("success"):
                print(f" Consensus failed: {result.get('error', 'Unknown error')}")
                return result

        else:
            # Traditional agent-based implementation
            # Step 3a: Spawn engineer agent for implementation
            print(f"\n The engineer agent is {'fixing the bug' if is_bug else 'implementing the task'}...")
            engineer_result = self._spawn_engineer_agent(context)

            # Attach engineer report
            if engineer_result.get("response") and not self.dry_run:
                self._attach_report(item_id, engineer_result["response"], "implementation")

            if not engineer_result.get("success"):
                print(f" Engineer agent failed: {engineer_result.get('error', 'Unknown error')}")
                return engineer_result

            # Step 3b: Spawn tester agent for validation
            if not self.skip_tests:
                print(f"\n 🧪 The tester agent is validating the implementation...")
                tester_result = self._spawn_tester_agent(context, verification_mode=is_bug)

                # Attach tester report
                if tester_result.get("response") and not self.dry_run:
                    self._attach_report(item_id, tester_result["response"], "test-results")

                result = self._merge_results(engineer_result, tester_result)
            else:
                result = engineer_result

        # Step 5: Handle issues - create conformant bugs
        if result.get("issues"):
            issues = result.get("issues", [])
            print(f"\n ⚠ Found {len(issues)} issue(s)")

            # Check for divergence
            if self._check_divergence(item_id, issues):
                print(f"\n DIVERGENCE WARNING: Issues not converging for #{item_id}")
                print(f" Retry count: {self.retry_counts.get(item_id, 0)}")
                print(f" Issue history: {self.issue_history.get(item_id, [])}")

            # Create conformant bug work items for issues
            for issue in issues:
                self._create_conformant_bug(item_id, issue)

        # Step 6: Check if we can mark as Done (must have no open child bugs)
        if result.get("success"):
            can_complete, reason = self._can_mark_done(item_id)

            if can_complete:
                # All work item types in this project use "Done" as final state
                final_state = "Done"
                print(f"\n Marking #{item_id} as '{final_state}'...")
                if not self.dry_run:
                    self._update_work_item_state(item_id, final_state)

                # Step 7: Auto-commit if successful
                if not self.dry_run:
                    self._auto_commit_if_eligible(item_id, result, title)
            else:
                print(f"\n ⏸ Cannot mark #{item_id} as Done: {reason}")
                print(f" Keeping in '{active_state}' until child issues are resolved")
        else:
            # Keep in progress for retry
            print(f"\n ⏸ Keeping #{item_id} in '{active_state}' for retry")
            self.retry_counts[item_id] = self.retry_counts.get(item_id, 0) + 1

        return result

    async def _execute_single_item_sdk(self, item: Dict) -> Dict[str, Any]:
        """
        Execute a single work item using Claude Agent SDK with multi-turn sessions.

        Key improvements over CLI version:
        - Session continuity: Tester agent sees engineer's full context
        - Token efficiency: Cached prompts reduce costs by ~90%
        - Better quality: Agent can explore codebase with tools
        - Crash recovery: Sessions saved for resume
        """
        item_id = item.get("id")
        fields = item.get("fields", {})
        item_type = fields.get("System.WorkItemType", "Task")
        title = fields.get("System.Title", "Untitled")

        # Step 1: Mark as in-progress
        is_bug = item_type == "Bug"
        active_state = "Committed" if is_bug else "In Progress"
        print(f"\n Marking #{item_id} as '{active_state}'...")
        if not self.dry_run:
            state_result = self._update_work_item_state(item_id, active_state)
            if not state_result.get("success"):
                return {"success": False, "error": f"Failed to update state: {state_result.get('error')}"}

        # Step 2: Fetch full context with attachments
        print(f" Fetching full context for the work item...")
        context = self._get_work_item_context(item_id)
        if context.get("error"):
            return {"success": False, "error": f"Failed to get context: {context.get('error')}"}

        # Step 3: Build prompts (same as CLI version)
        engineer_prompt = self._build_engineer_prompt(context, is_bug)
        tester_prompt = self._build_tester_prompt(context, is_bug) if not self.skip_tests else None

        # Step 4: Execute with multi-turn session (KEY SDK BENEFIT!)
        # Both agents share the same session context, so tester sees all of
        # engineer's work, file changes, test results, etc.
        print(f"\n The engineer agent is {'fixing the bug' if is_bug else 'implementing the task'}...")

        # Engineer agent (first turn - creates session)
        engineer_result = await self._invoke_agent_sdk(
            agent_type="engineer",
            prompt=engineer_prompt,
            work_item_id=item_id,
            continue_session=False  # Fresh session for this work item
        )

        # Attach engineer report
        if engineer_result.get("response") and not self.dry_run:
            self._attach_report(item_id, engineer_result["response"], "implementation")

        if not engineer_result.get("success"):
            print(f" Engineer agent failed: {engineer_result.get('error', 'Unknown error')}")
            return engineer_result

        # Step 5: Tester agent (continues session - has full context!)
        if not self.skip_tests:
            print(f"\n 🧪 The tester agent is validating the implementation...")
            print(f"    (Session continuity: tester sees engineer's full context)")

            tester_result = await self._invoke_agent_sdk(
                agent_type="tester",
                prompt=tester_prompt,
                work_item_id=item_id,
                continue_session=True  # IMPORTANT: Continue engineer's session!
            )

            # Attach tester report
            if tester_result.get("response") and not self.dry_run:
                self._attach_report(item_id, tester_result["response"], "test-results")

            result = self._merge_results(engineer_result, tester_result)
        else:
            result = engineer_result

        # Step 6: Handle issues - create conformant bugs
        if result.get("issues"):
            issues = result.get("issues", [])
            print(f"\n ⚠ Found {len(issues)} issue(s)")

            # Check for divergence
            if self._check_divergence(item_id, issues):
                print(f"\n DIVERGENCE WARNING: Issues not converging for #{item_id}")
                print(f" Retry count: {self.retry_counts.get(item_id, 0)}")
                print(f" Issue history: {self.issue_history.get(item_id, [])}")

            # Create conformant bug work items for issues
            for issue in issues:
                self._create_conformant_bug(item_id, issue)

        # Step 7: Check if we can mark as Done (must have no open child bugs)
        if result.get("success"):
            can_complete, reason = self._can_mark_done(item_id)

            if can_complete:
                final_state = "Done"
                print(f"\n Marking #{item_id} as '{final_state}'...")
                if not self.dry_run:
                    self._update_work_item_state(item_id, final_state)

                # Step 8: Auto-commit if successful
                if not self.dry_run:
                    self._auto_commit_if_eligible(item_id, result, title)

                # Mark sessions as completed
                self.session_manager.mark_completed("engineer")
                if not self.skip_tests:
                    self.session_manager.mark_completed("tester")
            else:
                print(f"\n ⏸ Cannot mark #{item_id} as Done: {reason}")
                print(f" Keeping in '{active_state}' until child issues are resolved")
        else:
            # Keep in progress for retry
            print(f"\n ⏸ Keeping #{item_id} in '{active_state}' for retry")
            self.retry_counts[item_id] = self.retry_counts.get(item_id, 0) + 1

        return result

    async def _execute_single_item_pingpong(self, item: Dict) -> Dict[str, Any]:
        """
        Execute a single work item using ping-pong engineer-tester flow.

        Key difference from _execute_single_item_sdk:
        - Engineer and tester use SEPARATE sessions (isolated contexts)
        - Issues found by tester are passed back to engineer for immediate fixes
        - Script orchestrates the ping-pong, not the agents themselves
        - Bug work items only created for issues that persist after all iterations

        Flow:
        1. Engineer implements task (fresh session)
        2. Tester tests implementation (fresh session)
        3. If issues found:
           a. Pass issue JSON to engineer (continue engineer's session)
           b. Engineer fixes and returns fix report
           c. Pass fix report to tester (continue tester's session)
           d. Repeat up to max_fix_iterations times
        4. Create bug work items for any remaining issues
        """
        item_id = item.get("id")
        fields = item.get("fields", {})
        item_type = fields.get("System.WorkItemType", "Task")
        title = fields.get("System.Title", "Untitled")

        # Step 1: Mark as in-progress
        is_bug = item_type == "Bug"
        active_state = "Committed" if is_bug else "In Progress"
        print(f"\n Marking #{item_id} as '{active_state}'...")
        if not self.dry_run:
            state_result = self._update_work_item_state(item_id, active_state)
            if not state_result.get("success"):
                return {"success": False, "error": f"Failed to update state: {state_result.get('error')}"}

        # Step 2: Fetch full context with attachments
        print(f" Fetching full context for the work item...")
        context = self._get_work_item_context(item_id)
        if context.get("error"):
            return {"success": False, "error": f"Failed to get context: {context.get('error')}"}

        # Step 3: Check if we should use consensus-based implementation
        if self.use_consensus:
            # Consensus-based implementation (Feature #1344)
            print(f"\n CONSENSUS Running multi-agent consensus for implementation...")
            result = self._run_implementation_consensus(context, is_bug)

            # Attach consensus report
            if result.get("response") and not self.dry_run:
                self._attach_report(item_id, result["response"], "consensus-implementation")

            if not result.get("success"):
                print(f" Consensus failed: {result.get('error', 'Unknown error')}")
                return result

            # Finalize work item after successful consensus
            return self._finalize_work_item(item_id, active_state, title, result)

        # Step 4: Create SEPARATE SDK wrapper instances for isolated sessions
        engineer_wrapper = AgentSDKWrapper(
            workflow_name="sprint-execution-engineer",
            tool_preset="implementation",
            max_turns=50,  # More turns for complex implementations
            model=self.sdk_wrapper.model,
        )
        tester_wrapper = AgentSDKWrapper(
            workflow_name="sprint-execution-tester",
            tool_preset="implementation",
            max_turns=30,
            model=self.sdk_wrapper.model,
        )

        # Step 4: Engineer implements task (fresh session)
        print(f"\n The engineer agent is {'fixing the bug' if is_bug else 'implementing the task'}...")
        engineer_prompt = self._build_engineer_prompt(context, is_bug)

        engineer_result = await engineer_wrapper.query(
            prompt=engineer_prompt,
            agent_type="engineer",
            continue_session=False  # Fresh session
        )

        # Attach engineer report
        if engineer_result.response and not self.dry_run:
            self._attach_report(item_id, engineer_result.response, "implementation")

        if not engineer_result.success:
            print(f" Engineer agent failed: {engineer_result.error}")
            return {"success": False, "error": engineer_result.error or "Engineer failed"}

        # Update token tracking for engineer
        self._update_token_tracking("engineer", item_id, engineer_result)

        # Step 5: Tester tests implementation (fresh session)
        if self.skip_tests:
            print(f"\n ⏭ Skipping tester (--skip-tests)")
            return self._finalize_work_item(
                item_id, active_state, title,
                {"success": True, "response": engineer_result.response, "issues": []}
            )

        print(f"\n The tester agent is validating the implementation...")
        print(f"    (Separate session: tester starts fresh)")
        tester_initial_prompt = self._build_tester_initial_prompt(context, is_bug)

        tester_result = await tester_wrapper.query(
            prompt=tester_initial_prompt,
            agent_type="tester",
            continue_session=False  # Fresh session for tester
        )

        if not tester_result.success:
            print(f" Tester agent failed: {tester_result.error}")
            return {"success": False, "error": tester_result.error or "Tester failed"}

        # Update token tracking for tester
        self._update_token_tracking("tester", item_id, tester_result)

        # Step 6: Parse tester's issue report
        issue_report = self._parse_tester_issue_report(tester_result.response, item_id)

        # Attach initial test report
        if tester_result.response and not self.dry_run:
            self._attach_report(item_id, tester_result.response, "test-results-initial")

        # Step 7: Ping-pong loop
        iteration = 1
        ping_pong_history = []

        while issue_report.issues_found and iteration <= self.max_fix_iterations:
            print(f"\n Ping-pong iteration {iteration}/{self.max_fix_iterations}")
            print(f"    Issues to fix: {len(issue_report.issues_found)}")

            ping_pong_history.append({
                "iteration": iteration,
                "issues_count": len(issue_report.issues_found),
                "issues": [i.title for i in issue_report.issues_found]
            })

            # 7a: Pass issues to engineer (continue engineer's session)
            print(f"\n Engineer fixing {len(issue_report.issues_found)} issue(s)...")
            fix_prompt = self._build_engineer_fix_prompt(issue_report)
            fix_result = await engineer_wrapper.query(
                prompt=fix_prompt,
                agent_type="engineer",
                continue_session=True  # Continue engineer's session
            )

            if not fix_result.success:
                print(f" Engineer fix attempt failed: {fix_result.error}")
                break

            self._update_token_tracking("engineer", item_id, fix_result)
            fix_report = self._parse_engineer_fix_report(fix_result.response, item_id, iteration)

            # 7b: Pass fixes to tester for validation (continue tester's session)
            print(f"\n Tester validating {len(fix_report.fixes_applied)} fix(es)...")
            validation_prompt = self._build_tester_validation_prompt(fix_report)
            validation_result = await tester_wrapper.query(
                prompt=validation_prompt,
                agent_type="tester",
                continue_session=True  # Continue tester's session
            )

            if not validation_result.success:
                print(f" Tester validation failed: {validation_result.error}")
                break

            self._update_token_tracking("tester", item_id, validation_result)
            validation_report = self._parse_tester_validation_report(
                validation_result.response, item_id, iteration
            )

            # 7c: Convert validation to issue report for next iteration
            issue_report = self._validation_to_issue_report(validation_report)

            print(f"    Resolved: {len(validation_report.issues_resolved)}")
            print(f"    Remaining: {len(validation_report.issues_remaining)}")
            print(f"    New issues: {len(validation_report.new_issues_found)}")

            iteration += 1

        # Step 8: Create bugs for unresolved issues
        bugs_created = []
        if issue_report.issues_found:
            print(f"\n ⚠ Creating bug work items for {len(issue_report.issues_found)} unresolved issue(s)")

            # Ask tester to format bugs properly
            bug_prompt = self._build_bug_creation_prompt(issue_report)
            bug_result = await tester_wrapper.query(
                prompt=bug_prompt,
                agent_type="tester",
                continue_session=True
            )

            if bug_result.success:
                bug_contents = self._parse_bug_contents(bug_result.response)
                for bug in bug_contents:
                    created = self._create_conformant_bug(item_id, {
                        "title": bug.title,
                        "description": bug.description,
                        "severity": bug.severity,
                        "repro_steps": bug.reproduction_steps,
                        "expected_behavior": bug.expected_behavior,
                        "actual_behavior": bug.actual_behavior,
                        "affected_files": bug.related_files,
                        "type": "bug",
                        "parent_work_item_id": item_id
                    })
                    if created:
                        bugs_created.append(created)
            else:
                # Fallback: create bugs from issue_report directly
                for issue in issue_report.issues_found:
                    created = self._create_conformant_bug(item_id, {
                        "title": issue.title,
                        "description": issue.description,
                        "severity": issue.severity,
                        "repro_steps": "\n".join(issue.reproduction_steps),
                        "expected_behavior": issue.expected_behavior,
                        "actual_behavior": issue.actual_behavior,
                        "affected_files": [issue.file_path] if issue.file_path else [],
                        "type": issue.category,
                        "parent_work_item_id": item_id
                    })
                    if created:
                        bugs_created.append(created)

        # Attach final test report
        if not self.dry_run:
            final_report = {
                "ping_pong_iterations": iteration - 1,
                "bugs_created": len(bugs_created),
                "all_issues_resolved": len(issue_report.issues_found) == 0,
                "history": ping_pong_history
            }
            self._attach_report(item_id, json.dumps(final_report, indent=2), "ping-pong-summary")

        # Build final result
        result = {
            "success": len(issue_report.issues_found) == 0,
            "response": engineer_result.response,
            "issues": [asdict(i) for i in issue_report.issues_found],
            "ping_pong_iterations": iteration - 1,
            "bugs_created": bugs_created
        }

        # Step 9: Finalize work item state
        return self._finalize_work_item(item_id, active_state, title, result)

    def _update_token_tracking(
        self,
        agent_type: str,
        work_item_id: int,
        result: 'AgentResult'
    ) -> None:
        """Update token usage tracking from an AgentResult."""
        token_stats = {
            "input_tokens": result.token_usage.input_tokens,
            "output_tokens": result.token_usage.output_tokens,
            "cache_read_tokens": result.token_usage.cache_read_tokens,
            "cache_creation_tokens": result.token_usage.cache_creation_tokens,
        }

        self.token_usage["totals"]["input_tokens"] += token_stats["input_tokens"]
        self.token_usage["totals"]["output_tokens"] += token_stats["output_tokens"]
        self.token_usage["totals"]["cache_read_tokens"] += token_stats["cache_read_tokens"]
        self.token_usage["totals"]["cache_creation_tokens"] += token_stats["cache_creation_tokens"]
        self.token_usage["totals"]["total_cost_usd"] += result.cost_usd

        if agent_type not in self.token_usage["by_agent"]:
            self.token_usage["by_agent"][agent_type] = {
                "calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0
            }
        self.token_usage["by_agent"][agent_type]["calls"] += 1
        self.token_usage["by_agent"][agent_type]["input_tokens"] += token_stats["input_tokens"]
        self.token_usage["by_agent"][agent_type]["output_tokens"] += token_stats["output_tokens"]
        self.token_usage["by_agent"][agent_type]["cost_usd"] += result.cost_usd

        wi_key = str(work_item_id)
        if wi_key not in self.token_usage["by_work_item"]:
            self.token_usage["by_work_item"][wi_key] = {
                "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0
            }
        self.token_usage["by_work_item"][wi_key]["input_tokens"] += token_stats["input_tokens"]
        self.token_usage["by_work_item"][wi_key]["output_tokens"] += token_stats["output_tokens"]
        self.token_usage["by_work_item"][wi_key]["cost_usd"] += result.cost_usd

    def _finalize_work_item(
        self,
        item_id: int,
        active_state: str,
        title: str,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Finalize work item state based on execution result."""
        if result.get("success"):
            can_complete, reason = self._can_mark_done(item_id)

            if can_complete:
                final_state = "Done"
                print(f"\n Marking #{item_id} as '{final_state}'...")
                if not self.dry_run:
                    self._update_work_item_state(item_id, final_state)

                # Auto-commit if successful
                if not self.dry_run:
                    self._auto_commit_if_eligible(item_id, result, title)

                # Mark sessions as completed
                self.session_manager.mark_completed("engineer")
                if not self.skip_tests:
                    self.session_manager.mark_completed("tester")
            else:
                print(f"\n ⏸ Cannot mark #{item_id} as Done: {reason}")
                print(f" Keeping in '{active_state}' until child issues are resolved")
        else:
            print(f"\n ⏸ Keeping #{item_id} in '{active_state}' for retry")
            self.retry_counts[item_id] = self.retry_counts.get(item_id, 0) + 1

        return result

    def _build_engineer_prompt(self, context: Dict[str, Any], is_bug: bool) -> str:
        """Build the engineer agent prompt."""
        agent_def = self._load_agent_definition("engineer")
        return f"""{agent_def}

---

## Task: Implement Work Item #{context['id']}

{self._format_context_for_prompt(context)}

## CRITICAL: Execute Immediately

**DO NOT ask for permission or confirmation.** This work item has already been approved for implementation through sprint planning. You are being invoked by an automated sprint execution workflow that has already verified this work is ready.

**Implement immediately. Do not present a plan and ask to proceed.**

## Instructions

1. Implement the code changes required to complete this work item
2. Create appropriate tests for your implementation
3. Run the tests to verify they pass
4. Report any issues or blockers encountered

Work autonomously to completion. Only stop if you encounter a genuine blocker that prevents implementation.

## CRITICAL: Token Optimization Rules

To minimize token consumption and costs, you MUST follow these rules:

### Test Execution (MANDATORY)
- **Always use short tracebacks**: `pytest --tb=short` or `pytest --tb=line`
- **Never use**: `pytest --tb=long` or `pytest -v --tb=full`
- **Run only relevant tests**: `pytest tests/test_specific_module.py -k "test_name"`
- **Avoid full test suite runs** unless absolutely necessary for final verification

### Iteration Limits (MANDATORY)
- **Maximum {self.MAX_FIX_ITERATIONS} fix-and-rerun cycles** - if tests still fail after {self.MAX_FIX_ITERATIONS} attempts, report the issues and stop
- Do NOT loop indefinitely trying to fix failing tests
- After {self.MAX_FIX_ITERATIONS} cycles, include remaining failures in the issues array

### Output Efficiency
- Summarize test results rather than including full pytest output
- Only include relevant code snippets, not entire files
- Be concise in explanations

## Output Format

Provide your work summary, then at the END of your response include a structured JSON block.

**IMPORTANT**: The `issues` array should ONLY contain genuine bugs that need to be fixed.
Do NOT include issues for work you successfully completed.

```json
{{
  "status": "success" | "partial" | "blocked",
  "files_modified": ["path/to/file1.py", "path/to/file2.py"],
  "tests_created": ["path/to/test_file.py"],
  "tests_passed": true | false,
  "issues": [
    {{
      "title": "Short descriptive title for the bug (max 80 chars)",
      "description": "Detailed description of what is wrong and why it's a problem",
      "severity": "high" | "medium" | "low",
      "affected_files": ["path/to/file1.py", "path/to/file2.py"],
      "repro_steps": "1. Do X\\n2. Do Y\\n3. Observe bug",
      "expected_behavior": "What should happen",
      "actual_behavior": "What actually happens",
      "suggested_fix": "Brief suggestion for how to fix",
      "type": "bug" | "security" | "performance" | "incomplete",
      "parent_work_item_id": {context['id']}
    }}
  ]
}}
```

If implementation was successful with no issues, use: `"issues": []`
"""

    def _build_tester_prompt(self, context: Dict[str, Any], verification_mode: bool) -> str:
        """Build the tester agent prompt."""
        mode_desc = "verification" if verification_mode else "adversarial"
        agent_def = self._load_agent_definition("tester")
        taxonomy_section = "\n".join([
            f"- **{cat}**: {desc}" for cat, desc in self.TEST_TAXONOMY.items()
        ])

        return f"""{agent_def}

---

## Task: {mode_desc.title()} Testing for Work Item #{context['id']}

{self._format_context_for_prompt(context)}

## CRITICAL: Execute Immediately

**DO NOT ask for permission or confirmation.** This work item has already been implemented and you are being invoked by an automated sprint execution workflow to perform {mode_desc} testing.

**Execute tests immediately. Do not present a plan and ask to proceed.**

## Test Taxonomy

All tests must be labelled with appropriate taxonomy:
{taxonomy_section}

## Testing Requirements

### Parent Epic/Feature Validation (CRITICAL - per workflow-flow.md)
Your validation must ensure the implementation achieves the PARENT goals, not just Task completion:

1. **Epic Goal Alignment**: Verify implementation achieves the parent Epic's stated goals
2. **Feature Completeness**: Check all parent Feature acceptance criteria are satisfied
3. **Task Implementation**: Validate specific Task requirements

### Standard Checks
4. **Completeness Check**: Verify all acceptance criteria are met
5. **Falsifiability Check**: Ensure tests can definitively fail
6. **Stub Detection**: Check for empty/todo/stub implementations
7. **Adversarial Testing**: Apply red team mindset to find edge cases

## CRITICAL: Token Optimization Rules

To minimize token consumption and costs, you MUST follow these rules:

### Test Execution (MANDATORY)
- **Always use short tracebacks**: `pytest --tb=short` or `pytest --tb=line`
- **Never use**: `pytest --tb=long` or `pytest -v --tb=full`
- **Run only relevant tests**: `pytest tests/test_specific_module.py -k "test_name"`
- **Avoid full test suite runs** - only run tests relevant to this work item

### Iteration Limits (MANDATORY)
- **Maximum {self.MAX_FIX_ITERATIONS} test-fix cycles** - if issues persist after {self.MAX_FIX_ITERATIONS} attempts, report them and stop
- Do NOT loop indefinitely trying to make tests pass
- After {self.MAX_FIX_ITERATIONS} cycles, include remaining failures in the issues array

### Output Efficiency
- Summarize test results (e.g., "5 passed, 2 failed") rather than full output
- Only include relevant failure details, not full stack traces
- Be concise in explanations

## Test Labelling Format

Every test must include:
```python
@pytest.mark.{{taxonomy}} # e.g., unit, integration, adversarial
@pytest.mark.falsifiable
def test_specific_behavior():
    \"\"\"
    Test Taxonomy: {{taxonomy}}, falsifiable
    Purpose: {{clear description}}
    Related Bug: #{{bug_id}} (if regression test)
    \"\"\"
    # Test implementation
```

## Output Format

Provide your testing summary, then at the END of your response include a structured JSON block.

**CRITICAL**: The `issues` array should ONLY contain genuine bugs/problems that need fixing.
Do NOT include:
- Work that was successfully completed
- Tests that passed
- Code that was intentionally fixed during this session
- Documentation of what you tested

```json
{{
  "status": "pass" | "fail" | "partial",
  "tests_created": ["path/to/test_file.py"],
  "tests_run": 10,
  "tests_passed": 10,
  "tests_failed": 0,
  "coverage_met": true | false,
  "issues": [
    {{
      "title": "Short descriptive title for the bug (max 80 chars)",
      "description": "Detailed description of the defect found during testing",
      "severity": "high" | "medium" | "low",
      "affected_files": ["path/to/file1.py"],
      "repro_steps": "1. Run test X\\n2. Observe failure\\n3. Root cause is Y",
      "expected_behavior": "What the code should do per acceptance criteria",
      "actual_behavior": "What the code actually does (the bug)",
      "test_evidence": "Name of failing test or verification that found this",
      "suggested_fix": "Brief suggestion for how to fix",
      "type": "bug" | "security" | "incomplete" | "test_gap" | "stub",
      "parent_work_item_id": {context['id']}
    }}
  ]
}}
```

If all tests pass and implementation is complete, use: `"issues": []`
"""

    def _build_tester_initial_prompt(self, context: Dict[str, Any], is_bug: bool) -> str:
        """
        Build initial tester prompt for ping-pong flow.

        Unlike _build_tester_prompt, this prompt instructs the tester to return
        structured JSON issues instead of prose that gets parsed later.
        """
        mode_desc = "verification" if is_bug else "adversarial"
        agent_def = self._load_agent_definition("tester")
        taxonomy_section = "\n".join([
            f"- **{cat}**: {desc}" for cat, desc in self.TEST_TAXONOMY.items()
        ])

        return f"""{agent_def}

---

## Task: {mode_desc.title()} Testing for Work Item #{context['id']}

{self._format_context_for_prompt(context)}

## CRITICAL: Execute Immediately

**DO NOT ask for permission or confirmation.** This work item has been implemented and you are performing {mode_desc} testing as part of the ping-pong workflow.

**Execute tests immediately. Do not present a plan and ask to proceed.**

## Test Taxonomy

All tests must be labelled with appropriate taxonomy:
{taxonomy_section}

## Testing Requirements

### Parent Epic/Feature Validation (CRITICAL - per workflow-flow.md)
Your validation must ensure the implementation achieves the PARENT goals, not just Task completion:

1. **Epic Goal Alignment**: Verify implementation achieves the parent Epic's stated goals
2. **Feature Completeness**: Check all parent Feature acceptance criteria are satisfied
3. **Task Implementation**: Validate specific Task requirements

### Standard Checks
4. **Completeness Check**: Verify all acceptance criteria are met
5. **Falsifiability Check**: Ensure tests can definitively fail
6. **Stub Detection**: Check for empty/todo/stub implementations
7. **Adversarial Testing**: Apply red team mindset to find edge cases

## CRITICAL: Token Optimization Rules

- **Always use short tracebacks**: `pytest --tb=short` or `pytest --tb=line`
- **Run only relevant tests**: `pytest tests/test_specific_module.py -k "test_name"`
- **Summarize results** rather than including full pytest output

## Output Format

IMPORTANT: Return your findings as a structured JSON block. Do NOT create bug work items directly.
The orchestrating script will handle bug creation for unresolved issues after the ping-pong cycle.

```json
{{
  "work_item_id": {context['id']},
  "iteration": 1,
  "issues_found": [
    {{
      "issue_id": "ISS-001",
      "severity": "critical|major|minor",
      "category": "test_failure|missing_test|code_issue|requirement_gap",
      "title": "Short description of the issue",
      "description": "Detailed description of what is wrong",
      "file_path": "path/to/file.py",
      "line_number": 42,
      "reproduction_steps": ["step 1", "step 2", "step 3"],
      "expected_behavior": "What should happen",
      "actual_behavior": "What actually happened"
    }}
  ],
  "tests_executed": ["tests/test_file.py::test_name"],
  "tests_passed": 5,
  "tests_failed": 2,
  "overall_assessment": "pass|fail|partial",
  "recommendation": "Summary of what needs fixing"
}}
```

If all tests pass and implementation is correct:
```json
{{
  "work_item_id": {context['id']},
  "iteration": 1,
  "issues_found": [],
  "tests_executed": ["tests/test_file.py"],
  "tests_passed": 10,
  "tests_failed": 0,
  "overall_assessment": "pass",
  "recommendation": "Implementation complete, all tests passing"
}}
```
"""

    def _build_engineer_fix_prompt(self, issue_report: TesterIssueReport) -> str:
        """Build prompt for engineer to fix issues found by tester."""
        issues_json = json.dumps([asdict(i) for i in issue_report.issues_found], indent=2)

        return f"""## Fix Request: Iteration {issue_report.iteration}

The tester found the following issues with your implementation:

```json
{issues_json}
```

**Tester's Recommendation:** {issue_report.recommendation}

## Instructions

1. Fix each issue listed above
2. Run tests to verify your fixes work
3. Report what you fixed and any issues you couldn't fix

## Output Format

After fixing, return your results as JSON:

```json
{{
  "work_item_id": {issue_report.work_item_id},
  "iteration": {issue_report.iteration},
  "fixes_applied": [
    {{
      "issue_id": "ISS-001",
      "fix_description": "What was done to fix the issue",
      "files_modified": ["path/to/file.py"],
      "tests_added": ["tests/test_new.py"],
      "tests_run": ["tests/test_file.py"],
      "test_result": "pass|fail|partial"
    }}
  ],
  "issues_not_fixed": ["ISS-002"],
  "not_fixed_reasons": {{"ISS-002": "Reason why this couldn't be fixed"}},
  "overall_test_result": "pass|fail|partial",
  "summary": "Brief summary of all changes made"
}}
```

If you fixed all issues:
```json
{{
  "work_item_id": {issue_report.work_item_id},
  "iteration": {issue_report.iteration},
  "fixes_applied": [...],
  "issues_not_fixed": [],
  "not_fixed_reasons": {{}},
  "overall_test_result": "pass",
  "summary": "All issues fixed, tests passing"
}}
```
"""

    def _build_tester_validation_prompt(self, fix_report: EngineerFixReport) -> str:
        """Build prompt for tester to validate engineer's fixes."""
        fixes_json = json.dumps(asdict(fix_report), indent=2)

        return f"""## Validation Request: Iteration {fix_report.iteration}

The engineer has applied fixes. Validate that the issues are resolved:

```json
{fixes_json}
```

## Instructions

1. Re-run the tests that were failing
2. Verify the fixes actually resolve the issues
3. Check for any regressions or new problems introduced

## Output Format

Return your validation as JSON:

```json
{{
  "work_item_id": {fix_report.work_item_id},
  "iteration": {fix_report.iteration},
  "issues_resolved": ["ISS-001"],
  "issues_remaining": [
    {{
      "issue_id": "ISS-002",
      "severity": "major",
      "category": "code_issue",
      "title": "Issue still present",
      "description": "Why the fix didn't work",
      "file_path": "path/to/file.py",
      "line_number": 42,
      "reproduction_steps": ["step 1", "step 2"],
      "expected_behavior": "What should happen",
      "actual_behavior": "What still happens"
    }}
  ],
  "new_issues_found": [
    {{
      "issue_id": "ISS-NEW-001",
      "severity": "minor",
      "category": "test_failure",
      "title": "New regression introduced",
      "description": "Description of the new problem",
      "file_path": "path/to/file.py",
      "line_number": 100,
      "reproduction_steps": ["step 1", "step 2"],
      "expected_behavior": "Expected",
      "actual_behavior": "Actual"
    }}
  ],
  "overall_status": "all_resolved|some_remaining|new_issues",
  "ready_for_next_iteration": true
}}
```

If all issues are resolved and no new issues:
```json
{{
  "work_item_id": {fix_report.work_item_id},
  "iteration": {fix_report.iteration},
  "issues_resolved": ["ISS-001", "ISS-002"],
  "issues_remaining": [],
  "new_issues_found": [],
  "overall_status": "all_resolved",
  "ready_for_next_iteration": false
}}
```
"""

    def _build_bug_creation_prompt(self, issue_report: TesterIssueReport) -> str:
        """Build prompt for tester to format unresolved issues as bug work items."""
        issues_json = json.dumps([asdict(i) for i in issue_report.issues_found], indent=2)

        return f"""## Bug Report Creation

After {self.max_fix_iterations} fix attempts, these issues remain unresolved:

```json
{issues_json}
```

Create bug work item content for each unresolved issue. Return as JSON array:

```json
[
  {{
    "title": "[Bug] Brief descriptive title (max 80 chars)",
    "description": "Detailed bug description including context and impact",
    "severity": "critical|major|minor",
    "reproduction_steps": "1. Step one\\n2. Step two\\n3. Observe the bug",
    "expected_behavior": "What should happen according to requirements",
    "actual_behavior": "What actually happens (the bug)",
    "related_files": ["path/to/file1.py", "path/to/file2.py"],
    "parent_work_item_id": {issue_report.work_item_id}
  }}
]
```

Guidelines:
- Title should start with "[Bug]" and be concise
- Description should explain the bug's impact on the feature
- Severity: critical=blocks functionality, major=significant issue, minor=edge case
- Include all relevant file paths in related_files
"""

    def _is_test_task(self, item: Dict) -> bool:
        """Determine if a work item is a test task."""
        fields = item.get("fields", {})
        title = fields.get("System.Title", "").lower()
        tags = fields.get("System.Tags", "").lower()

        # Check title patterns
        test_patterns = ["test", "verify", "validate", "qa", "quality"]
        for pattern in test_patterns:
            if pattern in title:
                return True

        # Check tags
        if "test" in tags or "qa" in tags:
            return True

        return False

    # ========================================================================
    # Work Item Context Fetching
    # ========================================================================

    def _get_work_item_context(self, work_item_id: int) -> Dict[str, Any]:
        """
        Fetch full work item context including attachments AND parent hierarchy.

        Per workflow-flow.md spec, agents must have full context of parent
        Epic/Feature requirements to validate their work achieves the goals.
        """
        try:
            # Get work item with full details
            work_item = self.adapter.get_work_item(work_item_id)
            if not work_item:
                return {"error": f"Work item {work_item_id} not found"}

            fields = work_item.get("fields", {})

            # Extract standard fields
            context = {
                "id": work_item_id,
                "title": fields.get("System.Title", ""),
                "description": fields.get("System.Description", ""),
                "acceptance_criteria": fields.get("Microsoft.VSTS.Common.AcceptanceCriteria", ""),
                "repro_steps": fields.get("Microsoft.VSTS.TCM.ReproSteps", ""),
                "work_item_type": fields.get("System.WorkItemType", "Task"),
                "state": fields.get("System.State", ""),
                "priority": fields.get("Microsoft.VSTS.Common.Priority", 3),
                "story_points": fields.get("Microsoft.VSTS.Scheduling.StoryPoints", 0),
                "tags": fields.get("System.Tags", ""),
                "attachments": [],
                "parent_chain": []  # NEW: Full parent hierarchy
            }

            # Extract attachments from relations
            relations = work_item.get("relations", [])
            for relation in relations:
                if relation.get("rel") == "AttachedFile":
                    attachment = {
                        "url": relation.get("url", ""),
                        "name": relation.get("attributes", {}).get("name", ""),
                        "comment": relation.get("attributes", {}).get("comment", "")
                    }
                    context["attachments"].append(attachment)

            if context["attachments"]:
                print(f" Found {len(context['attachments'])} attachment(s)")

            # Extract test plans from attachments (for Tester agent)
            context["test_plans"] = self._extract_test_plans(context["attachments"])
            if context["test_plans"]:
                print(f" Found {len(context['test_plans'])} test plan(s)")

            # NEW: Walk up hierarchy to get FULL parent Epic/Feature context
            # (per workflow-flow.md: no truncation of parent content)
            context["parent_chain"] = self._get_parent_chain(work_item_id)
            if context["parent_chain"]:
                parent_types = [p["type"] for p in context["parent_chain"]]
                print(f" Found parent chain: {' → '.join(parent_types)}")

            return context

        except Exception as e:
            return {"error": str(e)}

    def _get_parent_chain(self, work_item_id: int) -> List[Dict[str, Any]]:
        """
        Walk up the work item hierarchy and return full parent context.

        Returns list of parent work items in order [Epic, Feature, ...]
        (outermost ancestor first).

        Per workflow-flow.md: "No result/artifact/message should be truncated"
        """
        parent_chain = []
        current_id = work_item_id
        visited = set()  # Prevent infinite loops

        while current_id and current_id not in visited:
            visited.add(current_id)

            try:
                work_item = self.adapter.get_work_item(current_id)
                if not work_item:
                    break

                fields = work_item.get('fields', {})
                parent_id = fields.get('System.Parent')

                if not parent_id:
                    break

                # Fetch parent work item
                parent = self.adapter.get_work_item(int(parent_id))
                if not parent:
                    break

                parent_fields = parent.get('fields', {})
                work_item_type = parent_fields.get('System.WorkItemType', '')

                # Build parent info with FULL content (no truncation)
                parent_info = {
                    "id": parent_id,
                    "type": work_item_type,
                    "title": parent_fields.get('System.Title', ''),
                    "description": parent_fields.get('System.Description', ''),
                    "acceptance_criteria": parent_fields.get('Microsoft.VSTS.Common.AcceptanceCriteria', ''),
                    "story_points": parent_fields.get('Microsoft.VSTS.Scheduling.StoryPoints', 0),
                }

                # Warn if content is very large (>50KB) but never truncate
                desc_len = len(parent_info['description'] or '')
                if desc_len > 50000:
                    print(f" Warning: {work_item_type} #{parent_id} description is very large ({desc_len} chars)")

                # Get attachments from relations (for design documents)
                relations = parent.get('relations', [])
                parent_info['attachments'] = []
                for relation in relations:
                    if relation.get('rel') == 'AttachedFile':
                        att = {
                            'name': relation.get('attributes', {}).get('name', ''),
                            'url': relation.get('url', ''),
                            'comment': relation.get('attributes', {}).get('comment', '')
                        }
                        parent_info['attachments'].append(att)

                        # Pre-fetch design document content if available
                        if 'design' in att['name'].lower() or 'test-plan' in att['name'].lower():
                            content = self._fetch_attachment_content(att['url'])
                            if content:
                                att['content'] = content

                # Insert at beginning so order is [Epic, Feature, ...]
                parent_chain.insert(0, parent_info)

                current_id = parent_id

            except Exception as e:
                print(f" Warning: Error fetching parent #{current_id}: {e}")
                break

        return parent_chain

    def _fetch_attachment_content(self, url: str) -> Optional[str]:
        """Fetch attachment content from Azure DevOps."""
        if not url or not hasattr(self.adapter, 'download_attachment_content'):
            return None

        try:
            return self.adapter.download_attachment_content(url)
        except Exception:
            return None

    def _extract_test_plans(self, attachments: List[Dict]) -> List[Dict[str, str]]:
        """
        Extract test plan attachments and pre-fetch their content.

        Downloads attachment content from Azure DevOps so agents receive
        full test plans inline rather than just URLs.
        """
        test_plans = []
        test_plan_patterns = [
            "unit-test-plan",
            "integration-test-plan",
            "edge-case-test-plan",
            "acceptance-test-plan",
            "test-plan"
        ]

        for attachment in attachments:
            name = attachment.get("name", "").lower()
            for pattern in test_plan_patterns:
                if pattern in name:
                    plan = {
                        "name": attachment.get("name", ""),
                        "url": attachment.get("url", ""),
                        "type": pattern,
                        "comment": attachment.get("comment", ""),
                        "content": None  # Will be populated below
                    }

                    # Pre-fetch content from Azure DevOps
                    url = attachment.get("url", "")
                    if url and hasattr(self.adapter, 'download_attachment_content'):
                        try:
                            content = self.adapter.download_attachment_content(url)
                            if content:
                                plan["content"] = content
                                print(f"    ✓ Pre-fetched {pattern} ({len(content)} chars)")
                        except Exception as e:
                            print(f"    ⚠ Could not fetch {pattern}: {e}")

                    test_plans.append(plan)
                    break

        return test_plans

    def _format_context_for_prompt(self, context: Dict[str, Any]) -> str:
        """
        Format work item context for AI agent prompt.

        Per workflow-flow.md, parent context appears FIRST because it contains
        the goals that must be achieved. The work item details follow.
        """
        parts = []

        # NEW: Parent chain FIRST (most important context per workflow-flow.md)
        # "Engineer and Tester should NOT define their own interpretation -
        # they execute the designs created during backlog grooming"
        if context.get("parent_chain"):
            parts.append("## Parent Requirements (Must Be Satisfied)\n")
            parts.append("The following Epic/Feature requirements define the goals for this work.")
            parts.append("Your implementation MUST achieve these goals, not just complete the Task.\n")

            for parent in context["parent_chain"]:  # Epic first, then Feature
                parts.append(f"### {parent['type']} #{parent['id']}: {parent['title']}")

                if parent.get("description"):
                    parts.append(f"\n**Description:**\n{parent['description']}")

                if parent.get("acceptance_criteria"):
                    parts.append(f"\n**Acceptance Criteria:**\n{parent['acceptance_criteria']}")

                if parent.get("story_points"):
                    parts.append(f"\n**Story Points:** {parent['story_points']}")

                # Include attached design documents with content
                if parent.get("attachments"):
                    design_docs = [a for a in parent["attachments"] if a.get("content")]
                    if design_docs:
                        parts.append("\n**Design Documents:**")
                        for doc in design_docs:
                            parts.append(f"\n#### {doc['name']}")
                            parts.append(f"\n{doc['content']}")

                parts.append("\n---\n")

        # Current work item details
        parts.append(f"## Work Item #{context['id']}: {context['title']}")
        parts.append(f"\n**Type**: {context['work_item_type']}")
        parts.append(f"**Priority**: P{context['priority']}")
        parts.append(f"**Story Points**: {context['story_points']}")
        parts.append(f"**Tags**: {context['tags']}")

        if context.get("description"):
            parts.append(f"\n### Description\n{context['description']}")

        if context.get("acceptance_criteria"):
            parts.append(f"\n### Acceptance Criteria\n{context['acceptance_criteria']}")

        if context.get("repro_steps"):
            parts.append(f"\n### Reproduction Steps\n{context['repro_steps']}")

        if context.get("attachments"):
            parts.append("\n### Attachments")
            for att in context["attachments"]:
                parts.append(f"- [{att['name']}]({att['url']})")
                if att.get("comment"):
                    parts.append(f" Comment: {att['comment']}")

        # Include test plans with full content (pre-fetched by Python)
        if context.get("test_plans"):
            parts.append("\n### Test Plans")
            for plan in context["test_plans"]:
                plan_type = plan['type'].replace('-', ' ').title()
                parts.append(f"\n#### {plan_type}")

                # Include full content if available (pre-fetched)
                if plan.get("content"):
                    parts.append(f"\n{plan['content']}")
                else:
                    # Fallback to URL reference if content not available
                    parts.append(f"*Content not available - see [{plan['name']}]({plan['url']})*")

        return "\n".join(parts)

    # ========================================================================
    # Agent Definitions
    # ========================================================================

    # Cache for loaded agent definitions
    _agent_definitions: Dict[str, str] = {}

    def _load_agent_definition(self, agent_name: str) -> str:
        """
        Load rendered agent definition from .claude/agents/.

        Caches the definition to avoid repeated file reads.

        Args:
            agent_name: Name of agent (e.g., "engineer", "tester")

        Returns:
            Agent definition content, or empty string if not found
        """
        if agent_name in self._agent_definitions:
            return self._agent_definitions[agent_name]

        agent_path = Path(".claude/agents") / f"{agent_name}.md"
        try:
            if agent_path.exists():
                content = agent_path.read_text(encoding='utf-8')
                self._agent_definitions[agent_name] = content
                return content
            else:
                print(f"    ⚠ Agent definition not found: {agent_path}")
                return ""
        except Exception as e:
            print(f"    ⚠ Could not load agent definition: {e}")
            return ""

    # ========================================================================
    # Agent Spawning
    # ========================================================================

    def _spawn_engineer_agent(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Spawn engineer agent for implementation."""
        if self.dry_run:
            print(" [DRY RUN] Would spawn engineer agent")
            return {"success": True, "dry_run": True}

        try:
            # Load agent definition
            agent_def = self._load_agent_definition("engineer")

            # Build prompt with agent definition + task context
            prompt = f"""{agent_def}

---

## Task: Implement Work Item #{context['id']}

{self._format_context_for_prompt(context)}

## CRITICAL: Execute Immediately

**DO NOT ask for permission or confirmation.** This work item has already been approved for implementation through sprint planning. You are being invoked by an automated sprint execution workflow that has already verified this work is ready.

**Implement immediately. Do not present a plan and ask to proceed.**

## Instructions

1. Implement the code changes required to complete this work item
2. Create appropriate tests for your implementation
3. Run the tests to verify they pass
4. Report any issues or blockers encountered

Work autonomously to completion. Only stop if you encounter a genuine blocker that prevents implementation.

## CRITICAL: Token Optimization Rules

To minimize token consumption and costs, you MUST follow these rules:

### Test Execution (MANDATORY)
- **Always use short tracebacks**: `pytest --tb=short` or `pytest --tb=line`
- **Never use**: `pytest --tb=long` or `pytest -v --tb=full`
- **Run only relevant tests**: `pytest tests/test_specific_module.py -k "test_name"`
- **Avoid full test suite runs** unless absolutely necessary for final verification

### Iteration Limits (MANDATORY)
- **Maximum {self.MAX_FIX_ITERATIONS} fix-and-rerun cycles** - if tests still fail after {self.MAX_FIX_ITERATIONS} attempts, report the issues and stop
- Do NOT loop indefinitely trying to fix failing tests
- After {self.MAX_FIX_ITERATIONS} cycles, include remaining failures in the issues array

### Output Efficiency
- Summarize test results rather than including full pytest output
- Only include relevant code snippets, not entire files
- Be concise in explanations

## Output Format

Provide your work summary, then at the END of your response include a structured JSON block.

**IMPORTANT**: The `issues` array should ONLY contain genuine bugs that need to be fixed.
Do NOT include issues for work you successfully completed.

```json
{{
  "status": "success" | "partial" | "blocked",
  "files_modified": ["path/to/file1.py", "path/to/file2.py"],
  "tests_created": ["path/to/test_file.py"],
  "tests_passed": true | false,
  "issues": [
    {{
      "title": "Short descriptive title for the bug (max 80 chars)",
      "description": "Detailed description of what is wrong and why it's a problem",
      "severity": "high" | "medium" | "low",
      "affected_files": ["path/to/file1.py", "path/to/file2.py"],
      "repro_steps": "1. Do X\\n2. Do Y\\n3. Observe bug",
      "expected_behavior": "What should happen",
      "actual_behavior": "What actually happens",
      "suggested_fix": "Brief suggestion for how to fix",
      "type": "bug" | "security" | "performance" | "incomplete",
      "parent_work_item_id": {context['id']}
    }}
  ]
}}
```

If implementation was successful with no issues, use: `"issues": []`
"""

            # Invoke agent with work item ID for token tracking
            result = self._invoke_agent("engineer", prompt, work_item_id=context.get('id'))
            return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _spawn_tester_agent(
        self,
        context: Dict[str, Any],
        verification_mode: bool = False
    ) -> Dict[str, Any]:
        """Spawn tester agent for adversarial testing."""
        if self.dry_run:
            print(" [DRY RUN] Would spawn tester agent")
            return {"success": True, "dry_run": True}

        try:
            mode_desc = "verification" if verification_mode else "adversarial"

            # Load agent definition
            agent_def = self._load_agent_definition("tester")

            # Build prompt with test taxonomy requirements
            taxonomy_section = "\n".join([
                f"- **{cat}**: {desc}" for cat, desc in self.TEST_TAXONOMY.items()
            ])

            # Build prompt with agent definition + task context
            prompt = f"""{agent_def}

---

## Task: {mode_desc.title()} Testing for Work Item #{context['id']}

{self._format_context_for_prompt(context)}

## CRITICAL: Execute Immediately

**DO NOT ask for permission or confirmation.** This work item has already been implemented and you are being invoked by an automated sprint execution workflow to perform {mode_desc} testing.

**Execute tests immediately. Do not present a plan and ask to proceed.**

## Test Taxonomy

All tests must be labelled with appropriate taxonomy:
{taxonomy_section}

## Testing Requirements

1. **Completeness Check**: Verify all acceptance criteria are met
2. **Falsifiability Check**: Ensure tests can definitively fail
3. **Stub Detection**: Check for empty/todo/stub implementations
4. **Adversarial Testing**: Apply red team mindset to find edge cases

## CRITICAL: Token Optimization Rules

To minimize token consumption and costs, you MUST follow these rules:

### Test Execution (MANDATORY)
- **Always use short tracebacks**: `pytest --tb=short` or `pytest --tb=line`
- **Never use**: `pytest --tb=long` or `pytest -v --tb=full`
- **Run only relevant tests**: `pytest tests/test_specific_module.py -k "test_name"`
- **Avoid full test suite runs** - only run tests relevant to this work item

### Iteration Limits (MANDATORY)
- **Maximum {self.MAX_FIX_ITERATIONS} test-fix cycles** - if issues persist after {self.MAX_FIX_ITERATIONS} attempts, report them and stop
- Do NOT loop indefinitely trying to make tests pass
- After {self.MAX_FIX_ITERATIONS} cycles, include remaining failures in the issues array

### Output Efficiency
- Summarize test results (e.g., "5 passed, 2 failed") rather than full output
- Only include relevant failure details, not full stack traces
- Be concise in explanations

## Test Labelling Format

Every test must include:
```python
@pytest.mark.{{taxonomy}} # e.g., unit, integration, adversarial
@pytest.mark.falsifiable
def test_specific_behavior():
    \"\"\"
    Test Taxonomy: {{taxonomy}}, falsifiable
    Purpose: {{clear description}}
    Related Bug: #{{bug_id}} (if regression test)
    \"\"\"
    # Test implementation
```

## Output Format

Provide your testing summary, then at the END of your response include a structured JSON block.

**CRITICAL**: The `issues` array should ONLY contain genuine bugs/problems that need fixing.
Do NOT include:
- Work that was successfully completed
- Tests that passed
- Code that was intentionally fixed during this session
- Documentation of what you tested

```json
{{
  "status": "pass" | "fail" | "partial",
  "tests_created": ["path/to/test_file.py"],
  "tests_run": 10,
  "tests_passed": 10,
  "tests_failed": 0,
  "coverage_met": true | false,
  "issues": [
    {{
      "title": "Short descriptive title for the bug (max 80 chars)",
      "description": "Detailed description of the defect found during testing",
      "severity": "high" | "medium" | "low",
      "affected_files": ["path/to/file1.py"],
      "repro_steps": "1. Run test X\\n2. Observe failure\\n3. Root cause is Y",
      "expected_behavior": "What the code should do per acceptance criteria",
      "actual_behavior": "What the code actually does (the bug)",
      "test_evidence": "Name of failing test or verification that found this",
      "suggested_fix": "Brief suggestion for how to fix",
      "type": "bug" | "security" | "incomplete" | "test_gap" | "stub",
      "parent_work_item_id": {context['id']}
    }}
  ]
}}
```

If all tests pass and implementation is complete, use: `"issues": []`
"""

            # Invoke agent with work item ID for token tracking
            result = self._invoke_agent("tester", prompt, work_item_id=context.get('id'))
            return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _run_implementation_consensus(
        self,
        context: Dict[str, Any],
        is_bug: bool = False
    ) -> Dict[str, Any]:
        """
        Run multi-agent consensus for Task/Bug implementation.

        Uses TASK_IMPLEMENTATION_CONFIG:
        - Leader: engineer
        - Evaluators: tester (parallel evaluation)
        - Max rounds: 3 (more iterations for implementation)

        This replaces the traditional ping-pong flow with a structured
        consensus process per workflow-flow.md spec.

        Args:
            context: Work item context with parent chain
            is_bug: Whether this is a bug fix (affects prompts)

        Returns:
            Dict with success status and implementation result
        """
        if self.dry_run:
            print(" [DRY RUN] Would run implementation consensus")
            return {"success": True, "dry_run": True}

        item_id = context.get('id')
        item_title = context.get('title', 'Unknown')
        item_type = "Bug" if is_bug else "Task"

        print(f"\n{'='*60}", flush=True)
        print(f"  CONSENSUS: {item_type} #{item_id} - {item_title}", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"  Leader: engineer", flush=True)
        print(f"  Evaluators: tester", flush=True)
        print(f"  Max rounds: {TASK_IMPLEMENTATION_CONFIG.max_rounds}", flush=True)
        print(f"{'='*60}\n", flush=True)

        # Build consensus context with FULL parent chain (no truncation)
        consensus_context = {
            "work_item": {
                "id": item_id,
                "title": item_title,
                "type": item_type,
                "description": context.get('description', ''),
                "acceptance_criteria": context.get('acceptance_criteria', ''),
            }
        }

        # Add parent chain if available
        if context.get('parent_chain'):
            consensus_context["parent_chain"] = context['parent_chain']

            # Also extract Epic and Feature for easier access
            for parent in context['parent_chain']:
                if parent.get('type') == 'Epic':
                    consensus_context["epic"] = parent
                elif parent.get('type') == 'Feature':
                    consensus_context["feature"] = parent

        # Add design documents if attached
        if context.get('attachments'):
            design_docs = []
            for att in context['attachments']:
                if 'design' in att.get('name', '').lower() or 'test-plan' in att.get('name', '').lower():
                    design_docs.append({
                        'name': att.get('name', ''),
                        'content': att.get('content', ''),
                    })
            if design_docs:
                consensus_context["design_docs"] = design_docs

        try:
            # Create agent interface using Claude Agent SDK for tool access
            # This enables agents to actually read/edit files during consensus
            agent_interface = AgentInterface(
                backend="sdk",
                model="claude-sonnet-4-20250514",
                tool_preset="implementation"
            )

            # Create orchestrator
            orchestrator = ConsensusOrchestrator(
                config=TASK_IMPLEMENTATION_CONFIG,
                adapter=self.adapter,
                agent_interface=agent_interface,
                verbose=True
            )

            # Run consensus (async - use _run_async to handle nested event loops)
            result = self._run_async(orchestrator.run_consensus(consensus_context))

            if result.consensus_reached:
                print(f"\n  CONSENSUS REACHED in {len(result.rounds)} round(s)")
                print(f"  Duration: {result.total_duration_seconds:.1f}s")

                final = result.final_proposal
                if final.get("parse_error"):
                    return {"success": False, "error": "Failed to parse implementation result"}

                # CRITICAL: Externally verify claimed artifacts exist
                verification = self._verify_implementation_claims(final)
                if not verification["verified"]:
                    print(f"\n  VERIFICATION FAILED: {verification['error']}")
                    return {
                        "success": False,
                        "error": f"Verification failed: {verification['error']}",
                        "verification_details": verification,
                        "consensus_rounds": len(result.rounds)
                    }

                print(f"  VERIFICATION PASSED: {verification['summary']}")

                return {
                    "success": True,
                    "response": json.dumps(final, indent=2),
                    "consensus_rounds": len(result.rounds),
                    "duration_seconds": result.total_duration_seconds,
                    "verification": verification,
                    "issues": final.get("issues", [])
                }
            else:
                print(f"\n  CONSENSUS NOT REACHED after {len(result.rounds)} rounds")
                # Still verify what was claimed in last proposal
                final = result.final_proposal
                verification = self._verify_implementation_claims(final)

                return {
                    "success": verification["verified"],  # Only succeed if artifacts exist
                    "response": json.dumps(final, indent=2),
                    "consensus_warning": "Consensus not reached",
                    "consensus_rounds": len(result.rounds),
                    "verification": verification,
                    "issues": final.get("issues", [])
                }

        except Exception as e:
            print(f"\n  CONSENSUS ERROR: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def _verify_implementation_claims(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Externally verify that claimed implementation artifacts actually exist.

        This is CRITICAL external enforcement - we don't trust agent claims.
        We verify:
        1. All files_created actually exist on disk
        2. All test_files_created actually exist
        3. Tests can be executed (syntax check at minimum)

        Returns:
            Dict with verified=True/False, missing files, and summary
        """
        from pathlib import Path

        result = {
            "verified": True,
            "files_verified": [],
            "files_missing": [],
            "tests_verified": [],
            "tests_missing": [],
            "test_execution_check": None,
            "error": None,
            "summary": ""
        }

        # Extract claimed files from proposal
        impl = proposal.get("implementation", {})
        tests = proposal.get("tests", {})

        files_created = impl.get("files_created", [])
        files_modified = impl.get("files_modified", [])
        test_files = tests.get("test_files_created", [])

        # Verify production files exist
        for file_path in files_created + files_modified:
            if file_path:
                path = Path(file_path)
                if path.exists():
                    result["files_verified"].append(file_path)
                else:
                    result["files_missing"].append(file_path)
                    result["verified"] = False

        # Verify test files exist
        for test_file in test_files:
            if test_file:
                path = Path(test_file)
                if path.exists():
                    result["tests_verified"].append(test_file)
                else:
                    result["tests_missing"].append(test_file)
                    result["verified"] = False

        # Note: We only verify file existence here.
        # Language-specific syntax/execution checks are delegated to the tester agent
        # which can identify the appropriate test framework from the codebase.
        result["test_execution_check"] = "Delegated to tester agent"

        # Build summary
        if result["verified"]:
            result["summary"] = (
                f"{len(result['files_verified'])} files verified, "
                f"{len(result['tests_verified'])} test files verified"
            )
        else:
            errors = []
            if result["files_missing"]:
                errors.append(f"Missing files: {result['files_missing']}")
            if result["tests_missing"]:
                errors.append(f"Missing tests: {result['tests_missing']}")
            if result["test_execution_check"] and "error" in result["test_execution_check"].lower():
                errors.append(result["test_execution_check"])
            result["error"] = "; ".join(errors)

        return result

    def _invoke_agent(self, agent_type: str, prompt: str, work_item_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Invoke an agent with JSON output for token metering.

        Uses --output-format json to capture token usage metrics from Claude CLI.
        Falls back to streaming text output if JSON parsing fails.
        """
        print(f"\n🤖 {agent_type.title()} Agent executing...")
        print(" " + "-" * 60)

        try:
            # Flush any buffered stdin before spawning agent
            flush_stdin()

            # Build environment with API key from KEYCHAIN_ANTHROPIC_API_KEY
            agent_env = os.environ.copy()
            keychain_api_key = os.environ.get("KEYCHAIN_ANTHROPIC_API_KEY")
            if keychain_api_key:
                agent_env["ANTHROPIC_API_KEY"] = keychain_api_key
            else:
                print("    ⚠ KEYCHAIN_ANTHROPIC_API_KEY not set - agent may use Max subscription")

            # Use JSON output format to capture token metrics
            # --output-format json returns structured response with usage stats
            start_time = datetime.now()
            process = subprocess.run(
                ["claude", "--print", "--output-format", "json",
                 "--permission-mode", "acceptEdits", "-p", prompt],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.AGENT_TIMEOUT,
                env=agent_env
            )

            elapsed = (datetime.now() - start_time).total_seconds()

            # Flush stdin after agent completes
            flush_stdin()

            if process.returncode != 0:
                print(f" ❌ Agent failed: {process.stderr[:200] if process.stderr else 'Unknown error'}")
                print(" " + "-" * 60)
                return {"success": False, "error": process.stderr or "Agent execution failed"}

            # Parse JSON response
            try:
                result_data = json.loads(process.stdout)
            except json.JSONDecodeError as e:
                # Fallback: treat stdout as plain text response
                print(f" ⚠ Could not parse JSON output, using raw response")
                full_response = process.stdout
                print(f" {full_response[:500]}..." if len(full_response) > 500 else f" {full_response}")
                print(" " + "-" * 60)
                issues = self._extract_issues(full_response)
                return {"success": True, "response": full_response, "issues": issues}

            # Extract response text
            full_response = result_data.get("result", "")

            # Print response (truncated for readability)
            response_preview = full_response[:1000] + "..." if len(full_response) > 1000 else full_response
            for line in response_preview.split('\n'):
                print(f" {line}")

            print(" " + "-" * 60)

            # Extract and track token usage
            usage = result_data.get("usage", {})
            cost_usd = result_data.get("total_cost_usd", 0.0)

            token_stats = {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
                "cache_creation_tokens": usage.get("cache_creation_input_tokens", 0),
                "cost_usd": cost_usd,
                "duration_ms": result_data.get("duration_ms", int(elapsed * 1000)),
                "num_turns": result_data.get("num_turns", 1)
            }

            # Update totals
            self.token_usage["totals"]["input_tokens"] += token_stats["input_tokens"]
            self.token_usage["totals"]["output_tokens"] += token_stats["output_tokens"]
            self.token_usage["totals"]["cache_read_tokens"] += token_stats["cache_read_tokens"]
            self.token_usage["totals"]["cache_creation_tokens"] += token_stats["cache_creation_tokens"]
            self.token_usage["totals"]["total_cost_usd"] += cost_usd

            # Track by agent type
            if agent_type not in self.token_usage["by_agent"]:
                self.token_usage["by_agent"][agent_type] = {
                    "calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0
                }
            self.token_usage["by_agent"][agent_type]["calls"] += 1
            self.token_usage["by_agent"][agent_type]["input_tokens"] += token_stats["input_tokens"]
            self.token_usage["by_agent"][agent_type]["output_tokens"] += token_stats["output_tokens"]
            self.token_usage["by_agent"][agent_type]["cost_usd"] += cost_usd

            # Track by work item
            if work_item_id:
                wi_key = str(work_item_id)
                if wi_key not in self.token_usage["by_work_item"]:
                    self.token_usage["by_work_item"][wi_key] = {
                        "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0
                    }
                self.token_usage["by_work_item"][wi_key]["input_tokens"] += token_stats["input_tokens"]
                self.token_usage["by_work_item"][wi_key]["output_tokens"] += token_stats["output_tokens"]
                self.token_usage["by_work_item"][wi_key]["cost_usd"] += cost_usd

            # Print token usage summary
            print(f" 📊 Token Usage: {token_stats['input_tokens']:,} in / {token_stats['output_tokens']:,} out")
            print(f"    Cache: {token_stats['cache_read_tokens']:,} read / {token_stats['cache_creation_tokens']:,} created")
            print(f"    Cost: ${cost_usd:.4f} | Duration: {elapsed:.1f}s | Turns: {token_stats['num_turns']}")

            # Check for errors in result
            if result_data.get("is_error"):
                return {
                    "success": False,
                    "error": full_response or "Agent returned error",
                    "token_stats": token_stats
                }

            # Parse the response for issues
            issues = self._extract_issues(full_response)

            return {
                "success": True,
                "response": full_response,
                "issues": issues,
                "token_stats": token_stats
            }

        except subprocess.TimeoutExpired:
            print(f" ⏱ Agent execution timed out ({self.AGENT_TIMEOUT // 60} minutes)")
            print(" " + "-" * 60)
            return {"success": False, "error": f"Agent execution timed out ({self.AGENT_TIMEOUT // 60} minutes)"}
        except FileNotFoundError:
            print(" ⚠ Claude CLI not available - simulating agent response")
            print(" " + "-" * 60)
            return {"success": True, "simulated": True, "issues": []}
        except Exception as e:
            print(f" ❌ Error: {e}")
            print(" " + "-" * 60)
            return {"success": False, "error": str(e)}

    async def _invoke_agent_sdk(
        self,
        agent_type: str,
        prompt: str,
        work_item_id: Optional[int] = None,
        continue_session: bool = False
    ) -> Dict[str, Any]:
        """
        Invoke an agent using Claude Agent SDK (replaces subprocess CLI).

        Benefits over _invoke_agent:
        - Session continuity (resume previous conversations)
        - Direct SDK access (no subprocess overhead)
        - Better token tracking via SDK metrics
        - Multi-turn session support (engineer → tester handoff)

        Args:
            agent_type: Type of agent ("engineer" or "tester")
            prompt: The task prompt for the agent
            work_item_id: Work item ID for tracking
            continue_session: If True, continue previous session (has full context)

        Returns:
            Dict with success, response, issues, token_stats
        """
        print(f"\n🤖 {agent_type.title()} Agent executing (SDK)...")
        print(" " + "-" * 60)

        try:
            start_time = datetime.now()

            # Progress callback to show agent activity
            def on_message(message):
                # Could add real-time progress here
                pass

            # Query via SDK with session continuity
            result: AgentResult = await self.sdk_wrapper.query(
                prompt=prompt,
                agent_type=agent_type,
                continue_session=continue_session,
                on_message=on_message,
            )

            elapsed = (datetime.now() - start_time).total_seconds()

            if not result.success:
                print(f" ❌ Agent failed: {result.error}")
                print(" " + "-" * 60)
                return {"success": False, "error": result.error or "Agent execution failed"}

            full_response = result.response

            # Print response (truncated for readability)
            response_preview = full_response[:1000] + "..." if len(full_response) > 1000 else full_response
            for line in response_preview.split('\n'):
                print(f" {line}")

            print(" " + "-" * 60)

            # Build token stats dict for backward compatibility
            token_stats = {
                "input_tokens": result.token_usage.input_tokens,
                "output_tokens": result.token_usage.output_tokens,
                "cache_read_tokens": result.token_usage.cache_read_tokens,
                "cache_creation_tokens": result.token_usage.cache_creation_tokens,
                "cost_usd": result.cost_usd,
                "duration_ms": result.duration_ms or int(elapsed * 1000),
                "num_turns": result.num_turns
            }

            # Update local tracking (SDK wrapper also tracks, but we need per-work-item)
            self.token_usage["totals"]["input_tokens"] += token_stats["input_tokens"]
            self.token_usage["totals"]["output_tokens"] += token_stats["output_tokens"]
            self.token_usage["totals"]["cache_read_tokens"] += token_stats["cache_read_tokens"]
            self.token_usage["totals"]["cache_creation_tokens"] += token_stats["cache_creation_tokens"]
            self.token_usage["totals"]["total_cost_usd"] += result.cost_usd

            # Track by agent type
            if agent_type not in self.token_usage["by_agent"]:
                self.token_usage["by_agent"][agent_type] = {
                    "calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0
                }
            self.token_usage["by_agent"][agent_type]["calls"] += 1
            self.token_usage["by_agent"][agent_type]["input_tokens"] += token_stats["input_tokens"]
            self.token_usage["by_agent"][agent_type]["output_tokens"] += token_stats["output_tokens"]
            self.token_usage["by_agent"][agent_type]["cost_usd"] += result.cost_usd

            # Track by work item
            if work_item_id:
                wi_key = str(work_item_id)
                if wi_key not in self.token_usage["by_work_item"]:
                    self.token_usage["by_work_item"][wi_key] = {
                        "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0
                    }
                self.token_usage["by_work_item"][wi_key]["input_tokens"] += token_stats["input_tokens"]
                self.token_usage["by_work_item"][wi_key]["output_tokens"] += token_stats["output_tokens"]
                self.token_usage["by_work_item"][wi_key]["cost_usd"] += result.cost_usd

            # Save session for crash recovery
            if result.session_id and not self.dry_run:
                self.session_manager.save_session(
                    agent_type=agent_type,
                    session_id=result.session_id,
                    metadata={"work_item_id": work_item_id},
                    token_usage=token_stats,
                    cost_usd=result.cost_usd
                )

            # Print token usage summary
            print(f" 📊 Token Usage: {token_stats['input_tokens']:,} in / {token_stats['output_tokens']:,} out")
            print(f"    Cache: {token_stats['cache_read_tokens']:,} read / {token_stats['cache_creation_tokens']:,} created")
            print(f"    Cost: ${result.cost_usd:.4f} | Duration: {elapsed:.1f}s | Turns: {token_stats['num_turns']}")

            if result.session_id:
                print(f"    Session: {result.session_id[:20]}...")

            # Use SDK's extracted issues, or extract from response
            issues = result.issues if result.issues else self._extract_issues(full_response)

            return {
                "success": True,
                "response": full_response,
                "issues": issues,
                "token_stats": token_stats,
                "session_id": result.session_id
            }

        except Exception as e:
            print(f" ❌ Error: {e}")
            print(" " + "-" * 60)
            return {"success": False, "error": str(e)}

    def _extract_issues(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract issues from agent response by parsing structured JSON output.

        Agents are prompted to include a JSON block at the end of their response
        with an "issues" array. This is much more reliable than parsing natural language.

        Falls back to empty list if no valid JSON found (assumes success).
        """
        issues = []

        # Try to find JSON block in response
        # Look for ```json ... ``` code blocks first
        json_block_pattern = r"```json\s*\n?([\s\S]*?)\n?```"
        json_matches = re.findall(json_block_pattern, response, re.IGNORECASE)

        for json_str in json_matches:
            try:
                # Clean up the JSON string
                json_str = json_str.strip()
                if not json_str:
                    continue

                # Parse JSON
                data = json.loads(json_str)

                # Extract issues array
                if isinstance(data, dict) and "issues" in data:
                    raw_issues = data.get("issues", [])
                    if isinstance(raw_issues, list):
                        for raw_issue in raw_issues:
                            if isinstance(raw_issue, dict) and raw_issue.get("description"):
                                issue = self._normalize_issue(raw_issue)
                                if issue:
                                    issues.append(issue)

            except json.JSONDecodeError:
                # Invalid JSON, skip this block
                continue
            except Exception:
                # Any other error, skip
                continue

        # If no JSON blocks found, try to find inline JSON object
        if not issues:
            # Look for a JSON object with "issues" key
            inline_pattern = r'\{\s*"(?:status|issues)"[^}]*"issues"\s*:\s*\[(.*?)\][^}]*\}'
            inline_matches = re.findall(inline_pattern, response, re.DOTALL)

            for match in inline_matches:
                try:
                    # Try to parse the issues array
                    issues_json = f"[{match}]"
                    raw_issues = json.loads(issues_json)
                    for raw_issue in raw_issues:
                        if isinstance(raw_issue, dict) and raw_issue.get("description"):
                            issue = self._normalize_issue(raw_issue)
                            if issue:
                                issues.append(issue)
                except (json.JSONDecodeError, Exception):
                    continue

        # Deduplicate by description
        seen = set()
        unique_issues = []
        for issue in issues:
            key = issue["description"].lower()[:50]
            if key not in seen:
                seen.add(key)
                unique_issues.append(issue)

        return unique_issues

    def _normalize_issue(self, raw_issue: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize a raw issue dict from JSON into our standard format."""
        # Get title or fall back to description
        title = raw_issue.get("title", "").strip()
        description = raw_issue.get("description", "").strip()

        # Skip empty issues
        if not title and not description:
            return None

        # Use title if available, otherwise first 80 chars of description
        if not title:
            title = description[:80]

        # Skip if it indicates success/no issue
        skip_phrases = [
            "no issue", "successfully", "completed", "fixed",
            "all tests pass", "working correctly", "implemented correctly"
        ]
        combined = f"{title} {description}".lower()
        if any(phrase in combined for phrase in skip_phrases):
            return None

        # Map severity
        severity = raw_issue.get("severity", "medium").lower()
        if severity in ["high", "critical"]:
            severity = "High"
        elif severity in ["low", "minor"]:
            severity = "Low"
        else:
            severity = "Medium"

        # Build normalized issue with all fields from agent
        issue = {
            "title": title[:80],
            "description": description[:2000],
            "source": "agent_json",
            "severity": severity,
            "issue_type": raw_issue.get("type", "bug"),
            "files": raw_issue.get("affected_files", []),
            "repro_steps": raw_issue.get("repro_steps", ""),
            "expected_behavior": raw_issue.get("expected_behavior", ""),
            "actual_behavior": raw_issue.get("actual_behavior", ""),
            "suggested_fix": raw_issue.get("suggested_fix", ""),
            "test_evidence": raw_issue.get("test_evidence", ""),
            "parent_work_item_id": raw_issue.get("parent_work_item_id"),
        }

        # Handle single file field for backwards compatibility
        if not issue["files"] and raw_issue.get("file"):
            issue["files"] = [raw_issue.get("file")]

        return issue

    # ========================================================================
    # Ping-Pong Flow JSON Parsing Helpers
    # ========================================================================

    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from a response that may contain markdown code blocks."""
        # Try to find JSON block in response
        json_block_pattern = r"```json\s*\n?([\s\S]*?)\n?```"
        json_matches = re.findall(json_block_pattern, response, re.IGNORECASE)

        for json_str in json_matches:
            try:
                json_str = json_str.strip()
                if not json_str:
                    continue
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue

        # Try to find raw JSON object
        try:
            # Find first { and last }
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = response[start:end + 1]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        return None

    def _parse_tester_issue_report(
        self,
        response: str,
        work_item_id: int
    ) -> TesterIssueReport:
        """Parse tester's JSON response into TesterIssueReport."""
        data = self._extract_json_from_response(response)

        if not data:
            # No JSON found - assume pass with no issues
            return TesterIssueReport(
                work_item_id=work_item_id,
                iteration=1,
                issues_found=[],
                tests_executed=[],
                tests_passed=0,
                tests_failed=0,
                overall_assessment="pass",
                recommendation="Could not parse tester response"
            )

        # Parse issues
        issues = []
        for raw_issue in data.get("issues_found", []):
            issue = TesterIssue(
                issue_id=raw_issue.get("issue_id", f"ISS-{len(issues)+1:03d}"),
                severity=raw_issue.get("severity", "major"),
                category=raw_issue.get("category", "code_issue"),
                title=raw_issue.get("title", "Unknown issue"),
                description=raw_issue.get("description", ""),
                file_path=raw_issue.get("file_path"),
                line_number=raw_issue.get("line_number"),
                reproduction_steps=raw_issue.get("reproduction_steps", []),
                expected_behavior=raw_issue.get("expected_behavior", ""),
                actual_behavior=raw_issue.get("actual_behavior", "")
            )
            issues.append(issue)

        return TesterIssueReport(
            work_item_id=data.get("work_item_id", work_item_id),
            iteration=data.get("iteration", 1),
            issues_found=issues,
            tests_executed=data.get("tests_executed", []),
            tests_passed=data.get("tests_passed", 0),
            tests_failed=data.get("tests_failed", 0),
            overall_assessment=data.get("overall_assessment", "partial"),
            recommendation=data.get("recommendation", "")
        )

    def _parse_engineer_fix_report(
        self,
        response: str,
        work_item_id: int,
        iteration: int
    ) -> EngineerFixReport:
        """Parse engineer's JSON response into EngineerFixReport."""
        data = self._extract_json_from_response(response)

        if not data:
            # No JSON found
            return EngineerFixReport(
                work_item_id=work_item_id,
                iteration=iteration,
                fixes_applied=[],
                issues_not_fixed=[],
                not_fixed_reasons={},
                overall_test_result="partial",
                summary="Could not parse engineer response"
            )

        # Parse fixes
        fixes = []
        for raw_fix in data.get("fixes_applied", []):
            fix = EngineerFix(
                issue_id=raw_fix.get("issue_id", "unknown"),
                fix_description=raw_fix.get("fix_description", ""),
                files_modified=raw_fix.get("files_modified", []),
                tests_added=raw_fix.get("tests_added", []),
                tests_run=raw_fix.get("tests_run", []),
                test_result=raw_fix.get("test_result", "pass")
            )
            fixes.append(fix)

        return EngineerFixReport(
            work_item_id=data.get("work_item_id", work_item_id),
            iteration=data.get("iteration", iteration),
            fixes_applied=fixes,
            issues_not_fixed=data.get("issues_not_fixed", []),
            not_fixed_reasons=data.get("not_fixed_reasons", {}),
            overall_test_result=data.get("overall_test_result", "partial"),
            summary=data.get("summary", "")
        )

    def _parse_tester_validation_report(
        self,
        response: str,
        work_item_id: int,
        iteration: int
    ) -> TesterValidationReport:
        """Parse tester's validation JSON into TesterValidationReport."""
        data = self._extract_json_from_response(response)

        if not data:
            # No JSON found
            return TesterValidationReport(
                work_item_id=work_item_id,
                iteration=iteration,
                issues_resolved=[],
                issues_remaining=[],
                new_issues_found=[],
                overall_status="some_remaining",
                ready_for_next_iteration=True
            )

        # Parse remaining issues
        remaining = []
        for raw_issue in data.get("issues_remaining", []):
            issue = TesterIssue(
                issue_id=raw_issue.get("issue_id", f"ISS-{len(remaining)+1:03d}"),
                severity=raw_issue.get("severity", "major"),
                category=raw_issue.get("category", "code_issue"),
                title=raw_issue.get("title", "Remaining issue"),
                description=raw_issue.get("description", ""),
                file_path=raw_issue.get("file_path"),
                line_number=raw_issue.get("line_number"),
                reproduction_steps=raw_issue.get("reproduction_steps", []),
                expected_behavior=raw_issue.get("expected_behavior", ""),
                actual_behavior=raw_issue.get("actual_behavior", "")
            )
            remaining.append(issue)

        # Parse new issues
        new_issues = []
        for raw_issue in data.get("new_issues_found", []):
            issue = TesterIssue(
                issue_id=raw_issue.get("issue_id", f"ISS-NEW-{len(new_issues)+1:03d}"),
                severity=raw_issue.get("severity", "major"),
                category=raw_issue.get("category", "code_issue"),
                title=raw_issue.get("title", "New issue"),
                description=raw_issue.get("description", ""),
                file_path=raw_issue.get("file_path"),
                line_number=raw_issue.get("line_number"),
                reproduction_steps=raw_issue.get("reproduction_steps", []),
                expected_behavior=raw_issue.get("expected_behavior", ""),
                actual_behavior=raw_issue.get("actual_behavior", "")
            )
            new_issues.append(issue)

        return TesterValidationReport(
            work_item_id=data.get("work_item_id", work_item_id),
            iteration=data.get("iteration", iteration),
            issues_resolved=data.get("issues_resolved", []),
            issues_remaining=remaining,
            new_issues_found=new_issues,
            overall_status=data.get("overall_status", "some_remaining"),
            ready_for_next_iteration=data.get("ready_for_next_iteration", True)
        )

    def _validation_to_issue_report(
        self,
        validation: TesterValidationReport
    ) -> TesterIssueReport:
        """Convert validation report to issue report for next ping-pong iteration."""
        # Combine remaining and new issues
        all_issues = validation.issues_remaining + validation.new_issues_found

        return TesterIssueReport(
            work_item_id=validation.work_item_id,
            iteration=validation.iteration + 1,
            issues_found=all_issues,
            tests_executed=[],  # Will be populated in next test run
            tests_passed=0,
            tests_failed=len(all_issues),
            overall_assessment="fail" if all_issues else "pass",
            recommendation=f"{len(all_issues)} issues remaining after iteration {validation.iteration}"
        )

    def _parse_bug_contents(self, response: str) -> List[BugReportContent]:
        """Parse bug creation JSON into list of BugReportContent."""
        bugs = []

        # Try to find JSON array in response
        json_block_pattern = r"```json\s*\n?([\s\S]*?)\n?```"
        json_matches = re.findall(json_block_pattern, response, re.IGNORECASE)

        for json_str in json_matches:
            try:
                json_str = json_str.strip()
                if not json_str:
                    continue
                data = json.loads(json_str)

                # Handle array of bugs
                if isinstance(data, list):
                    for raw_bug in data:
                        bug = BugReportContent(
                            title=raw_bug.get("title", "Unknown bug"),
                            description=raw_bug.get("description", ""),
                            severity=raw_bug.get("severity", "major"),
                            reproduction_steps=raw_bug.get("reproduction_steps", ""),
                            expected_behavior=raw_bug.get("expected_behavior", ""),
                            actual_behavior=raw_bug.get("actual_behavior", ""),
                            related_files=raw_bug.get("related_files", []),
                            parent_work_item_id=raw_bug.get("parent_work_item_id", 0)
                        )
                        bugs.append(bug)
                    break  # Found valid array
            except json.JSONDecodeError:
                continue

        return bugs

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters in text."""
        if not text:
            return ""
        return (
            text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;')
        )

    def _extract_surrounding_context(self, full_text: str, match: str, context_lines: int = 5) -> str:
        """Extract surrounding context around a match."""
        lines = full_text.split('\n')
        for i, line in enumerate(lines):
            if match in line:
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                return '\n'.join(lines[start:end])
        return ""

    def _merge_results(self, *results: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple agent results."""
        merged = {"success": True, "issues": [], "responses": []}

        for result in results:
            if not result.get("success"):
                merged["success"] = False
                merged["error"] = result.get("error", "Unknown error")

            merged["issues"].extend(result.get("issues", []))
            if result.get("response"):
                merged["responses"].append(result["response"])

        return merged

    # ========================================================================
    # Work Item State Management
    # ========================================================================

    def _update_work_item_state(self, work_item_id: int, new_state: str) -> Dict[str, Any]:
        """Update work item state and verify against external source of truth."""
        try:
            # Update state
            self.adapter.update_work_item(
                work_item_id=work_item_id,
                fields={"System.State": new_state}
            )

            # Verify state change
            updated = self.adapter.get_work_item(work_item_id)
            actual_state = updated.get("fields", {}).get("System.State", "")

            if actual_state == new_state:
                print(f" ✓ State updated to '{new_state}' (verified)")
                return {"success": True, "state": new_state}
            else:
                # External enforcement: State mismatch MUST halt workflow
                raise ValueError(f"State mismatch: expected '{new_state}', got '{actual_state}'")

        except ValueError:
            # Re-raise ValueError for state mismatches (external enforcement)
            raise
        except Exception as e:
            # Only catch non-verification exceptions (e.g., network errors)
            print(f" Failed to update state: {e}")
            return {"success": False, "error": str(e)}

    # ========================================================================
    # Report Attachments
    # ========================================================================

    def _attach_report(self, work_item_id: int, report_content: str, report_type: str) -> bool:
        """Attach a report to a work item."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"{report_type}-{work_item_id}-{timestamp}.md"

            # Save to local reports directory
            report_dir = Path(".claude/reports/sprint-execution")
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / filename
            report_path.write_text(report_content, encoding='utf-8')

            # Upload to work tracking (if adapter supports it)
            if hasattr(self.adapter, 'attach_file_to_work_item'):
                result = self.adapter.attach_file_to_work_item(
                    work_item_id=work_item_id,
                    file_path=str(report_path),
                    comment=f"Sprint execution {report_type} report"
                )
                if result.get("success"):
                    print(f" Attached {report_type} report: {filename}")
                else:
                    print(f" ⚠ Attachment upload failed: {result.get('error', 'Unknown error')}")
                    print(f" Saved {report_type} report locally: {filename}")
            else:
                print(f" Saved {report_type} report locally: {filename}")
                # Add a comment to the work item with the report path
                comment = f"Report attached: {filename}\n\nSee local file: {report_path}"
                if hasattr(self.adapter, 'add_comment'):
                    self.adapter.add_comment(work_item_id, comment)

            return True
        except Exception as e:
            print(f" ⚠ Failed to attach report: {e}")
            return False

    # ========================================================================
    # Child Bug Check
    # ========================================================================

    def _can_mark_done(self, work_item_id: int) -> Tuple[bool, str]:
        """Check if work item can be marked as Done (no open child bugs)."""
        try:
            # Get child work items
            if hasattr(self.adapter, 'get_child_work_items'):
                children = self.adapter.get_child_work_items(work_item_id)
            else:
                # Fallback: query by parent link
                children = []

            if not children:
                return True, "No child items"

            # Check for open children
            open_children = [
                c for c in children
                if c.get("fields", {}).get("System.State", c.get("state", "")) not in self.COMPLETED_STATES
            ]

            if open_children:
                bug_count = sum(
                    1 for c in open_children
                    if c.get("fields", {}).get("System.WorkItemType", c.get("type", "")) == "Bug"
                )
                return False, f"{len(open_children)} open child item(s) ({bug_count} bug(s))"

            return True, "All child items completed"

        except Exception as e:
            # On error, allow completion but log warning
            print(f" ⚠ Could not check child items: {e}")
            return True, "Could not verify child items"

    # ========================================================================
    # Auto-Commit
    # ========================================================================

    def _auto_commit_if_eligible(self, work_item_id: int, result: Dict[str, Any], title: str) -> bool:
        """Auto-commit if task completed successfully (local only, no push)."""
        # Check eligibility
        if not result.get("success"):
            return False

        if result.get("issues"):
            print(f" ⏸ Skipping auto-commit: issues found")
            return False

        # Check for child bugs
        can_commit, reason = self._can_mark_done(work_item_id)
        if not can_commit:
            print(f" ⏸ Skipping auto-commit: {reason}")
            return False

        try:
            # Stage all changes
            stage_result = subprocess.run(
                ["git", "add", "-A"],
                capture_output=True,
                text=True
            )

            if stage_result.returncode != 0:
                print(f" ⚠ Git stage failed: {stage_result.stderr}")
                return False

            # Check if there are changes to commit
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True
            )

            if not status_result.stdout.strip():
                print(f" ℹ No changes to commit")
                return False

            # Create commit message
            commit_msg = f"""{title}

Completes #{work_item_id}

AI Generated with Claude Code

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"""

            # Commit (local only)
            commit_result = subprocess.run(
                ["git", "commit", "-m", commit_msg],
                capture_output=True,
                text=True
            )

            if commit_result.returncode == 0:
                print(f" Auto-committed: {title[:50]}...")
                return True
            else:
                print(f" ⚠ Commit failed: {commit_result.stderr}")
                return False

        except Exception as e:
            print(f" ⚠ Auto-commit error: {e}")
            return False

    # ========================================================================
    # Duplicate Detection
    # ========================================================================

    def _check_for_duplicate_bug(self, title: str, description: str) -> Optional[int]:
        """Check if a similar bug already exists in the sprint."""
        try:
            # Query existing bugs in the sprint
            items = self.adapter.query_sprint_work_items(self.sprint_name)
            bugs = [
                item for item in items
                if item.get("fields", {}).get("System.WorkItemType") == "Bug"
            ]

            # Check similarity
            for bug in bugs:
                bug_title = bug.get("fields", {}).get("System.Title", "")
                similarity = SequenceMatcher(None, title.lower(), bug_title.lower()).ratio()

                if similarity > 0.95:
                    return bug.get("id")

            return None

        except Exception:
            return None

    def _create_conformant_bug(self, parent_id: int, issue: Dict[str, Any]) -> Optional[int]:
        """
        Create a bug work item from structured issue data provided by agent.

        The issue dict contains fields populated by the agent's JSON output:
        - title: Short descriptive title
        - description: Detailed description
        - severity: high/medium/low
        - affected_files: List of file paths
        - repro_steps: Steps to reproduce
        - expected_behavior: What should happen
        - actual_behavior: What actually happens
        - suggested_fix: How to fix it
        - test_evidence: Test that found this
        - parent_work_item_id: Parent work item (from agent)
        """
        if self.dry_run:
            print(f" [DRY RUN] Would create bug: {issue.get('title', issue.get('description', ''))[:50]}")
            return None

        # Use agent-provided title, or construct from description
        title = issue.get('title', '')
        if not title:
            title = issue.get('description', 'Issue')[:80]

        # Ensure title has "Bug:" prefix
        if not title.lower().startswith('bug:'):
            title = f"Bug: {title}"

        # Check for duplicates
        duplicate_id = self._check_for_duplicate_bug(title, issue.get("description", ""))
        if duplicate_id:
            print(f" ℹ Similar bug already exists: #{duplicate_id}")
            return None

        try:
            # Use parent_id from issue if available (agent knows the context)
            actual_parent_id = issue.get('parent_work_item_id') or parent_id

            # Extract structured fields from agent
            description_text = issue.get('description', 'No description provided')
            severity = issue.get('severity', 'Medium')
            files = issue.get('files', [])
            repro_steps = issue.get('repro_steps', '')
            expected_behavior = issue.get('expected_behavior', '')
            actual_behavior = issue.get('actual_behavior', '')
            suggested_fix = issue.get('suggested_fix', '')
            test_evidence = issue.get('test_evidence', '')
            issue_type = issue.get('issue_type', 'bug')

            # Build files section
            files_section = chr(10).join('- `' + f + '`' for f in files) if files else "- (No specific files identified)"

            # Build acceptance criteria from agent-provided data
            acceptance_items = []
            if actual_behavior:
                acceptance_items.append(f"- Bug behavior no longer reproduces: {actual_behavior[:100]}")
            else:
                acceptance_items.append(f"- Bug behavior no longer reproduces: {description_text[:100]}")

            if expected_behavior:
                acceptance_items.append(f"- System behaves correctly: {expected_behavior[:100]}")

            acceptance_items.append("- Root cause identified and documented")
            acceptance_items.append("- Fix implemented and code reviewed")

            if files:
                acceptance_items.append(f"- Changes verified in: {', '.join(files[:3])}")

            acceptance_items.append("- Regression test added to prevent recurrence")
            acceptance_items.append("- No new issues introduced by the fix")

            acceptance_criteria = '\n'.join(acceptance_items)

            # Build test plan
            test_plan_parts = [f"# Test Plan: {title[:60]}"]

            if test_evidence:
                test_plan_parts.append(f"\n## Discovery\nFound by: {test_evidence}")

            if repro_steps:
                test_plan_parts.append(f"\n## Reproduction Steps\n{repro_steps}")
            else:
                test_plan_parts.append("\n## Reproduction Steps\n1. Navigate to the affected component\n2. Perform the action that triggers the bug\n3. Observe the incorrect behavior")

            if expected_behavior and actual_behavior:
                test_plan_parts.append(f"\n## Expected vs Actual\n- **Expected**: {expected_behavior}\n- **Actual**: {actual_behavior}")

            if suggested_fix:
                test_plan_parts.append(f"\n## Suggested Fix\n{suggested_fix}")

            test_plan_parts.append("\n## Verification Checklist")
            test_plan_parts.append(f"- [ ] Verify fix for: {description_text[:80]}")
            if files:
                for f in files[:3]:
                    test_plan_parts.append(f"- [ ] Test changes in `{f}`")
            test_plan_parts.append("- [ ] Run existing unit tests - all must pass")
            test_plan_parts.append("- [ ] No regression in related functionality")

            test_plan = '\n'.join(test_plan_parts)

            # Build Repro Steps field (main content field for Bugs in Azure DevOps)
            repro_steps_html = f"""<div>
<h2>Issue Summary</h2>
<p>{self._escape_html(description_text)}</p>

<h2>Found During</h2>
<p>Testing of Work Item #{actual_parent_id}</p>
<p><strong>Type</strong>: {issue_type}</p>

<h2>Affected Files</h2>
<ul>
{''.join('<li><code>' + self._escape_html(f) + '</code></li>' for f in files) if files else '<li>(No specific files identified)</li>'}
</ul>
"""

            if expected_behavior or actual_behavior:
                repro_steps_html += f"""
<h2>Expected vs Actual Behavior</h2>
<table>
<tr><td><strong>Expected</strong></td><td>{self._escape_html(expected_behavior) if expected_behavior else 'Not specified'}</td></tr>
<tr><td><strong>Actual</strong></td><td>{self._escape_html(actual_behavior) if actual_behavior else 'Not specified'}</td></tr>
</table>
"""

            if repro_steps:
                repro_steps_html += f"""
<h2>Steps to Reproduce</h2>
<ol>
{''.join('<li>' + self._escape_html(step.strip().lstrip('0123456789.-) ')) + '</li>' for step in repro_steps.split(chr(10)) if step.strip())}
</ol>
"""
            else:
                repro_steps_html += """
<h2>Steps to Reproduce</h2>
<ol>
<li>Navigate to the affected component</li>
<li>Perform the action that triggers the bug</li>
<li>Observe the incorrect behavior</li>
</ol>
"""

            if suggested_fix:
                repro_steps_html += f"""
<h2>Suggested Fix</h2>
<p>{self._escape_html(suggested_fix)}</p>
"""

            repro_steps_html += "</div>"

            # Build System Info field
            system_info_html = f"""<div>
<p><strong>Auto-generated by sprint execution workflow</strong></p>
<p>Parent Work Item: #{actual_parent_id}</p>
<p>Issue Type: {issue_type}</p>
<p>Severity: {severity}</p>
"""
            if test_evidence:
                system_info_html += f"<p>Discovered by: {self._escape_html(test_evidence)}</p>"
            system_info_html += "</div>"

            # Map severity to priority
            priority_map = {"High": 1, "Medium": 2, "Low": 3}
            priority = priority_map.get(severity, 2)

            # Create bug work item
            result = self.adapter.create_work_item(
                work_item_type="Bug",
                title=title,
                description="", # Bugs don't display this field in Azure DevOps
                assigned_to=self.current_user,
                fields={
                    "Microsoft.VSTS.Common.Priority": priority,
                    "Microsoft.VSTS.TCM.ReproSteps": repro_steps_html,
                    "Microsoft.VSTS.TCM.SystemInfo": system_info_html,
                    "Microsoft.VSTS.Common.AcceptanceCriteria": acceptance_criteria,
                    "System.Tags": f"auto-created; from-#{actual_parent_id}; {issue_type}; groomed"
                }
            )

            bug_id = result.get("id")
            print(f" Created conformant bug #{bug_id}: {title[:50]}...")

            # Attach test plan
            self._attach_report(bug_id, test_plan, "test-plan")

            # Link bug properly in hierarchy:
            # - Bug should be RELATED to the Task (traceability)
            # - Bug should be a CHILD of the Task's parent Feature (proper hierarchy)
            # Using parent-child link to Task breaks the taskboard (same category error)
            if hasattr(self.adapter, 'link_work_items'):
                # Link as Related to the source Task (traceability)
                try:
                    self.adapter.link_work_items(
                        bug_id, actual_parent_id, "System.LinkTypes.Related"
                    )
                    print(f" Linked bug #{bug_id} as Related to #{actual_parent_id}")
                except Exception as e:
                    print(f" ⚠ Could not link bug to task: {e}")

                # Find the Task's parent (Feature) and link Bug as its child
                try:
                    task_item = self.adapter.get_work_item(actual_parent_id)
                    task_parent_id = task_item.get("fields", {}).get("System.Parent")
                    if task_parent_id:
                        self.adapter.link_work_items(
                            bug_id, int(task_parent_id), "System.LinkTypes.Hierarchy-Reverse"
                        )
                        print(f" Linked bug #{bug_id} as child of Feature #{task_parent_id}")
                except Exception as e:
                    print(f" ⚠ Could not link bug to parent feature: {e}")

            # Assign to sprint
            self._assign_to_sprint(bug_id)

            return bug_id

        except Exception as e:
            print(f" Failed to create bug: {e}")
            return None

    def _assign_to_sprint(self, work_item_id: int) -> None:
        """Assign work item to the current sprint."""
        try:
            iteration_path = self.adapter.get_iteration_path(self.sprint_name)
            self.adapter.update_work_item(
                work_item_id=work_item_id,
                fields={"System.IterationPath": iteration_path}
            )
        except Exception as e:
            print(f" ⚠ Failed to assign to sprint: {e}")

    # ========================================================================
    # Divergence Detection
    # ========================================================================

    def _check_divergence(self, work_item_id: int, issues: List[Dict]) -> bool:
        """Check if issues are diverging (not converging to zero)."""
        issue_count = len(issues)

        # Initialize history
        if work_item_id not in self.issue_history:
            self.issue_history[work_item_id] = []

        history = self.issue_history[work_item_id]
        history.append(issue_count)

        # Increment retry count
        retry_count = self.retry_counts.get(work_item_id, 0)

        # Warn if:
        # - 2+ retries
        # - Issue count is increasing (not converging)
        if retry_count >= 2:
            return True

        if len(history) >= 2 and history[-1] >= history[-2]:
            # Issues not decreasing
            return True

        return False

    # ========================================================================
    # Step 4: Summary
    # ========================================================================

    def _step_4_summary(self) -> Dict[str, Any]:
        """Generate execution summary."""
        execution = self.step_evidence.get("3-execute-items", {})

        executed = execution.get("executed", [])
        skipped = execution.get("skipped", [])
        failed = execution.get("failed", [])

        print("\n" + "=" * 70)
        print(" SPRINT EXECUTION SUMMARY")
        print("=" * 70)

        print(f"\n Executed: {len(executed)}")
        for item in executed[:5]:
            print(f" #{item['id']}: {item['title'][:40]}")
        if len(executed) > 5:
            print(f" ... and {len(executed) - 5} more")

        if skipped:
            print(f"\n ⏭ Skipped: {len(skipped)}")
            for item in skipped[:3]:
                print(f" #{item['id']}: {item.get('reason', 'skipped')}")

        if failed:
            print(f"\n Failed: {len(failed)}")
            for item in failed[:3]:
                print(f" #{item['id']}: {item.get('error', 'unknown error')[:40]}")

        success_rate = len(executed) / max(len(executed) + len(failed), 1) * 100
        print(f"\n✅ Success Rate: {success_rate:.1f}%")

        # Print token usage summary
        totals = self.token_usage["totals"]
        if totals["input_tokens"] > 0 or totals["output_tokens"] > 0:
            print(f"\n" + "=" * 70)
            print(" 📊 TOKEN USAGE SUMMARY")
            print("=" * 70)

            print(f"\n Total Tokens:")
            print(f"   Input:  {totals['input_tokens']:,}")
            print(f"   Output: {totals['output_tokens']:,}")
            print(f"   Cache Read: {totals['cache_read_tokens']:,}")
            print(f"   Cache Created: {totals['cache_creation_tokens']:,}")
            print(f"\n💰 Total Cost: ${totals['total_cost_usd']:.4f}")

            # By agent breakdown
            if self.token_usage["by_agent"]:
                print(f"\n By Agent:")
                for agent_type, stats in self.token_usage["by_agent"].items():
                    print(f"   {agent_type.title()}: {stats['calls']} calls, "
                          f"{stats['input_tokens']:,} in / {stats['output_tokens']:,} out, "
                          f"${stats['cost_usd']:.4f}")

            # Top 3 most expensive work items
            if self.token_usage["by_work_item"]:
                sorted_items = sorted(
                    self.token_usage["by_work_item"].items(),
                    key=lambda x: x[1]["cost_usd"],
                    reverse=True
                )[:3]
                if sorted_items:
                    print(f"\n🔥 Top 3 Most Expensive Work Items:")
                    for wi_id, stats in sorted_items:
                        print(f"   #{wi_id}: {stats['input_tokens']:,} in / {stats['output_tokens']:,} out, "
                              f"${stats['cost_usd']:.4f}")

        return {
            "executed_count": len(executed),
            "skipped_count": len(skipped),
            "failed_count": len(failed),
            "success_rate": success_rate,
            "token_usage": self.token_usage
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sprint Execution Workflow - Execute sprint work items with specialized agents"
    )
    parser.add_argument(
        "--sprint",
        required=True,
        help="Sprint name (e.g., 'Sprint 8')"
    )
    parser.add_argument(
        "--work-items",
        nargs='+',
        type=int,
        help="Specific work item IDs to execute (optional)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without making changes"
    )
    parser.add_argument(
        "--confirm-start",
        action="store_true",
        help="Prompt for confirmation before starting execution"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts per work item (default: 3)"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip tester agent validation (for debugging)"
    )
    parser.add_argument(
        "--use-consensus",
        action="store_true",
        help="Use multi-agent consensus for implementation (Feature #1344)"
    )
    parser.add_argument(
        "--max-iteration-cycles",
        type=int,
        default=10,
        help="Maximum number of execution cycles before stopping (default: 10, 0 for unlimited)"
    )
    parser.add_argument(
        "--use-sdk",
        action="store_true",
        default=True,
        help="Use Claude Agent SDK for agent invocations (default: True, enables session continuity)"
    )
    parser.add_argument(
        "--use-cli",
        action="store_true",
        help="Use CLI subprocess for agent invocations (legacy, no session continuity)"
    )
    parser.add_argument(
        "--max-fix-iterations",
        type=int,
        default=3,
        help="Maximum ping-pong iterations between engineer and tester before creating bugs (default: 3)"
    )

    args = parser.parse_args()

    # Determine SDK mode: --use-cli overrides --use-sdk
    use_sdk = not args.use_cli

    # Normalize sprint name (e.g., "9" → "Sprint 9")
    sprint_name = normalize_sprint_name(args.sprint)
    if sprint_name != args.sprint:
        print(f"ℹ Normalized sprint name: '{args.sprint}' → '{sprint_name}'")

    from cli.console import console

    # Print header with trustable-ai persona
    console.print()
    console.print("─" * 80)
    console.print("[bold #71E4D1]  SPRINT EXECUTION[/bold #71E4D1]")
    console.print("─" * 80)
    console.print()
    console.print(f"[#D9EAFC]Sprint:[/#D9EAFC] [bold]{sprint_name}[/bold]")
    console.print(f"[#D9EAFC]Mode:[/#D9EAFC] {'Dry Run (no changes)' if args.dry_run else 'Automatic Execution'}")
    console.print(f"[#D9EAFC]Agent Backend:[/#D9EAFC] {'Claude Agent SDK' if use_sdk else 'CLI subprocess (legacy)'}")
    if args.work_items:
        console.print(f"[#D9EAFC]Work Items:[/#D9EAFC] {', '.join(map(str, args.work_items))}")
    if args.confirm_start:
        console.print(f"[#D9EAFC]Confirmation:[/#D9EAFC] Will prompt before starting")
    max_cycles = args.max_iteration_cycles
    if max_cycles == 0:
        console.print(f"[#D9EAFC]Cycles:[/#D9EAFC] Unlimited (will loop until all work is done)")
    else:
        console.print(f"[#D9EAFC]Max Cycles:[/#D9EAFC] {max_cycles}")
    if use_sdk:
        console.print(f"[#D9EAFC]Fix Iterations:[/#D9EAFC] {args.max_fix_iterations} (engineer-tester ping-pong)")
    console.print()

    # Loop until done (default behavior)
    # Each cycle re-queries the sprint to find any remaining or new work items
    cycle = 0
    total_executed = 0
    total_failed = 0
    total_skipped = 0

    try:
        while True:
            cycle += 1

            # Check max cycles limit (0 = unlimited)
            if max_cycles > 0 and cycle > max_cycles:
                console.print()
                console.print("─" * 80)
                console.print(f"[bold #FFA500]  Reached maximum cycles ({max_cycles})[/bold #FFA500]")
                console.print("─" * 80)
                console.print()
                console.print(f"[#D9EAFC]Completed {max_cycles} cycle(s) as configured.[/#D9EAFC]")
                console.print(f"[#758B9B]To continue, run the script again or increase --max-iteration-cycles.[/#758B9B]")
                break

            if cycle > 1:
                console.print()
                console.print("─" * 80)
                console.print(f"[bold #67CFEE]  Starting Cycle {cycle}[/bold #67CFEE]")
                console.print("─" * 80)
                console.print()
                console.print("[#D9EAFC]Checking for remaining or newly created work items...[/#D9EAFC]")

            # Initialize and execute workflow for this cycle
            workflow = SprintExecutionWorkflow(
                sprint_name=sprint_name,
                work_item_ids=args.work_items,
                dry_run=args.dry_run,
                confirm_start=args.confirm_start if cycle == 1 else False, # Only confirm on first cycle
                max_retries=args.max_retries,
                skip_tests=args.skip_tests,
                use_sdk=use_sdk,  # SDK mode for session continuity
                max_fix_iterations=args.max_fix_iterations,  # Ping-pong iterations
                use_consensus=args.use_consensus,  # Multi-agent consensus (Feature #1344)
                args=args
            )

            # Execute this cycle
            success = workflow.execute()

            # Collect stats from this cycle
            execution_evidence = workflow.step_evidence.get("3-execute-items", {})
            cycle_executed = len(execution_evidence.get("executed", []))
            cycle_failed = len(execution_evidence.get("failed", []))
            cycle_skipped = len(execution_evidence.get("skipped", []))

            total_executed += cycle_executed
            total_failed += cycle_failed
            total_skipped += cycle_skipped

            # Check if any work was done this cycle
            if cycle_executed == 0 and cycle_failed == 0:
                # No work items found or all already complete
                console.print()
                console.print("─" * 80)
                console.print("[bold #71E4D1]  All work complete![/bold #71E4D1]")
                console.print("─" * 80)
                console.print()
                console.print(f"[#D9EAFC]No more incomplete work items found in {sprint_name}.[/#D9EAFC]")
                console.print()
                console.print(f"[bold]Final Totals Across {cycle} Cycle(s):[/bold]")
                console.print(f"  [#71E4D1]Executed:[/#71E4D1] {total_executed}")
                console.print(f"  [#758B9B]Skipped:[/#758B9B] {total_skipped}")
                console.print(f"  [#FF6B6B]Failed:[/#FF6B6B] {total_failed}")
                break

            # If we only had failures and no successful executions, don't loop forever
            if cycle_executed == 0 and cycle_failed > 0:
                console.print()
                console.print("─" * 80)
                console.print(f"[bold #FFA500]  Cycle {cycle} had only failures[/bold #FFA500]")
                console.print("─" * 80)
                console.print()
                console.print("[#D9EAFC]No items were successfully executed this cycle.[/#D9EAFC]")
                console.print(f"[#FF6B6B]{cycle_failed} item(s) failed. Review errors above.[/#FF6B6B]")
                console.print()
                console.print(f"[bold]Final Totals Across {cycle} Cycle(s):[/bold]")
                console.print(f"  [#71E4D1]Executed:[/#71E4D1] {total_executed}")
                console.print(f"  [#758B9B]Skipped:[/#758B9B] {total_skipped}")
                console.print(f"  [#FF6B6B]Failed:[/#FF6B6B] {total_failed}")
                break

            # Continue to next cycle to pick up any newly created bugs or remaining work
            console.print()
            console.print(f"[#71E4D1]Cycle {cycle} complete:[/#71E4D1] {cycle_executed} executed, {cycle_failed} failed")

        # Final exit
        sys.exit(0 if total_failed == 0 else 1)

    except KeyboardInterrupt:
        console.print()
        console.print("[#758B9B]Execution interrupted by user.[/#758B9B]")
        console.print()
        console.print(f"[bold]Totals Before Interruption ({cycle} cycle(s)):[/bold]")
        console.print(f"  [#71E4D1]Executed:[/#71E4D1] {total_executed}")
        console.print(f"  [#758B9B]Skipped:[/#758B9B] {total_skipped}")
        console.print(f"  [#FF6B6B]Failed:[/#FF6B6B] {total_failed}")
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
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
