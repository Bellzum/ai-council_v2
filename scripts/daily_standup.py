#!/usr/bin/env python3
"""
Daily Standup Report Workflow with External Enforcement

Lightweight workflow for generating daily standup reports showing:
- Yesterday's completed work (last 24 hours)
- Today's active work items
- Blockers and stale items
- Sprint progress metrics
- Work item state verification (external source of truth)

Usage:
 # Basic daily standup (Mode 1: Pure Python)
    python3 scripts/daily_standup.py --sprint "Sprint 8"

 # With AI-generated summary (Mode 2: AI + JSON)
    python3 scripts/daily_standup.py --sprint "Sprint 8" --use-ai

 # Custom time window
    python3 scripts/daily_standup.py --sprint "Sprint 8" --hours 48

 # Save report to file
    python3 scripts/daily_standup.py --sprint "Sprint 8" --output-file standup-report.md

 # With debug trace output
    python3 scripts/daily_standup.py --sprint "Sprint 8" --debug-trace
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import workflow executor base
from scripts.workflow_executor.base import WorkflowOrchestrator, ExecutionMode

# Import work tracking adapter
sys.path.insert(0, '.claude/skills')
from work_tracking import get_adapter

# Import workflow utilities
try:
    from workflows.utilities import (
    analyze_sprint,
    verify_work_item_states,
    get_recent_activity,
    identify_blockers,
    normalize_azure_timestamp
    )
except ImportError:
    from scripts.workflow_executor.progress import print_warning, print_status
    print_warning("Could not import workflow utilities")
    print_status("Ensure workflows/utilities.py is accessible")
    sys.exit(1)


class DailyStandupWorkflow(WorkflowOrchestrator):
    """
    Daily Standup workflow with external enforcement.

    Generates daily standup reports with:
    - Recent activity (configurable time window)
    - Active work items
    - Blockers and stale items
    - Sprint progress metrics
    - External verification of work item states
    """

    def __init__(
        self,
        sprint_name: str,
        hours: int = 24,
        use_ai: bool = False,
        output_file: Optional[str] = None,
        args: Optional[argparse.Namespace] = None,
        verbose: bool = False,
        debug_trace: bool = False
    ):
        """
        Initialize daily standup workflow.

        Args:
            sprint_name: Sprint name (e.g., "Sprint 8", "sprint 8", or "8")
            hours: Time window for recent activity (default: 24 hours)
            use_ai: If True, use AI for summary generation (Mode 2), otherwise mock (Mode 1)
            output_file: Optional output file path for report
            args: Command-line arguments for targeted execution
            verbose: If True, show detailed step-by-step output (default: clean summary)
            debug_trace: If True, enable debug trace output with timestamps
        """
        # Normalize sprint name to standard format (e.g., "9" → "Sprint 9")
        self.sprint_name = normalize_sprint_name(sprint_name)
        self.hours = hours
        self.use_ai = use_ai
        self.output_file = output_file
        self.args = args
        self.verbose = verbose
        self.debug_trace = debug_trace

        # Configure debug logging if enabled
        # Use unique logger name per workflow instance to avoid shared state
        workflow_id_for_logger = f"{sprint_name.replace(' ', '-')}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
        self.logger = logging.getLogger(f"{__name__}.{workflow_id_for_logger}")
        # Prevent propagation to parent loggers to ensure isolation
        self.logger.propagate = False

        if debug_trace:
            # Set logger level to DEBUG
            self.logger.setLevel(logging.DEBUG)
            # Add console handler (always add since this is a unique logger)
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('[DEBUG] %(asctime)s - %(message)s', '%Y-%m-%d %H:%M:%S')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        else:
            self.logger.setLevel(logging.WARNING)

        # Initialize work tracking adapter
        from scripts.workflow_executor.progress import print_warning, print_status
        try:
            self.adapter = get_adapter()
        except Exception as e:
            print_warning(f"Could not initialize work tracking adapter: {e}")
            print_status("Continuing with limited functionality...")
            self.adapter = None

        # Load configuration for model selection
        self.config = None
        try:
            from config.loader import load_config
            self.config = load_config()
        except Exception as e:
            if verbose:
                print_warning(f"Could not load configuration: {e}")
                print_status("Using default model settings...")

        # Initialize Claude API client if using AI
        self.claude_client = None
        if use_ai:
            try:
                import anthropic
                api_key = os.getenv("KEYCHAIN_ANTHROPIC_API_KEY")
                if api_key:
                    self.claude_client = anthropic.Anthropic(api_key=api_key)
                    if verbose:
                        from scripts.workflow_executor.progress import print_success
                        print_success("Claude API client initialized")
                else:
                    print_warning("KEYCHAIN_ANTHROPIC_API_KEY not set, using simple logic (no AI)")
                    self.use_ai = False
            except ImportError:
                print_warning("anthropic package not installed, using simple logic (no AI)")
                self.use_ai = False

        # Determine execution mode
        mode = ExecutionMode.AI_JSON_VALIDATION if use_ai else ExecutionMode.PURE_PYTHON

        # Include timestamp in workflow ID to ensure fresh data on each run
        # (prevents caching external state like work item queries across runs)
        workflow_id = f"{sprint_name.replace(' ', '-')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        super().__init__(
            workflow_name="daily-standup",
            workflow_id=workflow_id,
            mode=mode,
            enable_checkpoints=True,
            quiet_mode=not verbose
        )

    def _define_steps(self) -> List[Dict[str, Any]]:
        """Define workflow steps."""
        steps = [
            {"id": "1-gather-recent", "name": "Gather Yesterday's Activity"},
            {"id": "2-verify-states", "name": "Verify Work Item States"},
            {"id": "3-identify-focus", "name": "Identify Today's Focus"},
            {"id": "4-detect-blockers", "name": "Detect Blockers"},
            {"id": "5-analyze-progress", "name": "Analyze Sprint Progress"},
            {"id": "6-sprint-status", "name": "Generate Sprint Status Hierarchy"},
        ]

        # Add AI recommendations step if using AI
        if self.use_ai:
            steps.append({"id": "7-ai-recommendations", "name": "Generate AI Recommendations"})

        steps.extend([
            {"id": "8-generate-report", "name": "Generate Standup Report"},
            {"id": "9-save-report", "name": "Save Report"},
        ])

        return steps

    def _execute_step(
        self,
        step: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single workflow step."""
        step_id = step["id"]
        step_name = step.get("name", step_id)

        # Debug trace: step entry
        if self.debug_trace:
            self.logger.debug(f">>> ENTER Step {step_id}: {step_name}")

        # Execute step
        if step_id == "1-gather-recent":
            result = self._step_1_gather_recent()
        elif step_id == "2-verify-states":
            result = self._step_2_verify_states()
        elif step_id == "3-identify-focus":
            result = self._step_3_identify_focus()
        elif step_id == "4-detect-blockers":
            result = self._step_4_detect_blockers()
        elif step_id == "5-analyze-progress":
            result = self._step_5_analyze_progress()
        elif step_id == "6-sprint-status":
            result = self._step_6_sprint_status()
        elif step_id == "7-ai-recommendations":
            result = self._step_7_ai_recommendations()
        elif step_id == "8-generate-report":
            result = self._step_8_generate_report()
        elif step_id == "9-save-report":
            result = self._step_9_save_report()
        else:
            raise ValueError(f"Unknown step: {step_id}")

        # Debug trace: step exit
        if self.debug_trace:
            self.logger.debug(f"<<< EXIT Step {step_id}: {step_name}")

        return result

    def _step_1_gather_recent(self) -> Dict[str, Any]:
        """Step 1: Gather yesterday's activity (Mode 1)."""
        from scripts.workflow_executor.progress import Spinner, print_success, print_warning, print_error

        if not self.adapter:
            return {
                "error": "Adapter not available",
                "recent_items": [],
                "recent_count": 0
            }

        try:
            if self.debug_trace:
                self.logger.debug(f"Calling get_recent_activity(sprint={self.sprint_name}, hours={self.hours})")

            with Spinner(f"Gathering activity from last {self.hours} hours"):
                activity_result = get_recent_activity(
                    self.adapter,
                    self.sprint_name,
                    hours=self.hours
                )

            if self.debug_trace:
                recent_count = activity_result.get('recent_count', 0)
                recent_items = activity_result.get('recent_items', [])
                item_ids = [item.get('id') for item in recent_items]
                self.logger.debug(f"Found {recent_count} recent items: IDs={item_ids}")

            if self.verbose:
                print_success(f"Found {activity_result['recent_count']} items updated in last {self.hours} hours")

            if activity_result.get('errors'):
                print_warning("Warnings during activity query:")
                from cli.console import console
                for error in activity_result['errors']:
                    console.print(f"  - {error}", style="warning")

            return activity_result

        except Exception as e:
            print_error(f"Error gathering recent activity: {e}")
            return {
                "error": str(e),
                "recent_items": [],
                "recent_count": 0,
                "total_items": 0,
                "errors": [str(e)]
            }

    def _step_2_verify_states(self) -> Dict[str, Any]:
        """Step 2: Verify work item states against external source of truth (Mode 1)."""
        from scripts.workflow_executor.progress import Spinner, print_success, print_warning, print_error, print_info

        recent_activity = self.step_evidence.get("1-gather-recent", {})
        recent_items = recent_activity.get("recent_items", [])

        if self.debug_trace:
            self.logger.debug(f"Verifying {len(recent_items)} recent items")

        if not recent_items:
            if self.verbose:
                print_info("No recent items to verify")
            return {
                "verified_count": 0,
                "divergence_count": 0,
                "divergences": []
            }

        if not self.adapter:
            return {
                "error": "Adapter not available",
                "verified_count": 0,
                "divergence_count": 0
            }

        try:
            if self.debug_trace:
                item_ids = [item.get('id') for item in recent_items]
                self.logger.debug(f"Calling verify_work_item_states(item_ids={item_ids})")

            with Spinner("Verifying work item states against external source of truth"):
                verification_result = verify_work_item_states(self.adapter, recent_items)

            if self.debug_trace:
                verified_count = verification_result.get('verified_count', 0)
                divergence_count = verification_result.get('divergence_count', 0)
                self.logger.debug(f"Verification complete: {verified_count} verified, {divergence_count} divergences")

            if self.verbose:
                print_success(f"Verified {verification_result['verified_count']} work items")

            # Always show divergence warnings (important)
            if verification_result['divergence_count'] > 0:
                from cli.console import console
                print_warning(f"DIVERGENCE DETECTED: {verification_result['divergence_count']} work item(s) need attention")
                if self.verbose:
                    console.print("=" * 80, style="dim")

                for div in verification_result['divergences']:
                    if div['severity'] == 'ERROR':
                        print_error(f"{div['id']}: {div['title']}")
                        console.print(f"  {div['message']}", style="error")
                    else:
                        print_warning(f"{div['id']}: {div['title']}")
                        console.print(f"  CLAIMED: {div['claimed_state']} | ACTUAL: {div['actual_state']}", style="warning")

                console.print("\nDivergence Summary:", style="bold_warning")
                console.print(f"  - {verification_result['summary']['errors']} ERROR(s)", style="error")
                console.print(f"  - {verification_result['summary']['warnings']} WARNING(s)", style="warning")
                console.print(f"  - Action: Review work items and sync states in {self.adapter.platform}", style="warning")
                console.print("=" * 80, style="dim")
            elif self.verbose:
                print_success("No divergence detected - all work item states match external source of truth")

            return verification_result

        except Exception as e:
            print_error(f"Error during verification: {e}")
            return {
                "error": str(e),
                "verified_count": 0,
                "divergence_count": 0,
                "divergences": [],
                "errors": [str(e)]
            }

    def _step_3_identify_focus(self) -> Dict[str, Any]:
        """Step 3: Identify today's focus (active work items) (Mode 1)."""
        from scripts.workflow_executor.progress import print_status, print_success, print_error
        from cli.console import console

        if self.verbose:
            print_status("Identifying today's focus (active work items)")

        if not self.adapter:
            return {"error": "Adapter not available", "active_items": []}

        try:
            if self.debug_trace:
                self.logger.debug(f"Calling adapter.query_sprint_work_items(sprint={self.sprint_name})")

            # Get all sprint work items
            all_items = self.adapter.query_sprint_work_items(self.sprint_name)

            if self.debug_trace:
                self.logger.debug(f"Retrieved {len(all_items)} total work items from sprint")

            # Filter to active states
            # - Task: "In Progress"
            # - Feature/Epic: "In Progress"
            # - Bug: "Committed" or "Approved" (Scrum template)
            active_states = ['In Progress', 'Committed', 'Approved', 'Active', 'Doing']
            active_items = [
                item for item in all_items
                if (item.get('state') or item.get('fields', {}).get('System.State')) in active_states
            ]

            if self.debug_trace:
                active_ids = [item.get('id') for item in active_items]
                self.logger.debug(f"Filtered to {len(active_items)} active items: IDs={active_ids}")

            if self.verbose:
                print_success(f"Found {len(active_items)} active work items")

            # Group by assignee
            by_assignee = {}
            for item in active_items:
                # Extract assigned_to from Azure DevOps fields (raw format: fields.System.AssignedTo.displayName)
                assignee = item.get('assigned_to')
                if not assignee:
                    assigned_to_obj = item.get('fields', {}).get('System.AssignedTo')
                    if assigned_to_obj and isinstance(assigned_to_obj, dict):
                        assignee = assigned_to_obj.get('displayName', 'Unassigned')
                    else:
                        assignee = 'Unassigned'
                if assignee not in by_assignee:
                    by_assignee[assignee] = []
                by_assignee[assignee].append(item)

            if self.verbose:
                console.print("\nActive work by assignee:", style="bold_primary")
                for assignee, items in by_assignee.items():
                    total_points = sum(
                        item.get('story_points', 0) or 0
                        for item in items
                    )
                    console.print(f"  {assignee}: {len(items)} items ({total_points} points)", style="primary")

            return {
                "active_items": active_items,
                "active_count": len(active_items),
                "by_assignee": {k: len(v) for k, v in by_assignee.items()}
            }

        except Exception as e:
            print_error(f"Error identifying active items: {e}")
            return {
                "error": str(e),
                "active_items": [],
                "active_count": 0,
                "errors": [str(e)]
            }

    def _step_4_detect_blockers(self) -> Dict[str, Any]:
        """Step 4: Detect blockers (Mode 1)."""
        if self.verbose:
            print("\n Detecting blockers...")

        if not self.adapter:
            return {"error": "Adapter not available", "total_blockers": 0}

        try:
            blocker_result = identify_blockers(
                self.adapter,
                self.sprint_name,
                stale_threshold_days=3
            )

            if self.verbose:
                print(f"\n Blocker Analysis:")
                print(f"Total blockers detected: {blocker_result['total_blockers']}")
                print(f"Blocked state: {len(blocker_result['blocked_items'])}")
                print(f"Tagged as blocker: {len(blocker_result['tagged_items'])}")
                print(f"Stale (no updates 3+ days): {len(blocker_result['stale_items'])}")

                print(f"\n Impact:")
                print(f"Affected people: {blocker_result['impact']['affected_people']}")
                print(f"Story points at risk: {blocker_result['impact']['story_points_at_risk']}")

            if blocker_result.get('errors'):
                print(f"\n Warnings during blocker detection:")
                for error in blocker_result['errors']:
                    print(f" - {error}")

            return blocker_result

        except Exception as e:
            print(f" Error detecting blockers: {e}")
            return {
                "error": str(e),
                "total_blockers": 0,
                "blocked_items": [],
                "tagged_items": [],
                "stale_items": [],
                "errors": [str(e)]
            }

    def _step_5_analyze_progress(self) -> Dict[str, Any]:
        """Step 5: Analyze sprint progress (Mode 1)."""
        if self.verbose:
            print("\n Analyzing sprint progress...")

        if not self.adapter:
            return {"error": "Adapter not available"}

        try:
            sprint_stats = analyze_sprint(self.adapter, self.sprint_name)

            if self.verbose:
                print(f"\nSprint Progress:")
                print(f"Total items: {sprint_stats['total_items']}")
                print(f"Completion rate: {sprint_stats['completion_rate']:.1f}%")
                print(f"Velocity (completed points): {sprint_stats['velocity']}")

                print(f"\nBy State:")
                for state, count in sprint_stats['by_state'].items():
                    print(f" {state}: {count}")

                print(f"\nStory Points:")
                print(f" Total: {sprint_stats['story_points']['total']}")
                print(f" Completed: {sprint_stats['story_points']['completed']}")
                print(f" In Progress: {sprint_stats['story_points']['in_progress']}")
                print(f" Not Started: {sprint_stats['story_points']['not_started']}")

            if sprint_stats.get('errors'):
                print(f"\n Warnings during sprint analysis:")
                for error in sprint_stats['errors']:
                    print(f" - {error}")

            return sprint_stats

        except Exception as e:
            print(f" Error analyzing sprint: {e}")
            return {
                "error": str(e),
                "total_items": 0,
                "completion_rate": 0,
                "errors": [str(e)]
            }

    def _step_6_sprint_status(self) -> Dict[str, Any]:
        """Step 6: Generate sprint status hierarchy (Mode 1)."""
        if self.verbose:
            print("\n Generating sprint status hierarchy...")

        try:
            # Call sprint_status.py script
            script_path = Path(__file__).parent / "sprint_status.py"
            # Include done items to show full sprint hierarchy
            cmd = [sys.executable, str(script_path), "--sprint", self.sprint_name, "--include-done"]

            # Run sprint_status.py and capture output
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                if self.verbose:
                    print(f" sprint_status.py returned non-zero exit code: {result.returncode}")
                    print(f" stderr: {result.stderr}")
                return {
                    "error": f"sprint_status.py failed: {result.stderr}",
                    "sprint_status_output": ""
                }

            sprint_status_output = result.stdout
            if self.verbose:
                print(f" Sprint status generated ({len(sprint_status_output.splitlines())} lines)")

            return {
                "sprint_status_output": sprint_status_output
            }

        except subprocess.TimeoutExpired:
            if self.verbose:
                print(" sprint_status.py timed out after 60 seconds")
            return {
                "error": "Timeout",
                "sprint_status_output": ""
            }
        except Exception as e:
            if self.verbose:
                print(f" Error running sprint_status.py: {e}")
            return {
                "error": str(e),
                "sprint_status_output": ""
            }

    def _step_7_ai_recommendations(self) -> Dict[str, Any]:
        """Step 7: Generate AI recommendations from senior-engineer agent (Mode 2)."""
        if self.verbose:
            print("\n Generating AI recommendations from senior-engineer agent...")

        # Gather context for AI
        recent_activity = self.step_evidence.get("1-gather-recent", {})
        blockers = self.step_evidence.get("4-detect-blockers", {})
        progress = self.step_evidence.get("5-analyze-progress", {})
        sprint_status = self.step_evidence.get("6-sprint-status", {})

        # Build context prompt - request concise output for terminal display
        context_prompt = f"""You are a senior engineer reviewing {self.sprint_name}.

First, read the project CLAUDE.md to understand the project context, then provide recommendations.

Sprint: {progress.get('total_items', 0)} items, {progress.get('completion_rate', 0):.1f}% complete, {progress.get('velocity', 0)} story points
Blockers: {blockers.get('total_blockers', 0)} total, {len(blockers.get('stale_items', []))} stale (3+ days)
Recent: {recent_activity.get('recent_count', 0)} items updated (last {self.hours}h)

Sprint Status:
{sprint_status.get('sprint_status_output', 'Not available')[:1500]}

Provide exactly 3 concise, actionable recommendations. Be brief - this displays in a terminal.

Format each as:
[number]. [ACTION] (Impact: HIGH/MEDIUM/LOW)
    [One sentence why this matters]

Example:
1. Add story points to all work items (Impact: HIGH)
    Enables velocity tracking and capacity planning for future sprints."""

        # Try Claude Agent SDK first (provides tool access to read codebase)
        try:
            from scripts.workflow_executor.interactive_session import InteractiveSession

            # Get model from config
            model = "claude-sonnet-4-20250514"  # Default
            if self.config and hasattr(self.config, 'agent_config'):
                model = self.config.agent_config.models.get("engineer", model)

            if self.verbose:
                print(f" Using Claude Agent SDK with model: {model}", flush=True)

            # Create interactive session with tool access enabled
            session = InteractiveSession(
                workflow_name="daily-standup",
                session_id=f"standup-{self.sprint_name.replace(' ', '-')}",
                model=model,
                max_tokens=500,
                enable_tools=True,
                allowed_tools=["Read", "Grep", "Glob"]  # Read-only access
            )

            # Load senior-engineer agent definition if available
            agent_definition = None
            agent_path = Path(".claude/agents/senior-engineer.md")
            if agent_path.exists():
                agent_definition = agent_path.read_text(encoding='utf-8')

            system_context = f"""You are a Senior Engineer providing sprint recommendations.

{f"**Agent Context:**{chr(10)}{agent_definition[:2000]}" if agent_definition else ""}

Read the project CLAUDE.md to understand the codebase context before providing recommendations."""

            # Use ask_with_tools for tool access (Read, Grep, Glob)
            result = session.ask_with_tools(
                prompt=f"{system_context}\n\n{context_prompt}",
                tools=["Read", "Grep", "Glob"]
            )

            recommendations = result.get("response", "")

            if self.verbose:
                print(f" Generated {len(recommendations.splitlines())} lines of recommendations", flush=True)
                token_usage = result.get("token_usage", {})
                if token_usage:
                    print(f" Tokens: input={token_usage.get('input_tokens', 0)}, output={token_usage.get('output_tokens', 0)}", flush=True)

            return {
                "recommendations": recommendations
            }

        except ImportError as e:
            # ALWAYS report import errors - these indicate missing dependencies
            from cli.console import console
            console.print()
            console.print("[bold #FF6B6B]AI Recommendations Failed[/bold #FF6B6B]")
            console.print(f"[#FF6B6B]Import error: {e}[/#FF6B6B]")
            console.print("[#758B9B]The Claude Agent SDK is not available. AI recommendations require tool access to read the codebase.[/#758B9B]")
            console.print("[#758B9B]Install with: pip install claude-code-sdk[/#758B9B]")
            console.print()
            return {
                "error": f"ImportError: {e}",
                "recommendations": "AI recommendations unavailable - Claude Agent SDK import failed (see error above)"
            }

        except Exception as e:
            # ALWAYS report errors - no silent failures
            from cli.console import console
            console.print()
            console.print("[bold #FF6B6B]AI Recommendations Failed[/bold #FF6B6B]")
            console.print(f"[#FF6B6B]{type(e).__name__}: {e}[/#FF6B6B]")
            console.print("[#758B9B]The workflow will continue without AI recommendations.[/#758B9B]")
            console.print()
            return {
                "error": f"{type(e).__name__}: {e}",
                "recommendations": f"AI recommendations unavailable - {type(e).__name__}: {e}"
            }

    def _step_8_generate_report(self) -> Dict[str, Any]:
        """Step 8: Generate standup report (Mode 1 or 2)."""
        if self.verbose:
            print("\n Generating standup report...")

        # Gather all evidence
        recent_activity = self.step_evidence.get("1-gather-recent", {})
        verification = self.step_evidence.get("2-verify-states", {})
        focus = self.step_evidence.get("3-identify-focus", {})
        blockers = self.step_evidence.get("4-detect-blockers", {})
        progress = self.step_evidence.get("5-analyze-progress", {})
        sprint_status = self.step_evidence.get("6-sprint-status", {})
        ai_recommendations = self.step_evidence.get("7-ai-recommendations", {})

        if self.use_ai:
            report_content = self._generate_report_with_ai(
                recent_activity, verification, focus, blockers, progress,
                sprint_status, ai_recommendations
            )
        else:
            report_content = self._generate_report_simple(
                recent_activity, verification, focus, blockers, progress,
                sprint_status, ai_recommendations
            )

        if self.verbose:
            print(" Report generated")

        return {"report_content": report_content}

    def _generate_report_simple(
        self,
        recent_activity: Dict[str, Any],
        verification: Dict[str, Any],
        focus: Dict[str, Any],
        blockers: Dict[str, Any],
        progress: Dict[str, Any],
        sprint_status: Dict[str, Any],
        ai_recommendations: Dict[str, Any]
    ) -> str:
        """Generate simple markdown report (Mode 1)."""
        report = f"""# Daily Standup Report

**Date**: {datetime.now().strftime('%Y-%m-%d')}
**Sprint**: {self.sprint_name}
**Time Window**: Last {self.hours} hours

---

## Yesterday's Accomplishments

**Items Updated**: {recent_activity.get('recent_count', 0)}

"""

        # List recent items
        recent_items = recent_activity.get('recent_items', [])
        if recent_items:
            for item in recent_items[:10]: # Limit to 10 items
                # Extract fields from Azure DevOps format
                item_type = item.get('type') or item.get('fields', {}).get('System.WorkItemType', 'Item')
                item_title = item.get('title') or item.get('fields', {}).get('System.Title', 'Untitled')
                item_state = item.get('state') or item.get('fields', {}).get('System.State', 'Unknown')
                report += f"- **{item_type} #{item.get('id')}**: {item_title}\n"
                report += f" - State: {item_state}\n"
        else:
            report += "_No items updated in the last {self.hours} hours_\n"

        report += "\n---\n\n## Work Item State Verification\n\n"

        # Divergence summary
        divergence_count = verification.get('divergence_count', 0)
        if divergence_count > 0:
            report += f" **DIVERGENCE DETECTED**: {divergence_count} work item(s) need attention\n\n"

            divergences = verification.get('divergences', [])
            errors = [d for d in divergences if d.get('severity') == 'ERROR']
            warnings = [d for d in divergences if d.get('severity') == 'WARNING']

            if errors:
                report += f"### Errors ({len(errors)})\n\n"
                for error in errors:
                    report += f"- **{error.get('id')}**: {error.get('title', 'Untitled')}\n"
                    report += f" - {error.get('message', 'Unknown error')}\n"
                report += "\n"

            if warnings:
                report += f"### Warnings ({len(warnings)})\n\n"
                for warning in warnings:
                    report += f"-  **{warning.get('id')}**: {warning.get('title', 'Untitled')}\n"
                    report += f" - Claimed: {warning.get('claimed_state', 'Unknown')} | Actual: {warning.get('actual_state', 'Unknown')}\n"
                report += "\n"

            report += f"**Action Required**: Review and sync work item states in {self.adapter.platform if self.adapter else 'work tracking system'}\n\n"
        else:
            report += " **All work item states verified** - no divergence detected\n\n"

        report += "---\n\n## Today's Focus\n\n"

        # Active items
        active_items = focus.get('active_items', [])
        active_count = focus.get('active_count', 0)

        report += f"**Active Items**: {active_count}\n\n"

        if active_items:
            # Group by assignee
            by_assignee = {}
            for item in active_items:
                assignee = item.get('assigned_to', 'Unassigned')
                if assignee not in by_assignee:
                    by_assignee[assignee] = []
                by_assignee[assignee].append(item)

            for assignee, items in by_assignee.items():
                report += f"### {assignee}\n\n"
                for item in items[:5]: # Limit to 5 per assignee
                    # Extract fields from Azure DevOps format
                    item_type = item.get('type') or item.get('fields', {}).get('System.WorkItemType', 'Item')
                    item_title = item.get('title') or item.get('fields', {}).get('System.Title', 'Untitled')
                    points = item.get('story_points') or item.get('fields', {}).get('Microsoft.VSTS.Scheduling.StoryPoints', 0) or 0
                    report += f"- **{item_type} #{item.get('id')}**: {item_title}\n"
                    if points:
                        report += f" - Story Points: {points}\n"
                report += "\n"
        else:
            report += "_No active work items_\n\n"

        report += "---\n\n## Blockers\n\n"

        # Blockers
        total_blockers = blockers.get('total_blockers', 0)
        blocked_items = blockers.get('blocked_items', [])
        stale_items = blockers.get('stale_items', [])

        report += f"**Total Blockers**: {total_blockers}\n"
        report += f"**Affected People**: {blockers.get('impact', {}).get('affected_people', 0)}\n"
        report += f"**Story Points at Risk**: {blockers.get('impact', {}).get('story_points_at_risk', 0)}\n\n"

        if blocked_items:
            report += "### Blocked Items\n\n"
            for item in blocked_items:
                # Extract fields from Azure DevOps format
                item_type = item.get('type') or item.get('fields', {}).get('System.WorkItemType', 'Item')
                item_title = item.get('title') or item.get('fields', {}).get('System.Title', 'Untitled')
                points = item.get('story_points') or item.get('fields', {}).get('Microsoft.VSTS.Scheduling.StoryPoints', 0) or 0
                report += f"- **{item_type} #{item.get('id')}**: {item_title}\n"
                if points:
                    report += f" - Story Points: {points}\n"
            report += "\n"

        if stale_items:
            report += "### Stale Items (3+ days no updates)\n\n"
            for item in stale_items:
                # Extract fields from Azure DevOps format
                item_type = item.get('type') or item.get('fields', {}).get('System.WorkItemType', 'Item')
                item_title = item.get('title') or item.get('fields', {}).get('System.Title', 'Untitled')
                report += f"- **{item_type} #{item.get('id')}**: {item_title}\n"
                report += f" - Days Stale: {item.get('days_stale', 0)}\n"
            report += "\n"

        if not blocked_items and not stale_items:
            report += "_No blockers detected_\n\n"

        report += "---\n\n## Sprint Progress\n\n"

        # Sprint progress
        total_items = progress.get('total_items', 0)
        completion_rate = progress.get('completion_rate', 0)
        velocity = progress.get('velocity', 0)
        story_points = progress.get('story_points', {})

        report += f"**Total Items**: {total_items}\n"
        report += f"**Completion Rate**: {completion_rate:.1f}%\n"
        report += f"**Velocity**: {velocity} story points completed\n\n"

        report += "### Story Points Breakdown\n\n"
        report += f"- Total: {story_points.get('total', 0)}\n"
        report += f"- Completed: {story_points.get('completed', 0)}\n"
        report += f"- In Progress: {story_points.get('in_progress', 0)}\n"
        report += f"- Not Started: {story_points.get('not_started', 0)}\n\n"

        # Add sprint status hierarchy section
        report += "---\n\n## Sprint Status Hierarchy\n\n"
        sprint_status_output = sprint_status.get('sprint_status_output', '')
        if sprint_status_output:
            report += "```\n"
            report += sprint_status_output
            report += "```\n\n"
        else:
            report += "_Sprint status not available_\n\n"

        # Add AI recommendations section if available
        if ai_recommendations:
            recommendations_text = ai_recommendations.get('recommendations', '')
            if recommendations_text and not recommendations_text.startswith("AI recommendations not available"):
                report += "---\n\n## Senior Engineer Recommendations\n\n"
                report += recommendations_text
                report += "\n\n"

        report += "---\n\n"
        report += "_Generated by Trustable AI Development Workbench_\n"

        return report

    def _generate_report_with_ai(
        self,
        recent_activity: Dict[str, Any],
        verification: Dict[str, Any],
        focus: Dict[str, Any],
        blockers: Dict[str, Any],
        progress: Dict[str, Any],
        sprint_status: Dict[str, Any],
        ai_recommendations: Dict[str, Any]
    ) -> str:
        """Generate AI-enhanced report (Mode 2)."""
        # For now, use the same report as simple mode (with AI recommendations already included)
        return self._generate_report_simple(
            recent_activity, verification, focus, blockers, progress,
            sprint_status, ai_recommendations
        )

    def _step_9_save_report(self) -> Dict[str, Any]:
        """Step 9: Save report to file (Mode 1)."""
        if self.verbose:
            print("\n Saving report...")

        report_content = self.step_evidence.get("8-generate-report", {}).get("report_content", "")

        if not report_content:
            if self.verbose:
                print(" No report content to save")
            return {"saved": False}

        # Determine output file path
        if self.output_file:
            output_path = Path(self.output_file)
        else:
            # Default: .claude/reports/daily/standup-YYYYMMDD.md
            reports_dir = Path(".claude/reports/daily")
            reports_dir.mkdir(parents=True, exist_ok=True)
            filename = f"standup-{datetime.now().strftime('%Y%m%d')}.md"
            output_path = reports_dir / filename

        # Write report
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)

            if self.verbose:
                print(f" Report saved: {output_path}")

            return {
                "saved": True,
                "file_path": str(output_path)
            }

        except Exception as e:
            if self.verbose:
                print(f" Error saving report: {e}")
            return {
                "saved": False,
                "error": str(e)
            }


def markdown_to_terminal(text: str) -> str:
    """
    Convert markdown formatting to terminal-friendly plain text.

    Handles:
    - Headers (# ## ###) -> UPPERCASE with underlines
    - Bold (**text**) -> text (removes markers)
    - Horizontal rules (---) -> solid lines
    - Bullet points preserved
    - Code blocks preserved

    Args:
        text: Markdown formatted text

    Returns:
        Terminal-friendly plain text
    """
    import re

    lines = text.split('\n')
    result = []

    for line in lines:
        # Convert headers to uppercase with visual separator
        if line.startswith('### '):
            header_text = line[4:].strip()
            # Remove any bold markers from header
            header_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', header_text)
            result.append(f"\n {header_text.upper()}")
        elif line.startswith('## '):
            header_text = line[3:].strip()
            header_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', header_text)
            result.append(f"\n{'─' * 40}")
            result.append(f" {header_text.upper()}")
            result.append(f"{'─' * 40}")
        elif line.startswith('# '):
            header_text = line[2:].strip()
            header_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', header_text)
            result.append(f"\n{'─' * 60}")
            result.append(f" {header_text.upper()}")
            result.append(f"{'─' * 60}")
        elif line.strip() == '---':
            # Horizontal rule
            result.append(f"\n{'─' * 40}\n")
        else:
            # Remove bold markers
            line = re.sub(r'\*\*([^*]+)\*\*', r'\1', line)
            # Remove italic markers
            line = re.sub(r'\*([^*]+)\*', r'\1', line)
            result.append(line)

    return '\n'.join(result)


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


def get_current_sprint(adapter):
    """
    Detect the current active sprint.

    Strategy:
    1. Query all sprints from adapter
    2. Find sprint with current date between start/end dates
    3. If no date-based match, find most recently started sprint
    4. If no sprints found, return None

    Args:
    adapter: Work tracking adapter

    Returns:
    str: Current sprint name, or None if not found
    """
    try:
        # Query all sprints
        sprints = adapter.list_sprints()

        if not sprints:
            return None

        from datetime import datetime
        now = datetime.now()

        # Strategy 1: Find sprint where current date is between start/end
        for sprint in sprints:
            # Check if sprint has date fields
            start_date = sprint.get('start_date')
            end_date = sprint.get('finish_date') or sprint.get('end_date')

            if start_date and end_date:
                try:
                    # Parse dates (Azure DevOps format: YYYY-MM-DDTHH:MM:SSZ)
                    if isinstance(start_date, str):
                        normalized_start = normalize_azure_timestamp(start_date)
                        start = datetime.fromisoformat(normalized_start.replace('Z', '+00:00'))
                    else:
                        start = start_date

                    if isinstance(end_date, str):
                        normalized_end = normalize_azure_timestamp(end_date)
                        end = datetime.fromisoformat(normalized_end.replace('Z', '+00:00'))
                    else:
                        end = end_date

                    # Check if current date is within sprint dates
                    if start <= now <= end:
                        sprint_name = sprint.get('name', sprint.get('path', ''))
                        # Extract simple name if full path (e.g., "Project\\Sprint 8" -> "Sprint 8")
                        if '\\' in sprint_name:
                            sprint_name = sprint_name.split('\\')[-1]
                        return sprint_name
                except (ValueError, AttributeError):
                    # Date parsing failed, skip this sprint
                    continue

        # Strategy 2: Find most recently started sprint
        recent_sprint = None
        recent_start = None

        for sprint in sprints:
            start_date = sprint.get('start_date')
            if start_date:
                try:
                    if isinstance(start_date, str):
                        normalized_start = normalize_azure_timestamp(start_date)
                        start = datetime.fromisoformat(normalized_start.replace('Z', '+00:00'))
                    else:
                        start = start_date

                    # Only consider sprints that have started
                    if start <= now:
                        if recent_start is None or start > recent_start:
                            recent_start = start
                            sprint_name = sprint.get('name', sprint.get('path', ''))
                            # Extract simple name if full path
                            if '\\' in sprint_name:
                                sprint_name = sprint_name.split('\\')[-1]
                            recent_sprint = sprint_name
                except (ValueError, AttributeError):
                    continue

        if recent_sprint:
            return recent_sprint

        # Strategy 3: Fall back to last sprint in list
        last_sprint = sprints[-1]
        sprint_name = last_sprint.get('name', last_sprint.get('path', ''))
        # Extract simple name if full path
        if '\\' in sprint_name:
            sprint_name = sprint_name.split('\\')[-1]
        return sprint_name

    except Exception as e:
        print(f" Warning: Could not auto-detect current sprint: {e}")
        return None


def main():
    """Main entry point."""
    from cli.console import console

    parser = argparse.ArgumentParser(
        description="Daily Standup Report Workflow with External Enforcement"
    )
    parser.add_argument(
        "--sprint",
        required=False,
        help="Sprint name (e.g., 'Sprint 8'). If not specified, uses current active sprint."
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Time window for recent activity in hours (default: 24)"
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Disable AI recommendations (default: AI enabled)"
    )
    parser.add_argument(
        "--output-file",
        help="Optional output file path for report"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed step-by-step output (default: clean summary)"
    )
    parser.add_argument(
        "--debug-trace",
        action="store_true",
        help="Enable verbose debug output with timestamps and work item IDs"
    )

    args = parser.parse_args()

    # Auto-detect current sprint if not specified
    sprint_name = args.sprint
    if not sprint_name:
        console.print("[#BB93DD]I'm auto-detecting the current sprint...[/#BB93DD]")
        try:
            # Initialize adapter temporarily to detect sprint
            sys.path.insert(0, '.claude/skills')
            from work_tracking import get_adapter
            temp_adapter = get_adapter()
            sprint_name = get_current_sprint(temp_adapter)

            if sprint_name:
                console.print(f"[#71E4D1]Found it! Current sprint: {sprint_name}[/#71E4D1]")
            else:
                console.print("[#FF6B6B]I couldn't auto-detect the current sprint[/#FF6B6B]")
                console.print("[#D9EAFC]Please specify sprint with --sprint argument[/#D9EAFC]")
                console.print("[dim]Example: python scripts/daily_standup.py --sprint 'Sprint 8'[/dim]")
                sys.exit(1)
        except Exception as e:
            console.print(f"[#FF6B6B]I had trouble detecting the sprint: {e}[/#FF6B6B]")
            console.print("[#D9EAFC]Please specify sprint with --sprint argument[/#D9EAFC]")
            sys.exit(1)
    else:
        # Normalize sprint name (e.g., "9" → "Sprint 9")
        normalized = normalize_sprint_name(sprint_name)
        if normalized != sprint_name:
            console.print(f"[dim]I normalized the sprint name: '{sprint_name}' -> '{normalized}'[/dim]")
        sprint_name = normalized

    # Print header
    use_ai = not args.no_ai
    console.print()
    console.print("─" * 80)
    console.print("[bold #71E4D1]  DAILY STANDUP REPORT[/bold #71E4D1]")
    console.print("─" * 80)
    console.print()
    console.print(f"[#D9EAFC]Sprint:[/#D9EAFC] [bold]{sprint_name}[/bold]")
    console.print(f"[#D9EAFC]Time Window:[/#D9EAFC] Last {args.hours} hours")
    console.print(f"[#D9EAFC]Mode:[/#D9EAFC] {'AI-Assisted' if use_ai else 'Fast (no AI)'}")
    console.print()

    # Initialize and execute workflow
    workflow = DailyStandupWorkflow(
        sprint_name=sprint_name,
        hours=args.hours,
        use_ai=use_ai,
        output_file=args.output_file,
        args=args,
        verbose=args.verbose,
        debug_trace=args.debug_trace
    )

    try:
        result = workflow.execute()

        # Print sprint status hierarchy (ticket display)
        sprint_status_output = workflow.step_evidence.get("6-sprint-status", {}).get("sprint_status_output", "")
        if sprint_status_output:
            console.print(sprint_status_output)

        # Print summary header
        console.print()
        console.print("─" * 80)
        console.print("[bold #71E4D1]  DAILY STANDUP COMPLETE[/bold #71E4D1]")
        console.print("─" * 80)

        # Print key metrics
        recent_count = workflow.step_evidence.get("1-gather-recent", {}).get("recent_count", 0)
        active_count = workflow.step_evidence.get("3-identify-focus", {}).get("active_count", 0)
        total_blockers = workflow.step_evidence.get("4-detect-blockers", {}).get("total_blockers", 0)
        completion_rate = workflow.step_evidence.get("5-analyze-progress", {}).get("completion_rate", 0)

        console.print()
        console.print(f"[bold]{sprint_name}[/bold] [dim]|[/dim] [#71E4D1]{completion_rate:.0f}% complete[/#71E4D1] [dim]|[/dim] {active_count} active [dim]|[/dim] {total_blockers} blockers [dim]|[/dim] {recent_count} updated")

        # Divergence warning if detected
        divergence_count = workflow.step_evidence.get("2-verify-states", {}).get("divergence_count", 0)
        if divergence_count > 0:
            console.print(f"[#FFA500]Heads up! {divergence_count} work item(s) have state divergence - they need attention[/#FFA500]")

        # AI recommendations (if available)
        ai_recommendations = workflow.step_evidence.get("7-ai-recommendations", {})
        recommendations_text = ai_recommendations.get("recommendations", "")
        if recommendations_text and not recommendations_text.startswith("AI recommendations not available"):
            console.print()
            console.print("─" * 80)
            console.print("[bold #BB93DD]  SENIOR ENGINEER RECOMMENDATIONS[/bold #BB93DD]")
            console.print("─" * 80)
            # Convert markdown to terminal-friendly format
            terminal_text = markdown_to_terminal(recommendations_text)
            console.print(terminal_text)

        # Report path
        report_path = workflow.step_evidence.get("9-save-report", {}).get("file_path")
        if report_path:
            console.print()
            console.print(f"[#71E4D1]Full report saved:[/#71E4D1] {report_path}")

        console.print()

    except KeyboardInterrupt:
        console.print("\n[#FFA500]I was interrupted - stopping workflow[/#FFA500]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[#FF6B6B]Oops! The workflow failed: {e}[/#FF6B6B]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
