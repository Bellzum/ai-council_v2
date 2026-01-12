#!/usr/bin/env python3
"""
Sprint Review Workflow with External Enforcement

Implements Phase 3: Medium Workflows - Sprint Review Script

13-Step Workflow:
1. Sprint Selection - User selects sprint to review
2. Query Completed Work Items - Query all Tasks and Bugs in Done/Closed state (actual sprint work)
3. Generate Demo Checklist - AI generates checklist for Tasks/Bugs to demonstrate (Mode 2)
4. Approval Gate - BLOCKING approval for parent completion with evidence review option
5. Parent Completion - Auto-complete Features (all Tasks done) then EPICs (all Features done)
6. Test Plan Verification - Check for test coverage on Tasks/Bugs
7. Acceptance Tests - Tester agent runs acceptance tests for completed work
8. Security Review - Security specialist agent performs final security review
9. Deployment Readiness - Engineer agent assesses deployment readiness
10. Sprint Closure Decision - Scrum master agent recommends closure decision
11. Final Human Approval - BLOCKING approval for sprint closure (close/extend/cancel)
12. Save Sprint Review Report - Generate and save comprehensive report
13. Checkpoint - Save workflow state

Key Concepts:
- Sprint work = Tasks and Bugs (NOT Features/EPICs)
- Features/EPICs are parent work items, completed when all children are done
- Cascading completion: Tasks→Features→EPICs
- Agent-based quality gates: tester, security, engineer, scrum-master

Design Pattern:
- Extends WorkflowOrchestrator from Phase 1
- Uses adapter for ALL work item operations (NO `az boards` CLI)
- Parent auto-completion with external verification
- Real input() blocking for approval gates with evidence review
- Agent-based reviews spawn fresh context windows
- UTF-8 encoding for all file writes
- Rollback safety: parent transition failures don't crash workflow
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

from scripts.workflow_executor.base import WorkflowOrchestrator, ExecutionMode
from scripts.workflow_executor.schemas import StepType
from scripts.workflow_executor.validators import (
    check_epic_completion_eligibility,
    auto_complete_epic
)

# Import unified console infrastructure
from core.console_workflow import (
    print_workflow_header,
    print_step_header,
    print_work_items_table,
    print_approval_gate,
    print_metrics_table,
    print_summary_panel,
    print_section_divider,
    WorkItem,
    WorkItemType,
    ApprovalGateData,
    CapacityMetrics
)
from cli.console import print_success, print_error, print_warning, print_info

# Import JSON schema validation
try:
    from jsonschema import validate, ValidationError
except ImportError:
    print("⚠️  jsonschema package not installed - install with: pip install jsonschema")
    ValidationError = Exception  # Fallback


def get_work_item_field(item: Dict[str, Any], field: str, default: Any = None) -> Any:
    """
    Extract a field from a work item dict, handling different response formats.

    Azure DevOps REST API returns fields in a nested structure:
    - item['fields']['System.Title'] for full work item fetch
    - item['title'] for simplified responses

    Args:
        item: Work item dict from adapter
        field: Field name (e.g., 'title', 'state', 'type')
        default: Default value if field not found

    Returns:
        Field value or default
    """
    # Map common field names to Azure DevOps field names
    field_map = {
        'title': 'System.Title',
        'state': 'System.State',
        'type': 'System.WorkItemType',
        'description': 'System.Description',
        'assigned_to': 'System.AssignedTo',
        'iteration_path': 'System.IterationPath',
    }

    azure_field = field_map.get(field, field)

    # Try direct access first (simplified response)
    if field in item:
        return item[field]

    # Try nested fields structure (full work item)
    fields = item.get('fields', {})
    if azure_field in fields:
        return fields[azure_field]

    # Try with System. prefix if not already prefixed
    if not azure_field.startswith('System.') and f'System.{azure_field}' in fields:
        return fields[f'System.{azure_field}']

    return default


def convert_to_work_item(item: Dict[str, Any]) -> WorkItem:
    """
    Convert adapter work item dict to WorkItem dataclass.

    Args:
        item: Work item dict from adapter

    Returns:
        WorkItem object
    """
    item_id = item.get('id')
    item_title = get_work_item_field(item, 'title', 'Unknown')
    item_type_str = get_work_item_field(item, 'type', 'Task')
    item_state = get_work_item_field(item, 'state', 'Unknown')

    # Map string type to WorkItemType enum
    type_map = {
        'Epic': WorkItemType.EPIC,
        'Feature': WorkItemType.FEATURE,
        'Task': WorkItemType.TASK,
        'Bug': WorkItemType.BUG
    }
    item_type = type_map.get(item_type_str, WorkItemType.TASK)

    # Try to get story points
    fields = item.get('fields', {})
    story_points = fields.get('Microsoft.VSTS.Scheduling.StoryPoints') or fields.get('Story Points')

    return WorkItem(
        id=item_id,
        title=item_title,
        type=item_type,
        state=item_state,
        story_points=int(story_points) if story_points else None
    )


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


class SprintReviewWorkflow(WorkflowOrchestrator):
    """
    Sprint Review workflow with external enforcement.

    Implements the 13-step sprint review process with:
    - Query completed Tasks and Bugs from sprint (actual sprint work)
    - AI-generated demo checklist for sprint work items
    - Parent completion: Features (all Tasks done) → EPICs (all Features done)
    - External verification of state changes
    - Agent-based quality gates: tester, security-specialist, engineer, scrum-master
    - Blocking approval gates with evidence review option
    - Final human approval for sprint closure (close/extend/cancel)
    """

    def __init__(
        self,
        sprint_name: str,
        workflow_id: str,
        enable_checkpoints: bool = True,
        use_ai: bool = False,
        interactive: bool = False
    ):
        """
        Initialize sprint review workflow.

        Args:
            sprint_name: Sprint to review (e.g., "Sprint 7")
            workflow_id: Unique ID for this execution
            enable_checkpoints: Enable state checkpointing
            use_ai: If True, use AI for demo checklist (Mode 2), otherwise use simple (Mode 1)
            interactive: If True, use Mode 3 interactive collaboration with Claude Agent SDK
        """
        # Normalize sprint name (e.g., "9" → "Sprint 9")
        self.sprint_name = normalize_sprint_name(sprint_name)
        self.use_ai = use_ai
        self.interactive = interactive

        # Interactive mode overrides use_ai (Mode 3 vs Mode 2)
        if interactive and use_ai:
            print_warning("Both --use-ai and --interactive specified - using interactive mode (Mode 3)")
            self.use_ai = False  # Interactive mode takes precedence

        mode = ExecutionMode.INTERACTIVE_AI if interactive else (ExecutionMode.AI_JSON_VALIDATION if use_ai else ExecutionMode.PURE_PYTHON)

        super().__init__(
            workflow_name="sprint-review",
            workflow_id=workflow_id,
            mode=mode,
            enable_checkpoints=enable_checkpoints
        )

        # Initialize adapter
        try:
            sys.path.insert(0, '.claude/skills')
            from work_tracking import get_adapter
            self.adapter = get_adapter()
        except Exception as e:
            print_warning(f"Could not initialize adapter: {e}")
            print_info("Continuing with limited functionality...")
            self.adapter = None

        # Initialize Claude API client if using AI
        self.claude_client = None
        self.token_usage = {}
        if use_ai:
            try:
                import anthropic
                api_key = os.getenv("KEYCHAIN_ANTHROPIC_API_KEY")
                if api_key:
                    self.claude_client = anthropic.Anthropic(api_key=api_key)
                    print_success("Claude API client initialized")
                else:
                    print_warning("KEYCHAIN_ANTHROPIC_API_KEY not set, falling back to simple logic mode")
                    self.use_ai = False
            except ImportError:
                print_warning("anthropic package not installed, falling back to simple logic mode")
                self.use_ai = False

        # Initialize interactive session if interactive mode
        self.interactive_session = None
        if interactive:
            try:
                from scripts.workflow_executor.interactive_session import InteractiveSession
                self.interactive_session = InteractiveSession(
                    workflow_name="sprint-review",
                    session_id=sprint_name.replace(' ', '-'),
                    model="claude-sonnet-4-5",
                    max_tokens=4000
                )
                if self.interactive_session.is_available():
                    print_success("Interactive mode initialized (Mode 3)")
                else:
                    print_warning("Interactive mode unavailable - falling back to mock data")
                    self.interactive = False
            except ImportError as e:
                print_warning(f"Interactive mode unavailable: {e}")
                print_info("Falling back to mock data")
                self.interactive = False

    def _define_steps(self) -> List[Dict[str, Any]]:
        """Define the 13 workflow steps."""
        return [
            {
                "id": "1-sprint-selection",
                "name": "Sprint Selection",
                "step_type": StepType.DATA_COLLECTION,
                "description": "Confirm sprint to review",
                "required": True
            },
            {
                "id": "2-query-completed",
                "name": "Query Completed Work Items",
                "step_type": StepType.DATA_COLLECTION,
                "description": "Query all Tasks and Bugs in Done/Closed state",
                "required": True,
                "depends_on": ["1-sprint-selection"]
            },
            {
                "id": "3-demo-checklist",
                "name": "Generate Demo Checklist",
                "step_type": StepType.AI_REVIEW,
                "description": "AI generates demo checklist for completed work",
                "required": True,
                "depends_on": ["2-query-completed"]
            },
            {
                "id": "4-approval-gate",
                "name": "Human Approval Gate",
                "step_type": StepType.APPROVAL_GATE,
                "description": "BLOCKING approval for parent completion (Features/EPICs)",
                "required": True,
                "depends_on": ["3-demo-checklist"]
            },
            {
                "id": "5-parent-completion",
                "name": "Parent Work Item Completion",
                "step_type": StepType.ACTION,
                "description": "Auto-complete Features (all Tasks done) and EPICs (all Features done)",
                "required": True,
                "depends_on": ["4-approval-gate"]
            },
            {
                "id": "6-test-verification",
                "name": "Test Plan Verification",
                "step_type": StepType.VERIFICATION,
                "description": "Check test coverage for Tasks and Bugs",
                "required": True,
                "depends_on": ["5-parent-completion"]
            },
            {
                "id": "7-acceptance-tests",
                "name": "Run Acceptance Tests",
                "step_type": StepType.AI_REVIEW,
                "description": "Tester agent runs acceptance tests for completed work",
                "required": True,
                "depends_on": ["6-test-verification"]
            },
            {
                "id": "8-security-review",
                "name": "Security Review",
                "step_type": StepType.AI_REVIEW,
                "description": "Security specialist agent performs final security review",
                "required": True,
                "depends_on": ["7-acceptance-tests"]
            },
            {
                "id": "9-deployment-readiness",
                "name": "Deployment Readiness Assessment",
                "step_type": StepType.AI_REVIEW,
                "description": "Engineer agent assesses deployment readiness",
                "required": True,
                "depends_on": ["8-security-review"]
            },
            {
                "id": "10-closure-decision",
                "name": "Sprint Closure Decision",
                "step_type": StepType.AI_REVIEW,
                "description": "Scrum master recommends sprint closure decision",
                "required": True,
                "depends_on": ["9-deployment-readiness"]
            },
            {
                "id": "11-final-approval",
                "name": "Final Human Approval",
                "step_type": StepType.APPROVAL_GATE,
                "description": "BLOCKING approval for sprint closure",
                "required": True,
                "depends_on": ["10-closure-decision"]
            },
            {
                "id": "12-save-report",
                "name": "Save Sprint Review Report",
                "step_type": StepType.ACTION,
                "description": "Generate and save comprehensive report",
                "required": True,
                "depends_on": ["11-final-approval"]
            },
            {
                "id": "13-checkpoint",
                "name": "Save Checkpoint",
                "step_type": StepType.ACTION,
                "description": "Save workflow state",
                "required": True,
                "depends_on": ["12-save-report"]
            }
        ]

    def _execute_step(
        self,
        step: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single workflow step."""
        step_id = step["id"]

        # Route to step-specific handler
        if step_id == "1-sprint-selection":
            return self._step_1_sprint_selection(context)
        elif step_id == "2-query-completed":
            return self._step_2_query_completed(context)
        elif step_id == "3-demo-checklist":
            return self._step_3_demo_checklist(context)
        elif step_id == "4-approval-gate":
            return self._step_4_approval_gate(context)
        elif step_id == "5-parent-completion":
            return self._step_5_parent_completion(context)
        elif step_id == "5-epic-completion":
            # Legacy step ID - redirect to new method
            return self._step_5_parent_completion(context)
        elif step_id == "6-test-verification":
            return self._step_6_test_verification(context)
        elif step_id == "7-acceptance-tests":
            return self._step_7_acceptance_tests(context)
        elif step_id == "8-security-review":
            return self._step_8_security_review(context)
        elif step_id == "9-deployment-readiness":
            return self._step_9_deployment_readiness(context)
        elif step_id == "10-closure-decision":
            return self._step_10_closure_decision(context)
        elif step_id == "11-final-approval":
            return self._step_11_final_approval(context)
        elif step_id == "12-save-report":
            return self._step_12_save_report(context)
        elif step_id == "13-checkpoint":
            return self._step_13_checkpoint(context)
        # Legacy step IDs for backwards compatibility
        elif step_id == "7-save-report":
            return self._step_12_save_report(context)
        elif step_id == "8-checkpoint":
            return self._step_13_checkpoint(context)
        else:
            raise ValueError(f"Unknown step: {step_id}")

    # ========================================================================
    # Interactive Mode 3 Methods (Feature #1215)
    # ========================================================================

    def _demo_preparation_interactive(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Interactive demo preparation with AI (Mode 3).

        User collaborates with AI to prepare demo for completed features.

        Returns:
            Demo preparation guide with key points and talking points
        """
        if not self.interactive_session or not self.interactive_session.is_available():
            return {"features": features, "skipped": True}

        context = {
            "sprint_name": self.sprint_name,
            "feature_count": len(features),
            "features": [f.get("title", "Unknown") for f in features[:3]]
        }

        initial_prompt = f"""You are preparing a sprint demo for {self.sprint_name}.

Help prepare talking points and key demonstrations for {len(features)} feature(s)."""

        try:
            result = self.interactive_session.discuss(
                initial_prompt=initial_prompt,
                context=context,
                max_iterations=3
            )
            return {"features": features, "demo_guide": result.get("final_response")}
        except Exception as e:
            print_warning(f"Interactive preparation failed: {e}")
            return {"features": features, "skipped": True}

    def _completion_assessment_interactive(self, epics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Interactive completion assessment with AI (Mode 3).

        User collaborates with AI to assess EPIC completion readiness.

        Returns:
            Assessment of which EPICs are ready to auto-complete
        """
        if not self.interactive_session or not self.interactive_session.is_available():
            return {"epics": epics, "skipped": True}

        context = {
            "sprint_name": self.sprint_name,
            "epic_count": len(epics),
            "epics": [e.get("title", "Unknown") for e in epics[:3]]
        }

        initial_prompt = f"""You are assessing completion readiness for {len(epics)} EPIC(s) in {self.sprint_name}.

Review which EPICs have all child features completed and are ready for auto-completion."""

        try:
            result = self.interactive_session.discuss(
                initial_prompt=initial_prompt,
                context=context,
                max_iterations=3
            )
            return {"epics": epics, "assessment": result.get("final_response")}
        except Exception as e:
            print_warning(f"Interactive assessment failed: {e}")
            return {"epics": epics, "skipped": True}

    def _generate_demo_checklist_interactive(self, work_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate demo checklist interactively using user prompts.

        Args:
            work_items: List of work items from the sprint

        Returns:
            List of checklist items with structure:
            [{
                'work_item_id': int,
                'work_item_title': str,
                'work_item_type': str,
                'demo_points': List[str],
                'stakeholder_questions': List[str]
            }]
        """
        print_section_divider()
        print_info("Demo Checklist Generation (Interactive)")
        print_info(
            f"Found {len(work_items)} work items in this sprint.\n"
            "Let's build a demo checklist for the items you want to showcase."
        )

        checklist_items = []

        # Step 1: Select work items to include
        print_info("\nStep 1: Select work items to include in demo")
        for idx, item in enumerate(work_items, 1):
            item_id = item.get('id')
            item_title = get_work_item_field(item, 'title', 'Untitled')
            item_type = get_work_item_field(item, 'type', 'Unknown')

            item_display = f"[{item_type}] {item_title} (ID: {item_id})"
            while True:
                response = input(f"Include item {idx}/{len(work_items)}: {item_display}? (yes/no) [yes]: ").strip().lower()
                if response in ['', 'yes', 'y', 'no', 'n']:
                    break
                print_warning("Please enter 'yes' or 'no'")

            include = response in ['', 'yes', 'y']

            if include:
                # Step 2: Add demo points for this item
                print_info(f"\nAdding demo points for: {item_title}")
                demo_points = []

                # Suggest default demo point based on type
                default_point = self._suggest_demo_point(item)
                demo_points.append(default_point)
                print_success(f"Added default demo point: {default_point}")

                # Allow adding more demo points
                while True:
                    while True:
                        response = input("Add another demo point? (yes/no) [no]: ").strip().lower()
                        if response in ['', 'yes', 'y', 'no', 'n']:
                            break
                        print_warning("Please enter 'yes' or 'no'")

                    add_more = response in ['yes', 'y']
                    if not add_more:
                        break

                    custom_point = input("Enter demo point: ").strip()
                    if custom_point:
                        demo_points.append(custom_point)
                        print_success(f"Added: {custom_point}")
                    else:
                        print_warning("Empty demo point, skipping")

                # Step 3: Add stakeholder questions
                print_info(f"\nAdding stakeholder questions for: {item_title}")
                stakeholder_questions = []

                # Suggest default question
                default_question = self._suggest_stakeholder_question(item)
                stakeholder_questions.append(default_question)
                print_success(f"Added default question: {default_question}")

                # Allow adding more questions
                while True:
                    while True:
                        response = input("Add another stakeholder question? (yes/no) [no]: ").strip().lower()
                        if response in ['', 'yes', 'y', 'no', 'n']:
                            break
                        print_warning("Please enter 'yes' or 'no'")

                    add_more = response in ['yes', 'y']
                    if not add_more:
                        break

                    custom_question = input("Enter stakeholder question: ").strip()
                    if custom_question:
                        stakeholder_questions.append(custom_question)
                        print_success(f"Added: {custom_question}")
                    else:
                        print_warning("Empty question, skipping")

                # Build checklist item
                checklist_items.append({
                    'work_item_id': item_id,
                    'work_item_title': item_title,
                    'work_item_type': item_type,
                    'demo_points': demo_points,
                    'stakeholder_questions': stakeholder_questions
                })

        # Step 4: Review and confirm
        print_section_divider()
        print_info("Demo Checklist Review")
        print_info(f"Generated checklist with {len(checklist_items)} items:\n")

        for idx, item in enumerate(checklist_items, 1):
            print_info(f"{idx}. [{item['work_item_type']}] {item['work_item_title']}")
            print_info(f"   Demo Points: {len(item['demo_points'])}")
            print_info(f"   Stakeholder Questions: {len(item['stakeholder_questions'])}")

        while True:
            response = input("\nConfirm this demo checklist? (yes/no) [yes]: ").strip().lower()
            if response in ['', 'yes', 'y', 'no', 'n']:
                break
            print_warning("Please enter 'yes' or 'no'")

        confirmed = response in ['', 'yes', 'y']

        if not confirmed:
            print_warning("Checklist rejected. Falling back to AI generation.")
            return self._generate_demo_checklist_ai(work_items)

        return checklist_items

    def _suggest_demo_point(self, work_item: Dict[str, Any]) -> str:
        """Suggest a demo point based on work item type and title."""
        work_type = get_work_item_field(work_item, 'type', '').lower()
        title = get_work_item_field(work_item, 'title', 'this feature')

        if 'bug' in work_type:
            return f"Verify that {title} no longer reproduces"
        elif 'feature' in work_type or 'user story' in work_type:
            return f"Demonstrate {title} functionality"
        else:
            return f"Show completed work for {title}"

    def _suggest_stakeholder_question(self, work_item: Dict[str, Any]) -> str:
        """Suggest a stakeholder question based on work item."""
        title = get_work_item_field(work_item, 'title', 'this item')
        return f"Does {title} meet your expectations?"

    def _step_1_sprint_selection(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 1: Confirm sprint selection."""
        print_step_header(1, "Sprint Selection", "Confirm sprint to review")

        print_info(f"Sprint selected for review: {self.sprint_name}")

        evidence = {
            "sprint_name": self.sprint_name,
            "selected_at": datetime.now().isoformat()
        }

        return evidence

    def _step_2_query_completed(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Query completed Tasks and Bugs in sprint (the actual sprint work)."""
        print_step_header(2, "Query Completed Work Items", "Query all Tasks and Bugs in Done/Closed state")

        if not self.adapter:
            print_warning("No adapter - using simple data")
            return self._get_simple_completed_work_items()

        try:
            # Query all work items in sprint
            sprint_items = self.adapter.query_sprint_work_items(self.sprint_name)

            # Filter for Tasks and Bugs in Done/Closed state
            done_states = ['Done', 'Closed', 'Resolved']
            completed_tasks = []
            completed_bugs = []

            for item in sprint_items:
                item_type = get_work_item_field(item, 'type')
                item_state = get_work_item_field(item, 'state')

                if item_state in done_states:
                    if item_type == 'Task':
                        completed_tasks.append(item)
                    elif item_type == 'Bug':
                        completed_bugs.append(item)

            # Identify parent Features from completed Tasks/Bugs
            parent_features = {}  # feature_id -> feature_info
            for item in completed_tasks + completed_bugs:
                relations = item.get('relations', [])
                for relation in relations:
                    rel_type = relation.get('rel', '')
                    if 'Hierarchy-Reverse' in rel_type or 'Parent' in rel_type:
                        url = relation.get('url', '')
                        if url:
                            parent_id = url.split('/')[-1]
                            try:
                                parent_id = int(parent_id)
                                if parent_id not in parent_features:
                                    # Fetch parent to check if it's a Feature
                                    try:
                                        parent = self.adapter.get_work_item(parent_id)
                                        parent_type = get_work_item_field(parent, 'type')
                                        if parent_type == 'Feature':
                                            parent_features[parent_id] = {
                                                'id': parent_id,
                                                'title': get_work_item_field(parent, 'title', 'Unknown'),
                                                'state': get_work_item_field(parent, 'state'),
                                                'child_tasks_done': 0,
                                                'child_tasks_total': 0,
                                                'child_bugs_done': 0,
                                                'child_bugs_total': 0
                                            }
                                    except Exception:
                                        continue
                            except (ValueError, TypeError):
                                continue

            # Count children for each parent Feature
            for feature_id in parent_features:
                try:
                    feature = self.adapter.get_work_item(feature_id)
                    relations = feature.get('relations', [])
                    for relation in relations:
                        rel_type = relation.get('rel', '')
                        if 'Hierarchy-Forward' in rel_type or 'Child' in rel_type:
                            url = relation.get('url', '')
                            if url:
                                child_id = int(url.split('/')[-1])
                                try:
                                    child = self.adapter.get_work_item(child_id)
                                    child_type = get_work_item_field(child, 'type')
                                    child_state = get_work_item_field(child, 'state')
                                    if child_type == 'Task':
                                        parent_features[feature_id]['child_tasks_total'] += 1
                                        if child_state in done_states:
                                            parent_features[feature_id]['child_tasks_done'] += 1
                                    elif child_type == 'Bug':
                                        parent_features[feature_id]['child_bugs_total'] += 1
                                        if child_state in done_states:
                                            parent_features[feature_id]['child_bugs_done'] += 1
                                except Exception:
                                    continue
                except Exception:
                    continue

            # Identify Features ready for completion (all children done)
            features_ready_for_completion = []
            for feature_id, feature_info in parent_features.items():
                total = feature_info['child_tasks_total'] + feature_info['child_bugs_total']
                done = feature_info['child_tasks_done'] + feature_info['child_bugs_done']
                if total > 0 and done == total and feature_info['state'] != 'Done':
                    feature_info['all_children_done'] = True
                    features_ready_for_completion.append(feature_info)

            # Find parent EPICs of Features ready for completion
            parent_epics = {}
            for feature_info in features_ready_for_completion:
                try:
                    feature = self.adapter.get_work_item(feature_info['id'])
                    relations = feature.get('relations', [])
                    for relation in relations:
                        rel_type = relation.get('rel', '')
                        if 'Hierarchy-Reverse' in rel_type or 'Parent' in rel_type:
                            url = relation.get('url', '')
                            if url:
                                epic_id = int(url.split('/')[-1])
                                if epic_id not in parent_epics:
                                    try:
                                        epic = self.adapter.get_work_item(epic_id)
                                        epic_type = get_work_item_field(epic, 'type')
                                        if epic_type == 'Epic':
                                            parent_epics[epic_id] = {
                                                'id': epic_id,
                                                'title': get_work_item_field(epic, 'title', 'Unknown'),
                                                'state': get_work_item_field(epic, 'state'),
                                                'child_features': []
                                            }
                                    except Exception:
                                        continue
                                if epic_id in parent_epics:
                                    parent_epics[epic_id]['child_features'].append(feature_info['id'])
                except Exception:
                    continue

            evidence = {
                "sprint_name": self.sprint_name,
                "completed_tasks": completed_tasks,
                "completed_bugs": completed_bugs,
                "task_count": len(completed_tasks),
                "bug_count": len(completed_bugs),
                "parent_features": list(parent_features.values()),
                "features_ready_for_completion": features_ready_for_completion,
                "parent_epics": list(parent_epics.values()),
                "queried_at": datetime.now().isoformat()
            }

            # Convert to WorkItem objects and display in tables
            task_work_items = [convert_to_work_item(t) for t in completed_tasks]
            bug_work_items = [convert_to_work_item(b) for b in completed_bugs]

            print_success("Sprint work items query complete")
            print_work_items_table(task_work_items, title=f"Completed Tasks ({len(completed_tasks)})")
            print_work_items_table(bug_work_items, title=f"Completed Bugs ({len(completed_bugs)})")

            # Print summary
            summary_data = {
                "Parent Features Identified": len(parent_features),
                "Features Ready for Completion": len(features_ready_for_completion),
                "Parent EPICs to Check": len(parent_epics)
            }
            print_summary_panel("Parent Work Items Summary", summary_data, style="info")

            return evidence

        except Exception as e:
            print_error(f"Query error: {e}")
            import traceback
            traceback.print_exc()
            return self._get_simple_completed_work_items()

    def _get_simple_completed_work_items(self) -> Dict[str, Any]:
        """Get simple completed work items for testing."""
        return {
            "sprint_name": self.sprint_name,
            "completed_tasks": [
                {
                    "id": "MOCK-301",
                    "title": "Implement login form",
                    "state": "Done",
                    "type": "Task"
                },
                {
                    "id": "MOCK-302",
                    "title": "Add password validation",
                    "state": "Done",
                    "type": "Task"
                }
            ],
            "completed_bugs": [
                {
                    "id": "MOCK-401",
                    "title": "Fix login timeout issue",
                    "state": "Done",
                    "type": "Bug"
                }
            ],
            "task_count": 2,
            "bug_count": 1,
            "parent_features": [
                {
                    "id": "MOCK-201",
                    "title": "User Authentication",
                    "state": "Active",
                    "child_tasks_done": 2,
                    "child_tasks_total": 2,
                    "child_bugs_done": 1,
                    "child_bugs_total": 1,
                    "all_children_done": True
                }
            ],
            "features_ready_for_completion": [
                {
                    "id": "MOCK-201",
                    "title": "User Authentication",
                    "state": "Active",
                    "all_children_done": True
                }
            ],
            "parent_epics": [
                {
                    "id": "MOCK-100",
                    "title": "Authentication System",
                    "state": "Active",
                    "child_features": ["MOCK-201"]
                }
            ],
            "mock": True
        }

    def _step_3_demo_checklist(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Generate demo checklist for completed Tasks and Bugs (Mode 2: AI with JSON)."""
        print_step_header(3, "Generate Demo Checklist", "AI generates demo checklist for completed work")

        completed = self.step_evidence.get("2-query-completed", {})
        tasks = completed.get("completed_tasks", [])
        bugs = completed.get("completed_bugs", [])
        work_items = tasks + bugs

        if not work_items:
            print_warning("No Tasks or Bugs to demo")
            return {"checklist_items": [], "task_items": [], "bug_items": []}

        if self.interactive:
            checklist_items = self._generate_demo_checklist_interactive(work_items)
        elif self.use_ai:
            checklist_items = self._generate_demo_checklist_ai(work_items)
        else:
            checklist_items = self._generate_demo_checklist_simple(work_items)

        # Separate by type for the report
        task_items = [item for item in checklist_items if item.get('work_item_type') == 'Task']
        bug_items = [item for item in checklist_items if item.get('work_item_type') == 'Bug']

        evidence = {
            "checklist_items": checklist_items,
            "task_items": task_items,
            "bug_items": bug_items,
            "item_count": len(checklist_items),
            "generated_at": datetime.now().isoformat()
        }

        # Print summary
        summary_data = {
            "Total Checklist Items": len(checklist_items),
            "Task Demos": len(task_items),
            "Bug Fix Demos": len(bug_items)
        }
        print_summary_panel("Demo Checklist Generated", summary_data, style="success")

        return evidence

    def _generate_demo_checklist_simple(self, work_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate simple demo checklist for Tasks and Bugs (Mode 1: Pure Python)."""
        checklist_items = []

        for item in work_items:
            item_id = item.get('id')
            item_title = get_work_item_field(item, 'title', 'Unknown')
            item_type = get_work_item_field(item, 'type', 'Task')

            if item_type == 'Bug':
                demo_points = [
                    f"Show the bug fix: {item_title}",
                    f"Demonstrate the issue is resolved",
                    f"Verify no regression in related functionality"
                ]
                questions = [
                    "Does the fix fully resolve the reported issue?",
                    "Are there related areas that should be tested?"
                ]
            else:
                demo_points = [
                    f"Demonstrate the implemented functionality: {item_title}",
                    f"Show user workflow and interaction",
                    f"Highlight acceptance criteria met"
                ]
                questions = [
                    "Does this meet the requirements?",
                    "Are there any edge cases to consider?"
                ]

            checklist_items.append({
                "work_item_id": item_id,
                "work_item_title": item_title,
                "work_item_type": item_type,
                "demo_points": demo_points,
                "stakeholder_questions": questions
            })

        return checklist_items

    def _generate_demo_checklist_ai(self, work_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate AI-based demo checklist for Tasks and Bugs (Mode 2: AI with JSON validation).

        Uses Claude Agent SDK with tool access so the agent can:
        - Read CLAUDE.md for project context
        - Explore the codebase to understand what was implemented
        """
        # Define JSON schema
        schema = {
            "type": "object",
            "properties": {
                "demo_checklist": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "work_item_id": {"type": ["string", "integer"]},
                            "work_item_title": {"type": "string"},
                            "work_item_type": {"type": "string", "enum": ["Task", "Bug"]},
                            "demo_points": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 2
                            },
                            "stakeholder_questions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1
                            }
                        },
                        "required": ["work_item_id", "work_item_title", "work_item_type", "demo_points", "stakeholder_questions"]
                    },
                    "minItems": 1
                }
            },
            "required": ["demo_checklist"]
        }

        # Build contextual prompt
        item_list = []
        for item in work_items:
            item_id = item.get('id')
            item_title = get_work_item_field(item, 'title', 'Unknown')
            item_type = get_work_item_field(item, 'type', 'Task')
            item_desc = get_work_item_field(item, 'description', 'No description')
            item_list.append(f"- {item_type} {item_id}: {item_title}")
            if item_desc and item_desc != 'No description':
                item_list.append(f"  Description: {str(item_desc)[:200]}")

        items_text = "\n".join(item_list)

        prompt = f"""You are creating a demo checklist for a sprint review. For each completed Task or Bug, suggest what to demonstrate and questions to ask stakeholders.

**IMPORTANT: First read the project context**

Before generating demo points, use your Read tool to read `CLAUDE.md` in the project root to understand:
- Project purpose and key features
- Architecture and user-facing components
- Quality standards and demo expectations

If work items mention specific files or features, use Grep/Glob to explore what was actually implemented.

Sprint: {self.sprint_name}

Completed Work Items:
{items_text}

For each work item, provide:
1. Demo points: What to show (minimum 2 points, be specific about functionality)
   - For Tasks: Focus on new functionality implemented
   - For Bugs: Focus on the fix and verification that the issue is resolved
2. Stakeholder questions: What to ask for feedback (minimum 1 question)

Return ONLY valid JSON matching this exact schema:
{json.dumps(schema, indent=2)}"""

        # Use Agent SDK with tool access
        try:
            from scripts.workflow_executor.agent_sdk import AgentSDKWrapper
            import asyncio

            print("   Using Agent SDK with tool access...")

            wrapper = AgentSDKWrapper(
                workflow_name="sprint-review-demo",
                tool_preset="read_only",
                max_turns=15,
                model="claude-sonnet-4-5",
            )

            async def _run_demo_checklist():
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
                        future = executor.submit(asyncio.run, _run_demo_checklist())
                        result = future.result()
                else:
                    result = loop.run_until_complete(_run_demo_checklist())
            except RuntimeError:
                result = asyncio.run(_run_demo_checklist())

            if not result.success:
                print_warning(f"Agent SDK query failed: {result.error}")
                return self._generate_demo_checklist_simple(work_items)

            # Extract JSON from response
            response_text = result.response

            json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

            parsed_result = json.loads(response_text)
            validate(parsed_result, schema)

            # Track token usage
            self.token_usage["generate_demo_checklist"] = {
                "input_tokens": result.token_usage.input_tokens,
                "output_tokens": result.token_usage.output_tokens,
                "cost_usd": result.cost_usd
            }

            return parsed_result["demo_checklist"]

        except ImportError as e:
            # ALWAYS report import errors - no silent fallbacks
            print_error(f"AI Demo Checklist Failed: Import error: {e}")
            print_error("The Claude Agent SDK is not available. AI demo checklist requires tool access.")
            print_info("Install with: pip install claude-code-sdk")
            print_info("Workflow will continue without AI-generated demo checklist.")
            return []  # Return empty - don't pretend with degraded functionality

        except (json.JSONDecodeError, ValidationError) as e:
            # Report validation errors clearly
            print_error(f"AI Demo Checklist Failed: {type(e).__name__}: {e}")
            print_info("Workflow will continue without AI-generated demo checklist.")
            return []  # Return empty - don't pretend with degraded functionality

        except Exception as e:
            # ALWAYS report errors - no silent fallbacks
            print_error(f"AI Demo Checklist Failed: {type(e).__name__}: {e}")
            print_info("Workflow will continue without AI-generated demo checklist.")
            return []  # Return empty - don't pretend with degraded functionality

    def _generate_demo_checklist_anthropic_fallback(
        self,
        work_items: List[Dict[str, Any]],
        schema: Dict[str, Any],
        prompt: str
    ) -> List[Dict[str, Any]]:
        """Fallback to Anthropic API when Agent SDK is unavailable."""
        if not self.claude_client:
            return self._generate_demo_checklist_simple(work_items)

        for attempt in range(3):
            try:
                response = self.claude_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2500,
                    messages=[{"role": "user", "content": prompt}]
                )

                response_text = response.content[0].text

                json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(1)

                result = json.loads(response_text)
                validate(result, schema)

                self.token_usage["generate_demo_checklist"] = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cost_usd": self._calculate_cost(response.usage)
                }

                return result["demo_checklist"]

            except (json.JSONDecodeError, ValidationError) as e:
                print_warning(f"Attempt {attempt + 1}/3 failed: {type(e).__name__}: {e}")
                if attempt == 2:
                    print_error("All retries exhausted, falling back to mock")
                    return self._generate_demo_checklist_simple(work_items)
            except Exception as e:
                print_error(f"API error: {type(e).__name__}: {e}, falling back to mock")
                return self._generate_demo_checklist_simple(work_items)

        return self._generate_demo_checklist_simple(work_items)

    def _calculate_cost(self, usage) -> float:
        """Calculate cost in USD based on Claude API token usage."""
        input_cost = (usage.input_tokens / 1_000_000) * 3.0   # $3 per million
        output_cost = (usage.output_tokens / 1_000_000) * 15.0  # $15 per million
        return input_cost + output_cost

    def _step_4_approval_gate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 4: Human approval gate (BLOCKING).

        Execution HALTS here until user approves/rejects parent work item auto-completion.
        User can request evidence review before making a decision.
        """
        completed = self.step_evidence.get("2-query-completed", {})
        features_ready = completed.get("features_ready_for_completion", [])
        parent_epics = completed.get("parent_epics", [])
        completed_tasks = completed.get("completed_tasks", [])
        completed_bugs = completed.get("completed_bugs", [])

        # Build summary for approval gate
        summary_lines = [
            f"Sprint: {self.sprint_name}",
            "",
            "SPRINT WORK SUMMARY:",
            f"  • Completed Tasks: {len(completed_tasks)}",
            f"  • Completed Bugs: {len(completed_bugs)}",
            "",
            "PARENT COMPLETION CANDIDATES:",
            f"  • Features ready for completion: {len(features_ready)}"
        ]

        for feature in features_ready:
            summary_lines.append(f"    - #{feature['id']}: {feature.get('title', 'Unknown')}")

        summary_lines.append(f"  • Parent EPICs to check: {len(parent_epics)}")
        for epic in parent_epics:
            summary_lines.append(f"    - #{epic['id']}: {epic.get('title', 'Unknown')}")

        # Create approval gate data
        gate = ApprovalGateData(
            title="Step 4: Human Approval Gate",
            summary=summary_lines,
            options=[
                ("e", "Show EVIDENCE (detailed proof of completion claims)"),
                ("y", "APPROVE parent completion (Features→Done, then EPICs→Done if all Features complete)"),
                ("n", "SKIP parent completion (keep current states)")
            ],
            question="Enter choice (e/y/n): "
        )

        print_approval_gate(gate)

        # BLOCKING CALL - Execution halts here
        evidence_shown = False
        while True:
            try:
                response = input("Enter choice (e/y/n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print_error("\nApproval cancelled by user")
                response = "n"
                break

            if response == 'e':
                # Show detailed evidence
                evidence_shown = True
                self._show_completion_evidence(completed)
                print_section_divider()
                print_info("Now choose:")
                print_info("  y = APPROVE parent completion")
                print_info("  n = SKIP parent completion")
                print_section_divider()
                continue
            elif response in ['y', 'yes']:
                response = 'yes'
                break
            elif response in ['n', 'no']:
                response = 'no'
                break
            else:
                print_warning("Invalid choice. Enter 'e' for evidence, 'y' to approve, or 'n' to skip.")

        approved = response == "yes"

        evidence = {
            "approved": approved,
            "response": response,
            "evidence_reviewed": evidence_shown,
            "timestamp": datetime.now().isoformat(),
            "features_to_complete": len(features_ready),
            "epics_to_check": len(parent_epics)
        }

        if approved:
            print_success(f"User APPROVED - Will complete {len(features_ready)} Feature(s) and check {len(parent_epics)} EPIC(s)")
        else:
            print_warning("User DECLINED - Skipping parent completion")

        return evidence

    def _show_completion_evidence(self, completed: Dict[str, Any]) -> None:
        """Display detailed evidence for parent work item completion."""
        from core.console_workflow import WorkItem, WorkItemType, print_work_items_table

        print_section_divider("EVIDENCE REPORT")

        # Show completed Tasks
        tasks = completed.get("completed_tasks", [])
        if tasks:
            task_items = [
                WorkItem(
                    id=task.get('id'),
                    title=get_work_item_field(task, 'title', 'Unknown'),
                    type=WorkItemType.TASK,
                    state=get_work_item_field(task, 'state', 'Unknown'),
                    story_points=task.get('story_points')
                )
                for task in tasks
            ]
            print_work_items_table(task_items, title=f"✅ Completed Tasks ({len(tasks)})")
        else:
            print_info("No completed tasks to display")

        # Show completed Bugs
        bugs = completed.get("completed_bugs", [])
        if bugs:
            bug_items = [
                WorkItem(
                    id=bug.get('id'),
                    title=get_work_item_field(bug, 'title', 'Unknown'),
                    type=WorkItemType.BUG,
                    state=get_work_item_field(bug, 'state', 'Unknown'),
                    story_points=bug.get('story_points')
                )
                for bug in bugs
            ]
            print_work_items_table(bug_items, title=f"🐛 Completed Bugs ({len(bugs)})")
        else:
            print_info("No completed bugs to display")

        # Show Features ready for completion
        features = completed.get("features_ready_for_completion", [])
        if features:
            print_section_divider("Features Ready for Completion")
            for feature in features:
                feature_id = feature.get('id')
                feature_title = feature.get('title', 'Unknown')
                tasks_done = feature.get('child_tasks_done', 0)
                tasks_total = feature.get('child_tasks_total', 0)
                bugs_done = feature.get('child_bugs_done', 0)
                bugs_total = feature.get('child_bugs_total', 0)

                print_info(f"Feature #{feature_id}: {feature_title}")
                print_info(f"  Tasks: {tasks_done}/{tasks_total} done")
                print_info(f"  Bugs: {bugs_done}/{bugs_total} done")
                print_success(f"  Status: ALL CHILDREN COMPLETE ✓")
        else:
            print_info("No features ready for completion")

        # Show EPICs
        epics = completed.get("parent_epics", [])
        if epics:
            print_section_divider("Parent EPICs")
            for epic in epics:
                epic_id = epic.get('id')
                epic_title = epic.get('title', 'Unknown')
                epic_state = epic.get('state')
                child_features = epic.get('child_features', [])

                print_info(f"EPIC #{epic_id}: {epic_title}")
                print_info(f"  Current State: {epic_state}")
                print_info(f"  Child Features ready: {len(child_features)}")
        else:
            print_info("No parent EPICs to display")

        print_section_divider()

    def _step_5_parent_completion(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 5: Parent work item completion logic.

        Cascading completion:
        1. Complete Features where all child Tasks/Bugs are Done
        2. Complete EPICs where all child Features are Done
        """
        print_section_divider("Processing Parent Work Item Completions")

        # Check if approved
        approval = self.step_evidence.get("4-approval-gate", {})
        if not approval.get("approved"):
            print_info("Skipped - User did not approve parent completion")
            return {"feature_completions": [], "epic_completions": [], "skipped": True}

        completed = self.step_evidence.get("2-query-completed", {})
        features_ready = completed.get("features_ready_for_completion", [])
        parent_epics = completed.get("parent_epics", [])

        if not self.adapter:
            print_warning("No adapter - cannot complete parent items")
            return {"feature_completions": [], "epic_completions": [], "mock": True}

        feature_completions = []
        epic_completions = []

        # Phase 1: Complete Features with all children done
        print_section_divider("Phase 1: Feature Completion")

        if not features_ready:
            print_success("No Features ready for completion")
        else:
            for feature_info in features_ready:
                feature_id = feature_info['id']
                feature_title = feature_info.get('title', 'Unknown')

                print_info(f"Processing Feature #{feature_id}: {feature_title}")

                try:
                    # Transition Feature to Done
                    result = self.adapter.update_work_item(
                        work_item_id=feature_id,
                        fields={"System.State": "Done"}
                    )

                    if result:
                        print_success(f"Feature #{feature_id} completed")
                        feature_completions.append({
                            "feature_id": feature_id,
                            "title": feature_title,
                            "success": True,
                            "completed_at": datetime.now().isoformat()
                        })
                    else:
                        print_warning(f"Feature #{feature_id} transition returned no result")
                        feature_completions.append({
                            "feature_id": feature_id,
                            "title": feature_title,
                            "success": False,
                            "error": "No result from update",
                            "attempted_at": datetime.now().isoformat()
                        })

                except Exception as e:
                    print_error(f"Feature #{feature_id} transition failed: {e}")
                    feature_completions.append({
                        "feature_id": feature_id,
                        "title": feature_title,
                        "success": False,
                        "error": str(e),
                        "attempted_at": datetime.now().isoformat()
                    })

        # Phase 2: Complete EPICs with all child Features done
        print_section_divider("Phase 2: EPIC Completion")

        if not parent_epics:
            print_success("No EPICs to check")
        else:
            for epic_info in parent_epics:
                epic_id = epic_info['id']
                epic_title = epic_info.get('title', 'Unknown')

                print_info(f"Checking EPIC #{epic_id}: {epic_title}")

                try:
                    # Use the existing validator for EPIC completion eligibility
                    eligible, reason, evidence_data = check_epic_completion_eligibility(
                        self.adapter,
                        epic_id
                    )

                    if eligible:
                        print_info(f"EPIC #{epic_id} eligible: {reason}")
                        print_info(f"  - Total Features: {evidence_data.get('total_features', 0)}")
                        print_info(f"  - Done Features: {evidence_data.get('done_features', 0)}")

                        # Auto-complete EPIC
                        result = auto_complete_epic(self.adapter, epic_id, evidence_data)

                        if result['success']:
                            print_success(f"EPIC #{epic_id} auto-completed")
                            epic_completions.append({
                                "epic_id": epic_id,
                                "title": epic_title,
                                "reason": reason,
                                "evidence": evidence_data,
                                "success": True,
                                "completed_at": datetime.now().isoformat()
                            })
                        else:
                            print_warning(f"EPIC #{epic_id} transition failed: {result.get('error')}")
                            epic_completions.append({
                                "epic_id": epic_id,
                                "title": epic_title,
                                "reason": reason,
                                "evidence": evidence_data,
                                "success": False,
                                "error": result.get('error'),
                                "attempted_at": datetime.now().isoformat()
                            })
                    else:
                        print_info(f"EPIC #{epic_id} not eligible: {reason}")
                        if evidence_data:
                            incomplete_count = evidence_data.get('total_features', 0) - evidence_data.get('done_features', 0)
                            if incomplete_count > 0:
                                print_info(f"  - Incomplete Features: {incomplete_count}")

                        epic_completions.append({
                            "epic_id": epic_id,
                            "title": epic_title,
                            "eligible": False,
                            "reason": reason,
                            "evidence": evidence_data
                        })

                except Exception as e:
                    print_error(f"EPIC #{epic_id} check failed: {e}")
                    epic_completions.append({
                        "epic_id": epic_id,
                        "title": epic_title,
                        "success": False,
                        "error": str(e),
                        "attempted_at": datetime.now().isoformat()
                    })

        # Summary
        features_successful = [f for f in feature_completions if f.get('success')]
        features_failed = [f for f in feature_completions if f.get('success') is False]
        epics_successful = [e for e in epic_completions if e.get('success')]
        epics_failed = [e for e in epic_completions if e.get('success') is False]
        epics_ineligible = [e for e in epic_completions if e.get('eligible') is False]

        evidence = {
            "feature_completions": feature_completions,
            "epic_completions": epic_completions,
            "features_completed": len(features_successful),
            "features_failed": len(features_failed),
            "epics_completed": len(epics_successful),
            "epics_failed": len(epics_failed),
            "epics_ineligible": len(epics_ineligible),
            "processed_at": datetime.now().isoformat()
        }

        print_section_divider("Parent Completion Summary")
        print_info(f"Features: {len(features_successful)} completed, {len(features_failed)} failed")
        print_info(f"EPICs: {len(epics_successful)} completed, {len(epics_failed)} failed, {len(epics_ineligible)} ineligible")

        return evidence

    def _step_6_test_verification(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 6: Verify test plans/coverage for completed Tasks and Bugs."""
        print_step_header(6, "Test Plan Verification", "Check test coverage for Tasks and Bugs")

        completed = self.step_evidence.get("2-query-completed", {})
        tasks = completed.get("completed_tasks", [])
        bugs = completed.get("completed_bugs", [])
        work_items = tasks + bugs

        if not work_items:
            print_success("No Tasks or Bugs to verify")
            return {"verified": True, "task_count": 0, "bug_count": 0}

        # Check for test plan attachments or relations
        tasks_with_tests = []
        tasks_without_tests = []
        bugs_with_tests = []
        bugs_without_tests = []

        for item in work_items:
            item_id = item.get('id')
            item_title = get_work_item_field(item, 'title', 'Unknown')
            item_type = get_work_item_field(item, 'type', 'Task')

            # Check for AttachedFile or Test Case relations
            relations = item.get('relations', [])
            has_test_plan = False

            for relation in relations:
                rel_type = relation.get('rel', '')
                # Check for test-related attachments or relations
                if any(x in rel_type for x in ['AttachedFile', 'Tested By', 'Test Case']):
                    has_test_plan = True
                    break

            item_info = {"id": item_id, "title": item_title, "type": item_type}

            if item_type == 'Bug':
                if has_test_plan:
                    bugs_with_tests.append(item_id)
                else:
                    bugs_without_tests.append(item_info)
            else:  # Task
                if has_test_plan:
                    tasks_with_tests.append(item_id)
                else:
                    tasks_without_tests.append(item_info)

        total_items = len(work_items)
        items_with_tests = len(tasks_with_tests) + len(bugs_with_tests)
        coverage_rate = (items_with_tests / total_items) * 100 if total_items > 0 else 0

        evidence = {
            "tasks_with_tests": tasks_with_tests,
            "tasks_without_tests": tasks_without_tests,
            "bugs_with_tests": bugs_with_tests,
            "bugs_without_tests": bugs_without_tests,
            "task_coverage_rate": (len(tasks_with_tests) / len(tasks)) * 100 if tasks else 0,
            "bug_coverage_rate": (len(bugs_with_tests) / len(bugs)) * 100 if bugs else 0,
            "overall_coverage_rate": coverage_rate,
            "verified_at": datetime.now().isoformat()
        }

        print_section_divider("Test Coverage Verification Complete")
        print_info(f"Tasks ({len(tasks)}):")
        print_info(f"  - With tests: {len(tasks_with_tests)}")
        print_info(f"  - Without tests: {len(tasks_without_tests)}")
        print_info(f"  - Coverage: {evidence['task_coverage_rate']:.1f}%")

        print_info(f"Bugs ({len(bugs)}):")
        print_info(f"  - With tests: {len(bugs_with_tests)}")
        print_info(f"  - Without tests: {len(bugs_without_tests)}")
        print_info(f"  - Coverage: {evidence['bug_coverage_rate']:.1f}%")

        print_info(f"Overall Coverage: {coverage_rate:.1f}%")

        if tasks_without_tests:
            print_warning("Tasks without test plans:")
            for item in tasks_without_tests:
                print_info(f"  - #{item['id']}: {item['title']}")

        if bugs_without_tests:
            print_warning("Bugs without regression tests:")
            for item in bugs_without_tests:
                print_info(f"  - #{item['id']}: {item['title']}")

        return evidence

    def _step_7_acceptance_tests(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 7: Run acceptance tests via tester agent.

        Spawns tester agent to run acceptance tests for completed work items.
        Agent runs in fresh context window with full codebase access.
        """
        print_step_header(7, "Run Acceptance Tests", "Tester agent runs acceptance tests for completed work")

        completed = self.step_evidence.get("2-query-completed", {})
        tasks = completed.get("completed_tasks", [])
        bugs = completed.get("completed_bugs", [])

        if not tasks and not bugs:
            print_success("No work items to test")
            return {
                "agent": "tester",
                "skipped": True,
                "reason": "No completed work items",
                "timestamp": datetime.now().isoformat()
            }

        # Build context for tester agent
        work_items_summary = []
        for task in tasks[:10]:  # Limit to first 10 for context
            task_id = task.get('id')
            task_title = get_work_item_field(task, 'title', 'Unknown')
            work_items_summary.append(f"- Task #{task_id}: {task_title}")

        for bug in bugs[:10]:
            bug_id = bug.get('id')
            bug_title = get_work_item_field(bug, 'title', 'Unknown')
            work_items_summary.append(f"- Bug #{bug_id}: {bug_title}")

        work_items_text = "\n".join(work_items_summary)

        print_section_divider("Requesting Tester Agent")
        print_info(f"Work items to test: {len(tasks)} tasks, {len(bugs)} bugs")

        # Agent prompt for tester
        agent_prompt = f"""You are the Tester agent for sprint review.

## Sprint: {self.sprint_name}

## Completed Work Items:
{work_items_text}

## Your Task:
1. Run acceptance tests for the completed work items
2. Execute: `pytest -m "acceptance or system" --tb=short -q`
3. Report test results summary
4. Identify any failing tests that block sprint closure

## Output Format:
Provide a structured summary:
- Tests run: [count]
- Tests passed: [count]
- Tests failed: [count]
- Blocking issues: [list any critical failures]
- Recommendation: PASS/FAIL for acceptance testing phase
"""

        # For non-interactive mode, provide mock response
        if not self.interactive:
            print_warning("Non-interactive mode - using mock acceptance test results")
            evidence = {
                "agent": "tester",
                "tests_run": len(tasks) + len(bugs),
                "tests_passed": len(tasks) + len(bugs),
                "tests_failed": 0,
                "blocking_issues": [],
                "recommendation": "PASS",
                "mock": True,
                "timestamp": datetime.now().isoformat()
            }
            print_success(f"Mock: {evidence['tests_run']} tests passed")
            return evidence

        # Interactive mode - would spawn actual tester agent
        try:
            if self.interactive_session and self.interactive_session.is_available():
                result = self.interactive_session.discuss(
                    initial_prompt=agent_prompt,
                    context={"sprint": self.sprint_name, "work_items": len(tasks) + len(bugs)},
                    max_iterations=2
                )
                evidence = {
                    "agent": "tester",
                    "response": result.get("final_response", ""),
                    "recommendation": "PASS" if "pass" in result.get("final_response", "").lower() else "REVIEW",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                evidence = {
                    "agent": "tester",
                    "tests_run": len(tasks) + len(bugs),
                    "tests_passed": len(tasks) + len(bugs),
                    "tests_failed": 0,
                    "recommendation": "PASS",
                    "mock": True,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            print_warning(f"Tester agent error: {e}")
            evidence = {
                "agent": "tester",
                "error": str(e),
                "recommendation": "REVIEW",
                "timestamp": datetime.now().isoformat()
            }

        print_success(f"Acceptance test phase: {evidence.get('recommendation', 'REVIEW')}")
        return evidence

    def _step_8_security_review(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 8: Security review via security-specialist agent.

        Spawns security-specialist agent to perform final security review
        of completed work items before sprint closure.
        """
        print_step_header(8, "Security Review", "Security specialist agent performs final security review")

        completed = self.step_evidence.get("2-query-completed", {})
        tasks = completed.get("completed_tasks", [])
        bugs = completed.get("completed_bugs", [])

        # Build work items summary for security review
        work_items_summary = []
        for task in tasks[:10]:
            task_id = task.get('id')
            task_title = get_work_item_field(task, 'title', 'Unknown')
            work_items_summary.append(f"- Task #{task_id}: {task_title}")

        for bug in bugs[:10]:
            bug_id = bug.get('id')
            bug_title = get_work_item_field(bug, 'title', 'Unknown')
            work_items_summary.append(f"- Bug #{bug_id}: {bug_title}")

        work_items_text = "\n".join(work_items_summary) if work_items_summary else "No work items"

        print_section_divider("Requesting Security Specialist")
        print_info(f"Items for security review: {len(tasks)} tasks, {len(bugs)} bugs")

        # Agent prompt for security specialist
        agent_prompt = f"""You are the Security Specialist agent for sprint review.

## Sprint: {self.sprint_name}

## Completed Work Items:
{work_items_text}

## Your Task:
1. Review completed work for security implications
2. Check for OWASP Top 10 vulnerabilities in new code
3. Verify authentication/authorization changes are secure
4. Check for sensitive data exposure risks
5. Review any security-related bugs that were fixed

## Output Format:
Provide a structured security assessment:
- Security issues found: [count]
- Critical issues: [list any blockers]
- Recommendations: [security improvements]
- Security clearance: APPROVED/NEEDS_REVIEW/BLOCKED
"""

        # For non-interactive mode, provide mock response
        if not self.interactive:
            print_warning("Non-interactive mode - using mock security review")
            evidence = {
                "agent": "security-specialist",
                "issues_found": 0,
                "critical_issues": [],
                "recommendations": ["Continue regular security scanning"],
                "clearance": "APPROVED",
                "mock": True,
                "timestamp": datetime.now().isoformat()
            }
            print_success(f"Mock security clearance: {evidence['clearance']}")
            return evidence

        # Interactive mode
        try:
            if self.interactive_session and self.interactive_session.is_available():
                result = self.interactive_session.discuss(
                    initial_prompt=agent_prompt,
                    context={"sprint": self.sprint_name},
                    max_iterations=2
                )
                clearance = "APPROVED"
                response_lower = result.get("final_response", "").lower()
                if "blocked" in response_lower or "critical" in response_lower:
                    clearance = "BLOCKED"
                elif "review" in response_lower:
                    clearance = "NEEDS_REVIEW"

                evidence = {
                    "agent": "security-specialist",
                    "response": result.get("final_response", ""),
                    "clearance": clearance,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                evidence = {
                    "agent": "security-specialist",
                    "issues_found": 0,
                    "clearance": "APPROVED",
                    "mock": True,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            print_warning(f"Security specialist error: {e}")
            evidence = {
                "agent": "security-specialist",
                "error": str(e),
                "clearance": "NEEDS_REVIEW",
                "timestamp": datetime.now().isoformat()
            }

        print_success(f"Security clearance: {evidence.get('clearance', 'NEEDS_REVIEW')}")
        return evidence

    def _step_9_deployment_readiness(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 9: Deployment readiness assessment via engineer agent.

        Spawns engineer agent to assess deployment readiness
        for the completed sprint work.
        """
        print_step_header(9, "Deployment Readiness Assessment", "Engineer agent assesses deployment readiness")

        completed = self.step_evidence.get("2-query-completed", {})
        test_verification = self.step_evidence.get("6-test-verification", {})
        acceptance_tests = self.step_evidence.get("7-acceptance-tests", {})
        security_review = self.step_evidence.get("8-security-review", {})

        tasks = completed.get("completed_tasks", [])
        bugs = completed.get("completed_bugs", [])

        print_section_divider("Requesting Engineer Deployment Assessment")

        # Agent prompt for engineer
        agent_prompt = f"""You are the Engineer agent for sprint review deployment assessment.

## Sprint: {self.sprint_name}

## Sprint Summary:
- Completed Tasks: {len(tasks)}
- Completed Bugs: {len(bugs)}

## Quality Gates Status:
- Test Coverage: {test_verification.get('overall_coverage_rate', 0):.1f}%
- Acceptance Tests: {acceptance_tests.get('recommendation', 'N/A')}
- Security Review: {security_review.get('clearance', 'N/A')}

## Your Task:
1. Assess deployment readiness based on quality gates
2. Check for any deployment blockers
3. Verify rollback procedures are in place
4. Confirm monitoring and alerting is ready

## Output Format:
Provide deployment assessment:
- Deployment readiness: READY/NOT_READY
- Quality gate summary: [pass/fail for each]
- Blockers: [list any deployment blockers]
- Recommendations: [deployment considerations]
"""

        # For non-interactive mode, provide mock response
        if not self.interactive:
            print_warning("Non-interactive mode - using mock deployment assessment")

            # Determine readiness based on previous steps
            acceptance_pass = acceptance_tests.get('recommendation') == 'PASS'
            security_pass = security_review.get('clearance') == 'APPROVED'
            coverage_ok = test_verification.get('overall_coverage_rate', 0) >= 50

            readiness = "READY" if (acceptance_pass and security_pass and coverage_ok) else "NOT_READY"

            evidence = {
                "agent": "engineer",
                "readiness": readiness,
                "quality_gates": {
                    "acceptance_tests": acceptance_pass,
                    "security_review": security_pass,
                    "test_coverage": coverage_ok
                },
                "blockers": [] if readiness == "READY" else ["Quality gates not fully passed"],
                "mock": True,
                "timestamp": datetime.now().isoformat()
            }
            print_success(f"Mock deployment readiness: {evidence['readiness']}")
            return evidence

        # Interactive mode
        try:
            if self.interactive_session and self.interactive_session.is_available():
                result = self.interactive_session.discuss(
                    initial_prompt=agent_prompt,
                    context={"sprint": self.sprint_name},
                    max_iterations=2
                )
                readiness = "READY"
                response_lower = result.get("final_response", "").lower()
                if "not ready" in response_lower or "blocked" in response_lower:
                    readiness = "NOT_READY"

                evidence = {
                    "agent": "engineer",
                    "response": result.get("final_response", ""),
                    "readiness": readiness,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                evidence = {
                    "agent": "engineer",
                    "readiness": "READY",
                    "mock": True,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            print_warning(f"Engineer agent error: {e}")
            evidence = {
                "agent": "engineer",
                "error": str(e),
                "readiness": "NOT_READY",
                "timestamp": datetime.now().isoformat()
            }

        print_success(f"Deployment readiness: {evidence.get('readiness', 'NOT_READY')}")
        return evidence

    def _step_10_closure_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 10: Sprint closure decision via scrum-master agent.

        Spawns scrum-master agent to recommend sprint closure decision
        based on all review steps.
        """
        print_step_header(10, "Sprint Closure Decision", "Scrum master recommends sprint closure decision")

        # Gather all evidence from previous steps
        completed = self.step_evidence.get("2-query-completed", {})
        approval = self.step_evidence.get("4-approval-gate", {})
        parent_completion = self.step_evidence.get("5-parent-completion", {})
        test_verification = self.step_evidence.get("6-test-verification", {})
        acceptance_tests = self.step_evidence.get("7-acceptance-tests", {})
        security_review = self.step_evidence.get("8-security-review", {})
        deployment = self.step_evidence.get("9-deployment-readiness", {})

        print_section_divider("Requesting Scrum Master Closure Recommendation")

        # Summarize sprint status
        summary = {
            "tasks_completed": len(completed.get("completed_tasks", [])),
            "bugs_completed": len(completed.get("completed_bugs", [])),
            "features_completed": parent_completion.get("features_completed", 0),
            "epics_completed": parent_completion.get("epics_completed", 0),
            "test_coverage": test_verification.get("overall_coverage_rate", 0),
            "acceptance_result": acceptance_tests.get("recommendation", "N/A"),
            "security_clearance": security_review.get("clearance", "N/A"),
            "deployment_readiness": deployment.get("readiness", "N/A")
        }

        # Agent prompt for scrum master
        agent_prompt = f"""You are the Scrum Master agent making the sprint closure decision.

## Sprint: {self.sprint_name}

## Sprint Summary:
- Tasks Completed: {summary['tasks_completed']}
- Bugs Completed: {summary['bugs_completed']}
- Features Completed: {summary['features_completed']}
- EPICs Completed: {summary['epics_completed']}

## Quality Gates:
- Test Coverage: {summary['test_coverage']:.1f}%
- Acceptance Tests: {summary['acceptance_result']}
- Security Clearance: {summary['security_clearance']}
- Deployment Readiness: {summary['deployment_readiness']}

## Your Task:
1. Evaluate overall sprint success
2. Determine if sprint should be closed
3. Identify any incomplete work to carry forward
4. Provide final recommendation

## Output Format:
Provide closure recommendation:
- Decision: CLOSE_SPRINT/EXTEND_SPRINT
- Sprint success rating: [1-5]
- Rationale: [brief explanation]
- Carry-forward items: [if any]
- Retrospective topics: [suggested topics for retro]
"""

        # For non-interactive mode, determine based on gates
        if not self.interactive:
            print_warning("Non-interactive mode - using mock closure decision")

            # Determine closure based on gates
            acceptance_pass = summary['acceptance_result'] == 'PASS'
            security_pass = summary['security_clearance'] == 'APPROVED'
            deployment_ready = summary['deployment_readiness'] == 'READY'

            can_close = acceptance_pass and security_pass and deployment_ready
            decision = "CLOSE_SPRINT" if can_close else "EXTEND_SPRINT"

            # Calculate success rating
            success_rating = 3  # Default
            if can_close and summary['test_coverage'] >= 80:
                success_rating = 5
            elif can_close and summary['test_coverage'] >= 60:
                success_rating = 4
            elif can_close:
                success_rating = 3
            else:
                success_rating = 2

            evidence = {
                "agent": "scrum-master",
                "decision": decision,
                "success_rating": success_rating,
                "rationale": "All quality gates passed" if can_close else "Quality gates incomplete",
                "carry_forward": [] if can_close else ["Review failing quality gates"],
                "retrospective_topics": [
                    "Test coverage improvements",
                    "Security review process",
                    "Deployment automation"
                ],
                "summary": summary,
                "mock": True,
                "timestamp": datetime.now().isoformat()
            }
            print_success(f"Mock closure decision: {evidence['decision']} (Rating: {success_rating}/5)")
            return evidence

        # Interactive mode
        try:
            if self.interactive_session and self.interactive_session.is_available():
                result = self.interactive_session.discuss(
                    initial_prompt=agent_prompt,
                    context={"sprint": self.sprint_name, "summary": summary},
                    max_iterations=2
                )
                decision = "CLOSE_SPRINT"
                response_lower = result.get("final_response", "").lower()
                if "extend" in response_lower:
                    decision = "EXTEND_SPRINT"

                evidence = {
                    "agent": "scrum-master",
                    "response": result.get("final_response", ""),
                    "decision": decision,
                    "summary": summary,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                evidence = {
                    "agent": "scrum-master",
                    "decision": "CLOSE_SPRINT",
                    "summary": summary,
                    "mock": True,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            print_warning(f"Scrum master agent error: {e}")
            evidence = {
                "agent": "scrum-master",
                "error": str(e),
                "decision": "EXTEND_SPRINT",
                "summary": summary,
                "timestamp": datetime.now().isoformat()
            }

        print_success(f"Scrum master recommendation: {evidence.get('decision', 'EXTEND_SPRINT')}")
        return evidence

    def _step_11_final_approval(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 11: Final human approval gate for sprint closure.

        BLOCKING approval gate - execution halts until user approves/rejects
        the sprint closure decision.
        """
        print_step_header(11, "Final Human Approval", "BLOCKING approval for sprint closure")

        from core.console_workflow import print_summary_panel, ApprovalGateData, print_approval_gate

        # Get scrum master recommendation
        closure_decision = self.step_evidence.get("10-closure-decision", {})
        recommendation = closure_decision.get("decision", "EXTEND_SPRINT")
        summary = closure_decision.get("summary", {})

        # Display sprint closure summary
        print_section_divider(f"Sprint Closure Summary: {self.sprint_name}")

        # Work completed summary
        work_completed = {
            "Tasks": summary.get('tasks_completed', 0),
            "Bugs": summary.get('bugs_completed', 0),
            "Features": summary.get('features_completed', 0),
            "EPICs": summary.get('epics_completed', 0)
        }
        print_summary_panel("📋 Work Completed", work_completed, style="success")

        # Quality gates summary
        quality_gates = {
            "Test Coverage": f"{summary.get('test_coverage', 0):.1f}%",
            "Acceptance Tests": summary.get('acceptance_result', 'N/A'),
            "Security Clearance": summary.get('security_clearance', 'N/A'),
            "Deployment Readiness": summary.get('deployment_readiness', 'N/A')
        }
        print_summary_panel("✓ Quality Gates", quality_gates, style="info")

        # Scrum Master recommendation
        print_info(f"🎯 Scrum Master Recommendation: {recommendation}")
        if closure_decision.get('rationale'):
            print_info(f"   Rationale: {closure_decision['rationale']}")

        # Display approval gate
        gate = ApprovalGateData(
            title="Sprint Closure Decision",
            summary=[
                f"Recommendation: {recommendation}",
                f"Sprint: {self.sprint_name}"
            ],
            options=[
                ("y", "CLOSE SPRINT (mark sprint as complete)"),
                ("e", "EXTEND SPRINT (continue sprint with incomplete work)"),
                ("n", "CANCEL (abort sprint review)")
            ],
            question="Enter choice (y/e/n):"
        )
        print_approval_gate(gate)

        # BLOCKING CALL
        while True:
            try:
                response = input("Enter choice (y/e/n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print_error("\nApproval cancelled by user")
                response = "n"
                break

            if response in ['y', 'yes']:
                response = 'close'
                break
            elif response in ['e', 'extend']:
                response = 'extend'
                break
            elif response in ['n', 'no', 'cancel']:
                response = 'cancel'
                break
            else:
                print_warning("Invalid choice. Enter 'y' to close, 'e' to extend, or 'n' to cancel.")

        evidence = {
            "response": response,
            "sprint_closed": response == "close",
            "sprint_extended": response == "extend",
            "cancelled": response == "cancel",
            "scrum_master_recommendation": recommendation,
            "user_agreed_with_recommendation": (
                (recommendation == "CLOSE_SPRINT" and response == "close") or
                (recommendation == "EXTEND_SPRINT" and response == "extend")
            ),
            "timestamp": datetime.now().isoformat()
        }

        if response == "close":
            print_success(f"SPRINT CLOSED: {self.sprint_name}")
            print_info("Sprint marked as complete. Proceeding to save report.")
        elif response == "extend":
            print_info(f"SPRINT EXTENDED: {self.sprint_name}")
            print_info("Sprint will continue. Incomplete work remains active.")
        else:
            print_error(f"SPRINT REVIEW CANCELLED")
            print_info("No changes made to sprint state.")

        return evidence

    def _step_12_save_report(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 12: Save sprint review report."""
        print_step_header(12, "Save Sprint Review Report", "Generate and save comprehensive report")

        completed = self.step_evidence.get("2-query-completed", {})
        demo_checklist = self.step_evidence.get("3-demo-checklist", {})
        parent_completion = self.step_evidence.get("5-parent-completion", {})
        test_verification = self.step_evidence.get("6-test-verification", {})
        acceptance_tests = self.step_evidence.get("7-acceptance-tests", {})
        security_review = self.step_evidence.get("8-security-review", {})
        deployment = self.step_evidence.get("9-deployment-readiness", {})
        closure_decision = self.step_evidence.get("10-closure-decision", {})
        final_approval = self.step_evidence.get("11-final-approval", {})

        # Generate report content
        report_lines = [
            f"# Sprint Review: {self.sprint_name}",
            "",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Workflow ID:** {self.workflow_id}",
            "",
            "## Sprint Work Summary",
            "",
            f"| Metric | Count |",
            f"|--------|-------|",
            f"| Completed Tasks | {completed.get('task_count', 0)} |",
            f"| Completed Bugs | {completed.get('bug_count', 0)} |",
            f"| **Total Work Items** | {completed.get('task_count', 0) + completed.get('bug_count', 0)} |",
            ""
        ]

        # List completed Tasks
        tasks = completed.get("completed_tasks", [])
        if tasks:
            report_lines.extend([
                "### Completed Tasks",
                ""
            ])
            for task in tasks:
                task_id = task.get('id')
                task_title = get_work_item_field(task, 'title', 'Unknown')
                report_lines.append(f"- **#{task_id}:** {task_title}")
            report_lines.append("")

        # List completed Bugs
        bugs = completed.get("completed_bugs", [])
        if bugs:
            report_lines.extend([
                "### Completed Bug Fixes",
                ""
            ])
            for bug in bugs:
                bug_id = bug.get('id')
                bug_title = get_work_item_field(bug, 'title', 'Unknown')
                report_lines.append(f"- **#{bug_id}:** {bug_title}")
            report_lines.append("")

        # Add demo checklist
        checklist = demo_checklist.get("checklist_items", [])
        if checklist:
            report_lines.extend([
                "## Demo Checklist",
                ""
            ])

            # Group by type
            task_items = [item for item in checklist if item.get('work_item_type') == 'Task']
            bug_items = [item for item in checklist if item.get('work_item_type') == 'Bug']

            if task_items:
                report_lines.append("### Task Demonstrations")
                report_lines.append("")
                for item in task_items:
                    report_lines.append(f"#### #{item.get('work_item_id')}: {item.get('work_item_title', 'Unknown')}")
                    report_lines.append("")
                    for point in item.get('demo_points', []):
                        report_lines.append(f"- {point}")
                    report_lines.append("")

            if bug_items:
                report_lines.append("### Bug Fix Demonstrations")
                report_lines.append("")
                for item in bug_items:
                    report_lines.append(f"#### #{item.get('work_item_id')}: {item.get('work_item_title', 'Unknown')}")
                    report_lines.append("")
                    for point in item.get('demo_points', []):
                        report_lines.append(f"- {point}")
                    report_lines.append("")

        # Add parent completion results
        if not parent_completion.get("skipped"):
            # Feature completions
            feature_completions = parent_completion.get("feature_completions", [])
            features_completed = [f for f in feature_completions if f.get('success')]

            if features_completed:
                report_lines.extend([
                    "## Completed Features",
                    "",
                    "The following Features were marked Done (all child Tasks/Bugs completed):",
                    ""
                ])
                for feature in features_completed:
                    report_lines.append(f"- **Feature #{feature['feature_id']}:** {feature.get('title', 'Unknown')}")
                report_lines.append("")

            # EPIC completions
            epic_completions = parent_completion.get("epic_completions", [])
            epics_completed = [e for e in epic_completions if e.get('success')]

            if epics_completed:
                report_lines.extend([
                    "## Completed EPICs",
                    "",
                    "The following EPICs were marked Done (all child Features completed):",
                    ""
                ])
                for epic in epics_completed:
                    report_lines.append(f"- **EPIC #{epic['epic_id']}:** {epic.get('title', 'Unknown')}")
                report_lines.append("")

        # Add test verification
        report_lines.extend([
            "## Test Coverage",
            "",
            f"| Category | With Tests | Without Tests | Coverage |",
            f"|----------|------------|---------------|----------|",
            f"| Tasks | {len(test_verification.get('tasks_with_tests', []))} | {len(test_verification.get('tasks_without_tests', []))} | {test_verification.get('task_coverage_rate', 0):.1f}% |",
            f"| Bugs | {len(test_verification.get('bugs_with_tests', []))} | {len(test_verification.get('bugs_without_tests', []))} | {test_verification.get('bug_coverage_rate', 0):.1f}% |",
            f"| **Overall** | - | - | **{test_verification.get('overall_coverage_rate', 0):.1f}%** |",
            ""
        ])

        tasks_without_tests = test_verification.get("tasks_without_tests", [])
        bugs_without_tests = test_verification.get("bugs_without_tests", [])

        if tasks_without_tests or bugs_without_tests:
            report_lines.extend([
                "### Test Coverage Warnings",
                ""
            ])

            if tasks_without_tests:
                report_lines.append("**Tasks without test plans:**")
                report_lines.append("")
                for item in tasks_without_tests:
                    report_lines.append(f"- #{item['id']}: {item['title']}")
                report_lines.append("")

            if bugs_without_tests:
                report_lines.append("**Bugs without regression tests:**")
                report_lines.append("")
                for item in bugs_without_tests:
                    report_lines.append(f"- #{item['id']}: {item['title']}")
                report_lines.append("")

        # Add Quality Gates Summary
        report_lines.extend([
            "## Quality Gates Summary",
            "",
            "| Gate | Status | Agent |",
            "|------|--------|-------|",
            f"| Acceptance Tests | {acceptance_tests.get('recommendation', 'N/A')} | Tester |",
            f"| Security Review | {security_review.get('clearance', 'N/A')} | Security Specialist |",
            f"| Deployment Readiness | {deployment.get('readiness', 'N/A')} | Engineer |",
            ""
        ])

        # Add Agent Review Details
        report_lines.extend([
            "### Acceptance Tests (Tester Agent)",
            ""
        ])
        if acceptance_tests.get('skipped'):
            report_lines.append("*Skipped - no work items to test*")
        else:
            report_lines.append(f"- **Recommendation:** {acceptance_tests.get('recommendation', 'N/A')}")
            if acceptance_tests.get('tests_run'):
                report_lines.append(f"- **Tests Run:** {acceptance_tests.get('tests_run', 0)}")
                report_lines.append(f"- **Tests Passed:** {acceptance_tests.get('tests_passed', 0)}")
                report_lines.append(f"- **Tests Failed:** {acceptance_tests.get('tests_failed', 0)}")
            if acceptance_tests.get('blocking_issues'):
                report_lines.append(f"- **Blocking Issues:** {', '.join(acceptance_tests.get('blocking_issues', []))}")
        report_lines.append("")

        report_lines.extend([
            "### Security Review (Security Specialist Agent)",
            ""
        ])
        report_lines.append(f"- **Clearance:** {security_review.get('clearance', 'N/A')}")
        if security_review.get('issues_found') is not None:
            report_lines.append(f"- **Issues Found:** {security_review.get('issues_found', 0)}")
        if security_review.get('critical_issues'):
            report_lines.append(f"- **Critical Issues:** {', '.join(security_review.get('critical_issues', []))}")
        if security_review.get('recommendations'):
            report_lines.append("- **Recommendations:**")
            for rec in security_review.get('recommendations', []):
                report_lines.append(f"  - {rec}")
        report_lines.append("")

        report_lines.extend([
            "### Deployment Readiness (Engineer Agent)",
            ""
        ])
        report_lines.append(f"- **Readiness:** {deployment.get('readiness', 'N/A')}")
        if deployment.get('quality_gates'):
            report_lines.append("- **Quality Gate Details:**")
            gates = deployment.get('quality_gates', {})
            for gate, passed in gates.items():
                status = "PASS" if passed else "FAIL"
                report_lines.append(f"  - {gate}: {status}")
        if deployment.get('blockers'):
            report_lines.append("- **Blockers:**")
            for blocker in deployment.get('blockers', []):
                report_lines.append(f"  - {blocker}")
        report_lines.append("")

        # Add Sprint Closure Decision
        report_lines.extend([
            "## Sprint Closure Decision",
            ""
        ])
        report_lines.append(f"**Scrum Master Recommendation:** {closure_decision.get('decision', 'N/A')}")
        if closure_decision.get('success_rating'):
            report_lines.append(f"**Success Rating:** {closure_decision.get('success_rating')}/5")
        if closure_decision.get('rationale'):
            report_lines.append(f"**Rationale:** {closure_decision.get('rationale')}")
        report_lines.append("")

        if closure_decision.get('retrospective_topics'):
            report_lines.append("**Suggested Retrospective Topics:**")
            for topic in closure_decision.get('retrospective_topics', []):
                report_lines.append(f"- {topic}")
            report_lines.append("")

        # Add Final Approval Decision
        report_lines.extend([
            "## Final Approval",
            ""
        ])
        if final_approval.get('sprint_closed'):
            report_lines.append("**Decision:** SPRINT CLOSED")
        elif final_approval.get('sprint_extended'):
            report_lines.append("**Decision:** SPRINT EXTENDED")
        elif final_approval.get('cancelled'):
            report_lines.append("**Decision:** REVIEW CANCELLED")
        else:
            report_lines.append("**Decision:** Pending")

        if final_approval.get('user_agreed_with_recommendation') is not None:
            agreed = "Yes" if final_approval.get('user_agreed_with_recommendation') else "No"
            report_lines.append(f"**User Agreed with Scrum Master:** {agreed}")
        report_lines.append("")

        # Add footer
        report_lines.extend([
            "---",
            "",
            "*Generated by Trustable AI Development Workbench*"
        ])

        report_content = "\n".join(report_lines)

        # Save report to .claude/sprint-reviews/
        report_dir = Path(".claude/sprint-reviews")
        report_dir.mkdir(parents=True, exist_ok=True)

        safe_sprint_name = self.sprint_name.lower().replace(' ', '-')
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        report_file = report_dir / f"{safe_sprint_name}-{timestamp}.md"

        # Write with UTF-8 encoding
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        evidence = {
            "report_file": str(report_file),
            "saved_at": datetime.now().isoformat()
        }

        print_success(f"Report saved: {report_file}")
        return evidence

    def _step_13_checkpoint(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 13: Save final checkpoint."""
        print_step_header(13, "Save Checkpoint", "Save workflow state")

        # State is automatically saved by WorkflowOrchestrator
        # This step just confirms completion

        evidence = {
            "checkpoint_saved": True,
            "timestamp": datetime.now().isoformat(),
            "workflow_complete": True
        }

        print_success("Checkpoint saved")
        return evidence


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sprint Review Workflow - External enforcement with EPIC auto-completion"
    )
    parser.add_argument(
        "--sprint",
        required=True,
        help="Sprint name to review (e.g., 'Sprint 7')"
    )
    parser.add_argument(
        "--workflow-id",
        help="Workflow ID (defaults to sprint-based ID)"
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Disable state checkpointing"
    )
    parser.add_argument(
        "--use-ai",
        action="store_true",
        help="Use AI for demo checklist generation (Mode 2)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Use interactive AI collaboration (Mode 3) - requires claude-agent-sdk and KEYCHAIN_ANTHROPIC_API_KEY"
    )

    args = parser.parse_args()

    # Normalize sprint name (e.g., "9" → "Sprint 9")
    sprint_name = normalize_sprint_name(args.sprint)
    if sprint_name != args.sprint:
        print_info(f"Normalized sprint name: '{args.sprint}' → '{sprint_name}'")

    # Generate workflow ID with timestamp (prevents caching external state across runs)
    workflow_id = args.workflow_id or f"review-{sprint_name.lower().replace(' ', '-')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Print workflow header
    mode = "Mode 3: Interactive AI" if args.interactive else ("Mode 2: AI + JSON" if args.use_ai else "Mode 1: Pure Python")
    print_workflow_header("Sprint Review Workflow", sprint_name=sprint_name, mode=mode)

    # Create and execute workflow
    workflow = SprintReviewWorkflow(
        sprint_name=sprint_name,
        workflow_id=workflow_id,
        enable_checkpoints=not args.no_checkpoints,
        use_ai=args.use_ai,
        interactive=args.interactive
    )

    try:
        success = workflow.execute()
        if success:
            console.print()
            console.print("─" * 80)
            console.print("[bold #71E4D1]  Sprint review complete![/bold #71E4D1]")
            console.print("─" * 80)
            console.print()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        console.print()
        console.print("[#758B9B]Sprint review cancelled by user.[/#758B9B]")
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
