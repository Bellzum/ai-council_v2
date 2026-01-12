#!/usr/bin/env python3
"""
Backlog Grooming Workflow with External Enforcement

Implements Phase 3: Medium Workflows - Backlog Grooming Script

7-Step Workflow:
1. Query Backlog - Query all backlog items (Features, Tasks)
2. Hierarchy Validation - Find orphan Features, empty EPICs
3. AI Refinement Analysis - Analyze Features for readiness (Mode 2: AI + JSON validation)
4. Approval Gate - BLOCKING approval for state transitions
5. State Transitions - Transition ready Features from New → Ready
6. External Verification - Verify state changes in tracking system
7. Save Grooming Report - Generate and save report to .claude/grooming/

Design Pattern:
- Extends WorkflowOrchestrator from Phase 1
- Uses adapter for ALL work item operations (NO `az boards` CLI)
- External verification after state transitions
- Real input() blocking for approval gate
- UTF-8 encoding for all file writes
- Hierarchy validation using validators.py utilities
"""

import argparse
import concurrent.futures
import json
import os
import select
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


def flush_stdin():
    """
    Flush any buffered input from stdin.

    This prevents accidental input during long-running operations
    from being consumed by subsequent input() calls.
    """
    try:
        # Unix/Linux/Mac: use select to check if there's input available
        if hasattr(select, 'select'):
            while select.select([sys.stdin], [], [], 0.0)[0]:
                sys.stdin.read(1)
    except Exception:
        # On Windows or if select fails, try alternative approach
        try:
            import msvcrt
            while msvcrt.kbhit():
                msvcrt.getch()
        except ImportError:
            # Neither select nor msvcrt available, skip flushing
            pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.workflow_executor.base import WorkflowOrchestrator, ExecutionMode
from scripts.workflow_executor.schemas import StepType
from scripts.workflow_executor.validators import (
    find_orphan_features,
    find_empty_epics,
    validate_feature_readiness
)
from scripts.workflow_executor.agent_prompts import (
    build_epic_breakdown_prompt,
    build_feature_conformance_prompt,
    validate_epic_breakdown_response,
    validate_feature_conformance_response,
)
from scripts.workflow_executor.consensus import (
    ConsensusOrchestrator,
    AgentInterface,
    ConsensusResult,
)
from scripts.workflow_executor.consensus_configs import (
    EPIC_TO_FEATURE_CONFIG,
    FEATURE_TO_TASK_CONFIG,
)


class BacklogGroomingWorkflow(WorkflowOrchestrator):
    """
    Backlog Grooming workflow with external enforcement.

    Implements the 7-step backlog grooming process with:
    - Hierarchy validation (orphan Features, empty EPICs)
    - Mode 2 AI refinement analysis
    - External verification after state changes
    - Blocking approval gates
    """

    # ========================================================================
    # Decomposition Requirements Constants
    # ========================================================================
    # Goal: Make all tickets ready for sprint planning with sufficient context
    # for the next agent to understand and execute the work.
    #
    # Reference: .claude/commands/backlog-grooming.md "Decomposition Requirements"

    # EPIC requirements - Business context for strategic planning
    EPIC_MIN_TOTAL_HOURS = 50 # EPICs must have Features totaling >= 50 hours
    EPIC_REQUIRED_CONTENT = [
        "business_analysis", # Business value, ROI, stakeholder impact
        "success_criteria", # Measurable outcomes
        "scope_definition" # What's in/out of scope
    ]

    # Feature requirements - Minimum 50 hours (~10 story points) per Feature
    # Features should be cohesive, measurable, testable, independently deliverable
    FEATURE_MIN_STORY_POINTS = 10 # Minimum 10 pts per Feature (~50 hours)
    FEATURE_MAX_STORY_POINTS = 30 # Typically 10-30 pts per Feature
    FEATURE_MIN_TASKS = 1 # Minimum 1 task per Feature (merged impl+test)
    FEATURE_MAX_TASKS = 5 # Maximum 5 tasks per Feature
    FEATURE_REQUIRED_TASKS = {
        "implementation": 1 # Single Implementation Task with test plans attached
        # Note: Testing Task is now merged into Implementation Task
        # The Tester agent validates during sprint execution, not as separate task
    }

    # Content attachment threshold - if description exceeds this, save as attachment
    CONTENT_ATTACHMENT_THRESHOLD = 2000 # characters
    FEATURE_REQUIRED_CONTENT = [
        "detailed_description", # Full scope and requirements
        "architecture_analysis", # Technical design and approach
        "security_analysis", # Security considerations and mitigations
        "acceptance_criteria" # Clear, testable acceptance criteria
    ]

    # Task story point range
    TASK_MIN_STORY_POINTS = 1 # Minimum 1 pt per Task
    TASK_MAX_STORY_POINTS = 5 # Maximum 5 pts per Task

    # Implementation Task requirements - Full context for engineer + test plans for tester
    # Note: Testing Task is merged into Implementation Task - all test plans are attached
    IMPLEMENTATION_TASK_REQUIRED_CONTENT = [
        # Implementation content (for Engineer agent)
        "detailed_design", # Implementation approach and decisions
        "function_specifications", # Detailed specs of each testable component
        # Test plans (attached as files for Tester agent validation)
        "unit_test_plan", # Unit test plan with 80% minimum coverage target
        "integration_test_plan", # Integration test plan for component interactions
        "edge_case_test_plan", # Edge-case/whitebox test plan
        "acceptance_test_plan", # Acceptance test plan for each criterion
        "falsifiability_requirements", # How tests detect actual failures
        "evidence_requirements" # Evidence of complete implementation
    ]

    # Test plan fields that should be attached as files (not inline in description)
    TEST_PLAN_ATTACHMENT_FIELDS = [
        "unit_test_plan",
        "integration_test_plan",
        "edge_case_test_plan",
        "acceptance_test_plan"
    ]

    # Deprecated: Testing Task merged into Implementation Task
    # Kept for backward compatibility with existing work items
    TESTING_TASK_REQUIRED_CONTENT = [
        "test_type_validation",
        "coverage_validation",
        "feature_coverage_validation",
        "falsifiability_validation",
        "test_results_report"
    ]

    # Legacy compatibility - keep for existing code
    TASK_REQUIRED_CONTENT = [
        "detailed_design", # Implementation approach and decisions
        "unit_test_design", # Unit test cases and coverage
        "integration_test_design", # Integration test scenarios
        "acceptance_test_design" # Acceptance test criteria
    ]

    # Bug requirements - Full context for diagnosis and fix
    BUG_REQUIRED_CONTENT = [
        "reproduction_steps", # Step-by-step repro instructions
        "root_cause_analysis", # Why the bug exists
        "solution_design", # How to fix it
        "acceptance_test_design" # How to verify the fix
    ]

    def __init__(
        self,
        workflow_id: str,
        enable_checkpoints: bool = True,
        use_ai: bool = True,
        interactive: bool = True,
        auto_conform: bool = False,
        use_agents: bool = False,
        use_consensus: bool = False,
        args: Optional[argparse.Namespace] = None
    ):
        """
        Initialize backlog grooming workflow.

        Args:
            workflow_id: Unique ID for this execution (e.g., "grooming-2024-12-21")
            enable_checkpoints: Enable state checkpointing
            use_ai: If True (default), use AI for analysis (Mode 2)
            interactive: If True (default), use Mode 3 interactive collaboration
            auto_conform: If True, automatically add missing fields to conform to criteria
            use_agents: If True, use tree-based agent spawning (parallel, context-isolated)
            use_consensus: If True, use multi-agent consensus for Epic/Feature breakdown
            args: Command-line arguments for non-interactive mode (Task #1243)
        """
        self.use_ai = use_ai
        self.interactive = interactive
        self.auto_conform = auto_conform
        self.use_agents = use_agents
        self.use_consensus = use_consensus
        self.args = args

        # Interactive mode overrides use_ai (Mode 3 vs Mode 2)
        if interactive and use_ai:
            self._print_verbose("ℹ Both AI and interactive enabled - using interactive mode (Mode 3)")
            self.use_ai = False # Interactive mode takes precedence

        mode = ExecutionMode.INTERACTIVE_AI if interactive else (ExecutionMode.AI_JSON_VALIDATION if use_ai else ExecutionMode.PURE_PYTHON)

        # Determine quiet mode from args (verbose = not quiet)
        verbose = getattr(args, 'verbose', False) if args else False

        super().__init__(
            workflow_name="backlog-grooming",
            workflow_id=workflow_id,
            mode=mode,
            enable_checkpoints=enable_checkpoints,
            quiet_mode=not verbose
        )

        # Initialize adapter
        try:
            sys.path.insert(0, '.claude/skills')
            from work_tracking import get_adapter
            self.adapter = get_adapter()
        except Exception as e:
            print(f"⚠ Warning: Could not initialize adapter: {e}")
            print(" Continuing with limited functionality...")
            self.adapter = None

        # Get current user for work item assignment
        self.current_user = None
        if self.adapter:
            try:
                user_info = self.adapter.get_current_user()
                if user_info:
                    self.current_user = user_info.get('display_name') or user_info.get('email')
                    self._print_verbose(f"✓ Authenticated as: {self.current_user}")
            except Exception as e:
                print(f"⚠ Could not get current user: {e}")
                print(" Work items will be created without assignment")

        # Initialize interactive session if AI or interactive mode is enabled
        # (AI content generation methods require the interactive session)
        self.interactive_session = None
        if use_ai or interactive:
            self._init_interactive_session()

        # Configuration for Feature readiness (legacy, kept for compatibility)
        self.min_description_length = 500 # characters
        self.min_acceptance_criteria = 3 # criteria count

    def _init_interactive_session(self):
        """Initialize interactive session with lazy SDK loading."""
        try:
            from scripts.workflow_executor.interactive_session import InteractiveSession
            self.interactive_session = InteractiveSession(
                workflow_name="backlog-grooming",
                session_id="grooming",
                model="claude-sonnet-4-5",
                max_tokens=16000 # Large for comprehensive implementation task descriptions (8 sections)
            )
            if self.interactive_session.is_available():
                self._print_verbose("✓ Interactive mode initialized (Mode 3)")
            else:
                print("⚠ Interactive mode unavailable - falling back to AI mode (Mode 2)")
                self.interactive = False
                self.use_ai = True
        except ImportError as e:
            print(f"⚠ Interactive mode unavailable: {e}")
            print(" Falling back to AI mode (Mode 2)")
            self.interactive = False
            self.use_ai = True

    @property
    def verbose(self) -> bool:
        """Check if verbose output is enabled."""
        return getattr(self.args, 'verbose', False) if self.args else False

    def _print_verbose(self, message: str):
        """Print message only if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _truncate_content(self, content: str, max_length: int = 150) -> str:
        """Truncate content for display with ellipsis."""
        if not content:
            return "(empty)"
        # Clean up whitespace and newlines for single-line display
        clean = ' '.join(content.split())
        if len(clean) <= max_length:
            return clean
        return clean[:max_length - 3] + "..."

    def _format_content_preview(self, content: Dict[str, Any], indent: str = " ") -> List[str]:
        """Format content dictionary for preview display."""
        lines = []
        for key, value in content.items():
            if value:
                # Convert key from snake_case to Title Case
                display_key = key.replace('_', ' ').title()
                preview = self._truncate_content(str(value), 100)
                lines.append(f"{indent}• {display_key}: {preview}")
        return lines

    def _define_steps(self) -> List[Dict[str, Any]]:
        """Define the 7 workflow steps."""
        return [
            {
                "id": "1-query-backlog",
                "name": "Query Backlog Items",
                "step_type": StepType.DATA_COLLECTION,
                "description": "Query all backlog items (Features, Tasks)",
                "required": True
            },
            {
                "id": "2-hierarchy-validation",
                "name": "Hierarchy Validation",
                "step_type": StepType.VERIFICATION,
                "description": "Find orphan Features and empty EPICs",
                "required": True,
                "depends_on": ["1-query-backlog"]
            },
            {
                "id": "3-ai-refinement",
                "name": "AI Refinement Analysis",
                "step_type": StepType.AI_REVIEW,
                "description": "Analyze Features for readiness using AI",
                "required": True,
                "depends_on": ["2-hierarchy-validation"]
            },
            {
                "id": "4-approval-gate",
                "name": "Human Approval Gate",
                "step_type": StepType.APPROVAL_GATE,
                "description": "BLOCKING approval for state transitions",
                "required": True,
                "depends_on": ["3-ai-refinement"]
            },
            {
                "id": "5-state-transitions",
                "name": "State Transitions",
                "step_type": StepType.ACTION,
                "description": "Transition ready Features from New → Ready",
                "required": True,
                "depends_on": ["4-approval-gate"]
            },
            {
                "id": "6-verification",
                "name": "External Verification",
                "step_type": StepType.VERIFICATION,
                "description": "Verify state changes in tracking system",
                "required": True,
                "depends_on": ["5-state-transitions"]
            },
            {
                "id": "7-save-report",
                "name": "Save Grooming Report",
                "step_type": StepType.ACTION,
                "description": "Generate and save report to .claude/grooming/",
                "required": True,
                "depends_on": ["6-verification"]
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
        if step_id == "1-query-backlog":
            return self._step_1_query_backlog(context)
        elif step_id == "2-hierarchy-validation":
            return self._step_2_hierarchy_validation(context)
        elif step_id == "3-ai-refinement":
            return self._step_3_ai_refinement(context)
        elif step_id == "4-approval-gate":
            return self._step_4_approval_gate(context)
        elif step_id == "5-state-transitions":
            return self._step_5_state_transitions(context)
        elif step_id == "6-verification":
            return self._step_6_verification(context)
        elif step_id == "7-save-report":
            return self._step_7_save_report(context)
        else:
            raise ValueError(f"Unknown step: {step_id}")

    # ========================================================================
    # Interactive Mode 3 Methods (Feature #1216)
    # ========================================================================

    def _epic_breakdown_interactive(self, epics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Interactive EPIC breakdown with AI (Mode 3).

        User collaborates with AI to decompose EPICs into well-structured Features.

        Returns:
            EPIC breakdown with feature suggestions
        """
        if not self.interactive_session or not self.interactive_session.is_available():
            return {"epics": epics, "skipped": True}

        context = {
            "epic_count": len(epics),
            "epics": [e.get("title", "Unknown") for e in epics[:3]]
        }

        initial_prompt = f"""You are helping break down {len(epics)} EPIC(s) into well-structured Features.

For each EPIC, suggest Features that are:
- Independently valuable
- Completable within a sprint
- Have clear acceptance criteria"""

        try:
            result = self.interactive_session.discuss(
                initial_prompt=initial_prompt,
                context=context,
                max_iterations=3
            )
            return {"epics": epics, "breakdown": result.get("final_response")}
        except Exception as e:
            print(f"⚠ Interactive breakdown failed: {e}")
            return {"epics": epics, "skipped": True}

    def _readiness_assessment_interactive(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Interactive feature readiness assessment with AI (Mode 3).

        User collaborates with AI to assess which Features are ready for sprint planning.

        Returns:
            Readiness assessment with recommendations
        """
        if not self.interactive_session or not self.interactive_session.is_available():
            return {"features": features, "skipped": True}

        context = {
            "feature_count": len(features),
            "features": [f.get("title", "Unknown") for f in features[:3]]
        }

        initial_prompt = f"""You are assessing readiness of {len(features)} Feature(s) for sprint planning.

For each Feature, evaluate:
- Description completeness (recommend minimum 500 characters)
- Acceptance criteria clarity (recommend minimum 3 criteria)
- Technical dependencies
- Business value clarity"""

        try:
            result = self.interactive_session.discuss(
                initial_prompt=initial_prompt,
                context=context,
                max_iterations=3
            )
            return {"features": features, "assessment": result.get("final_response")}
        except Exception as e:
            print(f"⚠ Interactive assessment failed: {e}")
            return {"features": features, "skipped": True}

    # ========================================================================
    # Acceptance Criteria Validation Methods
    # ========================================================================

    def _validate_epic(self, epic: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate an EPIC against acceptance criteria.

        Requirements:
        - Must have Features totaling >= 50 hours of work
        - Must have business analysis (value, ROI, stakeholder impact)
        - Must have success criteria (measurable outcomes)
        - Must have scope definition (what's in/out)

        Returns:
            {
                "valid": bool,
                "missing": List[str],
                "missing_content": List[str],
                "children": List[Dict],
                "total_hours": float
            }
        """
        epic_id = epic.get('id') or epic.get('fields', {}).get('System.Id')
        fields = epic.get('fields', {})
        description = (fields.get('System.Description', '') or '').lower()
        missing = []
        missing_content = []
        children = []
        total_hours = 0

        # Check for business analysis content
        has_business_analysis = any(kw in description for kw in [
            'business value', 'roi', 'return on investment', 'stakeholder',
            'business impact', 'business case', 'value proposition',
            'market', 'revenue', 'cost saving', 'efficiency'
        ])
        if not has_business_analysis:
            missing_content.append("business_analysis")
            missing.append("business_analysis: No business value/ROI analysis found")

        # Check for success criteria
        has_success_criteria = any(kw in description for kw in [
            'success criteria', 'success metric', 'kpi', 'measurable outcome',
            'definition of done', 'acceptance criteria', 'target metric',
            'goal:', 'objective:'
        ])
        if not has_success_criteria:
            missing_content.append("success_criteria")
            missing.append("success_criteria: No measurable success criteria found")

        # Check for scope definition
        has_scope = any(kw in description for kw in [
            'scope', 'in scope', 'out of scope', 'included', 'excluded',
            'boundary', 'deliverable', 'not included'
        ])
        if not has_scope:
            missing_content.append("scope_definition")
            missing.append("scope_definition: No scope definition found")

        # Query child Features for total hours
        if self.adapter:
            try:
                all_features = self.adapter.query_work_items(work_item_type='Feature')
                for feature in all_features:
                    parent_id = feature.get('fields', {}).get('System.Parent')
                    if parent_id and str(parent_id) == str(epic_id):
                        children.append(feature)
                        story_points = feature.get('fields', {}).get(
                            'Microsoft.VSTS.Scheduling.StoryPoints', 0
                        ) or 0
                        total_hours += story_points * 8
            except Exception as e:
                print(f" ⚠ Error querying child Features: {e}")

        if total_hours < self.EPIC_MIN_TOTAL_HOURS:
            missing.append(
                f"insufficient_work: {total_hours}h of {self.EPIC_MIN_TOTAL_HOURS}h required"
            )

        if not children:
            missing.append("no_features: No child Features found")

        return {
            "valid": len(missing) == 0,
            "missing": missing,
            "missing_content": missing_content,
            "children": children,
            "total_hours": total_hours
        }

    def _validate_feature(self, feature: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a Feature against acceptance criteria.

        Requirements:
        - Detailed description (full scope and requirements)
        - Architecture analysis (technical design and approach)
        - Security analysis (security considerations and mitigations)
        - Acceptance criteria (clear, testable criteria)
        - Exactly 1 implementation Task (testing merged into implementation)

        Note: Testing Tasks are no longer required as separate items.
        The Tester agent validates during sprint execution.

        Returns:
            {
                "valid": bool,
                "missing": List[str],
                "missing_content": List[str],
                "implementation_tasks": List[Dict]
            }
        """
        feature_id = feature.get('id') or feature.get('fields', {}).get('System.Id')
        fields = feature.get('fields', {})
        description = (fields.get('System.Description', '') or '').lower()
        acceptance_criteria = (fields.get('Microsoft.VSTS.Common.AcceptanceCriteria', '') or '').lower()
        missing = []
        missing_content = []

        # Check for detailed description (substantial content)
        has_detailed_description = (
            len(description) > 300 and
            any(kw in description for kw in [
                'requirement', 'scope', 'goal', 'objective', 'user story',
                'functional', 'feature', 'capability', 'must', 'should'
            ])
        )
        if not has_detailed_description:
            missing_content.append("detailed_description")
            missing.append("detailed_description: Insufficient detail for implementation")

        # Check for architecture analysis
        has_architecture = any(kw in description for kw in [
            'architecture', 'design', 'technical approach', 'component',
            'api', 'database', 'service', 'interface', 'integration',
            'data flow', 'system design', 'module'
        ])
        if not has_architecture:
            missing_content.append("architecture_analysis")
            missing.append("architecture_analysis: No technical design found")

        # Check for security analysis
        has_security = any(kw in description for kw in [
            'security', 'authentication', 'authorization', 'encryption',
            'vulnerability', 'threat', 'risk', 'access control', 'audit',
            'compliance', 'privacy', 'data protection', 'injection', 'xss'
        ])
        if not has_security:
            missing_content.append("security_analysis")
            missing.append("security_analysis: No security considerations found")

        # Check for acceptance criteria
        has_acceptance = (
            len(acceptance_criteria) > 100 or
            any(kw in description for kw in [
                'acceptance criteria', 'acceptance test', 'test case',
                'given', 'when', 'then', 'verify', 'validate'
            ])
        )
        if not has_acceptance:
            missing_content.append("acceptance_criteria")
            missing.append("acceptance_criteria: No testable acceptance criteria found")

        # Check story points (minimum 10 per Feature)
        story_points = fields.get('Microsoft.VSTS.Scheduling.StoryPoints', 0) or 0
        if story_points < self.FEATURE_MIN_STORY_POINTS:
            missing.append(
                f"story_points: Feature has {story_points} pts, "
                f"minimum is {self.FEATURE_MIN_STORY_POINTS} pts (~50 hours)"
            )

        # Query child Tasks
        # Note: Testing is now merged into Implementation Task, so we count all child tasks
        # as implementation tasks. The Tester agent validates during sprint execution.
        implementation_tasks = []

        if self.adapter:
            try:
                all_tasks = self.adapter.query_work_items(work_item_type='Task')
                for task in all_tasks:
                    parent_id = task.get('fields', {}).get('System.Parent')
                    if parent_id and str(parent_id) == str(feature_id):
                        implementation_tasks.append(task)
            except Exception as e:
                print(f" ⚠ Error querying child Tasks: {e}")

        # Check task counts - only implementation tasks required
        # Testing is handled by Tester agent during sprint execution, not as separate task
        if len(implementation_tasks) != self.FEATURE_REQUIRED_TASKS["implementation"]:
            missing.append(
                f"implementation_task: Need {self.FEATURE_REQUIRED_TASKS['implementation']} "
                f"implementation task(s), found {len(implementation_tasks)}"
            )

        return {
            "valid": len(missing) == 0,
            "missing": missing,
            "missing_content": missing_content,
            "implementation_tasks": implementation_tasks
        }

    def _get_parent_feature_context(self, parent_id: int) -> Dict[str, Any]:
        """
        Fetch full parent Feature context including all fields and attachments.

        This provides comprehensive context for AI agents generating Task/Bug content,
        ensuring they have access to:
        - All Feature fields (title, description, acceptance criteria, etc.)
        - Architecture and security analysis from the parent Feature
        - File attachments that may contain design documents, specs, etc.

        Args:
            parent_id: Work item ID of the parent Feature

        Returns:
            Dict with parent context:
            {
                "found": bool,
                "title": str,
                "description": str,
                "acceptance_criteria": str,
                "architecture_analysis": str,
                "security_analysis": str,
                "story_points": int,
                "state": str,
                "tags": str,
                "attachments": List[Dict] - [{name, url, comment}]
            }
        """
        if not self.adapter or not parent_id:
            return {"found": False}

        try:
            parent = self.adapter.get_work_item(parent_id)
            if not parent:
                return {"found": False}

            fields = parent.get('fields', {})
            relations = parent.get('relations', []) or []

            # Extract attachments from relations
            attachments = []
            for relation in relations:
                if relation.get('rel') == 'AttachedFile':
                    attachment_info = {
                        "name": relation.get('attributes', {}).get('name', 'unknown'),
                        "url": relation.get('url', ''),
                        "comment": relation.get('attributes', {}).get('comment', '')
                    }
                    attachments.append(attachment_info)
                    self._print_verbose(f" Found attachment: {attachment_info['name']}")

            return {
                "found": True,
                "id": parent_id,
                "title": fields.get('System.Title', ''),
                "description": fields.get('System.Description', '') or '',
                "acceptance_criteria": fields.get('Microsoft.VSTS.Common.AcceptanceCriteria', '') or '',
                "architecture_analysis": self._extract_section(
                    fields.get('System.Description', ''), 'architecture'
                ),
                "security_analysis": self._extract_section(
                    fields.get('System.Description', ''), 'security'
                ),
                "story_points": fields.get('Microsoft.VSTS.Scheduling.StoryPoints'),
                "state": fields.get('System.State', ''),
                "tags": fields.get('System.Tags', '') or '',
                "attachments": attachments
            }
        except Exception as e:
            self._print_verbose(f" ⚠ Could not fetch parent context: {e}")
            return {"found": False, "error": str(e)}

    def _extract_section(self, content: str, section_name: str) -> str:
        """
        Extract a specific section from content (e.g., architecture, security).

        Looks for section headers like "## Architecture" or "### Security Analysis"
        and extracts the content until the next section header.

        Args:
            content: Full content to search
            section_name: Section keyword to look for (case-insensitive)

        Returns:
            Extracted section content or empty string if not found
        """
        if not content:
            return ""

        import re
        # Look for markdown-style headers containing the section name
        pattern = rf'(#{1,3}\s*[^#\n]*{section_name}[^#\n]*\n)(.*?)(?=\n#{1,3}\s|\Z)'
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(0).strip()
        return ""

    def _format_parent_context_for_prompt(self, parent_context: Dict[str, Any]) -> str:
        """
        Format parent Feature context for inclusion in AI prompts.

        Args:
            parent_context: Dict from _get_parent_feature_context()

        Returns:
            Formatted string for prompt inclusion
        """
        if not parent_context.get("found"):
            return ""

        sections = [
            f"\n## Parent Feature Context (#{parent_context.get('id')})",
            f"\n### Title\n{parent_context.get('title', 'N/A')}",
        ]

        if parent_context.get('description'):
            desc = parent_context['description']
            # Warn if content is very large but never truncate (per workflow-flow.md spec)
            if len(desc) > 50000:
                print(f" ⚠ Warning: Parent description is very large ({len(desc)} chars)")
            sections.append(f"\n### Description\n{desc}")

        if parent_context.get('acceptance_criteria'):
            sections.append(f"\n### Acceptance Criteria\n{parent_context['acceptance_criteria']}")

        if parent_context.get('architecture_analysis'):
            sections.append(f"\n### Architecture Analysis\n{parent_context['architecture_analysis']}")

        if parent_context.get('security_analysis'):
            sections.append(f"\n### Security Analysis\n{parent_context['security_analysis']}")

        if parent_context.get('story_points'):
            sections.append(f"\n### Story Points: {parent_context['story_points']}")

        if parent_context.get('attachments'):
            attachment_list = []
            for att in parent_context['attachments']:
                att_entry = f"- **{att['name']}**"
                if att.get('comment'):
                    att_entry += f": {att['comment']}"
                if att.get('url'):
                    att_entry += f"\n URL: {att['url']}"
                attachment_list.append(att_entry)
            sections.append(f"\n### Attachments\n" + "\n".join(attachment_list))

        return "\n".join(sections)

    def _build_full_parent_context(self, work_item_id: int) -> Dict[str, Any]:
        """
        Build complete parent hierarchy context for a work item.

        Walks up the hierarchy (Task → Feature → Epic) and returns full content
        for each level WITHOUT truncation, per workflow-flow.md spec:
        "No result/artifact/message should be truncated unless the data is grossly inflated"

        Args:
            work_item_id: ID of the work item to get parent context for

        Returns:
            Dict with keys:
                - parent_chain: List of parent work items in order [Epic, Feature, ...] (outermost first)
                - full_context: Formatted string with all parent content
                - has_epic: bool indicating if Epic was found
                - has_feature: bool indicating if Feature was found
        """
        result = {
            "parent_chain": [],
            "full_context": "",
            "has_epic": False,
            "has_feature": False
        }

        if not self.adapter:
            return result

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

                # Warn if content is very large (>50KB)
                desc_len = len(parent_info['description'] or '')
                if desc_len > 50000:
                    print(f" ⚠ Warning: {work_item_type} #{parent_id} description is very large ({desc_len} chars)")

                # Get attachments from relations
                relations = parent.get('relations', [])
                parent_info['attachments'] = []
                for relation in relations:
                    if relation.get('rel') == 'AttachedFile':
                        parent_info['attachments'].append({
                            'name': relation.get('attributes', {}).get('name', ''),
                            'url': relation.get('url', ''),
                            'comment': relation.get('attributes', {}).get('comment', '')
                        })

                # Insert at beginning so order is [Epic, Feature, ...]
                result["parent_chain"].insert(0, parent_info)

                if work_item_type == 'Epic':
                    result["has_epic"] = True
                elif work_item_type == 'Feature':
                    result["has_feature"] = True

                current_id = parent_id

            except Exception as e:
                print(f" ⚠ Error fetching parent context: {e}")
                break

        # Format the full context string
        if result["parent_chain"]:
            context_parts = ["## Parent Requirements (Must Be Satisfied)\n"]

            for parent in result["parent_chain"]:
                context_parts.append(f"### {parent['type']} #{parent['id']}: {parent['title']}\n")

                if parent.get('description'):
                    context_parts.append(f"**Description:**\n{parent['description']}\n")

                if parent.get('acceptance_criteria'):
                    context_parts.append(f"**Acceptance Criteria:**\n{parent['acceptance_criteria']}\n")

                if parent.get('story_points'):
                    context_parts.append(f"**Story Points:** {parent['story_points']}\n")

                if parent.get('attachments'):
                    context_parts.append("**Attachments:**\n")
                    for att in parent['attachments']:
                        context_parts.append(f"- {att['name']}")
                        if att.get('comment'):
                            context_parts.append(f" ({att['comment']})")
                        context_parts.append(f": {att['url']}\n")

                context_parts.append("---\n")

            result["full_context"] = "\n".join(context_parts)

        return result

    def _validate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a Task against acceptance criteria.

        Requirements:
        - Detailed design (implementation approach and decisions)
        - Unit test design (test cases and coverage)
        - Integration test design (integration scenarios)
        - Acceptance test design (acceptance criteria)

        Returns:
            {
                "valid": bool,
                "missing": List[str],
                "missing_content": List[str]
            }
        """
        fields = task.get('fields', {})
        description = (fields.get('System.Description', '') or '').lower()
        missing = []
        missing_content = []

        # Check for detailed design
        has_detailed_design = (
            len(description) > 200 and
            any(kw in description for kw in [
                'design', 'approach', 'implementation', 'algorithm',
                'component', 'class', 'function', 'method', 'module',
                'step', 'process', 'flow', 'logic'
            ])
        )
        if not has_detailed_design:
            missing_content.append("detailed_design")
            missing.append("detailed_design: No implementation approach found")

        # Check for unit test design
        has_unit_tests = any(kw in description for kw in [
            'unit test', 'unit-test', 'unittest', 'test case',
            'mock', 'stub', 'assert', 'expect', 'test coverage'
        ])
        if not has_unit_tests:
            missing_content.append("unit_test_design")
            missing.append("unit_test_design: No unit test design found")

        # Check for integration test design
        has_integration_tests = any(kw in description for kw in [
            'integration test', 'integration-test', 'e2e', 'end-to-end',
            'api test', 'system test', 'component test', 'contract test'
        ])
        if not has_integration_tests:
            missing_content.append("integration_test_design")
            missing.append("integration_test_design: No integration test design found")

        # Check for acceptance test design
        has_acceptance_tests = any(kw in description for kw in [
            'acceptance test', 'acceptance-test', 'acceptance criteria',
            'user acceptance', 'uat', 'bdd', 'given', 'when', 'then'
        ])
        if not has_acceptance_tests:
            missing_content.append("acceptance_test_design")
            missing.append("acceptance_test_design: No acceptance test design found")

        return {
            "valid": len(missing) == 0,
            "missing": missing,
            "missing_content": missing_content
        }

    def _validate_bug(self, bug: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a Bug against acceptance criteria.

        Requirements:
        - Reproduction steps (step-by-step repro instructions)
        - Root cause analysis (why the bug exists)
        - Solution design (how to fix it)
        - Acceptance test design (how to verify the fix)

        Returns:
            {
                "valid": bool,
                "missing": List[str],
                "missing_content": List[str]
            }
        """
        fields = bug.get('fields', {})
        description = (fields.get('System.Description', '') or '').lower()
        repro_steps = (fields.get('Microsoft.VSTS.TCM.ReproSteps', '') or '').lower()
        combined = description + ' ' + repro_steps
        missing = []
        missing_content = []

        # Check for reproduction steps
        has_repro_steps = (
            len(repro_steps) > 50 or
            any(kw in combined for kw in [
                'steps to reproduce', 'reproduction steps', 'repro steps',
                'step 1', '1.', '1)', 'to reproduce', 'how to reproduce',
                'reproduce the issue', 'precondition'
            ])
        )
        if not has_repro_steps:
            missing_content.append("reproduction_steps")
            missing.append("reproduction_steps: No reproduction steps found")

        # Check for root cause analysis
        has_root_cause = any(kw in combined for kw in [
            'root cause', 'cause', 'reason', 'because', 'due to',
            'investigation', 'analysis', 'diagnosed', 'identified',
            'the issue is', 'the problem is', 'occurs when', 'happens because'
        ])
        if not has_root_cause:
            missing_content.append("root_cause_analysis")
            missing.append("root_cause_analysis: No root cause analysis found")

        # Check for solution design
        has_solution = any(kw in combined for kw in [
            'solution', 'fix', 'resolution', 'approach', 'to resolve',
            'will fix', 'should fix', 'proposed fix', 'remediation',
            'corrective action', 'implementation'
        ])
        if not has_solution:
            missing_content.append("solution_design")
            missing.append("solution_design: No solution design found")

        # Check for acceptance test design
        has_acceptance = any(kw in combined for kw in [
            'acceptance test', 'verify', 'validate', 'test case',
            'acceptance criteria', 'how to test', 'verification steps',
            'expected result', 'expected behavior'
        ])
        if not has_acceptance:
            missing_content.append("acceptance_test_design")
            missing.append("acceptance_test_design: No acceptance test design found")

        return {
            "valid": len(missing) == 0,
            "missing": missing,
            "missing_content": missing_content
        }

    # ========================================================================
    # Agent-Based Content Generation (Tree-Based Architecture)
    # ========================================================================
    #
    # These methods spawn independent AI agents for content generation,
    # providing context isolation and parallel execution capabilities.
    #
    # Architecture:
    # - EpicBreakdownAgent: Breaks Epic into Features
    # - FeatureConformanceAgent: Conforms Feature + creates Implementation Task
    # - Parallel execution via ThreadPoolExecutor (max 3 concurrent)
    #
    # Benefits over single-session approach:
    # - Fresh context per agent (no context overflow)
    # - Failure isolation (one Feature failure doesn't affect others)
    # - Parallel processing for Features
    # - Immediate write-back after each agent completes

    # Agent configuration
    AGENT_MODEL = "claude-opus-4-5-20251101"
    AGENT_MAX_TOKENS = 16000
    AGENT_MAX_WORKERS = 3

    def _get_agent_sdk_wrapper(self):
        """Get or create Agent SDK wrapper for agent spawning with codebase access."""
        if not hasattr(self, '_agent_sdk_wrapper') or self._agent_sdk_wrapper is None:
            try:
                from scripts.workflow_executor.agent_sdk import AgentSDKWrapper
                from pathlib import Path

                self._agent_sdk_wrapper = AgentSDKWrapper(
                    model=self.AGENT_MODEL,
                    max_tokens=self.AGENT_MAX_TOKENS,
                    working_directory=str(Path.cwd()),
                    tool_preset="implementation"  # Agents write design/test plan files for consensus
                )
                return self._agent_sdk_wrapper
            except ImportError as e:
                # ALWAYS report import errors - no silent fallbacks
                from cli.console import print_error, print_info
                print_error(f"Agent SDK Import Failed: {e}")
                print_error("The Claude Agent SDK is not available. Agents require tool access to read the codebase.")
                print_info("Install with: pip install claude-code-sdk")
                print_info("Workflow cannot spawn agents without codebase access.")
                return None
            except Exception as e:
                from cli.console import print_error
                print_error(f"Agent SDK Initialization Failed: {e}")
                return None
        return self._agent_sdk_wrapper

    def _spawn_epic_breakdown_agent(
        self,
        epic: Dict[str, Any],
        missing_content: List[str]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Spawn an agent to break down an Epic into Features.

        Args:
            epic: Epic work item
            missing_content: List of missing content types

        Returns:
            (success, result) tuple where result contains:
            - epic_updates: Content to update on Epic
            - features_to_create: List of Feature definitions
        """
        epic_id = epic.get('id') or epic.get('fields', {}).get('System.Id')
        epic_title = epic.get('fields', {}).get('System.Title', 'Unknown')
        epic_description = epic.get('fields', {}).get('System.Description', '')

        prompt = build_epic_breakdown_prompt(
            epic_id=epic_id,
            epic_title=epic_title,
            epic_description=epic_description,
            missing_content=missing_content
        )

        # Add instruction to read CLAUDE.md for context
        enhanced_prompt = f"""First, read the project CLAUDE.md to understand the project context, architecture, and conventions.

{prompt}"""

        # Try Agent SDK first (provides codebase access)
        sdk = self._get_agent_sdk_wrapper()
        if sdk:
            try:
                result = sdk.ask(
                    prompt=enhanced_prompt,
                    system_prompt="""You are a Senior Engineer breaking down Epics into Features.
Read the project CLAUDE.md and relevant code files to understand the architecture before generating Feature definitions."""
                )

                # Extract text content
                text = result.response

                # Parse JSON from response
                parsed = self._extract_json_from_response(text)
                if not parsed:
                    return False, {"error": "Failed to parse JSON from agent response"}

                # Validate response
                is_valid, errors = validate_epic_breakdown_response(parsed)
                if not is_valid:
                    return False, {"error": f"Invalid response: {errors}"}

                return True, parsed

            except Exception as e:
                # ALWAYS report errors - no silent fallbacks to Anthropic API
                from cli.console import print_error, print_info
                print_error(f"Agent SDK Execution Failed: {e}")
                print_error("Epic breakdown requires codebase access to read CLAUDE.md and understand architecture.")
                print_info("This epic will be skipped. Fix the Agent SDK issue to enable AI-assisted breakdown.")
                return False, {"error": f"Agent SDK execution failed: {e}"}

        # No Agent SDK available - report and fail (don't fall back to degraded functionality)
        from cli.console import print_error, print_info
        print_error("Epic Breakdown Skipped: No Agent SDK available")
        print_error("Epic breakdown requires codebase access. Cannot proceed without Agent SDK.")
        print_info("Install with: pip install claude-code-sdk")
        return False, {"error": "Agent SDK not available - cannot breakdown epic without codebase access"}

    def _spawn_feature_conformance_agent(
        self,
        feature: Dict[str, Any],
        missing_content: List[str],
        needs_implementation_task: bool = True
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Spawn an agent to conform a Feature and create its Implementation Task.

        Args:
            feature: Feature work item
            missing_content: List of missing content types
            needs_implementation_task: Whether to generate Implementation Task

        Returns:
            (success, result) tuple where result contains:
            - feature_updates: Content to update on Feature
            - implementation_task: Task definition with test plans
        """
        feature_id = feature.get('id') or feature.get('fields', {}).get('System.Id')
        feature_title = feature.get('fields', {}).get('System.Title', 'Unknown')
        feature_description = feature.get('fields', {}).get('System.Description', '')

        # Get parent Epic context if available - FULL context, no truncation
        # (per workflow-flow.md: "No result/artifact/message should be truncated")
        parent_epic_context = ""
        parent_id = feature.get('fields', {}).get('System.Parent')
        if parent_id and self.adapter:
            try:
                parent = self.adapter.get_work_item(int(parent_id))
                if parent:
                    parent_fields = parent.get('fields', {})
                    epic_desc = parent_fields.get('System.Description', '')
                    epic_acc_criteria = parent_fields.get('Microsoft.VSTS.Common.AcceptanceCriteria', '')

                    # Warn if content is very large (>50KB) but never truncate
                    if len(epic_desc) > 50000:
                        print(f" ⚠ Warning: Epic description is very large ({len(epic_desc)} chars)")

                    parent_epic_context = f"""
## Parent Epic #{parent_id}: {parent_fields.get('System.Title', '')}

### Epic Description
{epic_desc}

### Epic Acceptance Criteria
{epic_acc_criteria}
"""
            except Exception:
                pass

        prompt = build_feature_conformance_prompt(
            feature_id=feature_id,
            feature_title=feature_title,
            feature_description=feature_description,
            parent_epic_context=parent_epic_context,
            missing_content=missing_content,
            needs_implementation_task=needs_implementation_task
        )

        # Add instruction to read CLAUDE.md for context
        enhanced_prompt = f"""First, read the project CLAUDE.md to understand the project context, architecture, and conventions.

{prompt}"""

        # Try Agent SDK first (provides codebase access)
        sdk = self._get_agent_sdk_wrapper()
        if sdk:
            try:
                result = sdk.ask(
                    prompt=enhanced_prompt,
                    system_prompt="""You are a Senior Engineer conforming Features and creating Implementation Tasks.
Read the project CLAUDE.md and relevant code files to understand the architecture before generating Task definitions."""
                )

                # Extract text content
                text = result.response

                # Parse JSON from response
                parsed = self._extract_json_from_response(text)
                if not parsed:
                    return False, {"error": "Failed to parse JSON from agent response"}

                # Validate response
                is_valid, errors = validate_feature_conformance_response(parsed)
                if not is_valid:
                    return False, {"error": f"Invalid response: {errors}"}

                return True, parsed

            except Exception as e:
                # ALWAYS report errors - no silent fallbacks to Anthropic API
                from cli.console import print_error, print_info
                print_error(f"Agent SDK Execution Failed: {e}")
                print_error("Feature conformance requires codebase access to read CLAUDE.md and understand architecture.")
                print_info("This feature will be skipped. Fix the Agent SDK issue to enable AI-assisted conformance.")
                return False, {"error": f"Agent SDK execution failed: {e}"}

        # No Agent SDK available - report and fail (don't fall back to degraded functionality)
        from cli.console import print_error, print_info
        print_error("Feature Conformance Skipped: No Agent SDK available")
        print_error("Feature conformance requires codebase access. Cannot proceed without Agent SDK.")
        print_info("Install with: pip install claude-code-sdk")
        return False, {"error": "Agent SDK not available - cannot conform feature without codebase access"}

    def _spawn_feature_agents_parallel(
        self,
        features: List[Dict[str, Any]],
        validations: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], bool, Dict[str, Any]]]:
        """
        Spawn multiple Feature conformance agents in parallel.

        Args:
            features: List of Feature work items
            validations: Corresponding validation results for each Feature

        Returns:
            List of (feature, success, result) tuples
        """
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.AGENT_MAX_WORKERS) as executor:
            # Submit all tasks
            future_to_feature = {}
            for feature, validation in zip(features, validations):
                missing_content = validation.get("missing_content", [])
                impl_missing = any('implementation_task' in m for m in validation.get("missing", []))
                test_missing = any('test_task' in m for m in validation.get("missing", []))
                needs_task = impl_missing or test_missing

                future = executor.submit(
                    self._spawn_feature_conformance_agent,
                    feature,
                    missing_content,
                    needs_task
                )
                future_to_feature[future] = feature

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_feature):
                feature = future_to_feature[future]
                try:
                    success, result = future.result()
                    results.append((feature, success, result))
                except Exception as e:
                    results.append((feature, False, {"error": str(e)}))

        return results

    def _extract_json_from_response(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from agent response text.

        Handles both pure JSON and markdown-wrapped JSON.
        """
        if not text:
            return None

        # Try direct JSON parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract from markdown code block
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in text
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        return None

    # ========================================================================
    # Multi-Agent Consensus Methods (Feature #1344)
    # ========================================================================
    #
    # These methods implement k-phase commit consensus per workflow-flow.md:
    # - Leader proposes, evaluators critique in parallel
    # - Feedback aggregated and relayed to leader for revision
    # - Repeats for k rounds or until all evaluators accept
    #
    # Two configurations:
    # - EPIC_TO_FEATURE_CONFIG: senior-engineer + architect, security-specialist, tester
    # - FEATURE_TO_TASK_CONFIG: senior-engineer + architect, security-specialist, tester

    def _run_epic_breakdown_consensus(
        self,
        epic: Dict[str, Any],
        missing_content: List[str]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Run multi-agent consensus for Epic-to-Feature breakdown.

        Uses EPIC_TO_FEATURE_CONFIG:
        - Leader: senior-engineer
        - Evaluators: architect, security-specialist, tester (parallel)
        - Max rounds: 2

        Args:
            epic: Epic work item
            missing_content: List of missing content types

        Returns:
            (success, result) tuple where result contains:
            - epic_updates: Content to update on Epic
            - features_to_create: List of Feature definitions
        """
        import asyncio

        epic_id = epic.get('id') or epic.get('fields', {}).get('System.Id')
        epic_title = epic.get('fields', {}).get('System.Title', 'Unknown')
        epic_description = epic.get('fields', {}).get('System.Description', '')
        epic_acceptance = epic.get('fields', {}).get('Microsoft.VSTS.Common.AcceptanceCriteria', '')

        print(f"\n{'='*60}")
        print(f"  CONSENSUS: Epic #{epic_id} - {epic_title}")
        print(f"{'='*60}")
        print(f"  Leader: senior-engineer")
        print(f"  Evaluators: architect, security-specialist, tester")
        print(f"  Max rounds: {EPIC_TO_FEATURE_CONFIG.max_rounds}")
        print(f"{'='*60}\n")

        # Build context with FULL Epic content (no truncation per workflow-flow.md)
        context = {
            "epic": {
                "id": epic_id,
                "title": epic_title,
                "description": epic_description,
                "acceptance_criteria": epic_acceptance,
                "missing_content": missing_content
            }
        }

        # Warn if content is very large but never truncate
        if len(epic_description) > 50000:
            print(f"  Warning: Epic description is very large ({len(epic_description)} chars)")

        try:
            # Create agent interface using Claude Agent SDK for tool access
            agent_interface = AgentInterface(
                backend="sdk",
                model=self.AGENT_MODEL,
                tool_preset="implementation"
            )

            # Create orchestrator
            orchestrator = ConsensusOrchestrator(
                config=EPIC_TO_FEATURE_CONFIG,
                adapter=self.adapter,
                agent_interface=agent_interface,
                verbose=True
            )

            # Run consensus (async)
            result = asyncio.run(orchestrator.run_consensus(context))

            if result.consensus_reached:
                print(f"\n  CONSENSUS REACHED in {len(result.rounds)} round(s)")
                # Extract the final proposal
                final = result.final_proposal
                if final.get("parse_error"):
                    return False, {"error": "Failed to parse leader proposal"}
                return True, final
            else:
                print(f"\n  CONSENSUS NOT REACHED after {len(result.rounds)} rounds")
                # Return last proposal with warning
                final = result.final_proposal
                if final.get("parse_error"):
                    return False, {"error": "Failed to parse leader proposal"}
                return True, {**final, "_consensus_warning": "Consensus not reached"}

        except Exception as e:
            print(f"\n  CONSENSUS ERROR: {e}")
            return False, {"error": str(e)}

    def _run_feature_to_task_consensus(
        self,
        feature: Dict[str, Any],
        missing_content: List[str],
        needs_implementation_task: bool = True
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Run multi-agent consensus for Feature-to-Task creation.

        Uses FEATURE_TO_TASK_CONFIG:
        - Leader: senior-engineer
        - Evaluators: architect, security-specialist, tester (parallel)
        - Max rounds: 2

        Args:
            feature: Feature work item
            missing_content: List of missing content types
            needs_implementation_task: Whether to generate Implementation Task

        Returns:
            (success, result) tuple where result contains:
            - feature_updates: Content to update on Feature
            - implementation_task: Task definition with test plans
        """
        import asyncio

        feature_id = feature.get('id') or feature.get('fields', {}).get('System.Id')
        feature_title = feature.get('fields', {}).get('System.Title', 'Unknown')
        feature_description = feature.get('fields', {}).get('System.Description', '')
        feature_acceptance = feature.get('fields', {}).get('Microsoft.VSTS.Common.AcceptanceCriteria', '')

        print(f"\n{'='*60}")
        print(f"  CONSENSUS: Feature #{feature_id} - {feature_title}")
        print(f"{'='*60}")
        print(f"  Leader: senior-engineer")
        print(f"  Evaluators: architect, security-specialist, tester")
        print(f"  Max rounds: {FEATURE_TO_TASK_CONFIG.max_rounds}")
        print(f"{'='*60}\n")

        # Get parent Epic context if available - FULL content, no truncation
        parent_context = {}
        parent_id = feature.get('fields', {}).get('System.Parent')
        if parent_id and self.adapter:
            try:
                parent = self.adapter.get_work_item(int(parent_id))
                if parent:
                    parent_fields = parent.get('fields', {})
                    parent_context = {
                        "id": parent_id,
                        "title": parent_fields.get('System.Title', ''),
                        "description": parent_fields.get('System.Description', ''),
                        "acceptance_criteria": parent_fields.get('Microsoft.VSTS.Common.AcceptanceCriteria', ''),
                    }
                    if len(parent_context.get('description', '')) > 50000:
                        print(f"  Warning: Parent Epic description is very large")
            except Exception:
                pass

        # Build context
        context = {
            "feature": {
                "id": feature_id,
                "title": feature_title,
                "description": feature_description,
                "acceptance_criteria": feature_acceptance,
                "missing_content": missing_content,
                "needs_implementation_task": needs_implementation_task
            }
        }
        if parent_context:
            context["epic"] = parent_context

        try:
            # Create agent interface using Claude Agent SDK for tool access
            agent_interface = AgentInterface(
                backend="sdk",
                model=self.AGENT_MODEL,
                tool_preset="implementation"
            )

            # Create orchestrator
            orchestrator = ConsensusOrchestrator(
                config=FEATURE_TO_TASK_CONFIG,
                adapter=self.adapter,
                agent_interface=agent_interface,
                verbose=True
            )

            # Run consensus (async)
            result = asyncio.run(orchestrator.run_consensus(context))

            if result.consensus_reached:
                print(f"\n  CONSENSUS REACHED in {len(result.rounds)} round(s)")
                final = result.final_proposal
                if final.get("parse_error"):
                    return False, {"error": "Failed to parse leader proposal"}
                return True, final
            else:
                print(f"\n  CONSENSUS NOT REACHED after {len(result.rounds)} rounds")
                final = result.final_proposal
                if final.get("parse_error"):
                    return False, {"error": "Failed to parse leader proposal"}
                return True, {**final, "_consensus_warning": "Consensus not reached"}

        except Exception as e:
            print(f"\n  CONSENSUS ERROR: {e}")
            return False, {"error": str(e)}

    # ========================================================================
    # AI-Driven Content Generation Methods (Legacy - Single Session)
    # ========================================================================

    def _generate_epic_content_with_ai(
        self,
        epic: Dict[str, Any],
        missing_content: List[str]
    ) -> Dict[str, Any]:
        """
        Generate missing content for an EPIC using AI.

        Args:
            epic: EPIC work item
            missing_content: List of missing content types

        Returns:
            Dict with generated content (business_analysis, success_criteria, scope_definition)
        """
        if not self.interactive_session or not self.interactive_session.is_available():
            print(" ⚠ AI unavailable - cannot generate content")
            return {}

        epic_title = epic.get('fields', {}).get('System.Title', 'Unknown')
        epic_description = epic.get('fields', {}).get('System.Description', '')

        prompt = f"""Analyze this EPIC and generate the missing business documentation.

EPIC: {epic_title}
Current Description: {epic_description}

Generate the following sections (only generate what's marked as missing):
Missing: {', '.join(missing_content)}

For each missing section, provide comprehensive content:

1. **Business Analysis** (if missing):
   - Business value and ROI justification
   - Stakeholder impact analysis
   - Market opportunity or efficiency gains
   - Cost-benefit analysis

2. **Success Criteria** (if missing):
   - Measurable outcomes and KPIs
   - Definition of Done at EPIC level
   - Target metrics with specific numbers
   - Timeline expectations

3. **Scope Definition** (if missing):
   - What is IN scope (specific deliverables)
   - What is OUT of scope (explicit exclusions)
   - Dependencies and assumptions
   - Boundaries and constraints

Return as JSON:
{{
  "business_analysis": "Detailed business value analysis...",
  "success_criteria": "Measurable success criteria...",
  "scope_definition": "Clear scope boundaries..."
}}"""

        try:
            response = self.interactive_session.ask(prompt)
            content = self.interactive_session.extract_json_from_response(response)
            return content if isinstance(content, dict) else {}
        except Exception as e:
            print(f" ⚠ AI generation failed: {e}")
            return {}

    def _generate_epic_features_with_ai(
        self,
        epic: Dict[str, Any],
        target_hours: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Generate Feature breakdown for an EPIC using AI.

        Args:
            epic: EPIC work item
            target_hours: Minimum hours of work to generate

        Returns:
            List of Feature definitions to create
        """
        if not self.interactive_session or not self.interactive_session.is_available():
            print(" ⚠ AI unavailable - cannot generate Features")
            return []

        epic_title = epic.get('fields', {}).get('System.Title', 'Unknown')
        epic_description = epic.get('fields', {}).get('System.Description', '')

        prompt = f"""Break down this EPIC into Features following the Decomposition Requirements.

EPIC: {epic_title}
Description: {epic_description}

**Feature Extraction Requirements:**

1. **Minimum Size**: Each Feature must be at least 10 story points (~50 hours)
   - 1 story point = ~5 hours of work
   - Features should represent minimum logically-related functionality
   - If remaining EPIC functionality < 10 points, bundle into one final Feature

2. **Feature Characteristics**:
   - Cohesive: Related functionality grouped together
   - Measurable: Clear success metrics
   - Testable: Can be verified through tests
   - Independently Deliverable: Can be released on its own
   - Valuable: Provides user/business value when complete

3. **Size Guidance**:
   - Minimum: 10 story points (~50 hours)
   - Typical: 10-30 story points
   - Each Feature will have 2-5 Tasks under it

4. **Task Preview**: Each Feature will get:
   - Exactly 1 Implementation Task (code + comprehensive tests)
   - Exactly 1 Testing/Validation Task (verify test quality)
   - Optional Deployment Tasks if needed
   - Tasks will be 1-5 story points each

For each Feature, provide:
1. Title (clear, actionable capability name)
2. Description (full scope, requirements, technical approach)
3. Story points (minimum 10, typically 10-30)
4. Acceptance criteria (3+ testable criteria)
5. Dependencies (other Features this depends on, if any)

Return as JSON array:
[
  {{
    "title": "Feature title",
    "description": "Comprehensive description including scope and approach",
    "story_points": 15,
    "acceptance_criteria": ["Testable criterion 1", "Testable criterion 2", "Testable criterion 3"],
    "dependencies": []
  }}
]

IMPORTANT:
- Each Feature must have >= 10 story points
- Sum of Feature story points should approximate EPIC total ({target_hours} hours = ~{target_hours // 5} points)
- All EPIC functionality must be covered by Features
- Features must be independently deliverable when possible"""

        try:
            response = self.interactive_session.ask(prompt)
            features = self.interactive_session.extract_json_from_response(response)
            return features if isinstance(features, list) else []
        except Exception as e:
            print(f" ⚠ AI generation failed: {e}")
            return []

    def _generate_feature_content_with_ai(
        self,
        feature: Dict[str, Any],
        missing_content: List[str]
    ) -> Dict[str, Any]:
        """
        Generate missing content for a Feature using AI.

        Args:
            feature: Feature work item
            missing_content: List of missing content types

        Returns:
            Dict with generated content
        """
        if not self.interactive_session or not self.interactive_session.is_available():
            print(" ⚠ AI unavailable - cannot generate content")
            return {}

        feature_title = feature.get('fields', {}).get('System.Title', 'Unknown')
        feature_description = feature.get('fields', {}).get('System.Description', '')

        prompt = f"""Generate CONCISE technical documentation for this Feature.

Feature: {feature_title}
Description: {feature_description}

Generate ONLY these missing sections: {', '.join(missing_content)}

IMPORTANT: Keep each section to 150-300 words. Use bullet points. Be actionable.

Return as JSON with ONLY the missing sections:
{{
  "detailed_description": "• Scope summary\\n• Key requirements (3-5 bullets)\\n• User story",
  "architecture_analysis": "• Technical approach\\n• Key components\\n• Integration points",
  "security_analysis": "• Auth requirements\\n• Data protection\\n• Key mitigations",
  "acceptance_criteria": "• Given/When/Then criteria (3-5 items)\\n• Key edge cases"
}}"""

        try:
            response = self.interactive_session.ask(prompt)
            content = self.interactive_session.extract_json_from_response(response)
            return content if isinstance(content, dict) else {}
        except Exception as e:
            print(f" ⚠ AI generation failed: {e}")
            return {}

    def _generate_task_content_with_ai(
        self,
        task: Dict[str, Any],
        missing: List[str]
    ) -> Dict[str, Any]:
        """
        Generate missing content for a Task using AI.

        Uses full parent Feature context including all fields and attachments
        to provide comprehensive context for content generation.

        Args:
            task: Task work item
            missing: List of missing fields

        Returns:
            Dict with generated content
        """
        if not self.interactive_session or not self.interactive_session.is_available():
            print(" ⚠ AI unavailable - cannot generate content")
            return {}

        task_title = task.get('fields', {}).get('System.Title', 'Unknown')
        task_description = task.get('fields', {}).get('System.Description', '')
        parent_id = task.get('fields', {}).get('System.Parent')

        # Get full parent Feature context including attachments
        parent_context_str = ""
        if parent_id:
            self._print_verbose(f" Fetching parent Feature #{parent_id} context...")
            parent_context = self._get_parent_feature_context(parent_id)
            if parent_context.get("found"):
                parent_context_str = self._format_parent_context_for_prompt(parent_context)
                attachment_count = len(parent_context.get("attachments", []))
                if attachment_count > 0:
                    self._print_verbose(f" ✓ Found {attachment_count} attachment(s) in parent Feature")

        needs = [m.split(':')[0] for m in missing]

        prompt = f"""Generate detailed design and test design content for this Task.

# Task Information
**Title**: {task_title}
**Current Description**: {task_description}
{parent_context_str}

# Requirements
Generate comprehensive content for the following missing sections:
{chr(10).join(f'- {need}' for need in needs)}

Use the parent Feature context above (including any referenced attachments) to ensure
the generated content is aligned with the Feature's requirements, architecture decisions,
and acceptance criteria.

Return as JSON:
{{
  "detailed_design": "Implementation approach and design decisions...",
  "unit_test_design": "Unit test cases covering...",
  "integration_test_design": "Integration test scenarios...",
  "acceptance_test_design": "Acceptance test criteria..."
}}"""

        try:
            response = self.interactive_session.ask(prompt)
            content = self.interactive_session.extract_json_from_response(response)
            return content if isinstance(content, dict) else {}
        except Exception as e:
            print(f" ⚠ AI generation failed: {e}")
            return {}

    def _generate_new_task_content_with_ai(
        self,
        feature: Dict[str, Any],
        task_type: str
    ) -> Dict[str, Any]:
        """
        Generate content for a new Task being created under a Feature.

        Conforms to Decomposition Requirements from .claude/commands/backlog-grooming.md:
        - Implementation Task: Full context for engineer, detailed specs, test types, 80% coverage
        - Testing Task: Validate test types, completeness, falsifiability, report results

        Uses full Feature context including all fields and attachments.

        Args:
            feature: Parent Feature work item
            task_type: Type of task ("implementation" or "test")

        Returns:
            Dict with generated content for the new task
        """
        if not self.interactive_session or not self.interactive_session.is_available():
            print(" ⚠ AI unavailable - cannot generate content")
            return {}

        feature_id = feature.get('id') or feature.get('fields', {}).get('System.Id')
        feature_title = feature.get('fields', {}).get('System.Title', 'Unknown')
        feature_description = feature.get('fields', {}).get('System.Description', '')
        acceptance_criteria = feature.get('fields', {}).get('Microsoft.VSTS.Common.AcceptanceCriteria', '')

        # Get full Feature context including attachments
        # (The feature passed in may not have relations, so fetch fresh)
        attachment_info = ""
        if feature_id and self.adapter:
            self._print_verbose(f" Fetching Feature #{feature_id} attachments...")
            full_feature = self._get_parent_feature_context(feature_id)
            if full_feature.get("found") and full_feature.get("attachments"):
                attachments = full_feature["attachments"]
                self._print_verbose(f" ✓ Found {len(attachments)} attachment(s)")
                attachment_lines = ["\n## Feature Attachments"]
                for att in attachments:
                    att_entry = f"- **{att['name']}**"
                    if att.get('comment'):
                        att_entry += f": {att['comment']}"
                    if att.get('url'):
                        att_entry += f"\n URL: {att['url']}"
                    attachment_lines.append(att_entry)
                attachment_info = "\n".join(attachment_lines)

                # Also include architecture/security analysis if found
                if full_feature.get("architecture_analysis"):
                    attachment_info += f"\n\n## Architecture Analysis\n{full_feature['architecture_analysis']}"
                if full_feature.get("security_analysis"):
                    attachment_info += f"\n\n## Security Analysis\n{full_feature['security_analysis']}"

        if task_type == "implementation":
            # Implementation Task: Full context for engineer to implement code + tests
            # Note: No truncation per workflow-flow.md spec
            prompt = f"""Generate comprehensive Implementation Task content for this Feature.

# Feature Information
**Title**: {feature_title}
**Description**: {feature_description}
**Acceptance Criteria**: {acceptance_criteria}
{attachment_info}

This Implementation Task must contain enough context for an engineer to implement
both the code AND comprehensive tests solely based on this Task description.

Generate the following sections:

1. **detailed_design**: Implementation approach and technical decisions
   - Key components/modules to create or modify
   - Data structures and algorithms
   - API contracts and interfaces
   - Error handling strategy

2. **function_specifications**: Detailed specs of each testable component
   - Function signatures with parameters and return types
   - Preconditions and postconditions
   - Side effects and state changes

3. **unit_test_requirements**: Unit tests (80% minimum code coverage)
   - Test individual functions/methods in isolation
   - Mock external dependencies
   - Cover normal flows and error conditions
   - Include boundary value tests
   - **Marker**: `@pytest.mark.unit` + `@pytest.mark.functional`
   - **SDLC Stage**: Feature Testing, CI Pipeline

4. **integration_test_requirements**: Integration tests with coverage targets
   - **Whitebox**: Component interactions using internal knowledge
     - Marker: `@pytest.mark.integration_whitebox`
     - SDLC Stage: Feature Testing, CI Pipeline
   - **Blackbox**: Component interactions via public APIs only
     - Marker: `@pytest.mark.integration_blackbox`
     - SDLC Stage: CI Pipeline
   - Test external API integrations
   - Test database operations if applicable

5. **edge_case_test_requirements**: Whitebox edge-case testing
   - Boundary conditions (empty, null, max values)
   - Error handling and exception paths
   - Race conditions if applicable
   - Cases that black-box testing might miss
   - **Marker**: `@pytest.mark.integration_whitebox` + `@pytest.mark.functional`

6. **acceptance_test_requirements**: Tests for ALL acceptance criteria
   - Map each acceptance criterion to specific test cases
   - Verify business rules are enforced
   - Validate user-facing behavior
   - **Marker**: `@pytest.mark.acceptance` + `@pytest.mark.functional`
   - **SDLC Stage**: Sprint Review

7. **falsifiability_requirements**: Tests must detect actual failures
   - Tests should fail when implementation is incorrect
   - Avoid tests that always pass (e.g., assert True)
   - Include negative test cases

8. **evidence_requirements**: Evidence of complete implementation
   - Code review checklist items
   - Test coverage report requirements
   - Documentation updates needed

**IMPORTANT: Test plans will be attached as separate files for the Tester agent.**

## Test Classification (pytest markers for SDLC stages)
| Test Level | Marker | SDLC Stage |
|------------|--------|------------|
| Unit | `@pytest.mark.unit` | Feature Testing, CI Pipeline |
| Integration (Whitebox) | `@pytest.mark.integration_whitebox` | Feature Testing, CI Pipeline |
| Integration (Blackbox) | `@pytest.mark.integration_blackbox` | CI Pipeline |
| Acceptance | `@pytest.mark.acceptance` | Feature Testing, Sprint Review |
| System (E2E) | `@pytest.mark.system` | Feature Testing, Sprint Review |

Return as JSON with these fields (test plans will be extracted and attached):
{{
  "detailed_design": "...",
  "function_specifications": "...",
  "unit_test_plan": "Unit tests with @pytest.mark.unit marker (will be attached as file)",
  "integration_test_plan": "Integration tests with @pytest.mark.integration_whitebox or integration_blackbox markers (will be attached as file)",
  "edge_case_test_plan": "Edge-case whitebox tests with @pytest.mark.integration_whitebox marker (will be attached as file)",
  "acceptance_test_plan": "Acceptance tests with @pytest.mark.acceptance marker mapping criteria to tests (will be attached as file)",
  "falsifiability_requirements": "...",
  "evidence_requirements": "..."
}}"""
        else: # test task (Testing/Validation Task)
            # Testing Task: Validate test quality and completeness
            # Note: No truncation per workflow-flow.md spec
            prompt = f"""Generate comprehensive Testing/Validation Task content for this Feature.

# Feature Information
**Title**: {feature_title}
**Description**: {feature_description}
**Acceptance Criteria**: {acceptance_criteria}
{attachment_info}

This Testing Task validates the quality and completeness of tests written
for the Implementation Task. The tester must verify:

## Test Classification (pytest markers for SDLC stages)
| Test Level | Marker | SDLC Stage |
|------------|--------|------------|
| Unit | `@pytest.mark.unit` | Feature Testing, CI Pipeline |
| Integration (Whitebox) | `@pytest.mark.integration_whitebox` | Feature Testing, CI Pipeline |
| Integration (Blackbox) | `@pytest.mark.integration_blackbox` | CI Pipeline |
| Acceptance | `@pytest.mark.acceptance` | Feature Testing, Sprint Review |
| System (E2E) | `@pytest.mark.system` | Feature Testing, Sprint Review |

Generate the following sections:

1. **test_type_validation**: Validate presence of all required test types WITH proper markers
   - Confirm unit tests exist with `@pytest.mark.unit` marker
   - Confirm integration tests exist with `@pytest.mark.integration_whitebox` or `integration_blackbox`
   - Confirm edge-case/whitebox tests exist with `@pytest.mark.integration_whitebox`
   - Confirm acceptance tests exist with `@pytest.mark.acceptance` marker
   - Checklist of test categories and their markers to verify

2. **coverage_validation**: Validate 80% minimum code coverage
   - Instructions for running coverage tools
   - Expected coverage thresholds by module
   - Areas that may need additional coverage
   - How to generate and interpret coverage report

3. **feature_coverage_validation**: All acceptance criteria have tests
   - Mapping of acceptance criteria to test cases
   - Checklist: each criterion must have corresponding passing test
   - Verify no acceptance criteria are untested
   - All acceptance tests must have `@pytest.mark.acceptance` marker

4. **falsifiability_validation**: Validate tests can detect failures
   - Instructions to introduce intentional bugs
   - Expected tests that should fail for each bug
   - Steps to remove intentional bugs after validation
   - Evidence that tests are not "always pass" tests

5. **sdlc_stage_validation**: Validate tests run at correct SDLC stages
   - Feature Testing: `pytest -m "unit or integration_whitebox or acceptance or system"`
   - CI Pipeline: `pytest -m "unit or integration_whitebox or integration_blackbox"`
   - Sprint Review: `pytest -m "acceptance or system"`
   - Verify tests have correct markers for their intended stage

6. **test_results_report**: Test execution and evidence requirements
   - Full test suite execution steps with marker filters
   - Expected output format and pass/fail criteria
   - Evidence artifacts to collect (logs, screenshots, coverage)
   - Final verification sign-off checklist

Return as JSON:
{{
  "test_type_validation": "...",
  "coverage_validation": "...",
  "feature_coverage_validation": "...",
  "falsifiability_validation": "...",
  "sdlc_stage_validation": "...",
  "test_results_report": "..."
}}"""

        try:
            response = self.interactive_session.ask(prompt)
            content = self.interactive_session.extract_json_from_response(response)
            return content if isinstance(content, dict) else {}
        except Exception as e:
            print(f" ⚠ AI generation failed: {e}")
            return {}

    def _generate_bug_content_with_ai(
        self,
        bug: Dict[str, Any],
        missing_content: List[str]
    ) -> Dict[str, Any]:
        """
        Generate missing content for a Bug using AI.

        Uses full parent Feature context (if bug is linked to a Feature)
        including all fields and attachments to provide comprehensive context.

        Args:
            bug: Bug work item
            missing_content: List of missing content types

        Returns:
            Dict with generated content
        """
        if not self.interactive_session or not self.interactive_session.is_available():
            print(" ⚠ AI unavailable - cannot generate content")
            return {}

        bug_title = bug.get('fields', {}).get('System.Title', 'Unknown')
        bug_description = bug.get('fields', {}).get('System.Description', '')
        repro_steps = bug.get('fields', {}).get('Microsoft.VSTS.TCM.ReproSteps', '')
        parent_id = bug.get('fields', {}).get('System.Parent')

        # Get full parent Feature context including attachments (if bug has parent)
        parent_context_str = ""
        if parent_id:
            self._print_verbose(f" Fetching parent Feature #{parent_id} context...")
            parent_context = self._get_parent_feature_context(parent_id)
            if parent_context.get("found"):
                parent_context_str = self._format_parent_context_for_prompt(parent_context)
                attachment_count = len(parent_context.get("attachments", []))
                if attachment_count > 0:
                    self._print_verbose(f" ✓ Found {attachment_count} attachment(s) in parent Feature")

        prompt = f"""Analyze this Bug and generate the missing diagnostic and fix documentation.

# Bug Information
**Title**: {bug_title}
**Current Description**: {bug_description}
**Current Repro Steps**: {repro_steps}
{parent_context_str}

# Requirements
Generate the following sections (only generate what's marked as missing):
**Missing**: {', '.join(missing_content)}

Use the parent Feature context above (including any referenced attachments) to ensure
the generated content is aligned with the Feature's requirements and architecture.

For each missing section, provide comprehensive content:

1. **Reproduction Steps** (if missing):
   - Preconditions and environment setup
   - Step-by-step instructions to reproduce
   - Expected behavior vs actual behavior
   - Frequency (always/intermittent)

2. **Root Cause Analysis** (if missing):
   - Technical investigation of why the bug occurs
   - Code path analysis
   - Conditions that trigger the bug
   - Related components/systems affected

3. **Solution Design** (if missing):
   - Proposed fix approach
   - Code changes required
   - Alternative solutions considered
   - Risk assessment of the fix
   - Rollback plan if needed

4. **Acceptance Test Design** (if missing):
   - Test cases to verify the fix
   - Regression test scenarios
   - Edge cases to test
   - Performance impact verification

Return as JSON:
{{
  "reproduction_steps": "Step-by-step reproduction...",
  "root_cause_analysis": "Technical root cause...",
  "solution_design": "Proposed fix approach...",
  "acceptance_test_design": "Verification test cases..."
}}"""

        try:
            response = self.interactive_session.ask(prompt)
            content = self.interactive_session.extract_json_from_response(response)
            return content if isinstance(content, dict) else {}
        except Exception as e:
            print(f" ⚠ AI generation failed: {e}")
            return {}

    def _generate_summary_for_attachment(
        self,
        full_content: str,
        task_title: str,
        attachment_filename: str
    ) -> str:
        """
        Generate a summary description for content that will be attached.

        Args:
            full_content: The full content that will be attached
            task_title: Title of the task for context
            attachment_filename: Filename of the attachment for reference

        Returns:
            Summary description referencing the attachment
        """
        if not self.interactive_session or not self.interactive_session.is_available():
            # Fallback: generate a simple summary without AI
            return self._generate_fallback_summary(full_content, attachment_filename)

        prompt = f"""Summarize this task content in 150-250 words. Focus on:
- Key implementation approach
- Main testing considerations
- Critical acceptance criteria

Task: {task_title}

Full Content:
{full_content[:3000]}

Return a plain text summary (NOT JSON). End with:
" See attached file '{attachment_filename}' for complete details."
"""
        try:
            response = self.interactive_session.ask(prompt)
            # Clean up the response - remove any markdown code blocks
            summary = response.strip()
            if summary.startswith("```"):
                lines = summary.split("\n")
                summary = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

            # Ensure attachment reference is included
            if attachment_filename not in summary:
                summary += f"\n\n See attached file '{attachment_filename}' for complete details."

            return summary
        except Exception as e:
            print(f" ⚠ AI summary generation failed: {e}")
            return self._generate_fallback_summary(full_content, attachment_filename)

    def _generate_fallback_summary(self, full_content: str, attachment_filename: str) -> str:
        """Generate a simple summary without AI."""
        # Extract section headers from the content
        sections = []
        for line in full_content.split("\n"):
            if line.startswith("## "):
                sections.append(line.replace("## ", ""))

        summary = "This task includes comprehensive design and test documentation.\n\n"
        if sections:
            summary += "**Sections included:**\n"
            for section in sections:
                summary += f"- {section}\n"
        summary += f"\n See attached file '{attachment_filename}' for complete details."
        return summary

    def _save_content_and_attach(
        self,
        work_item_id: int,
        content: str,
        task_title: str,
        task_type: str
    ) -> Optional[str]:
        """
        Save content to a file and attach it to a work item.

        Args:
            work_item_id: ID of the work item to attach to
            content: Full content to save and attach
            task_title: Title of the task (for filename)
            task_type: Type of task (implementation/test)

        Returns:
            Attachment filename if successful, None otherwise
        """
        if not self.adapter:
            print(" ⚠ No adapter - cannot attach file")
            return None

        # Generate filename
        safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in task_title[:30])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_title}_{task_type}_{timestamp}.md"

        try:
            # Create temp directory and save with proper filename
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, filename)

            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(f"# {task_title}\n\n")
                f.write(f"*Generated: {datetime.now().isoformat()}*\n\n")
                f.write(content)

            # Attach to work item (filename is derived from file_path)
            result = self.adapter.attach_file_to_work_item(
                work_item_id=work_item_id,
                file_path=temp_path,
                comment=f"Auto-generated {task_type} design documentation"
            )

            # Clean up temp file and directory
            try:
                os.unlink(temp_path)
                os.rmdir(temp_dir)
            except Exception:
                pass

            if result:
                print(f" Attached '{filename}' to Task #{work_item_id}")
                return filename
            else:
                print(f" ⚠ Failed to attach file to Task #{work_item_id}")
                return None

        except Exception as e:
            print(f" ⚠ Attachment failed: {e}")
            return None

    def _save_test_plan_attachment(
        self,
        work_item_id: int,
        plan_type: str,
        plan_content: str,
        task_title: str
    ) -> Optional[str]:
        """
        Save a test plan to a file and attach it to a work item.

        Test plans are attached as separate files for the Tester agent to consume
        during sprint execution validation.

        Args:
            work_item_id: ID of the work item to attach to
            plan_type: Type of test plan (unit_test_plan, integration_test_plan, etc.)
            plan_content: Test plan content
            task_title: Title of the task (for filename)

        Returns:
            Attachment filename if successful, None otherwise
        """
        if not self.adapter:
            print(" ⚠ No adapter - cannot attach test plan")
            return None

        # Generate descriptive filename
        plan_label = plan_type.replace("_", "-") # e.g., "unit-test-plan"
        safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in task_title[:20])
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"{plan_label}-{safe_title}-{timestamp}.md"

        try:
            # Create temp directory and save with proper filename
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, filename)

            # Format the test plan content
            plan_title = plan_type.replace("_", " ").title()
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(f"# {plan_title}\n\n")
                f.write(f"**Task**: {task_title}\n")
                f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
                f.write("---\n\n")
                f.write(plan_content)

            # Attach to work item
            result = self.adapter.attach_file_to_work_item(
                work_item_id=work_item_id,
                file_path=temp_path,
                comment=f"Test plan: {plan_title} - for Tester agent validation"
            )

            # Clean up temp file and directory
            try:
                os.unlink(temp_path)
                os.rmdir(temp_dir)
            except Exception:
                pass

            if result:
                return filename
            else:
                return None

        except Exception as e:
            print(f" ⚠ Test plan attachment failed: {e}")
            return None

    def _step_1_query_backlog(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 1: Query target work items (EPICs, Features, Tasks, or Bugs)."""
        self._print_verbose("\n Querying work items...")

        # Check for argument-based filtering
        max_items = None
        target_ids = []

        if self.args:
            max_items = getattr(self.args, 'max_epics', None)
            target_ids = getattr(self.args, 'target_ids', []) or []

            if max_items:
                self._print_verbose(f" Limiting to {max_items} item(s) (argument provided)")
            if target_ids:
                self._print_verbose(f" Targeting specific IDs: {', '.join(map(str, target_ids))} (argument provided)")

        if not self.adapter:
            print("⚠ No adapter - using mock data")
            return self._get_mock_backlog()

        try:
            epics = []
            features = []
            tasks = []
            bugs = []

            if target_ids:
                # Query specific work items by ID
                self._print_verbose(" Fetching targeted work items...")
                feature_ids_for_children = [] # Track Features to get children

                epic_ids_for_children = [] # Track Epics to get child Features

                for work_item_id in target_ids:
                    work_item = self.adapter.get_work_item(work_item_id)
                    if work_item:
                        work_item_type = work_item.get('fields', {}).get('System.WorkItemType', '')
                        if work_item_type == 'Epic':
                            epics.append(work_item)
                            epic_ids_for_children.append(work_item_id)
                        elif work_item_type == 'Feature':
                            features.append(work_item)
                            feature_ids_for_children.append(work_item_id)
                        elif work_item_type == 'Task':
                            tasks.append(work_item)
                        elif work_item_type == 'Bug':
                            bugs.append(work_item)
                        else:
                            print(f" ⚠ Work item {work_item_id} has unknown type: {work_item_type}")
                    else:
                        print(f" ⚠ Work item {work_item_id} not found")

                # Fetch child Features of targeted Epics
                if epic_ids_for_children:
                    self._print_verbose(f" Fetching child Features for {len(epic_ids_for_children)} Epic(s)...")

                    for epic_id in epic_ids_for_children:
                        epic_item = next((e for e in epics if e.get('id') == epic_id), None)
                        if epic_item:
                            relations = epic_item.get('relations', [])
                            for rel in relations:
                                if rel.get('rel') == 'System.LinkTypes.Hierarchy-Forward':
                                    url = rel.get('url', '')
                                    if '/workItems/' in url:
                                        child_id = int(url.split('/workItems/')[-1])
                                        child_item = self.adapter.get_work_item(child_id)
                                        if child_item:
                                            child_type = child_item.get('fields', {}).get('System.WorkItemType', '')
                                            if child_type == 'Feature':
                                                features.append(child_item)
                                                feature_ids_for_children.append(child_id)
                                                self._print_verbose(f" Found child Feature #{child_id}")

                # Also fetch child Tasks/Bugs of targeted Features (using relations)
                if feature_ids_for_children:
                    self._print_verbose(f" Fetching child work items for {len(feature_ids_for_children)} Feature(s)...")

                    # Use relations to find children (same pattern as Epic->Feature)
                    for feature in features:
                        feature_id = feature.get('id')
                        relations = feature.get('relations', [])
                        for rel in relations:
                            if rel.get('rel') == 'System.LinkTypes.Hierarchy-Forward':
                                url = rel.get('url', '')
                                if '/workItems/' in url:
                                    child_id = int(url.split('/workItems/')[-1])
                                    # Avoid duplicates
                                    if child_id in [t.get('id') for t in tasks] or child_id in [b.get('id') for b in bugs]:
                                        continue
                                    child_item = self.adapter.get_work_item(child_id)
                                    if child_item:
                                        child_type = child_item.get('fields', {}).get('System.WorkItemType', '')
                                        if child_type == 'Task':
                                            tasks.append(child_item)
                                            self._print_verbose(f" Found child Task #{child_id} (parent: #{feature_id})")
                                        elif child_type == 'Bug':
                                            bugs.append(child_item)
                                            self._print_verbose(f" Found child Bug #{child_id} (parent: #{feature_id})")
            else:
                # Query all backlog items (Features in New state)
                features = self.adapter.query_work_items(
                    work_item_type='Feature',
                    state='New'
                )

            # Limit by max_items if specified
            total_items = len(epics) + len(features) + len(tasks) + len(bugs)
            if max_items and total_items > max_items:
                self._print_verbose(f" Limiting to first {max_items} item(s)...")
                # Prioritize: EPICs > Features > Bugs > Tasks
                remaining = max_items
                epics = epics[:remaining]
                remaining -= len(epics)
                features = features[:remaining]
                remaining -= len(features)
                bugs = bugs[:remaining]
                remaining -= len(bugs)
                tasks = tasks[:remaining]

            evidence = {
                "epics": epics,
                "features": features,
                "tasks": tasks,
                "bugs": bugs,
                "total_epics": len(epics),
                "total_features": len(features),
                "total_tasks": len(tasks),
                "total_bugs": len(bugs),
                "queried_at": datetime.now().isoformat(),
                "targeted_ids": bool(target_ids),
                "limited_by_max": bool(max_items)
            }

            print(f"\n✓ Found {len(epics)} EPIC(s), {len(features)} Feature(s), {len(tasks)} Task(s), {len(bugs)} Bug(s)")
            return evidence

        except Exception as e:
            print(f"⚠ Query error: {e}")
            return self._get_mock_backlog()

    def _get_mock_backlog(self) -> Dict[str, Any]:
        """Get mock backlog data for testing."""
        return {
            "total_features": 3,
            "features": [
                {
                    "id": "MOCK-101",
                    "title": "User Authentication",
                    "state": "New",
                    "fields": {
                        "System.Description": "Implement user authentication with OAuth",
                        "Microsoft.VSTS.Common.AcceptanceCriteria": "1. Users can login\n2. Tokens stored securely\n3. Auto-refresh implemented",
                        "Microsoft.VSTS.Scheduling.StoryPoints": 8
                    }
                },
                {
                    "id": "MOCK-102",
                    "title": "Data Export",
                    "state": "New",
                    "fields": {
                        "System.Description": "Export data to CSV",
                        "Microsoft.VSTS.Common.AcceptanceCriteria": "1. Export button",
                        "Microsoft.VSTS.Scheduling.StoryPoints": None
                    }
                }
            ],
            "mock": True
        }

    def _step_2_hierarchy_validation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Validate work item hierarchy."""
        self._print_verbose("\n Validating work item hierarchy...")

        # Check for auto-fix argument
        auto_fix = False
        if self.args:
            auto_fix = getattr(self.args, 'auto_fix_hierarchy', False)
            if auto_fix:
                self._print_verbose(" Auto-fix enabled (argument provided)")

        if not self.adapter:
            print("⚠ No adapter - skipping hierarchy validation")
            return {"orphan_features": [], "empty_epics": [], "skipped": True}

        try:
            # Find orphan Features (no parent EPIC)
            self._print_verbose(" - Finding orphan Features...")
            orphan_features = find_orphan_features(self.adapter)

            # Find empty EPICs (no child Features)
            self._print_verbose(" - Finding empty EPICs...")
            empty_epics = find_empty_epics(self.adapter)

            evidence = {
                "orphan_features": orphan_features,
                "empty_epics": empty_epics,
                "orphan_count": len(orphan_features),
                "empty_count": len(empty_epics),
                "validated_at": datetime.now().isoformat(),
                "auto_fixed": False,
                "fixes_applied": []
            }

            print(f"\n✓ Hierarchy validation complete:")
            print(f" - Orphan Features: {len(orphan_features)}")
            print(f" - Empty EPICs: {len(empty_epics)}")

            # Check if we have targeted empty Epics that need breakdown
            target_ids = getattr(self.args, 'target_ids', None) or []
            targeted_empty_epics = [eid for eid in empty_epics if eid in target_ids]

            # Auto-breakdown targeted empty Epics (always, regardless of --auto-fix-hierarchy)
            fixes = []
            if targeted_empty_epics:
                print(f"\n Breaking down {len(targeted_empty_epics)} targeted empty EPIC(s) into Features...")
                for epic_id in targeted_empty_epics:
                    try:
                        epic = self.adapter.get_work_item(epic_id)
                        epic_title = epic.get('fields', {}).get('System.Title', f'EPIC {epic_id}')
                        print(f"\n EPIC #{epic_id}: {epic_title}")

                        # Generate Features using AI
                        features = self._generate_epic_features_with_ai(epic)

                        if features:
                            print(f" ✓ Generated {len(features)} Feature(s)")
                            created_features = []

                            for feature_def in features:
                                # Create Feature in work tracking
                                feature = self.adapter.create_work_item(
                                    work_item_type='Feature',
                                    title=feature_def.get('title', 'Untitled Feature'),
                                    description=feature_def.get('description', ''),
                                    fields={
                                        'Microsoft.VSTS.Scheduling.StoryPoints': feature_def.get('story_points', 13),
                                        'Microsoft.VSTS.Common.AcceptanceCriteria': '<br>'.join(
                                            f"- {c}" for c in feature_def.get('acceptance_criteria', [])
                                        )
                                    }
                                )
                                feature_id = feature.get('id')

                                # Link Feature to Epic
                                self.adapter.link_work_items(
                                    source_id=feature_id,
                                    target_id=epic_id,
                                    relation_type="System.LinkTypes.Hierarchy-Reverse"
                                )

                                created_features.append(feature_id)
                                fixes.append(f"Created Feature {feature_id} under EPIC {epic_id}")
                                print(f" ✓ Created Feature #{feature_id}: {feature_def.get('title')}")

                            # Add created features to backlog for further processing
                            # Step 3 (AI refinement) will validate these Features and create Tasks
                            if created_features:
                                backlog = self.step_evidence.get("1-query-backlog", {})
                                existing_features = backlog.get("features", [])
                                for fid in created_features:
                                    new_feature = self.adapter.get_work_item(fid)
                                    if new_feature:
                                        existing_features.append(new_feature)
                                backlog["features"] = existing_features
                                self.step_evidence["1-query-backlog"] = backlog

                                # Remove this Epic from empty_epics since it now has children
                                empty_epics = [e for e in empty_epics if e != epic_id]
                        else:
                            print(f" ⚠ No Features generated - AI unavailable or generation failed")

                    except Exception as breakdown_err:
                        print(f" Failed to break down EPIC {epic_id}: {breakdown_err}")
                        fixes.append(f"Failed to break down EPIC {epic_id}: {breakdown_err}")

            # Auto-fix other hierarchy issues if requested
            if auto_fix and (orphan_features or empty_epics):
                self._print_verbose(f"\n Applying automatic fixes...")
                # Note: fixes list already initialized above with Epic breakdown fixes

                # Create "Uncategorized" EPIC for orphan Features
                if orphan_features:
                    self._print_verbose(f" - Creating 'Uncategorized' EPIC for {len(orphan_features)} orphan Feature(s)...")
                    try:
                        uncategorized_epic = self.adapter.create_work_item(
                            work_item_type='Epic',
                            title='Uncategorized Features',
                            description='Auto-created EPIC to hold orphaned Features',
                            assigned_to=self.current_user,
                            fields={'System.Tags': 'auto-created; needs-review'}
                        )
                        epic_id = uncategorized_epic.get('id')

                        # Link orphan Features to this EPIC using parent-child relationship
                        for feature_id in orphan_features:
                            try:
                                self.adapter.link_work_items(
                                    source_id=feature_id,
                                    target_id=epic_id,
                                    relation_type="System.LinkTypes.Hierarchy-Reverse"
                                )
                                fixes.append(f"Linked Feature {feature_id} to EPIC {epic_id}")
                            except Exception as link_err:
                                print(f" ⚠ Failed to link {feature_id}: {link_err}")

                        self._print_verbose(f" ✓ Created EPIC {epic_id} and linked {len(orphan_features)} Feature(s)")
                    except Exception as create_err:
                        print(f" Failed to create Uncategorized EPIC: {create_err}")

                # Tag remaining empty EPICs for review (targeted ones already handled above)
                remaining_empty = [eid for eid in empty_epics if eid not in target_ids]
                if remaining_empty:
                    self._print_verbose(f" - Tagging {len(remaining_empty)} empty EPIC(s) for review...")
                    for epic_id in remaining_empty:
                        try:
                            epic = self.adapter.get_work_item(epic_id)
                            current_tags = epic.get('fields', {}).get('System.Tags', '')
                            new_tags = f"{current_tags}; empty-epic; needs-review" if current_tags else "empty-epic; needs-review"
                            self.adapter.update_work_item(work_item_id=epic_id, fields={'System.Tags': new_tags})
                            fixes.append(f"Tagged EPIC {epic_id} as empty")
                        except Exception as tag_err:
                            print(f" ⚠ Failed to tag {epic_id}: {tag_err}")
                    self._print_verbose(f" ✓ Tagged {len(remaining_empty)} empty EPIC(s)")

                evidence["auto_fixed"] = True
                evidence["fixes_applied"] = fixes
                print(f"\n✓ Auto-fix complete: {len(fixes)} fix(es) applied")

            elif fixes:
                # Epic breakdown happened but no other auto-fix was requested
                evidence["auto_fixed"] = True
                evidence["fixes_applied"] = fixes
                print(f"\n✓ Epic breakdown complete: {len(fixes)} change(s) applied")

            if orphan_features or empty_epics:
                # Only show warning if there are remaining issues (not targeted for breakdown)
                remaining_issues = len(orphan_features) + len([e for e in empty_epics if e not in target_ids])
                if remaining_issues > 0 and not auto_fix:
                    print(f"\n⚠ Hierarchy issues found (use --auto-fix-hierarchy to automatically fix)")

            return evidence

        except Exception as e:
            print(f" Hierarchy validation failed: {e}")
            raise

    def _step_3_ai_refinement(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Validate and conform work items against acceptance criteria."""
        self._print_verbose("\nAI Validating work items against acceptance criteria...")

        backlog = self.step_evidence.get("1-query-backlog", {})
        epics = backlog.get("epics", [])
        features = backlog.get("features", [])
        tasks = backlog.get("tasks", [])
        bugs = backlog.get("bugs", [])

        # Results
        valid_items = []
        invalid_items = []
        conformance_actions = []

        # ====================================================================
        # Validate EPICs
        # ====================================================================
        if epics:
            self._print_verbose(f"\n Validating {len(epics)} EPIC(s)...")
            self._print_verbose(f" Requirement: >= {self.EPIC_MIN_TOTAL_HOURS}h of work in child Features")

            for epic in epics:
                epic_id = epic.get('id') or epic.get('fields', {}).get('System.Id')
                epic_title = epic.get('fields', {}).get('System.Title', 'Unknown')

                validation = self._validate_epic(epic)

                if validation["valid"]:
                    valid_items.append({
                        "id": epic_id,
                        "type": "Epic",
                        "title": epic_title,
                        "total_hours": validation["total_hours"]
                    })
                    self._print_verbose(f" ✓ #{epic_id}: {epic_title} ({validation['total_hours']}h)")
                else:
                    invalid_items.append({
                        "id": epic_id,
                        "type": "Epic",
                        "title": epic_title,
                        "missing": validation["missing"],
                        "total_hours": validation["total_hours"]
                    })
                    self._print_verbose(f" ✗ #{epic_id}: {epic_title}")
                    for m in validation["missing"]:
                        self._print_verbose(f" - {m}")

                    # Auto-conform if enabled
                    if self.auto_conform:
                        missing_content = validation.get("missing_content", [])
                        needs_features = validation["total_hours"] < self.EPIC_MIN_TOTAL_HOURS

                        # Use consensus or agent-based approach if enabled
                        if (self.use_consensus or self.use_agents) and (missing_content or needs_features):
                            if self.use_consensus:
                                print(f" CONSENSUS Spawning multi-agent consensus for Epic #{epic_id}...")
                                success, result = self._run_epic_breakdown_consensus(epic, missing_content)
                            else:
                                print(f" AI Spawning Epic breakdown agent for #{epic_id}...")
                                success, result = self._spawn_epic_breakdown_agent(epic, missing_content)

                            if success:
                                # Process Epic updates
                                if result.get("epic_updates"):
                                    conformance_actions.append({
                                        "type": "update_epic",
                                        "epic_id": epic_id,
                                        "content": result["epic_updates"]
                                    })
                                    print(f" ✓ Epic #{epic_id}: Content generated")

                                # Process Features to create
                                if result.get("features_to_create"):
                                    conformance_actions.append({
                                        "type": "create_features",
                                        "epic_id": epic_id,
                                        "features": result["features_to_create"]
                                    })
                                    print(f" ✓ Epic #{epic_id}: {len(result['features_to_create'])} Feature(s) defined")
                            else:
                                print(f" ⚠ Epic #{epic_id}: Agent failed - {result.get('error', 'unknown error')}")

                        else:
                            # Legacy: Use single-session approach
                            if missing_content:
                                self._print_verbose(f" Generating missing EPIC content...")
                                generated = self._generate_epic_content_with_ai(epic, missing_content)
                                if generated:
                                    conformance_actions.append({
                                        "type": "update_epic",
                                        "epic_id": epic_id,
                                        "content": generated
                                    })
                                    self._print_verbose(f" ✓ Will update EPIC with {', '.join(generated.keys())}")

                            # Generate Features to reach minimum hours
                            if needs_features:
                                self._print_verbose(f" Generating Features to reach {self.EPIC_MIN_TOTAL_HOURS}h...")
                                generated_features = self._generate_epic_features_with_ai(
                                    epic,
                                    target_hours=self.EPIC_MIN_TOTAL_HOURS - validation["total_hours"]
                                )
                                if generated_features:
                                    conformance_actions.append({
                                        "type": "create_features",
                                        "epic_id": epic_id,
                                        "features": generated_features
                                    })
                                    self._print_verbose(f" ✓ Will create {len(generated_features)} Feature(s)")

        # ====================================================================
        # Validate Features
        # ====================================================================
        if features:
            print(f"\n Validating {len(features)} Feature(s)...")
            self._print_verbose(f" Requirements: acceptance tests, architecture analysis, 1 impl task, 1 test task")

            # First pass: Validate all Features and collect invalid ones
            features_to_conform = [] # (feature, validation) tuples
            for feature in features:
                feature_id = feature.get('id') or feature.get('fields', {}).get('System.Id')
                feature_title = feature.get('fields', {}).get('System.Title', 'Unknown')

                validation = self._validate_feature(feature)

                if validation["valid"]:
                    valid_items.append({
                        "id": feature_id,
                        "type": "Feature",
                        "title": feature_title
                    })
                    self._print_verbose(f" ✓ #{feature_id}: {feature_title}")
                else:
                    invalid_items.append({
                        "id": feature_id,
                        "type": "Feature",
                        "title": feature_title,
                        "missing": validation["missing"]
                    })
                    self._print_verbose(f" ✗ #{feature_id}: {feature_title}")
                    for m in validation["missing"]:
                        self._print_verbose(f" - {m}")

                    # Collect for conformance if enabled
                    if self.auto_conform:
                        features_to_conform.append((feature, validation))

            # Second pass: Conform invalid Features
            if features_to_conform:
                if self.use_consensus:
                    # Consensus-based: Process Features with multi-agent consensus
                    print(f"\nCONSENSUS Processing {len(features_to_conform)} Feature(s) with multi-agent consensus...")
                    skipped_count = 0
                    results = []

                    for idx, (feature, validation) in enumerate(features_to_conform, 1):
                        feature_id = feature.get('id') or feature.get('fields', {}).get('System.Id')
                        print(f"\n[{idx}/{len(features_to_conform)}] Running consensus for Feature #{feature_id}...")

                        missing_content = validation.get("missing_content", [])
                        impl_missing = any('implementation_task' in m for m in validation.get("missing", []))
                        test_missing = any('test_task' in m for m in validation.get("missing", []))
                        needs_task = impl_missing or test_missing

                        success, result = self._run_feature_to_task_consensus(
                            feature, missing_content, needs_task
                        )
                        results.append((feature, success, result))

                elif self.use_agents:
                    # Agent-based: Process in parallel batches of 3
                    print(f"\nAI Processing {len(features_to_conform)} Feature(s) with parallel agents...")
                    batch_size = self.AGENT_MAX_WORKERS
                    skipped_count = 0
                    results = []

                    for batch_start in range(0, len(features_to_conform), batch_size):
                        batch = features_to_conform[batch_start:batch_start + batch_size]
                        batch_features = [f for f, v in batch]
                        batch_validations = [v for f, v in batch]

                        batch_num = batch_start // batch_size + 1
                        total_batches = (len(features_to_conform) + batch_size - 1) // batch_size
                        print(f" Batch {batch_num}/{total_batches}: Processing {len(batch)} Feature(s) in parallel...")

                        # Spawn agents in parallel
                        batch_results = self._spawn_feature_agents_parallel(batch_features, batch_validations)
                        results.extend(batch_results)

                # Process results from either consensus or agent approach
                if self.use_consensus or self.use_agents:
                    print(f"\nProcessing {len(results)} consensus result(s)...")
                    for feature, success, result in results:
                        feature_id = feature.get('id') or feature.get('fields', {}).get('System.Id')
                        feature_title = feature.get('fields', {}).get('System.Title', 'Unknown')
                        self._print_verbose(f"  Feature #{feature_id}: success={success}, keys={list(result.keys()) if isinstance(result, dict) else 'not-dict'}")

                        if success:
                            # Process Feature updates - handle both nested and flat formats
                            feature_updates = result.get("feature_updates")
                            if not feature_updates:
                                # Check for flat format (keys at top level)
                                update_keys = ["detailed_description", "architecture_analysis", "security_analysis", "acceptance_criteria"]
                                if any(key in result for key in update_keys):
                                    feature_updates = {k: result[k] for k in update_keys if k in result}

                            if feature_updates:
                                conformance_actions.append({
                                    "type": "update_feature",
                                    "feature_id": feature_id,
                                    "content": feature_updates
                                })
                                print(f" ✓ Feature #{feature_id}: Content generated")

                            # Process Implementation Task (handles both consensus and agent output)
                            task_data = None
                            if result.get("implementation_task"):
                                task_data = result["implementation_task"]
                            elif result.get("tasks_to_create"):
                                tasks = result["tasks_to_create"]
                                if tasks and len(tasks) > 0:
                                    task_data = tasks[0]

                            if task_data:
                                conformance_actions.append({
                                    "type": "create_task",
                                    "parent_id": feature_id,
                                    "task_type": "implementation",
                                    "title": task_data.get("title", f"Implement: {feature_title}"),
                                    "content": task_data,
                                    "attach_test_plans": True
                                })
                                print(f" ✓ Feature #{feature_id}: Implementation task generated")
                        else:
                            skipped_count += 1
                            print(f" ⚠ Feature #{feature_id}: Failed - {result.get('error', 'unknown')}")

                    if skipped_count > 0:
                        print(f"\n⚠ {skipped_count} Feature(s) skipped due to failures")

                else:
                    # Legacy: Sequential single-session approach
                    for idx, (feature, validation) in enumerate(features_to_conform, 1):
                        feature_id = feature.get('id') or feature.get('fields', {}).get('System.Id')
                        feature_title = feature.get('fields', {}).get('System.Title', 'Unknown')

                        # Generate missing content
                        missing_content = validation.get("missing_content", [])
                        if missing_content:
                            print(f" [{idx}/{len(features_to_conform)}] Generating content for Feature #{feature_id}...")
                            generated = self._generate_feature_content_with_ai(feature, missing_content)
                            if generated:
                                conformance_actions.append({
                                    "type": "update_feature",
                                    "feature_id": feature_id,
                                    "content": generated
                                })
                                print(f" [{idx}/{len(features_to_conform)}] ✓ Feature #{feature_id}: Content generated")
                            else:
                                print(f" [{idx}/{len(features_to_conform)}] ⚠ Feature #{feature_id}: AI generation failed")

                        # Create missing implementation task
                        impl_missing = any('implementation_task' in m for m in validation["missing"])
                        test_missing = any('test_task' in m for m in validation["missing"])

                        if impl_missing or test_missing:
                            print(f" [{idx}/{len(features_to_conform)}] Generating implementation task for Feature #{feature_id}...")
                            impl_content = self._generate_new_task_content_with_ai(feature, "implementation")
                            conformance_actions.append({
                                "type": "create_task",
                                "parent_id": feature_id,
                                "task_type": "implementation",
                                "title": f"Implement: {feature_title}",
                                "content": impl_content,
                                "attach_test_plans": True
                            })
                            if impl_content:
                                print(f" [{idx}/{len(features_to_conform)}] ✓ Implementation task generated")
                            else:
                                print(f" [{idx}/{len(features_to_conform)}] ⚠ Implementation task: AI generation failed")

        # ====================================================================
        # Validate Tasks
        # ====================================================================
        if tasks:
            print(f"\n Validating {len(tasks)} Task(s)...")
            self._print_verbose(f" Requirements: detailed design, unit/integration/acceptance test design")

            for idx, task in enumerate(tasks, 1):
                task_id = task.get('id') or task.get('fields', {}).get('System.Id')
                task_title = task.get('fields', {}).get('System.Title', 'Unknown')
                current_tags = (task.get('fields', {}).get('System.Tags', '') or '').lower()

                validation = self._validate_task(task)

                if validation["valid"]:
                    valid_items.append({
                        "id": task_id,
                        "type": "Task",
                        "title": task_title
                    })
                    self._print_verbose(f" ✓ #{task_id}: {task_title}")

                    # Re-entrant: Add 'groomed' tag if valid but not already tagged
                    if 'groomed' not in current_tags:
                        conformance_actions.append({
                            "type": "add_groomed_tag",
                            "work_item_id": task_id,
                            "work_item_type": "Task"
                        })
                        self._print_verbose(f"  Will add 'groomed' tag")
                else:
                    invalid_items.append({
                        "id": task_id,
                        "type": "Task",
                        "title": task_title,
                        "missing": validation["missing"]
                    })
                    self._print_verbose(f" ✗ #{task_id}: {task_title}")
                    for m in validation["missing"]:
                        self._print_verbose(f" - {m}")

                    # Auto-conform if enabled
                    if self.auto_conform:
                        print(f" [{idx}/{len(tasks)}] Generating content for Task #{task_id}...")
                        generated = self._generate_task_content_with_ai(task, validation["missing"])
                        if generated:
                            conformance_actions.append({
                                "type": "update_task",
                                "task_id": task_id,
                                "content": generated
                            })
                            print(f" [{idx}/{len(tasks)}] ✓ Task #{task_id}: Content generated")
                        else:
                            print(f" [{idx}/{len(tasks)}] ⚠ Task #{task_id}: AI generation failed or unavailable")

        # ====================================================================
        # Validate Bugs
        # ====================================================================
        if bugs:
            print(f"\n Validating {len(bugs)} Bug(s)...")
            self._print_verbose(f" Requirements: repro steps, root cause, solution design, acceptance tests")

            for idx, bug in enumerate(bugs, 1):
                bug_id = bug.get('id') or bug.get('fields', {}).get('System.Id')
                bug_title = bug.get('fields', {}).get('System.Title', 'Unknown')
                current_tags = (bug.get('fields', {}).get('System.Tags', '') or '').lower()

                validation = self._validate_bug(bug)

                if validation["valid"]:
                    valid_items.append({
                        "id": bug_id,
                        "type": "Bug",
                        "title": bug_title
                    })
                    self._print_verbose(f" ✓ #{bug_id}: {bug_title}")

                    # Re-entrant: Add 'groomed' tag if valid but not already tagged
                    if 'groomed' not in current_tags:
                        conformance_actions.append({
                            "type": "add_groomed_tag",
                            "work_item_id": bug_id,
                            "work_item_type": "Bug"
                        })
                        self._print_verbose(f"  Will add 'groomed' tag")
                else:
                    invalid_items.append({
                        "id": bug_id,
                        "type": "Bug",
                        "title": bug_title,
                        "missing": validation["missing"]
                    })
                    self._print_verbose(f" ✗ #{bug_id}: {bug_title}")
                    for m in validation["missing"]:
                        self._print_verbose(f" - {m}")

                    # Auto-conform if enabled
                    if self.auto_conform:
                        print(f" [{idx}/{len(bugs)}] Generating content for Bug #{bug_id}...")
                        generated = self._generate_bug_content_with_ai(bug, validation["missing_content"])
                        if generated:
                            conformance_actions.append({
                                "type": "update_bug",
                                "bug_id": bug_id,
                                "content": generated
                            })
                            print(f" [{idx}/{len(bugs)}] ✓ Bug #{bug_id}: Content generated")
                        else:
                            print(f" [{idx}/{len(bugs)}] ⚠ Bug #{bug_id}: AI generation failed or unavailable")

        # Build evidence - map to legacy format for compatibility
        ready_features = [
            {"id": i["id"], "title": i["title"], "recommendation": "Ready"}
            for i in valid_items if i["type"] == "Feature"
        ]
        unready_features = [
            {"id": i["id"], "title": i["title"], "missing": i.get("missing", []),
             "recommendation": f"Needs: {', '.join(i.get('missing', []))}"}
            for i in invalid_items if i["type"] == "Feature"
        ]

        evidence = {
            # Full validation results
            "valid_items": valid_items,
            "invalid_items": invalid_items,
            "conformance_actions": conformance_actions,
            # Legacy format for compatibility
            "ready_features": ready_features,
            "unready_features": unready_features,
            "ready_count": len(ready_features),
            "unready_count": len(unready_features),
            # Summary counts
            "valid_count": len(valid_items),
            "invalid_count": len(invalid_items),
            "actions_count": len(conformance_actions),
            "analyzed_at": datetime.now().isoformat()
        }

        # Print summary
        print(f"\n✓ Validation complete:")
        print(f" - Valid items: {len(valid_items)}")
        print(f" - Invalid items: {len(invalid_items)}")
        if conformance_actions:
            print(f" - Conformance actions queued: {len(conformance_actions)}")

        return evidence

    def _step_4_approval_gate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 4: Human approval gate (BLOCKING).

        Execution HALTS here until user approves/rejects.
        """
        print("\n" + "=" * 70)
        print("⏸ STEP 4: HUMAN APPROVAL GATE")
        print("=" * 70)
        print("\n BLOCKING CHECKPOINT - Execution halted pending approval")

        refinement = self.step_evidence.get("3-ai-refinement", {})
        valid_items = refinement.get("valid_items", [])
        invalid_items = refinement.get("invalid_items", [])
        conformance_actions = refinement.get("conformance_actions", [])
        ready_features = refinement.get("ready_features", [])

        # Display summary for approval
        print(f"\n Validation Summary:")
        print(f" - Valid items: {len(valid_items)}")
        print(f" - Invalid items: {len(invalid_items)}")

        if conformance_actions:
            print(f"\n Conformance Actions to Apply ({len(conformance_actions)}):")
            for i, action in enumerate(conformance_actions, 1):
                action_type = action.get("type", "unknown")
                content = action.get("content", {})

                if action_type == "create_features":
                    features = action.get('features', [])
                    print(f"\n {i}. CREATE {len(features)} Feature(s) under EPIC #{action.get('epic_id')}")
                    for feat in features:
                        print(f" └─ {feat.get('title', 'Untitled')}")
                        if feat.get('description'):
                            print(f" {self._truncate_content(feat['description'], 80)}")

                elif action_type == "update_epic":
                    print(f"\n {i}. UPDATE EPIC #{action.get('epic_id')} with business analysis")
                    if content:
                        for line in self._format_content_preview(content):
                            print(line)

                elif action_type == "update_feature":
                    print(f"\n {i}. UPDATE Feature #{action.get('feature_id')} with generated content")
                    if content:
                        for line in self._format_content_preview(content):
                            print(line)

                elif action_type == "create_task":
                    task_type = action.get('task_type', 'unknown')
                    title = action.get('title', 'Untitled')
                    parent_id = action.get('parent_id')
                    print(f"\n {i}. CREATE {task_type.upper()} Task under Feature #{parent_id}")
                    print(f" Title: {title}")
                    if content:
                        for line in self._format_content_preview(content):
                            print(line)
                    else:
                        print(f" ⚠ NO AI CONTENT - Task will have generic description")
                        print(f" (AI generation failed - consider re-running)")

                elif action_type == "update_task":
                    print(f"\n {i}. UPDATE Task #{action.get('task_id')} with generated content")
                    if content:
                        for line in self._format_content_preview(content):
                            print(line)

                elif action_type == "update_bug":
                    print(f"\n {i}. UPDATE Bug #{action.get('bug_id')} with root cause/solution analysis")
                    if content:
                        for line in self._format_content_preview(content):
                            print(line)

        if ready_features:
            print(f"\n✓ Features to transition to Ready state:")
            for feature in ready_features:
                print(f" - #{feature['id']}: {feature['title']}")

        print("\n" + "─" * 70)
        print("DECISION REQUIRED:")
        if conformance_actions:
            print(" yes = Apply conformance actions and transition ready items")
        else:
            print(" yes = Transition ready items to Ready state")
        print(" no = Cancel (no changes)")
        print("─" * 70 + "\n")

        # BLOCKING CALL - Execution halts here
        # Flush any buffered stdin input from keystrokes during AI processing
        flush_stdin()

        # Re-prompt on empty/invalid input to prevent accidental rejection
        response = None
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                user_input = input("Approve changes? (yes/no): ").strip().lower()

                if user_input in ("yes", "y"):
                    response = "yes"
                    break
                elif user_input in ("no", "n"):
                    response = "no"
                    break
                elif user_input == "":
                    print("⚠ Empty input - please type 'yes' or 'no'")
                else:
                    print(f"⚠ Invalid input '{user_input}' - please type 'yes' or 'no'")

            except EOFError:
                print("\n⚠ Unexpected end of input (EOF received)")
                print(" This may happen if stdin was closed unexpectedly.")
                print(" Treating as rejection for safety. Re-run the script to try again.")
                response = "no"
                break
            except KeyboardInterrupt:
                print("\n\n Approval cancelled by user (Ctrl+C)")
                response = "no"
                break

        # If max attempts reached without valid input, reject for safety
        if response is None:
            print(f"\n⚠ No valid response after {max_attempts} attempts - rejecting for safety")
            response = "no"

        approved = response == "yes"

        evidence = {
            "approved": approved,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "features_to_transition": len(ready_features) if approved else 0,
            "conformance_actions_approved": len(conformance_actions) if approved else 0
        }

        if approved:
            actions_msg = f", {len(conformance_actions)} conformance action(s)" if conformance_actions else ""
            print(f"\n User APPROVED - Will apply changes: {len(ready_features)} transition(s){actions_msg}")
        else:
            print("\n User REJECTED - No changes will be made")

        return evidence

    def _step_5_state_transitions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 5: Apply conformance actions and state transitions."""
        self._print_verbose("\n Applying approved changes...")

        # Check if approved
        approval = self.step_evidence.get("4-approval-gate", {})
        if not approval.get("approved"):
            print("⏭ Skipped - User did not approve changes")
            return {"transitions": [], "conformance_applied": [], "skipped": True}

        refinement = self.step_evidence.get("3-ai-refinement", {})
        ready_features = refinement.get("ready_features", [])
        conformance_actions = refinement.get("conformance_actions", [])

        if not self.adapter:
            print("⚠ No adapter - cannot apply changes")
            return {"transitions": [], "conformance_applied": [], "mock": True}

        transitions = []
        conformance_applied = []

        try:
            # ================================================================
            # Apply conformance actions
            # ================================================================
            if conformance_actions:
                self._print_verbose(f"\n Applying {len(conformance_actions)} conformance action(s)...")

                for action in conformance_actions:
                    action_type = action.get("type")

                    if action_type == "create_features":
                        # Create Features under an EPIC
                        epic_id = action.get("epic_id")
                        for feature_def in action.get("features", []):
                            try:
                                result = self.adapter.create_work_item(
                                    work_item_type='Feature',
                                    title=feature_def.get("title", "Generated Feature"),
                                    description=feature_def.get("description", ""),
                                    assigned_to=self.current_user,
                                    parent_id=epic_id,
                                    fields={
                                        'Microsoft.VSTS.Scheduling.StoryPoints': feature_def.get("story_points", 5),
                                        'Microsoft.VSTS.Common.AcceptanceCriteria': "\n".join(
                                            f"- {c}" for c in feature_def.get("acceptance_criteria", [])
                                        )
                                    }
                                )
                                conformance_applied.append({
                                    "action": "create_feature",
                                    "epic_id": epic_id,
                                    "feature_id": result.get("id"),
                                    "title": feature_def.get("title")
                                })
                                self._print_verbose(f" ✓ Created Feature #{result.get('id')}: {feature_def.get('title')}")
                            except Exception as e:
                                print(f" ⚠ Failed to create Feature: {e}")

                    elif action_type == "update_feature":
                        # Update Feature with generated content
                        feature_id = action.get("feature_id")
                        content = action.get("content", {})
                        try:
                            # Build description update
                            current = self.adapter.get_work_item(feature_id)
                            current_desc = current.get('fields', {}).get('System.Description', '') or ''
                            current_criteria = current.get('fields', {}).get('Microsoft.VSTS.Common.AcceptanceCriteria', '') or ''
                            feature_title = current.get('fields', {}).get('System.Title', 'Feature')

                            additions = []
                            if content.get("detailed_description"):
                                additions.append(f"## Detailed Description\n\n{content['detailed_description']}")
                            if content.get("architecture_analysis"):
                                additions.append(f"## Architecture Analysis\n\n{content['architecture_analysis']}")
                            if content.get("security_analysis"):
                                additions.append(f"## Security Analysis\n\n{content['security_analysis']}")

                            new_content = "\n\n".join(additions)

                            # Check if additions exceed attachment threshold
                            use_attachment = len(new_content) > self.CONTENT_ATTACHMENT_THRESHOLD

                            if use_attachment:
                                self._print_verbose(f" New content is {len(new_content)} chars - will use attachment")
                                # Attach the new content as a file
                                attachment_filename = self._save_content_and_attach(
                                    work_item_id=feature_id,
                                    content=new_content,
                                    task_title=feature_title,
                                    task_type="feature-analysis"
                                )

                                if attachment_filename:
                                    # Generate summary and append to existing description
                                    summary = self._generate_summary_for_attachment(
                                        full_content=new_content,
                                        task_title=feature_title,
                                        attachment_filename=attachment_filename
                                    )
                                    new_desc = current_desc + "\n\n---\n\n" + summary
                                else:
                                    # Attachment failed, inline the content
                                    new_desc = current_desc + "\n\n" + new_content
                            else:
                                new_desc = current_desc + "\n\n" + new_content

                            fields_to_update = {'System.Description': new_desc}

                            # Add acceptance criteria if generated
                            if content.get("acceptance_criteria"):
                                if current_criteria:
                                    new_criteria = current_criteria + f"\n\n## Generated Acceptance Criteria\n{content['acceptance_criteria']}"
                                else:
                                    new_criteria = content['acceptance_criteria']
                                fields_to_update['Microsoft.VSTS.Common.AcceptanceCriteria'] = new_criteria

                            self.adapter.update_work_item(
                                work_item_id=feature_id,
                                fields=fields_to_update
                            )
                            conformance_applied.append({
                                "action": "update_feature",
                                "feature_id": feature_id,
                                "content_added": list(content.keys()),
                                "used_attachment": use_attachment
                            })
                            attachment_msg = " (with attachment)" if use_attachment else ""
                            self._print_verbose(f" ✓ Updated Feature #{feature_id} with generated content{attachment_msg}")
                        except Exception as e:
                            print(f" ⚠ Failed to update Feature #{feature_id}: {e}")

                    elif action_type == "create_task":
                        # Create Task under a Feature with AI-generated content
                        parent_id = action.get("parent_id")
                        task_title = action.get("title", "Generated Task")
                        task_type = action.get("task_type", "implementation")
                        content = action.get("content", {})

                        # Check if we need to attach test plans as separate files
                        attach_test_plans = action.get("attach_test_plans", False)

                        # Build comprehensive description from ALL content keys dynamically
                        # This handles Implementation Tasks, Testing Tasks, and legacy formats
                        description_parts = []
                        test_plan_attachments = {} # Store test plans to attach as files

                        for key, value in content.items():
                            if value and isinstance(value, str) and value.strip():
                                # Check if this is a test plan that should be attached as file
                                if attach_test_plans and key in self.TEST_PLAN_ATTACHMENT_FIELDS:
                                    # Store for separate attachment
                                    test_plan_attachments[key] = value
                                else:
                                    # Convert key to readable header (e.g., "detailed_design" -> "Detailed Design")
                                    header = key.replace("_", " ").title()
                                    description_parts.append(f"## {header}\n\n{value}")

                        if description_parts:
                            full_content = "\n\n".join(description_parts)
                        else:
                            # Fallback description when AI generation failed
                            # Include reference to parent Feature for context
                            full_content = f"""## {task_type.title()} Task

**Note**: AI content generation was unavailable. This task requires manual elaboration.

### Parent Feature
- Feature #{parent_id}: {task_title.replace('Implement: ', '').replace('Test: ', '')}

### Required Content
Please add the following sections manually:
"""
                            if task_type == "implementation":
                                full_content += """
- **Detailed Design**: Implementation approach and technical decisions
- **Function Specifications**: Detailed specs of each component
- **Unit Test Requirements**: Tests with 80% coverage target
- **Integration Test Requirements**: Component interaction tests
- **Acceptance Test Requirements**: Tests for acceptance criteria
"""
                            else: # test task
                                full_content += """
- **Test Type Validation**: Verify presence of all required test types
- **Coverage Validation**: Verify 80% minimum coverage
- **Feature Coverage Validation**: All acceptance criteria have tests
- **Falsifiability Validation**: Tests can detect actual failures
- **Test Results Report**: Execution results with evidence
"""

                        # Check if content exceeds attachment threshold
                        use_attachment = len(full_content) > self.CONTENT_ATTACHMENT_THRESHOLD

                        try:
                            if use_attachment:
                                # Create task first with placeholder, then attach file
                                self._print_verbose(f" Content is {len(full_content)} chars - will use attachment")
                                result = self.adapter.create_work_item(
                                    work_item_type='Task',
                                    title=task_title,
                                    description="*Full design documentation attached - see attachment for details.*",
                                    assigned_to=self.current_user,
                                    parent_id=parent_id,
                                    fields={
                                        'System.Tags': f"auto-generated; groomed; {task_type}; has-attachment"
                                    }
                                )
                                task_id = result.get("id")

                                # Attach the full content as a file
                                attachment_filename = self._save_content_and_attach(
                                    work_item_id=task_id,
                                    content=full_content,
                                    task_title=task_title,
                                    task_type=task_type
                                )

                                # Generate summary description referencing the attachment
                                if attachment_filename:
                                    summary = self._generate_summary_for_attachment(
                                        full_content=full_content,
                                        task_title=task_title,
                                        attachment_filename=attachment_filename
                                    )
                                    # Update the task with the summary
                                    self.adapter.update_work_item(
                                        work_item_id=task_id,
                                        fields={'System.Description': summary}
                                    )
                                    self._print_verbose(f" ✓ Created Task #{task_id}: {task_title} (with attachment)")
                                else:
                                    # Attachment failed, use inline content as fallback
                                    self.adapter.update_work_item(
                                        work_item_id=task_id,
                                        fields={'System.Description': full_content}
                                    )
                                    self._print_verbose(f" ✓ Created Task #{task_id}: {task_title} (inline - attachment failed)")
                            else:
                                # Content is short enough, use inline description
                                result = self.adapter.create_work_item(
                                    work_item_type='Task',
                                    title=task_title,
                                    description=full_content,
                                    assigned_to=self.current_user,
                                    parent_id=parent_id,
                                    fields={
                                        'System.Tags': f"auto-generated; groomed; {task_type}"
                                    }
                                )
                                task_id = result.get("id")
                                self._print_verbose(f" ✓ Created Task #{task_id}: {task_title}")

                            # Attach test plans as separate files (for Tester agent)
                            attached_test_plans = []
                            if test_plan_attachments and task_id:
                                for plan_key, plan_content in test_plan_attachments.items():
                                    try:
                                        plan_filename = self._save_test_plan_attachment(
                                            work_item_id=task_id,
                                            plan_type=plan_key,
                                            plan_content=plan_content,
                                            task_title=task_title
                                        )
                                        if plan_filename:
                                            attached_test_plans.append(plan_filename)
                                            self._print_verbose(f" Attached {plan_key}: {plan_filename}")
                                    except Exception as e:
                                        print(f" ⚠ Failed to attach {plan_key}: {e}")

                            conformance_applied.append({
                                "action": "create_task",
                                "parent_id": parent_id,
                                "task_id": task_id,
                                "title": task_title,
                                "task_type": task_type,
                                "content_added": list(content.keys()) if content else [],
                                "used_attachment": use_attachment,
                                "test_plan_attachments": attached_test_plans
                            })
                        except Exception as e:
                            print(f" ⚠ Failed to create Task: {e}")

                    elif action_type == "update_task":
                        # Update Task with generated content
                        task_id = action.get("task_id")
                        content = action.get("content", {})
                        try:
                            # Build description update
                            current = self.adapter.get_work_item(task_id)
                            current_desc = current.get('fields', {}).get('System.Description', '') or ''
                            task_title = current.get('fields', {}).get('System.Title', 'Task')

                            additions = []
                            if content.get("detailed_design"):
                                additions.append(f"## Detailed Design\n\n{content['detailed_design']}")
                            if content.get("unit_test_design"):
                                additions.append(f"## Unit Test Design\n\n{content['unit_test_design']}")
                            if content.get("integration_test_design"):
                                additions.append(f"## Integration Test Design\n\n{content['integration_test_design']}")
                            if content.get("acceptance_test_design"):
                                additions.append(f"## Acceptance Test Design\n\n{content['acceptance_test_design']}")

                            new_content = "\n\n".join(additions)

                            # Check if additions exceed attachment threshold
                            use_attachment = len(new_content) > self.CONTENT_ATTACHMENT_THRESHOLD

                            if use_attachment:
                                self._print_verbose(f" New content is {len(new_content)} chars - will use attachment")
                                # Attach the new content as a file
                                attachment_filename = self._save_content_and_attach(
                                    work_item_id=task_id,
                                    content=new_content,
                                    task_title=task_title,
                                    task_type="update"
                                )

                                if attachment_filename:
                                    # Generate summary and append to existing description
                                    summary = self._generate_summary_for_attachment(
                                        full_content=new_content,
                                        task_title=task_title,
                                        attachment_filename=attachment_filename
                                    )
                                    new_desc = current_desc + "\n\n---\n\n" + summary
                                else:
                                    # Attachment failed, inline the content
                                    new_desc = current_desc + "\n\n" + new_content
                            else:
                                new_desc = current_desc + "\n\n" + new_content

                            # Add 'groomed' tag to existing tags
                            current_tags = current.get('fields', {}).get('System.Tags', '') or ''
                            if 'groomed' not in current_tags.lower():
                                new_tags = f"{current_tags}; groomed" if current_tags else "groomed"
                            else:
                                new_tags = current_tags

                            self.adapter.update_work_item(
                                work_item_id=task_id,
                                fields={
                                    'System.Description': new_desc,
                                    'System.Tags': new_tags
                                }
                            )
                            conformance_applied.append({
                                "action": "update_task",
                                "task_id": task_id,
                                "content_added": list(content.keys()),
                                "used_attachment": use_attachment
                            })
                            attachment_msg = " (with attachment)" if use_attachment else ""
                            self._print_verbose(f" ✓ Updated Task #{task_id} with generated content{attachment_msg}")
                        except Exception as e:
                            print(f" ⚠ Failed to update Task #{task_id}: {e}")

                    elif action_type == "update_epic":
                        # Update EPIC with business analysis content
                        epic_id = action.get("epic_id")
                        content = action.get("content", {})
                        try:
                            # Build description update
                            current = self.adapter.get_work_item(epic_id)
                            current_desc = current.get('fields', {}).get('System.Description', '') or ''

                            additions = []
                            if content.get("business_analysis"):
                                additions.append(f"\n\n## Business Analysis\n{content['business_analysis']}")
                            if content.get("success_criteria"):
                                additions.append(f"\n\n## Success Criteria\n{content['success_criteria']}")
                            if content.get("scope_definition"):
                                additions.append(f"\n\n## Scope Definition\n{content['scope_definition']}")

                            new_desc = current_desc + "".join(additions)

                            self.adapter.update_work_item(
                                work_item_id=epic_id,
                                fields={'System.Description': new_desc}
                            )
                            conformance_applied.append({
                                "action": "update_epic",
                                "epic_id": epic_id,
                                "content_added": list(content.keys())
                            })
                            self._print_verbose(f" ✓ Updated EPIC #{epic_id} with business analysis")
                        except Exception as e:
                            print(f" ⚠ Failed to update EPIC #{epic_id}: {e}")

                    elif action_type == "update_bug":
                        # Update Bug with root cause/solution analysis
                        bug_id = action.get("bug_id")
                        content = action.get("content", {})
                        try:
                            # Build description and repro steps update
                            current = self.adapter.get_work_item(bug_id)
                            current_desc = current.get('fields', {}).get('System.Description', '') or ''
                            current_repro = current.get('fields', {}).get('Microsoft.VSTS.TCM.ReproSteps', '') or ''

                            # Add analysis to description
                            desc_additions = []
                            if content.get("root_cause_analysis"):
                                desc_additions.append(f"\n\n## Root Cause Analysis\n{content['root_cause_analysis']}")
                            if content.get("solution_design"):
                                desc_additions.append(f"\n\n## Solution Design\n{content['solution_design']}")
                            if content.get("acceptance_test_design"):
                                desc_additions.append(f"\n\n## Acceptance Test Design\n{content['acceptance_test_design']}")

                            new_desc = current_desc + "".join(desc_additions)

                            # Add reproduction steps if generated
                            fields_to_update = {'System.Description': new_desc}
                            if content.get("reproduction_steps"):
                                if current_repro:
                                    new_repro = current_repro + f"\n\n## Enhanced Reproduction Steps\n{content['reproduction_steps']}"
                                else:
                                    new_repro = content['reproduction_steps']
                                fields_to_update['Microsoft.VSTS.TCM.ReproSteps'] = new_repro

                            # Add 'groomed' tag to existing tags
                            current_tags = current.get('fields', {}).get('System.Tags', '') or ''
                            if 'groomed' not in current_tags.lower():
                                new_tags = f"{current_tags}; groomed" if current_tags else "groomed"
                            else:
                                new_tags = current_tags
                            fields_to_update['System.Tags'] = new_tags

                            self.adapter.update_work_item(
                                work_item_id=bug_id,
                                fields=fields_to_update
                            )
                            conformance_applied.append({
                                "action": "update_bug",
                                "bug_id": bug_id,
                                "content_added": list(content.keys())
                            })
                            self._print_verbose(f" ✓ Updated Bug #{bug_id} with root cause/solution analysis")
                        except Exception as e:
                            print(f" ⚠ Failed to update Bug #{bug_id}: {e}")

                    elif action_type == "add_groomed_tag":
                        # Add 'groomed' tag to valid work items that don't have it
                        work_item_id = action.get("work_item_id")
                        work_item_type = action.get("work_item_type", "Item")
                        try:
                            # Get current work item to read existing tags
                            current = self.adapter.get_work_item(work_item_id)
                            current_tags = current.get('fields', {}).get('System.Tags', '') or ''

                            # Add 'groomed' tag if not present
                            if 'groomed' not in current_tags.lower():
                                new_tags = f"{current_tags}; groomed" if current_tags else "groomed"
                                self.adapter.update_work_item(
                                    work_item_id=work_item_id,
                                    fields={'System.Tags': new_tags}
                                )
                                conformance_applied.append({
                                    "action": "add_groomed_tag",
                                    "work_item_id": work_item_id,
                                    "work_item_type": work_item_type
                                })
                                self._print_verbose(f"  Added 'groomed' tag to {work_item_type} #{work_item_id}")
                            else:
                                self._print_verbose(f" ✓ {work_item_type} #{work_item_id} already has 'groomed' tag")
                        except Exception as e:
                            print(f" ⚠ Failed to add 'groomed' tag to #{work_item_id}: {e}")

                print(f"\n✓ Applied {len(conformance_applied)} conformance action(s)")

            # ================================================================
            # Mark ready Features with 'ready-for-sprint' tag
            # Note: We use tags instead of state transitions because
            # Features only have states: New → In Progress → Done
            # ================================================================
            if ready_features:
                print(f"\n Marking {len(ready_features)} Feature(s) as ready for sprint...")

                for feature in ready_features:
                    feature_id = feature['id']
                    feature_title = feature['title']

                    try:
                        # Add 'ready-for-sprint' tag to indicate grooming is complete
                        current_tags = feature.get('tags', '')
                        if 'ready-for-sprint' not in current_tags.lower():
                            new_tags = f"{current_tags}; ready-for-sprint" if current_tags else "ready-for-sprint"
                            self.adapter.update_work_item(
                                work_item_id=feature_id,
                                fields={'System.Tags': new_tags},
                                verify=False
                            )

                        transitions.append({
                            "feature_id": feature_id,
                            "title": feature_title,
                            "action": "tagged-ready-for-sprint",
                            "tagged_at": datetime.now().isoformat()
                        })

                        print(f" ✓ Marked #{feature_id}: {feature_title}")
                    except Exception as e:
                        print(f" ⚠ Failed to mark #{feature_id}: {e}")

                print(f"\n✓ Marked {len(transitions)} Feature(s) as ready for sprint")

            evidence = {
                "transitions": transitions,
                "conformance_applied": conformance_applied,
                "transition_count": len(transitions),
                "conformance_count": len(conformance_applied),
                "applied_at": datetime.now().isoformat()
            }

            return evidence

        except Exception as e:
            print(f" Changes failed: {e}")
            raise

    def _step_6_verification(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 6: Verify state changes (external source of truth)."""
        self._print_verbose("\n Verifying state transitions...")

        transitions_data = self.step_evidence.get("5-state-transitions", {})

        if transitions_data.get("skipped"):
            self._print_verbose("⏭ Skipped - No transitions to verify")
            return {"verified": True, "skipped": True}

        transitions = transitions_data.get("transitions", [])

        if not transitions:
            self._print_verbose("✓ No transitions to verify")
            return {"verified": True, "count": 0}

        if not self.adapter:
            print("⚠ No adapter - cannot verify transitions")
            return {"verified": False, "error": "No adapter"}

        try:
            verified_transitions = []

            for transition in transitions:
                feature_id = transition['feature_id']

                # Query adapter for current state (external source of truth)
                feature = self.adapter.get_work_item(feature_id)

                if not feature:
                    raise ValueError(
                        f"Feature {feature_id} claimed marked but doesn't exist in tracking system"
                    )

                # Get actual tags - we verify the tag was applied, not state
                actual_tags = feature.get('fields', {}).get('System.Tags', '')

                # Verify 'ready-for-sprint' tag is present
                if 'ready-for-sprint' not in actual_tags.lower():
                    raise ValueError(
                        f"Feature {feature_id} missing 'ready-for-sprint' tag: got '{actual_tags}'"
                    )

                verified_transitions.append({
                    "feature_id": feature_id,
                    "title": transition['title'],
                    "verified_tags": actual_tags
                })

                print(f" ✓ Verified {feature_id}: has 'ready-for-sprint' tag")

            evidence = {
                "verified": True,
                "verified_transitions": verified_transitions,
                "verified_count": len(verified_transitions),
                "verified_at": datetime.now().isoformat()
            }

            print(f"\n✓ Verified {len(verified_transitions)} transition(s)")
            return evidence

        except Exception as e:
            print(f" Verification failed: {e}")
            raise

    def _step_7_save_report(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 7: Save grooming report."""
        self._print_verbose("\n Saving grooming report...")

        backlog = self.step_evidence.get("1-query-backlog", {})
        hierarchy = self.step_evidence.get("2-hierarchy-validation", {})
        refinement = self.step_evidence.get("3-ai-refinement", {})
        transitions = self.step_evidence.get("5-state-transitions", {})
        verification = self.step_evidence.get("6-verification", {})

        # Generate report content
        report_lines = [
            "# Backlog Grooming Report",
            "",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Workflow ID:** {self.workflow_id}",
            "",
            "## Backlog Summary",
            "",
            f"- **Total Features:** {backlog.get('total_features', 0)}",
            "",
            "## Hierarchy Validation",
            "",
            f"- **Orphan Features:** {hierarchy.get('orphan_count', 0)}",
            f"- **Empty EPICs:** {hierarchy.get('empty_count', 0)}",
            ""
        ]

        if hierarchy.get('orphan_features'):
            report_lines.extend([
                "### Orphan Features (No Parent EPIC)",
                ""
            ])
            for feature_id in hierarchy['orphan_features']:
                report_lines.append(f"- {feature_id}")
            report_lines.append("")

        if hierarchy.get('empty_epics'):
            report_lines.extend([
                "### Empty EPICs (No Child Features)",
                ""
            ])
            for epic_id in hierarchy['empty_epics']:
                report_lines.append(f"- {epic_id}")
            report_lines.append("")

        # Add refinement analysis
        ready_features = refinement.get("ready_features", [])
        unready_features = refinement.get("unready_features", [])

        report_lines.extend([
            "## Refinement Analysis",
            "",
            f"- **Ready Features:** {len(ready_features)}",
            f"- **Needs Refinement:** {len(unready_features)}",
            ""
        ])

        if ready_features:
            report_lines.extend([
                "### Ready for Sprint",
                ""
            ])
            for feature in ready_features:
                report_lines.append(f"- **{feature['id']}:** {feature['title']}")
            report_lines.append("")

        if unready_features:
            report_lines.extend([
                "### Needs Refinement",
                ""
            ])
            for feature in unready_features:
                report_lines.append(f"- **{feature['id']}:** {feature['title']}")
                report_lines.append(f" - Missing: {', '.join(feature['missing'])}")
            report_lines.append("")

        # Add transitions if any
        if not transitions.get("skipped"):
            transitioned = transitions.get("transitions", [])
            if transitioned:
                report_lines.extend([
                    "## State Transitions",
                    "",
                    f"Transitioned {len(transitioned)} Feature(s) to Ready state:",
                    ""
                ])
                for trans in transitioned:
                    report_lines.append(f"- **{trans['feature_id']}:** {trans['title']}")
                report_lines.append("")

                # Add verification status
                if verification.get("verified"):
                    report_lines.append(" All transitions verified in tracking system")
                else:
                    report_lines.append("⚠ Verification incomplete")
                report_lines.append("")

        # Add footer
        report_lines.extend([
            "---",
            "",
            "*Generated by Trustable AI Development Workbench*"
        ])

        report_content = "\n".join(report_lines)

        # Save report to .claude/grooming/
        report_dir = Path(".claude/grooming")
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        report_file = report_dir / f"grooming-{timestamp}.md"

        # Write with UTF-8 encoding
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        evidence = {
            "report_file": str(report_file),
            "saved_at": datetime.now().isoformat()
        }

        print(f"\n✓ Report saved: {report_file}")
        return evidence


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Backlog Grooming Workflow - External enforcement with hierarchy validation"
    )
    parser.add_argument(
        "--workflow-id",
        help="Workflow ID (defaults to timestamp-based ID)"
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Disable state checkpointing"
    )

    # Non-interactive mode arguments (Task #1243)
    parser.add_argument(
        "--max-epics",
        type=int,
        help="Maximum number of EPICs to process (limits scope of grooming)"
    )
    parser.add_argument(
        "--target-ids",
        nargs='+',
        type=int,
        help="Specific work item IDs to target (EPICs, Features, or Tasks)"
    )
    parser.add_argument(
        "--auto-fix-hierarchy",
        action="store_true",
        help="Automatically fix hierarchy issues (orphan Features, empty EPICs)"
    )
    parser.add_argument(
        "--no-auto-conform",
        action="store_true",
        help="Disable automatic field conformance (auto-conform is enabled by default)"
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Disable AI analysis (use simple heuristics instead)"
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Disable interactive AI collaboration"
    )
    parser.add_argument(
        "--use-agents",
        action="store_true",
        help="Use tree-based agent spawning for content generation (parallel, context-isolated)"
    )
    parser.add_argument(
        "--use-consensus",
        action="store_true",
        help="Use multi-agent consensus for Epic/Feature breakdown (Feature #1344)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output (show detailed workflow trace)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.max_epics is not None and args.max_epics < 1:
        print(" Error: --max-epics must be >= 1")
        sys.exit(1)

    # Generate workflow ID if not provided
    workflow_id = args.workflow_id or f"grooming-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Create and execute workflow (AI, interactive, and auto-conform enabled by default)
    workflow = BacklogGroomingWorkflow(
        workflow_id=workflow_id,
        enable_checkpoints=not args.no_checkpoints,
        use_ai=not args.no_ai,
        interactive=not args.no_interactive,
        auto_conform=not args.no_auto_conform, # Default: True
        use_agents=args.use_agents, # Tree-based agent spawning
        use_consensus=args.use_consensus, # Multi-agent consensus (Feature #1344)
        args=args
    )

    from cli.console import console

    try:
        success = workflow.execute()
        if success:
            console.print()
            console.print("─" * 80)
            console.print("[bold #71E4D1]  Backlog grooming complete![/bold #71E4D1]")
            console.print("─" * 80)
            console.print()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        console.print()
        console.print("[#758B9B]Backlog grooming cancelled by user.[/#758B9B]")
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
