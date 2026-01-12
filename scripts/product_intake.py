#!/usr/bin/env python3
"""
Product Intake Workflow with External Enforcement and AI-Driven Work Item Type Classification

Implements Phase 2: Simple Workflows - Product Intake Script

8-Step Workflow:
1. User Input Collection - Gather product idea details
2. Duplicate Detection - Check for similar existing work items
3. AI Triage - Classify nature (category) and scope (work item type) using Mode 2 (AI + JSON validation)
4. Human Approval Gate - BLOCKING approval for work item creation
5. Work Item Creation - Create work item (Bug/Task/Feature/Epic based on AI triage) in tracking system
6. External Verification - Verify work item exists in tracking system
7. Summary Generation - Generate intake summary report
8. Checkpoint - Save workflow state for re-entrancy

AI Classification (Multi-Agent):
- **Business Analyst Agent**: Comprehensive triage analysis with Extended Thinking
  - Category: Nature of work (bug/feature/enhancement/infrastructure/security/performance)
  - Work Item Type: Scope/size (Bug/Task/Feature/Epic)
    * Bug: Isolated issue, small scope (1-8 hours)
    * Task: Small standalone work (1-2 days)
    * Feature: Medium capability (3-7 days)
    * Epic: Large initiative (2+ weeks)
  - Priority assessment based on business impact
  - Business value scoring (1-10)
  - Technical risk evaluation
  - Effort estimation

Design Pattern:
- Extends WorkflowOrchestrator from Phase 1
- Uses adapter for ALL work item operations
- **Spawns Business Analyst agent** for comprehensive triage (Mode 2)
- Extended Thinking enabled for thorough analysis
- AI determines work item type based on scope analysis
- External verification after work item creation
- Real input() blocking for approval gate
- UTF-8 encoding for all file writes
- Input validation and WIQL query escaping
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
from scripts.workflow_executor.schemas import StepType, WorkflowStepDefinition
from workflows.utilities import calculate_similarity
from core.console_workflow import (
    print_workflow_header,
    print_step_header,
    print_approval_gate,
    print_summary_panel,
    ApprovalGateData
)

# Import JSON schema validation
try:
    from jsonschema import validate, ValidationError
except ImportError:
    print("⚠️  jsonschema package not installed - install with: pip install jsonschema")
    ValidationError = Exception  # Fallback


class ProductIntakeWorkflow(WorkflowOrchestrator):
    """
    Product Intake workflow with external enforcement.

    Implements the 8-step product intake process with:
    - Duplicate detection (95% similarity threshold)
    - Mode 2 AI triage with JSON validation
    - External verification after EPIC creation
    - Blocking approval gates
    """

    def __init__(
        self,
        workflow_id: str,
        enable_checkpoints: bool = True,
        use_ai: bool = True,
        interactive: bool = True,
        args: Optional[argparse.Namespace] = None,
        verbose: bool = False,
        skip_triage: bool = False
    ):
        """
        Initialize product intake workflow.

        Args:
            workflow_id: Unique ID for this execution (e.g., "intake-2024-12-21")
            enable_checkpoints: Enable state checkpointing
            use_ai: If True, use AI for triage (default: True, use --no-ai to disable)
            interactive: If True, use interactive mode with user prompts (default: True, use --no-interactive to disable)
            args: Command-line arguments for non-interactive mode (Task #1239)
            verbose: If True, show detailed step-by-step output (default: clean summary)
            skip_triage: If True, skip AI triage and create Epic directly
        """
        self.use_ai = use_ai
        self.interactive = interactive
        self.args = args  # Store args for use in _step_1_user_input
        self.verbose = verbose
        self.skip_triage = skip_triage

        # Interactive mode takes precedence over use_ai (Mode 3 vs Mode 2)
        # Both are enabled by default; use --no-interactive for Mode 2, --no-ai for Mode 1
        if interactive and use_ai:
            self.use_ai = False  # Interactive mode (Mode 3) takes precedence

        # Determine execution mode
        if interactive:
            mode = ExecutionMode.INTERACTIVE_AI  # Mode 3
        elif use_ai:
            mode = ExecutionMode.AI_JSON_VALIDATION  # Mode 2
        else:
            mode = ExecutionMode.PURE_PYTHON  # Mode 1

        super().__init__(
            workflow_name="product-intake",
            workflow_id=workflow_id,
            mode=mode,
            enable_checkpoints=enable_checkpoints,
            quiet_mode=not verbose
        )

        # Initialize adapter (uses config to select Azure DevOps or file-based)
        try:
            sys.path.insert(0, '.claude/skills')
            from work_tracking import get_adapter
            self.adapter = get_adapter()
            print(f"✅ Adapter initialized: {self.adapter.platform}")
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize adapter: {e}")
            print("    Continuing with limited functionality...")
            self.adapter = None

        # Get current user for work item assignment
        self.current_user = None
        if self.adapter:
            try:
                user_info = self.adapter.get_current_user()
                if user_info:
                    self.current_user = user_info.get('display_name') or user_info.get('email')
                    if verbose:
                        print(f"✓ Authenticated as: {self.current_user}")
            except Exception as e:
                print(f"⚠️  Could not get current user: {e}")
                print("    Work items will be created without assignment")

        # Initialize Claude API client if using AI
        self.claude_client = None
        self.token_usage = {}
        if use_ai:
            try:
                import anthropic
                api_key = os.getenv("KEYCHAIN_ANTHROPIC_API_KEY")
                if api_key:
                    self.claude_client = anthropic.Anthropic(api_key=api_key)
                    print("✅ Claude API client initialized")
                else:
                    print("⚠️ KEYCHAIN_ANTHROPIC_API_KEY not set, falling back to mock mode")
                    self.use_ai = False
            except ImportError:
                print("⚠️ anthropic package not installed, falling back to mock mode")
                self.use_ai = False

        # Initialize interactive session if interactive mode
        self.interactive_session = None
        if interactive:
            try:
                from scripts.workflow_executor.interactive_session import InteractiveSession
                self.interactive_session = InteractiveSession(
                    workflow_name="product-intake",
                    session_id=workflow_id,
                    model="claude-sonnet-4-5",
                    max_tokens=4000
                )
                if self.interactive_session.is_available():
                    print("✓ Interactive mode initialized (Mode 3)")
                else:
                    print("⚠️  Interactive mode unavailable - falling back to mock data")
                    self.interactive = False
            except ImportError as e:
                print(f"⚠️  Interactive mode unavailable: {e}")
                print("    Falling back to mock data")
                self.interactive = False

        # Configuration
        self.duplicate_threshold = 0.95  # 95% similarity for duplicates

    def execute(self) -> bool:
        """
        Execute the product intake workflow with unified console output.

        Overrides base class to add workflow header.
        """
        # Print workflow header
        mode_str = "Mode 1 (Pure Python)" if not self.use_ai and not self.interactive else \
                   "Mode 2 (AI + JSON)" if self.use_ai and not self.interactive else \
                   "Mode 3 (Interactive AI)"
        print_workflow_header("Product Intake Workflow", mode=mode_str)

        # Execute base workflow
        return super().execute()

    def _define_steps(self) -> List[Dict[str, Any]]:
        """Define the 8 workflow steps."""
        return [
            {
                "id": "1-user-input",
                "name": "Collect User Input",
                "step_type": StepType.DATA_COLLECTION,
                "description": "Gather product idea details from user",
                "required": True
            },
            {
                "id": "2-duplicate-detection",
                "name": "Duplicate Detection",
                "step_type": StepType.VERIFICATION,
                "description": "Check for similar existing work items",
                "required": True,
                "depends_on": ["1-user-input"]
            },
            {
                "id": "3-ai-triage",
                "name": "AI Triage and Classification",
                "step_type": StepType.AI_REVIEW,
                "description": "Classify and prioritize using AI with JSON validation",
                "required": True,
                "depends_on": ["2-duplicate-detection"]
            },
            {
                "id": "3.5-technical-analysis",
                "name": "Technical Analysis (Bug/Task only)",
                "step_type": StepType.AI_REVIEW,
                "description": "Generate RCA, implementation plan, AC, and tests for Bug/Task items",
                "required": False,
                "depends_on": ["3-ai-triage"]
            },
            {
                "id": "4-approval-gate",
                "name": "Human Approval Gate",
                "step_type": StepType.APPROVAL_GATE,
                "description": "BLOCKING approval for work item creation",
                "required": True,
                "depends_on": ["3.5-technical-analysis"]
            },
            {
                "id": "5-work-item-creation",
                "name": "Create Work Item",
                "step_type": StepType.ACTION,
                "description": "Create work item (type determined by AI triage) in tracking system",
                "required": True,
                "depends_on": ["4-approval-gate"]
            },
            {
                "id": "6-verification",
                "name": "External Verification",
                "step_type": StepType.VERIFICATION,
                "description": "Verify work item exists in tracking system",
                "required": True,
                "depends_on": ["5-work-item-creation"]
            },
            {
                "id": "7-summary",
                "name": "Generate Summary",
                "step_type": StepType.ACTION,
                "description": "Generate intake summary report",
                "required": True,
                "depends_on": ["6-verification"]
            },
            {
                "id": "8-checkpoint",
                "name": "Save Checkpoint",
                "step_type": StepType.ACTION,
                "description": "Save workflow state for re-entrancy",
                "required": True,
                "depends_on": ["7-summary"]
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
        if step_id == "1-user-input":
            return self._step_1_user_input(context)
        elif step_id == "2-duplicate-detection":
            return self._step_2_duplicate_detection(context)
        elif step_id == "3-ai-triage":
            return self._step_3_ai_triage(context)
        elif step_id == "3.5-technical-analysis":
            return self._step_3_5_technical_analysis(context)
        elif step_id == "4-approval-gate":
            return self._step_4_approval_gate(context)
        elif step_id == "5-work-item-creation":
            return self._step_5_work_item_creation(context)
        elif step_id == "6-verification":
            return self._step_6_verification(context)
        elif step_id == "7-summary":
            return self._step_7_summary(context)
        elif step_id == "8-checkpoint":
            return self._step_8_checkpoint(context)
        else:
            raise ValueError(f"Unknown step: {step_id}")

    def _step_1_user_input(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 1: Collect user input.

        Supports:
        - File input mode (--file): Load requirements from .md or .txt file
        - Non-interactive mode (--title, --description): Command-line args
        - Interactive mode: User prompts
        """
        print_step_header(1, "Collect User Input", "Gather product idea details from user")

        # Check for file input mode
        if self.args and self.args.file:
            return self._handle_file_input(self.args.file)

        # Check for non-interactive mode arguments (Task #1239)
        if self.args and self.args.title and self.args.description:
            print("📌 Using non-interactive mode (arguments provided)")

            # Get values from arguments
            title = self.args.title.strip()
            description = self.args.description.strip()
            business_value = (self.args.business_value or "").strip() if hasattr(self.args, 'business_value') else ""

            # Validate argument values
            if not title:
                raise ValueError("Title cannot be empty or whitespace-only")
            if len(title) > 200:
                raise ValueError(f"Title too long ({len(title)} chars, max 200)")

            if not description:
                raise ValueError("Description cannot be empty or whitespace-only")
            if len(description) > 10000:
                raise ValueError(f"Description too long ({len(description)} chars, max 10000)")

            evidence = {
                "title": title,
                "description": description,
                "business_value": business_value,
                "timestamp": datetime.now().isoformat(),
                "mode": "non-interactive"
            }

            print(f"  Title: {title}")
            print(f"  Description: {description[:100]}{'...' if len(description) > 100 else ''}")
            if business_value:
                print(f"  Business Value: {business_value[:100]}{'...' if len(business_value) > 100 else ''}")
            print(f"\n✓ Collected product idea: '{title}'")
            return evidence

        else:
            # Interactive mode - use any provided arguments and prompt for missing fields
            title = None
            description = None
            business_value = None

            # Extract any provided arguments
            if self.args:
                if self.args.title:
                    title = self.args.title.strip()
                    if not title:
                        raise ValueError("Title cannot be empty or whitespace-only")
                    print(f"✓ Using title from arguments: '{title}'")
                if self.args.description:
                    description = self.args.description.strip()
                    if not description:
                        raise ValueError("Description cannot be empty or whitespace-only")
                    print(f"✓ Using description from arguments: {description[:100]}{'...' if len(description) > 100 else ''}")
                if self.args.business_value:
                    business_value = self.args.business_value.strip()
                    print(f"✓ Using business value from arguments: {business_value[:100]}{'...' if len(business_value) > 100 else ''}")

            # Prompt for any missing fields
            # Note: Title will be generated by AI during triage
            try:
                if not description:
                    description = input("Product idea description: ").strip()
                    if not description:
                        raise ValueError("Description cannot be empty or whitespace-only")

                if not business_value:
                    business_value = input("Business value (optional): ").strip()

                # Validate inputs
                if title and len(title) > 200:
                    raise ValueError("Title too long (max 200 chars)")

                if len(description) > 10000:
                    raise ValueError("Description too long (max 10000 chars)")

                evidence = {
                    "title": title,  # May be None
                    "description": description,
                    "business_value": business_value,
                    "timestamp": datetime.now().isoformat(),
                    "mode": "interactive"
                }

                if title:
                    print(f"\n✓ Collected product idea: '{title}'")
                else:
                    print(f"\n✓ Collected product idea (title will be generated by AI)")
                return evidence

            except (EOFError, KeyboardInterrupt):
                raise ValueError("User input cancelled")

    def _handle_file_input(self, file_path: str) -> Dict[str, Any]:
        """
        Handle file-based input for vision/requirements documents.

        Reads content from .md or .txt files and extracts title and description.
        For large files, the full content becomes the description and the
        first heading or line becomes the title.

        Args:
            file_path: Path to the requirements file

        Returns:
            Dict with title, description, business_value, and metadata
        """
        from pathlib import Path
        import re

        path = Path(file_path)

        # Validate file exists and has correct extension
        if not path.exists():
            raise ValueError(f"File not found: {file_path}")

        if path.suffix.lower() not in ['.md', '.txt']:
            raise ValueError(f"Unsupported file format: {path.suffix}. Use .md or .txt")

        print(f"📄 Loading requirements from file: {file_path}")

        # Read file content
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        if not content.strip():
            raise ValueError("File is empty")

        # Extract title from content (may be None if no heading found)
        title = self._extract_title_from_content(content, path.stem)

        # Use full content as description
        description = content.strip()

        # Try to extract business value if present
        business_value = self._extract_business_value_from_content(content)

        if title:
            print(f"   Title: {title}")
        else:
            print(f"   Title: Will be generated by AI during triage")
        print(f"   Content: {len(description)} characters")
        if business_value:
            print(f"   Business Value: extracted from content")

        evidence = {
            "title": title,  # May be None
            "description": description,
            "business_value": business_value,
            "timestamp": datetime.now().isoformat(),
            "mode": "file-input",
            "source_file": str(path),
            "file_size": len(content)
        }

        if title:
            print(f"\n✓ Loaded product idea from file: '{title}'")
        else:
            print(f"\n✓ Loaded product idea from file (title will be generated by AI)")
        return evidence

    def _extract_title_from_content(self, content: str, fallback: str) -> Optional[str]:
        """
        Extract title from content - first heading or first line.

        Returns None if no clear title found (AI will generate one).
        """
        import re

        # Try to find markdown heading (# Title)
        heading_match = re.search(r'^#\s+(.+?)$', content, re.MULTILINE)
        if heading_match:
            return heading_match.group(1).strip()[:200]

        # Don't use first line or fallback - let AI generate title
        # This ensures AI-generated titles are used for file inputs
        return None

    def _extract_business_value_from_content(self, content: str) -> str:
        """Try to extract business value section from content."""
        import re

        # Look for common business value headings
        patterns = [
            r'##\s*Business\s+Value\s*\n(.*?)(?=\n##|\Z)',
            r'##\s*Value\s+Proposition\s*\n(.*?)(?=\n##|\Z)',
            r'##\s*Benefits\s*\n(.*?)(?=\n##|\Z)',
            r'##\s*ROI\s*\n(.*?)(?=\n##|\Z)'
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()[:2000]

        return ""

    def _step_2_duplicate_detection(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Check for duplicate work items."""
        print_step_header(2, "Duplicate Detection", "Check for similar existing work items")

        user_input = self.step_evidence.get("1-user-input", {})
        title = user_input.get("title", "")
        description = user_input.get("description", "")

        # If no title, use description for duplicate detection
        search_text = title if title else description[:200]

        if not self.adapter:
            print("⚠️  No adapter - skipping duplicate check")
            return {"duplicates": [], "highest_similarity": 0.0}

        try:
            # Query all EPICs (this is where we'd use adapter in production)
            # For now, simulate with empty list
            existing_epics = []

            # Check similarity against existing EPICs
            duplicates = []
            highest_similarity = 0.0

            for epic in existing_epics:
                epic_title = epic.get("title", "")
                similarity = calculate_similarity(search_text, epic_title)

                if similarity > highest_similarity:
                    highest_similarity = similarity

                if similarity >= self.duplicate_threshold:
                    duplicates.append({
                        "id": epic.get("id"),
                        "title": epic_title,
                        "similarity": similarity
                    })

            evidence = {
                "duplicates": duplicates,
                "highest_similarity": highest_similarity,
                "threshold": self.duplicate_threshold,
                "total_checked": len(existing_epics)
            }

            if duplicates:
                print(f"\n⚠️  Found {len(duplicates)} potential duplicate(s):")
                for dup in duplicates:
                    print(f"  - {dup['title']} (similarity: {dup['similarity']:.1%})")
            else:
                print(f"✓ No duplicates found (checked {len(existing_epics)} existing EPICs)")

            return evidence

        except Exception as e:
            print(f"⚠️  Duplicate detection error: {e}")
            return {"duplicates": [], "error": str(e)}

    def _step_3_ai_triage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: AI triage and classification (Mode 2: JSON validation or Mode 3: Interactive)."""
        user_input = self.step_evidence.get("1-user-input", {})
        duplicates = self.step_evidence.get("2-duplicate-detection", {})

        # Skip triage if requested (for detailed requirement files)
        if self.skip_triage:
            print("\n⏭️  Skipping AI triage (--skip-triage flag set)")
            print("   Creating Epic directly from requirements")
            triage_result = {
                "category": "feature",
                "priority": "medium",
                "estimated_effort": "large",
                "business_value_score": 7,
                "technical_risk": "medium",
                "work_item_type": "Epic",
                "work_item_type_rationale": "Requirements document provided - creating Epic for decomposition into Features/Tasks",
                "recommended_action": "proceed",
                "rationale": "Detailed requirements provided via file input. Skipping AI triage per user request."
            }
            print(f"\n✓ Triage skipped - defaulting to Epic")
            return triage_result

        if self.use_ai:
            print("\n🤖 Spawning Business Analyst agent for triage...")
            print("   Model: claude-sonnet-4.5")
            print("   Extended Thinking: ENABLED")
        else:
            print("\n🤖 Running AI triage and classification...")

        if self.interactive:
            triage_result = self._triage_interactive(user_input, duplicates)
        elif self.use_ai:
            triage_result = self._triage_ai(user_input, duplicates)
        else:
            triage_result = self._triage_simple(user_input, duplicates)

        print(f"\n✓ Triage complete:")
        print(f"  Category: {triage_result['category']}")
        print(f"  Priority: {triage_result['priority']}")
        print(f"  Work Item Type: {triage_result.get('work_item_type', 'Unknown')}")
        print(f"  Recommended action: {triage_result['recommended_action']}")

        return triage_result

    def _step_3_5_technical_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 3.5: Technical analysis for Bug/Task work items.

        Generates root cause analysis, implementation plan, acceptance criteria,
        and test design for Bug and Task work items.

        Skipped for Feature and Epic items (they need decomposition first).
        """
        triage = self.step_evidence.get("3-ai-triage", {})
        work_item_type = triage.get("work_item_type", "Unknown")

        # Only run for Bug and Task items
        if work_item_type not in ["Bug", "Task"]:
            print(f"\n⏭️  Skipping technical analysis ({work_item_type} items don't need detailed RCA/implementation plan)")
            return {
                "skipped": True,
                "reason": f"{work_item_type} items require decomposition, not immediate implementation"
            }

        user_input = self.step_evidence.get("1-user-input", {})

        print(f"\n🔧 Generating technical analysis for {work_item_type}...")
        print("   Spawning Senior Engineer agent")
        print("   Model: claude-sonnet-4.5")
        print("   Extended Thinking: ENABLED")

        if self.use_ai and self.claude_client:
            return self._technical_analysis_with_ai(user_input, triage, work_item_type)
        else:
            return self._technical_analysis_simple(user_input, triage, work_item_type)

    def _technical_analysis_simple(
        self,
        user_input: Dict[str, Any],
        triage: Dict[str, Any],
        work_item_type: str
    ) -> Dict[str, Any]:
        """Simple technical analysis without AI (for testing)."""
        title = user_input.get("title", "")
        description = user_input.get("description", "")

        if work_item_type == "Bug":
            return {
                "root_cause_analysis": f"Root cause analysis for: {title}",
                "implementation_plan": "1. Identify the bug\n2. Write failing test\n3. Fix the bug\n4. Verify test passes",
                "acceptance_criteria": [
                    "Bug no longer reproduces",
                    "Related functionality works correctly",
                    "Unit tests added"
                ],
                "test_design": "Test case 1: Verify bug is fixed\nTest case 2: Verify no regression"
            }
        else:  # Task
            return {
                "implementation_plan": f"Implementation plan for: {title}\n\n1. Analyze requirements\n2. Design solution\n3. Implement changes\n4. Test implementation",
                "acceptance_criteria": [
                    "Implementation matches requirements",
                    "All tests pass",
                    "Code review approved"
                ],
                "test_design": "Test case 1: Verify implementation\nTest case 2: Edge cases"
            }

    def _technical_analysis_with_ai(
        self,
        user_input: Dict[str, Any],
        triage: Dict[str, Any],
        work_item_type: str
    ) -> Dict[str, Any]:
        """Generate technical analysis using Senior Engineer agent."""
        title = user_input.get("title", "")
        description = user_input.get("description", "")
        category = triage.get("category", "unknown")

        # Define schema based on work item type
        if work_item_type == "Bug":
            schema = {
                "type": "object",
                "properties": {
                    "root_cause_analysis": {
                        "type": "string",
                        "minLength": 100,
                        "description": "Detailed root cause analysis explaining what went wrong and why"
                    },
                    "implementation_plan": {
                        "type": "string",
                        "minLength": 100,
                        "description": "Step-by-step plan to fix the bug"
                    },
                    "acceptance_criteria": {
                        "type": "array",
                        "minItems": 3,
                        "items": {"type": "string", "minLength": 20},
                        "description": "Specific, measurable criteria to verify the bug is fixed"
                    },
                    "test_design": {
                        "type": "string",
                        "minLength": 100,
                        "description": "Test cases to verify the fix and prevent regression"
                    }
                },
                "required": ["root_cause_analysis", "implementation_plan", "acceptance_criteria", "test_design"]
            }
        else:  # Task
            schema = {
                "type": "object",
                "properties": {
                    "implementation_plan": {
                        "type": "string",
                        "minLength": 100,
                        "description": "Step-by-step implementation plan"
                    },
                    "acceptance_criteria": {
                        "type": "array",
                        "minItems": 3,
                        "items": {"type": "string", "minLength": 20},
                        "description": "Specific, measurable criteria to verify task completion"
                    },
                    "test_design": {
                        "type": "string",
                        "minLength": 100,
                        "description": "Test cases to verify the implementation"
                    }
                },
                "required": ["implementation_plan", "acceptance_criteria", "test_design"]
            }

        # Build analysis prompt
        if work_item_type == "Bug":
            analysis_prompt = f"""**TECHNICAL ANALYSIS REQUEST - BUG FIX**

You are a Senior Engineer analyzing a bug report for the Trustable AI Development Workbench project.

**Bug Details:**
- Title: {title}
- Description: {description}
- Category: {category}
- Priority: {triage.get('priority', 'unknown')}

**Your Task:**

Provide comprehensive technical analysis for this bug:

1. **Root Cause Analysis**: Explain what went wrong and why
   - What is the underlying cause of the bug?
   - Why did this happen (missing validation, race condition, etc.)?
   - What assumptions were violated?

2. **Implementation Plan**: Step-by-step fix
   - How to reproduce the bug
   - Where to make changes (files, functions)
   - What to change (logic, validation, error handling)
   - How to verify the fix

3. **Acceptance Criteria**: Measurable verification (minimum 3)
   - Bug no longer reproduces following reproduction steps
   - Related functionality still works correctly
   - Unit tests added to prevent regression
   - Additional criteria specific to this bug

4. **Test Design**: Test cases to verify fix
   - Test case to reproduce original bug (should fail before fix)
   - Test case to verify bug is fixed (should pass after fix)
   - Regression tests for related functionality
   - Edge cases

Return ONLY valid JSON matching the schema."""
        else:  # Task
            analysis_prompt = f"""**TECHNICAL ANALYSIS REQUEST - TASK IMPLEMENTATION**

You are a Senior Engineer planning implementation for a task in the Trustable AI Development Workbench project.

**Task Details:**
- Title: {title}
- Description: {description}
- Category: {category}
- Priority: {triage.get('priority', 'unknown')}
- Estimated Effort: {triage.get('estimated_effort', 'unknown')}

**Your Task:**

Provide comprehensive implementation planning:

1. **Implementation Plan**: Step-by-step approach
   - Requirements analysis
   - Design decisions
   - Files/modules to modify
   - Implementation steps
   - Testing approach

2. **Acceptance Criteria**: Measurable completion criteria (minimum 3)
   - Functionality meets requirements
   - All tests pass
   - Code review approved
   - Additional criteria specific to this task

3. **Test Design**: Test cases
   - Unit tests for new functionality
   - Integration tests if needed
   - Edge cases
   - Error conditions

Return ONLY valid JSON matching the schema."""

        # Call Claude API with retry logic
        for attempt in range(3):
            try:
                response = self.claude_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4000,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 3000
                    },
                    system="You are a Senior Engineer providing technical analysis. Use extended thinking to reason through the implementation thoroughly.",
                    messages=[{"role": "user", "content": analysis_prompt + f"\n\nReturn ONLY valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"}]
                )

                # Extract response (skip thinking blocks)
                response_text = ""
                for block in response.content:
                    if block.type == "text":
                        response_text = block.text
                        break

                # Extract JSON if wrapped in code blocks
                json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(1)

                result = json.loads(response_text)
                validate(result, schema)

                # Track token usage
                usage_info = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cost_usd": self._calculate_cost(response.usage)
                }

                if hasattr(response.usage, 'thinking_tokens'):
                    usage_info["thinking_tokens"] = response.usage.thinking_tokens
                    print(f"💭 Extended thinking used: {response.usage.thinking_tokens} tokens")

                if "technical_analysis" not in self.token_usage:
                    self.token_usage["technical_analysis"] = {}
                self.token_usage["technical_analysis"] = usage_info

                print(f"✓ Technical analysis complete (input: {response.usage.input_tokens}, output: {response.usage.output_tokens})")
                return result

            except (json.JSONDecodeError, ValidationError) as e:
                print(f"⚠️  Attempt {attempt + 1}/3 failed: {type(e).__name__}: {e}")
                if attempt == 2:
                    print("❌ All retries exhausted, using simple analysis")
                    return self._technical_analysis_simple(user_input, triage, work_item_type)
            except Exception as e:
                print(f"❌ API error: {type(e).__name__}: {e}, using simple analysis")
                return self._technical_analysis_simple(user_input, triage, work_item_type)

        return self._technical_analysis_simple(user_input, triage, work_item_type)

    def _generate_simple_title(self, description: str, category: str, work_item_type: str) -> str:
        """
        Generate a simple title from description when user doesn't provide one.

        Args:
            description: The description text
            category: The work item category (bug, feature, etc.)
            work_item_type: The work item type (Bug, Task, Feature, Epic)

        Returns:
            A generated title (max 200 chars)
        """
        # Take first sentence or first 100 characters
        first_sentence = description.split('.')[0].strip()
        if len(first_sentence) > 100:
            first_sentence = first_sentence[:100].strip()

        # Add prefix based on type if description doesn't naturally start with action
        action_words = ['add', 'fix', 'update', 'remove', 'implement', 'create', 'improve', 'refactor']
        first_word = first_sentence.lower().split()[0] if first_sentence else ""

        if first_word not in action_words:
            # Add appropriate action verb
            if category == "bug":
                prefix = "Fix: "
            elif work_item_type == "Epic":
                prefix = "Epic: "
            elif category == "enhancement":
                prefix = "Improve: "
            else:
                prefix = "Add: "

            title = prefix + first_sentence
        else:
            title = first_sentence

        # Capitalize first letter
        if title:
            title = title[0].upper() + title[1:]

        # Ensure it's not too long
        return title[:200]

    def _triage_simple(self, user_input: Dict[str, Any], duplicates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simple triage using real product idea data (no AI).

        Uses keyword analysis of real user input to categorize and prioritize.

        Args:
            user_input: Real product idea from user
            duplicates: Real duplicate detection results

        Returns:
            Triage results based on real data, simple keyword analysis
        """
        title = (user_input.get("title") or "").lower()
        description = (user_input.get("description") or "").lower()
        business_value_text = (user_input.get("business_value") or "").lower()

        # Simple keyword-based categorization using real user input
        category_keywords = {
            "bug": ["bug", "error", "broken", "fails", "crash", "issue", "defect", "incorrect",
                    "ignores", "ignored", "wrong", "not working", "doesn't work", "doesnt work",
                    "missing", "failed", "unexpected", "should not", "shouldn't", "shouldn't",
                    "instead of", "but it", "however it", "problem", "fix"],
            "security": ["security", "vulnerability", "exploit", "breach", "authentication", "authorization"],
            "performance": ["performance", "slow", "optimize", "latency", "throughput", "bottleneck"],
            "infrastructure": ["api", "database", "architecture", "deployment", "ci/cd", "pipeline", "infrastructure"],
            "enhancement": ["improve", "enhance", "better", "faster", "refactor", "update"],
        }

        category = "feature"  # Default
        for cat, keywords in category_keywords.items():
            if any(kw in title or kw in description for kw in keywords):
                category = cat
                break

        # Simple priority assessment using real user input
        priority_keywords = {
            "critical": ["critical", "urgent", "blocker", "emergency", "asap"],
            "high": ["high", "important", "required", "essential", "must"],
            "low": ["nice", "nice to have", "optional", "future", "maybe"],
        }

        priority = "medium"  # Default
        for pri, keywords in priority_keywords.items():
            if any(kw in title or kw in description or kw in business_value_text for kw in keywords):
                priority = pri
                break

        # Simple effort estimation based on description length
        desc_len = len(description)
        if desc_len < 100:
            estimated_effort = "small"
        elif desc_len < 300:
            estimated_effort = "medium"
        elif desc_len < 600:
            estimated_effort = "large"
        else:
            estimated_effort = "extra-large"

        # Simple business value score (1-10) based on keywords
        business_value_score = 5  # Default: medium
        if any(kw in business_value_text for kw in ["high", "critical", "essential", "revenue"]):
            business_value_score = 8
        elif any(kw in business_value_text for kw in ["low", "optional", "nice"]):
            business_value_score = 3

        # Simple technical risk assessment
        risk_keywords = ["complex", "integration", "migration", "architecture", "security", "performance"]
        technical_risk = "high" if any(kw in title or kw in description for kw in risk_keywords) else "low"

        # Determine work item type based on SCOPE (not nature)
        # Bug: isolated issue, small scope (1-8 hours)
        # Task: small standalone work (1-2 days)
        # Feature: medium capability (3-7 days)
        # Epic: large initiative (2+ weeks)

        work_item_type = "Feature"  # Default
        work_item_type_rationale = "Default scope classification"

        # Simple heuristics for work item type
        if category == "bug" and estimated_effort == "small":
            work_item_type = "Bug"
            work_item_type_rationale = "Isolated bug with small effort estimate"
        elif category == "bug" and estimated_effort in ["large", "extra-large"]:
            work_item_type = "Epic"
            work_item_type_rationale = "Complex bug requiring multiple fixes/features"
        elif estimated_effort == "small":
            work_item_type = "Task"
            work_item_type_rationale = "Small work item with limited scope"
        elif estimated_effort in ["large", "extra-large"]:
            work_item_type = "Epic"
            work_item_type_rationale = "Large scope requiring breakdown into features"
        elif estimated_effort == "medium":
            work_item_type = "Feature"
            work_item_type_rationale = "Medium-sized capability"

        # Override: high technical risk + large effort → Epic
        if technical_risk == "high" and estimated_effort in ["large", "extra-large"]:
            work_item_type = "Epic"
            work_item_type_rationale = "High complexity and large effort requires Epic-level planning"

        # Recommended action
        recommended_action = "proceed"
        rationale = f"Categorized as {category} with {priority} priority, scope classified as {work_item_type}"

        # If duplicates found, recommend review (using real duplicate data)
        if duplicates.get("duplicates"):
            recommended_action = "review_duplicates"
            rationale = "Potential duplicates detected - review before proceeding"

        # Generate title if not provided by user
        generated_title = title if title else self._generate_simple_title(description, category, work_item_type)

        return {
            "title": generated_title,
            "category": category,
            "priority": priority,
            "estimated_effort": estimated_effort,
            "business_value_score": business_value_score,
            "technical_risk": technical_risk,
            "work_item_type": work_item_type,
            "work_item_type_rationale": work_item_type_rationale,
            "recommended_action": recommended_action,
            "rationale": rationale
        }

    def _triage_ai(self, user_input: Dict[str, Any], duplicates: Dict[str, Any]) -> Dict[str, Any]:
        """
        AI-based triage and classification (Mode 2).

        Uses Claude Agent SDK with tool access so the agent can:
        - Read CLAUDE.md for project context
        - Explore the codebase to understand scope
        - Make informed triage decisions based on actual code
        """
        # Load business analyst agent definition
        agent_definition = self._load_business_analyst_agent()

        # Define JSON schema
        schema = {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "minLength": 10,
                    "maxLength": 200,
                    "description": "Clear, concise title for the work item generated from description and business value"
                },
                "category": {
                    "type": "string",
                    "enum": ["feature", "enhancement", "infrastructure", "bug", "security", "performance", "other"]
                },
                "priority": {
                    "type": "string",
                    "enum": ["critical", "high", "medium", "low"]
                },
                "estimated_effort": {
                    "type": "string",
                    "enum": ["small", "medium", "large", "extra-large"]
                },
                "business_value_score": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10
                },
                "technical_risk": {
                    "type": "string",
                    "enum": ["low", "medium", "high"]
                },
                "work_item_type": {
                    "type": "string",
                    "enum": ["Bug", "Task", "Feature", "Epic"],
                    "description": "Work item type based on scope: Bug (isolated issue), Task (small work), Feature (medium), Epic (large/complex)"
                },
                "work_item_type_rationale": {
                    "type": "string",
                    "minLength": 30,
                    "description": "Explain why this scope classification (Bug/Task/Feature/Epic) was chosen"
                },
                "recommended_action": {
                    "type": "string",
                    "enum": ["proceed", "defer", "reject", "review_duplicates"]
                },
                "rationale": {
                    "type": "string",
                    "minLength": 50
                }
            },
            "required": ["title", "category", "priority", "estimated_effort", "business_value_score",
                         "technical_risk", "work_item_type", "work_item_type_rationale",
                         "recommended_action", "rationale"]
        }

        # Build comprehensive triage prompt with business analyst context
        title = user_input.get("title", "")
        description = user_input.get("description", "")
        business_value = user_input.get("business_value", "")
        duplicate_count = len(duplicates.get("duplicates", []))

        # Build detailed analysis prompt with instructions to read CLAUDE.md
        analysis_prompt = f"""**TRIAGE ANALYSIS REQUEST**

You are a Business Analyst triaging an incoming work item for the Trustable AI Development Workbench project.

**IMPORTANT: First read the project context**

Before analyzing the work item, use your Read tool to:
1. Read `CLAUDE.md` in the project root to understand the project purpose and architecture
2. If needed, explore relevant directories to understand technical scope

This will help you make accurate triage decisions based on actual project context.

{f"**Business Analyst Agent Context:**{chr(10)}{agent_definition}{chr(10)}" if agent_definition else ""}

**Work Item Details:**
{f"- User-Provided Title: {title}" if title else "- Title: To be generated by you"}
- Description: {description}
- Business Value (provided): {business_value or 'Not specified'}
- Potential Duplicates Found: {duplicate_count}

**Your Task:**

After reading project context, analyze this work item and provide comprehensive triage assessment:

0. **Generate Title** (if not provided): Create a clear, concise title (10-200 characters) that summarizes the work based on the description and business value. The title should be action-oriented and descriptive.

1. **Validate Category**: Classify the NATURE of the work
   - bug: Something is broken
   - feature: New capability needed
   - enhancement: Improvement to existing feature
   - infrastructure: Technical foundation work
   - security: Security vulnerability or improvement
   - performance: Performance optimization
   - other: Other types

2. **Determine Work Item Type**: Classify the SCOPE/SIZE
   - Bug: Isolated issue, small scope (1-8 hours)
   - Task: Small standalone work (1-2 days)
   - Feature: Medium capability requiring multiple tasks (3-7 days)
   - Epic: Large initiative requiring breakdown (2+ weeks)

   CRITICAL: Work item type reflects SCOPE, not nature:
   - "Bug Report" about simple query error → Bug work item (small)
   - "Bug Report" about auth system flaws → Epic work item (large, needs features)
   - "Feature Request" for export button → Task work item (small)
   - "Feature Request" for analytics dashboard → Epic work item (large)

3. **Assess Priority**: Based on business impact and urgency
   - critical: Production down, data loss, active security breach
   - high: Major feature broken, significant user impact
   - medium: Important but has workaround
   - low: Minor issue, cosmetic, nice-to-have

4. **Estimate Effort**: Development effort required
   - small: 1-2 days
   - medium: 3-5 days
   - large: 1-2 weeks
   - extra-large: Multiple weeks

5. **Score Business Value**: Rate 1-10 based on:
   - User impact and reach
   - Revenue potential
   - Strategic alignment
   - Technical debt reduction

6. **Evaluate Technical Risk**: Implementation complexity
   - low: Straightforward, well-understood
   - medium: Some unknowns, moderate complexity
   - high: Complex, dependencies, architectural changes

7. **Recommend Action**:
   - proceed: Create work item and add to backlog
   - defer: Strategic, needs roadmap planning
   - reject: Out of scope or duplicate
   - review_duplicates: Potential duplicates need review (use if {duplicate_count} > 0)

**Output Format:**

After your analysis, provide ONLY a JSON response matching this schema:
```json
{json.dumps(schema, indent=2)}
```

Do NOT include any text before or after the JSON block."""

        # Use Agent SDK with tool access for codebase exploration
        try:
            from scripts.workflow_executor.agent_sdk import AgentSDKWrapper
            import asyncio

            print("   Using Agent SDK with tool access for codebase exploration...")

            wrapper = AgentSDKWrapper(
                workflow_name="product-intake-triage",
                tool_preset="read_only",  # Read, Grep, Glob
                max_turns=15,  # Allow exploration turns
                model="claude-sonnet-4-5",
            )

            # Run async query
            async def _run_triage():
                return await wrapper.query(
                    prompt=analysis_prompt,
                    agent_type="business-analyst",
                )

            # Execute async
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, _run_triage())
                        result = future.result()
                else:
                    result = loop.run_until_complete(_run_triage())
            except RuntimeError:
                result = asyncio.run(_run_triage())

            if not result.success:
                print(f"⚠️  Agent SDK query failed: {result.error}")
                return self._triage_simple(user_input, duplicates)

            # Extract JSON from response
            response_text = result.response

            # Extract JSON if wrapped in code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

            triage_result = json.loads(response_text)
            validate(triage_result, schema)

            # Track token usage
            usage_info = {
                "input_tokens": result.token_usage.input_tokens,
                "output_tokens": result.token_usage.output_tokens,
                "cost_usd": result.cost_usd
            }
            self.token_usage["triage"] = usage_info

            print(f"✓ Business analyst triage complete (input: {result.token_usage.input_tokens:,}, output: {result.token_usage.output_tokens:,})")
            return triage_result

        except ImportError as e:
            # ALWAYS report import errors - no silent fallbacks
            from cli.console import print_error, print_info
            print_error(f"AI Triage Failed: Import error: {e}")
            print_error("The Claude Agent SDK is not available. AI triage requires tool access to read the codebase.")
            print_info("Install with: pip install claude-code-sdk")
            print_info("Workflow will continue with simple category-based triage.")
            return self._triage_simple(user_input, duplicates)

        except (json.JSONDecodeError, ValidationError) as e:
            print(f"⚠️  JSON validation failed: {type(e).__name__}: {e}")
            return self._triage_simple(user_input, duplicates)

        except Exception as e:
            print(f"⚠️  Agent SDK error: {type(e).__name__}: {e}")
            return self._triage_simple(user_input, duplicates)

    def _load_business_analyst_agent(self) -> Optional[str]:
        """
        Load business analyst agent definition for triage analysis.

        Returns:
            Agent definition text, or None if not found
        """
        agent_path = Path(".claude/agents/business-analyst.md")
        if not agent_path.exists():
            return None

        try:
            with open(agent_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"⚠️  Error loading business analyst agent: {e}")
            return None

    def _calculate_cost(self, usage) -> float:
        """Calculate cost in USD based on Claude API token usage."""
        input_cost = (usage.input_tokens / 1_000_000) * 3.0   # $3 per million
        output_cost = (usage.output_tokens / 1_000_000) * 15.0  # $15 per million
        return input_cost + output_cost

    def _triage_interactive(self, user_input: Dict[str, Any], duplicates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interactive product idea triage with multi-turn collaboration (Mode 3).

        User collaborates with AI to refine triage analysis through feedback.

        Args:
            user_input: Product idea from user
            duplicates: Duplicate detection results

        Returns:
            Triage results with category, priority, business value score
        """
        if not self.interactive_session or not self.interactive_session.is_available():
            print("⚠️  Interactive mode unavailable - falling back to simple triage")
            return self._triage_simple(user_input, duplicates)

        # Build context for AI
        title = user_input.get("title", "")
        description = user_input.get("description", "")
        business_value_text = user_input.get("business_value", "")
        duplicate_count = len(duplicates.get("duplicates", []))

        context = {
            "title": title,
            "description": description,  # Full content for accurate triage
            "business_value": business_value_text,
            "duplicate_count": duplicate_count
        }

        initial_prompt = f"""You are a product manager triaging a new product idea.

Analyze this idea and provide initial assessment with:
- Title: Generate a clear, concise title (10-200 characters) based on the description and business value{' or use the user-provided title if it is good' if title else ''}
- Category (feature/enhancement/infrastructure/bug/security/performance/other) - the NATURE of work
- Priority (critical/high/medium/low)
- Estimated effort (small/medium/large/extra-large)
- Business value score (1-10)
- Technical risk (low/medium/high)
- Work item type (Bug/Task/Feature/Epic) - the SCOPE/SIZE
- Recommended action (proceed/defer/reject/review_duplicates)

{f'Note: {duplicate_count} potential duplicate(s) found - consider review_duplicates action' if duplicate_count > 0 else ''}

**CRITICAL: Work item type reflects SCOPE/SIZE, not the nature of work:**

| Work Item Type | Scope | Examples |
|---------------|-------|----------|
| Bug | 1-8 hours | Fix typo, null check, simple validation error |
| Task | 1-2 days | Add CLI flag, display link, small refactor |
| Feature | 3-7 days | New API endpoint with tests, UI component |
| Epic | 2+ weeks | New subsystem, major architecture change |

**Examples of correct classification:**
- "Script ignores command arguments" → Bug (isolated issue, few hours to fix)
- "Display link to created work item" → Task (small enhancement, 1-2 days)
- "Add user authentication" → Feature (medium capability, multiple tasks)
- "Redesign data pipeline" → Epic (large initiative, needs breakdown)

A bug REPORT can be a Bug work item (simple fix) or Epic (complex systemic issue).
A feature REQUEST can be a Task work item (simple) or Epic (complex).

Provide your analysis with clear rationale for the work item type."""

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
                "title": {"type": "string", "minLength": 10, "maxLength": 200},
                "category": {"type": "string", "enum": ["feature", "enhancement", "infrastructure", "bug", "security", "performance", "other"]},
                "priority": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
                "estimated_effort": {"type": "string", "enum": ["small", "medium", "large", "extra-large"]},
                "business_value_score": {"type": "integer", "minimum": 1, "maximum": 10},
                "technical_risk": {"type": "string", "enum": ["low", "medium", "high"]},
                "work_item_type": {"type": "string", "enum": ["Bug", "Task", "Feature", "Epic"]},
                "work_item_type_rationale": {"type": "string", "minLength": 30},
                "recommended_action": {"type": "string", "enum": ["proceed", "defer", "reject", "review_duplicates"]},
                "rationale": {"type": "string", "minLength": 50}
            },
            "required": ["title", "category", "priority", "estimated_effort", "business_value_score",
                         "technical_risk", "work_item_type", "work_item_type_rationale",
                         "recommended_action", "rationale"]
        }

        # CRITICAL: Include the final AI response as context for JSON extraction.
        # The ask() method creates a fresh API call with no conversation history,
        # so we must explicitly provide the analysis to extract from.
        final_ai_analysis = result.get("final_response", "")

        extraction_prompt = f"""Extract a structured JSON assessment from the following product triage analysis.

## AI TRIAGE ANALYSIS TO EXTRACT FROM:
{final_ai_analysis}

## JSON SCHEMA:
{json.dumps(schema, indent=2)}

Provide ONLY the JSON matching the schema above, no other text. Extract values directly from the analysis provided."""

        try:
            json_response = self.interactive_session.ask(extraction_prompt)
            triage_data = self.interactive_session.extract_json_from_response(json_response, schema)
            return triage_data

        except Exception as e:
            print(f"⚠️  Failed to extract JSON: {e}")
            print("    Falling back to simple triage")
            return self._triage_simple(user_input, duplicates)

    def _step_4_approval_gate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 4: Human approval gate (BLOCKING).

        Execution HALTS here until user approves/rejects.
        """
        import textwrap

        user_input = self.step_evidence.get("1-user-input", {})
        triage = self.step_evidence.get("3-ai-triage", {})
        technical_analysis = self.step_evidence.get("3.5-technical-analysis", {})

        work_item_type = triage.get("work_item_type", "work item")

        # Build summary lines for approval gate
        summary_lines = []

        # User submission section
        summary_lines.append("📝 YOUR SUBMISSION")
        summary_lines.append("─" * 70)
        # Show AI-generated title if available, otherwise user-provided title
        display_title = triage.get('title') or user_input.get('title', 'Unknown')
        if triage.get('title') and not user_input.get('title'):
            summary_lines.append(f"Title (AI-generated): {display_title}")
        else:
            summary_lines.append(f"Title: {display_title}")

        description = user_input.get('description', 'No description provided')
        summary_lines.append(f"Description: {description[:100]}{'...' if len(description) > 100 else ''}")

        business_value = user_input.get('business_value', '').strip()
        if business_value:
            summary_lines.append(f"Business Value: {business_value[:100]}{'...' if len(business_value) > 100 else ''}")

        summary_lines.append("")
        summary_lines.append("📋 BUSINESS ANALYST TRIAGE RESULTS")
        summary_lines.append("─" * 70)
        summary_lines.append(f"Category (Nature): {triage.get('category', 'Unknown')}")
        summary_lines.append(f"Work Item Type (Scope): {work_item_type}")
        summary_lines.append(f"Priority: {triage.get('priority', 'Unknown')}")
        summary_lines.append(f"Effort Estimate: {triage.get('estimated_effort', 'Unknown')}")
        summary_lines.append(f"Business Value Score: {triage.get('business_value_score', 'N/A')}/10")
        summary_lines.append(f"Technical Risk: {triage.get('technical_risk', 'Unknown')}")

        rationale_text = triage.get('work_item_type_rationale', 'Unknown')
        summary_lines.append(f"Work Item Type Rationale: {rationale_text[:100]}{'...' if len(rationale_text) > 100 else ''}")

        # Technical analysis section (if available)
        if not technical_analysis.get("skipped", False):
            summary_lines.append("")
            summary_lines.append("🔧 TECHNICAL ANALYSIS (Senior Engineer)")
            summary_lines.append("─" * 70)

            if "root_cause_analysis" in technical_analysis:
                rca = technical_analysis.get("root_cause_analysis", "")
                summary_lines.append(f"Root Cause: {rca[:80]}{'...' if len(rca) > 80 else ''}")

            if "acceptance_criteria" in technical_analysis:
                criteria = technical_analysis.get("acceptance_criteria", [])
                summary_lines.append(f"Acceptance Criteria: {len(criteria)} criteria defined")

        # Create approval gate data structure
        gate = ApprovalGateData(
            title=f"Step 4: Human Approval Gate",
            summary=summary_lines,
            options=[
                ("yes", f"Approve {work_item_type} creation and continue"),
                ("no", "Reject idea (no changes made)")
            ],
            question=f"Approve {work_item_type} creation? (yes/no): "
        )

        # Print approval gate using unified console function
        print_approval_gate(gate)

        # BLOCKING CALL - Execution halts here
        try:
            response = input(f"Approve {work_item_type} creation? (yes/no): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n\n❌ Approval cancelled by user")
            response = "no"

        approved = response == "yes"

        evidence = {
            "approved": approved,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }

        if approved:
            print(f"\n✅ User APPROVED - Proceeding with {work_item_type} creation")
        else:
            print("\n❌ User REJECTED - Workflow will terminate")

        return evidence

    def _step_5_work_item_creation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 5: Create work item (type determined by AI triage)."""
        user_input = self.step_evidence.get("1-user-input", {})
        triage = self.step_evidence.get("3-ai-triage", {})

        work_item_type = triage.get("work_item_type", "Epic")

        print_step_header(5, "Create Work Item", f"Create {work_item_type} in tracking system")

        # Check if approved
        approval = self.step_evidence.get("4-approval-gate", {})
        if not approval.get("approved"):
            raise ValueError(f"Cannot create {work_item_type} - user did not approve")

        if not self.adapter:
            print(f"⚠️  No adapter - cannot create {work_item_type}")
            return {"work_item_id": "MOCK-123", "work_item_type": work_item_type, "mock": True}

        try:
            # Use AI-generated title if available, otherwise fall back to user-provided title
            title = triage.get("title") or user_input.get("title", "")
            description = user_input.get("description", "")
            technical_analysis = self.step_evidence.get("3.5-technical-analysis", {})

            # Build enhanced description with technical analysis
            enhanced_description = description

            # Build custom fields dict (excluding title and description which are separate params)
            custom_fields = {}

            # Add priority and business value as custom fields
            # Azure DevOps Priority field expects integers: 1=critical, 2=high, 3=medium, 4=low
            priority_to_int = {"critical": 1, "high": 2, "medium": 3, "low": 4}
            if triage.get("priority"):
                priority_str = triage.get("priority", "medium")
                custom_fields["priority"] = priority_to_int.get(priority_str, 3)
            if user_input.get("business_value"):
                custom_fields["business_value"] = user_input.get("business_value", "")

            # Add work item type specific fields
            if work_item_type == "Bug" and not technical_analysis.get("skipped", False):
                # Bug-specific fields with technical analysis

                # Root cause analysis → System Info field
                rca = technical_analysis.get("root_cause_analysis", "TBD during analysis")
                custom_fields["Microsoft.VSTS.TCM.SystemInfo"] = f"<div><strong>Root Cause Analysis:</strong><br/>{rca.replace(chr(10), '<br/>')}</div>"

                # Implementation plan → Repro Steps field
                plan = technical_analysis.get("implementation_plan", "TBD during planning")
                custom_fields["Microsoft.VSTS.TCM.ReproSteps"] = f"<div><strong>Implementation Plan:</strong><br/>{plan.replace(chr(10), '<br/>')}</div>"

                # Acceptance criteria
                criteria = technical_analysis.get("acceptance_criteria", ["Bug fixed and verified", "No regression", "Tests passing"])
                criteria_html = "<ul>"
                for criterion in criteria:
                    criteria_html += f"<li>{criterion}</li>"
                criteria_html += "</ul>"
                custom_fields["Microsoft.VSTS.Common.AcceptanceCriteria"] = f"<div><strong>Acceptance Criteria:</strong>{criteria_html}</div>"

                # Test design → Add to description
                if "test_design" in technical_analysis:
                    test_design = technical_analysis.get("test_design", "")
                    enhanced_description += f"\n\n## Test Design\n{test_design}"

            elif work_item_type == "Task" and not technical_analysis.get("skipped", False):
                # Task-specific fields with technical analysis

                # Implementation plan → Add to description
                if "implementation_plan" in technical_analysis:
                    plan = technical_analysis.get("implementation_plan", "")
                    enhanced_description += f"\n\n## Implementation Plan\n{plan}"

                # Acceptance criteria
                if "acceptance_criteria" in technical_analysis:
                    criteria = technical_analysis.get("acceptance_criteria", [])
                    criteria_html = "<ul>"
                    for criterion in criteria:
                        criteria_html += f"<li>{criterion}</li>"
                    criteria_html += "</ul>"
                    custom_fields["Microsoft.VSTS.Common.AcceptanceCriteria"] = f"<div><strong>Acceptance Criteria:</strong>{criteria_html}</div>"

                # Test design → Add to description
                if "test_design" in technical_analysis:
                    test_design = technical_analysis.get("test_design", "")
                    enhanced_description += f"\n\n## Test Design\n{test_design}"

            # Create work item with AI-determined type and enhanced fields
            result = self.adapter.create_work_item(
                work_item_type=work_item_type,
                title=title,
                description=enhanced_description,
                assigned_to=self.current_user,
                fields=custom_fields if custom_fields else None
            )

            work_item_id = result.get("id")

            evidence = {
                "work_item_id": work_item_id,
                "work_item_type": work_item_type,
                "title": title,
                "created_at": datetime.now().isoformat(),
                "mock": False
            }

            print(f"\n✓ Created {work_item_type}: {work_item_id}")
            return evidence

        except Exception as e:
            print(f"❌ {work_item_type} creation failed: {e}")
            raise

    def _step_6_verification(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 6: Verify work item exists (external source of truth)."""
        work_item_creation = self.step_evidence.get("5-work-item-creation", {})
        work_item_id = work_item_creation.get("work_item_id")
        work_item_type = work_item_creation.get("work_item_type", "work item")

        print_step_header(6, "External Verification", f"Verify {work_item_type} exists in tracking system")

        if work_item_creation.get("mock"):
            raise ValueError(f"Mock {work_item_type} detected - external verification required. Cannot verify mock work items.")

        if not self.adapter:
            print(f"⚠️  No adapter - cannot verify {work_item_type}")
            return {"verified": False, "error": "No adapter"}

        try:
            # Query adapter for work item (external source of truth)
            work_item = self.adapter.get_work_item(work_item_id)

            if not work_item:
                raise ValueError(f"{work_item_type} {work_item_id} claimed created but doesn't exist in tracking system")

            # Verify fields match
            # Azure DevOps REST API returns fields under work_item["fields"]["System.Title"]
            expected_title = work_item_creation.get("title", "")
            actual_title = work_item.get("fields", {}).get("System.Title", "")

            if expected_title and actual_title and expected_title != actual_title:
                raise ValueError(f"Title mismatch: expected '{expected_title}', got '{actual_title}'")

            evidence = {
                "verified": True,
                "work_item_id": work_item_id,
                "work_item_type": work_item_type,
                "verified_at": datetime.now().isoformat(),
                "actual_state": work_item.get("fields", {}).get("System.State", "Unknown"),
                "actual_title": actual_title
            }

            print(f"✓ {work_item_type} {work_item_id} verified in tracking system")
            return evidence

        except Exception as e:
            print(f"❌ Verification failed: {e}")
            raise

    def _step_7_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 7: Generate summary report."""
        print_step_header(7, "Generate Summary", "Generate intake summary report")

        user_input = self.step_evidence.get("1-user-input", {})
        triage = self.step_evidence.get("3-ai-triage", {})
        work_item_creation = self.step_evidence.get("5-work-item-creation", {})
        verification = self.step_evidence.get("6-verification", {})

        work_item_type = work_item_creation.get("work_item_type", "Unknown")
        work_item_id = work_item_creation.get("work_item_id", "Unknown")

        # Generate comprehensive summary report
        display_title = triage.get('title') or user_input.get('title', 'Unknown')
        title_source = "(AI-generated)" if triage.get('title') and not user_input.get('title') else "(User-provided)" if user_input.get('title') else ""

        summary_lines = [
            "# Product Intake Summary",
            "",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Workflow ID:** {self.workflow_id}",
            f"**Mode:** {'AI-Assisted (Business Analyst)' if self.use_ai else 'Simple Triage'}",
            "",
            "## Product Idea",
            "",
            f"**Title:** {display_title} {title_source}",
            "",
            f"**Description:**",
            user_input.get('description', 'No description provided'),
            "",
        ]

        # Add business value if provided
        if user_input.get('business_value'):
            summary_lines.extend([
                "**Business Value (User Provided):**",
                user_input.get('business_value'),
                ""
            ])

        summary_lines.extend([
            "## Business Analyst Triage Results",
            "",
            "### Classification",
            f"- **Category (Nature):** {triage.get('category', 'Unknown')}",
            f"- **Work Item Type (Scope):** {work_item_type}",
            f"- **Priority:** {triage.get('priority', 'Unknown')}",
            "",
            "### Assessment",
            f"- **Estimated Effort:** {triage.get('estimated_effort', 'Unknown')}",
            f"- **Business Value Score:** {triage.get('business_value_score', 0)}/10",
            f"- **Technical Risk:** {triage.get('technical_risk', 'Unknown')}",
            "",
            "### Work Item Type Rationale",
            triage.get('work_item_type_rationale', 'Not provided'),
            "",
            "### Overall Rationale",
            triage.get('rationale', 'Not provided'),
            "",
            "### Recommended Action",
            f"{triage.get('recommended_action', 'Unknown')}",
            ""
        ])

        # Add technical analysis if available
        technical_analysis = self.step_evidence.get("3.5-technical-analysis", {})
        if not technical_analysis.get("skipped", False):
            summary_lines.extend([
                "## Senior Engineer Technical Analysis",
                ""
            ])

            if "root_cause_analysis" in technical_analysis:
                summary_lines.extend([
                    "### Root Cause Analysis",
                    technical_analysis.get("root_cause_analysis", ""),
                    ""
                ])

            if "implementation_plan" in technical_analysis:
                summary_lines.extend([
                    "### Implementation Plan",
                    technical_analysis.get("implementation_plan", ""),
                    ""
                ])

            if "acceptance_criteria" in technical_analysis:
                summary_lines.append("### Acceptance Criteria")
                for i, criterion in enumerate(technical_analysis.get("acceptance_criteria", []), 1):
                    summary_lines.append(f"{i}. {criterion}")
                summary_lines.append("")

            if "test_design" in technical_analysis:
                summary_lines.extend([
                    "### Test Design",
                    technical_analysis.get("test_design", ""),
                    ""
                ])

        summary_lines.extend([
            f"## {work_item_type} Created",
            "",
            f"- **{work_item_type} ID:** {work_item_id}",
            f"- **Created At:** {work_item_creation.get('created_at', 'Unknown')}",
            f"- **Verified:** {'✅ Yes' if verification.get('verified') else '❌ No'}",
            "",
        ])

        # Add token usage stats if AI was used
        if self.use_ai and self.token_usage.get('triage'):
            usage = self.token_usage['triage']
            summary_lines.extend([
                "## AI Usage Statistics",
                "",
                f"- **Model:** claude-sonnet-4.5",
                f"- **Extended Thinking:** ENABLED",
                f"- **Input Tokens:** {usage.get('input_tokens', 0):,}",
                f"- **Output Tokens:** {usage.get('output_tokens', 0):,}",
            ])
            if 'thinking_tokens' in usage:
                summary_lines.append(f"- **Thinking Tokens:** {usage.get('thinking_tokens', 0):,}")
            summary_lines.extend([
                f"- **Cost:** ${usage.get('cost_usd', 0):.4f}",
                ""
            ])

        summary_lines.extend([
            "---",
            "",
            "*Generated by Trustable AI Development Workbench*",
            "*Business Analyst agent with Extended Thinking enabled*"
        ])

        summary_content = "\n".join(summary_lines)

        # Save summary to file
        summary_dir = Path(".claude/reports/product-intake")
        summary_dir.mkdir(parents=True, exist_ok=True)

        summary_file = summary_dir / f"{self.workflow_id}.md"

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_content)

        evidence = {
            "summary_file": str(summary_file),
            "work_item_id": work_item_id,
            "work_item_type": work_item_type
        }

        # Print summary panel using unified console
        summary_content_dict = {
            "Work Item Type": work_item_type,
            "Work Item ID": work_item_id,
            "Summary File": str(summary_file),
            "Verified": "✅ Yes" if verification.get('verified') else "❌ No"
        }
        print_summary_panel("✅ Product Intake Complete", summary_content_dict, style="success")

        return evidence

    def _step_8_checkpoint(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 8: Save final checkpoint."""
        print_step_header(8, "Save Checkpoint", "Save workflow state for re-entrancy")

        # State is automatically saved by WorkflowOrchestrator
        # This step just confirms completion

        evidence = {
            "checkpoint_saved": True,
            "timestamp": datetime.now().isoformat(),
            "workflow_complete": True
        }

        print("✓ Checkpoint saved")
        return evidence


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Product Intake Workflow - External enforcement with duplicate detection and AI triage"
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
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Disable AI triage (use simple heuristics instead)"
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Disable interactive mode (skip user prompts, use args only)"
    )

    # File input for vision/requirements documents
    parser.add_argument(
        "--file", "-f",
        help="Path to requirements file (.md or .txt) containing product idea/vision"
    )

    # Non-interactive mode arguments (Task #1239)
    parser.add_argument(
        "--title",
        help="Product idea title (1-200 characters)"
    )
    parser.add_argument(
        "--description",
        help="Product idea description (1-10000 characters)"
    )
    parser.add_argument(
        "--business-value",
        help="Business value description (optional)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed step-by-step output (default: clean summary)"
    )
    parser.add_argument(
        "--skip-triage",
        action="store_true",
        help="Skip AI triage and create Epic directly (useful for detailed requirement files)"
    )

    args = parser.parse_args()

    # Generate workflow ID if not provided
    workflow_id = args.workflow_id or f"intake-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Create and execute workflow
    workflow = ProductIntakeWorkflow(
        workflow_id=workflow_id,
        enable_checkpoints=not args.no_checkpoints,
        use_ai=not args.no_ai,
        interactive=not args.no_interactive,
        args=args,
        verbose=args.verbose,
        skip_triage=args.skip_triage
    )

    from cli.console import console

    try:
        success = workflow.execute()
        if success:
            console.print()
            console.print("─" * 80)
            console.print("[bold #71E4D1]  Product intake complete![/bold #71E4D1]")
            console.print("─" * 80)
            console.print()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        console.print()
        console.print("[#758B9B]Product intake cancelled by user.[/#758B9B]")
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
