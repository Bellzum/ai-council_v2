#!/usr/bin/env python3
"""
Sprint Retrospective Workflow with External Enforcement

Implements Phase 2: Simple Workflows - Sprint Retrospective Script

11-Step Workflow:
1. Sprint Selection - User selects sprint to retrospect
2. Metrics Collection - Gather velocity, completion rate, bug counts
3. AI Analysis - Analyze patterns and trends using Mode 2 (AI + JSON validation)
4. Improvement Generation - Generate actionable improvement items
5. Human Approval Gate - BLOCKING approval for task creation
6. Task Creation - Create improvement tasks in tracking system
7. External Verification - Verify tasks exist in tracking system
8. Report Generation - Generate retrospective report
9. Save Report - Write report to .claude/retrospectives/
10. Artifact Hygiene (Optional) - Clean up sprint artifacts
11. Checkpoint - Save workflow state for re-entrancy

Design Pattern:
- Extends WorkflowOrchestrator from Phase 1
- Uses adapter for ALL work item operations
- External verification after task creation
- Real input() blocking for approval gate
- UTF-8 encoding for all file writes
- Comprehensive metrics collection
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
from scripts.workflow_executor.schemas import StepType, WorkflowStepDefinition, ApprovalGateConfig
from scripts.workflow_executor.approval_gates import ApprovalGate
from scripts.workflow_executor.progress import print_warning
from workflows.utilities import (
 analyze_sprint,
 calculate_sprint_velocity,
 calculate_completion_rate,
 count_bugs_by_severity,
 format_retrospective_report
)

# Import JSON schema validation
try:
 from jsonschema import validate, ValidationError
except ImportError:
 print("️ jsonschema package not installed - install with: pip install jsonschema")
 ValidationError = Exception # Fallback


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


class SprintRetrospectiveWorkflow(WorkflowOrchestrator):
 """
 Sprint Retrospective workflow with external enforcement.

 Implements the 10-step retrospective process with:
 - Comprehensive metrics collection
 - Mode 2 AI analysis with JSON validation
 - External verification after task creation
 - Blocking approval gates
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
  Initialize sprint retrospective workflow.

  Args:
   sprint_name: Sprint to retrospect (e.g., "Sprint 7")
   workflow_id: Unique ID for this execution
   enable_checkpoints: Enable state checkpointing
   use_ai: If True, use AI for analysis (Mode 2), otherwise use mock (Mode 1)
   interactive: If True, use Mode 3 interactive collaboration with Claude Agent SDK
  """
  # Normalize sprint name (e.g., "9" → "Sprint 9")
  self.sprint_name = normalize_sprint_name(sprint_name)
  self.use_ai = use_ai
  self.interactive = interactive

  # Interactive mode overrides use_ai (Mode 3 vs Mode 2)
  if interactive and use_ai:
   print("️ Both --use-ai and --interactive specified - using interactive mode (Mode 3)")
   self.use_ai = False  # Interactive mode takes precedence

  mode = ExecutionMode.INTERACTIVE_AI if interactive else (ExecutionMode.AI_JSON_VALIDATION if use_ai else ExecutionMode.PURE_PYTHON)

  super().__init__(
   workflow_name="sprint-retrospective",
   workflow_id=workflow_id,
   mode=mode,
   enable_checkpoints=enable_checkpoints
  )

  # Initialize adapter using config-based get_adapter (Bug #1 fix)
  try:
   sys.path.insert(0, '.claude/skills')
   from work_tracking import get_adapter
   self.adapter = get_adapter()
   print(f" Adapter initialized: {type(self.adapter).__name__}")
  except Exception as e:
   print(f"️ Warning: Could not initialize adapter: {e}")
   print(" Continuing with limited functionality...")
   self.adapter = None

  # Get current user for work item assignment
  self.current_user = None
  if self.adapter:
   try:
    user_info = self.adapter.get_current_user()
    if user_info:
     self.current_user = user_info.get('display_name') or user_info.get('email')
   except Exception as e:
    print(f"️ Could not get current user: {e}")
    print(" Work items will be created without assignment")

  # Load config
  try:
   from config.loader import load_config
   self.config = load_config().model_dump()
  except Exception as e:
   print(f"️ Warning: Could not load config: {e}")
   self.config = {}

  # Initialize Claude API client if using AI
  self.claude_client = None
  self.token_usage = {}
  if use_ai:
   try:
    import anthropic
    api_key = os.getenv("KEYCHAIN_ANTHROPIC_API_KEY")
    if api_key:
     self.claude_client = anthropic.Anthropic(api_key=api_key)
     print(" Claude API client initialized")
    else:
     print_warning("KEYCHAIN_ANTHROPIC_API_KEY not set, falling back to simple logic mode")
     self.use_ai = False
   except ImportError:
    print("️ anthropic package not installed, falling back to simple logic mode")
    self.use_ai = False

  # Initialize interactive session if interactive mode
  self.interactive_session = None
  if interactive:
   try:
    from scripts.workflow_executor.interactive_session import InteractiveSession
    self.interactive_session = InteractiveSession(
     workflow_name="sprint-retrospective",
     session_id=sprint_name.replace(' ', '-'),
     model="claude-sonnet-4-5",
     max_tokens=4000
    )
    if self.interactive_session.is_available():
     print(" Interactive mode initialized (Mode 3)")
    else:
     print("️ Interactive mode unavailable - falling back to mock data")
     self.interactive = False
   except ImportError as e:
    print(f"️ Interactive mode unavailable: {e}")
    print(" Falling back to mock data")
    self.interactive = False

 def _define_steps(self) -> List[Dict[str, Any]]:
  """Define the 11 workflow steps."""
  return [
   {
    "id": "1-sprint-selection",
    "name": "Sprint Selection",
    "step_type": StepType.DATA_COLLECTION,
    "description": "Confirm sprint to retrospect",
    "required": True
   },
   {
    "id": "2-metrics-collection",
    "name": "Metrics Collection",
    "step_type": StepType.DATA_COLLECTION,
    "description": "Gather velocity, completion rate, bug counts",
    "required": True,
    "depends_on": ["1-sprint-selection"]
   },
   {
    "id": "3-ai-analysis",
    "name": "AI Pattern Analysis",
    "step_type": StepType.AI_REVIEW,
    "description": "Analyze patterns and trends using AI",
    "required": True,
    "depends_on": ["2-metrics-collection"]
   },
   {
    "id": "4-improvement-generation",
    "name": "Generate Improvements",
    "step_type": StepType.AI_REVIEW,
    "description": "Generate actionable improvement items",
    "required": True,
    "depends_on": ["3-ai-analysis"]
   },
   {
    "id": "5-approval-gate",
    "name": "Human Approval Gate",
    "step_type": StepType.APPROVAL_GATE,
    "description": "BLOCKING approval for task creation",
    "required": True,
    "depends_on": ["4-improvement-generation"]
   },
   {
    "id": "6-task-creation",
    "name": "Create Improvement Tasks",
    "step_type": StepType.ACTION,
    "description": "Create tasks in tracking system",
    "required": True,
    "depends_on": ["5-approval-gate"]
   },
   {
    "id": "7-verification",
    "name": "External Verification",
    "step_type": StepType.VERIFICATION,
    "description": "Verify tasks exist in tracking system",
    "required": True,
    "depends_on": ["6-task-creation"]
   },
   {
    "id": "8-report-generation",
    "name": "Generate Report",
    "step_type": StepType.ACTION,
    "description": "Generate retrospective report",
    "required": True,
    "depends_on": ["7-verification"]
   },
   {
    "id": "9-save-report",
    "name": "Save Report",
    "step_type": StepType.ACTION,
    "description": "Write report to .claude/retrospectives/",
    "required": True,
    "depends_on": ["8-report-generation"]
   },
   {
    "id": "10-artifact-hygiene",
    "name": "Artifact Hygiene (Optional)",
    "step_type": StepType.ACTION,
    "description": "Clean up sprint artifacts",
    "required": False,
    "depends_on": ["9-save-report"]
   },
   {
    "id": "11-checkpoint",
    "name": "Save Checkpoint",
    "step_type": StepType.ACTION,
    "description": "Save workflow state for re-entrancy",
    "required": True,
    "depends_on": ["10-artifact-hygiene"]
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
  elif step_id == "2-metrics-collection":
   return self._step_2_metrics_collection(context)
  elif step_id == "3-ai-analysis":
   return self._step_3_ai_analysis(context)
  elif step_id == "4-improvement-generation":
   return self._step_4_improvement_generation(context)
  elif step_id == "5-approval-gate":
   return self._step_5_approval_gate(context)
  elif step_id == "6-task-creation":
   return self._step_6_task_creation(context)
  elif step_id == "7-verification":
   return self._step_7_verification(context)
  elif step_id == "8-report-generation":
   return self._step_8_report_generation(context)
  elif step_id == "9-save-report":
   return self._step_9_save_report(context)
  elif step_id == "10-artifact-hygiene":
   return self._step_10_artifact_hygiene(context)
  elif step_id == "11-checkpoint":
   return self._step_11_checkpoint(context)
  else:
   raise ValueError(f"Unknown step: {step_id}")

 # ========================================================================
 # Interactive Mode 3 Methods (Feature #1214)
 # ========================================================================

 def _feedback_analysis_interactive(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
  """
  Interactive feedback analysis with AI (Mode 3).

  User collaborates with AI to analyze retrospective feedback.

  Returns:
   Structured analysis of feedback themes and patterns
  """
  if not self.interactive_session or not self.interactive_session.is_available():
   return {"feedback": feedback, "skipped": True}

  context = {
   "sprint_name": self.sprint_name,
   "positive_items": len(feedback.get("positive", [])),
   "improvement_areas": len(feedback.get("improvements", []))
  }

  initial_prompt = f"""You are a scrum master analyzing retrospective feedback for {self.sprint_name}.

Analyze this feedback and identify key themes, patterns, and root causes."""

  try:
   result = self.interactive_session.discuss(
    initial_prompt=initial_prompt,
    context=context,
    max_iterations=3
   )
   return {"feedback": feedback, "analysis": result.get("final_response")}
  except Exception as e:
   print(f"️ Interactive analysis failed: {e}")
   return {"feedback": feedback, "skipped": True}

 def _action_prioritization_interactive(self, actions: List[str]) -> Dict[str, Any]:
  """
  Interactive action item prioritization with AI (Mode 3).

  User collaborates with AI to prioritize action items.

  Returns:
   Prioritized action items with rationale
  """
  if not self.interactive_session or not self.interactive_session.is_available():
   return {"actions": actions, "skipped": True}

  context = {
   "sprint_name": self.sprint_name,
   "action_count": len(actions),
   "actions": actions[:5]  # First 5 for context
  }

  initial_prompt = f"""You are a scrum master prioritizing {len(actions)} action item(s) for {self.sprint_name}.

Help prioritize these items based on impact, effort, and alignment with team goals."""

  try:
   result = self.interactive_session.discuss(
    initial_prompt=initial_prompt,
    context=context,
    max_iterations=3
   )
   return {"actions": actions, "prioritization": result.get("final_response")}
  except Exception as e:
   print(f"️ Interactive prioritization failed: {e}")
   return {"actions": actions, "skipped": True}

 def _step_1_sprint_selection(self, context: Dict[str, Any]) -> Dict[str, Any]:
  """Step 1: Confirm sprint selection."""
  print(f"\n Sprint selected for retrospective: {self.sprint_name}")

  evidence = {
   "sprint_name": self.sprint_name,
   "selected_at": datetime.now().isoformat()
  }

  return evidence

 def _step_2_metrics_collection(self, context: Dict[str, Any]) -> Dict[str, Any]:
  """Step 2: Collect comprehensive sprint metrics."""
  print("\n Collecting sprint metrics...")

  if not self.adapter:
   print("️ No adapter - using mock metrics")
   return self._get_mock_metrics()

  try:
   # Collect velocity
   print(" - Calculating velocity...")
   velocity_data = calculate_sprint_velocity(
    self.adapter,
    self.sprint_name,
    self.config
   )

   # Collect completion rate
   print(" - Calculating completion rate...")
   completion_data = calculate_completion_rate(
    self.adapter,
    self.sprint_name,
    self.config
   )

   # Collect bug data
   print(" - Counting bugs by severity...")
   bug_data = count_bugs_by_severity(
    self.adapter,
    self.sprint_name,
    self.config
   )

   # Collect comprehensive sprint analysis
   print(" - Analyzing sprint work items...")
   sprint_analysis = analyze_sprint(
    self.adapter,
    self.sprint_name,
    self.config
   )

   evidence = {
    "velocity": velocity_data,
    "completion": completion_data,
    "bugs": bug_data,
    "sprint_analysis": sprint_analysis,
    "collected_at": datetime.now().isoformat()
   }

   print(f"\n Metrics collected:")
   print(f" - Velocity: {velocity_data.get('velocity', 0):.0f} points")
   print(f" - Completion rate: {completion_data.get('completion_rate', 0):.1f}%")
   print(f" - Total bugs: {bug_data.get('total_bugs', 0)}")
   print(f" • Sprint-execution bugs: {bug_data.get('sprint_execution_bugs', 0)} (from sprint work)")
   print(f" • Imported bugs: {bug_data.get('imported_bugs', 0)} (external)")

   return evidence

  except Exception as e:
   print(f"️ Metrics collection error: {e}")
   return self._get_mock_metrics()

 def _get_mock_metrics(self) -> Dict[str, Any]:
  """Get mock metrics for testing."""
  return {
   "velocity": {
    "sprint_name": self.sprint_name,
    "velocity": 25.0,
    "total_points": 30.0,
    "completion_percentage": 83.3
   },
   "completion": {
    "sprint_name": self.sprint_name,
    "completion_rate": 85.0,
    "completed_items": 17,
    "total_items": 20
   },
   "bugs": {
    "sprint_name": self.sprint_name,
    "total_bugs": 3,
    "sprint_execution_bugs": 1,
    "imported_bugs": 2,
    "by_severity": {
     "critical": 0,
     "high": 1,
     "medium": 2,
     "low": 0
    },
    "by_origin": {
     "sprint_execution": {"critical": 0, "high": 0, "medium": 1, "low": 0},
     "imported": {"critical": 0, "high": 1, "medium": 1, "low": 0}
    },
    "by_state": {
     "Done": 2,
     "Active": 1
    }
   },
   "sprint_analysis": {
    "sprint_name": self.sprint_name,
    "total_items": 20,
    "completion_rate": 85.0,
    "velocity": 25.0
   },
   "mock": True
  }

 def _step_3_ai_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
  """Step 3: AI pattern analysis (Mode 2: JSON validation)."""
  print("\n Running AI pattern analysis...")

  metrics = self.step_evidence.get("2-metrics-collection", {})

  if self.use_ai:
   analysis = self._analyze_ai(metrics)
  else:
   analysis = self._analyze_simple(metrics)

  print(f"\n Analysis complete: {len(analysis.get('patterns', []))} pattern(s) identified")
  for pattern in analysis.get("patterns", []):
   print(f" - [{pattern['category'].upper()}] {pattern['observation']}")

  return analysis

 def _analyze_simple(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
  """Mock analysis (Mode 1: Pure Python)."""
  velocity_data = metrics.get("velocity", {})
  completion_data = metrics.get("completion", {})
  bug_data = metrics.get("bugs", {})

  velocity = velocity_data.get("velocity", 0)
  completion_rate = completion_data.get("completion_rate", 0)
  # Focus on sprint-execution bugs (bugs resulting from sprint work, not imported)
  sprint_exec_bugs = bug_data.get("sprint_execution_bugs", 0)
  imported_bugs = bug_data.get("imported_bugs", 0)
  total_bugs = bug_data.get("total_bugs", 0)
  # Get critical bugs from sprint execution only
  sprint_exec_critical = bug_data.get("by_origin", {}).get("sprint_execution", {}).get("critical", 0)

  # Analyze patterns
  patterns = []

  if velocity < 20:
   patterns.append({
    "category": "velocity",
    "observation": "Low velocity",
    "severity": "medium",
    "recommendation": "Investigate blockers and team capacity"
   })
  elif velocity > 30:
   patterns.append({
    "category": "velocity",
    "observation": f"High velocity ({velocity:.0f} points)",
    "severity": "info",
    "recommendation": "Sustain current practices - team is performing well"
   })

  if completion_rate < 80:
   patterns.append({
    "category": "completion",
    "observation": "Low completion rate",
    "severity": "high",
    "recommendation": "Review sprint planning and estimation accuracy"
   })
  elif completion_rate >= 95:
   patterns.append({
    "category": "completion",
    "observation": f"Excellent completion rate ({completion_rate:.0f}%)",
    "severity": "info",
    "recommendation": "Maintain current planning and execution practices"
   })

  # Only flag critical bugs from sprint execution (not imported bugs)
  if sprint_exec_critical > 0:
   patterns.append({
    "category": "quality",
    "observation": f"{sprint_exec_critical} critical bug(s) from sprint work",
    "severity": "critical",
    "recommendation": "Strengthen testing and code review processes"
   })

  # Only count sprint-execution bugs for quality assessment
  if sprint_exec_bugs > 5:
   patterns.append({
    "category": "quality",
    "observation": f"High sprint-execution bug count ({sprint_exec_bugs})",
    "severity": "medium",
    "recommendation": "Improve test coverage and development practices"
   })
  elif sprint_exec_bugs == 0 and total_bugs > 0:
   patterns.append({
    "category": "quality",
    "observation": f"No bugs from sprint work ({imported_bugs} imported bugs addressed)",
    "severity": "info",
    "recommendation": "Quality practices are effective"
   })

  return {
   "patterns": patterns,
   "pattern_count": len(patterns),
   "analyzed_at": datetime.now().isoformat()
  }

 def _analyze_ai(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
  """AI-based analysis (Mode 2: AI with JSON validation)."""
  if not self.use_ai:
   return self._analyze_simple(metrics)

  # Define JSON schema
  schema = {
   "type": "object",
   "properties": {
    "themes": {
     "type": "object",
     "properties": {
      "positive": {"type": "array", "items": {"type": "string"}, "minItems": 1},
      "negative": {"type": "array", "items": {"type": "string"}},
      "neutral": {"type": "array", "items": {"type": "string"}}
     },
     "required": ["positive", "negative", "neutral"]
    },
    "action_items": {
     "type": "array",
     "items": {
      "type": "object",
      "properties": {
       "action": {"type": "string"},
       "priority": {"type": "string", "enum": ["high", "medium", "low"]},
       "category": {"type": "string", "enum": ["process", "technical", "team", "other"]}
      },
      "required": ["action", "priority", "category"]
     },
     "minItems": 1
    },
    "improvement_suggestions": {
     "type": "array",
     "items": {"type": "string"}
    }
   },
   "required": ["themes", "action_items", "improvement_suggestions"]
  }

  # Build contextual prompt
  velocity_data = metrics.get("velocity", {})
  completion_data = metrics.get("completion", {})
  bug_data = metrics.get("bugs", {})

  velocity = velocity_data.get("velocity", 0)
  completion_rate = completion_data.get("completion_rate", 0)
  total_bugs = bug_data.get("total_bugs", 0)
  sprint_exec_bugs = bug_data.get("sprint_execution_bugs", 0)
  imported_bugs = bug_data.get("imported_bugs", 0)

  prompt = f"""You are analyzing a sprint retrospective. First, read the project CLAUDE.md to understand the project context, then identify themes, categorize action items, and suggest improvements.

Sprint: {self.sprint_name}

Sprint Metrics:
- Velocity: {velocity:.0f} points
- Completion Rate: {completion_rate:.1f}%
- Total Bugs: {total_bugs}
 • Sprint-execution bugs: {sprint_exec_bugs} (bugs resulting from sprint work - parent is in sprint)
 • Imported bugs: {imported_bugs} (pre-existing bugs added to sprint from backlog)

NOTE: For quality assessment, focus on sprint-execution bugs. Imported bugs were not caused by sprint work.
High velocity and high completion rate are POSITIVE observations, not problems to fix.

Analyze this retrospective and return JSON with:
1. Themes (positive, negative, neutral) - identify patterns from the metrics
2. Action items categorized by priority and category
3. Improvement suggestions (only for actual problems, not for positive metrics)

Return ONLY valid JSON matching this exact schema:
{json.dumps(schema, indent=2)}"""

  # Try Claude Agent SDK first (provides tool access to read codebase)
  try:
   from pathlib import Path
   from scripts.workflow_executor.agent_sdk import AgentSDKWrapper

   sdk = AgentSDKWrapper(
    model="claude-sonnet-4-20250514",
    max_tokens=2000,
    working_directory=str(Path.cwd()),
    tool_preset="read_only"
   )

   result = sdk.ask(
    prompt=prompt,
    system_prompt="""You are a Scrum Master analyzing sprint retrospectives.
Read the project CLAUDE.md to understand the codebase context before providing analysis."""
   )

   response_text = result.response

   # Extract JSON if wrapped in code blocks
   json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response_text, re.DOTALL)
   if json_match:
    response_text = json_match.group(1)

   parsed_result = json.loads(response_text)
   validate(parsed_result, schema)

   # Track token usage
   if result.token_usage:
    self.token_usage["analyze_ai"] = {
     "input_tokens": result.token_usage.input_tokens,
     "output_tokens": result.token_usage.output_tokens,
     "cost_usd": result.cost_usd
    }

   # Convert to expected format
   patterns = []
   for theme in parsed_result["themes"]["positive"]:
    patterns.append({
     "category": "positive",
     "observation": theme,
     "severity": "info",
     "recommendation": "Continue this practice"
    })
   for theme in parsed_result["themes"]["negative"]:
    patterns.append({
     "category": "negative",
     "observation": theme,
     "severity": "medium",
     "recommendation": "Address this issue"
    })

   return {
    "patterns": patterns,
    "themes": parsed_result["themes"],
    "action_items": parsed_result["action_items"],
    "improvement_suggestions": parsed_result["improvement_suggestions"],
    "pattern_count": len(patterns),
    "analyzed_at": datetime.now().isoformat()
   }

  except ImportError as e:
   # ALWAYS report import errors - no silent fallbacks
   from cli.console import console
   console.print()
   console.print("[bold #FF6B6B]AI Analysis Failed[/bold #FF6B6B]")
   console.print(f"[#FF6B6B]Import error: {e}[/#FF6B6B]")
   console.print("[#758B9B]The Claude Agent SDK is not available. AI analysis requires tool access.[/#758B9B]")
   console.print("[#758B9B]Install with: pip install claude-code-sdk[/#758B9B]")
   console.print("[#758B9B]Workflow will continue with simple pattern analysis.[/#758B9B]")
   return self._analyze_simple(metrics)

  except (json.JSONDecodeError, ValidationError) as e:
   # Report validation errors clearly
   from cli.console import console
   console.print()
   console.print("[bold #FF6B6B]AI Analysis Failed[/bold #FF6B6B]")
   console.print(f"[#FF6B6B]{type(e).__name__}: {e}[/#FF6B6B]")
   console.print("[#758B9B]Workflow will continue with simple pattern analysis.[/#758B9B]")
   return self._analyze_simple(metrics)

  except Exception as e:
   # ALWAYS report errors - no silent fallbacks
   from cli.console import console
   console.print()
   console.print("[bold #FF6B6B]AI Analysis Failed[/bold #FF6B6B]")
   console.print(f"[#FF6B6B]{type(e).__name__}: {e}[/#FF6B6B]")
   console.print("[#758B9B]Workflow will continue with simple pattern analysis.[/#758B9B]")
   return self._analyze_simple(metrics)

 def _analyze_anthropic_fallback(self, metrics: Dict[str, Any], schema: Dict[str, Any], prompt: str) -> Dict[str, Any]:
  """Fallback to Anthropic API when Agent SDK is unavailable."""
  if not self.claude_client:
   return self._analyze_simple(metrics)

  for attempt in range(3):
   try:
    response = self.claude_client.messages.create(
     model="claude-sonnet-4-20250514",
     max_tokens=2000,
     messages=[{"role": "user", "content": prompt}]
    )

    response_text = response.content[0].text

    # Extract JSON if wrapped in code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response_text, re.DOTALL)
    if json_match:
     response_text = json_match.group(1)

    result = json.loads(response_text)
    validate(result, schema)

    # Track token usage
    self.token_usage["analyze_ai"] = {
     "input_tokens": response.usage.input_tokens,
     "output_tokens": response.usage.output_tokens,
     "cost_usd": self._calculate_cost(response.usage)
    }

    # Convert to expected format
    patterns = []
    for theme in result["themes"]["positive"]:
     patterns.append({
      "category": "positive",
      "observation": theme,
      "severity": "info",
      "recommendation": "Continue this practice"
     })
    for theme in result["themes"]["negative"]:
     patterns.append({
      "category": "negative",
      "observation": theme,
      "severity": "medium",
      "recommendation": "Address this issue"
     })

    return {
     "patterns": patterns,
     "themes": result["themes"],
     "action_items": result["action_items"],
     "improvement_suggestions": result["improvement_suggestions"],
     "pattern_count": len(patterns),
     "analyzed_at": datetime.now().isoformat()
    }

   except (json.JSONDecodeError, ValidationError) as e:
    print(f" Attempt {attempt + 1}/3 failed: {type(e).__name__}: {e}")
    if attempt == 2:
     print(" All retries exhausted, falling back to simple logic")
     return self._analyze_simple(metrics)
   except Exception as e:
    print(f" API error: {type(e).__name__}: {e}, falling back to simple logic")
    return self._analyze_simple(metrics)

  return self._analyze_simple(metrics)

 def _calculate_cost(self, usage) -> float:
  """Calculate cost in USD based on Claude API token usage."""
  input_cost = (usage.input_tokens / 1_000_000) * 3.0  # $3 per million
  output_cost = (usage.output_tokens / 1_000_000) * 15.0  # $15 per million
  return input_cost + output_cost

 def _step_4_improvement_generation(self, context: Dict[str, Any]) -> Dict[str, Any]:
  """Step 4: Generate improvement items."""
  print("\n Generating improvement items...")

  analysis = self.step_evidence.get("3-ai-analysis", {})
  patterns = analysis.get("patterns", [])

  # Generate improvement items from patterns (skip "info" severity - those are positive observations)
  improvements = []
  positive_observations = []

  for idx, pattern in enumerate(patterns, 1):
   # Skip info-level patterns - they're positive observations, not improvements needed
   if pattern.get('severity') == 'info':
    positive_observations.append(pattern)
    continue

   improvement = {
    "id": f"improvement-{len(improvements) + 1}",
    "title": f"Improve {pattern['category']}: {pattern['observation']}",
    "description": pattern['recommendation'],
    "priority": self._map_severity_to_priority(pattern['severity']),
    "category": pattern['category'],
    "related_pattern": pattern
   }
   improvements.append(improvement)

  # Display positive observations separately
  if positive_observations:
   print(f"\n Positive observations ({len(positive_observations)}):")
   for obs in positive_observations:
    print(f" {obs['observation']}: {obs['recommendation']}")

  evidence = {
   "improvements": improvements,
   "improvement_count": len(improvements),
   "generated_at": datetime.now().isoformat()
  }

  print(f"\n Generated {len(improvements)} improvement item(s):")
  for imp in improvements:
   print(f" - [{imp['priority'].upper()}] {imp['title']}")

  return evidence

 def _map_severity_to_priority(self, severity: str) -> str:
  """Map analysis severity to task priority."""
  severity_map = {
   "critical": "high",
   "high": "high",
   "medium": "medium",
   "info": "low"
  }
  return severity_map.get(severity, "medium")

 def _step_5_approval_gate(self, context: Dict[str, Any]) -> Dict[str, Any]:
  """
  Step 5: Human approval gate (BLOCKING).

  Execution HALTS here until user approves/rejects.
  """
  improvements = self.step_evidence.get("4-improvement-generation", {}).get("improvements", [])
  metrics = self.step_evidence.get("2-metrics-collection", {})

  # Build approval criteria based on improvements
  approval_criteria = [
   f"Create {len(improvements)} improvement task(s) in work tracking system",
   "Improvements are actionable and specific",
   "Priorities are appropriate for sprint planning"
  ]

  # Create approval gate configuration
  gate_config = ApprovalGateConfig(
   gate_id="retrospective-task-creation",
   gate_name="Retrospective Task Creation Approval",
   description=f"Approve creation of {len(improvements)} improvement tasks for {self.sprint_name}",
   required_approvers=1,
   approval_criteria=approval_criteria,
   allow_comments=True
  )

  # Build context for approval display
  approval_context = {
   "sprint_name": self.sprint_name,
   "metrics": {
    "velocity": metrics.get('velocity', {}).get('velocity', 0),
    "completion_rate": metrics.get('completion', {}).get('completion_rate', 0)
   },
   "improvements": improvements,
   "improvement_count": len(improvements),
   "improvement_summary": [
    f"[{imp['priority'].upper()}] {imp['title']}"
    for imp in improvements
   ]
  }

  # Use standardized ApprovalGate class
  gate = ApprovalGate()
  decision = gate.request_approval(gate_config, approval_context)

  # Build evidence from decision
  evidence = {
   "approved": decision.approved,
   "timestamp": decision.timestamp,
   "items_approved": len(improvements) if decision.approved else 0,
   "approver": decision.approver,
   "comment": decision.comment
  }

  if decision.approved:
   print(f"\n User APPROVED - Will create {len(improvements)} task(s)")
  else:
   print("\n User DECLINED - Skipping task creation (will continue to report generation)")

  return evidence

 def _step_6_task_creation(self, context: Dict[str, Any]) -> Dict[str, Any]:
  """Step 6: Create improvement tasks."""
  print("\n Creating improvement tasks...")

  approval = self.step_evidence.get("5-approval-gate", {})
  improvements = self.step_evidence.get("4-improvement-generation", {}).get("improvements", [])

  if not approval.get("approved"):
   print(" Skipped - User did not approve task creation")
   return {"tasks_created": [], "skipped": True}

  if not self.adapter:
   print("️ No adapter - cannot create tasks")
   return {"tasks_created": [], "mock": True}

  created_tasks = []

  try:
   for improvement in improvements:
    # Create task using adapter
    result = self.adapter.create_work_item(
     item_type="Task",
     title=improvement['title'],
     description=improvement['description'],
     assigned_to=self.current_user,
     priority=improvement['priority']
    )

    task_id = result.get("id")
    created_tasks.append({
     "task_id": task_id,
     "title": improvement['title'],
     "improvement_id": improvement['id']
    })

    print(f" Created task: {task_id} - {improvement['title']}")

   evidence = {
    "tasks_created": created_tasks,
    "task_count": len(created_tasks),
    "created_at": datetime.now().isoformat()
   }

   print(f"\n Created {len(created_tasks)} task(s)")
   return evidence

  except Exception as e:
   print(f" Task creation failed: {e}")
   raise

 def _step_7_verification(self, context: Dict[str, Any]) -> Dict[str, Any]:
  """Step 7: Verify tasks exist (external source of truth)."""
  print("\n Verifying task creation...")

  task_creation = self.step_evidence.get("6-task-creation", {})

  if task_creation.get("skipped"):
   print(" Skipped - No tasks to verify")
   return {"verified": True, "skipped": True}

  created_tasks = task_creation.get("tasks_created", [])

  if not created_tasks:
   print(" No tasks to verify")
   return {"verified": True, "task_count": 0}

  if not self.adapter:
   print("️ No adapter - cannot verify tasks")
   return {"verified": False, "error": "No adapter"}

  try:
   verified_tasks = []

   for task in created_tasks:
    task_id = task['task_id']
    expected_title = task['title']

    # Query adapter for task (external source of truth) - Bug #2 fix
    work_item = self.adapter.get_work_item(task_id)

    if not work_item:
     raise ValueError(
      f"Task {task_id} claimed created but doesn't exist in tracking system"
     )

    # Verify fields match expected values
    actual_title = work_item.get("fields", {}).get("System.Title", work_item.get("title", ""))
    if actual_title and expected_title and actual_title != expected_title:
     raise ValueError(f"Title mismatch for task {task_id}: expected '{expected_title}', got '{actual_title}'")

    # Get state from fields or top-level
    actual_state = work_item.get("fields", {}).get("System.State", work_item.get("state", "Unknown"))

    verified_tasks.append({
     "task_id": task_id,
     "title": expected_title,
     "verified_title": actual_title,
     "verified_state": actual_state,
     "verified_at": datetime.now().isoformat()
    })

    print(f" Verified task: {task_id}")

   evidence = {
    "verified": True,
    "verified_tasks": verified_tasks,
    "verified_count": len(verified_tasks),
    "verified_at": datetime.now().isoformat()
   }

   print(f"\n Verified {len(verified_tasks)} task(s)")
   return evidence

  except Exception as e:
   print(f" Verification failed: {e}")
   raise

 def _step_8_report_generation(self, context: Dict[str, Any]) -> Dict[str, Any]:
  """Step 8: Generate retrospective report."""
  print("\n Generating retrospective report...")

  metrics = self.step_evidence.get("2-metrics-collection", {})
  improvements = self.step_evidence.get("4-improvement-generation", {}).get("improvements", [])

  # Use format_retrospective_report utility
  report_content = format_retrospective_report(
   sprint_name=self.sprint_name,
   metrics=metrics.get("sprint_analysis", {}),
   improvements=improvements
  )

  # Add additional sections
  task_creation = self.step_evidence.get("6-task-creation", {})

  if task_creation.get("tasks_created"):
   tasks_section = [
    "",
    "## Tasks Created",
    ""
   ]

   for task in task_creation['tasks_created']:
    tasks_section.append(f"- **{task['task_id']}:** {task['title']}")

   report_content += "\n".join(tasks_section)

  evidence = {
   "report_content": report_content,
   "report_length": len(report_content),
   "generated_at": datetime.now().isoformat()
  }

  print(" Report generated")
  return evidence

 def _step_9_save_report(self, context: Dict[str, Any]) -> Dict[str, Any]:
  """Step 9: Save report to file."""
  print("\n Saving retrospective report...")

  report_generation = self.step_evidence.get("8-report-generation", {})
  report_content = report_generation.get("report_content", "")

  # Save to .claude/retrospectives/
  report_dir = Path(".claude/retrospectives")
  report_dir.mkdir(parents=True, exist_ok=True)

  # Generate filename
  safe_sprint_name = self.sprint_name.lower().replace(' ', '-')
  timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
  report_file = report_dir / f"{safe_sprint_name}-{timestamp}.md"

  # Write report with UTF-8 encoding
  with open(report_file, 'w', encoding='utf-8') as f:
   f.write(report_content)

  evidence = {
   "report_file": str(report_file),
   "saved_at": datetime.now().isoformat()
  }

  print(f" Report saved: {report_file}")
  return evidence

 def _step_10_artifact_hygiene(self, context: Dict[str, Any]) -> Dict[str, Any]:
  """Step 10: Optional artifact hygiene after retrospective approval."""
  print("\n Step 10: Artifact Hygiene (Optional)")

  # Check if retrospective was approved
  approval = self.step_evidence.get("5-approval-gate", {})
  if not approval.get("approved", False):
   print(f" Skipping artifact hygiene (retrospective not approved)")
   return {"skipped": True, "reason": "retrospective_not_approved"}

  # Ask user if they want to run artifact hygiene
  print("\n" + "-" * 50)
  print("Sprint retrospective complete.")
  print("Would you like to clean up sprint artifacts?")
  print("-" * 50)

  try:
   response = input("Run artifact hygiene? (yes/no): ").strip().lower()
  except (EOFError, KeyboardInterrupt):
   response = "no"

  if response != "yes":
   print(" Reminder: Run `python scripts/artifact_hygiene.py` later to clean up")
   return {"skipped": True, "reason": "user_declined"}

  # Run artifact hygiene as sub-workflow
  try:
   from scripts.artifact_hygiene import ArtifactHygieneWorkflow

   print("\n Running artifact hygiene...")
   hygiene = ArtifactHygieneWorkflow(
    workflow_id=f"hygiene-{self.sprint_name}",
    use_ai=self.use_ai,
    current_sprint=self.sprint_name
   )
   result = hygiene.execute()
   return {
    "ran_hygiene": True,
    "hygiene_result": result,
    "timestamp": datetime.now().isoformat()
   }
  except Exception as e:
   print(f" ️ Artifact hygiene failed: {e}")
   print(" Run manually: python scripts/artifact_hygiene.py")
   return {"error": str(e), "ran_hygiene": False}

 def _step_11_checkpoint(self, context: Dict[str, Any]) -> Dict[str, Any]:
  """Step 11: Save final checkpoint."""
  print("\n Saving final checkpoint...")

  # State is automatically saved by WorkflowOrchestrator
  # This step just confirms completion

  evidence = {
   "checkpoint_saved": True,
   "timestamp": datetime.now().isoformat(),
   "workflow_complete": True
  }

  print(" Checkpoint saved")
  return evidence


def main():
 """Main entry point."""
 from cli.console import console

 parser = argparse.ArgumentParser(
  description="Sprint Retrospective Workflow - External enforcement with metrics collection and AI analysis"
 )
 parser.add_argument(
  "--sprint",
  required=True,
  help="Sprint name to retrospect (e.g., 'Sprint 7')"
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
  help="Use AI for retrospective analysis (Mode 2)"
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
  console.print(f"[dim]I normalized the sprint name: '{args.sprint}' -> '{sprint_name}'[/dim]")

 # Print header
 console.print()
 console.print("─" * 80)
 console.print("[bold #71E4D1]  SPRINT RETROSPECTIVE[/bold #71E4D1]")
 console.print("─" * 80)
 console.print()
 console.print(f"[#D9EAFC]Sprint:[/#D9EAFC] [bold]{sprint_name}[/bold]")
 mode_str = "Interactive AI" if args.interactive else ("AI-Assisted" if args.use_ai else "Fast (no AI)")
 console.print(f"[#D9EAFC]Mode:[/#D9EAFC] {mode_str}")
 console.print()

 # Generate workflow ID with timestamp (prevents caching external state across runs)
 workflow_id = args.workflow_id or f"retro-{sprint_name.lower().replace(' ', '-')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

 # Create and execute workflow
 workflow = SprintRetrospectiveWorkflow(
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
   console.print("[bold #71E4D1]  Retrospective complete![/bold #71E4D1]")
   console.print("─" * 80)
   console.print()
  sys.exit(0 if success else 1)
 except KeyboardInterrupt:
  console.print()
  console.print("[#758B9B]Retrospective cancelled by user.[/#758B9B]")
  sys.exit(130)
 except Exception as e:
  console.print()
  console.print("─" * 80)
  console.print(f"[bold #FF6B6B]  Oops! Something went wrong[/bold #FF6B6B]")
  console.print("─" * 80)
  console.print()
  console.print(f"[#FF6B6B]{e}[/#FF6B6B]")
  console.print()
  console.print("[#758B9B]You can resume from where you left off by running the same command again.[/#758B9B]")
  sys.exit(1)


if __name__ == "__main__":
 main()
