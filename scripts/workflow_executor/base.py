"""
WorkflowOrchestrator Base Class

Provides foundation for externally-enforced workflows with three execution modes:
- Mode 1: Pure Python (no AI)
- Mode 2: AI with JSON validation
- Mode 3: Interactive AI sessions

Key guarantees:
- Sequential step execution (no skipping)
- Evidence collection at each step
- Comprehensive audit logging
- Integration with state management for checkpoint/resume
"""

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import json
import sys
import threading
import queue

# Import core state management
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.state_manager import WorkflowState

# Import UX enhancement modules
from scripts.workflow_executor.error_handling import (
    format_error_with_guidance,
    ErrorClassifier,
    ErrorType,
    WorkflowError
)
from scripts.workflow_executor.progress import (
    print_step_header,
    print_success,
    print_warning,
    print_error,
    print_info,
    Spinner,
    ProgressBar
)


class ExecutionMode(Enum):
    """Workflow execution modes."""
    PURE_PYTHON = "pure_python"  # Mode 1: No AI, pure automation
    AI_JSON_VALIDATION = "ai_json_validation"  # Mode 2: AI with JSON schemas
    INTERACTIVE_AI = "interactive_ai"  # Mode 3: Interactive sessions


class WorkflowOrchestrator(ABC):
    """
    Base class for externally-enforced workflow orchestration.

    This class provides the foundation for workflows that guarantee compliance through:
    1. External flow control (Python enforces step order)
    2. Evidence collection (external verification at each step)
    3. Audit logging (comprehensive trail of all actions)
    4. State persistence (checkpoint/resume capability)

    Subclasses must implement:
    - _define_steps(): Return list of step definitions
    - _execute_step(): Execute a single workflow step

    Example:
        class SprintReviewWorkflow(WorkflowOrchestrator):
            def _define_steps(self):
                return [
                    {"id": "1-metrics", "name": "Collect Metrics"},
                    {"id": "2-analysis", "name": "Analyze Work Items"},
                ]

            def _execute_step(self, step, context):
                if step["id"] == "1-metrics":
                    return self._collect_metrics()
                # ...
    """

    def __init__(
        self,
        workflow_name: str,
        workflow_id: str,
        mode: ExecutionMode = ExecutionMode.PURE_PYTHON,
        enable_checkpoints: bool = True,
        quiet_mode: bool = False
    ):
        """
        Initialize workflow orchestrator.

        Args:
            workflow_name: Name of workflow (e.g., "sprint-review-enforced")
            workflow_id: Unique ID for this execution (e.g., "Sprint-7")
            mode: Execution mode (PURE_PYTHON, AI_JSON_VALIDATION, INTERACTIVE_AI)
            enable_checkpoints: Enable state checkpointing after each step
            quiet_mode: If True, suppress workflow progress output (default: False)
        """
        self.workflow_name = workflow_name
        self.workflow_id = workflow_id
        self.mode = mode
        self.enable_checkpoints = enable_checkpoints
        self.quiet_mode = quiet_mode

        # State tracking
        self.steps_completed: List[str] = []
        self.step_evidence: Dict[str, Any] = {}
        self.start_time = datetime.now()

        # State manager for persistence
        if enable_checkpoints:
            self.state = WorkflowState(workflow_name, workflow_id, quiet_mode=quiet_mode)

            # ENFORCEMENT: Check state signature verification
            if self.state.state_file.exists() and not self.state.signature_verified:
                self._handle_unverified_state()

            # Resume from existing state if available
            if self.state.state.get("completed_steps"):
                self.steps_completed = [
                    s["name"] for s in self.state.state["completed_steps"]
                ]
                # Load evidence from state
                for step in self.state.state["completed_steps"]:
                    step_id = step["name"]
                    self.step_evidence[step_id] = step.get("result", {})
        else:
            self.state = None

        # Workflow step definitions (loaded from subclass)
        self._steps: List[Dict[str, Any]] = []

        # Audit log
        self.audit_log: List[Dict[str, Any]] = []

    @abstractmethod
    def _define_steps(self) -> List[Dict[str, Any]]:
        """
        Define workflow steps.

        Must return list of step definitions with:
        - id: Unique step identifier (e.g., "1-metrics")
        - name: Human-readable step name
        - required: Whether step is mandatory (default True)
        - approval_gate: Whether this step is an approval gate

        Returns:
            List of step definitions
        """
        pass

    @abstractmethod
    def _execute_step(
        self,
        step: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single workflow step.

        Args:
            step: Step definition from _define_steps()
            context: Execution context (prior evidence, config, etc.)

        Returns:
            Evidence dict with results from this step

        Raises:
            Exception if step execution fails
        """
        pass

    def _validate_evidence(
        self,
        evidence: Dict[str, Any],
        step_id: str,
        required_keys: Optional[List[str]] = None
    ) -> None:
        """
        Validate step evidence for required keys and proper structure.

        Bug #1190: Validates evidence dict has required keys before using.
        Prevents crashes from missing evidence keys.

        Args:
            evidence: Evidence dict returned from step execution
            step_id: Step identifier for error messages
            required_keys: List of required keys (optional, defaults to basic validation)

        Raises:
            ValueError: If evidence is missing required keys or has invalid structure
        """
        if not isinstance(evidence, dict):
            raise ValueError(
                f"Evidence from step '{step_id}' must be a dict, got {type(evidence).__name__}"
            )

        if required_keys:
            missing_keys = [key for key in required_keys if key not in evidence]
            if missing_keys:
                raise ValueError(
                    f"Evidence from step '{step_id}' is missing required keys: {missing_keys}. "
                    f"Available keys: {list(evidence.keys())}"
                )

    def _get_user_input_with_timeout(
        self,
        prompt: str,
        timeout_seconds: int = 300
    ) -> str:
        """
        Get user input with timeout using threading.

        This implementation uses threading.Thread + queue.Queue to implement
        timeout for input() calls. This approach works cross-platform (Windows,
        Linux, macOS) unlike signal-based approaches which don't work on Windows.

        Args:
            prompt: Prompt to display to user
            timeout_seconds: Timeout in seconds (default: 300 = 5 minutes)

        Returns:
            User input string

        Raises:
            TimeoutError: If user doesn't respond within timeout
            EOFError: If input cancelled (EOF signal)
        """
        result_queue = queue.Queue()

        def read_input():
            """Thread function to read from stdin."""
            try:
                user_input = input(prompt)
                result_queue.put(user_input)
            except EOFError:
                result_queue.put(None)  # EOF signal
            except Exception as e:
                result_queue.put(e)  # Propagate exceptions

        # Start input thread (daemon=True ensures clean shutdown)
        input_thread = threading.Thread(target=read_input, daemon=True)
        input_thread.start()

        try:
            # Wait for input with timeout
            user_input = result_queue.get(timeout=timeout_seconds)

            # Check for exceptions from input thread
            if isinstance(user_input, Exception):
                raise user_input

            # Check for EOF signal
            if user_input is None:
                raise EOFError("Input cancelled (EOF)")

            return user_input

        except queue.Empty:
            # Timeout occurred
            raise TimeoutError(
                f"User input timeout after {timeout_seconds} seconds. "
                f"Workflow aborted.\n\n"
                f"Troubleshooting:\n"
                f"  - Increase timeout with --input-timeout parameter\n"
                f"  - Re-run workflow (will resume from last checkpoint)\n"
            )

    def _get_validated_input(
        self,
        prompt: str,
        valid_choices: list,
        max_retries: int = 5,
        timeout_seconds: int = 300
    ) -> str:
        """
        Get validated input from user with max retry limit and timeout.

        Integrates timeout support (Bug #1186) with validation logic (Bug #1184).

        Args:
            prompt: Prompt to display to user
            valid_choices: List of valid choice strings (e.g., ["1", "2", "3"])
            max_retries: Maximum number of retry attempts (default: 5)
            timeout_seconds: Timeout in seconds for each input attempt (default: 300 = 5 minutes)

        Returns:
            Valid user input (one of valid_choices)

        Raises:
            RuntimeError: If max retries exceeded
            TimeoutError: If user doesn't respond within timeout on any attempt
            EOFError: If input cancelled (EOF signal)
        """
        attempts = 0
        while attempts < max_retries:
            user_input = self._get_user_input_with_timeout(prompt, timeout_seconds).strip()

            # Validate input
            if not user_input:
                remaining = max_retries - attempts - 1
                if remaining > 0:
                    if remaining == 1:
                        print("Invalid choice: empty input. This is your last attempt.")
                    else:
                        print(f"Invalid choice: empty input. Please try again ({remaining} attempts remaining)")
                attempts += 1
                continue

            # Note: user_input is already stripped on line 299, so no need to check whitespace-only
            # The empty check above handles both empty and whitespace-only inputs

            # Check if in valid choices (case-insensitive)
            if user_input.lower() not in [choice.lower() for choice in valid_choices]:
                remaining = max_retries - attempts - 1
                if remaining > 0:
                    if remaining == 1:
                        print(f"Invalid choice: '{user_input}'. This is your last attempt.")
                    else:
                        print(f"Invalid choice: '{user_input}'. Please try again ({remaining} attempts remaining)")
                attempts += 1
                continue

            # Valid input - return it (preserving case from valid_choices)
            for choice in valid_choices:
                if user_input.lower() == choice.lower():
                    return choice

        # Max retries exceeded
        raise RuntimeError(
            f"Maximum retry attempts ({max_retries}) exceeded for user input. "
            f"Valid choices were: {', '.join(valid_choices)}"
        )

    def _handle_unverified_state(self):
        """
        Handle unverified state (signature verification failed or no signature).

        Prompts user for remediation decision:
        1. Continue with unverified state (risky)
        2. Cross-verify against external source of truth
        3. Abort workflow

        Raises:
            SystemExit if user aborts workflow
            RuntimeError if max retry attempts exceeded for user input
        """
        print("\n" + "=" * 80)
        print("⚠️  STATE INTEGRITY ISSUE")
        print("=" * 80)
        print()
        print(f"Workflow: {self.workflow_name}")
        print(f"State file: {self.state.state_file}")
        print()
        print("The workflow state file signature verification failed or is missing.")
        print("This may indicate:")
        print("  - State corruption (file modified, disk error)")
        print("  - AI tampering (agent modified state to fake completion)")
        print("  - Key rotation (secret key changed since state was signed)")
        print("  - Old format state (created before HMAC signatures)")
        print()
        print("=" * 80)
        print("REMEDIATION OPTIONS:")
        print("=" * 80)
        print()
        print("1. Continue with unverified state (RISKY)")
        print("   - Workflow will execute but state cannot be trusted")
        print("   - Use if you trust the state file (e.g., key rotation)")
        print()
        print("2. Cross-verify state against external source of truth")
        print("   - Query work tracking system to verify all claimed work items exist")
        print("   - Recommended if state may be corrupted")
        print()
        print("3. Abort workflow (SAFE)")
        print("   - Manually inspect state file")
        print("   - Fix corruption or delete state to start fresh")
        print()
        print("=" * 80)
        print()

        # Get validated input with max retries
        response = self._get_validated_input(
            prompt="Enter choice (1/2/3): ",
            valid_choices=["1", "2", "3"],
            max_retries=5
        )

        if response == "1":
            # Continue with unverified state
            print("\n⚠️  Continuing with UNVERIFIED state (user accepted risk)")
            print("    Workflow will proceed but state integrity is NOT guaranteed")
            return

        elif response == "2":
            # Cross-verify against external source of truth
            # This may require retries if cross-verification fails
            while True:
                print("\n🔍 Cross-verifying state against external source of truth...")

                # Check if adapter is available
                if not hasattr(self, 'adapter'):
                    print("\n❌ Cannot cross-verify: Workflow has no 'adapter' attribute")
                    print("   Cross-verification requires work tracking adapter")
                    print("   Please choose option 1 (continue) or 3 (abort)")

                    # Get new choice
                    response = self._get_validated_input(
                        prompt="Enter choice (1/3): ",
                        valid_choices=["1", "3"],
                        max_retries=5
                    )

                    if response == "1":
                        print("\n⚠️  Continuing with UNVERIFIED state (user accepted risk)")
                        print("    Workflow will proceed but state integrity is NOT guaranteed")
                        return
                    else:  # response == "3"
                        print("\n❌ Workflow aborted by user")
                        print(f"   State file: {self.state.state_file}")
                        print("   Next steps:")
                        print("     1. Inspect state file manually")
                        print("     2. Delete state file to start fresh")
                        print("     3. Fix corruption and re-run workflow")
                        sys.exit(1)

                # Cross-verify
                verification_result = self.state.cross_verify_with_adapter(self.adapter)

                if verification_result["verified"]:
                    print("\n✅ Cross-verification SUCCESSFUL")
                    print("   All work items verified in tracking system")
                    print("   State is trustworthy despite signature failure")
                    print()
                    print("   Continuing with workflow...")
                    return
                else:
                    print("\n❌ Cross-verification FAILED")
                    print(f"   Missing work items: {verification_result['missing_items']}")
                    print(f"   Errors: {len(verification_result['verification_errors'])}")
                    print()
                    print("   State file claims work items created that don't exist")
                    print("   This indicates:")
                    print("     - AI lied about creating work items")
                    print("     - State was tampered with after creation")
                    print("     - Work items were deleted externally")
                    print()
                    print("   RECOMMENDATION: Abort workflow (option 3), inspect state file")
                    print()

                    # Get new choice after failed verification
                    response = self._get_validated_input(
                        prompt="Enter choice (1/3): ",
                        valid_choices=["1", "3"],
                        max_retries=5
                    )

                    if response == "1":
                        print("\n⚠️  Continuing with UNVERIFIED state (user accepted risk)")
                        print("    Workflow will proceed but state integrity is NOT guaranteed")
                        return
                    else:  # response == "3"
                        print("\n❌ Workflow aborted by user")
                        print(f"   State file: {self.state.state_file}")
                        print("   Next steps:")
                        print("     1. Inspect state file manually")
                        print("     2. Delete state file to start fresh")
                        print("     3. Fix corruption and re-run workflow")
                        sys.exit(1)

        elif response == "3":
            # Abort workflow
            print("\n❌ Workflow aborted by user")
            print(f"   State file: {self.state.state_file}")
            print("   Next steps:")
            print("     1. Inspect state file manually")
            print("     2. Delete state file to start fresh")
            print("     3. Fix corruption and re-run workflow")
            sys.exit(1)

    def execute(self) -> bool:
        """
        Execute the complete workflow with external enforcement.

        This method guarantees:
        - All steps execute in order (sequential enforcement)
        - No steps can be skipped (external verification)
        - Evidence collected for each step
        - Audit trail persisted to disk

        Returns:
            True if workflow completed successfully, False otherwise
        """
        # Load step definitions
        self._steps = self._define_steps()

        # Track workflow timing
        workflow_start = datetime.now()

        if not self.quiet_mode:
            # Print structured workflow banner
            from cli.console import console
            console.print("\n" + "=" * 70, style="accent1")
            console.print(f"🔒 WORKFLOW: {self.workflow_name.upper()}", style="bold_primary")
            console.print("=" * 70, style="accent1")
            console.print(f"ID: {self.workflow_id}", style="secondary")
            console.print(f"Mode: {self.mode.value}", style="secondary")
            console.print(f"Steps: {len(self._steps)}", style="secondary")
            console.print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}", style="secondary")
            console.print("=" * 70, style="accent1")

        self._log_audit("workflow_started", {
            "workflow_name": self.workflow_name,
            "workflow_id": self.workflow_id,
            "mode": self.mode.value,
            "total_steps": len(self._steps)
        })

        try:
            # Execute each step sequentially
            for step in self._steps:
                step_id = step["id"]

                # Skip if already completed (resume scenario)
                if step_id in self.steps_completed:
                    if not self.quiet_mode:
                        print(f"\n⏭️  Step {step_id} already completed (resuming)")
                    continue

                # Execute step
                success = self._execute_step_with_enforcement(step)

                if not success:
                    if not self.quiet_mode:
                        print(f"\n❌ Workflow failed at step: {step_id}")
                    self._finalize_workflow(status="failed")
                    return False

            # All steps completed
            if not self.quiet_mode:
                # Print structured completion banner with timing
                from cli.console import console
                workflow_end = datetime.now()
                duration = (workflow_end - workflow_start).total_seconds()

                console.print("\n" + "=" * 70, style="success")
                console.print("✅ WORKFLOW COMPLETE", style="bold_success")
                console.print("=" * 70, style="success")
                console.print(f"Duration: {duration:.1f}s", style="success")
                console.print(f"Steps completed: {len(self.steps_completed)}/{len(self._steps)}", style="success")
                console.print("=" * 70, style="success")

            self._finalize_workflow(status="completed")
            return True

        except KeyboardInterrupt:
            print_warning("\nWorkflow interrupted by user (Ctrl+C)")
            self._finalize_workflow(status="interrupted")
            return False
        except Exception as e:
            # Use enhanced error handling with troubleshooting guidance
            error_message = format_error_with_guidance(
                e,
                workflow_name=self.workflow_name,
                step_name="workflow-execution"
            )
            print(error_message)
            self._finalize_workflow(status="error", error=str(e))
            return False

    def _execute_step_with_enforcement(self, step: Dict[str, Any]) -> bool:
        """
        Execute a single step with external enforcement.

        Args:
            step: Step definition

        Returns:
            True if step succeeded, False otherwise
        """
        step_id = step["id"]
        step_name = step.get("name", step_id)

        # Use enhanced step header from progress module
        step_number = len(self.steps_completed) + 1
        total_steps = len(self._steps)
        if not self.quiet_mode:
            print_step_header(step_number, step_name, total_steps)

        # Mark step as started in state
        if self.state:
            self.state.start_step(step_id)

        self._log_audit("step_started", {
            "step_id": step_id,
            "step_name": step_name
        })

        try:
            # Build execution context
            context = self._build_execution_context(step)

            # Execute step
            evidence = self._execute_step(step, context)

            # Validate evidence structure (Bug #1190)
            self._validate_evidence(evidence, step_id)

            # Store evidence
            self.step_evidence[step_id] = evidence
            self.steps_completed.append(step_id)

            # Persist to state manager
            if self.state:
                self.state.complete_step(step_id, result=evidence)

            self._log_audit("step_completed", {
                "step_id": step_id,
                "evidence_keys": list(evidence.keys())
            })

            # Bug #1193: Flush audit log after each step (prevents loss on crash)
            self._flush_audit_log()

            if not self.quiet_mode:
                print(f"✅ Step {step_id} complete")
            return True

        except Exception as e:
            # Use enhanced error handling with troubleshooting guidance
            error_message = format_error_with_guidance(
                e,
                context={"step_id": step_id, "step_name": step_name},
                workflow_name=self.workflow_name,
                step_name=step_name
            )
            print(error_message)

            self._log_audit("step_failed", {
                "step_id": step_id,
                "error": str(e),
                "error_type": ErrorClassifier.classify(e).value
            })

            if self.state:
                self.state.record_error(f"Step {step_id} failed", {"error": str(e)})

            return False

    def _build_execution_context(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build execution context for a step.

        Args:
            step: Step definition

        Returns:
            Context dict with prior evidence and metadata
        """
        return {
            "workflow_name": self.workflow_name,
            "workflow_id": self.workflow_id,
            "step_id": step["id"],
            "mode": self.mode.value,
            "prior_evidence": self.step_evidence.copy(),
            "steps_completed": self.steps_completed.copy(),
        }

    def _log_audit(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Log audit event.

        Args:
            event_type: Type of event (workflow_started, step_completed, etc.)
            data: Event data
        """
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        self.audit_log.append(audit_entry)

    def _flush_audit_log(self) -> None:
        """
        Flush audit log to disk immediately.

        Bug #1193: Saves audit log after each step instead of only at end.
        This prevents audit log loss if workflow crashes mid-execution.
        """
        if not self.audit_log:
            return

        audit_dir = Path(".claude/audit")
        audit_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        audit_file = audit_dir / f"{self.workflow_name}-{self.workflow_id}-{timestamp}.json"

        audit_data = {
            "workflow": self.workflow_name,
            "workflow_id": self.workflow_id,
            "mode": self.mode.value,
            "start_time": self.start_time.isoformat(),
            "last_update": datetime.now().isoformat(),
            "steps_completed": self.steps_completed,
            "step_evidence": self.step_evidence,
            "audit_log": self.audit_log,
            "enforcement": {
                "mode": "external",
                "guarantee": "All steps verified externally - AI cannot skip or bypass"
            }
        }

        # Write to temp file then rename (atomic operation)
        temp_file = audit_file.with_suffix('.json.tmp')
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(audit_data, f, indent=2)
            temp_file.rename(audit_file)
        except Exception as e:
            # Non-fatal: log warning but continue
            print(f"⚠️  Warning: Failed to flush audit log: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def _finalize_workflow(
        self,
        status: str,
        error: Optional[str] = None
    ) -> None:
        """
        Finalize workflow execution.

        Args:
            status: Workflow status (completed, failed, interrupted, error)
            error: Error message if status is error
        """
        # Update state manager
        if self.state:
            if status == "completed":
                self.state.complete_workflow()
            else:
                self.state.fail_workflow(f"Workflow {status}: {error or 'Unknown'}")

        # Save audit log
        audit_path = self._save_audit_log(status, error)
        if not self.quiet_mode:
            print(f"\n📋 Audit log: {audit_path}")

        self._log_audit("workflow_finalized", {
            "status": status,
            "error": error,
            "steps_completed": len(self.steps_completed),
            "total_steps": len(self._steps)
        })

    def _save_audit_log(
        self,
        status: str,
        error: Optional[str] = None
    ) -> Path:
        """
        Save comprehensive audit log to disk.

        Args:
            status: Workflow status
            error: Error message if applicable

        Returns:
            Path to audit log file
        """
        audit_dir = Path(".claude/audit")
        audit_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        audit_file = audit_dir / f"{self.workflow_name}-{self.workflow_id}-{timestamp}.json"

        audit_data = {
            "workflow": self.workflow_name,
            "workflow_id": self.workflow_id,
            "mode": self.mode.value,
            "status": status,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "steps_completed": self.steps_completed,
            "step_evidence": self.step_evidence,
            "audit_log": self.audit_log,
            "enforcement": {
                "mode": "external",
                "guarantee": "All steps verified externally - AI cannot skip or bypass"
            }
        }

        if error:
            audit_data["error"] = error

        with open(audit_file, 'w', encoding='utf-8') as f:
            json.dump(audit_data, f, indent=2)

        return audit_file

    def get_step_evidence(self, step_id: str) -> Optional[Dict[str, Any]]:
        """
        Get evidence from a completed step.

        Args:
            step_id: Step identifier

        Returns:
            Evidence dict if step completed, None otherwise
        """
        return self.step_evidence.get(step_id)

    def is_step_completed(self, step_id: str) -> bool:
        """
        Check if step was completed.

        Args:
            step_id: Step identifier

        Returns:
            True if step in completed list
        """
        return step_id in self.steps_completed

    # =========================================================================
    # UX Enhancement Helper Methods
    # =========================================================================

    def create_spinner(self, message: str) -> Spinner:
        """
        Create a spinner for long-running operations.

        Usage:
            with self.create_spinner("Analyzing with AI"):
                result = self.call_ai_api(...)

        Args:
            message: Message to display while spinning

        Returns:
            Spinner context manager
        """
        return Spinner(message)

    def create_progress_bar(
        self,
        total: int,
        label: str = "Progress"
    ) -> ProgressBar:
        """
        Create a progress bar for batch operations.

        Usage:
            with self.create_progress_bar(len(items), "Creating tasks") as bar:
                for item in items:
                    create_task(item)
                    bar.update()

        Args:
            total: Total number of items
            label: Label for progress bar

        Returns:
            ProgressBar context manager
        """
        return ProgressBar(total=total, label=label)

    def print_success(self, message: str) -> None:
        """Print success message with icon."""
        print_success(message)

    def print_warning(self, message: str) -> None:
        """Print warning message with icon."""
        print_warning(message)

    def print_error(self, message: str) -> None:
        """Print error message with icon."""
        print_error(message)

    def print_info(self, message: str) -> None:
        """Print info message with icon."""
        print_info(message)

    def format_error(self, error: Exception, step_name: str = "") -> str:
        """
        Format an error with troubleshooting guidance.

        Args:
            error: The exception that occurred
            step_name: Name of the step (optional)

        Returns:
            Formatted error message with guidance
        """
        return format_error_with_guidance(
            error,
            workflow_name=self.workflow_name,
            step_name=step_name
        )
