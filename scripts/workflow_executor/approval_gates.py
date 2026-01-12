"""
Approval Gate Orchestration

Provides blocking approval gates that physically halt workflow execution until
user provides approval.

CRITICAL: This uses Python input() which is a BLOCKING system call. AI cannot
bypass this gate because execution literally stops until user types input.

Key Features:
- Blocking enforcement (input() halts execution)
- Audit trail (all decisions logged to JSONL)
- Context presentation (show evidence to approver)
- Timeout support (auto-reject after timeout)
"""

from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json
import sys
import threading
import queue

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.workflow_executor.schemas import ApprovalGateConfig, ApprovalDecision


class ApprovalGate:
    """
    Blocking approval gate orchestration.

    This class guarantees workflow compliance through Python input() which
    physically halts execution until user provides input.

    Example:
        gate = ApprovalGate()

        config = ApprovalGateConfig(
            gate_id="sprint-closure",
            gate_name="Sprint Closure Approval",
            description="Approve closing Sprint 7",
            approval_criteria=[
                "All tests passing",
                "No critical bugs"
            ]
        )

        context = {
            "metrics": {"completion_rate": 100.0},
            "tests": {"reports_found": 2}
        }

        decision = gate.request_approval(config, context)

        if decision.approved:
            # Continue workflow
            close_sprint()
        else:
            # Cancel workflow
            print("Sprint closure cancelled")
    """

    def __init__(self, audit_log_dir: Optional[Path] = None):
        """
        Initialize approval gate.

        Args:
            audit_log_dir: Directory for audit logs (defaults to .claude/audit)
        """
        self.audit_log_dir = audit_log_dir or Path(".claude/audit")
        self.audit_log_dir.mkdir(parents=True, exist_ok=True)

        # Audit log file (JSONL format - one decision per line)
        today = datetime.now().strftime("%Y%m%d")
        self.audit_log_file = self.audit_log_dir / f"approvals-{today}.jsonl"

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
            valid_choices: List of valid choice strings (e.g., ["yes", "no"])
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

            # Note: user_input is already stripped on line 171, so no need to check whitespace-only
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

    def request_approval(
        self,
        gate_config: ApprovalGateConfig,
        context: Dict[str, Any],
        timeout_seconds: Optional[int] = None
    ) -> ApprovalDecision:
        """
        Request approval from user (BLOCKING).

        This method HALTS execution until user provides input. AI cannot bypass
        because Python input() is a blocking system call.

        Args:
            gate_config: Approval gate configuration
            context: Context for approval decision (evidence, metrics, etc.)
            timeout_seconds: Optional timeout (overrides config)

        Returns:
            ApprovalDecision with approval status and metadata

        Note:
            If timeout is set and expires, auto-rejects if auto_reject_on_timeout=True
        """
        print("\n" + "=" * 70)
        print(f"⏸️  APPROVAL GATE: {gate_config.gate_name}")
        print("=" * 70)
        print(f"\n🔒 BLOCKING CHECKPOINT - Execution halted pending approval")
        print(f"\nGate ID: {gate_config.gate_id}")
        print(f"Description: {gate_config.description}")

        # Show approval criteria
        if gate_config.approval_criteria:
            print(f"\nApproval Criteria:")
            for idx, criterion in enumerate(gate_config.approval_criteria, 1):
                print(f"  {idx}. {criterion}")

        # Show context summary
        self._display_context_summary(context)

        print("\n" + "-" * 70)
        print("DECISION REQUIRED:")
        print("  yes = Approve and continue")
        print("  no  = Reject and cancel")
        print("-" * 70 + "\n")

        # BLOCKING CALL - Execution halts here until user types input
        try:
            # Handle timeout if specified
            timeout = timeout_seconds or gate_config.timeout_seconds or 300

            if timeout:
                print(f"⏲️  Timeout: {timeout} seconds")

            # Get validated input with max retries and timeout
            response = self._get_validated_input(
                prompt="Approve? (yes/no): ",
                valid_choices=["yes", "no"],
                max_retries=5,
                timeout_seconds=timeout
            )

        except (EOFError, KeyboardInterrupt):
            print("\n\n❌ Approval cancelled by user (Ctrl+C or EOF)")
            response = "no"

        # Parse response
        approved = response == "yes"

        # Get comment if allowed (optional input with timeout - no validation needed)
        comment = None
        if gate_config.allow_comments:
            try:
                comment = self._get_user_input_with_timeout(
                    "Comment (optional): ",
                    timeout_seconds=timeout
                ).strip()
                if not comment:
                    comment = None
            except (EOFError, KeyboardInterrupt):
                comment = None
            except TimeoutError:
                # Timeout on optional comment is not critical - just skip
                comment = None
                print("  (comment skipped due to timeout)")

        # Create decision record
        decision = ApprovalDecision(
            gate_id=gate_config.gate_id,
            approved=approved,
            approver="user",
            timestamp=datetime.now().isoformat(),
            comment=comment,
            evidence=context
        )

        # Log to audit trail
        self._log_decision(decision, gate_config)

        # Display result
        if approved:
            print(f"\n✅ APPROVED - {gate_config.gate_name}")
            if comment:
                print(f"   Comment: {comment}")
        else:
            print(f"\n❌ REJECTED - {gate_config.gate_name}")
            if comment:
                print(f"   Reason: {comment}")

        print("=" * 70 + "\n")

        return decision

    def _display_context_summary(self, context: Dict[str, Any]) -> None:
        """
        Display context summary for approval decision.

        Args:
            context: Context dict with evidence
        """
        print(f"\nContext:")

        # Show metrics if available
        if "metrics" in context or "approval_summary" in context:
            summary = context.get("approval_summary", context.get("metrics", {}))

            if "metrics" in summary:
                metrics = summary["metrics"]
                print(f"  Completion Rate: {metrics.get('completion_rate', 'N/A')}%")
                print(f"  Tasks: {metrics.get('completed_tasks', 'N/A')} / {metrics.get('total_tasks', 'N/A')}")

            if "tests" in summary:
                tests = summary["tests"]
                print(f"  Test Reports: {tests.get('reports_found', 0)}")

            if "reviews" in summary:
                reviews = summary["reviews"]
                print(f"  Reviews:")
                for reviewer, review in reviews.items():
                    rec = review.get("recommendation", "N/A")
                    print(f"    - {reviewer}: {rec}")

        # Show quality gates if available
        if "quality_gates" in context:
            gates = context["quality_gates"]
            print(f"\n  Quality Gates:")
            for gate_name, passed in gates.items():
                status = "✓" if passed else "✗"
                print(f"    {status} {gate_name}")

        # Show steps completed
        if "prior_evidence" in context:
            evidence = context["prior_evidence"]
            print(f"\n  Steps Completed: {len(evidence)}")
            for step_id in list(evidence.keys())[:5]:  # Show first 5
                print(f"    - {step_id}")
            if len(evidence) > 5:
                print(f"    ... and {len(evidence) - 5} more")

    def _log_decision(
        self,
        decision: ApprovalDecision,
        gate_config: ApprovalGateConfig
    ) -> None:
        """
        Log approval decision to audit trail.

        Args:
            decision: Approval decision
            gate_config: Gate configuration
        """
        # Create audit entry (JSONL format - one JSON object per line)
        audit_entry = {
            "timestamp": decision.timestamp,
            "gate_id": decision.gate_id,
            "gate_name": gate_config.gate_name,
            "approved": decision.approved,
            "approver": decision.approver,
            "comment": decision.comment,
            "approval_criteria": gate_config.approval_criteria,
            # Don't log full evidence (too large), just summary
            "evidence_keys": list(decision.evidence.keys())
        }

        # Append to JSONL file
        with open(self.audit_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(audit_entry) + "\n")

    def get_approval_history(
        self,
        gate_id: Optional[str] = None,
        limit: int = 100
    ) -> list:
        """
        Get approval history from audit log.

        Args:
            gate_id: Optional filter by gate ID
            limit: Maximum number of entries to return

        Returns:
            List of approval decisions (most recent first)
        """
        if not self.audit_log_file.exists():
            return []

        entries = []

        with open(self.audit_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())

                    # Filter by gate ID if specified
                    if gate_id and entry.get("gate_id") != gate_id:
                        continue

                    entries.append(entry)

                except json.JSONDecodeError:
                    continue

        # Return most recent first
        return entries[-limit:][::-1]

    def get_approval_stats(self) -> Dict[str, Any]:
        """
        Get approval statistics.

        Returns:
            Dict with approval stats (total, approved, rejected, approval rate)
        """
        history = self.get_approval_history()

        total = len(history)
        approved = sum(1 for entry in history if entry.get("approved"))
        rejected = total - approved
        approval_rate = (approved / total * 100) if total > 0 else 0

        return {
            "total_decisions": total,
            "approved": approved,
            "rejected": rejected,
            "approval_rate": approval_rate
        }


# Example usage
if __name__ == "__main__":
    # Create approval gate
    gate = ApprovalGate()

    # Create gate configuration
    config = ApprovalGateConfig(
        gate_id="test-approval",
        gate_name="Test Approval Gate",
        description="Testing approval gate functionality",
        required_approvers=1,
        approval_criteria=[
            "All tests passing",
            "Code reviewed",
            "Documentation updated"
        ],
        allow_comments=True
    )

    # Create context
    context = {
        "approval_summary": {
            "metrics": {
                "completion_rate": 95.0,
                "total_tasks": 20,
                "completed_tasks": 19
            },
            "tests": {
                "reports_found": 3
            },
            "reviews": {
                "qa": {"recommendation": "APPROVE"},
                "security": {"recommendation": "APPROVE"}
            }
        },
        "quality_gates": {
            "completion_rate_80_percent": True,
            "test_reports_exist": True,
            "all_reviews_approve": True
        }
    }

    # Request approval
    print("Testing approval gate...")
    decision = gate.request_approval(config, context)

    print(f"\nDecision recorded:")
    print(f"  Approved: {decision.approved}")
    print(f"  Timestamp: {decision.timestamp}")
    print(f"  Comment: {decision.comment}")
    print(f"\n✓ Audit log: {gate.audit_log_file}")

    # Show stats
    stats = gate.get_approval_stats()
    print(f"\nApproval Stats:")
    print(f"  Total: {stats['total_decisions']}")
    print(f"  Approved: {stats['approved']}")
    print(f"  Rejected: {stats['rejected']}")
    print(f"  Approval Rate: {stats['approval_rate']:.1f}%")
