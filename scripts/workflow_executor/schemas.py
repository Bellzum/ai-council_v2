"""
JSON Schemas for Workflow Executor

Pydantic models with strict validation for workflow step definitions, execution
contexts, results, and approval gates.

All models are immutable (frozen=True) and use strict mode for validation.
Custom validators enforce security constraints (e.g., path traversal prevention).
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum
from pathlib import Path


class StepType(str, Enum):
    """Types of workflow steps."""
    DATA_COLLECTION = "data_collection"
    ANALYSIS = "analysis"
    VERIFICATION = "verification"
    AI_REVIEW = "ai_review"
    APPROVAL_GATE = "approval_gate"
    ACTION = "action"


class ExecutionStatus(str, Enum):
    """Execution status values."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStepDefinition(BaseModel):
    """
    Definition of a single workflow step.

    Immutable model that defines what a step does and how it should be executed.
    """
    model_config = ConfigDict(frozen=True, strict=True)

    id: str = Field(..., description="Unique step identifier (e.g., '1-metrics')")
    name: str = Field(..., description="Human-readable step name")
    step_type: StepType = Field(..., description="Type of step")
    description: str = Field("", description="Detailed step description")
    required: bool = Field(True, description="Whether step is mandatory")
    depends_on: List[str] = Field(default_factory=list, description="Step IDs this step depends on")
    timeout_seconds: Optional[int] = Field(None, description="Step timeout in seconds", ge=1)
    retry_count: int = Field(0, description="Number of retries on failure", ge=0, le=5)

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate step ID format."""
        if not v or not v.strip():
            raise ValueError("Step ID cannot be empty")
        if len(v) > 100:
            raise ValueError("Step ID too long (max 100 chars)")
        # Only allow alphanumeric, dash, underscore
        if not all(c.isalnum() or c in "-_" for c in v):
            raise ValueError("Step ID must be alphanumeric with dash/underscore only")
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate step name."""
        if not v or not v.strip():
            raise ValueError("Step name cannot be empty")
        if len(v) > 200:
            raise ValueError("Step name too long (max 200 chars)")
        return v


class StepExecutionContext(BaseModel):
    """
    Context for executing a workflow step.

    Contains all information needed to execute a step, including prior evidence,
    configuration, and runtime data.
    """
    model_config = ConfigDict(frozen=True, strict=True)

    workflow_name: str = Field(..., description="Name of the workflow")
    workflow_id: str = Field(..., description="Unique workflow execution ID")
    step_id: str = Field(..., description="Current step ID")
    mode: str = Field(..., description="Execution mode")
    prior_evidence: Dict[str, Any] = Field(default_factory=dict, description="Evidence from prior steps")
    steps_completed: List[str] = Field(default_factory=list, description="List of completed step IDs")
    config: Dict[str, Any] = Field(default_factory=dict, description="Workflow configuration")
    runtime_data: Dict[str, Any] = Field(default_factory=dict, description="Runtime-injected data")

    @field_validator("workflow_name", "workflow_id", "step_id", "mode")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        """Ensure required string fields are non-empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class WorkflowExecutionResult(BaseModel):
    """
    Result of workflow execution.

    Immutable record of what happened during workflow execution.
    """
    model_config = ConfigDict(frozen=True, strict=True)

    workflow_name: str = Field(..., description="Workflow name")
    workflow_id: str = Field(..., description="Workflow execution ID")
    status: ExecutionStatus = Field(..., description="Final execution status")
    start_time: str = Field(..., description="ISO format start time")
    end_time: str = Field(..., description="ISO format end time")
    duration_seconds: float = Field(..., description="Execution duration", ge=0)
    steps_completed: List[str] = Field(default_factory=list, description="Completed step IDs")
    steps_failed: List[str] = Field(default_factory=list, description="Failed step IDs")
    step_evidence: Dict[str, Any] = Field(default_factory=dict, description="Evidence from all steps")
    error: Optional[str] = Field(None, description="Error message if failed")
    audit_log_path: Optional[str] = Field(None, description="Path to audit log file")

    @field_validator("audit_log_path")
    @classmethod
    def validate_audit_log_path(cls, v: Optional[str]) -> Optional[str]:
        """Validate audit log path for security."""
        if v is None:
            return v

        # Reject path traversal attempts
        if ".." in v:
            raise ValueError("Path traversal not allowed in audit_log_path")

        # Reject absolute paths outside workspace
        path = Path(v)
        if path.is_absolute():
            # Check if it's within allowed directories
            allowed_prefixes = [".claude/audit", ".claude/workflow-state"]
            if not any(str(path).startswith(prefix) for prefix in allowed_prefixes):
                raise ValueError(f"Absolute paths must be under {allowed_prefixes}")

        return v


class ApprovalGateConfig(BaseModel):
    """
    Configuration for an approval gate.

    Defines what needs approval and who can approve.
    """
    model_config = ConfigDict(frozen=True, strict=True)

    gate_id: str = Field(..., description="Unique gate identifier")
    gate_name: str = Field(..., description="Human-readable gate name")
    description: str = Field("", description="What is being approved")
    required_approvers: int = Field(1, description="Number of approvals required", ge=1)
    approval_criteria: List[str] = Field(default_factory=list, description="Criteria for approval")
    timeout_seconds: Optional[int] = Field(None, description="Timeout for approval", ge=1)
    auto_reject_on_timeout: bool = Field(True, description="Auto-reject if timeout expires")
    allow_comments: bool = Field(True, description="Allow approver comments")

    @field_validator("gate_id")
    @classmethod
    def validate_gate_id(cls, v: str) -> str:
        """Validate gate ID format."""
        if not v or not v.strip():
            raise ValueError("Gate ID cannot be empty")
        if len(v) > 100:
            raise ValueError("Gate ID too long (max 100 chars)")
        if not all(c.isalnum() or c in "-_" for c in v):
            raise ValueError("Gate ID must be alphanumeric with dash/underscore only")
        return v


class ApprovalDecision(BaseModel):
    """
    Record of an approval decision.

    Immutable record of who approved/rejected and when.
    """
    model_config = ConfigDict(frozen=True, strict=True)

    gate_id: str = Field(..., description="Gate identifier")
    approved: bool = Field(..., description="Whether approved")
    approver: str = Field("user", description="Who approved (user, system, etc.)")
    timestamp: str = Field(..., description="ISO format decision time")
    comment: Optional[str] = Field(None, description="Approver comment")
    evidence: Dict[str, Any] = Field(default_factory=dict, description="Evidence presented for approval")

    @field_validator("comment")
    @classmethod
    def validate_comment(cls, v: Optional[str]) -> Optional[str]:
        """Validate comment length."""
        if v is not None and len(v) > 1000:
            raise ValueError("Comment too long (max 1000 chars)")
        return v


class FilePathValidation(BaseModel):
    """
    Helper model for validating file paths.

    Used to validate file paths in various contexts to prevent path traversal
    and other security issues.
    """
    model_config = ConfigDict(frozen=True, strict=True)

    path: str = Field(..., description="File path to validate")
    must_exist: bool = Field(False, description="Whether file must exist")
    allowed_extensions: List[str] = Field(default_factory=list, description="Allowed file extensions")
    allowed_directories: List[str] = Field(default_factory=list, description="Allowed base directories")

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """
        Validate file path for security.

        Rejects:
        - Path traversal attempts (..)
        - Absolute paths outside workspace
        - Null bytes
        """
        # Reject null bytes
        if "\0" in v:
            raise ValueError("Null bytes not allowed in paths")

        # Reject path traversal
        if ".." in v:
            raise ValueError("Path traversal (..) not allowed")

        # Reject absolute paths (enforce relative paths only)
        path = Path(v)
        if path.is_absolute():
            raise ValueError("Absolute paths not allowed - use relative paths only")

        return v

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization validation."""
        path = Path(self.path)

        # Check existence if required
        if self.must_exist and not path.exists():
            raise ValueError(f"Path does not exist: {self.path}")

        # Check extension if specified
        if self.allowed_extensions:
            if path.suffix not in self.allowed_extensions:
                raise ValueError(
                    f"File extension {path.suffix} not in allowed list: {self.allowed_extensions}"
                )

        # Check directory if specified
        if self.allowed_directories:
            # Check if path starts with any allowed directory
            allowed = any(
                str(path).startswith(allowed_dir)
                for allowed_dir in self.allowed_directories
            )
            if not allowed:
                raise ValueError(
                    f"Path {self.path} not in allowed directories: {self.allowed_directories}"
                )


# Example usage and validation
if __name__ == "__main__":
    # Test step definition
    step = WorkflowStepDefinition(
        id="1-metrics",
        name="Collect Sprint Metrics",
        step_type=StepType.DATA_COLLECTION,
        description="Query work items and calculate metrics",
        required=True,
        depends_on=[],
        timeout_seconds=300
    )
    print(f"✓ Step definition valid: {step.id}")

    # Test execution context
    context = StepExecutionContext(
        workflow_name="sprint-review",
        workflow_id="sprint-7",
        step_id="1-metrics",
        mode="pure_python",
        prior_evidence={},
        steps_completed=[]
    )
    print(f"✓ Execution context valid: {context.step_id}")

    # Test approval gate config
    gate = ApprovalGateConfig(
        gate_id="sprint-closure-approval",
        gate_name="Sprint Closure Approval",
        description="Approve closing Sprint 7",
        required_approvers=1,
        approval_criteria=[
            "All tests passing",
            "No critical bugs",
            "Documentation updated"
        ]
    )
    print(f"✓ Approval gate valid: {gate.gate_id}")

    # Test path validation - valid path
    try:
        valid_path = FilePathValidation(
            path=".claude/audit/test.json",
            allowed_directories=[".claude/audit", ".claude/workflow-state"]
        )
        print(f"✓ Path validation passed: {valid_path.path}")
    except ValueError as e:
        print(f"✗ Path validation failed: {e}")

    # Test path validation - path traversal (should fail)
    try:
        invalid_path = FilePathValidation(
            path="../../../etc/passwd",
            allowed_directories=[".claude/audit"]
        )
        print(f"✗ Path traversal NOT blocked: {invalid_path.path}")
    except ValueError as e:
        print(f"✓ Path traversal blocked: {e}")

    # Test path validation - absolute path (should fail)
    try:
        invalid_path = FilePathValidation(
            path="/etc/passwd",
            allowed_directories=[".claude/audit"]
        )
        print(f"✗ Absolute path NOT blocked: {invalid_path.path}")
    except ValueError as e:
        print(f"✓ Absolute path blocked: {e}")
