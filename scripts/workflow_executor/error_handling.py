"""
Enhanced Error Handling for External Enforcement Workflows.

Provides:
- Structured error messages with troubleshooting guidance
- Error classification (transient, permission, business rule)
- Retry logic with progress indicators
- Recovery suggestions

Usage:
    from scripts.workflow_executor.error_handling import (
        format_error_with_guidance,
        ErrorClassifier,
        WorkflowError
    )

    try:
        result = adapter.create_work_item(...)
    except Exception as e:
        formatted = format_error_with_guidance(e, context={"operation": "create_work_item"})
        print(formatted)
"""
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class ErrorType(Enum):
    """Classification of error types for guidance."""
    TRANSIENT = "transient"           # Network, timeout - retry likely to help
    PERMISSION = "permission"         # Auth, access denied - fix credentials
    BUSINESS_RULE = "business_rule"   # Validation, constraint - fix input
    CONFIGURATION = "configuration"   # Config error - fix config file
    VERIFICATION = "verification"     # External verification failed
    UNKNOWN = "unknown"


@dataclass
class ErrorGuidance:
    """Structured error guidance."""
    error_type: ErrorType
    what_happened: str
    troubleshooting: List[str]
    recovery: str
    can_retry: bool = False


class ErrorClassifier:
    """Classify errors by type for appropriate handling."""

    # Error patterns mapped to types
    PATTERNS = {
        ErrorType.TRANSIENT: [
            r"503",
            r"504",
            r"timeout",
            r"connection.*refused",
            r"temporary.*failure",
            r"service.*unavailable",
            r"network.*error",
            r"rate.*limit",
        ],
        ErrorType.PERMISSION: [
            r"401",
            r"403",
            r"unauthorized",
            r"forbidden",
            r"access.*denied",
            r"authentication.*failed",
            r"permission.*denied",
            r"token.*expired",
            r"invalid.*credentials",
        ],
        ErrorType.BUSINESS_RULE: [
            r"400",
            r"validation.*failed",
            r"invalid.*input",
            r"constraint.*violation",
            r"already.*exists",
            r"not.*allowed",
            r"required.*field",
        ],
        ErrorType.CONFIGURATION: [
            r"config.*not.*found",
            r"invalid.*config",
            r"missing.*setting",
            r"platform.*not.*configured",
        ],
        ErrorType.VERIFICATION: [
            r"doesn't exist in tracking system",
            r"verification.*failed",
            r"not.*found.*after.*creation",
            r"claimed.*created.*but",
        ],
    }

    @classmethod
    def classify(cls, error: Exception) -> ErrorType:
        """
        Classify an error by type.

        Args:
            error: The exception to classify

        Returns:
            ErrorType classification
        """
        error_text = str(error).lower()

        for error_type, patterns in cls.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, error_text, re.IGNORECASE):
                    return error_type

        return ErrorType.UNKNOWN


class TroubleshootingGuide:
    """Provides troubleshooting guidance based on error type."""

    GUIDES = {
        ErrorType.TRANSIENT: {
            "what_happened": "A temporary network or service issue occurred.",
            "troubleshooting": [
                "Check Azure DevOps status: https://status.dev.azure.com",
                "Wait 2-5 minutes and retry",
                "Verify network connection is stable",
            ],
            "recovery": "Workflow saved at last checkpoint - safe to retry",
            "can_retry": True,
        },
        ErrorType.PERMISSION: {
            "what_happened": "Authentication or authorization failed.",
            "troubleshooting": [
                "Verify AZURE_DEVOPS_PAT environment variable is set",
                "Check PAT token has required scopes (Work Items: Read & Write)",
                "Regenerate PAT if expired (Settings -> Personal Access Tokens)",
                "Verify organization/project access",
            ],
            "recovery": "Fix credentials, then retry workflow",
            "can_retry": False,
        },
        ErrorType.BUSINESS_RULE: {
            "what_happened": "The operation violated a business rule or validation constraint.",
            "troubleshooting": [
                "Review the input data for errors",
                "Check required fields are provided",
                "Verify work item type is valid",
                "Check for duplicate work items",
            ],
            "recovery": "Fix input data, then retry workflow",
            "can_retry": False,
        },
        ErrorType.CONFIGURATION: {
            "what_happened": "Configuration is missing or invalid.",
            "troubleshooting": [
                "Verify .claude/config.yaml exists",
                "Check work_tracking.platform is set to 'azure-devops' or 'file-based'",
                "Run: trustable-ai validate",
            ],
            "recovery": "Fix configuration, then retry workflow",
            "can_retry": False,
        },
        ErrorType.VERIFICATION: {
            "what_happened": "External verification failed - work item may not exist.",
            "troubleshooting": [
                "Work item was claimed created but cannot be found",
                "Check Azure DevOps for the work item manually",
                "Verify network was stable during creation",
                "Check for silent API failures in Azure DevOps",
            ],
            "recovery": "Investigate root cause, then decide whether to retry or recreate",
            "can_retry": False,
        },
        ErrorType.UNKNOWN: {
            "what_happened": "An unexpected error occurred.",
            "troubleshooting": [
                "Review the full error message for details",
                "Check .claude/audit/ logs for more context",
                "Verify all dependencies are installed correctly",
            ],
            "recovery": "Investigate error, then retry workflow",
            "can_retry": False,
        },
    }

    @classmethod
    def get_guidance(cls, error_type: ErrorType) -> ErrorGuidance:
        """
        Get troubleshooting guidance for an error type.

        Args:
            error_type: The classified error type

        Returns:
            ErrorGuidance with troubleshooting steps
        """
        guide = cls.GUIDES.get(error_type, cls.GUIDES[ErrorType.UNKNOWN])

        return ErrorGuidance(
            error_type=error_type,
            what_happened=guide["what_happened"],
            troubleshooting=guide["troubleshooting"],
            recovery=guide["recovery"],
            can_retry=guide["can_retry"],
        )


def format_error_with_guidance(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    workflow_name: str = "",
    step_name: str = ""
) -> str:
    """
    Format an error with comprehensive troubleshooting guidance.

    Args:
        error: The exception that occurred
        context: Optional context dict with additional info
        workflow_name: Name of the workflow that failed
        step_name: Name of the step that failed

    Returns:
        Formatted error message with guidance

    Example:
        try:
            adapter.create_work_item(...)
        except Exception as e:
            print(format_error_with_guidance(e, context={"operation": "create_work_item"}))
    """
    error_type = ErrorClassifier.classify(error)
    guidance = TroubleshootingGuide.get_guidance(error_type)

    context = context or {}

    lines = [
        f"\n{'=' * 70}",
        f"WHAT HAPPENED:",
        f"{'=' * 70}",
        f"",
        guidance.what_happened,
        f"",
        f"Error: {str(error)}",
    ]

    if workflow_name:
        lines.append(f"Workflow: {workflow_name}")
    if step_name:
        lines.append(f"Step: {step_name}")

    lines.extend([
        f"",
        f"{'=' * 70}",
        f"TROUBLESHOOTING:",
        f"{'=' * 70}",
        f"",
    ])

    for i, step in enumerate(guidance.troubleshooting, 1):
        lines.append(f"{i}. {step}")

    lines.extend([
        f"",
        f"{'=' * 70}",
        f"RECOVERY:",
        f"{'=' * 70}",
        f"",
        guidance.recovery,
    ])

    if guidance.can_retry:
        lines.extend([
            f"",
            f"This error is likely transient. You can retry the workflow:",
        ])
        if workflow_name:
            lines.append(f"  python scripts/{workflow_name.replace('-', '_')}.py [original args]")
    else:
        lines.extend([
            f"",
            f"Fix the issue described above before retrying.",
        ])

    return "\n".join(lines)


class WorkflowError(Exception):
    """
    Enhanced workflow exception with classification and guidance.

    Usage:
        raise WorkflowError(
            message="Task creation failed",
            error_type=ErrorType.VERIFICATION,
            context={"task_id": 1234}
        )
    """

    def __init__(
        self,
        message: str,
        error_type: ErrorType = ErrorType.UNKNOWN,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.error_type = error_type
        self.context = context or {}
        self.original_error = original_error

    def get_guidance(self) -> ErrorGuidance:
        """Get troubleshooting guidance for this error."""
        return TroubleshootingGuide.get_guidance(self.error_type)

    def format(self, workflow_name: str = "", step_name: str = "") -> str:
        """Format this error with full guidance."""
        return format_error_with_guidance(
            self.original_error or self,
            context=self.context,
            workflow_name=workflow_name,
            step_name=step_name
        )
