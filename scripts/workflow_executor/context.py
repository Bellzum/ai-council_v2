"""
Context Builders for Workflow Executor

Builds execution contexts for workflow steps by loading relevant CLAUDE.md files,
project configuration, and runtime data.

Integrates with:
- core/context_loader.py for hierarchical CLAUDE.md loading
- config/loader.py for project configuration
- Implements LRU caching for loaded files
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional
from functools import lru_cache
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import core modules
from core.context_loader import load_hierarchical_context
from config.loader import load_config


class ContextBuilder(ABC):
    """
    Base class for building execution contexts.

    Context builders load relevant documentation, configuration, and runtime
    data to provide workflow steps with everything they need.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize context builder.

        Args:
            config_path: Path to config.yaml (defaults to .claude/config.yaml)
        """
        self.config_path = config_path or Path(".claude/config.yaml")
        self._config: Optional[Any] = None

    @abstractmethod
    def build_context_for_step(
        self,
        step_id: str,
        step_type: str,
        prior_evidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build execution context for a workflow step.

        Args:
            step_id: Step identifier
            step_type: Type of step
            prior_evidence: Evidence from prior steps

        Returns:
            Context dict with all necessary data
        """
        pass

    def inject_runtime_data(
        self,
        context: Dict[str, Any],
        runtime_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Inject runtime data into context.

        Args:
            context: Existing context dict
            runtime_data: Runtime data to inject

        Returns:
            Updated context dict
        """
        context_copy = context.copy()
        context_copy["runtime_data"] = runtime_data
        return context_copy

    def _load_config(self) -> Any:
        """
        Load project configuration.

        Returns:
            FrameworkConfig instance

        Note: Cached after first load
        """
        if self._config is None:
            if self.config_path.exists():
                self._config = load_config(str(self.config_path))
            else:
                # Return minimal config if file doesn't exist
                from config.schema import FrameworkConfig, ProjectConfig, WorkTrackingConfig
                self._config = FrameworkConfig(
                    project=ProjectConfig(
                        name="default",
                        type="cli-tool",
                        tech_stack={"languages": ["Python"]}
                    ),
                    work_tracking=WorkTrackingConfig()
                )
        return self._config

    @staticmethod
    @lru_cache(maxsize=32)
    def _load_claude_md(directory: str) -> str:
        """
        Load CLAUDE.md files for a directory.

        Args:
            directory: Directory path

        Returns:
            Combined CLAUDE.md content

        Note: LRU cached to avoid re-reading same files
        """
        path = Path(directory)
        if not path.exists():
            return ""

        try:
            return load_hierarchical_context(path)
        except Exception as e:
            print(f"⚠️  Warning: Could not load context for {directory}: {e}")
            return ""

    def _get_base_context(self) -> Dict[str, Any]:
        """
        Get base context with config and common data.

        Returns:
            Base context dict
        """
        config = self._load_config()

        return {
            "config": {
                "project_name": config.project.name,
                "project_type": config.project.type,
                "tech_stack": config.project.tech_stack,
                "quality_standards": {
                    "test_coverage_min": config.quality_standards.test_coverage_min,
                    "critical_vulnerabilities_max": config.quality_standards.critical_vulnerabilities_max,
                    "code_complexity_max": config.quality_standards.code_complexity_max,
                },
                "work_tracking": {
                    "platform": config.work_tracking.platform,
                }
            }
        }


class TestVerificationContextBuilder(ContextBuilder):
    """
    Context builder for test verification steps.

    Loads test-related context including test directories, coverage requirements,
    and quality standards.
    """

    def build_context_for_step(
        self,
        step_id: str,
        step_type: str,
        prior_evidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build context for test verification.

        Args:
            step_id: Step identifier
            step_type: Type of step
            prior_evidence: Evidence from prior steps

        Returns:
            Context with test-specific data
        """
        # Get base context
        context = self._get_base_context()

        # Load test directory context
        config = self._load_config()
        test_dir = config.project.test_directory
        context["test_context"] = self._load_claude_md(test_dir)

        # Add test-specific config
        context["test_config"] = {
            "test_directory": test_dir,
            "coverage_min": config.quality_standards.test_coverage_min,
            "test_report_directory": ".claude/test-reports",
        }

        # Add prior evidence
        context["prior_evidence"] = prior_evidence

        # Check for existing test reports
        test_report_dir = Path(".claude/test-reports")
        if test_report_dir.exists():
            reports = list(test_report_dir.glob("*.md"))
            context["existing_reports"] = [str(r) for r in reports]
        else:
            context["existing_reports"] = []

        return context


class DeploymentContextBuilder(ContextBuilder):
    """
    Context builder for deployment steps.

    Loads deployment configuration, environment settings, and deployment history.
    """

    def build_context_for_step(
        self,
        step_id: str,
        step_type: str,
        prior_evidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build context for deployment.

        Args:
            step_id: Step identifier
            step_type: Type of step
            prior_evidence: Evidence from prior steps

        Returns:
            Context with deployment-specific data
        """
        # Get base context
        context = self._get_base_context()

        # Load deployment context
        config = self._load_config()

        # Add deployment config
        context["deployment_config"] = {
            "environments": config.deployment_config.environments,
            "default_environment": config.deployment_config.default_environment,
            "deployment_tasks_enabled": config.deployment_config.deployment_tasks_enabled,
        }

        # Add prior evidence
        context["prior_evidence"] = prior_evidence

        # Load deployment history if available
        deployment_reports_dir = Path(".claude/reports/deployments")
        if deployment_reports_dir.exists():
            reports = list(deployment_reports_dir.glob("*.md"))
            context["deployment_history"] = [
                {
                    "file": str(r),
                    "name": r.stem
                }
                for r in sorted(reports, reverse=True)[:10]  # Last 10
            ]
        else:
            context["deployment_history"] = []

        return context


class ApprovalGateContextBuilder(ContextBuilder):
    """
    Context builder for approval gates.

    Builds context for approval decisions including all prior evidence,
    quality metrics, and approval criteria.
    """

    def build_context_for_step(
        self,
        step_id: str,
        step_type: str,
        prior_evidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build context for approval gate.

        Args:
            step_id: Step identifier
            step_type: Type of step
            prior_evidence: Evidence from prior steps

        Returns:
            Context with approval-specific data
        """
        # Get base context
        context = self._get_base_context()

        # Add all prior evidence (approver needs full visibility)
        context["prior_evidence"] = prior_evidence

        # Build approval summary
        context["approval_summary"] = self._build_approval_summary(prior_evidence)

        # Add quality gate status
        context["quality_gates"] = self._check_quality_gates(prior_evidence)

        return context

    def _build_approval_summary(
        self,
        prior_evidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build summary of evidence for approval decision.

        Args:
            prior_evidence: Evidence from prior steps

        Returns:
            Summary dict
        """
        summary = {
            "steps_completed": len(prior_evidence),
            "evidence_available": list(prior_evidence.keys()),
        }

        # Extract key metrics if available
        if "1-metrics" in prior_evidence:
            metrics = prior_evidence["1-metrics"]
            summary["metrics"] = {
                "total_tasks": metrics.get("total_tasks", 0),
                "completed_tasks": metrics.get("completed_tasks", 0),
                "completion_rate": metrics.get("completion_rate", 0),
            }

        # Extract test results if available
        if "4-tests" in prior_evidence:
            test_info = prior_evidence["4-tests"]
            summary["tests"] = {
                "reports_found": test_info.get("reports_found", 0),
                "report_files": test_info.get("report_files", []),
            }

        # Extract review results if available
        if "5-reviews" in prior_evidence:
            reviews = prior_evidence["5-reviews"]
            summary["reviews"] = reviews

        return summary

    def _check_quality_gates(
        self,
        prior_evidence: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        Check quality gates based on evidence.

        Args:
            prior_evidence: Evidence from prior steps

        Returns:
            Dict of gate name -> pass/fail
        """
        gates = {}

        # Check completion rate gate
        if "1-metrics" in prior_evidence:
            metrics = prior_evidence["1-metrics"]
            completion_rate = metrics.get("completion_rate", 0)
            gates["completion_rate_80_percent"] = completion_rate >= 80

        # Check test reports gate
        if "4-tests" in prior_evidence:
            test_info = prior_evidence["4-tests"]
            reports_found = test_info.get("reports_found", 0)
            gates["test_reports_exist"] = reports_found > 0

        # Check reviews gate (all approve)
        if "5-reviews" in prior_evidence:
            reviews = prior_evidence["5-reviews"]
            all_approve = all(
                review.get("recommendation", "").startswith("APPROVE")
                for review in reviews.values()
            )
            gates["all_reviews_approve"] = all_approve

        return gates


# Factory function to get appropriate context builder
def get_context_builder(builder_type: str) -> ContextBuilder:
    """
    Get context builder by type.

    Args:
        builder_type: Type of builder (test_verification, deployment, approval_gate)

    Returns:
        ContextBuilder instance

    Raises:
        ValueError if builder_type is unknown
    """
    builders = {
        "test_verification": TestVerificationContextBuilder,
        "deployment": DeploymentContextBuilder,
        "approval_gate": ApprovalGateContextBuilder,
    }

    builder_class = builders.get(builder_type)
    if builder_class is None:
        raise ValueError(
            f"Unknown builder type: {builder_type}. "
            f"Available: {list(builders.keys())}"
        )

    return builder_class()


# Example usage
if __name__ == "__main__":
    # Test test verification context builder
    test_builder = TestVerificationContextBuilder()
    test_context = test_builder.build_context_for_step(
        step_id="4-tests",
        step_type="verification",
        prior_evidence={
            "1-metrics": {"total_tasks": 10, "completed_tasks": 10},
        }
    )
    print("✓ Test verification context:")
    print(f"  - Config loaded: {bool(test_context.get('config'))}")
    print(f"  - Test config: {bool(test_context.get('test_config'))}")
    print(f"  - Existing reports: {len(test_context.get('existing_reports', []))}")

    # Test approval gate context builder
    approval_builder = ApprovalGateContextBuilder()
    approval_context = approval_builder.build_context_for_step(
        step_id="7-approval",
        step_type="approval_gate",
        prior_evidence={
            "1-metrics": {
                "total_tasks": 10,
                "completed_tasks": 10,
                "completion_rate": 100.0
            },
            "4-tests": {
                "reports_found": 2,
                "report_files": ["test1.md", "test2.md"]
            },
            "5-reviews": {
                "qa": {"recommendation": "APPROVE"},
                "security": {"recommendation": "APPROVE"},
                "engineering": {"recommendation": "APPROVE"}
            }
        }
    )
    print("\n✓ Approval gate context:")
    print(f"  - Steps completed: {approval_context['approval_summary']['steps_completed']}")
    print(f"  - Quality gates: {approval_context['quality_gates']}")
    print(f"  - All gates passing: {all(approval_context['quality_gates'].values())}")
