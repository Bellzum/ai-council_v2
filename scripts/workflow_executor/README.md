# Workflow Executor

Core workflow execution engine that provides externally-enforced workflow orchestration with comprehensive compliance guarantees, blocking approval gates, and multi-mode execution capabilities.

## Overview

The Workflow Executor is the heart of the external enforcement pattern, ensuring workflows cannot be bypassed or skipped through Python-controlled execution flow. It provides three distinct execution modes and implements critical security measures including blocking approval gates that physically halt execution until user approval.

### Key Features

- **External Enforcement**: Python-controlled execution that prevents AI from bypassing workflow steps
- **Blocking Approval Gates**: Uses `input()` system calls that literally stop execution until user provides approval
- **Multi-Mode Execution**: Pure Python, AI with JSON validation, and interactive AI sessions
- **Comprehensive Auditing**: Full audit trail with JSONL logging for all decisions and executions
- **Security-First Design**: Command allowlists, no shell execution, path traversal prevention

## Key Components

### Core Execution Engine

- **`base.py`** - `WorkflowOrchestrator` base class providing foundation for all workflows with sequential step execution, evidence collection, and state management integration
- **`schemas.py`** - Pydantic models with strict validation for workflow definitions, execution contexts, and results. All models are immutable and enforce security constraints

### Approval and Security

- **`approval_gates.py`** - Blocking approval gates using Python `input()` calls that physically halt workflow execution. Includes audit trail, context presentation, and timeout support
- **`subprocess_security.py`** - Secure subprocess wrapper with command allowlist enforcement, comprehensive audit logging, and timeout protection. **Never uses shell=True**

### Context and Validation

- **`context.py`** - Context builders that load relevant CLAUDE.md files, project configuration, and runtime data. Integrates with core context loader and implements LRU caching
- **`validators.py`** - Hierarchy validation utilities for work item validation, Feature readiness checks, and orphan detection using work tracking adapters (no CLI commands)

### User Experience

- **`progress.py`** - Progress indicators including retry logic with progress bars, animated spinners for long operations, and user feedback during AI calls
- **`error_handling.py`** - Enhanced error handling with structured messages, error classification (transient, permission, business rule), and recovery suggestions
- **`interactive_session.py`** - Interactive session helper for Mode 3 AI collaboration using Claude Agent SDK for multi-turn conversations

## Usage Examples

### Basic Workflow Orchestration

```python
from scripts.workflow_executor import WorkflowOrchestrator, ExecutionMode
from scripts.workflow_executor.schemas import WorkflowStepDefinition

class MyWorkflow(WorkflowOrchestrator):
    def __init__(self):
        super().__init__(
            workflow_name="my-workflow",
            execution_mode=ExecutionMode.AI_WITH_VALIDATION
        )
    
    def define_steps(self) -> List[WorkflowStepDefinition]:
        return [
            WorkflowStepDefinition(
                step_id="validate_input",
                description="Validate input parameters",
                step_type=StepType.VALIDATION,
                required_approvals=["technical_lead"]
            )
        ]

workflow = MyWorkflow()
result = workflow.execute()
```

### Blocking Approval Gate

```python
from scripts.workflow_executor.approval_gates import ApprovalGate

gate = ApprovalGate(
    gate_id="feature_ready_gate",
    description="Confirm Feature is ready for sprint",
    required_approvals=["product_owner", "technical_lead"],
    timeout_seconds=300
)

# This BLOCKS until approval is provided
approval_result = gate.request_approval(context=feature_context)
```

### Interactive AI Session

```python
from scripts.workflow_executor.interactive_session import InteractiveSession

session = InteractiveSession(
    workflow_name="sprint-planning",
    session_id="sprint-8"
)

response = session.ask("Analyze the current backlog for sprint readiness")
```

## Security Guarantees

- **Command Execution**: All subprocess calls use allowlist validation and never use `shell=True`
- **Path Security**: Path traversal prevention in all file operations
- **Audit Trail**: Comprehensive logging of all executions and decisions
- **Blocking Gates**: Physical execution halts that cannot be bypassed by AI

## Dependencies

- **Core Context System**: `core/context_loader.py` for hierarchical CLAUDE.md loading
- **Configuration**: `config/loader.py` for project configuration
- **Work Tracking**: Integration with work tracking adapters for validation
- **Claude Agent SDK**: For interactive AI sessions
- **Pydantic**: For strict schema validation and security

## Related Systems

This module integrates with:
- State management for checkpoint/resume functionality
- Work tracking adapters for external verification
- Core context system for workflow context building
- Configuration system for project-specific settings