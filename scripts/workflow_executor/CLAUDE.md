---
keywords: [workflow, orchestration, execution, approval, security, ai-modes]
task_types: [workflow_execution, process_orchestration, approval_management, security_enforcement]
priority: critical
max_tokens: 4000
children: []
dependencies: [core/context_loader, config/loader, shared/state]
---

# Workflow Executor

## Purpose

Core workflow execution engine that provides externally-enforced orchestration with three execution modes: pure Python, AI with JSON validation, and interactive AI sessions. Features blocking approval gates, comprehensive audit logging, and secure subprocess execution with strict security controls. See [README.md](README.md) for full documentation.

## Key Components

**base.py** - WorkflowOrchestrator base class providing foundation for all workflow execution modes with sequential step enforcement, evidence collection, and state management integration.

**approval_gates.py** - Blocking approval gate system using Python `input()` to physically halt execution until user approval. Creates unbypassable checkpoints with full audit trail logging.

**context.py** - Context builders that load relevant CLAUDE.md files, project configuration, and runtime data for workflow steps. Integrates with core context loader and implements LRU caching.

**interactive_session.py** - Mode 3 AI collaboration helper using Claude Agent SDK for multi-turn conversations. Provides reusable interface for interactive analysis and planning.

**error_handling.py** - Enhanced error handling with structured messages, error classification (transient/permission/business), retry logic with progress indicators, and recovery suggestions.

**progress.py** - Progress indicators including retry with progress display, animated spinners for long operations, progress bars for batch tasks, and user feedback during AI calls.

**schemas.py** - Pydantic models with strict validation for workflow definitions, execution contexts, results, and approval gates. All models are immutable with security constraint validation.

**subprocess_security.py** - Security wrapper for subprocess execution with command allowlist enforcement, shell=True prevention, comprehensive audit logging, and timeout protection.

## Constraints for AI Agents

**CRITICAL SECURITY RULES:**
- NEVER use `shell=True` in subprocess calls - always use explicit argument lists
- All subprocess commands MUST be validated against allowlist in subprocess_security.py
- Approval gates using `input()` are UNBYPASSABLE - do not attempt to bypass or mock

**Execution Mode Requirements:**
- Mode 1 (Pure Python): No AI involvement, direct execution only
- Mode 2 (AI + JSON): AI generates JSON that gets validated against schemas before execution
- Mode 3 (Interactive): Multi-turn AI collaboration through InteractiveSession class

**Audit Trail Mandatory:**
- All workflow steps must be logged with evidence collection
- Approval gate decisions must be recorded to JSONL audit files
- Subprocess executions must be logged with full command details

**Sequential Enforcement:**
- Workflow steps execute in strict sequence - no skipping allowed
- Each step must complete and provide evidence before next step begins
- Use state management integration for checkpoint/resume capabilities

**Error Handling Protocol:**
- Use error_handling.py for all workflow exceptions
- Classify errors appropriately (transient/permission/business rule)
- Provide structured error messages with troubleshooting guidance
- Implement retry logic with progress indicators where appropriate

## Related

- [core/context_loader](../../core/context_loader/CLAUDE.md) - Hierarchical CLAUDE.md loading
- [config/loader](../../config/loader/CLAUDE.md) - Project configuration management
- [shared/state](../../shared/state/CLAUDE.md) - State persistence and checkpoint/resume
- [scripts/workflows](../workflows/CLAUDE.md) - Concrete workflow implementations