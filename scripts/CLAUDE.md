---
context:
  purpose: "External enforcement workflows that solve AI unreliability through Python-controlled execution with external verification"
  problem_solved: "Slash commands running inside Claude Code rely on AI to execute steps and verify work - which fails silently. External enforcement workflows use Python to control execution, enforce step order, and verify all work against external sources of truth."
  keywords: [scripts, external, enforcement, workflow, sprint, planning, execution, review, verification]
  task_types: [workflow, sprint-planning, sprint-execution, sprint-review, retrospective]
  priority: high
  max_tokens: 1000
  children: []
  dependencies: [core, workflows, adapters]
---
# External Enforcement Workflows

## Purpose

Solves **AI claiming work done that wasn't** by running workflow control outside Claude Code.

Traditional slash commands inside Claude Code:
- AI claims "I created EPIC #1234" → EPIC may not exist
- AI can skip verification steps
- Session crashes lose all progress

External enforcement workflows:
- Python script controls execution (not AI)
- Queries adapter after each operation to verify work exists
- State persisted to disk, resume on crash

See [docs/EXTERNAL_ENFORCEMENT.md](../docs/EXTERNAL_ENFORCEMENT.md) for complete documentation.

---

## Available Scripts

| Script | Purpose | Key Flags |
|--------|---------|-----------|
| `product-intake.py` | Triage ideas into EPICs | `--interactive`, `--file` |
| `backlog-grooming.py` | Break Features into Tasks | `--epic-id`, `--use-agents` |
| `sprint-planning.py` | Plan sprint, create work items | `--sprint-number`, `--capacity` |
| `sprint-execution.py` | Execute sprint tasks | `--sprint`, `--confirm-start` |
| `sprint-review.py` | Review sprint outcomes | `--sprint` |
| `sprint-retrospective.py` | Analyze sprint performance | `--sprint` |
| `daily-standup.py` | Generate standup report | `--sprint` |
| `context-generation.py` | Generate CLAUDE.md files | `--dry-run`, `--force` |
| `sprint-status.py` | Quick sprint status | `--sprint` |
| `artifact-hygiene.py` | Clean up sprint artifacts | `--dry-run`, `--use-ai` |

---

## Execution Modes

### Mode 1: Pure Python (No AI)
Fast data collection and validation. No Claude Code involved.

```bash
python scripts/daily-standup.py --sprint "Sprint 6"
```

**Uses:**
- Query adapter for sprint items
- Calculate metrics
- Generate reports

### Mode 2: AI + JSON Schema
AI analysis with structured output. Python controls flow.

```bash
python scripts/product-intake.py
```

**Uses:**
- AI categorizes/estimates items
- Output validated against JSON schema
- Work items created programmatically

### Mode 3: Interactive AI
User collaborates with AI for complex planning.

```bash
python scripts/sprint-planning.py --sprint-number 8 --capacity 80 --interactive
```

**Uses:**
- Sprint planning with user input
- Complex test plan generation
- Architecture decisions

---

## Tree-Based Agent Spawning (backlog-grooming.py)

The `--use-agents` flag enables tree-based agent spawning for content generation, which solves performance and context overflow issues.

### Problem Solved
- **Single session architecture**: Original approach used one AI session for ALL content generation
- **Context accumulation**: Risk of token overflow with many Features
- **Sequential processing**: Each Feature processed one-at-a-time, blocking on each response

### Solution: Tree-Based Architecture
```
backlog_grooming.py (Orchestrator)
    │
    ├── Epic Breakdown: Spawn EpicBreakdownAgent per Epic
    │   └── Returns: epic_updates + features_to_create
    │
    └── Feature Conformance: Spawn FeatureConformanceAgent in parallel (batches of 3)
        └── Returns: feature_updates + implementation_task + test_plans
```

### Key Benefits
| Aspect | Legacy (default) | With `--use-agents` |
|--------|------------------|---------------------|
| Context isolation | Shared session, accumulates | Fresh context per agent |
| Parallelism | None | 3 Features in parallel |
| Failure recovery | All-or-nothing | Per-Feature (skip & continue) |
| Model | Configurable | claude-opus-4-5-20251101 |

### Usage
```bash
# Enable tree-based agent spawning
python scripts/backlog_grooming.py --use-agents

# Combine with other flags
python scripts/backlog_grooming.py --use-agents --epic-id 1304 -v
```

---

## Execution Pattern

All scripts follow this pattern:

```python
from scripts.workflow_executor.base import WorkflowOrchestrator

class MyWorkflow(WorkflowOrchestrator):
    def _define_steps(self):
        return [
            {"id": "1", "name": "Query Data"},
            {"id": "2", "name": "Analyze", "mode": 2},  # AI step
            {"id": "3", "name": "Create Work Items"},
        ]

    def _step_1_query_data(self):
        items = self.adapter.query_work_items(...)
        return {"items": items}

    def _step_2_analyze(self):
        # AI analysis with JSON schema validation
        return self._invoke_ai_with_schema(...)

    def _step_3_create_work_items(self):
        for item in self.state["analyzed_items"]:
            created = self.adapter.create_work_item(...)
            # VERIFY: Query adapter to confirm creation
            verified = self.adapter.get_work_item(created["id"])
            if not verified:
                raise VerificationError(...)
```

---

## Key Classes

### WorkflowOrchestrator (base.py)
Base class for all external enforcement workflows.

- `_define_steps()`: Define workflow steps
- `_execute_step()`: Execute individual step
- `_checkpoint()`: Save state to disk
- `_resume()`: Resume from checkpoint

### State Persistence
State saved to `.claude/workflow-state/{workflow}-{id}.json`:

```json
{
  "workflow": "sprint-planning",
  "current_step": 3,
  "completed_steps": [1, 2],
  "state": { "items": [...], "analysis": {...} }
}
```

---

## Verification Pattern

**CRITICAL: All work item operations must be verified against external source of truth.**

```python
# Create work item
result = self.adapter.create_work_item(
    title="Implement auth",
    work_item_type="Task"
)

# VERIFY: Query adapter to confirm it exists
verified = self.adapter.get_work_item(result["id"])
if not verified:
    raise VerificationError(f"Created work item {result['id']} but verification failed")

# Log verification
self.audit_log.append({
    "action": "create_work_item",
    "id": result["id"],
    "verified": True
})
```

---

## Approval Gates

Mode 3 scripts use real `input()` for approval gates:

```python
def _approval_gate(self, items_to_create):
    print(f"About to create {len(items_to_create)} work items:")
    for item in items_to_create:
        print(f"  - {item['title']}")

    # BLOCKS until user types "yes"
    response = input("Proceed? (yes/no): ")
    if response.lower() != "yes":
        raise UserRejection("User rejected work item creation")
```

AI cannot bypass this - execution is controlled by Python.

---

## Quick Start

```bash
# Set up credentials
export AZURE_DEVOPS_PAT="your-52-character-token"

# Run product intake (triage new ideas)
python scripts/product-intake.py

# Run sprint planning
python scripts/sprint-planning.py --sprint-number 8 --capacity 80

# Run sprint execution
python scripts/sprint-execution.py --sprint "Sprint 8"

# Run sprint review
python scripts/sprint-review.py --sprint "Sprint 7"

# Generate daily standup
python scripts/daily-standup.py --sprint "Sprint 7"
```

---

## Constraints for AI Agents

1. **Use adapter, not CLI**: Never `az boards` - always use adapter methods
2. **Verify all operations**: Query adapter after create/update to confirm
3. **Respect approval gates**: Cannot bypass real `input()` blocking
4. **Follow step order**: Steps execute sequentially, no skipping
5. **Preserve state**: Save checkpoint after each step completion
6. **Log to audit**: All decisions logged to `.claude/audit/`

---

## Related Contexts

- **VISION.md**: Pillars #1-3 for external enforcement principles
- **docs/EXTERNAL_ENFORCEMENT.md**: Complete usage documentation
- **workflows/CLAUDE.md**: Workflow utilities (analyze_sprint, verify_work_item_states)
- **adapters/CLAUDE.md**: Work tracking adapter interface
- **core/CLAUDE.md**: State management patterns
