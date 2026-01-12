"""
Pre-defined Consensus Configurations

This module provides three standard consensus configurations per workflow-flow.md:

1. EPIC_TO_FEATURE_CONFIG - Epic breakdown into Features (backlog-grooming)
2. FEATURE_TO_TASK_CONFIG - Feature breakdown into Tasks (backlog-grooming)
3. TASK_IMPLEMENTATION_CONFIG - Task/Bug implementation (sprint-execution)

Each configuration specifies:
- Leader agent (who creates/revises the proposal)
- Evaluator agents (who critique in parallel)
- Max rounds (k-phase iterations)
- Prompts for each role

Usage:
    from scripts.workflow_executor.consensus import ConsensusOrchestrator
    from scripts.workflow_executor.consensus_configs import EPIC_TO_FEATURE_CONFIG

    orchestrator = ConsensusOrchestrator(EPIC_TO_FEATURE_CONFIG, adapter)
    result = await orchestrator.run_consensus({"epic": epic_context})
"""

from scripts.workflow_executor.consensus import ConsensusConfig, AgentSettings


# =============================================================================
# Model Tiers - Cost-optimized model selection by agent role
# =============================================================================
# Pricing (per MTok): Opus 4.5: $5/$25, Sonnet 4: $3/$15, Haiku 4.5: $1/$5

MODEL_OPUS_4_5 = "claude-opus-4-5-20251101"      # Complex design, architecture
MODEL_SONNET_4 = "claude-sonnet-4-20250514"      # Implementation, code generation
MODEL_HAIKU_4_5 = "claude-haiku-4-5-20241022"    # Review, verification, simple tasks


# =============================================================================
# Agent Settings - Model, Thinking, and Context Configuration
# =============================================================================

# Backlog Grooming: senior-engineer (Opus), architect (Sonnet), others (Haiku)
BACKLOG_GROOMING_AGENT_SETTINGS = {
    "senior-engineer": AgentSettings(
        extended_thinking=True,
        thinking_budget=10000,
        max_context=True,  # Maximum context window for leader
        model_override=MODEL_OPUS_4_5,  # Complex design decisions
    ),
    "architect": AgentSettings(
        extended_thinking=True,
        thinking_budget=8000,
        model_override=MODEL_SONNET_4,  # Architecture review
    ),
    "security-specialist": AgentSettings(
        extended_thinking=True,
        thinking_budget=8000,
        model_override=MODEL_HAIKU_4_5,  # Security pattern matching
    ),
    "tester": AgentSettings(
        extended_thinking=True,
        thinking_budget=8000,
        model_override=MODEL_HAIKU_4_5,  # Test review
    ),
    "scrum-master": AgentSettings(
        extended_thinking=False,
        model_override=MODEL_HAIKU_4_5,  # Process coordination
    ),
}

# Sprint Execution: engineer (Sonnet), tester (Haiku)
SPRINT_EXECUTION_AGENT_SETTINGS = {
    "engineer": AgentSettings(
        extended_thinking=True,
        thinking_budget=10000,
        max_context=True,  # Maximum context window for implementation
        model_override=MODEL_SONNET_4,  # Code generation
    ),
    "tester": AgentSettings(
        extended_thinking=True,
        thinking_budget=8000,
        model_override=MODEL_HAIKU_4_5,  # Verification
    ),
}


# =============================================================================
# Configuration 1: Epic-to-Feature Breakdown (Backlog Grooming)
# =============================================================================

EPIC_TO_FEATURE_CONFIG = ConsensusConfig(
    leader_agent="senior-engineer",
    evaluator_agents=["architect", "security-specialist", "tester"],
    max_rounds=2,
    leader_proposal_prompt="""## Epic-to-Feature Breakdown Task

You are a Senior Engineer breaking down an Epic into Features for the development team.

### Your Deliverables

Generate a comprehensive Feature breakdown with:

1. **Feature List** - Each Feature should be:
   - Cohesive (single responsibility)
   - Independently deliverable
   - 10-30 story points each
   - Total >= 50 story points for the Epic

2. **Per Feature, Include:**
   - Title (clear, actionable)
   - Description (scope, requirements, constraints)
   - Story Points (estimate based on complexity)
   - Architecture Analysis (technical approach, components, integration points)
   - Security Analysis (risks, mitigations, data handling)
   - Acceptance Criteria (testable Given/When/Then format)

3. **Epic Updates** (if needed):
   - Business Analysis (value proposition, ROI)
   - Success Criteria (measurable outcomes)
   - Scope Definition (what's in/out)

### Output Format

Return JSON:
```json
{
  "epic_updates": {
    "business_analysis": "...",
    "success_criteria": "...",
    "scope_definition": "..."
  },
  "features_to_create": [
    {
      "title": "Feature: ...",
      "description": "...",
      "story_points": 13,
      "architecture_analysis": "...",
      "security_analysis": "...",
      "acceptance_criteria": "Given... When... Then..."
    }
  ]
}
```
""",
    evaluator_prompts={
        "architect": """## Architecture Review with Codebase Verification

You have tool access. Use Read, Grep, and Glob to verify claims against the actual codebase.

### Verification Steps
1. Use Glob to find existing architecture patterns (e.g., "**/*.py", "**/config*.yaml")
2. Use Read to examine existing modules referenced in the proposal
3. Use Grep to find existing implementations of similar features

### Review Criteria (verify against actual code)
- Architecture alignment: Does proposal match existing patterns you found?
- Technology consistency: Are proposed technologies already in use?
- Integration points: Do the modules/APIs referenced actually exist?
- Component boundaries: Are dependencies on existing code accurate?

### REJECT if:
- Proposal references modules/files that don't exist
- Proposed architecture conflicts with existing patterns
- Integration points are inaccurate

Provide specific file paths and code references in your evaluation.""",

        "security-specialist": """## Security Review with Codebase Verification

You have tool access. Use Read and Grep to find existing security patterns.

### Verification Steps
1. Use Grep to find existing auth/security code: "auth", "permission", "validate"
2. Use Read to examine existing input validation patterns
3. Use Grep to find how secrets are currently managed

### Review Criteria (verify against actual code)
- OWASP Top 10: Are proposed mitigations consistent with existing patterns?
- Auth/authz: Does proposal align with existing auth mechanisms?
- Input validation: Are proposed validation approaches consistent?
- Secrets: Does proposal follow existing secrets management?

### REJECT if:
- Proposal introduces patterns inconsistent with existing security
- Required security infrastructure doesn't exist
- Proposal would weaken existing security measures

Provide specific file paths and code references in your evaluation.""",

        "tester": """## Testability Review with Codebase Verification

You have tool access. Use Read and Glob to examine existing test patterns.

### Verification Steps
1. Use Glob to find existing tests: "tests/**/*.py", "**/test_*.py"
2. Use Read to examine test patterns and fixtures in use
3. Use Grep to find pytest markers and coverage patterns

### Review Criteria (verify against actual code)
- Testability: Are proposed test approaches consistent with existing tests?
- Acceptance criteria: Are they falsifiable (can definitively fail)?
- Test infrastructure: Does required test infrastructure exist?
- Coverage: Are proposed coverage targets achievable given existing patterns?

### REJECT if:
- Proposed test approach conflicts with existing test patterns
- Acceptance criteria are vague or unfalsifiable
- Required test fixtures/infrastructure don't exist

Provide specific file paths and code references in your evaluation."""
    },
    agent_settings=BACKLOG_GROOMING_AGENT_SETTINGS
)


# =============================================================================
# Configuration 2: Feature-to-Task Creation (Backlog Grooming)
# =============================================================================

FEATURE_TO_TASK_CONFIG = ConsensusConfig(
    leader_agent="senior-engineer",
    evaluator_agents=["architect", "security-specialist", "tester"],
    max_rounds=2,
    leader_proposal_prompt="""## Feature-to-Task Creation Task

You are a Senior Engineer creating detailed Tasks for a Feature.

### Your Deliverables

For this Feature, create comprehensive Tasks with:

1. **Task Breakdown**
   - Implementation Task(s) with detailed design
   - Each Task should be 1-5 story points
   - Clear dependencies between Tasks

2. **Per Task, Include:**
   - Title (specific, actionable)
   - Detailed Design (implementation approach, decisions, patterns)
   - Function Specifications (interfaces, parameters, return types)
   - Story Points (1-5 range)

3. **Test Plans** (attached as documents):
   - Unit Test Plan (80% coverage target, pytest markers)
   - Integration Test Plan (component interactions)
   - Edge Case Test Plan (boundary conditions, error cases)
   - Acceptance Test Plan (maps to Feature acceptance criteria)

4. **Quality Requirements:**
   - Falsifiability Requirements (how tests detect real failures)
   - Evidence Requirements (artifacts proving completion)

5. **Feature Updates** (if needed):
   - Refined architecture analysis
   - Updated acceptance criteria

### Output Format

Return JSON:
```json
{
  "feature_updates": {
    "detailed_description": "...",
    "architecture_analysis": "...",
    "security_analysis": "...",
    "acceptance_criteria": "..."
  },
  "tasks_to_create": [
    {
      "title": "Implement: ...",
      "detailed_design": "...",
      "function_specifications": "...",
      "story_points": 3,
      "test_plans": {
        "unit_test_plan": "...",
        "integration_test_plan": "...",
        "edge_case_test_plan": "...",
        "acceptance_test_plan": "..."
      },
      "falsifiability_requirements": "...",
      "evidence_requirements": "..."
    }
  ]
}
```
""",
    evaluator_prompts={
        "architect": """## Implementation Design Review with Codebase Verification

You have tool access. VERIFY the proposed design against actual codebase.

### Verification Steps (MANDATORY)
1. Use Glob/Grep to find modules referenced in the design
2. Use Read to examine existing error handling patterns
3. Verify API contracts against existing interfaces

### Review Criteria
- Implementation reliability: Does approach match existing patterns?
- Error handling: Is strategy consistent with codebase conventions?
- API contracts: Do referenced interfaces actually exist?
- Dependencies: Are all imported modules/functions real?

### REJECT if:
- Design references non-existent modules or functions
- Proposed patterns conflict with existing codebase
- Error handling strategy is inconsistent with conventions

Include file paths and line numbers in your evaluation.""",

        "security-specialist": """## Security Design Review with Codebase Verification

You have tool access. VERIFY security claims against actual code.

### Verification Steps (MANDATORY)
1. Use Grep to find existing validation patterns for similar inputs
2. Use Read to examine how similar data is currently handled
3. Verify auth patterns match existing implementation

### Review Criteria
- Input validation: Does design cover all attack vectors?
- Auth gaps: Are proposed auth checks consistent with existing code?
- Data handling: Does approach match existing sensitive data patterns?
- Error messages: No information leakage in proposed error handling?

### REJECT if:
- Security approach weaker than existing patterns
- Validation coverage has obvious gaps
- Auth design inconsistent with existing mechanisms

Include file paths and specific code references.""",

        "tester": """## Test Design Review with Codebase Verification

You have tool access. VERIFY test design is achievable.

### Verification Steps (MANDATORY)
1. Use Glob to find existing test files in the codebase
2. Use Read to examine existing test patterns, frameworks, and fixtures
3. Verify proposed test infrastructure exists

### Review Criteria
- Falsifiability: Can each test definitively fail?
- Edge cases: Are boundary conditions covered?
- Independence: No test interdependencies?
- Coverage: Is target coverage achievable for proposed scope?

### REJECT if:
- Test design relies on non-existent fixtures or test utilities
- Acceptance criteria cannot be tested (unfalsifiable)
- Proposed tests would be interdependent
- Coverage target is unrealistic for scope

Include references to existing test patterns."""
    },
    agent_settings=BACKLOG_GROOMING_AGENT_SETTINGS
)


# =============================================================================
# Configuration 3: Task/Bug Implementation (Sprint Execution)
# =============================================================================

TASK_IMPLEMENTATION_CONFIG = ConsensusConfig(
    leader_agent="engineer",
    evaluator_agents=["tester"],
    max_rounds=5,  # More rounds for implementation (code iteration)
    leader_proposal_prompt="""## Task/Bug Implementation

You are an Engineer implementing a Task or fixing a Bug.

**CRITICAL: You have full tool access. You MUST actually create/modify files using the Write and Edit tools.**

### Your Required Actions

1. **Read Existing Code First**
   - Use the Read tool to examine existing code patterns
   - Use Grep/Glob to find relevant files and identify the tech stack
   - Understand the codebase conventions before making changes

2. **Actually Write Production Code**
   - Use Write tool to create new files
   - Use Edit tool to modify existing files
   - Follow the existing language, framework, and style conventions
   - DO NOT just describe what you would do - actually do it

3. **Actually Write Test Code**
   - Create real test files using Write tool
   - Follow the existing test framework and conventions in the codebase
   - Tests must be executable, not hypothetical

4. **Run Tests to Verify**
   - Use Bash tool to run the project's test command (check package.json, Makefile, etc.)
   - Capture actual test output
   - Report real pass/fail counts, not estimates

### Output Format

After completing the actual implementation, return JSON with VERIFIED information:
```json
{
  "implementation": {
    "files_created": ["path/to/file"],
    "files_modified": ["path/to/existing"],
    "key_decisions": ["Decision 1 because...", "Decision 2 because..."],
    "deviations": [],
    "limitations": []
  },
  "tests": {
    "test_files_created": ["path/to/test_file"],
    "test_command": "the command used to run tests",
    "tests_run": 15,
    "tests_passed": 15,
    "test_output": "actual test runner output here"
  },
  "acceptance_criteria_status": {
    "criterion_1": "PASS - verified by test_x",
    "criterion_2": "PASS - verified by test_y"
  },
  "verification": {
    "files_exist": true,
    "tests_executed": true,
    "all_tests_pass": true
  }
}
```

### CRITICAL REQUIREMENTS

- Every file listed in files_created MUST actually exist (verifiable via Read tool)
- Every test listed MUST have been actually executed (test output required)
- Do NOT claim work that was not actually done
- Do NOT estimate or guess test results - run them
""",
    evaluator_prompts={
        "tester": """## CRITICAL: You Must VERIFY, Not Just Review

You have full tool access. You MUST use Read, Grep, Glob, and Bash tools to verify claims.

**DO NOT accept claims at face value. VERIFY EVERYTHING.**

### Verification Steps (MANDATORY)

1. **Verify Files Exist**
   - Use Read tool to open EACH file listed in files_created
   - Use Read tool to verify modifications in files_modified
   - If a file cannot be read, REJECT with concern: "File does not exist: {path}"

2. **Verify Code Quality**
   - Read the actual code, not just the file list
   - Check for stubs, TODOs, placeholder values, incomplete implementations
   - Check for proper error handling conventions used in this codebase

3. **Verify Tests Are Real and Executable**
   - Use Read tool to examine test files
   - Use Bash to run the test command specified by the engineer (or find it in package.json, Makefile, etc.)
   - Capture actual test output
   - If tests fail or don't exist, REJECT

4. **Verify Test Quality**
   - Check that tests actually test behavior, not just existence
   - Verify tests can fail (are falsifiable)
   - If coverage tools are available in the project, run them

5. **Verify Acceptance Criteria**
   - For each criterion marked PASS, verify the corresponding test exists
   - Run the specific test to confirm it passes
   - Cross-reference with parent Feature requirements

### Rejection Criteria (MUST REJECT if any are true)

- Cannot read a file that was claimed to be created
- Tests fail when actually executed
- Stubs, TODOs, or "not implemented" found in code
- Test file exists but contains no actual test functions
- Claims don't match what's actually in the files

### Response Format

```json
{
  "accepted": false,
  "verification_results": {
    "files_verified": ["list of files you successfully read"],
    "files_missing": ["list of claimed files that don't exist"],
    "tests_executed": true/false,
    "test_command_used": "the command you ran",
    "test_output": "actual test runner output",
    "stubs_found": ["list of stubs/TODOs found"]
  },
  "concerns": ["specific concerns with evidence"],
  "suggestions": ["specific fixes needed"]
}
```

**IMPORTANT: If you cannot verify a claim using tools, you MUST reject.**"""
    },
    agent_settings=SPRINT_EXECUTION_AGENT_SETTINGS
)


# =============================================================================
# Helper Functions
# =============================================================================

def get_config_for_transition(transition_type: str) -> ConsensusConfig:
    """
    Get the appropriate consensus config for a transition type.

    Args:
        transition_type: One of "epic_to_feature", "feature_to_task", "task_implementation"

    Returns:
        The corresponding ConsensusConfig
    """
    configs = {
        "epic_to_feature": EPIC_TO_FEATURE_CONFIG,
        "feature_to_task": FEATURE_TO_TASK_CONFIG,
        "task_implementation": TASK_IMPLEMENTATION_CONFIG
    }

    if transition_type not in configs:
        raise ValueError(
            f"Unknown transition type: {transition_type}. "
            f"Valid types: {', '.join(configs.keys())}"
        )

    return configs[transition_type]


def list_available_configs() -> dict:
    """
    List all available consensus configurations.

    Returns:
        Dict mapping config name to description
    """
    return {
        "epic_to_feature": "Epic breakdown into Features (backlog-grooming, 4 agents, k=2)",
        "feature_to_task": "Feature breakdown into Tasks (backlog-grooming, 4 agents, k=2)",
        "task_implementation": "Task/Bug implementation (sprint-execution, 2 agents, k=3)"
    }
