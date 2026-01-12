"""
Agent Prompt Templates for Tree-Based Backlog Grooming

This module contains prompt templates and JSON schemas for the tree-based
agent spawning architecture in backlog_grooming.py.

Architecture:
- EpicBreakdownAgent: Analyzes Epic, generates content, breaks down into Features
- FeatureConformanceAgent: Conforms Feature, creates Implementation Task + test plans
"""

# =============================================================================
# JSON Schemas for Response Validation
# =============================================================================

EPIC_BREAKDOWN_SCHEMA = {
    "type": "object",
    "required": ["epic_updates", "features_to_create"],
    "properties": {
        "epic_updates": {
            "type": "object",
            "properties": {
                "business_analysis": {"type": "string"},
                "success_criteria": {"type": "string"},
                "scope_definition": {"type": "string"},
                "story_points": {"type": "integer", "minimum": 50}
            }
        },
        "features_to_create": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["title", "description", "story_points"],
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "story_points": {"type": "integer", "minimum": 10, "maximum": 30},
                    "architecture_analysis": {"type": "string"},
                    "security_analysis": {"type": "string"},
                    "acceptance_criteria": {"type": "string"}
                }
            }
        }
    }
}

FEATURE_CONFORMANCE_SCHEMA = {
    "type": "object",
    "required": ["feature_updates"],
    "properties": {
        "feature_updates": {
            "type": "object",
            "properties": {
                "detailed_description": {"type": "string"},
                "architecture_analysis": {"type": "string"},
                "security_analysis": {"type": "string"},
                "acceptance_criteria": {"type": "string"},
                "story_points": {"type": "integer", "minimum": 10, "maximum": 30}
            }
        },
        "implementation_task": {
            "type": "object",
            "required": ["title", "detailed_design"],
            "properties": {
                "title": {"type": "string"},
                "detailed_design": {"type": "string"},
                "function_specifications": {"type": "string"},
                "story_points": {"type": "integer", "minimum": 1, "maximum": 30},
                "test_plans": {
                    "type": "object",
                    "properties": {
                        "unit_test_plan": {"type": "string"},
                        "integration_test_plan": {"type": "string"},
                        "edge_case_test_plan": {"type": "string"},
                        "acceptance_test_plan": {"type": "string"}
                    }
                },
                "falsifiability_requirements": {"type": "string"},
                "evidence_requirements": {"type": "string"}
            }
        }
    }
}


# =============================================================================
# Prompt Templates
# =============================================================================

EPIC_BREAKDOWN_PROMPT = """You are a Business Analyst breaking down an Epic into Features for software development.

## Epic Information
- **ID**: #{epic_id}
- **Title**: {epic_title}
- **Current Description**:
{epic_description}

## Your Task

Generate the following:

### 1. Missing Epic Content
{missing_content_instructions}

### 2. Feature Breakdown
Break this Epic into cohesive, independently deliverable Features:
- **Minimum total**: 50 story points (approximately 250 hours of work)
- **Per Feature**: 10-30 story points each
- **Each Feature must be**:
  - Cohesive (single responsibility)
  - Measurable (clear completion criteria)
  - Testable (can be validated)
  - Independently deliverable

## Output Format

Return ONLY valid JSON matching this structure:
```json
{{
  "epic_updates": {{
    "business_analysis": "Business value, ROI, stakeholder impact...",
    "success_criteria": "Measurable outcomes...",
    "scope_definition": "What's in/out of scope...",
    "story_points": 50
  }},
  "features_to_create": [
    {{
      "title": "Feature title",
      "description": "Detailed description of the Feature scope",
      "story_points": 13,
      "architecture_analysis": "Technical approach and integration points",
      "security_analysis": "Security considerations and mitigations",
      "acceptance_criteria": "Given/When/Then criteria"
    }}
  ]
}}
```

## Guidelines
- Keep descriptions concise but actionable (150-300 words each)
- Use bullet points for clarity
- Ensure Features sum to >= 50 story points
- Each Feature should be completable in 1-2 sprints
"""

FEATURE_CONFORMANCE_PROMPT = """You are a Technical Lead conforming a Feature for sprint planning and creating its Implementation Task.

## Feature Information
- **ID**: #{feature_id}
- **Title**: {feature_title}
- **Current Description**:
{feature_description}

## Parent Epic Context
{parent_epic_context}

## Your Task

### 1. Conform Feature (if needed)
{missing_content_instructions}

### 2. Create Implementation Task
Generate a complete Implementation Task with:
- Detailed design and approach
- Function specifications for each component
- Comprehensive test plans (unit, integration, edge-case, acceptance)
- Falsifiability requirements (how tests prove implementation works)
- Evidence requirements (artifacts proving completion)

## Output Format

Return ONLY valid JSON matching this structure:
```json
{{
  "feature_updates": {{
    "detailed_description": "Full scope and requirements...",
    "architecture_analysis": "Technical design, integration points, patterns...",
    "security_analysis": "Security risks and mitigations...",
    "acceptance_criteria": "Given/When/Then criteria...",
    "story_points": 13
  }},
  "implementation_task": {{
    "title": "Implement: {feature_title}",
    "detailed_design": "Step-by-step implementation approach...",
    "function_specifications": "Detailed specs for each function/component...",
    "story_points": 13,
    "test_plans": {{
      "unit_test_plan": "Unit tests with pytest markers, 80% coverage target...",
      "integration_test_plan": "Integration tests for component interactions...",
      "edge_case_test_plan": "Edge cases and error conditions...",
      "acceptance_test_plan": "Tests for each acceptance criterion..."
    }},
    "falsifiability_requirements": "How tests detect actual failures vs false positives...",
    "evidence_requirements": "Artifacts proving implementation is complete..."
  }}
}}
```

## Guidelines
- Keep each section to 150-300 words
- Use bullet points for clarity
- Test plans should include specific test function names with pytest markers
- Acceptance criteria must be testable (Given/When/Then format)
- Evidence requirements must be verifiable by external tools
"""


# =============================================================================
# Helper Functions
# =============================================================================

def build_epic_breakdown_prompt(
    epic_id: int,
    epic_title: str,
    epic_description: str,
    missing_content: list
) -> str:
    """Build the Epic breakdown prompt with context."""
    if missing_content:
        missing_instructions = f"Generate these missing sections: {', '.join(missing_content)}"
    else:
        missing_instructions = "All Epic content is present. Focus on Feature breakdown."

    # No truncation per workflow-flow.md spec:
    # "No result/artifact/message should be truncated unless the data is grossly inflated"
    if epic_description and len(epic_description) > 50000:
        import warnings
        warnings.warn(f"Epic description is very large ({len(epic_description)} chars)")

    return EPIC_BREAKDOWN_PROMPT.format(
        epic_id=epic_id,
        epic_title=epic_title,
        epic_description=epic_description if epic_description else "(No description)",
        missing_content_instructions=missing_instructions
    )


def build_feature_conformance_prompt(
    feature_id: int,
    feature_title: str,
    feature_description: str,
    parent_epic_context: str,
    missing_content: list,
    needs_implementation_task: bool = True
) -> str:
    """Build the Feature conformance prompt with context."""
    if missing_content:
        missing_instructions = f"Generate these missing sections: {', '.join(missing_content)}"
    else:
        missing_instructions = "All Feature content is present. Focus on Implementation Task."

    if not needs_implementation_task:
        missing_instructions += "\n\nNote: Implementation Task already exists. Only update Feature fields."

    # No truncation per workflow-flow.md spec:
    # "No result/artifact/message should be truncated unless the data is grossly inflated"
    if feature_description and len(feature_description) > 50000:
        import warnings
        warnings.warn(f"Feature description is very large ({len(feature_description)} chars)")
    if parent_epic_context and len(parent_epic_context) > 50000:
        import warnings
        warnings.warn(f"Parent Epic context is very large ({len(parent_epic_context)} chars)")

    return FEATURE_CONFORMANCE_PROMPT.format(
        feature_id=feature_id,
        feature_title=feature_title,
        feature_description=feature_description if feature_description else "(No description)",
        parent_epic_context=parent_epic_context if parent_epic_context else "(No parent Epic)",
        missing_content_instructions=missing_instructions
    )


def validate_epic_breakdown_response(response: dict) -> tuple:
    """
    Validate Epic breakdown response against schema.

    Returns:
        (is_valid: bool, errors: list)
    """
    errors = []

    if not isinstance(response, dict):
        return False, ["Response is not a dictionary"]

    if "epic_updates" not in response:
        errors.append("Missing 'epic_updates' field")

    if "features_to_create" not in response:
        errors.append("Missing 'features_to_create' field")
    elif not isinstance(response["features_to_create"], list):
        errors.append("'features_to_create' must be a list")
    elif len(response["features_to_create"]) == 0:
        errors.append("'features_to_create' cannot be empty")
    else:
        total_points = 0
        for i, feature in enumerate(response["features_to_create"]):
            if "title" not in feature:
                errors.append(f"Feature {i}: missing 'title'")
            if "description" not in feature:
                errors.append(f"Feature {i}: missing 'description'")
            if "story_points" not in feature:
                errors.append(f"Feature {i}: missing 'story_points'")
            else:
                points = feature.get("story_points", 0)
                if not (10 <= points <= 30):
                    errors.append(f"Feature {i}: story_points must be 10-30, got {points}")
                total_points += points

        if total_points < 50:
            errors.append(f"Total story points ({total_points}) must be >= 50")

    return len(errors) == 0, errors


def validate_feature_conformance_response(response: dict) -> tuple:
    """
    Validate Feature conformance response against schema.

    Returns:
        (is_valid: bool, errors: list)
    """
    errors = []

    if not isinstance(response, dict):
        return False, ["Response is not a dictionary"]

    if "feature_updates" not in response:
        errors.append("Missing 'feature_updates' field")

    # Implementation task is optional (Feature may already have one)
    if "implementation_task" in response:
        task = response["implementation_task"]
        if "title" not in task:
            errors.append("Implementation task: missing 'title'")
        if "detailed_design" not in task:
            errors.append("Implementation task: missing 'detailed_design'")

    return len(errors) == 0, errors
