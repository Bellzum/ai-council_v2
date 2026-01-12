"""
Prompt templates for AI Council agents.
"""

LEADER_CREATE_PROMPT = """## Council Task

{council_prompt}

## Initial Context

{initial_context}

## Instructions

Create a comprehensive document that addresses the council task.
Write in markdown format with clear structure.

The document will be reviewed by {num_evaluators} evaluator(s):
{evaluator_list}

Be thorough and anticipate potential concerns from each perspective.

## Output

Provide ONLY the document content in markdown format. Do not include any preamble or explanation.
"""

LEADER_REVISE_PROMPT = """## Council Task

{council_prompt}

## Current Document

{current_document}

## Evaluator Feedback

The following concerns were raised by specialist agents. You MUST address each one:

{aggregated_feedback}

## Instructions

Revise the document to address ALL evaluator concerns.
For each concern:
1. Consider whether the feedback is valid
2. Make appropriate changes to the document
3. Ensure the revision maintains overall quality and coherence

## Output Format

First, briefly explain your revision approach (2-3 sentences).
Then provide the COMPLETE revised document in markdown format.

Start your response with "REASONING:" followed by your explanation.
Then "DOCUMENT:" followed by the full revised document.
"""

EVALUATOR_PROMPT = """## Document to Review

{document}

## Council Task

{council_prompt}

## Your Role

You are "{evaluator_name}" - {evaluator_role_description}

## Your Perspective

{evaluator_starting_prompt}

## Instructions

Evaluate this document from your unique perspective.

Consider:
1. Does it address the council task completely?
2. Are there any gaps, inaccuracies, or areas for improvement?
3. Is the quality sufficient for the stated purpose?
4. What specific improvements would you suggest?

## Output Format

Respond with ONLY valid JSON (no markdown code blocks):

{{
    "approved": true or false,
    "concerns": ["list of specific concerns if not approved, or empty if approved"],
    "suggestions": ["list of specific improvements, even if approved"],
    "reasoning": "brief explanation of your evaluation (2-3 sentences)"
}}

Only set "approved" to true if the document truly meets quality standards from your perspective.
Be specific and constructive in concerns and suggestions.
"""


def build_leader_create_prompt(
    council_prompt: str,
    initial_context: str,
    evaluators: list
) -> str:
    """Build the prompt for leader's initial document creation."""
    evaluator_list = "\n".join(
        f"- **{e.name}**: {e.role_description}"
        for e in evaluators
    )
    return LEADER_CREATE_PROMPT.format(
        council_prompt=council_prompt,
        initial_context=initial_context or "(No initial context provided)",
        num_evaluators=len(evaluators),
        evaluator_list=evaluator_list
    )


def build_leader_revise_prompt(
    council_prompt: str,
    current_document: str,
    aggregated_feedback: str
) -> str:
    """Build the prompt for leader's document revision."""
    return LEADER_REVISE_PROMPT.format(
        council_prompt=council_prompt,
        current_document=current_document,
        aggregated_feedback=aggregated_feedback
    )


def build_evaluator_prompt(
    document: str,
    council_prompt: str,
    evaluator: "CouncilAgent"
) -> str:
    """Build the prompt for an evaluator's review."""
    return EVALUATOR_PROMPT.format(
        document=document,
        council_prompt=council_prompt,
        evaluator_name=evaluator.name,
        evaluator_role_description=evaluator.role_description,
        evaluator_starting_prompt=evaluator.starting_prompt
    )


def aggregate_feedback(feedback_list: list) -> str:
    """Aggregate evaluator feedback for leader revision."""
    sections = []
    for fb in feedback_list:
        status = "APPROVED" if fb.approved else "CONCERNS RAISED"
        section = f"### {fb.agent_name} ({status})\n\n"

        if fb.concerns:
            section += "**Concerns:**\n"
            for concern in fb.concerns:
                section += f"- {concern}\n"
            section += "\n"

        if fb.suggestions:
            section += "**Suggestions:**\n"
            for suggestion in fb.suggestions:
                section += f"- {suggestion}\n"
            section += "\n"

        if fb.reasoning:
            section += f"**Reasoning:** {fb.reasoning}\n"

        sections.append(section)

    return "\n---\n\n".join(sections)
