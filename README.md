# AI Council - Implementation Guide

A generalized multi-agent consensus orchestration system for collaborative document synthesis.

## Overview

AI Council enables users to configure custom AI agents that collaborate on a shared text document through iterative refinement. The system implements a **k-phase commit consensus pattern** where a leader agent creates/revises content while evaluator agents provide structured feedback.

**Use Cases:**
- Design document synthesis
- Policy drafting with multiple perspectives
- Creative writing with editorial feedback
- Decision-making with diverse viewpoints
- **Drug screening** (compound evaluation with multi-expert consensus)
- Any collaborative content creation

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Streamlit Web App                            │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │  Page 1:   │  │  Page 2:   │  │  Page 3:   │  │  Page 4:   │    │
│  │  Configure │──▶│  Council   │──▶│  Live     │──▶│  Final    │    │
│  │  Agents    │  │  Prompt    │  │  Consensus │  │  Synthesis │    │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘    │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      CouncilOrchestrator                            │
│                                                                      │
│  run_council() ────▶ Leader creates ────▶ Evaluators critique       │
│                      (serial)              (parallel via gather)     │
│                              │                                       │
│                              ▼                                       │
│                      Leader revises ────▶ Repeat k rounds            │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Anthropic API                                   │
│                      (claude-sonnet-4 or claude-opus-4-5)           │
└─────────────────────────────────────────────────────────────────────┘
```

## Consensus Flow

```
Round 1:
  1. Leader receives council prompt + initial context
  2. Leader creates initial document (markdown)
  3. All evaluators receive document IN PARALLEL
  4. Each evaluator returns: {approved, concerns[], suggestions[], reasoning}
  5. If ALL approve → DONE

Round 2+ (if needed):
  1. Leader receives: current document + aggregated feedback
  2. Leader REVISES document addressing each concern
  3. Evaluators re-evaluate IN PARALLEL
  4. If ALL approve → DONE; else loop until max_rounds
```

## File Structure

```
council/
├── __init__.py           # Package exports
├── models.py             # Data classes
├── prompts.py            # Prompt templates
├── orchestrator.py       # CouncilOrchestrator
└── app.py                # Streamlit application

requirements.txt          # Dependencies
```

---

## Data Models (`models.py`)

```python
"""
Data models for AI Council consensus system.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class ApprovalStatus(Enum):
    """Evaluator approval status."""
    PENDING = "pending"
    APPROVED = "approved"
    CONCERNS = "concerns"


@dataclass
class CouncilAgent:
    """
    Configuration for a council agent.

    Attributes:
        name: Display name (e.g., "Technical Reviewer")
        role_description: What perspective this agent brings
        starting_prompt: System prompt defining the agent's persona
        is_leader: Whether this is the leader agent (first agent)
    """
    name: str
    role_description: str
    starting_prompt: str
    is_leader: bool = False


@dataclass
class AgentFeedback:
    """
    Feedback from a single evaluator agent.

    Attributes:
        agent_name: Name of the evaluator
        approved: Whether the evaluator approves the document
        concerns: List of specific concerns raised
        suggestions: List of improvement suggestions
        reasoning: Agent's reasoning for the evaluation
        tokens_used: Token count for this evaluation
    """
    agent_name: str
    approved: bool
    concerns: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    reasoning: str = ""
    tokens_used: int = 0


@dataclass
class CouncilRound:
    """
    Results from one round of council deliberation.

    Attributes:
        round_number: Which round (1-indexed)
        document_version: The markdown document at this stage
        leader_reasoning: Leader's explanation of changes (rounds 2+)
        feedback: List of evaluator feedback
        all_approved: Whether all evaluators approved
        duration_seconds: How long this round took
    """
    round_number: int
    document_version: str
    leader_reasoning: str = ""
    feedback: List[AgentFeedback] = field(default_factory=list)
    all_approved: bool = False
    duration_seconds: float = 0.0


@dataclass
class CouncilConfig:
    """
    Configuration for a council session.

    Attributes:
        agents: List of council agents (first is leader)
        council_prompt: The task/question for the council
        max_rounds: Maximum deliberation rounds (1-5)
        initial_context: Optional starting document/context
    """
    agents: List[CouncilAgent]
    council_prompt: str
    max_rounds: int = 3
    initial_context: str = ""

    @property
    def leader(self) -> CouncilAgent:
        """Get the leader agent (first agent)."""
        return self.agents[0]

    @property
    def evaluators(self) -> List[CouncilAgent]:
        """Get evaluator agents (all except first)."""
        return self.agents[1:]


@dataclass
class CouncilResult:
    """
    Final result of council deliberation.

    Attributes:
        final_document: The synthesized document
        rounds: History of all rounds
        consensus_reached: Whether consensus was achieved
        total_tokens: Total tokens used across all agents
        total_duration_seconds: Total time elapsed
    """
    final_document: str
    rounds: List[CouncilRound]
    consensus_reached: bool
    total_tokens: int = 0
    total_duration_seconds: float = 0.0
```

---

## Prompt Templates (`prompts.py`)

```python
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
```

---

## Council Orchestrator (`orchestrator.py`)

```python
"""
Council Orchestrator - Multi-agent consensus on text documents.

Implements a k-phase commit pattern:
1. Leader creates initial document
2. Evaluators critique in parallel
3. Leader revises based on feedback
4. Repeat until consensus or max rounds
"""

import asyncio
import json
import re
from datetime import datetime
from typing import Callable, Optional, List

import anthropic

from .models import (
    CouncilConfig, CouncilResult, CouncilRound,
    AgentFeedback, CouncilAgent
)
from .prompts import (
    build_leader_create_prompt,
    build_leader_revise_prompt,
    build_evaluator_prompt,
    aggregate_feedback
)


class CouncilOrchestrator:
    """
    Orchestrates multi-agent consensus on text documents.

    Example:
        config = CouncilConfig(
            agents=[
                CouncilAgent("Author", "Creates drafts", "You are a technical writer..."),
                CouncilAgent("Critic", "Reviews quality", "You critically evaluate...")
            ],
            council_prompt="Write a design document for authentication",
            max_rounds=3
        )

        orchestrator = CouncilOrchestrator(config, api_key="sk-ant-...")
        result = await orchestrator.run_council()
    """

    def __init__(
        self,
        config: CouncilConfig,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        on_round_start: Optional[Callable[[int], None]] = None,
        on_leader_response: Optional[Callable[[str], None]] = None,
        on_evaluator_response: Optional[Callable[[str, AgentFeedback], None]] = None,
        on_round_complete: Optional[Callable[[CouncilRound], None]] = None,
        verbose: bool = True
    ):
        """
        Initialize council orchestrator.

        Args:
            config: Council configuration with agents and prompt
            api_key: Anthropic API key
            model: Model to use (default: claude-sonnet-4-20250514)
            on_round_start: Callback when round begins
            on_leader_response: Callback when leader produces document
            on_evaluator_response: Callback when evaluator provides feedback
            on_round_complete: Callback when round finishes
            verbose: Print progress to stdout
        """
        self.config = config
        self.api_key = api_key
        self.model = model
        self.verbose = verbose

        # Callbacks for UI integration
        self._on_round_start = on_round_start
        self._on_leader_response = on_leader_response
        self._on_evaluator_response = on_evaluator_response
        self._on_round_complete = on_round_complete

        # Metrics
        self.total_tokens = 0

        # Initialize Anthropic client
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    def _log(self, message: str):
        """Log message if verbose mode enabled."""
        if self.verbose:
            print(message)

    async def run_council(self) -> CouncilResult:
        """
        Execute the council deliberation process.

        Returns:
            CouncilResult with final document and round history
        """
        start_time = datetime.now()
        rounds: List[CouncilRound] = []
        current_document = self.config.initial_context

        self._log(f"\n{'='*60}")
        self._log(f"Starting AI Council")
        self._log(f"Leader: {self.config.leader.name}")
        self._log(f"Evaluators: {', '.join(e.name for e in self.config.evaluators)}")
        self._log(f"Max Rounds: {self.config.max_rounds}")
        self._log(f"{'='*60}\n")

        for round_num in range(1, self.config.max_rounds + 1):
            round_start = datetime.now()

            if self._on_round_start:
                self._on_round_start(round_num)

            self._log(f"\n--- Round {round_num}/{self.config.max_rounds} ---\n")

            # Phase 1: Leader creates/revises
            if round_num == 1:
                self._log(f"[{self.config.leader.name}] Creating initial document...")
                current_document = await self._leader_create()
                leader_reasoning = ""
            else:
                self._log(f"[{self.config.leader.name}] Revising based on feedback...")
                feedback_text = aggregate_feedback(rounds[-1].feedback)
                current_document, leader_reasoning = await self._leader_revise(
                    current_document, feedback_text
                )

            if self._on_leader_response:
                self._on_leader_response(current_document)

            # Phase 2: Evaluators critique in parallel
            self._log(f"\nEvaluating with {len(self.config.evaluators)} agents in parallel...")
            feedback_list = await self._evaluate_parallel(current_document)

            # Notify about each evaluation
            for fb in feedback_list:
                if self._on_evaluator_response:
                    self._on_evaluator_response(fb.agent_name, fb)

            # Check consensus
            all_approved = all(fb.approved for fb in feedback_list)

            round_duration = (datetime.now() - round_start).total_seconds()

            council_round = CouncilRound(
                round_number=round_num,
                document_version=current_document,
                leader_reasoning=leader_reasoning,
                feedback=feedback_list,
                all_approved=all_approved,
                duration_seconds=round_duration
            )
            rounds.append(council_round)

            if self._on_round_complete:
                self._on_round_complete(council_round)

            # Log evaluation results
            for fb in feedback_list:
                status = "APPROVED" if fb.approved else "CONCERNS"
                self._log(f"  [{fb.agent_name}] {status}")
                if fb.concerns:
                    for concern in fb.concerns[:2]:
                        self._log(f"    - {concern[:80]}...")

            if all_approved:
                self._log(f"\nConsensus reached in round {round_num}!")
                break
            elif round_num < self.config.max_rounds:
                self._log(f"\nProceeding to round {round_num + 1} with feedback...")

        total_duration = (datetime.now() - start_time).total_seconds()

        return CouncilResult(
            final_document=current_document,
            rounds=rounds,
            consensus_reached=rounds[-1].all_approved if rounds else False,
            total_tokens=self.total_tokens,
            total_duration_seconds=total_duration
        )

    async def _leader_create(self) -> str:
        """Have leader create initial document."""
        prompt = build_leader_create_prompt(
            council_prompt=self.config.council_prompt,
            initial_context=self.config.initial_context,
            evaluators=self.config.evaluators
        )

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=self.config.leader.starting_prompt,
            messages=[{"role": "user", "content": prompt}]
        )

        self.total_tokens += response.usage.input_tokens + response.usage.output_tokens
        return response.content[0].text

    async def _leader_revise(
        self,
        current_document: str,
        aggregated_feedback: str
    ) -> tuple:
        """
        Have leader revise document based on feedback.

        Returns:
            Tuple of (revised_document, revision_reasoning)
        """
        prompt = build_leader_revise_prompt(
            council_prompt=self.config.council_prompt,
            current_document=current_document,
            aggregated_feedback=aggregated_feedback
        )

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=self.config.leader.starting_prompt,
            messages=[{"role": "user", "content": prompt}]
        )

        self.total_tokens += response.usage.input_tokens + response.usage.output_tokens

        text = response.content[0].text

        # Parse reasoning and document
        reasoning = ""
        document = text

        if "REASONING:" in text and "DOCUMENT:" in text:
            parts = text.split("DOCUMENT:", 1)
            reasoning_part = parts[0]
            document = parts[1].strip()

            if "REASONING:" in reasoning_part:
                reasoning = reasoning_part.split("REASONING:", 1)[1].strip()

        return document, reasoning

    async def _evaluate_parallel(self, document: str) -> List[AgentFeedback]:
        """Run all evaluators in parallel."""
        tasks = [
            self._single_evaluation(evaluator, document)
            for evaluator in self.config.evaluators
        ]
        return await asyncio.gather(*tasks)

    async def _single_evaluation(
        self,
        evaluator: CouncilAgent,
        document: str
    ) -> AgentFeedback:
        """Get feedback from a single evaluator."""
        prompt = build_evaluator_prompt(
            document=document,
            council_prompt=self.config.council_prompt,
            evaluator=evaluator
        )

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=evaluator.starting_prompt,
            messages=[{"role": "user", "content": prompt}]
        )

        tokens = response.usage.input_tokens + response.usage.output_tokens
        self.total_tokens += tokens

        text = response.content[0].text

        # Parse JSON response
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(text)

            return AgentFeedback(
                agent_name=evaluator.name,
                approved=data.get("approved", False),
                concerns=data.get("concerns", []),
                suggestions=data.get("suggestions", []),
                reasoning=data.get("reasoning", ""),
                tokens_used=tokens
            )
        except json.JSONDecodeError:
            # If JSON parsing fails, treat as concerns
            return AgentFeedback(
                agent_name=evaluator.name,
                approved=False,
                concerns=["Failed to parse structured response"],
                suggestions=[],
                reasoning=text[:500],
                tokens_used=tokens
            )
```

---

## Streamlit Application (`app.py`)

```python
"""
AI Council - Streamlit Web Application

Multi-agent consensus orchestration with live visualization.
"""

import asyncio
import streamlit as st

from .models import CouncilConfig, CouncilAgent
from .orchestrator import CouncilOrchestrator


def main():
    st.set_page_config(
        page_title="AI Council",
        page_icon="🏛️",
        layout="wide"
    )

    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = 1
    if "agents" not in st.session_state:
        st.session_state.agents = []
    if "council_result" not in st.session_state:
        st.session_state.council_result = None

    # Sidebar
    st.sidebar.title("🏛️ AI Council")
    st.sidebar.markdown("Multi-agent consensus orchestration")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Step {st.session_state.page} of 4**")

    # Page routing
    if st.session_state.page == 1:
        page_configure_agents()
    elif st.session_state.page == 2:
        page_council_prompt()
    elif st.session_state.page == 3:
        page_live_consensus()
    elif st.session_state.page == 4:
        page_final_synthesis()


def page_configure_agents():
    """Page 1: Configure council agents."""
    st.header("Step 1: Configure Your Council")
    st.markdown(
        "Define up to 5 agents. The **first agent is the Leader** who creates "
        "and revises the document. The rest are **Evaluators** who provide feedback."
    )

    # API Key input
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=st.session_state.get("api_key", ""),
        help="Your Claude API key (starts with sk-ant-)"
    )
    st.session_state.api_key = api_key

    st.markdown("---")

    # Number of agents selector
    num_agents = st.slider(
        "Number of Agents",
        min_value=2,
        max_value=5,
        value=st.session_state.get("num_agents", 3),
        help="First agent is Leader, rest are Evaluators"
    )
    st.session_state.num_agents = num_agents

    agents = []

    # Default agent configurations
    defaults = [
        ("Author", "Creates and refines the document",
         "You are a skilled writer who creates clear, comprehensive documents. Focus on clarity, structure, and completeness."),
        ("Critic", "Evaluates quality and identifies gaps",
         "You are a critical reviewer who identifies weaknesses, gaps, and areas for improvement. Be thorough but constructive."),
        ("Editor", "Reviews style and coherence",
         "You are an editor focused on style, flow, and readability. Ensure the document is well-organized and engaging."),
        ("Expert", "Provides domain expertise",
         "You are a domain expert who evaluates technical accuracy and completeness. Ensure claims are well-supported."),
        ("Advocate", "Represents the audience perspective",
         "You represent the target audience. Evaluate whether the document meets their needs and is accessible to them."),
    ]

    for i in range(num_agents):
        is_leader = (i == 0)
        role_type = "👑 Leader" if is_leader else f"📋 Evaluator {i}"
        default_name, default_role, default_prompt = defaults[i]

        with st.expander(f"Agent {i+1}: {role_type}", expanded=(i < 2)):
            col1, col2 = st.columns([1, 2])

            with col1:
                name = st.text_input(
                    "Name",
                    value=st.session_state.get(f"agent_{i}_name", default_name),
                    key=f"name_{i}"
                )
                st.session_state[f"agent_{i}_name"] = name

            with col2:
                role_desc = st.text_input(
                    "Role Description",
                    value=st.session_state.get(f"agent_{i}_role", default_role),
                    key=f"role_{i}"
                )
                st.session_state[f"agent_{i}_role"] = role_desc

            starting_prompt = st.text_area(
                "System Prompt",
                value=st.session_state.get(f"agent_{i}_prompt", default_prompt),
                height=100,
                key=f"prompt_{i}"
            )
            st.session_state[f"agent_{i}_prompt"] = starting_prompt

            agents.append(CouncilAgent(
                name=name,
                role_description=role_desc,
                starting_prompt=starting_prompt,
                is_leader=is_leader
            ))

    st.session_state.agents = agents

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Next →", type="primary", use_container_width=True):
            if not api_key or not api_key.startswith("sk-"):
                st.error("Please enter a valid Anthropic API key")
            elif len(agents) < 2:
                st.error("Need at least 2 agents")
            else:
                st.session_state.page = 2
                st.rerun()


def page_council_prompt():
    """Page 2: Enter council prompt and configure rounds."""
    st.header("Step 2: Define the Council Task")

    # Council prompt
    council_prompt = st.text_area(
        "Council Prompt",
        value=st.session_state.get("council_prompt", ""),
        height=200,
        placeholder="What should the council deliberate on?\n\nExample: Write a mission statement for a sustainable coffee company that emphasizes quality, environmental responsibility, and community impact.",
        help="This is the main task or question for the council"
    )
    st.session_state.council_prompt = council_prompt

    # Initial context (optional)
    with st.expander("Initial Context (Optional)"):
        initial_context = st.text_area(
            "Starting Document or Context",
            value=st.session_state.get("initial_context", ""),
            height=150,
            help="Provide any starting material the council should build upon"
        )
        st.session_state.initial_context = initial_context

    st.markdown("---")

    # Number of rounds
    max_rounds = st.slider(
        "Maximum Rounds",
        min_value=1,
        max_value=5,
        value=st.session_state.get("max_rounds", 2),
        help="Council will stop early if consensus is reached"
    )
    st.session_state.max_rounds = max_rounds

    # Agent summary
    st.markdown("---")
    st.subheader("Council Members")
    cols = st.columns(len(st.session_state.agents))
    for i, agent in enumerate(st.session_state.agents):
        with cols[i]:
            role = "👑 Leader" if agent.is_leader else "📋 Evaluator"
            st.markdown(f"**{agent.name}**")
            st.caption(f"{role}")
            st.caption(agent.role_description)

    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = 1
            st.rerun()
    with col3:
        if st.button("Start Council →", type="primary", use_container_width=True):
            if not council_prompt.strip():
                st.error("Please enter a council prompt")
            else:
                st.session_state.page = 3
                st.rerun()


def page_live_consensus():
    """Page 3: Watch consensus unfold with live updates."""
    st.header("Step 3: Council in Session")

    # Build config
    config = CouncilConfig(
        agents=st.session_state.agents,
        council_prompt=st.session_state.council_prompt,
        max_rounds=st.session_state.max_rounds,
        initial_context=st.session_state.get("initial_context", "")
    )

    # Status display
    status_container = st.container()

    # Create columns for document and feedback
    doc_col, feedback_col = st.columns([2, 1])

    with doc_col:
        st.subheader("Current Document")
        document_placeholder = st.empty()

    with feedback_col:
        st.subheader("Evaluator Feedback")
        feedback_placeholder = st.empty()

    # Progress tracking
    current_doc = ""
    feedback_list = []

    def on_round_start(round_num):
        with status_container:
            st.info(f"🔄 Round {round_num}/{config.max_rounds} in progress...")

    def on_leader_response(document):
        nonlocal current_doc
        current_doc = document
        with document_placeholder.container():
            st.markdown(document)

    def on_evaluator_response(agent_name, feedback):
        feedback_list.append(feedback)
        with feedback_placeholder.container():
            for fb in feedback_list:
                icon = "✅" if fb.approved else "⚠️"
                st.markdown(f"{icon} **{fb.agent_name}**")
                if fb.concerns:
                    for c in fb.concerns[:2]:
                        st.caption(f"- {c[:100]}...")
                st.markdown("---")

    def on_round_complete(council_round):
        feedback_list.clear()
        if council_round.all_approved:
            with status_container:
                st.success(f"✅ Consensus reached in round {council_round.round_number}!")

    # Run council
    orchestrator = CouncilOrchestrator(
        config=config,
        api_key=st.session_state.api_key,
        on_round_start=on_round_start,
        on_leader_response=on_leader_response,
        on_evaluator_response=on_evaluator_response,
        on_round_complete=on_round_complete,
        verbose=False
    )

    # Execute asynchronously
    try:
        result = asyncio.run(orchestrator.run_council())
        st.session_state.council_result = result

        # Final status
        with status_container:
            if result.consensus_reached:
                st.success(f"✅ Consensus reached in {len(result.rounds)} round(s)!")
            else:
                st.warning("⚠️ Maximum rounds reached without full consensus")

        # Navigation
        st.markdown("---")
        if st.button("View Final Synthesis →", type="primary"):
            st.session_state.page = 4
            st.rerun()

    except Exception as e:
        st.error(f"Error during council execution: {str(e)}")
        if st.button("← Go Back"):
            st.session_state.page = 2
            st.rerun()


def page_final_synthesis():
    """Page 4: Display final synthesized document."""
    st.header("Step 4: Final Synthesis")

    result = st.session_state.council_result

    if not result:
        st.error("No council result available")
        if st.button("Start Over"):
            st.session_state.page = 1
            st.rerun()
        return

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rounds", len(result.rounds))
    col2.metric("Consensus", "✅ Yes" if result.consensus_reached else "⚠️ No")
    col3.metric("Total Tokens", f"{result.total_tokens:,}")
    col4.metric("Duration", f"{result.total_duration_seconds:.1f}s")

    st.markdown("---")

    # Final document
    st.subheader("Final Document")
    st.markdown(result.final_document)

    # Download button
    st.download_button(
        "📥 Download Document",
        result.final_document,
        file_name="council_synthesis.md",
        mime="text/markdown"
    )

    st.markdown("---")

    # Round history
    st.subheader("Deliberation History")
    for council_round in result.rounds:
        with st.expander(
            f"Round {council_round.round_number} - "
            f"{'✅ Consensus' if council_round.all_approved else '🔄 Revised'}"
        ):
            if council_round.leader_reasoning:
                st.markdown("**Leader's Revision Notes:**")
                st.markdown(council_round.leader_reasoning)
                st.markdown("---")

            st.markdown("**Document Version:**")
            st.markdown(council_round.document_version)

            st.markdown("---")
            st.markdown("**Evaluator Feedback:**")
            for fb in council_round.feedback:
                icon = "✅" if fb.approved else "⚠️"
                st.markdown(f"{icon} **{fb.agent_name}**")
                if fb.reasoning:
                    st.caption(fb.reasoning)
                if fb.concerns:
                    st.markdown("Concerns:")
                    for concern in fb.concerns:
                        st.markdown(f"- {concern}")
                if fb.suggestions:
                    st.markdown("Suggestions:")
                    for suggestion in fb.suggestions:
                        st.markdown(f"- {suggestion}")
                st.markdown("---")

    # Start over
    st.markdown("---")
    if st.button("🔄 Start New Council", type="primary"):
        # Clear result but keep agent config
        st.session_state.council_result = None
        st.session_state.council_prompt = ""
        st.session_state.page = 2
        st.rerun()


if __name__ == "__main__":
    main()
```

---

## Dependencies (`requirements.txt`)

```
anthropic>=0.30.0
streamlit>=1.28.0
```

---

## Running the Application

```bash
# Install dependencies
pip install anthropic streamlit

# Run the Streamlit app
streamlit run council/app.py

# Or with specific port
streamlit run council/app.py --server.port 8501
```

---

## Package Init (`__init__.py`)

```python
"""
AI Council - Multi-agent consensus orchestration system.

Usage:
    from council import CouncilOrchestrator, CouncilConfig, CouncilAgent

    config = CouncilConfig(
        agents=[
            CouncilAgent("Author", "Creates drafts", "You are a writer..."),
            CouncilAgent("Critic", "Reviews quality", "You critically evaluate...")
        ],
        council_prompt="Write a design document",
        max_rounds=3
    )

    orchestrator = CouncilOrchestrator(config, api_key="sk-ant-...")
    result = await orchestrator.run_council()
"""

from .models import (
    CouncilConfig,
    CouncilAgent,
    CouncilResult,
    CouncilRound,
    AgentFeedback,
    ApprovalStatus,
)
from .orchestrator import CouncilOrchestrator

__all__ = [
    "CouncilConfig",
    "CouncilAgent",
    "CouncilResult",
    "CouncilRound",
    "AgentFeedback",
    "ApprovalStatus",
    "CouncilOrchestrator",
]
```

---

## Example Use Case: Drug Screening Council

A practical example using AI Council for pharmaceutical compound evaluation.

### Scenario

Evaluate a drug candidate for potential development, synthesizing perspectives from scientific experts AND business stakeholders to produce a comprehensive assessment covering both therapeutic viability and commercial feasibility.

### Agent Configuration

```python
from council import CouncilOrchestrator, CouncilConfig, CouncilAgent

config = CouncilConfig(
    agents=[
        CouncilAgent(
            name="Portfolio Lead",
            role_description="Synthesizes scientific and business analysis into investment recommendations",
            starting_prompt="""You are a senior pharmaceutical portfolio leader responsible for
synthesizing scientific and commercial analyses into drug candidate investment decisions.

Your role:
1. Create structured evaluation reports covering efficacy, safety, AND business viability
2. Integrate feedback from both scientific and commercial reviewers
3. Provide clear go/no-go recommendations with ROI projections
4. Balance scientific promise against commercial realities
5. Identify key risks (scientific, regulatory, IP, market) and mitigation strategies

Write in clear language suitable for executive review boards and investors.
Include both qualitative assessments and quantitative projections where possible.""",
            is_leader=True
        ),
        CouncilAgent(
            name="Pharmacologist",
            role_description="Evaluates mechanism of action, efficacy, and pharmacokinetics",
            starting_prompt="""You are an expert pharmacologist reviewing drug candidates.

Evaluate:
- Mechanism of action and target specificity
- Predicted efficacy based on molecular structure and preclinical data
- Pharmacokinetic properties (ADME: absorption, distribution, metabolism, excretion)
- Drug-drug interaction potential
- Dose-response considerations and therapeutic window

Flag any concerns about efficacy limitations or off-target effects."""
        ),
        CouncilAgent(
            name="Toxicologist",
            role_description="Assesses safety profile and toxicity risks",
            starting_prompt="""You are a toxicology expert evaluating drug safety.

Assess:
- Acute and chronic toxicity potential
- Genotoxicity and carcinogenicity signals
- Organ-specific toxicity (hepatotoxicity, cardiotoxicity, nephrotoxicity)
- Reproductive and developmental toxicity
- Safety margins and therapeutic index
- Black box warning likelihood

Be conservative - patient safety is paramount. Flag any red flags immediately.
Consider how safety profile impacts labeling and market positioning."""
        ),
        CouncilAgent(
            name="IP & Patent Strategist",
            role_description="Evaluates patentability, freedom to operate, and IP strategy",
            starting_prompt="""You are a pharmaceutical IP strategist evaluating patent position.

Assess:
- Composition of matter patentability (novelty, non-obviousness)
- Freedom to operate analysis (blocking patents, ANDA/biosimilar risks)
- Patent term and exclusivity projections (NCE, orphan, pediatric extensions)
- Trade secret vs. patent strategy for manufacturing processes
- Competitive patent landscape and white space opportunities
- Licensing opportunities or encumbrance risks
- Geographic filing strategy (US, EU, Japan, China)

Estimate remaining exclusivity window and IP strength score.
Flag any freedom-to-operate concerns or patent cliff risks."""
        ),
        CouncilAgent(
            name="Commercial Strategist",
            role_description="Evaluates market opportunity, pricing, and commercial viability",
            starting_prompt="""You are a pharmaceutical commercial strategist evaluating business potential.

Assess:
- Total addressable market (TAM) and realistic market share projections
- Competitive landscape (existing therapies, pipeline competitors)
- Pricing and reimbursement feasibility (payer attitudes, ICER thresholds)
- Peak sales projections and time to peak
- Development costs and probability-adjusted NPV
- Partnership/licensing attractiveness
- Market access barriers (prior authorization, step therapy)
- Geographic market prioritization

Provide realistic revenue projections with bull/base/bear scenarios.
Flag any commercial dealbreakers or market timing concerns."""
        ),
    ],
    council_prompt="""Evaluate the following drug candidate for potential development investment:

**Compound:** XYZ-4829
**Target:** GPR55 (G protein-coupled receptor 55)
**Indication:** Neuropathic pain (diabetic peripheral neuropathy)
**Stage:** Lead optimization complete, pre-IND

## Scientific Data

**Efficacy:**
- IC50: 12 nM (target), >10 µM (off-target panel of 50 receptors)
- Efficacy in rat STZ model: 65% pain reduction vs. vehicle (p<0.001)
- Efficacy in mouse SNL model: 58% pain reduction vs. vehicle (p<0.001)
- Head-to-head vs. pregabalin: Non-inferior efficacy, faster onset

**Pharmacokinetics:**
- Oral bioavailability: 45% (rat), 38% (dog)
- Half-life: 6.2 hours (rat), 8.1 hours (dog)
- Protein binding: 92%
- CNS penetration: Brain/plasma ratio 0.3

**Safety:**
- hERG IC50: 8.2 µM (>100x safety margin)
- Ames test: Negative
- 14-day rat tox: NOAEL 100 mg/kg (30x projected human dose)
- 14-day dog tox: NOAEL 50 mg/kg (15x projected human dose)
- CYP inhibition: Weak 3A4 inhibitor (IC50 15 µM)
- No CNS adverse effects observed in behavioral battery

## Business Context

**Intellectual Property:**
- Composition of matter patent filed: March 2024
- Key prior art: Expired GPR55 agonist patents (different chemotype)
- Manufacturing process: Novel synthesis, trade secret candidate
- Freedom to operate search: Preliminary clear, full FTO pending

**Market Information:**
- Diabetic neuropathy market: $4.2B (2024), growing 6% CAGR
- Current SOC: Pregabalin, duloxetine, gabapentin (all generic)
- Unmet need: 40% of patients inadequately controlled on current therapy
- Pipeline competitors: 3 Phase 2 assets (different MOAs)
- Estimated development cost to approval: $280M
- Projected launch: 2031 (if fast-tracked)

**Strategic Considerations:**
- Company has no CNS/pain franchise currently
- Partnership interest from 2 top-20 pharma companies
- Orphan designation possible for rare pain subtypes

## Requested Output

Produce a comprehensive investment assessment with:
1. **Executive summary** with go/no-go recommendation and confidence level
2. **Scientific assessment** (efficacy potential, safety risks, differentiation)
3. **IP and exclusivity analysis** (patent strength, FTO, exclusivity runway)
4. **Commercial viability** (market size, pricing, peak sales projections)
5. **Risk register** (scientific, regulatory, IP, commercial risks with mitigations)
6. **Financial projections** (NPV, IRR, probability-adjusted returns)
7. **Recommended next steps** with decision gates and kill criteria""",
    max_rounds=3,
    initial_context=""
)

# Run the council
orchestrator = CouncilOrchestrator(config, api_key="sk-ant-...")
result = await orchestrator.run_council()
```

### Expected Output

The council produces a synthesized investment assessment that:
- Integrates scientific (pharmacology, toxicology) AND business (IP, commercial) perspectives
- Provides probability-adjusted financial projections (NPV, IRR, peak sales)
- Assesses patent strength and exclusivity runway
- Highlights consensus areas and flags unresolved concerns across disciplines
- Includes a structured risk register with mitigation strategies
- Documents the deliberation process for board presentations and due diligence

### Why Multi-Agent Consensus Works Here

| Challenge | How Council Solves It |
|-----------|----------------------|
| Science-business disconnect | Forces integration of both perspectives in single document |
| Single-expert blind spots | Pharmacologist, toxicologist, IP, and commercial each catch different issues |
| Confirmation bias | Business strategist challenges scientific optimism; scientists ground commercial hype |
| IP risks overlooked | Dedicated patent strategist evaluates FTO, exclusivity, and competitive landscape |
| Incomplete analysis | Iterative refinement ensures all dimensions covered before investment decision |
| Audit trail needs | Full deliberation history preserved for board review and due diligence |
| Reproducibility | Same agents + prompt = consistent evaluation framework across portfolio |

---

## Extension Ideas

1. **Model Selection**: Allow users to choose between Claude models (Sonnet, Opus, Haiku)
2. **Export Formats**: Add PDF, DOCX export options
3. **Streaming**: Stream leader responses for better UX
4. **Presets**: Save/load agent configurations (e.g., "Drug Screening Panel")
5. **History**: Persist past councils for reference
6. **Themes**: Custom Streamlit themes
7. **API Mode**: REST API for programmatic access
8. **Voting**: Alternative consensus models (majority, weighted)
9. **Batch Processing**: Evaluate multiple compounds in sequence
10. **Integration**: Connect to molecular databases (PubChem, ChEMBL) for context
