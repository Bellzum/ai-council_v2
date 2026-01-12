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
