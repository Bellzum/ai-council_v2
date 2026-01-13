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
class UploadedDocument:
    """
    Represents an uploaded document for an agent.

    Attributes:
        filename: Original filename
        content_type: MIME type or extension (pdf, docx, txt)
        extracted_text: Parsed text content
        summary: AI-generated summary (for token management)
        token_count: Estimated tokens in extracted_text
    """
    filename: str
    content_type: str
    extracted_text: str
    summary: str = ""
    token_count: int = 0


@dataclass
class ConversationExchange:
    """
    A single exchange in agent conversation history.

    Attributes:
        role: 'user' or 'assistant'
        content: Message content
        token_count: Estimated tokens
        round_number: Council round this occurred in
    """
    role: str
    content: str
    token_count: int = 0
    round_number: int = 0


@dataclass
class AgentMemory:
    """
    Session-scoped memory for a single agent.

    Attributes:
        agent_name: Name of the agent this memory belongs to
        documents: List of uploaded documents with summaries
        conversation_history: Recent exchanges (capped at max_exchanges)
        context_summary: Rolling summary of older context
        max_exchanges: Maximum recent exchanges to keep (default 3)
        max_context_tokens: Token budget for context (default 2000)
    """
    agent_name: str
    documents: List["UploadedDocument"] = field(default_factory=list)
    conversation_history: List["ConversationExchange"] = field(default_factory=list)
    context_summary: str = ""
    max_exchanges: int = 3
    max_context_tokens: int = 2000

    def get_context_for_prompt(self) -> str:
        """Build context string for API call, respecting token limits."""
        parts = []

        # Add document summaries
        if self.documents:
            doc_section = "## Reference Documents\n"
            for doc in self.documents:
                doc_section += f"\n### {doc.filename}\n{doc.summary}\n"
            parts.append(doc_section)

        # Add context summary (older conversation)
        if self.context_summary:
            parts.append(f"## Previous Context Summary\n{self.context_summary}")

        # Add recent conversation history
        if self.conversation_history:
            recent = "## Recent Exchanges\n"
            for exchange in self.conversation_history[-self.max_exchanges:]:
                role_label = "You" if exchange.role == "assistant" else "User"
                content_preview = exchange.content[:500]
                if len(exchange.content) > 500:
                    content_preview += "..."
                recent += f"\n**{role_label}:** {content_preview}\n"
            parts.append(recent)

        return "\n\n".join(parts)


@dataclass
class CouncilAgent:
    """
    Configuration for a council agent.

    Attributes:
        name: Display name (e.g., "Technical Reviewer")
        role_description: What perspective this agent brings
        starting_prompt: System prompt defining the agent's persona
        is_leader: Whether this is the leader agent (first agent)
        memory_key: Unique key for session_state memory lookup
    """
    name: str
    role_description: str
    starting_prompt: str
    is_leader: bool = False
    memory_key: str = ""

    def __post_init__(self):
        if not self.memory_key:
            self.memory_key = f"memory_{self.name.lower().replace(' ', '_')}"


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
