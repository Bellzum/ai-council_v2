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

from council.models import (
    CouncilConfig,
    CouncilAgent,
    CouncilResult,
    CouncilRound,
    AgentFeedback,
    ApprovalStatus,
    UploadedDocument,
    ConversationExchange,
    AgentMemory,
)
from council.orchestrator import CouncilOrchestrator

__all__ = [
    "CouncilConfig",
    "CouncilAgent",
    "CouncilResult",
    "CouncilRound",
    "AgentFeedback",
    "ApprovalStatus",
    "CouncilOrchestrator",
    "UploadedDocument",
    "ConversationExchange",
    "AgentMemory",
]
