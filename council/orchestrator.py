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
from typing import Callable, Optional, List, Dict

import anthropic

from council.models import (
    CouncilConfig, CouncilResult, CouncilRound,
    AgentFeedback, CouncilAgent, AgentMemory, ConversationExchange
)
from council.document_processor import count_tokens
from council.prompts import (
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
        agent_memories: Optional[Dict[str, AgentMemory]] = None,
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
            agent_memories: Dict mapping memory_key to AgentMemory for context injection
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
        self.agent_memories = agent_memories or {}

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

    def _get_memory_context(self, agent: CouncilAgent) -> str:
        """Get memory context string for an agent."""
        if agent.memory_key and agent.memory_key in self.agent_memories:
            memory = self.agent_memories[agent.memory_key]
            return memory.get_context_for_prompt()
        return ""

    def _record_exchange(
        self,
        agent: CouncilAgent,
        user_content: str,
        assistant_content: str,
        round_number: int
    ):
        """Record conversation exchange in agent memory."""
        if not agent.memory_key or agent.memory_key not in self.agent_memories:
            return

        memory = self.agent_memories[agent.memory_key]

        # Add user message (truncated for storage)
        user_truncated = user_content[:1000]
        memory.conversation_history.append(ConversationExchange(
            role="user",
            content=user_truncated,
            token_count=count_tokens(user_truncated),
            round_number=round_number
        ))

        # Add assistant response (truncated for storage)
        assistant_truncated = assistant_content[:1000]
        memory.conversation_history.append(ConversationExchange(
            role="assistant",
            content=assistant_truncated,
            token_count=count_tokens(assistant_truncated),
            round_number=round_number
        ))

        # Trim to max_exchanges * 2 (pairs)
        max_messages = memory.max_exchanges * 2
        if len(memory.conversation_history) > max_messages:
            memory.conversation_history = memory.conversation_history[-max_messages:]

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
                    current_document, feedback_text, round_num
                )

            if self._on_leader_response:
                self._on_leader_response(current_document)

            # Phase 2: Evaluators critique in parallel
            self._log(f"\nEvaluating with {len(self.config.evaluators)} agents in parallel...")
            feedback_list = await self._evaluate_parallel(current_document, round_num)

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
        leader = self.config.leader
        agent_context = self._get_memory_context(leader)

        prompt = build_leader_create_prompt(
            council_prompt=self.config.council_prompt,
            initial_context=self.config.initial_context,
            evaluators=self.config.evaluators,
            agent_context=agent_context
        )

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=leader.starting_prompt,
            messages=[{"role": "user", "content": prompt}]
        )

        self.total_tokens += response.usage.input_tokens + response.usage.output_tokens
        result = response.content[0].text

        # Record exchange in memory
        self._record_exchange(leader, prompt, result, 1)

        return result

    async def _leader_revise(
        self,
        current_document: str,
        aggregated_feedback: str,
        round_number: int = 2
    ) -> tuple:
        """
        Have leader revise document based on feedback.

        Returns:
            Tuple of (revised_document, revision_reasoning)
        """
        leader = self.config.leader
        agent_context = self._get_memory_context(leader)

        prompt = build_leader_revise_prompt(
            council_prompt=self.config.council_prompt,
            current_document=current_document,
            aggregated_feedback=aggregated_feedback,
            agent_context=agent_context
        )

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=leader.starting_prompt,
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

        # Record exchange in memory
        self._record_exchange(leader, prompt, text, round_number)

        return document, reasoning

    async def _evaluate_parallel(
        self,
        document: str,
        round_number: int
    ) -> List[AgentFeedback]:
        """Run all evaluators in parallel."""
        tasks = [
            self._single_evaluation(evaluator, document, round_number)
            for evaluator in self.config.evaluators
        ]
        return await asyncio.gather(*tasks)

    async def _single_evaluation(
        self,
        evaluator: CouncilAgent,
        document: str,
        round_number: int
    ) -> AgentFeedback:
        """Get feedback from a single evaluator."""
        agent_context = self._get_memory_context(evaluator)

        prompt = build_evaluator_prompt(
            document=document,
            council_prompt=self.config.council_prompt,
            evaluator=evaluator,
            agent_context=agent_context
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

        # Record exchange in memory
        self._record_exchange(evaluator, prompt, text, round_number)

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
