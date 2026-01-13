"""
Summarization utilities for managing context tokens.
"""

from typing import List
import anthropic

from council.document_processor import truncate_to_tokens


SUMMARIZE_DOCUMENT_PROMPT = """Summarize the following document in 2-3 paragraphs.
Focus on key facts, main arguments, and critical details that would be relevant
for someone reviewing or building upon this content.

Document:
{document_text}

Provide a clear, factual summary:"""


SUMMARIZE_CONVERSATION_PROMPT = """Summarize the following conversation exchanges into
a brief context summary (1-2 paragraphs). Focus on:
- Key decisions made
- Important feedback given
- Main points of discussion

Conversation:
{conversation_text}

Summary:"""


async def summarize_document(
    client: anthropic.AsyncAnthropic,
    document_text: str,
    model: str = "claude-sonnet-4-20250514",
    max_summary_tokens: int = 500
) -> str:
    """
    Generate a summary of a document for context injection.

    Args:
        client: Anthropic async client
        document_text: Full document text
        model: Model to use for summarization
        max_summary_tokens: Maximum tokens for summary

    Returns:
        Summary string
    """
    # Truncate input if too long (leave room for prompt)
    truncated = truncate_to_tokens(document_text, 6000)

    prompt = SUMMARIZE_DOCUMENT_PROMPT.format(document_text=truncated)

    response = await client.messages.create(
        model=model,
        max_tokens=max_summary_tokens,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text


async def summarize_conversation(
    client: anthropic.AsyncAnthropic,
    exchanges: List,
    model: str = "claude-sonnet-4-20250514",
    max_summary_tokens: int = 300
) -> str:
    """
    Summarize older conversation exchanges for rolling context.

    Args:
        client: Anthropic async client
        exchanges: List of ConversationExchange objects
        model: Model to use
        max_summary_tokens: Maximum tokens for summary

    Returns:
        Summary string
    """
    if not exchanges:
        return ""

    conversation_text = "\n".join(
        f"{e.role.upper()}: {e.content}" for e in exchanges
    )

    truncated = truncate_to_tokens(conversation_text, 4000)

    prompt = SUMMARIZE_CONVERSATION_PROMPT.format(conversation_text=truncated)

    response = await client.messages.create(
        model=model,
        max_tokens=max_summary_tokens,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text
