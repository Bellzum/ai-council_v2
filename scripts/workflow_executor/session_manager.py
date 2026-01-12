"""
Session Manager for Claude Agent SDK

Provides persistent session storage and recovery for workflow executions.
Sessions can be:
- Saved after each agent call for crash recovery
- Resumed to continue previous conversations
- Forked to create alternative branches
- Archived after workflow completion

Session data is stored in .claude/workflow-state/sessions/ as JSON files.

Usage:
    # In a workflow script
    session_mgr = SessionManager("sprint-execution", "sprint-10-exec-001")

    # Save session after agent call
    session_mgr.save_session(
        agent_type="engineer",
        session_id="abc-123",
        metadata={"work_item_id": 1234}
    )

    # Resume later
    session_id = session_mgr.get_session("engineer")
    if session_id:
        result = await wrapper.query(prompt, continue_session=True)

See: /home/sundance/.claude/plans/claude-agent-sdk-migration.md
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class SessionRecord:
    """Record of a saved session."""

    session_id: str
    agent_type: str
    timestamp: str
    workflow_name: str
    workflow_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_usage: Dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    status: str = "active"  # active, completed, failed, archived

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionRecord":
        """Create from dictionary, ignoring extra fields."""
        # Filter to only known fields
        known_fields = {
            "session_id", "agent_type", "timestamp", "workflow_name",
            "workflow_id", "metadata", "token_usage", "cost_usd", "status"
        }
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)


class SessionManager:
    """
    Manages session persistence for workflow agent calls.

    Features:
    - Save session IDs for crash recovery
    - Track session history across workflow steps
    - Support session forking for parallel exploration
    - Archive completed sessions

    Storage structure:
        .claude/workflow-state/sessions/
        ├── {workflow_id}-sessions.json      # Current sessions
        ├── {workflow_id}-history.json       # Full history
        └── archived/
            └── {workflow_id}-{timestamp}.json
    """

    SESSIONS_DIR = Path(".claude/workflow-state/sessions")
    ARCHIVE_DIR = SESSIONS_DIR / "archived"

    def __init__(self, workflow_name: str, workflow_id: str):
        """
        Initialize session manager.

        Args:
            workflow_name: Name of the workflow (e.g., "sprint-execution")
            workflow_id: Unique ID for this workflow run
        """
        self.workflow_name = workflow_name
        self.workflow_id = workflow_id

        # Ensure directories exist
        self.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        self.ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

        # File paths
        self.sessions_file = self.SESSIONS_DIR / f"{workflow_id}-sessions.json"
        self.history_file = self.SESSIONS_DIR / f"{workflow_id}-history.json"

    def save_session(
        self,
        agent_type: str,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        token_usage: Optional[Dict[str, int]] = None,
        cost_usd: float = 0.0,
    ) -> SessionRecord:
        """
        Save a session for potential resume.

        Args:
            agent_type: Type of agent (e.g., "engineer", "tester")
            session_id: Claude session ID
            metadata: Optional metadata (work_item_id, etc.)
            token_usage: Token usage for this session
            cost_usd: Cost of this session

        Returns:
            SessionRecord that was saved
        """
        record = SessionRecord(
            session_id=session_id,
            agent_type=agent_type,
            timestamp=datetime.now().isoformat(),
            workflow_name=self.workflow_name,
            workflow_id=self.workflow_id,
            metadata=metadata or {},
            token_usage=token_usage or {},
            cost_usd=cost_usd,
            status="active",
        )

        # Update current sessions
        sessions = self._load_sessions()
        sessions[agent_type] = record.to_dict()
        self._save_sessions(sessions)

        # Append to history
        self._append_history(record)

        return record

    def get_session(self, agent_type: str) -> Optional[str]:
        """
        Get saved session ID for an agent type.

        Args:
            agent_type: Type of agent to get session for

        Returns:
            Session ID if found, None otherwise
        """
        sessions = self._load_sessions()
        if agent_type in sessions:
            return sessions[agent_type].get("session_id")
        return None

    def get_session_record(self, agent_type: str) -> Optional[SessionRecord]:
        """
        Get full session record for an agent type.

        Args:
            agent_type: Type of agent to get session for

        Returns:
            SessionRecord if found, None otherwise
        """
        sessions = self._load_sessions()
        if agent_type in sessions:
            return SessionRecord.from_dict(sessions[agent_type])
        return None

    def get_all_sessions(self) -> Dict[str, SessionRecord]:
        """Get all current sessions."""
        sessions = self._load_sessions()
        return {
            agent_type: SessionRecord.from_dict(data)
            for agent_type, data in sessions.items()
        }

    def get_latest_session(self) -> Optional[str]:
        """
        Get the most recent session ID (any agent type).

        Useful for continuing a workflow after crash.

        Returns:
            Most recent session ID, or None
        """
        sessions = self._load_sessions()
        if not sessions:
            return None

        # Find most recent by timestamp
        latest = None
        latest_time = None

        for agent_type, data in sessions.items():
            timestamp = data.get("timestamp")
            if timestamp:
                if latest_time is None or timestamp > latest_time:
                    latest_time = timestamp
                    latest = data.get("session_id")

        return latest

    def mark_completed(self, agent_type: str) -> bool:
        """
        Mark a session as completed.

        Args:
            agent_type: Agent type whose session is complete

        Returns:
            True if session was found and marked
        """
        sessions = self._load_sessions()
        if agent_type in sessions:
            sessions[agent_type]["status"] = "completed"
            sessions[agent_type]["completed_at"] = datetime.now().isoformat()
            self._save_sessions(sessions)
            return True
        return False

    def mark_failed(self, agent_type: str, error: str) -> bool:
        """
        Mark a session as failed.

        Args:
            agent_type: Agent type whose session failed
            error: Error message

        Returns:
            True if session was found and marked
        """
        sessions = self._load_sessions()
        if agent_type in sessions:
            sessions[agent_type]["status"] = "failed"
            sessions[agent_type]["error"] = error
            sessions[agent_type]["failed_at"] = datetime.now().isoformat()
            self._save_sessions(sessions)
            return True
        return False

    def clear_session(self, agent_type: str) -> bool:
        """
        Clear a session (e.g., to start fresh).

        Args:
            agent_type: Agent type to clear

        Returns:
            True if session was found and cleared
        """
        sessions = self._load_sessions()
        if agent_type in sessions:
            del sessions[agent_type]
            self._save_sessions(sessions)
            return True
        return False

    def archive(self) -> Path:
        """
        Archive all sessions for this workflow run.

        Moves session data to archived/ directory with timestamp.

        Returns:
            Path to archive file
        """
        if not self.sessions_file.exists():
            return None

        # Create archive filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        archive_path = self.ARCHIVE_DIR / f"{self.workflow_id}-{timestamp}.json"

        # Load and archive sessions
        sessions = self._load_sessions()
        history = self._load_history()

        archive_data = {
            "workflow_name": self.workflow_name,
            "workflow_id": self.workflow_id,
            "archived_at": datetime.now().isoformat(),
            "sessions": sessions,
            "history": history,
        }

        archive_path.write_text(json.dumps(archive_data, indent=2), encoding="utf-8")

        # Clean up current files
        if self.sessions_file.exists():
            self.sessions_file.unlink()
        if self.history_file.exists():
            self.history_file.unlink()

        return archive_path

    def get_history(self) -> List[SessionRecord]:
        """Get full session history."""
        history = self._load_history()
        return [SessionRecord.from_dict(h) for h in history]

    def get_cumulative_usage(self) -> Dict[str, Any]:
        """
        Get cumulative token usage across all sessions.

        Returns:
            Dictionary with total tokens and cost
        """
        history = self._load_history()

        totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "cost_usd": 0.0,
            "session_count": len(history),
        }

        for record in history:
            usage = record.get("token_usage", {})
            totals["input_tokens"] += usage.get("input_tokens", 0)
            totals["output_tokens"] += usage.get("output_tokens", 0)
            totals["cache_read_tokens"] += usage.get("cache_read_tokens", 0)
            totals["cache_creation_tokens"] += usage.get("cache_creation_tokens", 0)
            totals["cost_usd"] += record.get("cost_usd", 0.0)

        return totals

    def _load_sessions(self) -> Dict[str, Any]:
        """Load current sessions from file."""
        if self.sessions_file.exists():
            try:
                return json.loads(self.sessions_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_sessions(self, sessions: Dict[str, Any]) -> None:
        """Save sessions to file."""
        self.sessions_file.write_text(
            json.dumps(sessions, indent=2), encoding="utf-8"
        )

    def _load_history(self) -> List[Dict[str, Any]]:
        """Load session history from file."""
        if self.history_file.exists():
            try:
                return json.loads(self.history_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return []
        return []

    def _append_history(self, record: SessionRecord) -> None:
        """Append a record to history."""
        history = self._load_history()
        history.append(record.to_dict())
        self.history_file.write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )


def list_active_workflows() -> List[Dict[str, Any]]:
    """
    List all workflows with active sessions.

    Returns:
        List of workflow summaries with session counts
    """
    sessions_dir = SessionManager.SESSIONS_DIR
    if not sessions_dir.exists():
        return []

    workflows = []
    for sessions_file in sessions_dir.glob("*-sessions.json"):
        try:
            data = json.loads(sessions_file.read_text(encoding="utf-8"))
            workflow_id = sessions_file.stem.replace("-sessions", "")

            # Count active sessions
            active_count = sum(
                1 for s in data.values() if s.get("status") == "active"
            )

            # Get latest timestamp
            latest = max(
                (s.get("timestamp", "") for s in data.values()),
                default=""
            )

            workflows.append({
                "workflow_id": workflow_id,
                "session_count": len(data),
                "active_count": active_count,
                "latest_timestamp": latest,
            })
        except (json.JSONDecodeError, KeyError):
            continue

    return sorted(workflows, key=lambda w: w.get("latest_timestamp", ""), reverse=True)


def cleanup_old_sessions(max_age_days: int = 7) -> int:
    """
    Clean up archived sessions older than max_age_days.

    Args:
        max_age_days: Maximum age in days before cleanup

    Returns:
        Number of files removed
    """
    archive_dir = SessionManager.ARCHIVE_DIR
    if not archive_dir.exists():
        return 0

    from datetime import timedelta

    cutoff = datetime.now() - timedelta(days=max_age_days)
    removed = 0

    for archive_file in archive_dir.glob("*.json"):
        try:
            data = json.loads(archive_file.read_text(encoding="utf-8"))
            archived_at = data.get("archived_at", "")
            if archived_at:
                archive_time = datetime.fromisoformat(archived_at)
                if archive_time < cutoff:
                    archive_file.unlink()
                    removed += 1
        except (json.JSONDecodeError, ValueError):
            continue

    return removed
