from pydantic_ai.messages import ModelMessage
from typing import Dict, List
import uuid


class InMemorySessionStore:
    """Simple in-memory session store for conversation history."""

    def __init__(self, max_messages_per_session: int = 20):
        self._sessions: Dict[str, List[ModelMessage]] = {}
        self._max_messages = max_messages_per_session

    def create_session(self) -> str:
        """Create a new conversation session."""
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = []
        return session_id

    def get_history(self, session_id: str) -> List[ModelMessage]:
        """Get conversation history for a session."""
        return self._sessions.get(session_id, [])

    def add_messages(self, session_id: str, messages: List[ModelMessage]) -> None:
        """Add messages to session history."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        self._sessions[session_id].extend(messages)

        # Trim to max messages (keep most recent, sliding window strategy)
        if len(self._sessions[session_id]) > self._max_messages:
            self._sessions[session_id] = self._sessions[session_id][
                -self._max_messages :
            ]

    def clear_session(self, session_id: str) -> None:
        """Clear a session's history."""
        if session_id in self._sessions:
            del self._sessions[session_id]

    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())
