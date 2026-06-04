# Short-Term Memory in DLens Base Agent

This document describes the in-memory conversation state implementation for `DLensConversationalAgent`.

## Architecture

The base agent (`DLensBaseAgent`) is stateless — each `arun()` call starts fresh. For conversational agents, `DLensConversationalAgent` extends the base with session-based message history via `InMemorySessionStore`.

## Implementation

### InMemorySessionStore (`src/dlens/memory/session_store.py`)

A dictionary-based store with sliding window trimming:

- `create_session()` — returns a UUID session ID
- `get_history(session_id)` — returns `List[ModelMessage]`
- `add_messages(session_id, messages)` — appends and trims to max
- `clear_session(session_id)` — deletes session
- `list_sessions()` — lists active session IDs

### DLensConversationalAgent (`src/dlens/agents/_base.py`)

Extends `DLensBaseAgent` with:

- `create_session()` — creates a new conversation session
- `arun(query, session_id=...)` — runs with message history from the session store
- `clear_session(session_id)` — clears conversation history

When `session_id` is `None`, behaves like the stateless base agent.

## Usage

```python
from dlens.agents import DLensConversationalAgent, BaseAgentConfig
from dlens.agents._models import OllamaModel

config = BaseAgentConfig(
    name="Conversational Assistant",
    description="A helpful assistant that remembers conversation context",
    model=OllamaModel(model_name="qwen3:8b"),
    max_history_messages=50,
)

agent = DLensConversationalAgent(config=config)
session_id = agent.create_session()

response1 = await agent.arun("My name is Alice", session_id=session_id)
response2 = await agent.arun("What did I say my name was?", session_id=session_id)

agent.clear_session(session_id)
```

## Sliding Window Strategy

The store keeps only the most recent N messages per session (`max_messages_per_session`), preventing unbounded memory growth while maintaining recent context.

## References

- [Pydantic AI Messages Documentation](https://ai.pydantic.dev/api/messages/)
- [Pydantic AI Message History](https://ai.pydantic.dev/message-history/)
