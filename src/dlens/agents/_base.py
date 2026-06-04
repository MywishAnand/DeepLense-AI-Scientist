from pydantic import BaseModel, Field
from pydantic_ai import Agent, AgentStreamEvent
from pydantic_ai.messages import (
    PartDeltaEvent,
    PartStartEvent,
    ThinkingPartDelta,
    TextPartDelta,
    ToolCallPartDelta,
    PartEndEvent,
    FinalResultEvent,
)
from typing import Optional, Any, Callable, AsyncIterator
import textwrap
from loguru import logger
from dlens.memory import InMemorySessionStore


class InputSchema(BaseModel):
    """Input schema for the agent."""


class OutputSchema(BaseModel):
    """Output schema for the agent."""

    reasoning: str = Field(description="The reasoning process of the agent.")


class BaseAgentConfig(BaseModel):
    name: str = Field(description="The name of the agent.")
    description: str = Field(description="The description of the agent.")
    custom_system_prompt: Optional[str] = None
    model: Any
    max_history_messages: int = 10
    debug: bool = False


# Use DLensBaseAgent for stateless agents (sub-agents in deterministic workflows, single-turn agents, or agent-as-tool).
# Use DLensConversationalAgent for conversational agents with session memory.
class DLensBaseAgent:
    def __init__(
        self,
        *,
        config: BaseAgentConfig,
        deps_type: Any | None = None,
        output_type: Any | None = str,
        register_tools: Callable | None = None,
        **agent_kwargs: Any,
    ):
        self.config = config

        system_prompt = self._build_system_prompt(
            custom_prompt=config.custom_system_prompt
        )
        if self.config.debug:
            logger.debug(f"System prompt: {system_prompt}")

        self._agent = Agent(
            model=config.model,
            deps_type=deps_type,
            output_type=output_type,
            system_prompt=system_prompt,
            **agent_kwargs,
        )

        if register_tools:
            register_tools(self._agent)

    def _build_system_prompt(self, custom_prompt: Optional[str] = None) -> str:
        """Build the system prompt for the agent."""
        base_prompt = textwrap.dedent(f"""
            You are {self.config.name}, a specialized agent in the DLens (DeepLense AI Scientist) framework.

            Your purpose: {self.config.description}

            You are part of a multi-agent system for autonomous scientific workflows.
            Your responses should be precise, structured, and follow the defined output schema.

            Key principles:
            - Be concise and focused on your specific task
            - Provide clear explanations for your decisions
            - Flag uncertainties or edge cases in warnings
            - Suggest logical next steps when appropriate
            - Use available tools when needed to accomplish your task

            Output ONLY valid JSON, with double quotes, no trailing commas, no markdown fences, no extra commentary.
        """).strip()

        if custom_prompt:
            custom_block = textwrap.dedent(f"""
                Additional instructions:
                {custom_prompt}
            """).strip()
            return f"{base_prompt}\n\n{custom_block}"

        return base_prompt

    async def arun(
        self,
        query: str | InputSchema,
        *,
        deps: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Run the agent on a user query.
        """
        if isinstance(query, InputSchema):
            query = query.model_dump_json()
        else:
            query = str(query)
        return await self._agent.run(query, deps=deps, **kwargs)

    async def _arun_stream_events(
        self,
        query: str,
        *,
        deps: Any | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AgentStreamEvent]:
        """
        Stream raw LLM events (thinking, text deltas, tool calls) as they occur (Buffered).
        Use this for custom streaming handling.

        # Refer to https://ai.pydantic.dev/api/messages/ and use this method for UI streaming. The arun_stream is only for logging and not for UI streaming.
        """
        async for event in self._agent.run_stream_events(query, deps=deps, **kwargs):
            yield event

    async def arun_stream(
        self,
        query: str | InputSchema,
        *,
        deps: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Stream agent execution with event logging and return final result.
        Stream Format (From Pydantic AI Docs):
        ModelResponseStreamEvent = [PartStartEvent | PartDeltaEvent | PartEndEvent | FinalResultEvent]
        """
        thinking_buffer = []
        text_buffer = []
        tool_buffer = []
        current_tool_name = None
        result = None

        logger.info(f"[QUERY] {query}\n")

        if isinstance(query, InputSchema):
            query = query.model_dump_json()
        else:
            query = str(query)

        # Use run_stream to get both streaming events and final result
        async for event in self._arun_stream_events(query, deps=deps, **kwargs):
            last_event = (
                event  # explicitly store the last event to avoid race condition
            )

            # Stream through all events
            if isinstance(event, PartStartEvent):
                logger.info(f"[STARTED] {event.part.__class__.__name__}\n")
                if hasattr(event.part, "tool_name"):
                    current_tool_name = event.part.tool_name

            if isinstance(event, PartDeltaEvent):
                if isinstance(event.delta, ThinkingPartDelta):
                    thinking_buffer.append(event.delta.content_delta)
                elif isinstance(event.delta, TextPartDelta):
                    text_buffer.append(event.delta.content_delta)
                elif isinstance(event.delta, ToolCallPartDelta):
                    tool_buffer.append(event.delta.args_delta)

            if isinstance(event, PartEndEvent):
                if thinking_buffer:
                    logger.info(f"[THINKING]\n{''.join(thinking_buffer)}\n")
                    thinking_buffer = []
                if text_buffer:
                    logger.info(f"[TEXT]\n{''.join(text_buffer)}\n")
                    text_buffer = []
                if tool_buffer:
                    args_str = "".join(tool_buffer)
                    logger.info(f"[TOOL CALL INVOKED]\nTool: {current_tool_name}\n")
                    if self.config.debug:
                        logger.info(f"[TOOL CALL ARGUMENTS] {args_str}\n")
                    tool_buffer = []
                    current_tool_name = None

            # Capture the final result
            if isinstance(event, FinalResultEvent):
                logger.info("[RUN COMPLETE]")

        result = last_event.result
        if result:
            logger.info("[COMPLETE, RESULT OBTAINED]")

        return result

    def __getattr__(self, name: str):
        # Forward access to underlying Agent attributes (tools, name, etc.)
        return getattr(self._agent, name)


# Base agent for user-facing conversational agents.
# To-Do:
# 1. Implement stream support
# 2. Set max_history_messages dynamically based on the model's context window.
class DLensConversationalAgent(DLensBaseAgent):
    """Extended base agent with conversation memory support."""

    def __init__(
        self,
        *,
        config: BaseAgentConfig,
        deps_type: Any | None = None,
        output_type: Any | None = str,
        register_tools: Callable | None = None,
        **agent_kwargs: Any,
    ):
        super().__init__(
            config=config,
            deps_type=deps_type,
            output_type=output_type,
            register_tools=register_tools,
            **agent_kwargs,
        )
        self._session_store = InMemorySessionStore(
            max_messages_per_session=config.max_history_messages
        )

    def create_session(self) -> str:
        """Create a new conversation session."""
        return self._session_store.create_session()

    async def arun(
        self,
        query: str | InputSchema,
        *,
        session_id: Optional[str] = None,
        deps: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Run the agent with conversation memory.

        Args:
            query: User input
            session_id: Session ID for conversation continuity (None = stateless)
            deps: Runtime dependencies
        """
        if isinstance(query, InputSchema):
            query = query.model_dump_json()
        else:
            query = str(query)

        # Get conversation history if session provided
        message_history = []
        if session_id:
            message_history = self._session_store.get_history(session_id)

        # Run with history
        result = await self._agent.run(
            query, deps=deps, message_history=message_history, **kwargs
        )

        # Store new messages if session provided
        if session_id:
            self._session_store.add_messages(session_id, result.new_messages())

        return result

    def clear_session(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        self._session_store.clear_session(session_id)


# Abstract Base Agent for any static workflow that takes an input schema and returns an output schema
# Only has an arun method that takes in input schema and returns output schema
# Everything else is dependency injected in the constructor


class AbstractInputSchema(BaseModel):
    """Input schema for the abstract agent."""


class AbstractOutputSchema(BaseModel):
    """Output schema for the abstract agent."""


class AbstractBaseAgent[AbstractInputSchema, AbstractOutputSchema]:
    def __init__(
        self, input_schema: AbstractInputSchema, output_schema: AbstractOutputSchema
    ):
        self.input_schema = input_schema
        self.output_schema = output_schema

    def arun(self, input: AbstractInputSchema) -> AbstractOutputSchema:
        """Run the agent on the input schema and return the output schema."""
        return self.output_schema
