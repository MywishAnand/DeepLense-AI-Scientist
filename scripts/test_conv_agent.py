from dlens.agents._base import DLensConversationalAgent, BaseAgentConfig
from dlens.agents._models import LlamaCppModel
from pydantic import Field
import asyncio

config = BaseAgentConfig(
    name="Conversational Assistant",
    description="A helpful assistant that remembers conversation context",
    model=LlamaCppModel(model_name="gpt-oss:20b"),
    max_history_messages=10,
)

agent = DLensConversationalAgent(config=config)

session_id = agent.create_session()

response1 = asyncio.run(
    agent.arun(
        query="My name is Pranath and I work on gravitational lensing simulations",
        session_id=session_id,
    )
)

response2 = asyncio.run(
    agent.arun(
        query="What did I say my name was?",
        session_id=session_id,
    )
)

print(response1)
print(response2)

agent.clear_session(session_id)
