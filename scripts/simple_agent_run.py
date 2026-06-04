from dlens.agents._base import DLensBaseAgent, BaseAgentConfig, OutputSchema
from dlens.agents._models import OllamaModel, OpenAIModel
from pydantic import Field
import asyncio

config = BaseAgentConfig(
    name="Generic QA Agent",
    description="A generic QA agent that can answer questions about the given context.",
    model=OllamaModel(model_name="qwen3:8b", port="11434"),
    # model=OpenAIModel(model_name="gpt-4o-mini"),
)


class Result(OutputSchema):
    answer: str = Field(description="The answer to the question.")


agent = DLensBaseAgent(config=config, output_type=Result)

print(
    asyncio.run(agent.arun("Explain how Machine Learning is used for Flood Detection?"))
)
