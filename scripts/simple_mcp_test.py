from dlens.agents._base import DLensBaseAgent, BaseAgentConfig, OutputSchema
from dlens.agents._models import LlamaCppModel, OpenAIModel
from pydantic import Field
import asyncio
from dlens.tools.example_mcp_server import EXAMPLE_TOOLSET

from typing import Optional

config = BaseAgentConfig(
    name="Example MCP Agent",
    description="An agent that can answer questions using NASA's APOD and image search APIs.",
    model=OpenAIModel(model_name="gpt-4o-mini"),
)


class Result(OutputSchema):
    apod: Optional[dict] = Field(default=None, description="The APOD data.")
    images: Optional[dict] = Field(default=None, description="The images data.")


agent = DLensBaseAgent(config=config, toolsets=[EXAMPLE_TOOLSET], output_type=Result)

result = asyncio.run(
    agent.arun_stream(
        "1. Get the NASA's APOD for today and 2. search for images of the moon."
    )
)
print(result.output)
