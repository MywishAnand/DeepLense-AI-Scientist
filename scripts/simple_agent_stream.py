import logging

logging.basicConfig(level=logging.DEBUG)

from dlens.agents._base import DLensBaseAgent, BaseAgentConfig, OutputSchema
from dlens.agents._models import LlamaCppModel
from pydantic import Field
import asyncio
from loguru import logger

config = BaseAgentConfig(
    name="Generic QA Agent",
    description="A generic QA agent that can answer questions.",
    model=LlamaCppModel(model_name="gpt-oss:20b"),
)


class Result(OutputSchema):
    answer: str = Field(description="The answer to the question.")


agent = DLensBaseAgent(config=config, output_type=Result)


async def main():
    query = "Explain how Machine Learning is used for Flood Detection?"

    result = await agent.arun_stream(query)
    logger.info(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
