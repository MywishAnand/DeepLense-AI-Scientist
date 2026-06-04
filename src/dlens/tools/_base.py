from pydantic import BaseModel

# Abstract base classes for tools that are explicitly defined in deterministic workflows. Not meant for MCP (Model Context Protocol) based tools.


class AbstractToolInputSchema(BaseModel):
    """Input schema for the abstract tool."""


class AbstractToolOutputSchema(BaseModel):
    """Output schema for the abstract tool."""


class AbstractBaseTool[AbstractToolInputSchema, AbstractToolOutputSchema]:
    def __init__(
        self,
        input_schema: AbstractToolInputSchema,
        output_schema: AbstractToolOutputSchema,
    ):
        self.input_schema = input_schema
        self.output_schema = output_schema

    def arun(self, input: AbstractToolInputSchema) -> AbstractToolOutputSchema:
        """Run the agent on the input schema and return the output schema."""
        return self.output_schema
