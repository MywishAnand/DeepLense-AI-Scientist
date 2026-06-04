# Example MCP server demonstrating how to wrap REST tools for Pydantic AI agents.
# Uses NASA's public APIs as a reference implementation.
# credits to https://github.com/portkeys/nasa-mcp

from dlens.tools._mcp_wrapper import mcp_rest_tool
from dlens.tools.example_rest_tools import get_nasa_apod, search_nasa_images
from mcp.server.fastmcp import FastMCP
from pydantic_ai.toolsets.fastmcp import FastMCPToolset

mcp = FastMCP("Example MCP Server")

GET_APOD_DOC = """Get NASA's Astronomy Picture of the Day (APOD).

Retrieves the featured astronomy image or video for a specific date, along with its
title, explanation, and metadata.

Args:
    date: Optional date in YYYY-MM-DD format. If not provided, returns today's APOD.
          Must be between 1995-06-16 (first APOD) and today's date.

Returns:
    JSON string containing APOD data (title, explanation, URL, date, etc.),
    or {"error": "..."} if the request fails.
"""

SEARCH_IMAGES_DOC = """Search NASA's image and video library.

Searches NASA's Image and Video Library for images matching a query string.

Args:
    query: Search query string (e.g., "Mars rover", "International Space Station").
    size: Number of results to return (page_size). Recommended range: 1-20.

Returns:
    JSON string containing search results and metadata, or {"error": "..."} if the request fails.
"""


mcp_rest_tool(
    mcp=mcp,
    name="get_nasa_apod",
    rest_fn=get_nasa_apod,
    doc=GET_APOD_DOC,
    none_error="Failed to retrieve APOD data",
    indent=2,
)

mcp_rest_tool(
    mcp=mcp,
    name="search_nasa_images",
    rest_fn=search_nasa_images,
    doc=SEARCH_IMAGES_DOC,
    none_error="Failed to search NASA images",
    indent=2,
)


@mcp.tool()
def get_todays_date() -> str:
    """Return today's date in YYYY-MM-DD format."""
    from datetime import date

    return date.today().isoformat()


EXAMPLE_TOOLSET = FastMCPToolset(mcp)
