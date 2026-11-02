"""MCP Server for Agent Memory.

Exposes search_memory and add_memory as MCP tools that can be used by
any MCP-compatible client (kiro-cli, Claude Code, Cursor, etc.).

Configuration via environment variables:
    MEMORY_FRAMEWORK - Backend (only graphiti in this build)
    LLM_BASE_URL / LLM_API_KEY - OpenAI-compatible gateway (LiteLLM proxy)
    AWS_REGION - AWS region
    + Backend-specific endpoint vars (see .env.example)

Usage:
    uv run python mcp_server.py
"""

import os

from mcp.server.fastmcp import FastMCP

# stdio for local clients (kiro-cli / Claude Code); streamable-http when run as a service.
# Host must be passed at construction: FastMCP enables DNS-rebinding protection (Host
# header must be localhost) when constructed with a localhost host, and changing
# settings.host afterwards doesn't undo it -> 421 for http://mcp.agent-memory.local.
TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")
if TRANSPORT == "stdio":
    mcp = FastMCP("agentmemory")
else:
    mcp = FastMCP("agentmemory", host=os.getenv("MCP_HOST", "0.0.0.0"), port=int(os.getenv("MCP_PORT", "8000")))

# Lazy-loaded memory backend — initialized on first tool call to keep server
# startup fast (kiro-cli has a timeout for MCP server initialization).
_memory = None


def _get_memory():
    global _memory
    if _memory is None:
        from frameworks.memory_backend import get_memory_backend
        _memory = get_memory_backend()
    return _memory


@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(request):
    from starlette.responses import PlainTextResponse
    return PlainTextResponse("ok")


@mcp.tool()
def search_memory(query: str, user_id: str):
    """Search a user's stored memories for relevant information.

    Use this to recall past conversations, preferences, facts, or context
    that has been previously stored for a user.

    Args:
        query: What to search for in the user's memories
        user_id: The user whose memories to search
    """
    memory = _get_memory()
    results = memory.search(query, user_id)
    if not results:
        return f"No memories found for user '{user_id}'. This user has no stored memories yet — use add_memory to start building their memory."
    return str(results)


@mcp.tool()
def add_memory(content: str, user_id: str):
    """Store new information in a user's memory for future recall.

    Use this to save important facts, preferences, decisions, or context
    that should be remembered across conversations.

    Args:
        content: The information to remember
        user_id: The user to store the memory for
    """
    memory = _get_memory()
    result = memory.add(content, user_id)
    return str(result)


if __name__ == "__main__":
    mcp.run(transport=TRANSPORT)
