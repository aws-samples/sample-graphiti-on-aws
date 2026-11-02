"""Core memory agent — a generic memory-enabled agent with configurable persona.

The agent LLM is reached through the same OpenAI-compatible gateway (LiteLLM proxy)
as Graphiti, using the AGENT_MODEL alias (default "agent-llm"). The model behind it
must support OpenAI-style tool calling (DeepSeek-V3/V3.2, Qwen3, GLM-4.5+, Kimi K2 do).

Configuration:
    AGENT_CONFIG      - Which agent persona to use (default: "default")
    AGENT_MODEL       - Gateway alias for the agent LLM (default: "agent-llm")
    LLM_BASE_URL      - Gateway URL, e.g. http://litellm.agent-memory.local:4000/v1
    LLM_API_KEY       - Gateway virtual key
"""

import json

from dotenv import load_dotenv
from strands import tool, Agent
from strands.models.openai import OpenAIModel

from examples import load_agent_config
from frameworks.memory_backend import get_memory_backend
from frameworks.settings import get_settings

load_dotenv()

settings = get_settings()

# Load agent configuration (persona, system prompt, extra tools)
agent_config = load_agent_config()

model = OpenAIModel(
    client_args={"base_url": settings.llm_base_url, "api_key": settings.llm_api_key},
    model_id=settings.agent_model,
    params={"max_tokens": settings.llm_max_tokens, "temperature": 0.3},
)

memory = get_memory_backend()


def create_memory_agent() -> Agent:
    """Build a new agent. Each conversation (e.g. each Streamlit session) needs its own,
    because the agent holds the message history and the current user_id."""
    agent: Agent

    @tool
    def search_memory(query: str) -> str:
        """Search the user's stored memories for relevant information."""
        return json.dumps(memory.search(query, agent.state.get('user_id')), ensure_ascii=False)

    @tool
    def add_memory(query: str) -> str:
        """Store new information in the user's memory for future recall."""
        return json.dumps(memory.add(query, agent.state.get('user_id')), ensure_ascii=False)

    # Core memory tools + any extras from the agent config
    agent = Agent(
        model=model,
        system_prompt=agent_config.SYSTEM_PROMPT,
        tools=[search_memory, add_memory] + agent_config.EXTRA_TOOLS,
    )
    return agent


# Shared instance for single-conversation scripts; app.py creates one per session.
memory_agent = create_memory_agent()
