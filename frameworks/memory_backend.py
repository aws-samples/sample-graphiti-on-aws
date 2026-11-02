"""Memory backend interface (Graphiti only in the China-region build).

Keeps the same search/add API as the global sample so app.py / mcp_server.py /
tools/memory_tool.py are unchanged in shape.
"""

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class MemoryBackend(ABC):
    """Common interface for memory operations."""

    @abstractmethod
    def search(self, query: str, user_id: str) -> list:
        """Search memories for a user."""
        ...

    @abstractmethod
    def add(self, content: str, user_id: str) -> dict:
        """Store a new memory for a user."""
        ...


class GraphitiBackend(MemoryBackend):
    """Memory backend using Graphiti with Neptune Database + OpenSearch + LiteLLM gateway."""

    def __init__(self):
        # boto3 reads AWS_DEFAULT_REGION, users typically set AWS_REGION. Bridge the gap.
        if not os.getenv("AWS_DEFAULT_REGION") and os.getenv("AWS_REGION"):
            os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_REGION")

        from frameworks.graphiti_neptune import GraphitiDemo, run_sync
        self._run = run_sync
        self._demo = GraphitiDemo("")
        self._run(self._demo.ensure_indices())

    def search(self, query: str, user_id: str) -> list:
        edges = self._run(self._demo.client.search(query, group_ids=[user_id]))
        # Return plain facts; EntityEdge objects carry embeddings and are noisy for the agent
        return [
            {
                "fact": e.fact,
                "valid_at": e.valid_at.isoformat() if e.valid_at else None,
                "invalid_at": e.invalid_at.isoformat() if e.invalid_at else None,
            }
            for e in edges
        ]

    def add(self, content: str, user_id: str) -> dict:
        from graphiti_core.nodes import EpisodeType

        now = datetime.now(timezone.utc)
        result = self._run(self._demo.client.add_episode(
            name=f"memory_{now.isoformat()}",
            episode_body=content,
            source=EpisodeType.message,
            source_description="user_message",
            reference_time=now,
            group_id=user_id,
        ))
        self._demo.index_episode(result)
        return {"status": "added", "user_id": user_id}


BACKENDS = {
    "graphiti": GraphitiBackend,
}


def get_memory_backend() -> MemoryBackend:
    """Create the configured memory backend (MEMORY_FRAMEWORK, default graphiti)."""
    framework = os.getenv("MEMORY_FRAMEWORK", "graphiti").lower()
    if framework not in BACKENDS:
        raise ValueError(
            f"Unknown MEMORY_FRAMEWORK '{framework}'. "
            f"Valid options: {', '.join(BACKENDS.keys())}"
        )
    return BACKENDS[framework]()
