"""Central configuration, read from environment variables (or .env).

Everything model-related points at an OpenAI-compatible gateway (LiteLLM proxy by
default). The application only knows *aliases* such as ``graphiti-llm`` or
``embedding``; which real provider/model sits behind each alias is decided in
``litellm/config.yaml``. Switching DeepSeek -> Qwen -> a SageMaker endpoint is a
gateway config change, not an application change.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    # --- OpenAI-compatible model gateway (LiteLLM proxy) ---
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "http://localhost:4000/v1"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "sk-litellm-local"))

    # Aliases resolved by the gateway
    graphiti_llm_model: str = field(default_factory=lambda: _env("GRAPHITI_LLM_MODEL", "graphiti-llm"))
    graphiti_small_model: str = field(default_factory=lambda: _env("GRAPHITI_SMALL_MODEL", "graphiti-small"))
    agent_model: str = field(default_factory=lambda: _env("AGENT_MODEL", "agent-llm"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "embedding"))
    # Empty string disables the remote reranker (a no-op passthrough is used instead)
    rerank_model: str = field(default_factory=lambda: _env("RERANK_MODEL", ""))

    # json_object works with DeepSeek / Qwen / GLM / Kimi; json_schema only where supported
    structured_output_mode: str = field(default_factory=lambda: _env("LLM_STRUCTURED_OUTPUT_MODE", "json_object"))
    llm_max_tokens: int = field(default_factory=lambda: int(_env("LLM_MAX_TOKENS", "8192")))
    llm_temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0")))

    # Must equal the native output dimension of the embedding model behind the alias.
    # (graphiti_core reads EMBEDDING_DIM itself at import time, so keep them in sync.)
    embedding_dim: int = field(default_factory=lambda: int(_env("EMBEDDING_DIM", "1024")))
    # Many CN providers cap batch size (DashScope: 10, SiliconFlow: 32)
    embedding_batch_size: int = field(default_factory=lambda: int(_env("EMBEDDING_BATCH_SIZE", "10")))

    # --- Neptune Database ---
    neptune_endpoint: str = field(default_factory=lambda: _env("GRAPHITI_NEPTUNE_ENDPOINT"))
    neptune_port: int = field(default_factory=lambda: int(_env("GRAPHITI_NEPTUNE_PORT", "8182")))

    # --- OpenSearch (full-text / BM25 index used by Graphiti's Neptune driver) ---
    # China regions have no OpenSearch Serverless, so default to a managed domain ("es").
    opensearch_endpoint: str = field(default_factory=lambda: _env("GRAPHITI_OPENSEARCH_ENDPOINT"))
    opensearch_port: int = field(default_factory=lambda: int(_env("GRAPHITI_OPENSEARCH_PORT", "443")))
    opensearch_service: str = field(default_factory=lambda: _env("GRAPHITI_OPENSEARCH_SERVICE", "es"))
    # "iam" (SigV4) or "basic" (fine-grained access control master user)
    opensearch_auth: str = field(default_factory=lambda: _env("GRAPHITI_OPENSEARCH_AUTH", "iam"))
    opensearch_username: str = field(default_factory=lambda: _env("GRAPHITI_OPENSEARCH_USERNAME"))
    opensearch_password: str = field(default_factory=lambda: _env("GRAPHITI_OPENSEARCH_PASSWORD"))

    aws_region: str = field(default_factory=lambda: _env("AWS_REGION") or _env("AWS_DEFAULT_REGION") or "cn-northwest-1")


def get_settings() -> Settings:
    return Settings()
