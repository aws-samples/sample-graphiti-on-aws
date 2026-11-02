import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def cn_env(monkeypatch):
    """Fake China-region config; nothing here talks to real infrastructure."""
    monkeypatch.setenv("AWS_REGION", "cn-northwest-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "cn-northwest-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("GRAPHITI_NEPTUNE_ENDPOINT", "my-cluster.cluster-abc.cn-northwest-1.neptune.amazonaws.com.cn")
    monkeypatch.setenv("GRAPHITI_OPENSEARCH_ENDPOINT", "https://vpc-graphiti-abc.cn-northwest-1.es.amazonaws.com.cn/")
    monkeypatch.setenv("LLM_BASE_URL", "http://litellm.agent-memory.local:4000/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("EMBEDDING_DIM", "4")
