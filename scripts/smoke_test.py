"""Post-deploy smoke test. Run inside the VPC with scripts/run_in_vpc.sh (a one-off
task from the app task definition), which also sets LLM_API_KEY from the master-key secret.

Checks, in order: gateway aliases -> Neptune -> OpenSearch -> add + search one memory.
"""
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from openai import OpenAI

from frameworks.settings import get_settings

s = get_settings()
ok = True


def step(name, fn):
    global ok
    t = time.time()
    try:
        detail = fn()
        print(f"[PASS] {name} ({time.time() - t:.1f}s) {detail or ''}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


oai = OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key)


def chat(alias):
    def _():
        r = oai.chat.completions.create(
            model=alias, max_tokens=50, response_format={"type": "json_object"},
            messages=[{"role": "user", "content": 'Reply with JSON {"ok": true}'}])
        return r.choices[0].message.content.strip()[:60]
    return _


def embed():
    r = oai.embeddings.create(model=s.embedding_model, input=["测试", "test"])
    dims = {len(d.embedding) for d in r.data}
    assert dims == {s.embedding_dim}, f"got dims {dims}, EMBEDDING_DIM={s.embedding_dim}"
    return f"dim={s.embedding_dim}"


def rerank():
    r = httpx.post(s.llm_base_url.rstrip("/") + "/rerank", timeout=30,
                   headers={"Authorization": f"Bearer {s.llm_api_key}"},
                   json={"model": s.rerank_model, "query": "靠窗", "documents": ["喜欢靠窗座位", "天气"]})
    r.raise_for_status()
    return r.json()["results"][0]


def tools():
    r = oai.chat.completions.create(
        model=s.agent_model, max_tokens=100,
        messages=[{"role": "user", "content": "What's the weather in Beijing? Use the tool."}],
        tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {
            "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}])
    calls = r.choices[0].message.tool_calls
    assert calls, "agent-llm did not emit a tool call"
    return calls[0].function.arguments


step("gateway graphiti-llm json_object", chat(s.graphiti_llm_model))
step("gateway graphiti-small json_object", chat(s.graphiti_small_model))
step("gateway agent-llm tool calling", tools)
step("gateway embedding", embed)
if s.rerank_model:
    step("gateway rerank", rerank)

from frameworks.graphiti_neptune import GraphitiDemo, run_sync  # noqa: E402

demo = GraphitiDemo("")
step("neptune openCypher", lambda: demo.driver.client.query("RETURN 1 AS x"))
step("opensearch cluster health", lambda: demo.driver.aoss_client.cluster.health()["status"])
step("opensearch indices", lambda: f"created={run_sync(demo.ensure_indices())}")

from frameworks.memory_backend import GraphitiBackend  # noqa: E402

user = f"smoke-{uuid.uuid4().hex[:8]}"
backend = GraphitiBackend()
step("add_memory", lambda: backend.add("我是冒烟测试用户，喜欢喝乌龙茶。", user))
step("search_memory", lambda: backend.search("喜欢喝什么茶", user))
step("cleanup", lambda: demo.driver.client.query(f"MATCH (n) WHERE n.group_id = '{user}' DETACH DELETE n"))

sys.exit(0 if ok else 1)
