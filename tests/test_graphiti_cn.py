import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from frameworks import graphiti_neptune as gn
from frameworks.settings import get_settings


class TestOpenSearch:
    def test_managed_domain_uses_es_signer_and_cn_region(self):
        client = gn.build_opensearch_client(get_settings())
        conn = client.transport.get_connection()
        assert conn.host == "https://vpc-graphiti-abc.cn-northwest-1.es.amazonaws.com.cn:443"
        auth = client.transport.kwargs["http_auth"]
        assert auth.service == "es"
        assert auth.signer.region == "cn-northwest-1"

    def test_aoss_still_selectable(self, monkeypatch):
        monkeypatch.setenv("GRAPHITI_OPENSEARCH_SERVICE", "aoss")
        assert gn.build_opensearch_client(get_settings()).transport.kwargs["http_auth"].service == "aoss"

    def test_basic_auth(self, monkeypatch):
        monkeypatch.setenv("GRAPHITI_OPENSEARCH_AUTH", "basic")
        monkeypatch.setenv("GRAPHITI_OPENSEARCH_USERNAME", "admin")
        monkeypatch.setenv("GRAPHITI_OPENSEARCH_PASSWORD", "pw")
        assert gn.build_opensearch_client(get_settings()).transport.kwargs["http_auth"] == ("admin", "pw")


# langchain_aws.NeptuneGraph would build a neptunedata client and fetch the schema
@patch("graphiti_core.driver.neptune_driver.NeptuneGraph")
class TestGraphitiDemo:
    def test_wires_gateway_clients_and_replaces_aoss_client(self, neptune_graph):
        demo = gn.GraphitiDemo("u1")
        neptune_graph.assert_called_once_with("my-cluster.cluster-abc.cn-northwest-1.neptune.amazonaws.com.cn", 8182)
        assert demo.driver.aoss_client.transport.get_connection().host.startswith(
            "https://vpc-graphiti-abc.cn-northwest-1.es.amazonaws.com.cn")
        assert demo.driver.aoss_client.transport.kwargs["http_auth"].service == "es"
        llm = demo.client.llm_client
        assert isinstance(llm, gn.GatewayLLMClient)
        assert str(llm.client.base_url).startswith("http://litellm.agent-memory.local:4000/v1")
        assert llm.model == "graphiti-llm" and llm.small_model == "graphiti-small"
        assert llm.structured_output_mode == "json_object"
        assert isinstance(demo.client.embedder, gn.BatchedOpenAIEmbedder)
        assert isinstance(demo.client.cross_encoder, gn.PassthroughReranker)

    def test_rerank_model_enables_gateway_reranker(self, _, monkeypatch):
        monkeypatch.setenv("RERANK_MODEL", "rerank")
        ce = gn.GraphitiDemo("u1").client.cross_encoder
        assert isinstance(ce, gn.GatewayRerankerClient)
        assert ce.url == "http://litellm.agent-memory.local:4000/v1/rerank"

    def test_requires_endpoints(self, _, monkeypatch):
        monkeypatch.delenv("GRAPHITI_OPENSEARCH_ENDPOINT")
        with pytest.raises(ValueError, match="OPENSEARCH"):
            gn.GraphitiDemo("u1")


class TestTolerantJson:
    @pytest.mark.parametrize("raw", [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '{"a": 1}\n\nI extracted one entity.',
        'Here is the JSON:\n{"a": 1}',
        'Sure! ```json\n{"a": 1}\n``` done',
    ])
    def test_extracts_first_json_value(self, raw):
        assert json.loads(gn.GatewayLLMClient._strip_code_fences(raw)) == {"a": 1}

    def test_nested_and_braces_in_strings(self):
        raw = 'ok {"e": [{"name": "A {x}"}], "n": {"k": 2}} trailing }'
        assert json.loads(gn.GatewayLLMClient._strip_code_fences(raw)) == {"e": [{"name": "A {x}"}], "n": {"k": 2}}


def _emb_response(vectors):
    return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=v) for i, v in reversed(list(enumerate(vectors)))])


class TestEmbedder:
    def _embedder(self, batch_size=2):
        return gn.build_embedder(get_settings()) if batch_size is None else gn.BatchedOpenAIEmbedder(
            gn.OpenAIEmbedderConfig(api_key="k", base_url="http://x/v1", embedding_model="embedding", embedding_dim=4),
            batch_size=batch_size)

    async def test_batches_and_orders(self):
        e = self._embedder(batch_size=2)
        calls = []

        async def create(input, model):
            calls.append(list(input))
            return _emb_response([[float(len(t))] * 4 for t in input])

        e.client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
        out = await e.create_batch(["a", "bb", "ccc", "dddd", "eeeee"])
        assert calls == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]
        assert [v[0] for v in out] == [1, 2, 3, 4, 5]

    async def test_dimension_mismatch_is_loud(self):
        e = self._embedder()

        async def create(input, model):
            return _emb_response([[0.1] * 1024])

        e.client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
        with pytest.raises(ValueError, match="EMBEDDING_DIM=4"):
            await e.create("hello")


class TestReranker:
    async def test_maps_scores_back_to_passages(self, monkeypatch):
        sent = {}

        class FakeClient:
            def __init__(self, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def post(self, url, headers, json):
                sent.update(url=url, json=json, headers=headers)
                return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"results": [
                    {"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}]})

        monkeypatch.setattr(gn.httpx, "AsyncClient", FakeClient)
        r = gn.GatewayRerankerClient("http://gw/v1/", "sk-x", "rerank")
        out = await r.rank("q", ["p0", "p1", "p2"])
        assert sent["url"] == "http://gw/v1/rerank"
        assert sent["json"]["documents"] == ["p0", "p1", "p2"]
        assert sent["headers"]["Authorization"] == "Bearer sk-x"
        assert out == [("p1", 0.9), ("p0", 0.2), ("p2", 0.0)]

    async def test_passthrough_keeps_order(self):
        out = await gn.PassthroughReranker().rank("q", ["a", "b", "c"])
        assert [p for p, _ in out] == ["a", "b", "c"]
        assert out[0][1] > out[1][1] > out[2][1]


class TestNeptunePatch:
    def test_strips_unreferenced_and_none_params(self):
        drv = SimpleNamespace(client=MagicMock(), _sanitize_parameters=lambda q, p: q)
        drv.client.query.return_value = [{"x": 1}]
        gn._patched_run_query(drv, "MATCH (n) WHERE n.g IN $group_ids RETURN n",
                              {"params": {"group_ids": ["u"]}, "unused": 1, "group_ids_none": None})
        assert drv.client.query.call_args.kwargs["params"] == {"group_ids": ["u"]}


class TestRunSync:
    def test_reuses_one_loop_across_calls(self):
        async def loop_id():
            return id(asyncio.get_running_loop())
        assert gn.run_sync(loop_id()) == gn.run_sync(loop_id())

    async def test_callable_from_inside_running_loop(self):
        async def two():
            return 2
        assert await asyncio.to_thread(gn.run_sync, two()) == 2


def test_mcp_sdk_has_fastmcp():
    # mcp 2.x renamed FastMCP -> MCPServer; mcp_server.py targets 1.x (pinned <2)
    from mcp.server.fastmcp import FastMCP  # noqa: F401


def test_neptune_schema_failure_is_tolerated(monkeypatch):
    # Statistics are off on t3/t4g Neptune instances -> summary API fails on init
    import frameworks.graphiti_neptune as gn

    def boom(self):
        raise RuntimeError("Statistics are disabled, thus summary is also disabled.")

    monkeypatch.setattr(gn, "_orig_refresh_schema", boom)
    g = object.__new__(gn.NeptuneGraph)
    gn.NeptuneGraph._refresh_schema(g)
    assert g.schema == ""


def test_index_episode_writes_nodes_edges_and_episode():
    demo = gn.GraphitiDemo.__new__(gn.GraphitiDemo)
    demo.driver = MagicMock()
    demo.driver.save_to_aoss.side_effect = lambda name, data: len(data)
    node = SimpleNamespace(uuid="n1", group_id="u", name="Li Lei", summary=None)
    edge = SimpleNamespace(uuid="e1", group_id="u", name="ALLERGIC_TO", fact="Li Lei is allergic to peanuts")
    ep = SimpleNamespace(uuid="p1", group_id="u", content="...", source_description="user_message")
    result = SimpleNamespace(nodes=[node], edges=[edge], episode=ep)

    assert demo.index_episode(result) == 3
    calls = {c.args[0]: c.args[1] for c in demo.driver.save_to_aoss.call_args_list}
    assert calls["node_name_and_summary"] == [{"uuid": "n1", "group_id": "u", "name": "Li Lei", "summary": ""}]
    assert calls["edge_name_and_fact"][0]["fact"] == "Li Lei is allergic to peanuts"
    assert calls["episode_content"][0]["uuid"] == "p1"
