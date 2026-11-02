"""E2E check of the gateway contract: Graphiti extraction + search with every model
call going through the LiteLLM proxy. embedded FalkorDB (falkordblite) stands in for Neptune so this
runs on a laptop; the Neptune/OpenSearch path is covered by tests/ and smoke_test.py.

    LLM_BASE_URL=http://localhost:4011/v1 LLM_API_KEY=sk-test RERANK_MODEL=rerank \
      python scripts/testing/e2e_gateway_local.py
"""
import asyncio, sys, tempfile, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from datetime import datetime, timezone

from graphiti_core import Graphiti
from falkordb.asyncio import FalkorDB
from graphiti_core.driver.falkordb_driver import FalkorDriver
from redislite.falkordb_client import FalkorDB as EmbeddedFalkorDB
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config import SearchConfig, EdgeSearchConfig, EdgeSearchMethod, EdgeReranker

from frameworks.graphiti_neptune import build_llm_client, build_embedder, build_cross_encoder
from frameworks.settings import get_settings


async def main():
    s = get_settings()
    server = EmbeddedFalkorDB(os.path.join(tempfile.mkdtemp(), "g.rdb"))
    sock = server.connection.connection_pool.connection_kwargs["path"]
    driver = FalkorDriver(falkor_db=FalkorDB(unix_socket_path=sock), database="zhangwei")
    g = Graphiti(graph_driver=driver, llm_client=build_llm_client(s),
                 embedder=build_embedder(s), cross_encoder=build_cross_encoder(s))
    await g.build_indices_and_constraints()
    for text in ["我叫张伟，住在上海，坐飞机喜欢靠窗的座位。",
                 "张伟下个月要去成都出差，预算每晚 600 元以内的酒店。",
                 "Zhang Wei is vegetarian and prefers Air China."]:
        await g.add_episode(name="m", episode_body=text, source=EpisodeType.message,
                            source_description="user_message",
                            reference_time=datetime.now(timezone.utc), group_id="zhangwei")
    edges = await g.search("张伟订机票有什么偏好？", group_ids=["zhangwei"])
    print("hybrid RRF search (default recipe used by the app):"); [print("  -", e.fact) for e in edges]
    ce = SearchConfig(edge_config=EdgeSearchConfig(
        search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
        reranker=EdgeReranker.cross_encoder))
    res = await g.search_("张伟的饮食偏好", config=ce, group_ids=["zhangwei"])
    print("cross-encoder search:"); [print(f"  - {e.fact}  score={sc:.3f}") for e, sc in zip(res.edges[:5], res.edge_reranker_scores[:5])]
    assert edges and res.edges
    await g.close()
    print("OK")

asyncio.run(main())
