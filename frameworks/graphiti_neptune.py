"""Graphiti on Amazon Neptune Database for AWS China regions.

Differences from the global sample (aws-samples/amazon-neptune-generative-ai-samples,
solutions/agent-memory):

* LLM, embedding and rerank calls go through an OpenAI-compatible gateway
  (LiteLLM proxy) instead of Amazon Bedrock, which is not offered in cn-north-1 /
  cn-northwest-1. The gateway decides which provider is actually used.
* The full-text index lives in an Amazon OpenSearch Service *managed domain*
  (SigV4 service name ``es``) instead of OpenSearch Serverless (``aoss``), which is
  also not offered in China regions. Graphiti's Neptune driver only uses OpenSearch
  for BM25 text search; vectors are stored on Neptune nodes/edges.
"""

import asyncio
import json
import logging
import re
import threading
from collections.abc import Iterable

import boto3
import httpx
from langchain_aws.graphs import NeptuneGraph
from opensearchpy import OpenSearch, Urllib3AWSV4SignerAuth, Urllib3HttpConnection

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.neptune_driver import NeptuneDriver, aoss_indices
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

from frameworks.settings import Settings, get_settings

logger = logging.getLogger(__name__)


def _parse_endpoint(endpoint: str) -> str:
    """Strip protocol prefixes and trailing slashes from an endpoint string.

    Accepts: hostname, https://hostname, neptune-db://hostname
    Returns: just the hostname
    """
    if not endpoint:
        return endpoint
    for prefix in ("neptune-db://", "neptune-graph://", "https://", "http://"):
        if endpoint.startswith(prefix):
            endpoint = endpoint[len(prefix):]
            break
    return endpoint.rstrip("/")


# --- Monkey-patch NeptuneDriver at the class level (carried over from the global sample) ---
# Neptune Database's execute_open_cypher_query rejects queries with parameters
# that are not referenced in the query string OR have None values.
# graphiti_core's _sanitize_parameters can inline datetime values (removing $param)
# without removing the param from the dict. It also conditionally builds query clauses
# but unconditionally passes all params.
# This patch strips unreferenced and None-valued params after sanitization.
def _patched_run_query(self, cypher_query_, params):
    # Handle nested 'params' dict: search_utils.py passes params=filter_params as a kwarg,
    # which creates {'params': {'group_ids': [...]}, 'search_vector': [...], ...}.
    # Flatten by merging the nested 'params' dict into the top level.
    if 'params' in params and isinstance(params['params'], dict):
        nested = params.pop('params')
        params.update(nested)

    cypher_query_ = str(self._sanitize_parameters(cypher_query_, params))
    if params:
        referenced = set(re.findall(r'\$(\w+)', cypher_query_))
        params = {k: v for k, v in params.items() if k in referenced and v is not None}
    try:
        result = self.client.query(cypher_query_, params=params)
    except Exception as e:
        logger.error('Query: %s', cypher_query_)
        logger.error('Parameters: %s', params)
        logger.error('Error executing query: %s', e)
        raise e
    return result, None, None


NeptuneDriver._run_query = _patched_run_query


# langchain_aws NeptuneGraph loads the schema via the property-graph summary API in
# __init__ and raises if that fails. The summary needs Neptune statistics, which are
# off on t3/t4g instances (and when disabled explicitly). Graphiti never uses the
# schema, so a failure there must not block startup.
_orig_refresh_schema = NeptuneGraph._refresh_schema


def _tolerant_refresh_schema(self):
    try:
        _orig_refresh_schema(self)
    except Exception as e:  # noqa: BLE001
        logger.warning("Neptune schema summary unavailable, continuing without it: %s", e)
        self.schema = ""


NeptuneGraph._refresh_schema = _tolerant_refresh_schema


def build_opensearch_client(settings: Settings) -> OpenSearch:
    """OpenSearch client for a managed domain (default) or a serverless collection.

    In China regions only managed domains exist, so the SigV4 service name is ``es``.
    ``aoss`` is kept selectable so the same code still runs in global regions.
    """
    host = _parse_endpoint(settings.opensearch_endpoint)
    if settings.opensearch_auth == "basic":
        http_auth = (settings.opensearch_username, settings.opensearch_password)
    else:
        session = boto3.Session(region_name=settings.aws_region)
        http_auth = Urllib3AWSV4SignerAuth(
            session.get_credentials(), settings.aws_region, settings.opensearch_service
        )
    return OpenSearch(
        hosts=[{'host': host, 'port': settings.opensearch_port}],
        http_auth=http_auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=Urllib3HttpConnection,
        pool_maxsize=20,
        timeout=30,
    )


class BatchedOpenAIEmbedder(OpenAIEmbedder):
    """OpenAI-compatible embedder that respects provider batch-size limits.

    Graphiti sends whole lists to ``create_batch``; DashScope rejects >10 inputs and
    SiliconFlow >32, so the list is chunked. Vectors are *not* truncated here — the
    gateway alias must already return ``EMBEDDING_DIM``-sized vectors, because slicing
    a non-Matryoshka embedding silently degrades it.
    """

    def __init__(self, config: OpenAIEmbedderConfig, batch_size: int = 10):
        super().__init__(config=config)
        self.batch_size = max(1, batch_size)

    def _check_dim(self, vector: list[float]) -> list[float]:
        if len(vector) != self.config.embedding_dim:
            raise ValueError(
                f"Embedding model '{self.config.embedding_model}' returned {len(vector)} dims "
                f"but EMBEDDING_DIM={self.config.embedding_dim}. Set EMBEDDING_DIM to the model's "
                "native dimension (and keep it fixed once data has been written)."
            )
        return vector

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        result = await self.client.embeddings.create(
            input=input_data, model=self.config.embedding_model
        )
        return self._check_dim(result.data[0].embedding)

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(input_data_list), self.batch_size):
            chunk = input_data_list[i:i + self.batch_size]
            result = await self.client.embeddings.create(input=chunk, model=self.config.embedding_model)
            # Some providers don't guarantee order; sort by the returned index
            for item in sorted(result.data, key=lambda d: d.index):
                vectors.append(self._check_dim(item.embedding))
        return vectors


class GatewayRerankerClient(CrossEncoderClient):
    """Cross-encoder that calls the gateway's Cohere-style ``/rerank`` endpoint.

    LiteLLM proxy exposes ``POST /rerank`` and can route it to Cohere, Jina, Voyage,
    SiliconFlow / Infinity / vLLM hosted bge-reranker, SageMaker, etc. This replaces
    Graphiti's OpenAIRerankerClient, whose logprob + ``logit_bias`` trick relies on
    OpenAI tokenizer token IDs and does not carry over to other model families.
    """

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 30.0):
        self.url = base_url.rstrip('/') + '/rerank'
        self.model = model
        self.headers = {'Authorization': f'Bearer {api_key}'}
        self.timeout = timeout

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        if not passages:
            return []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                self.url,
                headers=self.headers,
                json={'model': self.model, 'query': query, 'documents': passages, 'top_n': len(passages)},
            )
            resp.raise_for_status()
            results = resp.json().get('results', [])
        scores = {r['index']: float(r['relevance_score']) for r in results}
        ranked = [(p, scores.get(i, 0.0)) for i, p in enumerate(passages)]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked


class PassthroughReranker(CrossEncoderClient):
    """Used when no rerank model is configured.

    Graphiti's default ``search()`` recipe uses RRF and never calls the cross-encoder,
    but Graphiti still constructs an OpenAI reranker (and demands OPENAI_API_KEY) unless
    one is supplied. This keeps the original order with monotonically decreasing scores.
    """

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        n = len(passages)
        return [(p, 1.0 - i / max(n, 1)) for i, p in enumerate(passages)]


class GatewayLLMClient(OpenAIGenericClient):
    """OpenAIGenericClient with tolerant JSON extraction.

    In ``json_object`` mode not every model behind the gateway honours the format
    strictly: some prepend a sentence, wrap the object in a fence, or append an
    explanation after it. Graphiti then fails on ``json.loads`` ("Extra data").
    This keeps only the first complete top-level JSON value.
    """

    _decoder = json.JSONDecoder()

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        stripped = OpenAIGenericClient._strip_code_fences(text)
        try:
            json.loads(stripped)
            return stripped
        except ValueError:
            pass
        for i, ch in enumerate(stripped):
            if ch in '{[':
                try:
                    _, end = GatewayLLMClient._decoder.raw_decode(stripped, i)
                    return stripped[i:end]
                except ValueError:
                    continue
        return stripped


def build_llm_client(settings: Settings) -> OpenAIGenericClient:
    llm_config = LLMConfig(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.graphiti_llm_model,
        small_model=settings.graphiti_small_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    return GatewayLLMClient(
        config=llm_config,
        max_tokens=settings.llm_max_tokens,
        structured_output_mode=settings.structured_output_mode,  # type: ignore[arg-type]
    )


def build_embedder(settings: Settings) -> EmbedderClient:
    return BatchedOpenAIEmbedder(
        OpenAIEmbedderConfig(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            embedding_model=settings.embedding_model,
            embedding_dim=settings.embedding_dim,
        ),
        batch_size=settings.embedding_batch_size,
    )


def build_cross_encoder(settings: Settings) -> CrossEncoderClient:
    if settings.rerank_model:
        return GatewayRerankerClient(settings.llm_base_url, settings.llm_api_key, settings.rerank_model)
    return PassthroughReranker()


class GraphitiDemo:
    """Wrapper over Graphiti using Neptune Database + OpenSearch managed domain,
    with every model call going through an OpenAI-compatible gateway.

    Required environment variables:
        GRAPHITI_NEPTUNE_ENDPOINT    - neptune-db://<cluster-endpoint>
        GRAPHITI_OPENSEARCH_ENDPOINT - OpenSearch managed domain (VPC) endpoint
        LLM_BASE_URL / LLM_API_KEY   - LiteLLM proxy URL and virtual key
        AWS_REGION                   - cn-north-1 or cn-northwest-1

    See frameworks/settings.py for the full list.
    """

    def __init__(self, group_id: str, settings: Settings | None = None):
        settings = settings or get_settings()

        if not settings.neptune_endpoint:
            raise ValueError("GRAPHITI_NEPTUNE_ENDPOINT must be set")
        if not settings.opensearch_endpoint:
            raise ValueError("GRAPHITI_OPENSEARCH_ENDPOINT must be set")

        neptune_uri = settings.neptune_endpoint
        if not neptune_uri.startswith("neptune-db://"):
            # Neptune Analytics is not available in China regions; only Neptune Database
            neptune_uri = "neptune-db://" + _parse_endpoint(neptune_uri)

        self.driver = NeptuneDriver(
            host=neptune_uri,
            port=settings.neptune_port,
            aoss_host=_parse_endpoint(settings.opensearch_endpoint),
            aoss_port=settings.opensearch_port,
        )
        # Replace the driver's AOSS-signed client with one for a managed domain ("es").
        # The driver keeps calling it "aoss_client"; only index CRUD/search/bulk are used,
        # which behave identically on a managed domain.
        self.driver.aoss_client = build_opensearch_client(settings)
        # graphiti_core expects _database on all drivers but NeptuneDriver doesn't set it.
        if not hasattr(self.driver, '_database'):
            self.driver._database = ""

        self.client = Graphiti(
            graph_driver=self.driver,
            llm_client=build_llm_client(settings),
            embedder=build_embedder(settings),
            cross_encoder=build_cross_encoder(settings),
        )
        self.group_id = group_id

    async def reset(self):
        await self.driver.delete_aoss_indices()
        await self.driver._delete_all_data()
        await self.client.build_indices_and_constraints()

    def index_episode(self, result) -> int:
        """Write an add_episode() result to the OpenSearch BM25 indices.

        graphiti-core 0.31.2's Neptune driver only indexes community nodes, so without
        this the full-text leg of search (node_name_and_summary, edge_name_and_fact,
        episode_content) never finds anything. Documents are keyed by uuid, so
        re-indexing a node whose summary changed overwrites the old document.
        """
        def docs(items, fields):
            return [{"uuid": i.uuid, "group_id": i.group_id, **{f: getattr(i, f) or "" for f in fields}}
                    for i in items]

        d = self.driver
        return (d.save_to_aoss("node_name_and_summary", docs(result.nodes, ["name", "summary"]))
                + d.save_to_aoss("edge_name_and_fact", docs(result.edges, ["name", "fact"]))
                + d.save_to_aoss("episode_content",
                                 docs([result.episode], ["content", "source_description"])))

    async def ensure_indices(self) -> list[str]:
        """Create missing OpenSearch indices.

        Unlike driver.build_indices_and_constraints() this does not sleep 60s afterwards
        (that wait exists for AOSS; managed-domain index creation is synchronous).
        """
        client = self.driver.aoss_client
        created = []
        for index in aoss_indices:
            if not client.indices.exists(index=index['index_name']):
                client.indices.create(index=index['index_name'], body=index['body'])
                created.append(index['index_name'])
        if created:
            logger.info("Created OpenSearch indices: %s", created)
        return created


class _LoopThread:
    """One long-lived event loop in a daemon thread.

    The async OpenAI/httpx clients inside Graphiti bind their connection pool to the
    loop they first ran on, so calling asyncio.run() per request (new loop each time)
    eventually fails with "Event loop is closed". Sync callers (Strands tools,
    Streamlit, MCP) submit coroutines to this loop instead.
    """

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="graphiti-loop", daemon=True)
        self._thread.start()

    def run(self, coro, timeout: float | None = None):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)


_loop_thread: _LoopThread | None = None
_loop_lock = threading.Lock()


def run_sync(coro, timeout: float | None = None):
    """Run a coroutine on the shared background loop and wait for the result."""
    global _loop_thread
    with _loop_lock:
        if _loop_thread is None:
            _loop_thread = _LoopThread()
    return _loop_thread.run(coro, timeout)
