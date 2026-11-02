# Graphiti Agent Memory on AWS China Regions (model-agnostic via LiteLLM)

[中文文档](README.zh-CN.md)

This sample deploys [Graphiti](https://github.com/getzep/graphiti), a temporal knowledge-graph memory for AI agents, in the AWS China regions **cn-north-1 (Beijing)** and **cn-northwest-1 (Ningxia)**. It is a port of the Graphiti part of [aws-samples/amazon-neptune-generative-ai-samples `solutions/agent-memory`](https://github.com/aws-samples/amazon-neptune-generative-ai-samples/tree/main/solutions/agent-memory).

Amazon Bedrock is not available in the China regions. So **every model call** (Graphiti's entity and fact extraction, embeddings, reranking and the chat agent) goes through an **OpenAI-compatible gateway, the [LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy)**. The application only knows five aliases. Which model sits behind each alias is decided in one YAML file. You can switch between DeepSeek, Qwen (Alibaba Cloud Model Studio / DashScope), Kimi, GLM, Doubao, SiliconFlow, a SageMaker endpoint or a vLLM server in your VPC without touching application code.

The sample deploys:

- **Amazon Neptune Database** as the graph store.
- **Amazon OpenSearch Service managed domain** as the full-text (BM25) index. It replaces OpenSearch Serverless, which is not available in the China regions.
- **Amazon ECS on AWS Fargate**, running the LiteLLM proxy, a Streamlit chat demo (a Strands agent with memory tools) and an MCP server.
- **One AWS CDK (Python) app** that creates all of the above, plus the VPC, Cloud Map, an internal ALB and AWS Secrets Manager secrets.

> [!IMPORTANT]
> **Disclaimer:** This sample is for demonstration and proof-of-concept purposes only. It is **not intended for production use** as is. Before using it with real users or real data, review and harden it for your own security, availability, compliance and data-residency requirements (see [Security and compliance](#security-and-compliance)). You are responsible for the cost of the AWS resources and third-party model APIs it uses, and for complying with the terms of any third-party model provider you configure.

---

## Contents

- [When to use Graphiti](#when-to-use-graphiti)
- [Architecture](#architecture)
- [What changed from the global sample](#what-changed-from-the-global-sample)
- [Deploy](#deploy)
- [Use](#use)
- [Configuration reference](#configuration-reference)
- [Choosing models](#choosing-models)
- [Sizing and cost](#sizing-and-cost)
- [Security and compliance](#security-and-compliance)
- [Operations](#operations)
- [Troubleshooting](#troubleshooting)
- [Clean up](#clean-up)
- [Local development and tests](#local-development-and-tests)
- [Known limitations](#known-limitations)

---

## When to use Graphiti

Agent memory is about letting an agent remember things **across sessions**: who the user is, what they prefer, what was decided, and what has changed since. Graphiti stores this as a **temporal knowledge graph**:

- Every message you add is called an *episode*. An LLM turns each episode into entities (people, places, products) and facts, which are the edges between entities, for example "Li Lei — PREFERS — window seat".
- Entities are deduplicated, so "张伟" and "Zhang Wei" end up as one node.
- Every fact has a validity interval (`valid_at` / `invalid_at`). When a new message contradicts an old fact, Graphiti **invalidates** the old fact instead of deleting it. You can still answer "what does the user prefer now?" and also "what did they prefer in March?".
- Retrieval is hybrid: BM25 (OpenSearch), vector similarity and graph traversal, fused with RRF and optionally reranked by a cross-encoder.
- Ingestion is incremental. Each episode updates the graph on its own; there is no batch re-indexing like in GraphRAG-style pipelines.

### Graphiti is a good fit when

| Signal | Why Graphiti helps |
|---|---|
| Facts **change over time** (preferences, status, owners, addresses, plans) | Old facts are invalidated with timestamps instead of coexisting with new ones, so the agent does not retrieve contradictory memories |
| **Relationships** matter: "who reports to whom", "which customer uses which product", "which supplier ships to which port" | Facts are edges between deduplicated entities, so multi-hop questions and graph traversal work |
| Memory is built **continuously from conversations or events**, not from a fixed document set | Episodes are ingested one at a time; no re-indexing |
| You need **per-user / per-tenant isolation** | Every node and edge carries a `group_id` (this sample uses the user ID) and searches are filtered by it |
| You need **explainability or audit**: where did this memory come from, and when was it true? | Facts link back to the episodes they were extracted from and carry validity times |
| Users mix **Chinese and English** or use several names for the same thing | LLM-based entity resolution merges aliases |

### Consider something simpler when

| Situation | Better choice |
|---|---|
| Answering questions over a **static document set** (manuals, policies, knowledge base) | Plain RAG: chunk, embed, OpenSearch k-NN. Cheaper, faster, no graph to maintain |
| You only need the **current conversation's** context | The agent's own message history, or a rolling summary |
| Memories are **independent snippets** ("user likes tea") that never contradict each other and have no relationships | A vector store with metadata filters, or a lightweight memory library such as Mem0 on top of OpenSearch |
| **High write throughput or tight write latency** (thousands of events per second, writes in under 1 s) | Graphiti makes several LLM calls per episode (extraction, deduplication, contradiction check). In our tests one `add_memory` took **7–10 s** and one `search_memory` about **0.5 s**. Queue writes asynchronously, or use a non-LLM store |
| **Very tight LLM budget** | Every write costs several LLM calls; reads cost only an embedding (plus a rerank if enabled) |
| You cannot operate stateful services | This sample runs Neptune and OpenSearch. Both are managed services, but they still have capacity, backups and cost |

**Rule of thumb:** if the hard part of your problem is *"what is true now, how did it change, and how are these things connected"*, use Graphiti. If the hard part is *"find the paragraph that answers this question"*, use RAG.

---

## Architecture

```
                 ┌──────────────────────────── VPC (2 AZs) ────────────────────────────┐
  Users (VPN / ──▶ ALB (internal by default) ──▶ app :8501  Streamlit + Strands agent    │
  DX / bastion)  │                                                                      │
  MCP clients ───▶ mcp.agent-memory.local:8000 ─▶ mcp :8000  MCP server (streamable-http)│
  in the VPC     │                                   │                                  │
                 │          Cloud Map namespace: agent-memory.local                     │
                 │                                   ├──▶ litellm :4000 ─NAT─▶ DeepSeek / DashScope / Kimi / ...
                 │                                   │       (2 tasks)     └─▶ SageMaker / vLLM in region (optional)
                 │                                   ├──▶ Neptune Database :8182   (IAM auth, isolated subnets)
                 │                                   └──▶ OpenSearch domain :443   (IAM / SigV4 "es", isolated subnets)
                 └──────────────────────────────────────────────────────────────────────┘
```

**Write path (`add_memory`):** the episode text goes to Graphiti. Through the `graphiti-llm` alias, Graphiti extracts entities and facts. Each name and fact is embedded through the `embedding` alias. Graphiti then resolves duplicates and checks for contradictions against existing facts (`graphiti-small` / `graphiti-llm`). Finally, nodes, edges and vectors are written to Neptune. After `add_episode` returns, `GraphitiBackend.add()` calls `GraphitiDemo.index_episode()`, which writes the entity names and summaries, the facts and the episode text to the OpenSearch BM25 indices (`node_name_and_summary`, `edge_name_and_fact`, `episode_content`). graphiti-core 0.31.2's Neptune driver only writes community nodes to OpenSearch itself. If you call `GraphitiDemo.client.add_episode()` directly, call `index_episode(result)` yourself afterwards, otherwise BM25 search does not see that data.

**Read path (`search_memory`):** the query is embedded. Graphiti runs BM25 on OpenSearch and cosine similarity over vectors stored on Neptune, filtered by `group_id`. The results are fused with RRF and returned as facts with `valid_at` / `invalid_at`. If `RERANK_MODEL` is set, Graphiti's cross-encoder recipes call `/rerank` on the gateway.

**Model aliases** (defined in `litellm/config.yaml`):

| Alias | Used for | Requirement |
|---|---|---|
| `graphiti-llm` | Entity / fact extraction, contradiction checks | Must support `response_format: {"type": "json_object"}` |
| `graphiti-small` | Lighter Graphiti calls (deduplication, summaries) | JSON output; a cheaper or faster model is fine |
| `agent-llm` | The Strands chat agent | Must support OpenAI-style tool calling |
| `embedding` | Embeddings for names, facts and queries | Output dimension **must equal** `embeddingDim` (default 1024) |
| `rerank` | Optional cross-encoder via `POST /rerank` (Cohere format) | Leave `rerankModel` empty to disable |

---

## What changed from the global sample

| | Global sample | This sample (China regions) |
|---|---|---|
| LLM | Bedrock via litellm SDK (`bedrock/` prefix) | LiteLLM **proxy**, OpenAI protocol, aliases only |
| Embedding | Bedrock Titan v2 (boto3) | `embedding` alias. Requests are batched to respect provider limits (DashScope 10, SiliconFlow 32); a dimension mismatch fails loudly instead of truncating vectors |
| Reranker | `OpenAIRerankerClient` wrapping `LiteLLMClient`. This does not actually work: the wrapper has no `.chat.completions`, which goes unnoticed only because the default RRF search never calls the reranker, and the `logit_bias` token IDs only fit OpenAI tokenizers | Gateway `/rerank` (Cohere format: qwen3-rerank, bge-reranker, Cohere, Jina…), or a passthrough when disabled |
| Full-text index | OpenSearch Serverless (SigV4 `aoss`) | OpenSearch managed domain (SigV4 `es`); `aoss` and basic auth remain selectable |
| BM25 indexing | Left to graphiti-core's Neptune driver, which (in 0.31.2) only writes community nodes to OpenSearch, so BM25 finds no entities, facts or episodes | `GraphitiDemo.index_episode()` writes entity nodes, facts and the episode to OpenSearch after every `add_episode` (called by `GraphitiBackend.add()`; covered by unit tests) |
| Neptune | Database or Analytics | Database only (Neptune Analytics is not available in China regions) |
| Agent model | Strands `BedrockModel` | Strands `OpenAIModel` pointing at the gateway |
| JSON parsing | Strips code fences only | Extracts the first complete JSON value; some models add text before or after the JSON |
| Sync → async | `asyncio.run` per call. A new event loop each time eventually fails with "Event loop is closed", because httpx pools are bound to a loop | One long-lived background event loop |
| Neptune schema | `NeptuneGraph` init fails if the summary API is unavailable (statistics are off on t3/t4g) | Schema load failure is tolerated (Graphiti never uses it) |
| IaC | None (bring your own endpoints) | CDK: VPC, Neptune, OpenSearch, ECS, Cloud Map, ALB, Secrets |

> Graphiti's Neptune driver only uses OpenSearch for BM25 full-text search. Vectors are stored on Neptune nodes and edges, and cosine similarity is computed in Python. Replacing Serverless with a managed domain therefore changes nothing functionally.

### Repository layout

```
frameworks/settings.py          All configuration (environment variables / .env)
frameworks/graphiti_neptune.py  Graphiti wiring: Neptune driver, OpenSearch ("es") and BM25 indexing, gateway LLM / embedding / rerank
frameworks/memory_backend.py    search / add API (same shape as the global sample)
tools/memory_tool.py            Strands agent with search_memory / add_memory tools
examples/                       Agent personas (default, travel_assistant) - add your own here
app.py                          Streamlit chat UI
mcp_server.py                   MCP server (stdio or streamable-http)
litellm/config.yaml             Alias → real model mapping (DeepSeek + DashScope by default, many alternatives commented)
infra/                          CDK app (Python)
scripts/mirror_images.sh        Copy the LiteLLM and Python base images into ECR in the China region
scripts/build_app_image.sh      Build and push the app image
scripts/run_in_vpc.sh           Run a one-off command inside the VPC as a Fargate task (smoke test, admin scripts)
scripts/smoke_test.py           Post-deploy check of every component
scripts/testing/                Gateway contract e2e test (e2e_gateway_local.py) and its test-only gateway config (litellm-bedrock-test.yaml)
tests/                          Unit tests and CDK template assertions (no AWS calls)
```

---

## Deploy

Deployment takes about 30–40 minutes, of which the CDK deploy is about 17 minutes (Neptune and OpenSearch dominate).

### Prerequisites

- An AWS China account (operated by Sinnet or NWCD), and an IAM principal that can create VPC, Neptune, OpenSearch, ECS, Elastic Load Balancing, Cloud Map, IAM, Secrets Manager, CloudWatch Logs, ECR and Lambda resources. The stack contains one Lambda function: the CDK custom resource behind `@aws-cdk/aws-ec2:restrictDefaultSecurityGroup`, which removes the rules of the VPC's default security group.
- AWS CLI v2 configured with a China-region profile (`aws configure --profile cn`).
- Node.js 18+ (for `npx aws-cdk@2`) and Python 3.12.
- Tools to get images into ECR:
  - to mirror the LiteLLM and Python base images: docker, or [skopeo](https://github.com/containers/skopeo) (no daemon needed);
  - to build the app image: docker, or an AWS CodeBuild project (see [Building without docker](#building-without-docker)). skopeo cannot build images.
- An API key for at least one model provider reachable from the China regions. The default config uses **DeepSeek** for chat and extraction, and **Alibaba Cloud Model Studio (DashScope)** for embedding, rerank and fallback. Or bring your own SageMaker / vLLM endpoint.

Set these for the whole session. `AWS_REGION` matters: with only `AWS_PROFILE` set, CDK may resolve credentials against a global region and fail.

```bash
export AWS_PROFILE=cn
export AWS_REGION=cn-northwest-1          # or cn-north-1
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export REGISTRY=$ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com.cn
```

### Step 1 — Pick your models (`litellm/config.yaml`)

Keep the alias names (`model_name`) unchanged; only edit `litellm_params`.

- `embedding` must return exactly `embeddingDim` dimensions (default **1024**). DashScope `text-embedding-v4` with `dimensions: 1024` and `bge-m3` both work. **Once data is written, the dimension and the model are fixed.** Switching embedding models means re-ingesting everything.
- `graphiti-llm` must support `json_object` output. `agent-llm` must support tool calling. DeepSeek-V3.x, Qwen3 / qwen-plus, GLM-4.5+ and Kimi K2 all do.
- The same alias listed twice forms a load-balanced or failover group (the default config pairs DeepSeek with qwen-plus).
- Every key referenced as `os.environ/XXX`, except `LITELLM_MASTER_KEY` (the stack sets it), must appear in the CDK context `providerKeyNames` (Step 3).
- For a setup where **data never leaves your AWS account**, use the SageMaker or in-VPC vLLM examples at the end of the file. For SageMaker, also pass `-c sagemakerEndpointArns='["arn:aws-cn:sagemaker:..."]'` (a JSON list or a comma-separated string).

See [Choosing models](#choosing-models) for guidance.

### Step 2 — Put the images into ECR

Fargate in the China regions cannot reliably pull from `ghcr.io` or `docker.io`, so every image must be in your own ECR.

```bash
# 2a. Mirror LiteLLM and the Python base image (run where ghcr.io is reachable).
#     USE_SKOPEO=1 copies registry-to-registry without a docker daemon.
LITELLM_TAG=main-stable USE_SKOPEO=1 ./scripts/mirror_images.sh

# 2b. Build the app image from the mirrored base, using a domestic PyPI mirror
TAG=v1 USE_MIRROR_BASE=1 PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
  ./scripts/build_app_image.sh
```

Both scripts read `REGION`, falling back to `AWS_REGION` and then `cn-northwest-1`, so the `AWS_REGION` exported above is enough.

For production, pin `LITELLM_TAG` to a specific release (for example `v1.79.x-stable`) instead of `main-stable`.

#### Building without docker

If your workstation has no docker, run `scripts/build_app_image.sh` in an AWS CodeBuild project in the same China region. Use an `aws/codebuild/standard:7.0` image with **privileged mode** enabled, a role that can push to ECR, the repository as the source, and this buildspec (CodeBuild sets `AWS_REGION`, which the script uses):

```yaml
version: 0.2
phases:
  build:
    commands:
      - TAG=v1 USE_MIRROR_BASE=1 PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ bash scripts/build_app_image.sh
```

This is how the sample was built for its own cn-northwest-1 test.

### Step 3 — Deploy with CDK

```bash
# 3a. Bootstrap once per account/region. Run it OUTSIDE infra/: inside infra/ CDK
#     synthesizes the app first and fails because the image parameters are missing.
#     Use aws://, not aws-cn://, even in China regions.
(cd /tmp && npx -y aws-cdk@2 bootstrap aws://$ACCOUNT/$AWS_REGION)

# 3b. New accounts only: OpenSearch needs its service-linked role before the first
#     VPC domain. Otherwise the first deploy fails. "already exists" is fine.
aws iam create-service-linked-role --aws-service-name opensearchservice.amazonaws.com || true

# 3c. Deploy
cd infra
python3 -m venv .venv && . .venv/bin/activate && pip install aws-cdk-lib==2.275.0 constructs==10.8.1
npx -y aws-cdk@2 deploy \
  -c region=$AWS_REGION -c account=$ACCOUNT \
  -c litellmImage=$REGISTRY/mirror/litellm:main-stable \
  -c appImage=$REGISTRY/graphiti-agent-memory:v1
```

For a low-cost PoC, add:

```bash
  -c neptuneInstanceClass=db.t4g.medium \
  -c opensearchInstanceType=t3.medium.search -c opensearchDataNodes=1 \
  -c retainData=false
```

(This is the size the sample was tested with. OpenSearch reports `yellow` with one node because replicas cannot be placed; that is expected.)

When the deploy finishes, note the stack outputs: `AppUrl`, `McpUrl`, `LiteLLMUrl`, `ProviderKeysSecretArn`, `ClusterName`, and the others.

### Step 4 — Store provider keys and restart LiteLLM

The stack creates a placeholder secret, with a `REPLACE_ME` value for each name in `providerKeyNames`. Put the real keys in it and roll the LiteLLM service:

```bash
STACK=GraphitiAgentMemoryCn
out() { aws cloudformation describe-stacks --stack-name $STACK \
  --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text; }

aws secretsmanager put-secret-value --secret-id "$(out ProviderKeysSecretArn)" \
  --secret-string '{"DEEPSEEK_API_KEY":"sk-...","DASHSCOPE_API_KEY":"sk-..."}'

CLUSTER=$(out ClusterName)
LITELLM_SVC=$(aws ecs list-services --cluster $CLUSTER \
  --query 'serviceArns[?contains(@, `LiteLLM`)]|[0]' --output text)
aws ecs update-service --cluster $CLUSTER --service $LITELLM_SVC --force-new-deployment
aws ecs wait services-stable --cluster $CLUSTER --services $LITELLM_SVC
```

To manage the secret yourself instead (for example with rotation), create it beforehand and pass `-c providerSecretArn=<arn>`.

### Step 5 — Smoke test

```bash
cd ..   # repository root
./scripts/run_in_vpc.sh            # runs scripts/smoke_test.py as a one-off Fargate task
```

Expected output (abridged, from the cn-northwest-1 test run):

```
[PASS] gateway graphiti-llm json_object (2.9s)
[PASS] gateway graphiti-small json_object (0.5s)
[PASS] gateway agent-llm tool calling (1.4s) {"city": "Beijing"}
[PASS] gateway embedding (0.8s) dim=1024
[PASS] gateway rerank (1.5s)
[PASS] neptune openCypher (0.1s) [{'x': 1}]
[PASS] opensearch cluster health (0.1s) yellow
[PASS] opensearch indices (0.0s) created=[]
[PASS] add_memory (9.8s) {'status': 'added', 'user_id': 'smoke-a5d56eda'}
[PASS] search_memory (0.7s) [{'fact': '冒烟测试用户喜欢喝乌龙茶。', 'valid_at': '...', 'invalid_at': None}]
[PASS] cleanup (0.1s)
>> exit=0
```

The smoke test checks, in order: every gateway alias (JSON mode, tool calling, embedding dimension, rerank), then Neptune openCypher, then OpenSearch health and indices, then one real memory write and search, then cleanup of the test user's nodes and edges in Neptune. The cleanup does not touch OpenSearch, so the test user's BM25 documents remain (see [Deleting a user's memory](#deleting-a-users-memory)). Each failure line names the component at fault; see [Troubleshooting](#troubleshooting).

`scripts/run_in_vpc.sh` is the recommended way; it needs neither ECS Exec nor the Session Manager plugin. ECS Exec is off by default. To use it instead, deploy with `-c enableExecuteCommand=true` (this enables it on the app service only), install the Session Manager plugin, and run:

```bash
aws ecs execute-command --cluster <ClusterName> --task <task-id> --container app --interactive \
  --command "sh -c 'LLM_API_KEY=sk-\$LLM_API_KEY_RAW python scripts/smoke_test.py'"
```

The `\$` keeps your local shell from expanding the variable; the container's shell does it.

---

## Use

### 1. Streamlit chat demo

`AppUrl` is an **internal** ALB, because public ports 80/443 in the China regions require an ICP filing. Reach it in one of these ways:

- From a network connected to the VPC (Client VPN, Direct Connect, Transit Gateway). The ALB (port 80) and MCP (port 8000) security group rules admit only the VPC CIDR by default. Client VPN works as is, because it NATs client traffic to VPC addresses. For Direct Connect or Transit Gateway, deploy with `-c albIngressCidr=<on-prem CIDR>` for the UI and `-c mcpIngressCidr=<on-prem CIDR>` for MCP. Each takes one CIDR, which replaces the VPC CIDR.
- Through an SSM port forward via any EC2 instance in the VPC:
  ```bash
  aws ssm start-session --target <instance-id> \
    --document-name AWS-StartPortForwardingSessionToRemoteHost \
    --parameters host=<AppUrl host>,portNumber=80,localPortNumber=8501
  # then open http://localhost:8501
  ```
- If you have an ICP filing, deploy with `-c publicAlb=true -c albIngressCidr=<your office CIDR>`, and put HTTPS and authentication in front of it.

Pick a user in the sidebar and chat. The agent calls `search_memory` at the start of a conversation and `add_memory` when you share facts or preferences. Each tool call and its result are shown under the answer (`st.info` / `st.code`), so you can see exactly what was stored and retrieved. Every browser session gets its own agent (`create_memory_agent()`), so conversations in different tabs don't mix.

A quick test of cross-session memory:

1. As `User1`: "I'm travelling to Hangzhou next week. I prefer hotels near West Lake, a high floor, and I'm allergic to peanuts."
2. Switch to `User2` and back to `User1`. This clears the conversation history.
3. Ask: "Book me a hotel and suggest dinner." The answer should use the West Lake, high-floor and no-peanuts preferences recalled from memory.
4. Later, say "Actually I now prefer hotels near the train station". The old preference is invalidated (`invalid_at` set), and the new one becomes the current fact.

Switch the persona with `-c agentConfig=travel_assistant` at deploy time, or with `AGENT_CONFIG` locally.

### 2. MCP server (for Claude Code, Kiro, Cursor, or any MCP client)

The stack runs the MCP server with the streamable-http transport at `http://mcp.agent-memory.local:8000/mcp` (output `McpUrl`). It exposes two tools:

| Tool | Arguments | Description |
|---|---|---|
| `add_memory` | `content`, `user_id` | Store a piece of information for a user |
| `search_memory` | `query`, `user_id` | Return the user's relevant facts with validity times |

`user_id` must match `^[a-zA-Z0-9_-]+$` (see [Python API](#3-python-api-embed-memory-in-your-own-agent)).

The endpoint is **reachable only inside the VPC and is not authenticated**. Any client that can reach it can read and write any user's memory. Connect from a host in the VPC (or over VPN, with Cloud Map DNS resolvable; for Direct Connect / Transit Gateway set `mcpIngressCidr`, see [Streamlit chat demo](#1-streamlit-chat-demo)):

```bash
# Claude Code
claude mcp add --transport http agentmemory http://mcp.agent-memory.local:8000/mcp
```

```jsonc
// Kiro / Cursor / other clients (mcp.json)
{ "mcpServers": { "agentmemory": { "url": "http://mcp.agent-memory.local:8000/mcp" } } }
```

To expose it outside the VPC, put an authenticating layer in front of it, such as API Gateway with a Lambda authorizer, or an ALB with OIDC.

You can also run the MCP server **locally over stdio**, from a machine with network access to Neptune, OpenSearch and LiteLLM, with a filled-in `.env`:

```jsonc
{ "mcpServers": { "agentmemory": {
    "command": "uv", "args": ["run", "--directory", "/path/to/repo", "python", "mcp_server.py"] } } }
```

### 3. Python API (embed memory in your own agent)

```python
from frameworks.memory_backend import get_memory_backend

memory = get_memory_backend()            # reads configuration from env / .env

memory.add("李雷下周去杭州出差，喜欢靠窗座位，对花生过敏。", user_id="lilei")
memory.search("李雷的饮食禁忌", user_id="lilei")
# -> [{'fact': '李雷对花生过敏', 'valid_at': '2026-11-02T06:35:52+00:00', 'invalid_at': None}, ...]
```

- `user_id` becomes Graphiti's `group_id`. All reads and writes are scoped to it, so use one per end user (or per tenant, if memory is shared within a tenant).
- `user_id` may contain only ASCII letters, digits, `-` and `_` (`^[a-zA-Z0-9_-]+$`, checked by graphiti-core's `validate_group_id`). Email addresses, for example, are rejected; map them to an internal ID first.
- `add()` is synchronous and takes several seconds, because of the LLM calls. In a latency-sensitive service, call it from a background worker or queue (SQS, for example).
- The search results include `invalid_at`. Facts with `invalid_at` set are historical; either filter them out or show them as history.

For full Graphiti features, such as custom entity types, JSON episodes, `search_()` recipes with the cross-encoder, communities and `remove_episode`, use the underlying client:

```python
from frameworks.graphiti_neptune import GraphitiDemo, run_sync
demo = GraphitiDemo(group_id="lilei")
results = run_sync(demo.client.search("hotel preferences", group_ids=["lilei"], num_results=10))

result = run_sync(demo.client.add_episode(...))
demo.index_episode(result)   # required: write the episode's nodes, facts and text to the BM25 indices
```

Always go through `run_sync()` (one shared event loop) rather than calling `asyncio.run()` per call. When you call `demo.client.add_episode()` directly, always follow it with `demo.index_episode(result)`, as `GraphitiBackend.add()` does; otherwise BM25 search does not see that data.

### 4. Strands agent

`tools/memory_tool.py` builds a Strands `Agent` with `OpenAIModel` pointing at the `agent-llm` alias, and two tools that read the current user from `agent.state["user_id"]`:

```python
from tools.memory_tool import memory_agent
memory_agent.state.set("user_id", "lilei")
print(memory_agent("帮我订下周去杭州的酒店"))
```

`memory_agent` is one shared module-level instance, which is fine for a single-conversation script. The agent holds the message history and the current user, so for several conversations create one agent each with `create_memory_agent()`, as `app.py` does per browser session.

### 5. Add your own persona

Copy `examples/default.py` to `examples/my_agent.py` and edit `AGENT_NAME`, `SYSTEM_PROMPT`, `EXTRA_TOOLS` (any Strands `@tool` functions) and `USERS`. Then either register it in `AVAILABLE_AGENTS` in `examples/__init__.py` and use the short name (`-c agentConfig=my_agent`, or `AGENT_CONFIG=my_agent` locally), or skip registering and use the module path (`-c agentConfig=examples.my_agent` / `AGENT_CONFIG=examples.my_agent`). An unregistered short name fails to import. Rebuild the app image and redeploy.

### Deleting a user's memory

Remove every node and edge for one user (for example, for a deletion request) with openCypher. Run it with `scripts/run_in_vpc.sh` or from any host in the VPC:

```python
from frameworks.graphiti_neptune import GraphitiDemo
demo = GraphitiDemo("")
demo.driver.client.query("MATCH (n) WHERE n.group_id = $g DETACH DELETE n", params={"g": "lilei"})
```

The OpenSearch documents for that user still exist until removed with a `delete_by_query` on `group_id` across the four Graphiti indices (`node_name_and_summary`, `edge_name_and_fact`, `episode_content`, `community_name`).

---

## Configuration reference

### CDK context (`-c key=value`, defaults in `infra/cdk.json` and `infra/stacks/agent_memory_stack.py`)

| Key | Default | Description |
|---|---|---|
| `litellmImage` / `appImage` | — (required) | ECR image URIs from Step 2 |
| `region` / `account` | env | Target account and region |
| `stackName` | `GraphitiAgentMemoryCn` | CloudFormation stack name |
| `providerKeyNames` | `["DEEPSEEK_API_KEY","DASHSCOPE_API_KEY"]` | Secret keys injected into LiteLLM as env vars; must match `os.environ/...` in `litellm/config.yaml`. JSON list or comma-separated string, both with `-c` and in `cdk.json` |
| `providerSecretArn` | (create placeholder) | Use an existing Secrets Manager secret (JSON with the keys above) |
| `neptuneInstanceClass` | `db.r6g.large` | `db.t4g.medium` for PoC |
| `neptuneEngineVersion` | region default | |
| `opensearchInstanceType` | `r6g.large.search` | `t3.medium.search` for PoC |
| `opensearchDataNodes` | `2` | 1 for PoC (no zone awareness) |
| `opensearchVersion` | `2.19` | |
| `natGateways` | `1` | 2 for AZ-redundant egress to model providers |
| `embeddingDim` | `1024` | Must match the `embedding` alias. Fixed once data exists |
| `rerankModel` | `rerank` | `""` disables the cross-encoder |
| `agentConfig` | `default` | Persona module in `examples/` |
| `publicAlb` / `albIngressCidr` | `false` / VPC CIDR | `albIngressCidr` is the source CIDR allowed to the ALB on port 80. A public ALB requires an ICP filing and an explicit CIDR |
| `mcpIngressCidr` | VPC CIDR | Source CIDR allowed to the MCP server on port 8000 (set it for Direct Connect / Transit Gateway clients) |
| `enableExecuteCommand` | `false` | Enables ECS Exec on the app service. Not needed by `scripts/run_in_vpc.sh` |
| `retainData` | `true` | When true, Neptune and OpenSearch are both kept on `cdk destroy`, and the Neptune cluster has deletion protection (OpenSearch does not) |
| `sagemakerEndpointArns` | `[]` | Grants the LiteLLM task role `sagemaker:InvokeEndpoint` and `sagemaker:InvokeEndpointWithResponseStream` on these ARNs. JSON list or comma-separated string, both with `-c` and in `cdk.json` |
| `litellmConfig` | `litellm/config.yaml` | Alternate gateway config path (relative to the repo root); used for tests |
| `bedrockAccess` | `false` | Test only: grants the LiteLLM task role `bedrock:InvokeModel`, `bedrock:InvokeModelWithResponseStream` and `bedrock:Rerank` on `*`, for Bedrock-backed test configs in global regions |

### Application environment variables (`frameworks/settings.py`, see `.env.example`)

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` / `LLM_API_KEY` | `http://localhost:4000/v1` / `sk-litellm-local` | Gateway URL and key. The stack injects the LiteLLM master key (`sk-` + the master key secret) |
| `GRAPHITI_LLM_MODEL` / `GRAPHITI_SMALL_MODEL` / `AGENT_MODEL` / `EMBEDDING_MODEL` | alias names | Override only if you renamed aliases |
| `RERANK_MODEL` | `""` (disabled) | Rerank alias; empty uses a passthrough instead of the cross-encoder. The stack sets it from `rerankModel` (`rerank`) |
| `LLM_STRUCTURED_OUTPUT_MODE` | `json_object` | `json_schema` only if every model behind `graphiti-llm` supports it |
| `LLM_MAX_TOKENS` / `LLM_TEMPERATURE` | `8192` / `0` | `LLM_TEMPERATURE` applies to Graphiti's calls only; the Strands agent uses a fixed `0.3` (`tools/memory_tool.py`) |
| `EMBEDDING_DIM` | `1024` | Must equal the embedding model's output |
| `EMBEDDING_BATCH_SIZE` | `10` | Provider limit (DashScope 10, SiliconFlow 32) |
| `GRAPHITI_NEPTUNE_ENDPOINT` / `_PORT` | — / `8182` | `neptune-db://<cluster endpoint>` |
| `GRAPHITI_OPENSEARCH_ENDPOINT` / `_PORT` | — / `443` | Domain endpoint, no scheme |
| `GRAPHITI_OPENSEARCH_SERVICE` | `es` | `aoss` for Serverless in global regions |
| `GRAPHITI_OPENSEARCH_AUTH` | `iam` | `basic` with `_USERNAME` / `_PASSWORD` for a fine-grained-access-control master user |
| `AGENT_CONFIG` | `default` | Persona |
| `AWS_REGION` | `cn-northwest-1` | Falls back to `AWS_DEFAULT_REGION`, then `cn-northwest-1`; used for SigV4 |
| `MCP_TRANSPORT` / `MCP_HOST` / `MCP_PORT` | `stdio` / `0.0.0.0` / `8000` | |

---

## Choosing models

Graphiti's quality depends mostly on the `graphiti-llm` model. It has to follow long extraction prompts and return valid JSON every time.

| Role | Recommended starting point | Notes |
|---|---|---|
| `graphiti-llm` | DeepSeek-V3.2 (`deepseek-chat`), Qwen3 / qwen-plus | Strong instruction following and JSON. Small models (<14B) often miss facts or break JSON |
| `graphiti-small` | qwen-flash, Qwen3-32B, DeepSeek | Called often; latency matters more than depth |
| `agent-llm` | DeepSeek-V3.2, Qwen3, Kimi K2, GLM-4.6 | Must do reliable tool calling |
| `embedding` | DashScope `text-embedding-v4` (1024), `bge-m3` (1024) | Multilingual models handle mixed Chinese and English |
| `rerank` | `qwen3-rerank`, `bge-reranker-v2-m3` | Optional; default search (RRF) does not use it |

**Tested combinations:**

| Where | graphiti-llm / agent-llm | graphiti-small | embedding | rerank | Result |
|---|---|---|---|---|---|
| cn-northwest-1, full stack | DeepSeek V3.2 | Qwen3-32B | Titan v2 (1024) | Cohere Rerank 3.5 | Smoke test 11/11, MCP, agent cross-session recall |
| us-east-2, full stack | Claude (Bedrock) | Claude (Bedrock) | Titan v2 | Cohere Rerank 3.5 | Smoke test 11/11, MCP |
| Local gateway e2e (`scripts/testing/litellm-bedrock-test.yaml`) | Claude Haiku 4.5 (Bedrock) | Claude Haiku 4.5 (Bedrock) | Titan v2 (1024) | Cohere Rerank 3.5 | Extraction and Chinese/English alias merge ("张伟" = "Zhang Wei") |

In the cn-northwest-1 test, the models were Bedrock models in a global region, called across the border through a test-only LiteLLM config (not included in this repository), because no domestic provider keys were available. That config is **for synthetic test data only**. The default `litellm/config.yaml` (DeepSeek + DashScope APIs) uses the same OpenAI-compatible code path but was not exercised with real keys. Run the smoke test after Step 4 to confirm your provider set.

**Language of stored facts:** Graphiti's extraction prompts are in English, so some models write facts in English even when the input is Chinese (one MCP test stored "韩梅梅对花生过敏" as an English sentence). Retrieval still works across languages with a multilingual embedding. If you need facts in the source language, choose a model that tends to keep it (DeepSeek did in the smoke test), or pass `custom_extraction_instructions` / custom prompts through the Graphiti API.

---

## Sizing and cost

Billable resources created by the stack:

| Resource | Default | PoC option |
|---|---|---|
| Neptune Database | 1 × `db.r6g.large` writer, storage and I/O | `db.t4g.medium` |
| OpenSearch domain | 2 × `r6g.large.search`, 50 GB gp3 each | 1 × `t3.medium.search` |
| ECS Fargate | LiteLLM 2 × (1 vCPU, 2 GB), app 1 × (1 vCPU, 2 GB), MCP 1 × (0.5 vCPU, 1 GB) | — |
| NAT gateway | 1, plus data processing | — |
| ALB, Secrets Manager (2 secrets), CloudWatch Logs (task logs 30-day retention), VPC flow logs (CDK default retention, 731 days / 2 years) | | |
| Model API usage | Billed by your provider, not AWS | |

Use the [AWS China pricing calculator](https://calculator.amazonaws.cn/) for your region. Neptune and OpenSearch are the largest fixed costs; stop or destroy the stack when you are not using it.

Model usage scales with writes: each `add_memory` makes several `graphiti-llm` / `graphiti-small` calls plus embeddings. Reads make one embedding call (plus a rerank, if enabled).

For production, consider:

- Neptune: add a read replica in a second AZ for HA.
- OpenSearch: 2+ data nodes across AZs (the default), and dedicated master nodes for large domains.
- `natGateways=2`.
- Scaling the LiteLLM service, which is stateless, with request volume.

---

## Security and compliance

- **Network:** Neptune and OpenSearch are in isolated subnets with no internet route. Only ECS tasks can reach them (security groups). The ALB is internal by default. Egress to model providers goes through the NAT gateway.
- **AuthN to data stores:** Neptune uses IAM database authentication. OpenSearch uses IAM (SigV4); only the app/MCP task role is granted read/write. Neither store has a username or password. Both are encrypted at rest; OpenSearch enforces TLS 1.2 and node-to-node encryption.
- **Secrets:** Provider API keys and the LiteLLM master key live in Secrets Manager and are injected as ECS secrets, never baked into images or task definitions.
- **MCP and UI are unauthenticated.** Anyone who can reach them can read or write any user's memory, because `user_id` is a plain parameter. Before production, add authentication, such as OIDC on the ALB or an authenticating proxy for MCP, and derive `user_id` from the authenticated identity instead of trusting the client.
- **LiteLLM** runs with the master key only, and no database. For per-application virtual keys, budgets and spend tracking, add PostgreSQL (`DATABASE_URL`), and make sure the image contains the Prisma engines (downloading Prisma binaries from inside China may fail).
- **Data residency:** calling models outside mainland China (global Bedrock, OpenAI, etc.) is a cross-border data transfer and needs a compliance assessment first. The default config only lists domestic providers and in-region self-hosted options. To keep all data inside your AWS account, use SageMaker or vLLM in the VPC.
- **Memory content is user data.** It may contain personal information extracted by the LLM. Define retention, provide a deletion path (see [Deleting a user's memory](#deleting-a-users-memory)), and review what your prompts ask the agent to store.

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for how to report security issues.

---

## Operations

- **Logs:** CloudWatch Logs groups `<stack>-LiteLLMTask...`, `<stack>-AppTask...`, `<stack>-McpTask...` (30-day retention). VPC flow logs go to their own log group with the CDK default retention of 731 days (2 years). LiteLLM logs every upstream error with the provider message, so check it first when a model call fails.
- **Change models:** edit `litellm/config.yaml`, then run `cdk deploy` again. The config is embedded in the LiteLLM task definition, so a deploy rolls the service. Re-run the smoke test afterwards.
- **Rotate provider keys:** `put-secret-value`, then force a new deployment of the LiteLLM service (Step 4).
- **Upgrade the app:** build a new tag, then `cdk deploy -c appImage=...:v2`. Deployments use a circuit breaker with rollback.
- **Upgrade graphiti-core:** it is pinned to `0.31.2` in `pyproject.toml`. These depend on its internals: the `NeptuneDriver._run_query` patch, the `_database` workaround, and `GraphitiDemo.index_episode()`, which fills the BM25 indices because the 0.31.2 Neptune driver only writes community nodes to OpenSearch. Run `pytest` and the smoke test before bumping, and check whether the new driver indexes nodes, edges and episodes itself.
- **Backups:** Neptune keeps 7 days of automated backups. OpenSearch takes automated hourly snapshots. The OpenSearch indices hold text copies of Neptune data, but this sample includes no script to rebuild them from Neptune, so rely on the snapshots.

---

## Troubleshooting

These are all real failures seen while building and testing this sample.

| Symptom | Cause | Fix |
|---|---|---|
| `cdk bootstrap aws-cn://...` rejects the environment | CDK environments always use `aws://`, even in China | `aws://<account>/cn-northwest-1` |
| `cdk bootstrap` in `infra/` fails with "Pass -c litellmImage=..." | Bootstrap synthesizes the app in the current directory | Run bootstrap from another directory (`cd /tmp`) |
| CDK cannot find credentials or resolves the wrong partition | `AWS_REGION` unset or set to a global region | `export AWS_REGION=cn-northwest-1` along with `AWS_PROFILE` |
| First deploy fails creating the OpenSearch domain with a service-linked role error | New account without the OpenSearch service-linked role | `aws iam create-service-linked-role --aws-service-name opensearchservice.amazonaws.com`, then redeploy |
| Tasks stuck in `PENDING` / `CannotPullContainerError` | Image not in China ECR, or wrong URI (`.amazonaws.com.cn`) | Step 2; check the image URIs passed to CDK |
| Smoke test `[FAIL] gateway ...: 401` | Provider key still `REPLACE_ME`, or LiteLLM not restarted | Step 4 |
| `[FAIL] gateway ...: Access to Anthropic models is not allowed from unsupported countries` | Anthropic models reject calls from China IPs, even via global Bedrock | Use another model family (DeepSeek, Qwen, ...) |
| `[FAIL] gateway embedding: got dims {1536}` | Embedding model's native dimension ≠ `embeddingDim` | Set `dimensions:` in the alias, or deploy with the matching `embeddingDim` (before any data is written) |
| `add_memory` fails with `json.decoder.JSONDecodeError` | `graphiti-llm` model doesn't honour JSON mode | Use a stronger model; check LiteLLM logs for the raw response |
| App starts, then Neptune `AccessDeniedException` on `GetGraphSummary` | IAM policy missing `neptune-db:GetGraphSummary` | Already granted by the stack; check custom roles |
| Log line "Neptune schema summary unavailable, continuing without it" | Neptune statistics are disabled (t3/t4g) | Harmless warning |
| MCP container exits with `ImportError: FastMCP` | `mcp` 2.x renamed the API | `mcp[cli]==1.30.0` is pinned in `pyproject.toml`; stay on 1.x |
| MCP returns `421 Misdirected Request` | FastMCP DNS-rebinding protection when constructed with a localhost host | Already handled in `mcp_server.py` (host passed at construction) |
| "Event loop is closed" after a few requests | `asyncio.run()` per call with shared httpx clients | Use `run_sync()` from `frameworks/graphiti_neptune.py` |
| OpenSearch health `yellow` | Single data node, so replicas can't be allocated | Expected for PoC; use 2+ nodes in production |

---

## Clean up

```bash
cd infra
npx -y aws-cdk@2 destroy -c region=$AWS_REGION -c account=$ACCOUNT \
  -c litellmImage=$REGISTRY/mirror/litellm:main-stable -c appImage=$REGISTRY/graphiti-agent-memory:v1
```

- With the default `retainData=true`, **the Neptune cluster and OpenSearch domain are kept** and keep billing. Neptune also has deletion protection enabled (OpenSearch does not). The stack delete will then likely fail on the subnets and security groups that the retained resources still use. Disable deletion protection on the Neptune cluster, delete the cluster (with its instance) and the domain in the console or CLI, then run `cdk destroy` again. For throwaway stacks, deploy with `-c retainData=false` before destroying.
- Delete the ECR repositories `mirror/litellm`, `mirror/python`, `graphiti-agent-memory`, and any CodeBuild project or S3 bucket you created for image builds.
- Delete the leftover CloudWatch log groups if you don't need them.
- The `CDKToolkit` bootstrap stack can stay for future CDK deploys, or be deleted with its S3 bucket.

---

## Local development and tests

```bash
uv venv && uv pip install -r requirements.txt pytest==9.1.1 pytest-asyncio==1.4.1 \
  aws-cdk-lib==2.275.0 constructs==10.8.1     # locked runtime set + pinned test/CDK tools
.venv/bin/python -m pytest   # unit tests + CDK template assertions; no AWS calls
```

**Gateway contract e2e** runs real Graphiti extraction, search and rerank through a local LiteLLM proxy. It uses embedded FalkorDB instead of Neptune, so it needs no China-region resources:

```bash
uv pip install "graphiti-core[falkordb]==0.31.2" falkordblite==0.11.0 "litellm[proxy]==1.102.4"
# point the aliases at any models you have; the example config uses global-region Bedrock
# (Claude Haiku 4.5, Titan v2, Cohere Rerank 3.5), so it needs AWS credentials for it
LITELLM_MASTER_KEY=sk-test litellm --config scripts/testing/litellm-bedrock-test.yaml --port 4011 &
LLM_BASE_URL=http://localhost:4011/v1 LLM_API_KEY=sk-test RERANK_MODEL=rerank \
  python scripts/testing/e2e_gateway_local.py
```

To run the app against a deployed stack from your workstation, you need network access to the VPC (VPN, or a bastion with port forwards for Neptune 8182, OpenSearch 443 and LiteLLM 4000), and also:

- ingress rules from your source (VPN or bastion) on `NeptuneSg` (8182), `OpenSearchSg` (443) and `TasksSg` (4000); by default they only admit the ECS tasks;
- IAM permissions equivalent to `AppTaskRole`: the `neptune-db:*` data actions on the cluster and `es:ESHttp*` on the domain;
- `LLM_API_KEY=sk-<value of the LiteLLM master key secret>` (output `LiteLLMMasterKeySecretArn`).

Copy `.env.example` to `.env`, fill in the stack outputs, then run `.venv/bin/streamlit run app.py`. Otherwise run your script inside the VPC with `scripts/run_in_vpc.sh "python my_script.py"` (the script must be in the app image).

---

## Known limitations

- Graphiti's Neptune vector search pulls candidate embeddings into Python and computes cosine similarity there. It slows down when a single `group_id` holds a very large graph. This comes from graphiti_core's implementation.
- `add_memory` is synchronous and takes seconds; the sample does not include a write queue.
- The UI and MCP server have no authentication (see [Security and compliance](#security-and-compliance)).
- There is no built-in "forget user" API; see [Deleting a user's memory](#deleting-a-users-memory).
- graphiti-core is pinned to 0.31.2 because of the monkey patches and the BM25 indexing workaround listed in [Operations](#operations).

## Security

This is sample code and is not intended for production use without further review; see the Disclaimer at the top of this file.

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file. Third-party dependencies are listed in [THIRD-PARTY-LICENSES](THIRD-PARTY-LICENSES).
