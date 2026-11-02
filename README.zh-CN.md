# Graphiti Agent Memory —— AWS 中国区部署（通过 LiteLLM 接入任意模型）

[English](README.md)

本示例把 [Graphiti](https://github.com/getzep/graphiti)（面向 AI Agent 的时序知识图谱记忆）部署到 AWS 中国区 **cn-north-1（北京）/ cn-northwest-1（宁夏）**。它改造自 [aws-samples/amazon-neptune-generative-ai-samples `solutions/agent-memory`](https://github.com/aws-samples/amazon-neptune-generative-ai-samples/tree/main/solutions/agent-memory) 中的 Graphiti 部分。

中国区没有 Amazon Bedrock，所以**所有模型调用**（Graphiti 的实体/事实抽取、embedding、rerank、对话 Agent）都通过 **OpenAI 兼容网关 [LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy)** 发出。应用代码里只出现 5 个 alias，每个 alias 背后接哪个模型由一个 YAML 文件决定。在 DeepSeek、通义千问（阿里云百炼 DashScope）、Kimi、GLM、豆包、SiliconFlow、区域内的 SageMaker endpoint 或 VPC 内的 vLLM 之间切换，都不用改代码。

本示例会部署：

- **Amazon Neptune Database**：图存储。
- **Amazon OpenSearch Service 托管域**：全文（BM25）索引，替代中国区没有的 OpenSearch Serverless。
- **Amazon ECS on AWS Fargate**：运行 LiteLLM proxy、Streamlit 对话 Demo（带记忆工具的 Strands Agent）和 MCP server。
- **一个 AWS CDK（Python）应用**：创建以上全部资源，以及 VPC、Cloud Map、内网 ALB 和 Secrets Manager 密钥。

> [!IMPORTANT]
> **免责声明：** 本示例仅用于演示和概念验证（PoC），**不适合直接用于生产环境**。用于真实用户或真实数据之前，请按你自己的安全、可用性、合规和数据驻留要求进行评审和加固（见[安全与合规](#安全与合规)）。所用 AWS 资源和第三方模型 API 的费用由你承担，你也需要遵守所配置的第三方模型服务商的使用条款。

---

## 目录

- [什么时候选择 Graphiti](#什么时候选择-graphiti)
- [架构](#架构)
- [与 global 版的差异](#与-global-版的差异)
- [部署](#部署)
- [使用](#使用)
- [配置参考](#配置参考)
- [如何选模型](#如何选模型)
- [规格与成本](#规格与成本)
- [安全与合规](#安全与合规)
- [运维](#运维)
- [排障](#排障)
- [清理](#清理)
- [本地开发与测试](#本地开发与测试)
- [已知限制](#已知限制)

---

## 什么时候选择 Graphiti

Agent 记忆要解决的是**跨会话**记住东西：用户是谁、偏好什么、做过什么决定、后来又有什么变化。Graphiti 把这些存成一张**时序知识图谱**：

- 每写入一条消息叫一个 *episode*。LLM 会从中抽取实体（人、地点、产品）和事实；事实是实体之间的边，例如「李雷 —— 偏好 —— 靠窗座位」。
- 实体会去重合并，「张伟」和「Zhang Wei」最终是同一个节点。
- 每条事实都带有效期（`valid_at` / `invalid_at`）。新消息和旧事实矛盾时，Graphiti 会把旧事实**标记为失效**，而不是删除。所以既能回答「用户现在偏好什么」，也能回答「用户三月份时偏好什么」。
- 检索是混合的：BM25（OpenSearch）、向量相似度和图遍历，用 RRF 融合，可选再用 cross-encoder 重排。
- 增量写入。每个 episode 单独更新图，不需要像 GraphRAG 那样整体重建索引。

### 适合用 Graphiti 的场景

| 信号 | Graphiti 的优势 |
|---|---|
| 事实**会随时间变化**（偏好、状态、负责人、地址、计划） | 旧事实带时间戳失效，不会和新事实同时被检索出来、互相矛盾 |
| **关系**很重要：谁向谁汇报、哪个客户用哪个产品、哪个供应商发往哪个港口 | 事实是去重后实体之间的边，多跳问题和图遍历都能做 |
| 记忆是从对话或事件中**持续积累**的，而不是一批固定文档 | 一条一条写入，无需重建索引 |
| 需要**按用户 / 租户隔离** | 每个节点和边都带 `group_id`（本示例用 user ID），检索时按它过滤 |
| 需要**可解释、可审计**：这条记忆从哪来、什么时候成立 | 事实可以追溯到抽取它的 episode，并带有效时间 |
| 用户**中英文混用**，或者同一个东西有多个叫法 | 基于 LLM 的实体消解会合并别名 |

### 这些情况建议用更简单的方案

| 场景 | 更合适的选择 |
|---|---|
| 基于**固定文档集**问答（手册、制度、知识库） | 普通 RAG：切块、embedding、OpenSearch k-NN。更便宜、更快，不用维护图 |
| 只需要**当前会话**的上下文 | Agent 自带的消息历史，或滚动摘要 |
| 记忆是**彼此独立的片段**（「用户喜欢喝茶」），既不矛盾也没有关系 | 带元数据过滤的向量库，或在 OpenSearch 上跑 Mem0 这类轻量记忆库 |
| **写入量大或写入延迟要求高**（每秒上千事件，或要求 1 秒内写完） | Graphiti 每写一个 episode 要调用多次 LLM（抽取、去重、矛盾判断）。实测一次 `add_memory` 需要 **7–10 秒**，一次 `search_memory` 约 **0.5 秒**。可以异步排队写入，或者换不依赖 LLM 的存储 |
| **LLM 预算很紧** | 每次写入都要多次调用 LLM；读取只需要一次 embedding（开启 rerank 时再加一次 rerank） |
| 没有能力运维有状态服务 | 本方案要运行 Neptune 和 OpenSearch。它们是托管服务，但仍然有容量、备份和成本要管 |

**经验法则：** 如果难点在「现在什么是真的、它怎么变过来的、这些东西之间什么关系」，用 Graphiti；如果难点在「找到回答这个问题的那段文字」，用 RAG。

---

## 架构

```
                 ┌──────────────────────────── VPC（2 个可用区）──────────────────────────┐
  用户（VPN / ───▶ ALB（默认内网）──▶ app :8501  Streamlit + Strands Agent              │
  专线 / 堡垒机）│                                                                        │
  VPC 内的    ───▶ mcp.agent-memory.local:8000 ─▶ mcp :8000  MCP server（streamable-http）│
  MCP 客户端     │                                    │                                   │
                 │           Cloud Map 命名空间：agent-memory.local                       │
                 │                                    ├──▶ litellm :4000 ─NAT─▶ DeepSeek / 百炼 / Kimi / ...
                 │                                    │      （2 个任务）   └─▶ 区域内 SageMaker / vLLM（可选）
                 │                                    ├──▶ Neptune Database :8182（IAM 认证，隔离子网）
                 │                                    └──▶ OpenSearch 托管域 :443（IAM / SigV4 "es"，隔离子网）
                 └────────────────────────────────────────────────────────────────────────┘
```

**写入路径（`add_memory`）：** episode 文本交给 Graphiti。Graphiti 通过 `graphiti-llm` 抽取实体和事实，再通过 `embedding` 为每个名称和事实生成向量。接着用 `graphiti-small` / `graphiti-llm` 与已有数据做去重和矛盾判断。最后把节点、边和向量写入 Neptune。`add_episode` 返回后，`GraphitiBackend.add()` 会调用 `GraphitiDemo.index_episode()`，把实体名称和摘要、事实以及 episode 文本写入 OpenSearch 的 BM25 索引（`node_name_and_summary`、`edge_name_and_fact`、`episode_content`）。graphiti-core 0.31.2 的 Neptune driver 自己只把 community 节点写入 OpenSearch。如果你直接调用 `GraphitiDemo.client.add_episode()`，之后需要自己调用 `index_episode(result)`，否则 BM25 检索看不到这些数据。

**读取路径（`search_memory`）：** 先对 query 生成向量。Graphiti 在 OpenSearch 上做 BM25，并在 Neptune 存的向量上算 cosine 相似度，都按 `group_id` 过滤。结果用 RRF 融合后返回，每条事实带 `valid_at` / `invalid_at`。设置了 `RERANK_MODEL` 时，Graphiti 的 cross-encoder 检索方案会调用网关的 `/rerank`。

**模型 alias**（在 `litellm/config.yaml` 中定义）：

| Alias | 用途 | 要求 |
|---|---|---|
| `graphiti-llm` | 实体 / 事实抽取、矛盾判断 | 必须支持 `response_format: {"type": "json_object"}` |
| `graphiti-small` | Graphiti 中较轻的调用（去重、摘要） | 能输出 JSON；可以用更便宜或更快的模型 |
| `agent-llm` | Strands 对话 Agent | 必须支持 OpenAI 风格的 tool calling |
| `embedding` | 名称、事实和 query 的向量 | 输出维度**必须等于** `embeddingDim`（默认 1024） |
| `rerank` | 可选的 cross-encoder，接口为 `POST /rerank`（Cohere 格式） | `rerankModel` 设为空即关闭 |

---

## 与 global 版的差异

| | global 原版 | 本版（中国区） |
|---|---|---|
| LLM | 通过 litellm SDK 调 Bedrock（`bedrock/` 前缀） | LiteLLM **proxy**，OpenAI 协议，应用里只出现 alias |
| Embedding | Bedrock Titan v2（boto3 直连） | `embedding` alias。按各平台上限分批请求（DashScope 10、SiliconFlow 32）；维度不符直接报错，不会截断向量 |
| Reranker | `OpenAIRerankerClient` 包了一层 `LiteLLMClient`。这条路实际跑不通：后者没有 `.chat.completions`，只是默认的 RRF 检索不会调用 reranker，所以一直没暴露；`logit_bias` 用的 token id 也只适用于 OpenAI tokenizer | 网关的 `/rerank`（Cohere 格式，可接 qwen3-rerank、bge-reranker、Cohere、Jina 等）；不配置时用 passthrough |
| 全文索引 | OpenSearch Serverless（SigV4 `aoss`） | OpenSearch 托管域（SigV4 `es`）；仍可切回 `aoss`，或改用 basic auth |
| BM25 索引写入 | 交给 graphiti-core 的 Neptune driver，而它（0.31.2）只把 community 节点写入 OpenSearch，所以 BM25 检索不到实体、事实和 episode | 每次 `add_episode` 之后，由 `GraphitiDemo.index_episode()` 把实体节点、事实和 episode 写入 OpenSearch（由 `GraphitiBackend.add()` 调用；有单元测试覆盖） |
| Neptune | Database 或 Analytics | 只有 Database（中国区没有 Neptune Analytics） |
| Agent 模型 | Strands `BedrockModel` | Strands `OpenAIModel`，指向网关 |
| JSON 解析 | 只剥掉 code fence | 提取第一个完整的 JSON 值（部分模型会在 JSON 前后多输出文字） |
| 同步调异步 | 每次调用都 `asyncio.run`。每次新建 event loop，而 httpx 连接池绑定在 loop 上，跑久了会报 "Event loop is closed" | 共享一个常驻的后台 event loop |
| Neptune schema | summary API 不可用时 `NeptuneGraph` 初始化失败（t3/t4g 实例默认关闭统计信息） | 容忍 schema 加载失败（Graphiti 用不到 schema） |
| IaC | 无（需要自己准备 endpoint） | CDK：VPC、Neptune、OpenSearch、ECS、Cloud Map、ALB、Secrets |

> Graphiti 的 Neptune driver 只用 OpenSearch 做 BM25 全文检索；向量存在 Neptune 的节点和边上，cosine 在 Python 端计算。所以把 Serverless 换成托管域，在功能上没有任何影响。

### 目录结构

```
frameworks/settings.py          全部配置（环境变量 / .env）
frameworks/graphiti_neptune.py  Graphiti 组装：Neptune driver、OpenSearch（es）及 BM25 索引写入、网关 LLM / embedding / rerank
frameworks/memory_backend.py    search / add 接口（与原版一致）
tools/memory_tool.py            带 search_memory / add_memory 工具的 Strands Agent
examples/                       Agent 人设（default、travel_assistant），自定义人设也放这里
app.py                          Streamlit 对话界面
mcp_server.py                   MCP server（stdio 或 streamable-http）
litellm/config.yaml             alias → 实际模型映射（默认 DeepSeek + 百炼，注释里有多种替代写法）
infra/                          CDK 应用（Python）
scripts/mirror_images.sh        把 LiteLLM 和 Python 基础镜像同步到中国区 ECR
scripts/build_app_image.sh      构建并推送应用镜像
scripts/run_in_vpc.sh           在 VPC 内以一次性 Fargate 任务运行命令（冒烟测试、管理脚本）
scripts/smoke_test.py           部署后逐个组件检查
scripts/testing/                网关契约 e2e 测试（e2e_gateway_local.py）及其仅供测试的网关配置（litellm-bedrock-test.yaml）
tests/                          单元测试 + CDK 模板断言（不调用 AWS）
```

---

## 部署

整个部署约 30–40 分钟，其中 CDK 部署约 17 分钟（主要花在 Neptune 和 OpenSearch 上）。

### 前置条件

- 一个 AWS 中国区账号（光环新网或西云数据运营），以及有权限创建 VPC、Neptune、OpenSearch、ECS、Elastic Load Balancing、Cloud Map、IAM、Secrets Manager、CloudWatch Logs、ECR 和 Lambda 资源的 IAM 身份。stack 里有一个 Lambda 函数：`@aws-cdk/aws-ec2:restrictDefaultSecurityGroup` 背后的 CDK custom resource，用来删除 VPC 默认安全组的规则。
- AWS CLI v2，并配置好中国区 profile（`aws configure --profile cn`）。
- Node.js 18+（用于 `npx aws-cdk@2`）和 Python 3.12。
- 把镜像放进 ECR 的工具：
  - 同步 LiteLLM 和 Python 基础镜像：docker，或 [skopeo](https://github.com/containers/skopeo)（不需要 daemon）；
  - 构建应用镜像：docker，或 AWS CodeBuild 项目（见[没有 docker 时如何构建](#没有-docker-时如何构建)）。skopeo 不能构建镜像。
- 至少一家能从中国区访问的模型平台的 API key。默认配置用 **DeepSeek** 做对话和抽取，用**阿里云百炼（DashScope）**做 embedding、rerank 和兜底；也可以用自己的 SageMaker / vLLM。

在当前 shell 中设置以下变量。`AWS_REGION` 很重要：只设 `AWS_PROFILE` 的话，CDK 可能按 global 区域去解析凭证，导致失败。

```bash
export AWS_PROFILE=cn
export AWS_REGION=cn-northwest-1          # 或 cn-north-1
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export REGISTRY=$ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com.cn
```

### 第 1 步：选模型（`litellm/config.yaml`）

alias 名（`model_name`）不要改，只改 `litellm_params`。

- `embedding` 的输出维度必须正好等于 `embeddingDim`（默认 **1024**）。DashScope `text-embedding-v4` 配 `dimensions: 1024`，或 `bge-m3`，都可以。**数据写入之后，维度和模型就固定了**；换 embedding 模型需要全部重新写入。
- `graphiti-llm` 必须支持 `json_object` 输出，`agent-llm` 必须支持 tool calling。DeepSeek-V3.x、Qwen3 / qwen-plus、GLM-4.5+、Kimi K2 都满足。
- 同一个 alias 写两遍，就组成负载均衡 / 故障切换组（默认配置里 DeepSeek 和 qwen-plus 就是这样组队的）。
- 文件里每个以 `os.environ/XXX` 引用的 key（`LITELLM_MASTER_KEY` 除外，它由 stack 设置），都要出现在 CDK context `providerKeyNames` 中（见第 3 步）。
- 如果要求**数据不出 AWS 账号**，用文件末尾的 SageMaker 或 VPC 内 vLLM 示例。用 SageMaker 时还需要传 `-c sagemakerEndpointArns='["arn:aws-cn:sagemaker:..."]'`（JSON 列表或逗号分隔的字符串）。

选型建议见[如何选模型](#如何选模型)。

### 第 2 步：把镜像放进 ECR

中国区的 Fargate 无法稳定地从 `ghcr.io` 或 `docker.io` 拉镜像，所以所有镜像都必须放在你自己的 ECR 里。

```bash
# 2a. 同步 LiteLLM 和 Python 基础镜像（在能访问 ghcr.io 的机器上执行）。
#     USE_SKOPEO=1 时直接在 registry 之间复制，不需要 docker daemon。
LITELLM_TAG=main-stable USE_SKOPEO=1 ./scripts/mirror_images.sh

# 2b. 基于同步过来的基础镜像构建应用镜像，并使用国内 PyPI 源
TAG=v1 USE_MIRROR_BASE=1 PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
  ./scripts/build_app_image.sh
```

两个脚本都读取 `REGION`，未设置时依次回退到 `AWS_REGION` 和 `cn-northwest-1`，所以前面 export 的 `AWS_REGION` 就够了。

生产环境建议把 `LITELLM_TAG` 固定到具体的 release（例如 `v1.79.x-stable`），不要用 `main-stable`。

#### 没有 docker 时如何构建

如果本机没有 docker，可以在同一个中国区建一个 AWS CodeBuild 项目来运行 `scripts/build_app_image.sh`。项目配置：镜像用 `aws/codebuild/standard:7.0` 并开启 **privileged 模式**，角色要能推送 ECR，源码用本仓库，buildspec 如下（CodeBuild 会设置 `AWS_REGION`，脚本直接使用它）：

```yaml
version: 0.2
phases:
  build:
    commands:
      - TAG=v1 USE_MIRROR_BASE=1 PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ bash scripts/build_app_image.sh
```

本示例在 cn-northwest-1 实测时，应用镜像就是这样构建的。

### 第 3 步：CDK 部署

```bash
# 3a. 每个账号 + 区域 bootstrap 一次。要在 infra/ 之外执行：在 infra/ 里 CDK 会先
#     synth 应用，因为缺少镜像参数而报错。中国区也写 aws://，不是 aws-cn://
(cd /tmp && npx -y aws-cdk@2 bootstrap aws://$ACCOUNT/$AWS_REGION)

# 3b. 仅新账号需要：首次在 VPC 里创建 OpenSearch 域之前，需要先有它的
#     service-linked role，否则首次部署会失败。提示已存在可以忽略。
aws iam create-service-linked-role --aws-service-name opensearchservice.amazonaws.com || true

# 3c. 部署
cd infra
python3 -m venv .venv && . .venv/bin/activate && pip install aws-cdk-lib==2.275.0 constructs==10.8.1
npx -y aws-cdk@2 deploy \
  -c region=$AWS_REGION -c account=$ACCOUNT \
  -c litellmImage=$REGISTRY/mirror/litellm:main-stable \
  -c appImage=$REGISTRY/graphiti-agent-memory:v1
```

低成本 PoC 可以追加：

```bash
  -c neptuneInstanceClass=db.t4g.medium \
  -c opensearchInstanceType=t3.medium.search -c opensearchDataNodes=1 \
  -c retainData=false
```

（本示例实测用的就是这个规格。OpenSearch 只有 1 个节点时，副本无法分配，健康状态会是 `yellow`，属于正常现象。）

部署完成后，记下 stack 输出：`AppUrl`、`McpUrl`、`LiteLLMUrl`、`ProviderKeysSecretArn`、`ClusterName` 等。

### 第 4 步：写入模型 key，重启 LiteLLM

stack 会创建一个占位 secret，`providerKeyNames` 里每个名字的值都是 `REPLACE_ME`。把真实的 key 写进去，然后滚动重启 LiteLLM 服务：

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

如果想自己管理 secret（例如要做轮换），可以先建好，再部署时传 `-c providerSecretArn=<arn>`。

### 第 5 步：冒烟测试

```bash
cd ..   # 回到仓库根目录
./scripts/run_in_vpc.sh            # 以一次性 Fargate 任务运行 scripts/smoke_test.py
```

预期输出（有删节，来自 cn-northwest-1 实测）：

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

冒烟测试按顺序检查：各网关 alias（JSON 模式、tool calling、embedding 维度、rerank），然后 Neptune openCypher，然后 OpenSearch 健康状态和索引，然后真实写入并检索一条记忆，最后清理测试用户在 Neptune 中的节点和边。清理不涉及 OpenSearch，测试用户的 BM25 文档会留下（见[删除某个用户的记忆](#删除某个用户的记忆)）。失败的那一行会指明出问题的组件，处理方法见[排障](#排障)。

推荐用 `scripts/run_in_vpc.sh`，它既不需要 ECS Exec，也不需要 Session Manager 插件。ECS Exec 默认关闭。如果想改用它，请用 `-c enableExecuteCommand=true` 部署（只对 app 服务开启），装好 Session Manager 插件，然后执行：

```bash
aws ecs execute-command --cluster <ClusterName> --task <task-id> --container app --interactive \
  --command "sh -c 'LLM_API_KEY=sk-\$LLM_API_KEY_RAW python scripts/smoke_test.py'"
```

`\$` 让本地 shell 不展开这个变量，由容器里的 shell 展开。

---

## 使用

### 1. Streamlit 对话 Demo

`AppUrl` 是**内网** ALB，因为中国区公网 80/443 端口需要 ICP 备案。可以用以下方式访问：

- 从与 VPC 打通的网络访问（Client VPN、专线、Transit Gateway）。ALB（80 端口）和 MCP（8000 端口）的安全组规则默认只放行 VPC CIDR。Client VPN 会把客户端流量 NAT 成 VPC 内的地址，所以不用改。专线或 Transit Gateway 需要部署时传 `-c albIngressCidr=<本地网络 CIDR>`（UI）和 `-c mcpIngressCidr=<本地网络 CIDR>`（MCP）。两者都只接受一个 CIDR，并会替代 VPC CIDR。
- 通过 VPC 内任意一台 EC2 做 SSM 端口转发：
  ```bash
  aws ssm start-session --target <instance-id> \
    --document-name AWS-StartPortForwardingSessionToRemoteHost \
    --parameters host=<AppUrl 的主机名>,portNumber=80,localPortNumber=8501
  # 然后打开 http://localhost:8501
  ```
- 如果已有 ICP 备案，可以用 `-c publicAlb=true -c albIngressCidr=<办公网 CIDR>` 部署，并在前面加上 HTTPS 和身份认证。

在侧边栏选一个用户就可以开始对话。Agent 会在对话开始时调用 `search_memory`，在你提供事实或偏好时调用 `add_memory`。每次工具调用和结果都会显示在回答下方（`st.info` / `st.code`），可以看到具体存了什么、检索到了什么。每个浏览器会话有自己的 Agent（`create_memory_agent()`），不同标签页的对话不会混在一起。

快速验证跨会话记忆：

1. 以 `User1` 身份说：「我下周去杭州出差，喜欢住西湖边、高楼层的酒店，对花生过敏。」
2. 切到 `User2` 再切回 `User1`，这会清空对话历史。
3. 问：「帮我订酒店，再推荐一家晚饭的餐厅。」回答应该用到从记忆中召回的「西湖边、高楼层、不吃花生」。
4. 之后再说「我现在更喜欢住火车站附近」。旧偏好会被标记为失效（设置 `invalid_at`），新偏好成为当前事实。

切换人设：部署时传 `-c agentConfig=travel_assistant`，本地运行时设 `AGENT_CONFIG`。

### 2. MCP server（Claude Code、Kiro、Cursor 等任意 MCP 客户端）

stack 以 streamable-http 方式运行 MCP server，地址是 `http://mcp.agent-memory.local:8000/mcp`（输出 `McpUrl`），提供两个工具：

| 工具 | 参数 | 说明 |
|---|---|---|
| `add_memory` | `content`、`user_id` | 为某个用户存一条信息 |
| `search_memory` | `query`、`user_id` | 返回该用户的相关事实，带有效时间 |

`user_id` 必须匹配 `^[a-zA-Z0-9_-]+$`（见 [Python API](#3-python-api把记忆嵌入你自己的-agent)）。

该地址**只能在 VPC 内访问，并且没有鉴权**。能访问到它的客户端，可以读写任何用户的记忆。请从 VPC 内的主机连接（或通过 VPN，且能解析 Cloud Map DNS；专线 / Transit Gateway 需要设置 `mcpIngressCidr`，见 [Streamlit 对话 Demo](#1-streamlit-对话-demo)）：

```bash
# Claude Code
claude mcp add --transport http agentmemory http://mcp.agent-memory.local:8000/mcp
```

```jsonc
// Kiro / Cursor / 其他客户端（mcp.json）
{ "mcpServers": { "agentmemory": { "url": "http://mcp.agent-memory.local:8000/mcp" } } }
```

如果要对 VPC 外开放，请在前面加一层鉴权，例如 API Gateway 配 Lambda authorizer，或 ALB 配 OIDC。

也可以在**本地以 stdio 方式**运行 MCP server。前提是本机能访问 Neptune、OpenSearch 和 LiteLLM，并填好 `.env`：

```jsonc
{ "mcpServers": { "agentmemory": {
    "command": "uv", "args": ["run", "--directory", "/path/to/repo", "python", "mcp_server.py"] } } }
```

### 3. Python API（把记忆嵌入你自己的 Agent）

```python
from frameworks.memory_backend import get_memory_backend

memory = get_memory_backend()            # 从环境变量 / .env 读取配置

memory.add("李雷下周去杭州出差，喜欢靠窗座位，对花生过敏。", user_id="lilei")
memory.search("李雷的饮食禁忌", user_id="lilei")
# -> [{'fact': '李雷对花生过敏', 'valid_at': '2026-11-02T06:35:52+00:00', 'invalid_at': None}, ...]
```

- `user_id` 就是 Graphiti 的 `group_id`，所有读写都限定在它的范围内。每个终端用户用一个（如果租户内共享记忆，也可以每个租户用一个）。
- `user_id` 只能包含 ASCII 字母、数字、`-` 和 `_`（`^[a-zA-Z0-9_-]+$`，由 graphiti-core 的 `validate_group_id` 校验）。例如邮箱地址会被拒绝，请先映射成内部 ID。
- `add()` 是同步的，因为要调用 LLM，耗时几秒。对延迟敏感的服务，请放到后台 worker 或队列（例如 SQS）中执行。
- 检索结果里有 `invalid_at`。设置了 `invalid_at` 的事实是历史事实，可以过滤掉，也可以作为历史展示。

如果要用 Graphiti 的完整功能（自定义实体类型、JSON episode、带 cross-encoder 的 `search_()` 检索方案、community、`remove_episode` 等），直接使用底层 client：

```python
from frameworks.graphiti_neptune import GraphitiDemo, run_sync
demo = GraphitiDemo(group_id="lilei")
results = run_sync(demo.client.search("酒店偏好", group_ids=["lilei"], num_results=10))

result = run_sync(demo.client.add_episode(...))
demo.index_episode(result)   # 必须：把该 episode 的节点、事实和文本写入 BM25 索引
```

请始终通过 `run_sync()`（共享同一个 event loop）调用，不要每次都 `asyncio.run()`。直接调用 `demo.client.add_episode()` 时，请像 `GraphitiBackend.add()` 一样紧接着调用 `demo.index_episode(result)`，否则 BM25 检索看不到这些数据。

### 4. Strands Agent

`tools/memory_tool.py` 用 `OpenAIModel` 构建了一个 Strands `Agent`，指向 `agent-llm` alias，并带两个工具；工具从 `agent.state["user_id"]` 读取当前用户：

```python
from tools.memory_tool import memory_agent
memory_agent.state.set("user_id", "lilei")
print(memory_agent("帮我订下周去杭州的酒店"))
```

`memory_agent` 是模块级共享的一个实例，适合只有一个对话的脚本。Agent 里保存着消息历史和当前用户，所以有多个对话时，请用 `create_memory_agent()` 为每个对话各建一个，`app.py` 就是按浏览器会话这样做的。

### 5. 自定义人设

把 `examples/default.py` 复制为 `examples/my_agent.py`，修改 `AGENT_NAME`、`SYSTEM_PROMPT`、`EXTRA_TOOLS`（任意 Strands `@tool` 函数）和 `USERS`。然后二选一：在 `examples/__init__.py` 的 `AVAILABLE_AGENTS` 里注册，再用短名（`-c agentConfig=my_agent`，本地用 `AGENT_CONFIG=my_agent`）；或者不注册，直接用模块路径（`-c agentConfig=examples.my_agent` / `AGENT_CONFIG=examples.my_agent`）。没注册的短名会导入失败。最后重新构建应用镜像并重新部署。

### 删除某个用户的记忆

要删除某个用户的全部节点和边（例如处理删除请求），可以执行 openCypher。用 `scripts/run_in_vpc.sh` 运行，或在 VPC 内任意主机上运行：

```python
from frameworks.graphiti_neptune import GraphitiDemo
demo = GraphitiDemo("")
demo.driver.client.query("MATCH (n) WHERE n.group_id = $g DETACH DELETE n", params={"g": "lilei"})
```

该用户在 OpenSearch 里的文档还在。要删掉它们，需要在 Graphiti 的 4 个索引（`node_name_and_summary`、`edge_name_and_fact`、`episode_content`、`community_name`）上按 `group_id` 执行 `delete_by_query`。

---

## 配置参考

### CDK context（`-c key=value`，默认值见 `infra/cdk.json` 和 `infra/stacks/agent_memory_stack.py`）

| key | 默认值 | 说明 |
|---|---|---|
| `litellmImage` / `appImage` | —（必填） | 第 2 步得到的 ECR 镜像 URI |
| `region` / `account` | 取环境 | 目标账号和区域 |
| `stackName` | `GraphitiAgentMemoryCn` | CloudFormation stack 名称 |
| `providerKeyNames` | `["DEEPSEEK_API_KEY","DASHSCOPE_API_KEY"]` | 以环境变量形式注入 LiteLLM 的 secret 字段，必须与 `litellm/config.yaml` 中的 `os.environ/...` 一致。JSON 列表或逗号分隔的字符串均可，`-c` 和 `cdk.json` 里都适用 |
| `providerSecretArn` | 空（创建占位 secret） | 使用已有的 Secrets Manager secret（JSON，字段同上） |
| `neptuneInstanceClass` | `db.r6g.large` | PoC 可用 `db.t4g.medium` |
| `neptuneEngineVersion` | 区域默认 | |
| `opensearchInstanceType` | `r6g.large.search` | PoC 可用 `t3.medium.search` |
| `opensearchDataNodes` | `2` | PoC 可用 1（不启用 zone awareness） |
| `opensearchVersion` | `2.19` | |
| `natGateways` | `1` | 设为 2 可以让访问模型平台的出口跨可用区冗余 |
| `embeddingDim` | `1024` | 必须与 `embedding` alias 一致；写入数据后不能再改 |
| `rerankModel` | `rerank` | 设为 `""` 关闭 cross-encoder |
| `agentConfig` | `default` | `examples/` 下的人设模块 |
| `publicAlb` / `albIngressCidr` | `false` / VPC CIDR | `albIngressCidr` 是允许访问 ALB 80 端口的来源 CIDR。公网 ALB 需要 ICP 备案，并且要显式指定 CIDR |
| `mcpIngressCidr` | VPC CIDR | 允许访问 MCP server 8000 端口的来源 CIDR（专线 / Transit Gateway 客户端需要设置） |
| `enableExecuteCommand` | `false` | 在 app 服务上开启 ECS Exec。`scripts/run_in_vpc.sh` 不需要它 |
| `retainData` | `true` | 为 true 时 `cdk destroy` 会保留 Neptune 和 OpenSearch，并且 Neptune 集群开启删除保护（OpenSearch 没有） |
| `sagemakerEndpointArns` | `[]` | 给 LiteLLM 任务角色授予这些 ARN 上的 `sagemaker:InvokeEndpoint` 和 `sagemaker:InvokeEndpointWithResponseStream`。JSON 列表或逗号分隔的字符串均可，`-c` 和 `cdk.json` 里都适用 |
| `litellmConfig` | `litellm/config.yaml` | 替代的网关配置路径（相对于仓库根目录），测试时使用 |
| `bedrockAccess` | `false` | 仅测试用：给 LiteLLM 任务角色授予 `*` 上的 `bedrock:InvokeModel`、`bedrock:InvokeModelWithResponseStream` 和 `bedrock:Rerank`，供 global 区域中基于 Bedrock 的测试配置使用 |

### 应用环境变量（`frameworks/settings.py`，参见 `.env.example`）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_BASE_URL` / `LLM_API_KEY` | `http://localhost:4000/v1` / `sk-litellm-local` | 网关地址和 key。stack 会注入 LiteLLM master key（`sk-` + master key secret 的值） |
| `GRAPHITI_LLM_MODEL` / `GRAPHITI_SMALL_MODEL` / `AGENT_MODEL` / `EMBEDDING_MODEL` | 对应的 alias 名 | 只有改了 alias 名才需要设置 |
| `RERANK_MODEL` | `""`（关闭） | rerank alias；为空时用 passthrough 代替 cross-encoder。stack 按 `rerankModel` 设置（`rerank`） |
| `LLM_STRUCTURED_OUTPUT_MODE` | `json_object` | 只有 `graphiti-llm` 背后的所有模型都支持时才用 `json_schema` |
| `LLM_MAX_TOKENS` / `LLM_TEMPERATURE` | `8192` / `0` | `LLM_TEMPERATURE` 只作用于 Graphiti 的调用；Strands Agent 固定用 `0.3`（`tools/memory_tool.py`） |
| `EMBEDDING_DIM` | `1024` | 必须等于 embedding 模型的输出维度 |
| `EMBEDDING_BATCH_SIZE` | `10` | 平台限制（DashScope 10、SiliconFlow 32） |
| `GRAPHITI_NEPTUNE_ENDPOINT` / `_PORT` | — / `8182` | `neptune-db://<cluster endpoint>` |
| `GRAPHITI_OPENSEARCH_ENDPOINT` / `_PORT` | — / `443` | 域的 endpoint，不带协议前缀 |
| `GRAPHITI_OPENSEARCH_SERVICE` | `es` | 在 global 区域用 Serverless 时设为 `aoss` |
| `GRAPHITI_OPENSEARCH_AUTH` | `iam` | 设为 `basic` 并配 `_USERNAME` / `_PASSWORD`，即可使用细粒度访问控制的主用户 |
| `AGENT_CONFIG` | `default` | 人设 |
| `AWS_REGION` | `cn-northwest-1` | 未设置时回退到 `AWS_DEFAULT_REGION`，再回退到 `cn-northwest-1`；用于 SigV4 |
| `MCP_TRANSPORT` / `MCP_HOST` / `MCP_PORT` | `stdio` / `0.0.0.0` / `8000` | |

---

## 如何选模型

Graphiti 的效果主要取决于 `graphiti-llm` 用的模型：它要能遵循很长的抽取 prompt，并且每次都返回合法的 JSON。

| 角色 | 建议起点 | 说明 |
|---|---|---|
| `graphiti-llm` | DeepSeek-V3.2（`deepseek-chat`）、Qwen3 / qwen-plus | 指令遵循和 JSON 输出都强。小模型（<14B）容易漏事实或输出坏 JSON |
| `graphiti-small` | qwen-flash、Qwen3-32B、DeepSeek | 调用频繁，延迟比深度更重要 |
| `agent-llm` | DeepSeek-V3.2、Qwen3、Kimi K2、GLM-4.6 | tool calling 必须稳定 |
| `embedding` | DashScope `text-embedding-v4`（1024）、`bge-m3`（1024） | 多语言模型能处理中英混合 |
| `rerank` | `qwen3-rerank`、`bge-reranker-v2-m3` | 可选；默认检索（RRF）不会用到 |

**已实测的组合：**

| 环境 | graphiti-llm / agent-llm | graphiti-small | embedding | rerank | 结果 |
|---|---|---|---|---|---|
| cn-northwest-1，完整部署 | DeepSeek V3.2 | Qwen3-32B | Titan v2（1024） | Cohere Rerank 3.5 | 冒烟测试 11/11 通过；MCP 通过；Agent 能跨会话召回 |
| us-east-2，完整部署 | Claude（Bedrock） | Claude（Bedrock） | Titan v2 | Cohere Rerank 3.5 | 冒烟测试 11/11 通过；MCP 通过 |
| 本地网关 e2e（`scripts/testing/litellm-bedrock-test.yaml`） | Claude Haiku 4.5（Bedrock） | Claude Haiku 4.5（Bedrock） | Titan v2（1024） | Cohere Rerank 3.5 | 抽取正常；中英文别名合并（「张伟」=「Zhang Wei」） |

cn-northwest-1 实测时手上没有国内平台的 key，所以通过一份仅用于测试的 LiteLLM 配置（未包含在本仓库中）跨境调用了 global 区域 Bedrock 上的模型。这个配置**只能用于合成的测试数据**。默认的 `litellm/config.yaml`（DeepSeek + 百炼 API）走的是同一套 OpenAI 兼容代码路径，但没有用真实 key 实测过。请在第 4 步之后跑一次冒烟测试，确认你选的模型组合可用。

**存储事实的语言：** Graphiti 的抽取 prompt 是英文的，所以即使输入是中文，有些模型也会用英文写事实（一次 MCP 测试中，「韩梅梅对花生过敏」就被存成了一句英文）。使用多语言 embedding 时，跨语言检索仍然有效。如果要求事实保持原文语言，可以选更倾向保留原语言的模型（冒烟测试中 DeepSeek 保留了中文），或者通过 Graphiti API 传入 `custom_extraction_instructions` / 自定义 prompt。

---

## 规格与成本

stack 创建的计费资源：

| 资源 | 默认 | PoC 选项 |
|---|---|---|
| Neptune Database | 1 个 `db.r6g.large` 写实例，外加存储和 I/O | `db.t4g.medium` |
| OpenSearch 托管域 | 2 个 `r6g.large.search`，每个 50 GB gp3 | 1 个 `t3.medium.search` |
| ECS Fargate | LiteLLM 2 个（1 vCPU，2 GB），app 1 个（1 vCPU，2 GB），MCP 1 个（0.5 vCPU，1 GB） | — |
| NAT 网关 | 1 个，外加数据处理费 | — |
| ALB、Secrets Manager（2 个 secret）、CloudWatch Logs（任务日志保留 30 天）、VPC flow logs（CDK 默认保留期 731 天 / 2 年） | | |
| 模型 API 调用 | 由模型平台计费，不在 AWS 账单里 | |

具体价格请用 [AWS 中国区价格计算器](https://calculator.amazonaws.cn/) 按区域估算。Neptune 和 OpenSearch 是最大的固定成本，不用时请停止或删除 stack。

模型费用随写入量增长：每次 `add_memory` 会调用若干次 `graphiti-llm` / `graphiti-small`，外加 embedding。读取只需要一次 embedding（开启 rerank 时再加一次 rerank）。

生产环境建议：

- Neptune：在第二个可用区加一个只读副本，实现高可用。
- OpenSearch：保持 2 个以上数据节点跨可用区（即默认配置）；数据量大时加专用主节点。
- 设 `natGateways=2`。
- LiteLLM 服务是无状态的，按请求量扩容即可。

---

## 安全与合规

- **网络：** Neptune 和 OpenSearch 位于没有公网路由的隔离子网中，只有 ECS 任务能访问（由安全组控制）。ALB 默认是内网的。访问模型平台的流量经 NAT 网关出去。
- **数据存储的认证：** Neptune 使用 IAM 数据库认证。OpenSearch 使用 IAM（SigV4），只有 app/MCP 的任务角色有读写权限。两者都没有用户名密码。两者都开启静态加密；OpenSearch 强制 TLS 1.2，并开启节点间加密。
- **密钥：** 模型 API key 和 LiteLLM master key 都存在 Secrets Manager 中，以 ECS secret 的方式注入，不会打进镜像或写进任务定义。
- **MCP 和 UI 没有鉴权。** `user_id` 只是一个普通参数，能访问到它们的人就能读写任意用户的记忆。上生产前请加上身份认证（例如 ALB 上的 OIDC、MCP 前面加鉴权代理），并从认证后的身份推导 `user_id`，不要信任客户端传来的值。
- **LiteLLM** 只用 master key，没有数据库。如果需要按应用分配的虚拟 key、预算和费用统计，请接 PostgreSQL（`DATABASE_URL`），并确认镜像里包含 Prisma 引擎（在国内下载 Prisma 二进制可能失败）。
- **数据出境：** 调用中国大陆以外的模型（global Bedrock、OpenAI 等）属于数据出境，需要先做合规评估。默认配置只列了国内平台和区域内自托管的选项。如果要求数据完全留在自己的 AWS 账号内，请用 SageMaker 或 VPC 内的 vLLM。
- **记忆内容属于用户数据**，其中可能包含 LLM 抽取出来的个人信息。请制定保留期限，提供删除途径（见[删除某个用户的记忆](#删除某个用户的记忆)），并审查你的 prompt 让 Agent 存了哪些内容。

安全问题的报告方式见 [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications)。

---

## 运维

- **日志：** CloudWatch Logs 日志组 `<stack>-LiteLLMTask...`、`<stack>-AppTask...`、`<stack>-McpTask...`（保留 30 天）。VPC flow logs 写入单独的日志组，使用 CDK 默认的保留期 731 天（2 年）。LiteLLM 会记录每个上游错误，包括模型平台返回的原始信息，所以模型调用失败时先看它。
- **更换模型：** 修改 `litellm/config.yaml` 后再执行一次 `cdk deploy`。配置是嵌在 LiteLLM 任务定义里的，所以部署会滚动更新服务。改完后请重跑冒烟测试。
- **轮换模型 key：** 执行 `put-secret-value`，然后强制重新部署 LiteLLM 服务（同第 4 步）。
- **升级应用：** 构建新 tag 的镜像，然后 `cdk deploy -c appImage=...:v2`。部署带 circuit breaker，失败会自动回滚。
- **升级 graphiti-core：** `pyproject.toml` 里固定为 `0.31.2`。以下几处依赖它的内部实现：`NeptuneDriver._run_query` 的补丁、`_database` 的 workaround，以及 `GraphitiDemo.index_episode()`（0.31.2 的 Neptune driver 只把 community 节点写入 OpenSearch，由它补齐 BM25 索引）。升级前请先跑 `pytest` 和冒烟测试，并确认新版 driver 是否已经自己写入节点、边和 episode。
- **备份：** Neptune 保留 7 天自动备份；OpenSearch 每小时自动快照。OpenSearch 索引存的是 Neptune 数据的文本副本，但本示例没有提供从 Neptune 重建索引的脚本，请依靠快照。

---

## 排障

以下都是开发和测试本示例时真实遇到过的问题。

| 现象 | 原因 | 解决方法 |
|---|---|---|
| `cdk bootstrap aws-cn://...` 报环境格式错误 | CDK 的 environment 格式总是 `aws://`，中国区也一样 | 写 `aws://<account>/cn-northwest-1` |
| 在 `infra/` 里执行 `cdk bootstrap`，报 "Pass -c litellmImage=..." | bootstrap 会先 synth 当前目录下的应用 | 换到别的目录执行（`cd /tmp`） |
| CDK 找不到凭证，或解析到了错误的分区 | 没有设置 `AWS_REGION`，或设成了 global 区域 | 在设置 `AWS_PROFILE` 的同时 `export AWS_REGION=cn-northwest-1` |
| 首次部署创建 OpenSearch 域失败，报 service-linked role 相关错误 | 新账号还没有 OpenSearch 的 service-linked role | 执行 `aws iam create-service-linked-role --aws-service-name opensearchservice.amazonaws.com` 后重新部署 |
| 任务一直 `PENDING`，或报 `CannotPullContainerError` | 镜像不在中国区 ECR，或 URI 写错（应为 `.amazonaws.com.cn`） | 按第 2 步操作，并检查传给 CDK 的镜像 URI |
| 冒烟测试报 `[FAIL] gateway ...: 401` | 模型 key 还是 `REPLACE_ME`，或写入后没有重启 LiteLLM | 按第 4 步操作 |
| `[FAIL] gateway ...: Access to Anthropic models is not allowed from unsupported countries` | Anthropic 模型拒绝来自中国 IP 的调用，即使通过 global Bedrock 也一样 | 换其他模型系列（DeepSeek、Qwen 等） |
| `[FAIL] gateway embedding: got dims {1536}` | embedding 模型的原生维度 ≠ `embeddingDim` | 在 alias 里设置 `dimensions:`，或在写入任何数据之前用匹配的 `embeddingDim` 重新部署 |
| `add_memory` 报 `json.decoder.JSONDecodeError` | `graphiti-llm` 用的模型不遵守 JSON 模式 | 换更强的模型；到 LiteLLM 日志里查看原始返回 |
| 应用启动后，Neptune 对 `GetGraphSummary` 报 `AccessDeniedException` | IAM 策略缺少 `neptune-db:GetGraphSummary` | stack 已授予该权限；如果用了自定义角色，请检查 |
| 日志出现 "Neptune schema summary unavailable, continuing without it" | Neptune 统计信息关闭（t3/t4g） | 无害的警告，可以忽略 |
| MCP 容器退出，报 `ImportError: FastMCP` | `mcp` 2.x 改了 API | `pyproject.toml` 已固定 `mcp[cli]==1.30.0`，请保持在 1.x |
| MCP 返回 `421 Misdirected Request` | 用 localhost 作为 host 构造 FastMCP 时，会开启 DNS rebinding 防护 | `mcp_server.py` 已处理（在构造时传入 host） |
| 跑几次请求后报 "Event loop is closed" | 每次调用都 `asyncio.run()`，而 httpx client 是共享的 | 使用 `frameworks/graphiti_neptune.py` 里的 `run_sync()` |
| OpenSearch 健康状态为 `yellow` | 只有 1 个数据节点，副本无法分配 | PoC 下属于正常；生产环境请用 2 个以上节点 |

---

## 清理

```bash
cd infra
npx -y aws-cdk@2 destroy -c region=$AWS_REGION -c account=$ACCOUNT \
  -c litellmImage=$REGISTRY/mirror/litellm:main-stable -c appImage=$REGISTRY/graphiti-agent-memory:v1
```

- 默认 `retainData=true` 时，**Neptune 集群和 OpenSearch 域会被保留**，并会继续计费；Neptune 还开着删除保护（OpenSearch 没有）。这时 stack 删除很可能卡在被保留资源仍在使用的子网和安全组上而失败。请先关闭 Neptune 集群的删除保护，在控制台或 CLI 中删除集群（连同实例）和域，再重新执行 `cdk destroy`。用完即删的 stack，可以先用 `-c retainData=false` 部署一次，再执行 destroy。
- 删除 ECR 仓库 `mirror/litellm`、`mirror/python`、`graphiti-agent-memory`，以及为构建镜像创建的 CodeBuild 项目和 S3 桶。
- 不需要的话，删除残留的 CloudWatch 日志组。
- `CDKToolkit` bootstrap stack 可以留着供以后 CDK 部署使用，也可以连同它的 S3 桶一起删除。

---

## 本地开发与测试

```bash
uv venv && uv pip install -r requirements.txt pytest==9.1.1 pytest-asyncio==1.4.1 \
  aws-cdk-lib==2.275.0 constructs==10.8.1     # 锁定的运行时依赖 + 固定版本的测试/CDK 工具
.venv/bin/python -m pytest   # 单元测试 + CDK 模板断言，不调用 AWS
```

**网关契约 e2e** 会通过本地 LiteLLM proxy 真实运行 Graphiti 的抽取、检索和 rerank。图库用嵌入式 FalkorDB 代替 Neptune，所以不需要任何中国区资源：

```bash
uv pip install "graphiti-core[falkordb]==0.31.2" falkordblite==0.11.0 "litellm[proxy]==1.102.4"
# 把 alias 指向你手头可用的模型；示例配置用的是 global 区域的 Bedrock
# （Claude Haiku 4.5、Titan v2、Cohere Rerank 3.5），需要对应的 AWS 凭证
LITELLM_MASTER_KEY=sk-test litellm --config scripts/testing/litellm-bedrock-test.yaml --port 4011 &
LLM_BASE_URL=http://localhost:4011/v1 LLM_API_KEY=sk-test RERANK_MODEL=rerank \
  python scripts/testing/e2e_gateway_local.py
```

要在本机连接已部署的 stack 运行应用，需要能访问 VPC（VPN，或通过堡垒机把 Neptune 8182、OpenSearch 443、LiteLLM 4000 端口转发出来），另外还需要：

- 在 `NeptuneSg`（8182）、`OpenSearchSg`（443）和 `TasksSg`（4000）上添加允许你的来源（VPN 或堡垒机）访问的入站规则；默认只放行 ECS 任务；
- 与 `AppTaskRole` 等价的 IAM 权限：集群上的 `neptune-db:*` 数据操作，以及域上的 `es:ESHttp*`；
- `LLM_API_KEY=sk-<LiteLLM master key secret 的值>`（输出 `LiteLLMMasterKeySecretArn`）。

把 `.env.example` 复制为 `.env`，填入 stack 输出，然后执行 `.venv/bin/streamlit run app.py`。否则就用 `scripts/run_in_vpc.sh "python my_script.py"` 在 VPC 内运行你的脚本（脚本需要包含在应用镜像里）。

---

## 已知限制

- Graphiti 的 Neptune 向量检索会把候选 embedding 拉到 Python 端计算 cosine。单个 `group_id` 下的图非常大时会变慢，这是由 graphiti_core 的实现决定的。
- `add_memory` 是同步调用，耗时以秒计；本示例没有包含写入队列。
- UI 和 MCP server 没有鉴权（见[安全与合规](#安全与合规)）。
- 没有内置的「忘记某个用户」API，做法见[删除某个用户的记忆](#删除某个用户的记忆)。
- 由于[运维](#运维)中提到的 monkey patch 和 BM25 索引写入的 workaround，graphiti-core 固定在 0.31.2。

## 安全

本项目是示例代码，未经进一步评审不适合用于生产环境，见本文开头的免责声明。

详见 [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications)。

## 许可证

本项目使用 MIT-0 许可证，见 [LICENSE](LICENSE) 文件。第三方依赖见 [THIRD-PARTY-LICENSES](THIRD-PARTY-LICENSES)。
