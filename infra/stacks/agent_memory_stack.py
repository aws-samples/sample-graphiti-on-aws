"""Graphiti agent memory on AWS China regions (cn-north-1 / cn-northwest-1).

    VPC (2 AZ, private subnets + NAT for outbound calls to model providers)
     ├─ Neptune Database cluster (IAM auth)           graph + embeddings
     ├─ OpenSearch Service managed domain (VPC, IAM)   BM25 full-text index
     └─ ECS Fargate cluster, Cloud Map namespace agent-memory.local
         ├─ litellm   :4000  OpenAI-compatible gateway -> DeepSeek / Qwen / SageMaker / ...
         ├─ app       :8501  Streamlit memory agent, behind an ALB (internal by default)
         └─ mcp       :8000  MCP server (streamable-http), VPC-internal only

Everything is partition-aware (Aws.PARTITION / Stack.format_arn), so the same stack
also synthesizes for global regions.
"""

import json
import base64
from pathlib import Path

from aws_cdk import (
    Aws,
    CfnOutput,
    Duration,
    RemovalPolicy,
    SecretValue,
    Stack,
    aws_ec2 as ec2,
    aws_ecr as ecr,
    aws_ecs as ecs,
    aws_elasticloadbalancingv2 as elbv2,
    aws_iam as iam,
    aws_logs as logs,
    aws_neptune as neptune,
    aws_opensearchservice as opensearch,
    aws_secretsmanager as sm,
    aws_servicediscovery as sd,
)
from constructs import Construct

REPO_ROOT = Path(__file__).resolve().parents[2]
LITELLM_CONFIG = REPO_ROOT / "litellm" / "config.yaml"
NAMESPACE = "agent-memory.local"


def _str_list(value) -> list[str]:
    """CDK context list from cdk.json (a list) or `-c` (always a string: JSON or comma-separated)."""
    if not value:
        return []
    if isinstance(value, str):
        value = json.loads(value) if value.lstrip().startswith("[") else value.split(",")
    return [v.strip() for v in value if v.strip()]


def _image(scope: Construct, cid: str, uri: str) -> ecs.ContainerImage:
    """ECR URIs -> from_ecr_repository (grants the execution role pull rights).

    Anything else is pulled as-is; from China regions that only works for registries
    reachable without the GFW in the way, so mirror images into ECR first.
    """
    if ".dkr.ecr." in uri:
        repo_path, _, tag = uri.split("/", 1)[1].partition(":")
        repo = ecr.Repository.from_repository_name(scope, cid, repo_path)
        return ecs.ContainerImage.from_ecr_repository(repo, tag or "latest")
    return ecs.ContainerImage.from_registry(uri)


class AgentMemoryStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        ctx = self.node.try_get_context
        litellm_image = ctx("litellmImage")
        app_image = ctx("appImage")
        if not litellm_image or not app_image:
            raise ValueError(
                "Pass -c litellmImage=<ecr-uri> -c appImage=<ecr-uri> "
                "(see scripts/mirror_images.sh and scripts/build_app_image.sh)"
            )
        provider_key_names = _str_list(ctx("providerKeyNames")) or ["DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY"]
        public_alb = str(ctx("publicAlb") or "false").lower() == "true"
        neptune_instance_class = ctx("neptuneInstanceClass") or "db.r6g.large"
        opensearch_instance_type = ctx("opensearchInstanceType") or "r6g.large.search"
        opensearch_nodes = int(ctx("opensearchDataNodes") or 2)
        nat_gateways = int(ctx("natGateways") or 1)
        sagemaker_endpoint_arns = _str_list(ctx("sagemakerEndpointArns"))
        embedding_dim = str(ctx("embeddingDim") or 1024)
        rerank_model = ctx("rerankModel") if ctx("rerankModel") is not None else "rerank"
        # Test-only knobs: alternative gateway config and Bedrock access (global regions)
        litellm_config = REPO_ROOT / ctx("litellmConfig") if ctx("litellmConfig") else LITELLM_CONFIG
        bedrock_access = str(ctx("bedrockAccess") or "false").lower() == "true"
        retain = str(ctx("retainData") or "true").lower() == "true"
        # ECS Exec on the app service (opt-in; scripts/run_in_vpc.sh does not need it)
        enable_exec = str(ctx("enableExecuteCommand") or "false").lower() == "true"
        data_removal = RemovalPolicy.RETAIN if retain else RemovalPolicy.DESTROY

        # ------------------------------------------------------------------ network
        vpc = ec2.Vpc(
            self, "Vpc",
            max_azs=2,
            nat_gateways=nat_gateways,
            subnet_configuration=[
                ec2.SubnetConfiguration(name="public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24),
                ec2.SubnetConfiguration(name="app", subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS, cidr_mask=22),
                ec2.SubnetConfiguration(name="data", subnet_type=ec2.SubnetType.PRIVATE_ISOLATED, cidr_mask=24),
            ],
        )
        vpc.add_flow_log("FlowLog")
        app_subnets = ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS)
        data_subnets = vpc.select_subnets(subnet_type=ec2.SubnetType.PRIVATE_ISOLATED)

        tasks_sg = ec2.SecurityGroup(self, "TasksSg", vpc=vpc, description="ECS tasks (app, mcp, litellm)")
        tasks_sg.add_ingress_rule(tasks_sg, ec2.Port.tcp(4000), "LiteLLM from app/mcp tasks")
        neptune_sg = ec2.SecurityGroup(self, "NeptuneSg", vpc=vpc, description="Neptune", allow_all_outbound=False)
        neptune_sg.add_ingress_rule(tasks_sg, ec2.Port.tcp(8182), "Neptune from ECS tasks")
        os_sg = ec2.SecurityGroup(self, "OpenSearchSg", vpc=vpc, description="OpenSearch", allow_all_outbound=False)
        os_sg.add_ingress_rule(tasks_sg, ec2.Port.tcp(443), "OpenSearch from ECS tasks")

        # ------------------------------------------------------------------ Neptune
        subnet_group = neptune.CfnDBSubnetGroup(
            self, "NeptuneSubnets",
            db_subnet_group_description="Graphiti Neptune subnets",
            subnet_ids=data_subnets.subnet_ids,
        )
        cluster = neptune.CfnDBCluster(
            self, "Neptune",
            iam_auth_enabled=True,
            storage_encrypted=True,
            deletion_protection=retain,
            backup_retention_period=7,
            db_subnet_group_name=subnet_group.ref,
            vpc_security_group_ids=[neptune_sg.security_group_id],
            engine_version=ctx("neptuneEngineVersion"),  # None -> region default
        )
        cluster.apply_removal_policy(data_removal)
        neptune.CfnDBInstance(
            self, "NeptuneWriter",
            db_instance_class=neptune_instance_class,
            db_cluster_identifier=cluster.ref,
        ).apply_removal_policy(data_removal)

        # ------------------------------------------------------------------ OpenSearch
        # OpenSearch Serverless is not available in China regions -> managed domain.
        domain = opensearch.Domain(
            self, "OpenSearch",
            version=opensearch.EngineVersion.open_search(ctx("opensearchVersion") or "2.19"),
            vpc=vpc,
            vpc_subnets=[ec2.SubnetSelection(subnets=data_subnets.subnets[:opensearch_nodes])],
            security_groups=[os_sg],
            capacity=opensearch.CapacityConfig(
                data_nodes=opensearch_nodes,
                data_node_instance_type=opensearch_instance_type,
                multi_az_with_standby_enabled=False,
            ),
            zone_awareness=opensearch.ZoneAwarenessConfig(
                enabled=opensearch_nodes > 1, availability_zone_count=min(opensearch_nodes, 2)
            ) if opensearch_nodes > 1 else opensearch.ZoneAwarenessConfig(enabled=False),
            ebs=opensearch.EbsOptions(volume_size=50, volume_type=ec2.EbsDeviceVolumeType.GP3),
            encryption_at_rest=opensearch.EncryptionAtRestOptions(enabled=True),
            node_to_node_encryption=True,
            enforce_https=True,
            tls_security_policy=opensearch.TLSSecurityPolicy.TLS_1_2,
            removal_policy=data_removal,
        )

        # ------------------------------------------------------------------ secrets
        provider_secret_arn = ctx("providerSecretArn")
        if provider_secret_arn:
            provider_secret = sm.Secret.from_secret_complete_arn(self, "ProviderKeys", provider_secret_arn)
        else:
            # Placeholders so ECS can resolve every JSON field; fill real keys after deploy:
            #   aws secretsmanager put-secret-value --secret-id <arn> --secret-string '{...}'
            provider_secret = sm.Secret(
                self, "ProviderKeys",
                description="Model provider API keys used by the LiteLLM proxy",
                secret_object_value={k: SecretValue.unsafe_plain_text("REPLACE_ME") for k in provider_key_names},
            )
        master_key = sm.Secret(
            self, "LiteLLMMasterKey",
            description="LiteLLM proxy master key (raw; containers prefix it with sk-)",
            generate_secret_string=sm.SecretStringGenerator(exclude_punctuation=True, password_length=40),
        )

        # ------------------------------------------------------------------ ECS
        ecs_cluster = ecs.Cluster(
            self, "Cluster", vpc=vpc, container_insights_v2=ecs.ContainerInsights.ENABLED
        )
        namespace = sd.PrivateDnsNamespace(self, "Namespace", name=NAMESPACE, vpc=vpc)

        # --- LiteLLM gateway
        litellm_role = iam.Role(self, "LiteLLMTaskRole", assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"))
        if sagemaker_endpoint_arns:
            litellm_role.add_to_policy(iam.PolicyStatement(
                actions=["sagemaker:InvokeEndpoint", "sagemaker:InvokeEndpointWithResponseStream"],
                resources=sagemaker_endpoint_arns,
            ))
        if bedrock_access:
            litellm_role.add_to_policy(iam.PolicyStatement(
                actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream", "bedrock:Rerank"],
                resources=["*"],
            ))
        litellm_td = ecs.FargateTaskDefinition(
            self, "LiteLLMTask", cpu=1024, memory_limit_mib=2048, task_role=litellm_role,
        )
        config_b64 = base64.b64encode(litellm_config.read_bytes()).decode()
        litellm_td.add_container(
            "litellm",
            image=_image(self, "LiteLLMRepo", litellm_image),
            port_mappings=[ecs.PortMapping(container_port=4000, name="litellm")],
            logging=ecs.LogDrivers.aws_logs(stream_prefix="litellm", log_retention=logs.RetentionDays.ONE_MONTH),
            environment={
                "LITELLM_CONFIG_B64": config_b64,
                "AWS_REGION_NAME": Aws.REGION,
                "LITELLM_LOG": "INFO",
                "LITELLM_TELEMETRY": "False",
                # Don't fetch the model cost map from GitHub at startup (slow / blocked from CN)
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            },
            secrets={
                "LITELLM_MASTER_KEY_RAW": ecs.Secret.from_secrets_manager(master_key),
                **{k: ecs.Secret.from_secrets_manager(provider_secret, k) for k in provider_key_names},
            },
            entry_point=["/bin/sh", "-c"],
            command=[
                "python3 -c \"import os,base64;open('/tmp/config.yaml','wb')"
                ".write(base64.b64decode(os.environ['LITELLM_CONFIG_B64']))\" && "
                "export LITELLM_MASTER_KEY=sk-$LITELLM_MASTER_KEY_RAW && "
                "exec litellm --config /tmp/config.yaml --port 4000 --num_workers 2"
            ],
            health_check=ecs.HealthCheck(
                command=["CMD-SHELL", "python3 -c \"import urllib.request;urllib.request.urlopen('http://localhost:4000/health/liveliness')\""],
                start_period=Duration.seconds(60),
            ),
        )
        litellm_svc = ecs.FargateService(
            self, "LiteLLMService",
            cluster=ecs_cluster, task_definition=litellm_td, desired_count=2,
            vpc_subnets=app_subnets, security_groups=[tasks_sg],
            cloud_map_options=ecs.CloudMapOptions(
                cloud_map_namespace=namespace, name="litellm", dns_record_type=sd.DnsRecordType.A,
                dns_ttl=Duration.seconds(10),
            ),
            circuit_breaker=ecs.DeploymentCircuitBreaker(rollback=True),
            min_healthy_percent=50,
        )

        # --- shared app role: Neptune IAM auth + OpenSearch IAM
        app_role = iam.Role(self, "AppTaskRole", assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"))
        app_role.add_to_policy(iam.PolicyStatement(
            actions=["neptune-db:connect", "neptune-db:ReadDataViaQuery", "neptune-db:WriteDataViaQuery",
                     "neptune-db:DeleteDataViaQuery", "neptune-db:GetQueryStatus", "neptune-db:CancelQuery",
                     # langchain_aws NeptuneGraph reads the property-graph summary on init
                     "neptune-db:GetGraphSummary"],
            resources=[f"arn:{Aws.PARTITION}:neptune-db:{Aws.REGION}:{Aws.ACCOUNT_ID}:{cluster.attr_cluster_resource_id}/*"],
        ))
        # Identity-based policy only; no domain resource (access) policy
        domain.grant_read_write(app_role)

        app_env = {
            "MEMORY_FRAMEWORK": "graphiti",
            "AWS_REGION": Aws.REGION,
            "AWS_DEFAULT_REGION": Aws.REGION,
            "GRAPHITI_NEPTUNE_ENDPOINT": f"neptune-db://{cluster.attr_endpoint}",
            "GRAPHITI_NEPTUNE_PORT": cluster.attr_port,
            "GRAPHITI_OPENSEARCH_ENDPOINT": domain.domain_endpoint,
            "GRAPHITI_OPENSEARCH_SERVICE": "es",
            "LLM_BASE_URL": f"http://litellm.{NAMESPACE}:4000/v1",
            "EMBEDDING_DIM": embedding_dim,
            "RERANK_MODEL": rerank_model,
            "AGENT_CONFIG": ctx("agentConfig") or "default",
        }
        app_secrets = {"LLM_API_KEY_RAW": ecs.Secret.from_secrets_manager(master_key)}

        app_image_obj = _image(self, "AppRepo", app_image)

        def app_container(td: ecs.FargateTaskDefinition, name: str, port: int, cmd: str, health: str):
            td.add_container(
                name,
                image=app_image_obj,
                port_mappings=[ecs.PortMapping(container_port=port, name=name)],
                logging=ecs.LogDrivers.aws_logs(stream_prefix=name, log_retention=logs.RetentionDays.ONE_MONTH),
                environment=app_env,
                secrets=app_secrets,
                entry_point=["/bin/sh", "-c"],
                command=[f"export LLM_API_KEY=sk-$LLM_API_KEY_RAW && exec {cmd}"],
                health_check=ecs.HealthCheck(
                    command=["CMD-SHELL", f"python -c \"import urllib.request;urllib.request.urlopen('{health}')\""],
                    start_period=Duration.seconds(90),
                ),
            )

        # --- Streamlit app
        app_td = ecs.FargateTaskDefinition(self, "AppTask", cpu=1024, memory_limit_mib=2048, task_role=app_role)
        app_container(app_td, "app", 8501,
                      "streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true",
                      "http://localhost:8501/_stcore/health")
        app_svc = ecs.FargateService(
            self, "AppService",
            cluster=ecs_cluster, task_definition=app_td, desired_count=1,
            vpc_subnets=app_subnets, security_groups=[tasks_sg],
            circuit_breaker=ecs.DeploymentCircuitBreaker(rollback=True),
            health_check_grace_period=Duration.seconds(120),
            min_healthy_percent=100,
            enable_execute_command=enable_exec,
        )
        app_svc.node.add_dependency(litellm_svc)

        # Internal ALB by default: public 80/443 in China regions requires an ICP filing.
        alb_sg = ec2.SecurityGroup(self, "AlbSg", vpc=vpc, description="App ALB")
        alb_cidr = ctx("albIngressCidr") or (vpc.vpc_cidr_block if not public_alb else None)
        if not alb_cidr:
            raise ValueError("publicAlb=true requires -c albIngressCidr=<your office CIDR>")
        alb_sg.add_ingress_rule(ec2.Peer.ipv4(alb_cidr), ec2.Port.tcp(80), "Streamlit UI")
        alb = elbv2.ApplicationLoadBalancer(
            self, "Alb", vpc=vpc, internet_facing=public_alb, security_group=alb_sg,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC if public_alb else ec2.SubnetType.PRIVATE_WITH_EGRESS),
            idle_timeout=Duration.seconds(300),  # Streamlit websocket + long agent turns
        )
        tasks_sg.add_ingress_rule(alb_sg, ec2.Port.tcp(8501), "ALB to Streamlit")
        listener = alb.add_listener("Http", port=80, open=False)
        listener.add_targets(
            "App", port=8501, protocol=elbv2.ApplicationProtocol.HTTP, targets=[app_svc],
            health_check=elbv2.HealthCheck(path="/_stcore/health", healthy_http_codes="200"),
            stickiness_cookie_duration=Duration.hours(8),
        )

        # --- MCP server (streamable-http), VPC-internal via Cloud Map
        mcp_td = ecs.FargateTaskDefinition(self, "McpTask", cpu=512, memory_limit_mib=1024, task_role=app_role)
        app_container(mcp_td, "mcp", 8000, "python mcp_server.py", "http://localhost:8000/healthz")
        mcp_td.find_container("mcp").add_environment("MCP_TRANSPORT", "streamable-http")
        mcp_cidr = ctx("mcpIngressCidr") or vpc.vpc_cidr_block
        tasks_sg.add_ingress_rule(ec2.Peer.ipv4(mcp_cidr), ec2.Port.tcp(8000), "MCP clients")
        mcp_svc = ecs.FargateService(
            self, "McpService",
            cluster=ecs_cluster, task_definition=mcp_td, desired_count=1,
            vpc_subnets=app_subnets, security_groups=[tasks_sg],
            cloud_map_options=ecs.CloudMapOptions(cloud_map_namespace=namespace, name="mcp",
                                                  dns_record_type=sd.DnsRecordType.A, dns_ttl=Duration.seconds(10)),
            circuit_breaker=ecs.DeploymentCircuitBreaker(rollback=True),
            min_healthy_percent=100,
        )
        mcp_svc.node.add_dependency(litellm_svc)

        # ------------------------------------------------------------------ outputs
        CfnOutput(self, "AppUrl", value=f"http://{alb.load_balancer_dns_name}")
        CfnOutput(self, "McpUrl", value=f"http://mcp.{NAMESPACE}:8000/mcp")
        CfnOutput(self, "LiteLLMUrl", value=f"http://litellm.{NAMESPACE}:4000/v1")
        CfnOutput(self, "NeptuneEndpoint", value=cluster.attr_endpoint)
        CfnOutput(self, "OpenSearchEndpoint", value=domain.domain_endpoint)
        CfnOutput(self, "ProviderKeysSecretArn", value=provider_secret.secret_arn)
        CfnOutput(self, "LiteLLMMasterKeySecretArn", value=master_key.secret_arn)
        CfnOutput(self, "ClusterName", value=ecs_cluster.cluster_name)
