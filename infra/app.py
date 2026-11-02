#!/usr/bin/env python3
import os

import aws_cdk as cdk

from stacks.agent_memory_stack import AgentMemoryStack

app = cdk.App()
AgentMemoryStack(
    app, app.node.try_get_context("stackName") or "GraphitiAgentMemoryCn",
    env=cdk.Environment(
        account=app.node.try_get_context("account") or os.getenv("CDK_DEFAULT_ACCOUNT"),
        region=app.node.try_get_context("region") or os.getenv("CDK_DEFAULT_REGION", "cn-northwest-1"),
    ),
    description="Graphiti agent memory on Neptune + OpenSearch, models via LiteLLM proxy (AWS China)",
)
app.synth()
