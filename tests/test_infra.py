"""CDK template assertions for the China-region stack (synth only, no AWS calls)."""
import json
import os
import sys

import pytest

aws_cdk = pytest.importorskip("aws_cdk")
from aws_cdk import assertions  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "infra"))
from stacks.agent_memory_stack import AgentMemoryStack  # noqa: E402

ACCOUNT, REGION = "123456789012", "cn-northwest-1"
REG = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com.cn"


@pytest.fixture(scope="module")
def template():
    app = aws_cdk.App(context={
        "litellmImage": f"{REG}/mirror/litellm:main-stable",
        "appImage": f"{REG}/graphiti-agent-memory:v1",
        # pre-seed the AZ lookup so synth needs no credentials
        f"availability-zones:account={ACCOUNT}:region={REGION}": [f"{REGION}a", f"{REGION}b", f"{REGION}c"],
    })
    stack = AgentMemoryStack(app, "T", env=aws_cdk.Environment(account=ACCOUNT, region=REGION))
    return assertions.Template.from_stack(stack)


def test_no_services_missing_in_china(template):
    types = {r["Type"] for r in template.to_json()["Resources"].values()}
    assert not {t for t in types if t.startswith(("AWS::Bedrock", "AWS::OpenSearchServerless", "AWS::NeptuneGraph"))}


def test_no_hardcoded_global_partition(template):
    assert "arn:aws:" not in json.dumps(template.to_json())


def test_neptune_iam_auth_and_opensearch_domain(template):
    template.has_resource_properties("AWS::Neptune::DBCluster", {"IamAuthEnabled": True, "StorageEncrypted": True})
    template.has_resource_properties("AWS::OpenSearchService::Domain", {
        "EncryptionAtRestOptions": {"Enabled": True},
        "NodeToNodeEncryptionOptions": {"Enabled": True},
        "VPCOptions": assertions.Match.any_value(),
    })


def test_alb_internal_by_default(template):
    template.has_resource_properties("AWS::ElasticLoadBalancingV2::LoadBalancer", {"Scheme": "internal"})


def test_app_points_at_gateway_and_es(template):
    tds = template.find_resources("AWS::ECS::TaskDefinition")
    envs = [
        {e["Name"]: e.get("Value") for e in c.get("Environment", [])}
        for td in tds.values() for c in td["Properties"]["ContainerDefinitions"]
    ]
    app_env = next(e for e in envs if "LLM_BASE_URL" in e)
    assert app_env["LLM_BASE_URL"] == "http://litellm.agent-memory.local:4000/v1"
    assert app_env["GRAPHITI_OPENSEARCH_SERVICE"] == "es"
    assert any("LITELLM_CONFIG_B64" in e for e in envs)


def test_images_pulled_from_cn_ecr_with_pull_permissions(template):
    policies = json.dumps(template.find_resources("AWS::IAM::Policy"))
    assert "ecr:BatchGetImage" in policies and "ecr:GetAuthorizationToken" in policies


def test_ecs_exec_is_opt_in(template):
    assert "ssmmessages:" not in json.dumps(template.find_resources("AWS::IAM::Policy"))


def test_context_lists_accept_cli_strings():
    from stacks.agent_memory_stack import _str_list
    assert _str_list('["arn:a", "arn:b"]') == ["arn:a", "arn:b"]
    assert _str_list("A, B") == ["A", "B"]
    assert _str_list(["A"]) == ["A"] and _str_list(None) == []
