#!/usr/bin/env bash
# Run a one-off command inside the VPC, using the app service's task definition and
# network config, then print its logs. Neptune, OpenSearch and LiteLLM are only
# reachable from inside the VPC, so this is how the smoke test (or any ad-hoc script)
# is run without a bastion. Unlike `aws ecs execute-command` it needs no SSM plugin.
#
#   AWS_PROFILE=cn AWS_REGION=cn-northwest-1 ./scripts/run_in_vpc.sh                # smoke test
#   AWS_PROFILE=cn AWS_REGION=cn-northwest-1 ./scripts/run_in_vpc.sh "python my.py"
set -euo pipefail

STACK="${STACK:-GraphitiAgentMemoryCn}"
CMD="${1:-python scripts/smoke_test.py}"

C=$(aws cloudformation describe-stacks --stack-name "$STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='ClusterName'].OutputValue" --output text)
SVC=$(aws ecs list-services --cluster "$C" --query 'serviceArns[?contains(@, `AppService`)]|[0]' --output text)
TD=$(aws ecs describe-services --cluster "$C" --services "$SVC" --query 'services[0].taskDefinition' --output text)
NETCFG=$(aws ecs describe-services --cluster "$C" --services "$SVC" --query 'services[0].networkConfiguration' --output json)
LOG_GROUP=$(aws ecs describe-task-definition --task-definition "$TD" \
  --query 'taskDefinition.containerDefinitions[0].logConfiguration.options."awslogs-group"' --output text)

# The container's entrypoint derives LLM_API_KEY from the injected master-key secret
OVERRIDES=$(python3 -c 'import json,sys; print(json.dumps({"containerOverrides": [{"name": "app",
  "command": ["export LLM_API_KEY=sk-$LLM_API_KEY_RAW && exec " + sys.argv[1]]}]}))' "$CMD")

TASK=$(aws ecs run-task --cluster "$C" --task-definition "$TD" --launch-type FARGATE \
  --network-configuration "$NETCFG" --overrides "$OVERRIDES" --query 'tasks[0].taskArn' --output text)
echo ">> $TASK: $CMD" >&2
aws ecs wait tasks-stopped --cluster "$C" --tasks "$TASK"

aws logs get-log-events --log-group-name "$LOG_GROUP" --log-stream-name "app/app/${TASK##*/}" \
  --start-from-head --query 'events[].message' --output text | tr '\t' '\n'
EXIT=$(aws ecs describe-tasks --cluster "$C" --tasks "$TASK" --query 'tasks[0].containers[0].exitCode' --output text)
echo ">> exit=$EXIT" >&2
[[ "$EXIT" == "0" ]]
