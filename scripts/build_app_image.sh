#!/usr/bin/env bash
# Build the app image and push it to ECR in an AWS China region.
#   AWS_PROFILE=cn REGION=cn-northwest-1 TAG=v1 ./scripts/build_app_image.sh
# Inside China, use the mirrored base image and a domestic PyPI mirror:
#   PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ USE_MIRROR_BASE=1 ./scripts/build_app_image.sh
set -euo pipefail
cd "$(dirname "$0")/.."

REGION="${REGION:-${AWS_REGION:-cn-northwest-1}}"
TAG="${TAG:-latest}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text --region "$REGION")"
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com.cn"
REPO="graphiti-agent-memory"
BASE_IMAGE="public.ecr.aws/docker/library/python:3.12-slim"
[[ "${USE_MIRROR_BASE:-0}" == "1" ]] && BASE_IMAGE="${REGISTRY}/mirror/python:3.12-slim"

aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPO" --region "$REGION" \
       --image-scanning-configuration scanOnPush=true >/dev/null
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"

docker build --platform linux/amd64 \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}" \
  -t "${REGISTRY}/${REPO}:${TAG}" .
docker push "${REGISTRY}/${REPO}:${TAG}"
echo "appImage=${REGISTRY}/${REPO}:${TAG}"
