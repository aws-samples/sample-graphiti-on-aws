#!/usr/bin/env bash
# Mirror upstream images into ECR in an AWS China region.
# Fargate in cn-north-1 / cn-northwest-1 cannot reliably pull from ghcr.io / docker.io,
# so every image the stack runs must live in your own ECR (*.amazonaws.com.cn).
#
# Run on a machine that can reach ghcr.io AND has credentials for the China account
# (a global-region EC2 / laptop with an aws-cn profile works). Requires docker,
# or skopeo (set USE_SKOPEO=1) which copies registry-to-registry without a local daemon.
#
#   AWS_PROFILE=cn REGION=cn-northwest-1 LITELLM_TAG=main-stable ./scripts/mirror_images.sh
set -euo pipefail

REGION="${REGION:-${AWS_REGION:-cn-northwest-1}}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text --region "$REGION")"
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com.cn"
LITELLM_TAG="${LITELLM_TAG:-main-stable}"   # pin a release tag for production

IMAGES=(
  "ghcr.io/berriai/litellm:${LITELLM_TAG}|mirror/litellm:${LITELLM_TAG}"
  "public.ecr.aws/docker/library/python:3.12-slim|mirror/python:3.12-slim"
)

if [[ "${USE_SKOPEO:-0}" == "1" ]]; then
  PASS="$(aws ecr get-login-password --region "$REGION")"
else
  aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"
fi

for pair in "${IMAGES[@]}"; do
  src="${pair%%|*}"; dst_repo_tag="${pair##*|}"; repo="${dst_repo_tag%%:*}"
  aws ecr describe-repositories --repository-names "$repo" --region "$REGION" >/dev/null 2>&1 \
    || aws ecr create-repository --repository-name "$repo" --region "$REGION" \
         --image-scanning-configuration scanOnPush=true >/dev/null
  dst="${REGISTRY}/${dst_repo_tag}"
  echo ">> $src -> $dst"
  if [[ "${USE_SKOPEO:-0}" == "1" ]]; then
    skopeo copy --override-os linux --override-arch amd64 --dest-creds "AWS:${PASS}" "docker://${src}" "docker://${dst}"
  else
    docker pull --platform linux/amd64 "$src"
    docker tag "$src" "$dst"
    docker push "$dst"
  fi
done

echo
echo "litellmImage=${REGISTRY}/mirror/litellm:${LITELLM_TAG}"
echo "pythonBase=${REGISTRY}/mirror/python:3.12-slim"
