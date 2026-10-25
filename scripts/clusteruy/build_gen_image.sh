#!/usr/bin/env bash
# Build the slm-thesis generation Docker image (linux/amd64) from the repo root.
# See docs/clusteruy.md — generation image section.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-srobaina99/slm-thesis:latest}"

cd "$REPO_ROOT"

echo "Building $IMAGE_TAG (platform=linux/amd64) from $REPO_ROOT"
echo "Dockerfile: scripts/clusteruy/Dockerfile"
echo "This compiles llama-cpp-python for sm_60 and downloads torch + bert-base-cased + NLTK (large)."

docker build \
  --platform=linux/amd64 \
  -f scripts/clusteruy/Dockerfile \
  -t "$IMAGE_TAG" \
  .

echo "Done: $IMAGE_TAG"
docker image ls "$IMAGE_TAG"
