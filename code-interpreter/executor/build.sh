#!/usr/bin/env bash
# Build the Arch-based research executor image and tag it for the compose
# deployment. Run from anywhere; images are only used locally (--pull never).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TAG="${1:-code-interpreter-executor:arch-20260920}"
echo "Building research executor image: $TAG"
docker build -t "$TAG" .
echo "Built $TAG"
