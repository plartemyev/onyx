#!/bin/bash
# Daily SearXNG image update: pull latest, recreate container if image changed.
# Runs via onyx-searxng-update.timer; logs to the systemd journal.
set -euo pipefail

cd /home/nameless/onyx/deployment/docker_compose

echo "[$(date --iso-8601=seconds)] pulling searxng image"
docker compose pull searxng

echo "[$(date --iso-8601=seconds)] applying (recreates container only if image changed)"
docker compose up -d searxng

echo "[$(date --iso-8601=seconds)] done"
