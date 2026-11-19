#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "$0")/../.." && pwd)
# In rootless Docker, container UID 0 maps to the invoking user on the host.
# In rootful Docker, pass the host UID instead so artifacts remain user-owned.
security=$(docker info --format '{{json .SecurityOptions}}')
if [[ "$security" == *rootless* ]]; then
    uid_gid=0:0
else
    uid_gid="$(id -u):$(id -g)"
fi
exec docker run --rm --gpus "device=${GPU_DEVICE:-0}" --shm-size=1g \
    --user "$uid_gid" --env HOME=/tmp --env PYTHONDONTWRITEBYTECODE=1 \
    --volume "$root:/workspace/gpt-neox" --workdir /workspace/gpt-neox \
    "${GPT_NEOX_IMAGE:-gpt-neox:latest}" python3 logs/Inference_Testing/run_matrix.py "$@"
