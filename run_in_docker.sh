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
device_args=(--gpus "device=${GPU_DEVICE:-0}")
for arg in "$@"; do
    case "$arg" in
        --cpu-filters|--cpu-decode|--list|--prepare-only)
            device_args=(--env NVIDIA_VISIBLE_DEVICES=void --env CUDA_VISIBLE_DEVICES=)
            break ;;
    esac
done
exec docker run --rm -i "${device_args[@]}" --shm-size=1g \
    --user "$uid_gid" --env HOME=/tmp --env PYTHONDONTWRITEBYTECODE=1 \
    --volume "$root:/workspace/gpt-neox" --workdir /workspace/gpt-neox \
    "${GPT_NEOX_IMAGE:-gpt-neox:latest}" bash -seu -o pipefail -- "$@" <<'IN_CONTAINER'
# Keep workarounds in the disposable container, writable even for a non-root UID.
compat=$(mktemp -d /tmp/neox-inference-compat.XXXXXX)
export PYTHONUSERBASE="$compat/user"
export PYTHONPATH="$compat${PYTHONPATH:+:$PYTHONPATH}"

# NeoX imports W&B even when logging is disabled. The standard image's recent
# W&B fails to import with protobuf 3.20; reuse the repository's TE image pin.
if ! python3 -c 'import wandb' >/dev/null 2>&1; then
    python3 -m pip install --user --disable-pip-version-check 'wandb==0.16.6'
    python3 -c 'import wandb'
fi

python3 - "$compat" <<'PY'
import shutil
import sys
from importlib.metadata import distribution
from pathlib import Path

ds = distribution("deepspeed")
if ds.version == "0.16.5+91d1c55b":
    package = Path(ds.locate_file("deepspeed"))
    engine = Path("runtime/engine.py")
    original = (package / engine).read_text()
    old = """        if load_module_only:
            deepspeed_states = ['module']
            if self.optimizer is not None:
                self.optimizer.refresh_fp32_params()"""
    new = old.replace("if self.optimizer is not None:",
                      "if self.optimizer is not None and hasattr(self.optimizer, 'refresh_fp32_params'):")
    if new not in original:
        if original.count(old) != 1:
            sys.exit("Unexpected DeeperSpeed checkpoint loader; refusing to patch it.")
        # Backport the upstream v0.17.0 guard; leave installed site-packages alone.
        overlay = Path(sys.argv[1]) / "deepspeed"
        shutil.copytree(package, overlay, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (overlay / engine).write_text(original.replace(old, new, 1))
        print("Applied DeeperSpeed weights-only checkpoint guard in", overlay, flush=True)
PY
exec python3 logs/Inference_Testing/run_matrix.py "$@"
IN_CONTAINER
