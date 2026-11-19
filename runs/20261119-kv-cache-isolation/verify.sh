#!/usr/bin/env bash
set -euo pipefail
suite=logs/Inference_Testing
run=$(dirname -- "$0")
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export NVIDIA_TF32_OVERRIDE=0
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
source <(sed -n '/^# Keep workarounds in the disposable container/,/^PY$/p' "$suite/run_in_docker.sh")
python3 -m pytest -q -p no:cacheprovider tests/unit/test_text_generation_utils.py
python3 deepy.py "$run/check_cache.py" "$suite/configs/tiny.json"
