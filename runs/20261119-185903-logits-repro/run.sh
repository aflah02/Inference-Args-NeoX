#!/usr/bin/env bash
# Run from the checkout root, inside gpt-neox:latest.
set -euo pipefail
suite=logs/Inference_Testing
out=$(dirname -- "$0")
# Use the same disposable W&B and checkpoint-loading workarounds as the passing
# single-token matrix. No change is made to the generator or to its logits.
source <(sed -n '/^# Keep workarounds in the disposable container/,/^PY$/p' "$suite/run_in_docker.sh")
: "${compat:?missing launcher compatibility setup}"

generate() {
    local case=$1
    python3 deepy.py generate.py "$suite/configs/tiny.json" "$out/generation.json" \
        "$out/$case.json" --sample_output_file "$out/$case.jsonl" >"$out/$case.log" 2>&1
}
on=0; generate logits_on || on=$?
echo "logits_on: exit $on"
off=0; generate logits_off || off=$?
echo "logits_off: exit $off"
(( on != 0 && off == 0 ))
grep -F 'IndexError: index 8 is out of bounds for dimension 0 with size 8' "$out/logits_on.log"
python3 - "$out/logits_off.jsonl" <<'PY'
import json
import sys
with open(sys.argv[1]) as f:
    record = json.load(f)
print({k: record[k] for k in ("context", "text", "length", "finished")})
assert (record["context"], record["text"], record["length"], record["finished"]) == ("QQ", "AAAAAAAA", 8, False)
PY
