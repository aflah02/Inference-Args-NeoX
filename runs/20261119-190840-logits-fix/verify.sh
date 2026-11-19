#!/usr/bin/env bash
set -euo pipefail
suite=logs/Inference_Testing
before=$suite/runs/20261119-185903-logits-repro
out=$(dirname -- "$0")
source <(sed -n '/^# Keep workarounds in the disposable container/,/^PY$/p' "$suite/run_in_docker.sh")
: "${compat:?missing launcher compatibility setup}"

for case in logits_on logits_off; do
    python3 deepy.py generate.py "$suite/configs/tiny.json" "$before/generation.json" \
        "$before/$case.json" --sample_output_file "$out/$case.jsonl" >"$out/$case.log" 2>&1
    echo "$case: exit 0"
done

python3 - "$out" <<'PY'
import json, math, sys
from pathlib import Path
d = Path(sys.argv[1])
on, off = (json.loads((d / (case + ".jsonl")).read_text()) for case in ("logits_on", "logits_off"))
for row in (on, off):
    assert (row["context"], row["text"], row["length"], row["finished"]) == ("QQ", "AAAAAAAA", 8, False)
assert "logits" not in off
scores = on["logits"]
assert len(scores) == 8
expected = [-50.] * 512
expected[0] = -math.inf
for token, probability in zip("ABCD", [.25, .12, .06, .02]):
    expected[ord(token)] = math.log(probability)
for i, row in enumerate(scores):
    assert len(row) == len(expected)
    assert all(math.isclose(s, e, rel_tol=1e-3, abs_tol=1e-5) for s, e in zip(row, expected)), i
print("Verified QQ -> AAAAAAAA with logits on and off, including all 8 x 512 scores with EOS masked.")
PY
python3 -m pytest -q -p no:cacheprovider tests/unit/test_text_generation_utils.py
