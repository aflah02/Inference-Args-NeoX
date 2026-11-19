# Inference sampling smoke tests

This matrix exercises the real `deepy.py generate.py` input-file path on one
CUDA GPU. It creates a native checkpoint locally: two GPT-NeoX layers, hidden
size 64, four attention heads, FP32, and the built-in 512-token character
tokenizer. No training, dataset, pretrained weights, or W&B login is needed.

## Running

From the root of this checkout, you can inspect the cases and render all the
configs on a CPU machine with only Python 3:

```bash
python3 logs/Inference_Testing/run_matrix.py --list
python3 logs/Inference_Testing/run_matrix.py --prepare-only
```

With NeoX dependencies installed but **no GPU**, use `--cpu-filters` to check
the actual logit processors, including two rows on opposite sides of the EOS
boundary. This does not exercise checkpoint loading or decoding. For example:

```bash
docker run --rm -v "$PWD:/workspace/gpt-neox:ro" -w /workspace/gpt-neox \
  gpt-neox:latest python3 logs/Inference_Testing/run_matrix.py --cpu-filters
```

On a machine with one GPU visible and GPT-NeoX dependencies installed:

```bash
python3 logs/Inference_Testing/run_matrix.py
# Or run just two cases:
python3 logs/Inference_Testing/run_matrix.py --case minimum_tokens --case eta_cutoff
```

With the standard Docker image, the equivalent full run is:

```bash
bash logs/Inference_Testing/run_in_docker.sh
# Optional: GPU_DEVICE=1 GPT_NEOX_IMAGE=gpt-neox:latest bash logs/Inference_Testing/run_in_docker.sh --case combined
```

The wrapper defaults to GPU 0 and `gpt-neox:latest`, and selects the host user's
mapped UID for both rootless and rootful Docker. The checkout must be writable:
NeoX compiles its dataset helper in `megatron/data/` at initialization.
Each invocation gets a fresh `runs/run-*/` directory with rendered configs.
A live run also writes the checkpoint, per-case logs and JSONL, and
`summary.tsv`; a failed assertion or launch makes the runner exit nonzero.
Use `--output-dir PATH` to choose a **new** directory. Paths in the rendered
configs belong to the environment that rendered them; run the driver inside
the container if that is where you will test. This suite lives under the
repository's ignored `logs/` tree.

## Where the ground truth comes from

The expected outputs are calculated **before** running inference. They are not
samples saved from a previous NeoX run, and the suite does not run Hugging Face
as a reference. [The checkpoint builder](make_checkpoint.py) makes the final
LayerNorm output the fixed vector `[1, 0, …, 0]`. The output layer therefore
reads just the first column of its weights at every decode step. That column
contains `log(p)` for EOS, A, B, C, D, with `p = [0.55, 0.25, 0.12, 0.06, 0.02]`;
all other logits are `-50` and contribute negligible probability. The two
transformer layers still execute, but neither the prompt nor a previous random
draw can change the next step's unfiltered scores.

For example, take token B at probability **0.12**. Epsilon sampling at 0.14
removes it because `0.12 < 0.14`. For eta sampling at the *same* setting,
the distribution's entropy is `H = -sum(p * ln(p)) ≈ 1.17686`. The effective
cutoff is `min(0.14, sqrt(0.14) * exp(-H)) ≈ 0.11533`, so B survives.
That is why [the expected masks](configs/cases.json) are `$A` for epsilon
and `$AB` for eta (`$` is EOS). These masks are inputs to the test, not
computed by the sampler being tested.

## What a pass means

Each GPU case uses the one-token prompt `Q` in a fresh process with KV caching
enabled. In greedy mode EOS wins immediately. With `minimum_tokens=3`, EOS
is ineligible for the first three draws, so A wins each time. On draw four
EOS wins again: the exact completion is `AAA`, length 3, finished. EOS is not
included in the returned text.

For stochastic sampling there is generally **no single correct string**.
[The verifier](run_matrix.py) checks every vocabulary logit at every executed
step: eligible tokens retain their known score (rescaled if temperature is
set), and all others must be `-inf`. It also checks that each draw was eligible
and that generation stopped at EOS or the eight-token cap. Unused logit rows
after termination are ignored. A pass establishes these behaviors for this
fixture, not language quality, sampling-frequency calibration, or full HF parity.

| Case | Expected behavior |
| --- | --- |
| Greedy / minimum tokens | Greedy stops immediately. With a minimum of 3 it emits `AAA`, then EOS. With a minimum of 8 it hits the cap without emitting EOS. |
| Temperature / top-k / top-p | Temperature 0.8 rescales scores; top-k 2 keeps EOS and A; top-p 0.85 adds B, the token crossing the mass threshold. |
| Min-p / top-h | Min-p 0.1 keeps EOS, A, B, C. Top-h 0.6 keeps only EOS and A. |
| Typical-p | At 0.2 only A survives, **not** the most likely token, so generation reaches the cap. |
| Epsilon / eta | At the same cutoff (0.14), epsilon removes B (0.12), but the entropy-adjusted eta threshold keeps it. |
| Combined | With all filters enabled, only A/B may be sampled for two steps; after EOS is unmasked, only EOS/A remain. |

See [the matrix](configs/cases.json) for settings and expected masks:
`$` means EOS, `*` all tokens, and `!$` all tokens except EOS. Cutoffs are
deliberately large enough to distinguish behavior on this tiny vocabulary;
they are not tuning recommendations for a trained model. Beam search is not
implemented and is not part of the matrix.
