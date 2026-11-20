# Inference boundary tests

From the NeoX root, with this repo checked out at `logs/Inference_Testing`:

```bash
bash logs/Inference_Testing/run_in_docker.sh --cpu-all
# Or, in an environment with NeoX's requirements and pytest installed:
python logs/Inference_Testing/run_matrix.py --cpu-all
```

This runs both fixed-logit matrices and the pytest suites below with CUDA hidden.
It exits nonzero on a failure. Tests use temporary files, not saved answers from
older runs. The flag takes no case selector: use `--case NAME --cpu-decode` to
rerun an individual golden case, or pytest's `-k` to select an edge regression.

| Area | What is checked |
| --- | --- |
| Filters and order | All 128 enabled/disabled subsets of the seven filters on a noncontiguous three-row tensor. A separate scalar probability/entropy calculation predicts the eligible IDs; remaining logit values must be unchanged. |
| Numeric boundaries | Exact cumulative mass, tied maxima, inclusive cutoffs, top-k above vocabulary size, top-h's top-100 cap at widths 99/100/101/512, singleton vocabularies and pre-masked scores in FP16/BF16/FP32/FP64. |
| Validation | Negative, nonfinite, boolean, string, fractional-count and out-of-range values; neutral defaults; inconsistent min/max. The CLI validates at config load without repeating warnings; Python and lm-eval requests validate their own options before any forward pass. |
| Generation | Cached and recomputed paths, returned scores on/off, no prompts, empty text, zero/one/inferred budgets, exact-fill and overlong contexts, stop groups, ragged prompts, per-row EOS masks, and early-EOS score lengths. |
| RNG and modes | Each of the 25 golden configurations runs on 20 seeds. Separate tests compare draws to a seeded PyTorch reference at low and high temperatures, and run lm-eval greedy/sampling requests through the real decoder. |
| CLI helpers and lifecycle | Delimiter splitting, duplicate prompts/order, output overwrite and empty files, unconditional generation, interactive EOF, and cache cleanup on exhaustion, close and exceptions. |
| Tensor parallelism | Two real CPU Gloo processes gather disjoint vocabulary shards and agree on generated tokens. A non-source rank must not advance its RNG. Both cache modes and score output modes are checked. |

The edge tests are [test_filter_edges.py](test_filter_edges.py) and
[test_decode_edges.py](test_decode_edges.py); the runner also invokes NeoX's
existing config, decoder, and eval-adapter regression tests. The Gloo fixture
uses a fixed model head: it exercises actual collectives, not DeepSpeed's GPU
layers, checkpoint loading, or NCCL. The 128 subsets cover all on/off
combinations, not every possible real-valued threshold, model or prompt.

## Regression cases

These four cases caused six failures on NeoX `866e5e1` even though all 25
golden configurations passed. On [NeoX `1060c99`](https://github.com/aflah02/gpt-neox/commit/1060c99)
all 394 CPU tests pass, as do the 25 filter cases and 500 seeded golden
generations. The regressions remain ordinary assertions:

- With four equiprobable tokens, `top_p=0.5` must retain exactly two. A third
  token is needed at `0.51`, but not at `0.49`. See the [HF cutoff rule](https://github.com/huggingface/transformers/blob/v5.14.0/src/transformers/generation/logits_process.py).
- In a batch with unequal prompt lengths, a stop sequence must be generated
  entirely by that row. A match inside its prompt or across the prompt boundary
  must not stop it; two generated stop tokens must.
- `temperature=1e-300` and `5e-324` must still sample. With EOS masked and
  two tied maxima, the decoder's probabilities, token draw and post-draw RNG
  state are compared to an independent 50/50 multinomial draw over 20 seeds.
- `top_k: true`, `temperature: "0.5"`, and `minimum_tokens: 1.5` must raise
  a `ValueError` naming the offending field before model setup.

## GPU checks

On a GPU host, run the standalone matrix **without** a CPU flag to exercise
checkpoint loading and file generation. After changes to config handling, a
shorter integration run covers greedy decoding with ignored filters, sampling
with all seven filters, and vocabulary-dependent top-k truncation. The driver
checks that each expected warning appears exactly once across launcher and worker:

```bash
bash logs/Inference_Testing/run_in_docker.sh --case greedy_all_filters --case combined --case top_k_over_vocab
```

To compare real CUDA engines against single-rank outputs, run
`python -m pytest -q -rs tests/model/test_model_generation.py -k matches_single_rank`
with NeoX's GPU dependencies installed. Two GPUs cover non-pipeline TP=2 and
the pipeline engine with PP=2, TP=2, or DP=2 separately. The combined PP=2 ×
TP=2 test requires four GPUs and is reported as skipped on a two-GPU host.

On NeoX `1060c99`, all three checkpoint cases and all four supported NCCL
layouts passed on a two-A100 host; only the four-GPU layout was skipped.
