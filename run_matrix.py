"""Prepare/run a one-GPU sampling smoke test. This driver uses only the stdlib."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROMPT = HERE / "prompt.txt"
CASES = json.loads((HERE / "configs/cases.json").read_text())
LIMIT = 8
VOCAB = 512
PROBS = dict(zip([0, *map(ord, "ABCD")], [0.55, 0.25, 0.12, 0.06, 0.02]))
RAW = [math.log(PROBS[i]) if i in PROBS else -50.0 for i in range(VOCAB)]


def expected_ids(spec):
    if spec == "*":
        return set(range(VOCAB))
    if spec == "!$":
        return set(range(1, VOCAB))
    return {0 if c == "$" else ord(c) for c in spec}


def check_scores(scores, allowed, temperature, step):
    if len(scores) != VOCAB:
        raise ValueError(f"step {step}: expected {VOCAB} logits, got {len(scores)}")
    for token, score in enumerate(scores):
        expected = RAW[token] / temperature if temperature > 0 else RAW[token]
        if token not in allowed:
            expected = -math.inf
        if not math.isclose(score, expected, rel_tol=1e-3, abs_tol=1e-5):
            raise ValueError(f"step {step}, token {token}: expected {expected}, got {score}")


def cpu_filters(names):
    """Exercise the production logit transforms on CPU, including per-row EOS masking."""
    sys.path.insert(0, str(ROOT))
    import torch
    from megatron.text_generation_utils import _mask_eos_before_minimum_tokens, filter_logits

    for name in names:
        case = CASES[name]
        settings = case["settings"].copy()
        minimum = settings.pop("minimum_tokens", 0)
        temperature = settings.pop("temperature", 0.0)
        counts = [0, minimum]
        logits = torch.tensor([RAW, RAW], dtype=torch.float32)
        _mask_eos_before_minimum_tokens(logits, 0, torch.tensor(counts), minimum)
        filtered = filter_logits(logits / temperature, **settings) if temperature > 0 else logits
        for scores, count in zip(filtered.tolist(), counts):
            mask = case["before"] if count < minimum else case["after"]
            check_scores(scores, expected_ids(mask), temperature, count)
        print(f"{name}\tPASS (CPU filters)")


def cpu_decode(names):
    """Run the production decoder with the checkpoint's known readout, on one Gloo rank."""
    sys.path.insert(0, str(ROOT))
    from types import SimpleNamespace
    import torch
    from megatron import mpu
    from megatron.text_generation_utils import generate_samples_from_prompt
    from megatron.tokenizer.tokenizer import CharLevelTokenizer

    class FixedReadout(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scores", torch.tensor(RAW, dtype=torch.float32))

        def forward(self, inputs):
            # A fresh tensor per forward: the decoder masks scores in place.
            return self.scores.expand(*inputs[0].shape, -1).clone()

        def clear_cache(self):
            pass  # The readout is independent of its input and has no KV cache.

    model = torch.nn.Module()
    model.device = torch.device("cpu")
    model.module = FixedReadout()
    args = SimpleNamespace(tokenizer=CharLevelTokenizer(VOCAB), padded_vocab_size=VOCAB,
                           seq_length=32, eod_mask_loss=False, sliding_window_width=None,
                           is_pipe_parallel=False, return_logits=True)
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", store=torch.distributed.HashStore(), rank=0, world_size=1)
    try:
        mpu.initialize_model_parallel(1)
        for name in names:
            case = CASES[name]
            settings = case["settings"]
            should_sample = settings.get("temperature", 0.0) > 0
            for seed in range(20):
                torch.random.default_generator.manual_seed(seed)
                before = torch.get_rng_state()
                row, = generate_samples_from_prompt(args, model, "Q", maximum_tokens=LIMIT, **settings)
                used_rng = not torch.equal(torch.get_rng_state(), before)
                if used_rng != should_sample:
                    raise ValueError(f"{name}, seed {seed}: expected sampling={should_sample}, used RNG={used_rng}")
                verify(row, case)
            print(f"{name}\tPASS (CPU decode, 20 seeds)")
    finally:
        mpu.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def cpu_all():
    """Run both golden matrices plus standalone and NeoX regression tests."""
    # Keep config-loader logs, pytest state, and file-based rendezvous outside the checkout.
    with tempfile.TemporaryDirectory(prefix="neox-cpu-suite-") as scratch:
        for name in ("tests", "configs"):
            (Path(scratch) / name).symlink_to(ROOT / name, target_is_directory=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                   OMP_NUM_THREADS="1")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT), env.get("PYTHONPATH"))))
        commands = [
            [str(HERE / "run_matrix.py"), "--cpu-filters"],
            [str(HERE / "run_matrix.py"), "--cpu-decode"],
            ["-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider",
             str(HERE / "test_filter_edges.py"), str(HERE / "test_decode_edges.py"),
             str(ROOT / "tests/unit/test_text_generation_utils.py"),
             str(ROOT / "tests/unit/test_eval_generation.py"),
             str(ROOT / "tests/neox_args/test_neoxargs_load.py")],
        ]
        failures = 0
        for cmd in commands:
            status = subprocess.run([sys.executable, *cmd], cwd=scratch, env=env)
            failures += status.returncode != 0
        return int(failures != 0)


def verify(row, case):
    """Check live processed scores and tokens; there must be no post-EOS rows."""
    minimum = case["settings"].get("minimum_tokens", 0)
    temperature = case["settings"].get("temperature", 0.0)
    n, finished = row["length"], row["finished"]
    text = row["text"] or ""
    if not (row["context"] == "Q" and len(text) == n and minimum <= n <= LIMIT):
        raise ValueError(f"unexpected prompt or completion: {text!r}, length={n}")
    steps = n + int(finished)
    if not (0 < steps <= LIMIT and (finished or n == LIMIT)):
        raise ValueError(f"unexpected stopping condition: length={n}, finished={finished}")
    if "text" in case and row["text"] != case["text"]:
        raise ValueError(f"expected {case['text']!r}, got {row['text']!r}")
    if len(row["logits"]) != steps:
        raise ValueError(f"expected exactly {steps} score rows, got {len(row['logits'])}")
    for step, scores in enumerate(row["logits"]):
        allowed = expected_ids(case["before"] if step < minimum else case["after"])
        check_scores(scores, allowed, temperature, step)
        token = ord(text[step]) if step < n else 0  # EOS is omitted from completion text.
        if token not in allowed:
            raise ValueError(f"step {step}: sampled ineligible token {token}")


def launch(script, config, log):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    with log.open("w") as stream:
        proc = subprocess.run([sys.executable, "deepy.py", str(script), str(config)],
                              cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"launcher exited {proc.returncode}; see {log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=list(CASES),
                        help="select one case; repeat to select more")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="list cases without writing files")
    mode.add_argument("--prepare-only", action="store_true", help="render configs without CUDA or NeoX")
    mode.add_argument("--cpu-filters", action="store_true", help="check production CPU logit transforms without launching a model (needs NeoX dependencies)")
    mode.add_argument("--cpu-decode", action="store_true", help="run the production decoder on fixed CPU logits across 20 seeds per case")
    mode.add_argument("--cpu-all", action="store_true", help="run all CPU matrices, boundary, two-rank Gloo, and NeoX regression tests without writing to the checkout")
    parser.add_argument("--output-dir", type=Path, help="new directory (default: fresh runs/run-* directory)")
    args = parser.parse_args()
    if args.cpu_all:
        if args.case or args.output_dir:
            parser.error("--cpu-all runs the full suite in a temporary directory; omit --case and --output-dir")
        return cpu_all()
    cases = dict.fromkeys(args.case or CASES)
    if args.list:
        print("\n".join(cases))
        return 0
    if args.cpu_filters:
        cpu_filters(cases)
        return 0
    if args.cpu_decode:
        cpu_decode(cases)
        return 0
    if args.output_dir:
        out = args.output_dir.resolve()
        out.mkdir(parents=True, exist_ok=False)
    else:
        (HERE / "runs").mkdir(exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix="run-", dir=HERE / "runs"))
    print(f"Run directory: {out}", flush=True)
    base = json.loads((HERE / "configs/tiny.json").read_text())
    checkpoint = str(out / "checkpoint")

    def config(name, **values):
        path = out / f"{name}.json"
        path.write_text(json.dumps({**base, **values}, indent=2) + "\n")
        return path

    bootstrap = config("bootstrap", save=checkpoint, checkpoint_factor=1)
    for name in cases:
        cases[name] = config(name, load=checkpoint, iteration=1, text_gen_type="input-file",
                             prompt_end="\n", sample_input_file=str(PROMPT),
                             sample_output_file=str(out / f"{name}.jsonl"),
                             maximum_tokens=LIMIT, return_logits=True, **CASES[name]["settings"])
    if args.prepare_only:
        print(f"Prepared bootstrap and {len(cases)} generation configs (not run).")
        return 0
    print("Creating toy checkpoint...", flush=True)
    launch(HERE / "make_checkpoint.py", bootstrap, out / "bootstrap.log")
    passed = 0
    with (out / "summary.tsv").open("w", buffering=1) as summary:
        print("case\tstatus\tdetail", file=summary)
        for name, path in cases.items():
            print(f"Running {name}...", flush=True)
            try:
                launch("generate.py", path, out / f"{name}.log")
                # There is exactly one one-token prompt per invocation.
                row = json.loads((out / f"{name}.jsonl").read_text())
                verify(row, CASES[name])
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                result = f"{name}\tFAIL\t{exc}"
            else:
                passed += 1
                result = f"{name}\tPASS\tlength={row['length']}, finished={row['finished']}"
            print(result, file=summary)
            print(result, flush=True)
    return int(passed != len(cases))


if __name__ == "__main__":
    sys.exit(main())
