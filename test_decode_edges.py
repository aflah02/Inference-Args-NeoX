"""CPU end-to-end decoding, file I/O, and two-process tensor-parallel checks."""

from datetime import timedelta
from itertools import product
import json
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron import mpu
from megatron import text_generation_utils as gen


class FixedReadout(torch.nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.register_buffer("scores", torch.tensor(scores, dtype=torch.float32))
        self.calls = 0

    def clear_cache(self):
        pass

    def inference_mode(self, use_cache):
        self.use_cache = use_cache
        self.eval()

    def train_mode(self):
        self.train()

    def forward(self, inputs):
        self.calls += 1
        return self.scores.expand(*inputs[0].shape, -1).clone()


def runtime(scores=(4, 3, 2, 1), *, width=None):
    engine = torch.nn.Module()
    engine.device = torch.device("cpu")
    engine.module = FixedReadout(scores)
    tokenizer = SimpleNamespace(
        eod=0,
        tokenize=lambda s: [1] * len(s),
        detokenize=lambda ids: "".join("!ABC"[i] for i in ids),
    )
    args = SimpleNamespace(
        seq_length=6,
        padded_vocab_size=width or len(scores),
        is_pipe_parallel=False,
        return_logits=True,
        eod_mask_loss=False,
        sliding_window_width=None,
        tokenizer=tokenizer,
    )
    return args, engine


@pytest.fixture(scope="module")
def single_rank():
    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        mpu.initialize_model_parallel(1)
        yield
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.parametrize("scores_on,recompute", list(product([False, True], repeat=2)))
@pytest.mark.parametrize(
    "prompt,budget,minimum,text,finished,steps",
    [
        ("", 0, 0, None, False, 0),
        ("", 1, 0, None, True, 1),
        ("p", 1, 1, "A", False, 1),
        ("p", 3, 2, "AA", True, 3),
        ("p" * 6, None, 0, None, False, 0),
        ("p" * 5, None, 1, "A", False, 1),
        ("p" * 5, 99, 0, None, True, 1),
    ],
)
def test_token_budget_and_output(
    single_rank, prompt, budget, minimum, text, finished, steps, scores_on, recompute
):
    args, model = runtime()
    args.return_logits = scores_on
    (row,) = gen.generate_samples_from_prompt(
        args,
        model,
        prompt,
        maximum_tokens=budget,
        minimum_tokens=minimum,
        recompute=recompute,
    )
    assert (row["context"], row["text"], row["length"], row["finished"]) == (
        prompt,
        text,
        len(text or ""),
        finished,
    )
    assert model.module.calls == steps
    assert ("logits" in row) == scores_on
    if scores_on:
        assert len(row["logits"]) == steps
        for i, row_scores in enumerate(row["logits"]):
            assert row_scores == (
                [-float("inf"), 3, 2, 1] if i < minimum else [4, 3, 2, 1]
            )


def test_zero_requests_and_overlong_prompt(single_rank):
    args, model = runtime()
    assert gen.generate_samples_from_prompt(args, model, []) == []
    with pytest.raises(ValueError, match="context_length"):
        gen.generate_samples_from_prompt(args, model, "x" * (args.seq_length + 1))
    assert model.module.calls == 0


@pytest.mark.parametrize("temp", [0.0001, 0.5, 1.0, 10000.0])
def test_seeded_draw_and_temperature_scaling(single_rank, temp):
    # A local generator predicts the draw without calling any NeoX filter.
    args, model = runtime(scores=(4.0, -2.0, 3.0, 0.0))
    scaled = torch.tensor([[4.0, -2.0, 3.0, 0.0]]) / temp
    with torch.random.fork_rng(devices=[]):
        for seed in (7, 23, 31):
            expected = torch.multinomial(
                scaled.softmax(-1), 1, generator=torch.Generator().manual_seed(seed)
            )
            torch.random.default_generator.manual_seed(seed)
            (out,) = gen.stream_tokens(
                args, model, [[2]], maximum_tokens=1, temperature=temp
            )
            assert out[0][0, 1].item() == expected.item()
            torch.testing.assert_close(out[3], scaled)


def test_harness_with_real_decoder(single_rank):
    from functools import partial
    from unittest.mock import Mock
    from eval_tasks.eval_adapter import EvalHarnessAdapter

    args, model = runtime()
    args.recompute = False
    args.tokenizer.encode = args.tokenizer.tokenize
    adapter = EvalHarnessAdapter.__new__(EvalHarnessAdapter)
    adapter._model, adapter.neox_args, adapter.tokenizer = model, args, args.tokenizer
    adapter._max_gen_toks, adapter.cache_hook = 4, Mock()
    adapter.generate = partial(gen.generate_samples_from_prompt, args, model)

    def request(**kwargs):
        return SimpleNamespace(args=("p", kwargs))

    assert adapter.generate_until(
        [
            request(max_gen_toks=0),
            request(do_sample=False, minimum_tokens=2, temperature=0.5),
            request(do_sample=True, minimum_tokens=2, top_k=1),
        ]
    ) == ["", "AA", "AA"]
    for bad in (
        {"do_sample": True, "temperature": 0},
        {"do_sample": "false"},
        {"until": 42},
    ):
        with pytest.raises(ValueError):
            adapter.generate_until([request(**bad)])
        assert model.module.training


def test_tiny_positive_temperature_can_still_sample(single_rank):
    # A finite positive config value that underflows if cast to FP32 before division.
    args, model = runtime(scores=(1.0, 4.0, 2.0, 3.0))
    with torch.random.fork_rng(devices=[]):
        (step,) = gen.stream_tokens(
            args, model, [[2]], temperature=1e-300, maximum_tokens=1
        )
    assert step[0][0, 1].item() == 1


@pytest.mark.parametrize("recompute", [False, True])
def test_ragged_prompts_mask_eos_per_row(single_rank, recompute):
    args, model = runtime()
    prompts = [[2], [3, 3, 3]]
    steps = list(
        gen.stream_tokens(
            args,
            model,
            prompts,
            minimum_tokens=1,
            maximum_tokens=4,
            recompute=recompute,
        )
    )
    assert prompts == [[2], [3, 3, 3]]  # Padding is private.
    assert len(steps) == 4
    tokens, starts, ends, first_scores, done = steps[-1]
    assert starts.tolist() == [1, 3] and ends.tolist() == [1, 3] and done.all()
    assert tokens[0, :3].tolist() == [2, 1, 0]
    assert tokens[1, :5].tolist() == [3, 3, 3, 1, 0]
    assert first_scores.shape == (4, 4)
    assert first_scores[:, 0].tolist() == [-float("inf"), 4, 4, 4]


def test_ragged_prompt_is_not_already_a_generated_stop(single_rank):
    args, model = runtime(scores=(0, 1, 3, 2))
    generator = gen.stream_tokens(
        args, model, [[3], [1, 1, 1]], maximum_tokens=4, stop_tokens=[[1, 1]]
    )
    try:
        first = next(generator)
        assert first[-1].tolist() == [
            False,
            False,
        ]  # Row 1 is still consuming its prompt.
    finally:
        generator.close()


@pytest.mark.parametrize(
    "stops,steps", [([1], 1), ([[2], [1, 1]], 2), ([[3]], 3), ([], 3)]
)
def test_stop_sequences_and_minimum(single_rank, stops, steps):
    args, model = runtime()
    (row,) = gen.generate_samples_from_prompt(
        args, model, "", maximum_tokens=3, minimum_tokens=3, stop_tokens=stops
    )
    assert len(row["logits"]) == steps
    assert row["finished"] == (steps < 3)


def test_input_file_delimiters_and_output_overwrite(single_rank, tmp_path):
    args, model = runtime()
    prompts, dest = tmp_path / "prompts", tmp_path / "results"
    dest.write_text("stale bytes\n")
    prompts.write_text("ABC|| || D\nE ||ABC||")
    rows = gen.generate_samples_input_from_file(
        args, model, prompts, output_file=dest, prompt_end="||"
    )
    assert [r["context"] for r in rows] == ["ABC", "D\nE", "ABC"]
    assert [json.loads(line) for line in dest.read_text().splitlines()] == rows
    prompts.write_text("")
    assert (
        gen.generate_samples_input_from_file(args, model, prompts, output_file=dest)
        == []
    )
    assert dest.read_text() == ""


def test_unconditional_and_interactive_eof(single_rank, monkeypatch, capsys):
    args, model = runtime()
    records = gen.generate_samples_unconditional(args, model, number_of_samples=2)
    assert [(row["context"], row["length"]) for row in records] == [("", 0), ("", 0)]
    args, model = runtime(scores=(0, 3, 2, 1))
    lines = iter(["Q", None])

    def read(_):
        line = next(lines)
        if line is None:
            raise EOFError
        return line

    monkeypatch.setattr("builtins.input", read)
    monkeypatch.setattr(gen.os, "system", lambda _: 0)
    gen.generate_samples_interactive(args, model, maximum_tokens=2)
    assert "Generated Text: AA" in capsys.readouterr().out


def _tp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        mpu.initialize_model_parallel(2)
        # The unique global winner (ID 3) is on the non-sampling rank's shard.
        args, model = runtime(([0, 1], [3, 5])[rank], width=4)
        for temp, recompute, scores_on in product(
            (0.0, 1.0), (False, True), (False, True)
        ):
            args.return_logits = scores_on
            torch.random.default_generator.manual_seed(13 + rank)
            before = torch.get_rng_state()
            opts = {"top_k": 1} if temp else {}
            result = list(
                gen.stream_tokens(
                    args,
                    model,
                    [[1]],
                    temperature=temp,
                    maximum_tokens=3,
                    recompute=recompute,
                    **opts
                )
            )
            tokens, _, _, scores, _ = result[-1]
            assert tokens[0, 1:4].tolist() == [3, 3, 3]
            used_rng = not torch.equal(before, torch.get_rng_state())
            assert used_rng == (temp > 0 and rank == 0)
            if scores_on:
                expected = [-float("inf")] * 3 + [5.0] if temp else [0.0, 1.0, 3.0, 5.0]
                torch.testing.assert_close(scores, torch.tensor([expected] * 3))
            else:
                assert scores is None
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


def test_two_rank_gloo_gathers_scores_and_samples_once(tmp_path):
    torch.multiprocessing.spawn(
        _tp_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2
    )
