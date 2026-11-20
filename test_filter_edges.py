"""Independent CPU oracles for sampler boundaries and all seven-filter subsets."""

from itertools import combinations
import json
import math
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from megatron.text_generation_utils import filter_logits, stream_tokens


FILTERS = dict(
    top_h=0.86,
    top_k=5,
    top_p=0.84,
    min_p=0.17,
    typical_p=0.78,
    epsilon_cutoff=0.13,
    eta_cutoff=0.19,
)
COUNTS = {"minimum_tokens", "maximum_tokens", "top_k"}
UPPER_ONE = {"top_p", "top_h", "min_p", "typical_p", "epsilon_cutoff", "eta_cutoff"}


def invalid_values(name):
    values = [-1, True, False, "0.5", math.nan, math.inf, -math.inf]
    if name != "maximum_tokens":
        values.append(None)
    if name in COUNTS:
        values += [0.5, 1.0]
    if name in UPPER_ONE:
        values.append(math.nextafter(1.0, math.inf))
    if name in {"eta_cutoff", "epsilon_cutoff"}:
        values.append(1.0)
    if name == "typical_p":
        values.append(0.0)
    return values


@pytest.mark.parametrize(
    "name,value",
    [
        (name, value)
        for name in sorted(FILTERS.keys() | COUNTS | {"temperature"})
        for value in invalid_values(name)
    ],
)
def test_invalid_request_fails_before_model_or_collectives(name, value):
    model = Mock()
    with pytest.raises(ValueError, match=name):
        stream_tokens(object(), model, [[1]], **{name: value})
    model.eval.assert_not_called()


@pytest.mark.parametrize(
    "name,value", [("top_k", True), ("temperature", "0.5"), ("minimum_tokens", 1.5)]
)
def test_mistyped_sampling_config_is_a_validation_error(name, value):
    from megatron.neox_arguments import NeoXArgs

    config = json.loads((Path(__file__).parent / "configs/tiny.json").read_text())
    config[name] = value
    with pytest.raises(ValueError):
        NeoXArgs.from_dict(config)


def expected_ids(raw, opts):
    """Scalar reference: reason about candidate IDs, never mutate a logit tensor."""
    ids = sorted(range(len(raw)), key=raw.__getitem__, reverse=True)

    def probabilities(pool):
        peak = max(raw[i] for i in pool)
        weights = {i: math.exp(raw[i] - peak) for i in pool}
        total = math.fsum(weights.values())
        return {i: w / total for i, w in weights.items()}

    def prefix(pool, mass, probs):
        result = []
        for i in pool:
            result.append(i)
            if math.fsum(probs[j] for j in result) >= mass:
                break
        return result

    def entropy(p):
        return -math.fsum(x * math.log(x) for x in p.values())

    if "top_h" in opts:
        pool = ids[:100]
        p = probabilities(pool)
        budget = opts["top_h"] * entropy(p)
        terms = [-p[i] * math.log(p[i]) for i in pool]
        ids = [
            i for n, i in enumerate(pool, 1) if n == 1 or math.fsum(terms[:n]) <= budget
        ]
    if "top_k" in opts:
        boundary = raw[ids[min(opts["top_k"], len(ids)) - 1]]
        ids = [i for i in ids if raw[i] >= boundary]
    if "top_p" in opts:
        ids = prefix(ids, opts["top_p"], probabilities(ids))
    if "min_p" in opts:
        p = probabilities(ids)
        ids = [i for i in ids if p[i] >= max(p.values()) * opts["min_p"]]
    if "typical_p" in opts:
        p = probabilities(ids)
        h = entropy(p)
        distance = {i: abs(-math.log(p[i]) - h) for i in ids}
        ordered = sorted(ids, key=distance.__getitem__)
        last = prefix(ordered, opts["typical_p"], p)[-1]
        ids = [i for i in ids if distance[i] <= distance[last]]
    for name in ("epsilon_cutoff", "eta_cutoff"):
        if name in opts:
            p = probabilities(ids)
            cutoff = opts[name]
            if name == "eta_cutoff":
                cutoff = min(cutoff, math.sqrt(cutoff) * math.exp(-entropy(p)))
            ids = [i for i in ids if p[i] >= cutoff or p[i] == max(p.values())]
    return set(ids)


SUBSETS = [c for n in range(len(FILTERS) + 1) for c in combinations(FILTERS, n)]


@pytest.mark.parametrize("names", SUBSETS, ids=["+".join(s) or "none" for s in SUBSETS])
def test_every_filter_subset_matches_scalar_reference(names):
    opts = {k: FILTERS[k] for k in names}
    # Three distinct distributions in a non-contiguous [batch, vocab] tensor.
    logits = (
        torch.tensor(
            [
                [4.2, 3.9, 3.2, 2.1, 1.7, 0.5, -0.6],
                [-2.0, -0.1, 0.0, 0.8, 1.3, 1.5, 2.1],
                [0.25, 0.5, 0.85, 0.12, 0.02, 0.31, 0.43],
            ],
            dtype=torch.float64,
        )
        .t()
        .contiguous()
        .t()
    )
    expected = logits.clone()
    for row, raw in enumerate(logits.tolist()):
        keep = expected_ids(raw, opts)
        for i in range(len(raw)):
            if i not in keep:
                expected[row, i] = -math.inf
    torch.testing.assert_close(filter_logits(logits, **opts), expected)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("name", list(FILTERS))
def test_only_one_finite_candidate_survives(dtype, name):
    scores = torch.tensor(
        [[-math.inf, 9.0, -math.inf], [-1.0, -math.inf, -math.inf]], dtype=dtype
    )
    torch.testing.assert_close(
        filter_logits(scores.clone(), **{name: 1 if name == "top_k" else 0.5}), scores
    )
    singleton = torch.tensor([[6.0]], dtype=dtype)
    torch.testing.assert_close(
        filter_logits(singleton.clone(), **{name: 1 if name == "top_k" else 0.5}),
        singleton,
    )


@pytest.mark.parametrize(
    "opts,raw,expected",
    [
        ({"top_k": 2}, [3.0, 2.0, 2.0, 1.0], [3.0, 2.0, 2.0, -math.inf]),
        ({"min_p": 1.0}, [2.0, 2.0, 0.0], [2.0, 2.0, -math.inf]),
        ({"epsilon_cutoff": 0.5}, [1.0, 1.0], [1.0, 1.0]),
        ({"epsilon_cutoff": 0.99}, [1.0, 1.0, 0.0], [1.0, 1.0, -math.inf]),
        ({"typical_p": 0.01}, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
        (
            {"eta_cutoff": 0.25},
            [math.log(0.5), math.log(0.25), math.log(0.25)],
            [math.log(0.5), math.log(0.25), math.log(0.25)],
        ),
    ],
)
def test_ties_and_inclusive_cutoffs(opts, raw, expected):
    actual = filter_logits(torch.tensor([raw], dtype=torch.float64), **opts)
    torch.testing.assert_close(actual, actual.new_tensor([expected]))


def test_top_p_at_exact_threshold_keeps_no_extra_token():
    # Equal logits give four exactly representable probabilities of 0.25.
    scores = torch.zeros(1, 4)
    assert filter_logits(scores, top_p=0.5).isfinite().sum().item() == 2


@pytest.mark.parametrize("width", [99, 100, 101, 512])
def test_top_h_candidate_cap(width):
    scores = torch.linspace(1, 0, width, dtype=torch.float64)[None, :]
    out = filter_logits(scores.clone(), top_h=1.0)
    # Allow rounding at the entropy boundary; never admit anything outside the top 100.
    assert out[0, 0] == scores[0, 0]
    assert out[0, : min(99, width - 1)].isfinite().all()
    assert out[0, 100:].isneginf().all()
