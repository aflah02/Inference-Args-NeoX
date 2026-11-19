"""Check prompt isolation with a real context-sensitive model, not the fixed-head checkpoint."""

import torch

from megatron.initialize import initialize_megatron
from megatron.neox_arguments import NeoXArgs
from megatron.text_generation_utils import generate_samples_from_prompt, stream_tokens
from megatron.training import setup_model_and_optimizer


def main():
    args = NeoXArgs.consume_neox_args(overwrite_values={"return_logits": True})
    args.configure_distributed_args()
    args.build_tokenizer()
    initialize_megatron(args)
    # Prefill and full recomputation use different GEMM shapes; compare in FP32.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    model, *_ = setup_model_and_optimizer(args, use_cache=True)
    model.module.inference_mode(use_cache=True)
    layers = [m for m in model.module.modules() if hasattr(m, "layer_past")]
    assert layers
    opts = dict(minimum_tokens=3, maximum_tokens=3)

    def generate(text, recompute=False):
        return generate_samples_from_prompt(args, model, text, recompute=recompute, **opts)

    def compare(actual, expected, **tolerances):
        assert actual["text"] == expected["text"]
        torch.testing.assert_close(
            torch.tensor(actual["logits"]), torch.tensor(expected["logits"]), **tolerances
        )

    inputs = ["QZ", "AZ"]
    controls = []
    for prompt in inputs:
        model.module.clear_cache()
        controls.extend(generate(prompt))
    assert controls[0]["logits"][0] != controls[1]["logits"][0], "readout ignores context"
    model.module.clear_cache()
    sequential = generate(inputs)
    assert len(sequential) == len(controls)
    for result, control in zip(sequential, controls):
        compare(result, control, rtol=0, atol=0)
    print("PASS: both prompts' text and all saved scores match isolated runs exactly.", flush=True)
    assert all(m.layer_past is None for m in layers)

    stream = stream_tokens(args, model, [args.tokenizer.tokenize("Longer")], **opts)
    next(stream)
    assert any(m.layer_past is not None for m in layers)
    stream.close()
    assert all(m.layer_past is None for m in layers)
    compare(generate("AZ")[0], controls[1], rtol=0, atol=0)
    print("PASS: early close released cache; the next prompt matches its isolated run.", flush=True)

    model.module.inference_mode(use_cache=False)
    for result, control in zip(generate(inputs, recompute=True), controls):
        compare(result, control, rtol=1e-4, atol=1e-5)
    print("PASS: cached decoding matches full recomputation for both prompts and all three steps.", flush=True)


if __name__ == "__main__":
    main()
