"""Create a native NeoX checkpoint with known logits, without training or downloads."""

import math

import torch

from megatron.checkpointing import save_checkpoint
from megatron.initialize import initialize_megatron
from megatron.neox_arguments import NeoXArgs
from megatron.training import setup_model_and_optimizer


def main():
    args = NeoXArgs.consume_neox_args()
    args.configure_distributed_args()
    args.build_tokenizer()
    initialize_megatron(args)
    model, optimizer, scheduler, _ = setup_model_and_optimizer(args)
    # The two transformer blocks still run; only the untied readout is controlled.
    norm = model.module.sequential[-2].norm
    weight = model.module.sequential[-1].final_linear.weight
    with torch.no_grad():
        norm.weight.zero_()
        norm.bias.zero_()
        norm.bias[0] = 1.0
        weight.zero_()
        weight[:, 0] = -50.0
        for token, p in zip([args.tokenizer.eod, *map(ord, "ABCD")], [0.55, 0.25, 0.12, 0.06, 0.02]):
            weight[token, 0] = math.log(p)
    save_checkpoint(args, 1, model, optimizer, scheduler)


if __name__ == "__main__":
    main()
