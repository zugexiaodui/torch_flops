import torch
from torch import nn, Tensor

from torch_flops import TorchFLOPsByFX
import torchanalyse
from thop import profile
from ptflops import get_model_complexity_info

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--linear_no_bias", action='store_true')
parser.add_argument("--add_one", action='store_true')
inp_args = parser.parse_args()


class SimpleModel(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.layer = nn.Linear(5, 4, bias=not args.linear_no_bias)
        self.__add_one = args.add_one

    def forward(self, x: Tensor):
        x = self.layer(x)
        if self.__add_one:
            x += 1.
        return x


if __name__ == "__main__":
    model = SimpleModel(inp_args)
    x = torch.randn(1, 5)
    y = model(x)
    print("*" * 40 + " Model " + "*" * 40)
    print(model)
    print(y)
    print("=" * 80)

    # =========
    print("*" * 40 + " flops_counter " + "*" * 40)
    flops_counter = TorchFLOPsByFX(model)
    # flops_counter.graph_model.graph.print_tabular()
    flops_counter.propagate(x)
    flops_counter.print_result_table()
    flops_1 = flops_counter.print_total_flops(show=False)
    print(f"torch_flops: {flops_1} FLOPs")
    print("=" * 80)

    # =========
    print("*" * 40 + " torchanalyse " + "*" * 40)
    unit = torchanalyse.Unit(unit_flop='mFLOP')
    system = torchanalyse.System(
        unit,
        frequency=940,
        flops=123,
        onchip_mem_bw=900,
        pe_min_density_support=0.0001,
        accelerator_type="structured",
        model_on_chip_mem_implications=False,
        on_chip_mem_size=32,
    )
    result_2 = torchanalyse.profiler(model, x, system, unit)
    flops_2 = sum(result_2['Flops (mFLOP)'].values) / 1e3
    print(f"torchanalyse: {flops_2:.0f} FLOPs")
    print("=" * 80)

    # =========
    print("*" * 40 + " thop " + "*" * 40)
    macs_1, params = profile(model, inputs=(x, ))
    print(f"thop: {macs_1:.0f} MACs")
    print("=" * 80)

    # =========
    print("*" * 40 + " ptflops " + "*" * 40)
    macs_2, params = get_model_complexity_info(model, tuple(x.shape), as_strings=False,
                                               print_per_layer_stat=True, verbose=True)
    print(f"ptflops: {macs_2:.0f} MACs")
    print("=" * 80)
