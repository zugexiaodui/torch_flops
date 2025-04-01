import torch
from torch import nn, Tensor

from torch_flops import TorchFLOPsByFX
from torch.utils.flop_counter import FlopCounterMode
import argparse

'''
Comparion with torch.utils.flop_counter
'''

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
    model = SimpleModel(inp_args).cuda()
    model.requires_grad_(False)
    model.eval()
    x = torch.randn(1, 5).cuda()
    y = model(x)
    print("*" * 40 + " Model " + "*" * 40)
    print(model)
    print(y)
    print("=" * 80)

    # =========
    print("*" * 40 + " torch_flops " + "*" * 40)
    flops_counter = TorchFLOPsByFX(model)
    # flops_counter.graph_model.graph.print_tabular()
    flops_counter.propagate(x)
    flops_counter.print_result_table()
    flops_1 = flops_counter.print_total_flops(show=False)
    print(f"torch_flops: {flops_1} FLOPs")
    print("=" * 80)

    # =========
    print("*" * 40 + " torch.utils.flop_counter " + "*" * 40)
    flops_counter = FlopCounterMode(model, depth=None, display=False)
    with flops_counter:
        model(x)
    flops_2 = flops_counter.get_total_flops()
    print(f"torch.utils.flop_counter: {flops_2} FLOPs")
    print("=" * 80)
