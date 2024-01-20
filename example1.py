import os
os.environ['TIMM_FUSED_ATTN'] = "0"

import torch
from torch import Tensor
import timm
import warnings
warnings.filterwarnings('ignore')
from typing import Literal
import argparse

from torch_flops import TorchFLOPsByFX

'''
Count the FLOPs of ViT-B16 and ResNet-50.
'''

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=('vitb16', 'resnet50'), default='vitb16')
args = parser.parse_args()

if __name__ == "__main__":
    device = 'cuda:0'
    # Input
    x = torch.randn([1, 3, 224, 224]).to(device)
    model_arch: Literal['vitb16', 'resnet50'] = args.model

    if model_arch == 'vitb16':
        print("=" * 10, "vit_base16", "=" * 10)
        # Define the models
        vit = timm.create_model('vit_base_patch16_224').to(device)

        # NOTE: First run the model once for accurate time measurement in the following process.
        with torch.no_grad():
            vit(x)

        with torch.no_grad():
            # Build the graph of the model. You can specify the operations (listed in `MODULE_FLOPs_MAPPING`, `FUNCTION_FLOPs_MAPPING` and `METHOD_FLOPs_MAPPING` in 'flops_ops.py') to ignore.
            flops_counter = TorchFLOPsByFX(vit)
            # # Print the grath (not essential)
            # print('*' * 120)
            # flops_counter.graph_model.graph.print_tabular()
            # Feed the input tensor
            flops_counter.propagate(x)
        # # Print the flops of each node in the graph. Note that if there are unsupported operations, the "flops" of these ops will be marked as 'not recognized'.
        print('*' * 120)
        result_table = flops_counter.print_result_table()
        # # Print the total FLOPs
        total_flops = flops_counter.print_total_flops(show=True)
        total_time = flops_counter.print_total_time()
        max_memory = flops_counter.print_max_memory()
    elif model_arch == 'resnet50':
        print("=" * 10, "resnet50", "=" * 10)
        resnet = timm.create_model('resnet50').to(device)

        # NOTE: First run the model once for accurate time measurement in the following process.
        with torch.no_grad():
            resnet(x)

        with torch.no_grad():
            flops_counter = TorchFLOPsByFX(resnet)
            flops_counter.propagate(x)
        result_table = flops_counter.print_result_table()
        total_flops = flops_counter.print_total_flops(show=True)
        total_time = flops_counter.print_total_time()
        max_memory = flops_counter.print_max_memory()
