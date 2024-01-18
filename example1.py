import os
os.environ['TIMM_FUSED_ATTN'] = "0"

import torch
from torch import Tensor
import timm
import warnings
warnings.filterwarnings('ignore')

from torch_flops import TorchFLOPsByFX

'''
Count the FLOPs of ViT-B16 and ResNet-50.
'''

if __name__ == "__main__":
    # Define the models
    vit = timm.create_model('vit_base_patch16_224')
    resnet = timm.create_model('resnet50')

    # Input
    x = torch.randn([1, 3, 224, 224])

    # Output
    # Build the graph of the model. You can specify the operations (listed in `MODULE_FLOPs_MAPPING`, `FUNCTION_FLOPs_MAPPING` and `METHOD_FLOPs_MAPPING` in 'flops_ops.py') to ignore.
    print("=" * 10, "vit_base16", "=" * 10)
    flops_counter = TorchFLOPsByFX(vit)
    # # Print the grath (not essential)
    # print('*' * 120)
    # flops_counter.graph_model.graph.print_tabular()
    # Feed the input tensor
    flops_counter.propagate(x)
    # # Print the flops of each node in the graph. Note that if there are unsupported operations, the "flops" of these ops will be marked as 'not recognized'.
    # print('*' * 120)
    flops_counter.print_result_table()
    # # Print the total FLOPs
    total_flops = flops_counter.print_total_flops(show=True)

    print("=" * 10, "resnet50", "=" * 10)
    flops_counter = TorchFLOPsByFX(resnet)
    flops_counter.propagate(x)
    flops_counter.print_result_table()
    total_flops = flops_counter.print_total_flops(show=True)
