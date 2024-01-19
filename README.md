# torch_flops

## Introduction
[torch_flops中文介绍 - 知乎](https://zhuanlan.zhihu.com/p/663566912)

This is a library for calculating FLOPs of pytorch models. Compared with other  libraries such as [`thop`](https://github.com/Lyken17/pytorch-OpCounter), [`ptflops`](https://github.com/sovrasov/flops-counter.pytorch), [`torchinfo`](https://github.com/TylerYep/torchinfo) and [`torchanalyse`](https://github.com/HaoKang-Timmy/torchanalyse), the **advantage of this library** is that it can capture **all calculation operations** in the `forward` process, **not limited to only the subclasses of** `nn.Module`.

**Update Note**: Introducing support for displaying the **execution time** of each operation. Please use `flops_counter.print_result_table()` to see the detailed results.

**Update Note**: Introducing support for displaying the **GPU memory usage** of each operation. In the result table, `mem_before_op`, `mem_after_op` represent the memories (counted using `torch.cuda.memory_allocated()`) before and after the operation. `mem_delta` represent the difference between `mem_after_op` and `mem_before_op`. Please note that just run one model each time in a program in order to obtain accurate memory statistics.


## Usage
### Installation
```
pip install torch_flops -i https://pypi.org/simple
```

### Requirements

+ python >= 3.10 (for new python features)
+ pytorch >= 1.8 (for `torch.fx` support)
+ tabulate (for printing the summary of operations)

### Example 1
An expamle for calculating the FLOPs of ViT-base16 and ResNet-50 is given in [`example1.py`](example1.py). The example requires the [`timm`](https://github.com/huggingface/pytorch-image-models) library. You can calculate the FLOPs in three lines:
```python
    flops_counter = TorchFLOPsByFX(resnet)
    flops_counter.propagate(x)
    total_flops = flops_counter.print_total_flops(show=True)
```
The output of `example1.py` is:
```
========== vit_base16 ==========
total_flops = 35,164,979,282 
========== resnet50 ==========
total_flops = 8,227,340,288
```

### Example 2
Another example of calculating the FLOPs for an attention block is provided in [`example2.py`](example2.py). However, You can define a simple model to check the result (see [`compare.py`](compare.py)).

```python
C = 768

# Define the model: an attention block (refer to "timm": https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py)
block = Block(C, num_heads=2, qkv_bias=True)
block.attn.fused_attn = False
block.eval()
model = block

# Input
# N: number of tokens
N = 14**2 + 1
B = 1
x = torch.randn([B, N, C])

# Output
# Build the graph of the model. You can specify the operations (listed in `MODULE_FLOPs_MAPPING`, `FUNCTION_FLOPs_MAPPING` and `METHOD_FLOPs_MAPPING` in 'flops_ops.py') to ignore.
flops_counter = TorchFLOPsByFX(model)
# Print the grath (not essential)
print('*' * 120)
flops_counter.graph_model.graph.print_tabular()
# Feed the input tensor
flops_counter.propagate(x)
# Print the FLOPs of each node in the graph. Note that if there are unsupported operations, the "flops" of these ops will be marked as 'not recognized'.
print('*' * 120)
flops_counter.print_result_table()
# Print the total FLOPs
total_flops = flops_counter.print_total_flops()
```
You can also feed more than one sequential arguments for the model in `propagate()` if the `model.forward()` function need not only one arguments.

# Advantage
`torch_flops` can capture all the operations excuted in the forward including the operations not wrapped by `nn.Module`, like `torch.matmul`, `@`, `+` and `tensor.exp`, and it can ignore the FLOPs of the modules not used in the forward process.

There is a comparison of `torch_flops` (this repo), `torchanalyse`, `thop` and `ptflops` in the script [`compare.py`](compare.py).
The output of

`python compare.py`:

```
**************************************** Model ****************************************
SimpleModel(
  (layer): Linear(in_features=5, out_features=4, bias=True)
)
tensor([[-0.2077,  0.2623,  1.3978, -0.4170]], grad_fn=<AddmmBackward0>)
================================================================================
**************************************** torch_flops ****************************************
===========  ===========  ===========  =====================  =======
node_name    node_op      op_target    nn_module_stack[-1]      flops
===========  ===========  ===========  =====================  =======
x            placeholder  x                                         0
layer        call_module  layer        Linear                      40
output       output       output                                    0
===========  ===========  ===========  =====================  =======
torch_flops: 40 FLOPs
================================================================================
**************************************** torchanalyse ****************************************
torchanalyse: 40 FLOPs
================================================================================
**************************************** thop ****************************************
[INFO] Register count_linear() for <class 'torch.nn.modules.linear.Linear'>.
thop: 20 MACs
================================================================================
**************************************** ptflops ****************************************
Warning: module SimpleModel is treated as a zero-op.
SimpleModel(
  24, 100.000% Params, 24.0 Mac, 100.000% MACs, 
  (layer): Linear(24, 100.000% Params, 24.0 Mac, 100.000% MACs, in_features=5, out_features=4, bias=True)
)
ptflops: 24 MACs
================================================================================
```

Now let's add an operation `x += 1.` in `forward()`.
The output of

`python compare.py --add_one`:

```
**************************************** Model ****************************************
SimpleModel(
  (layer): Linear(in_features=5, out_features=4, bias=True)
)
tensor([[1.0426, 0.6963, 1.7114, 1.6526]], grad_fn=<AddBackward0>)
================================================================================
**************************************** torch_flops ****************************************
===========  =============  =======================  =====================  =======
node_name    node_op        op_target                nn_module_stack[-1]      flops
===========  =============  =======================  =====================  =======
x            placeholder    x                                                     0
layer        call_module    layer                    Linear                      40
add          call_function  <built-in function add>                               4
output       output         output                                                0
===========  =============  =======================  =====================  =======
torch_flops: 44 FLOPs
================================================================================
**************************************** torchanalyse ****************************************
torchanalyse: 40 FLOPs
================================================================================
**************************************** thop ****************************************
[INFO] Register count_linear() for <class 'torch.nn.modules.linear.Linear'>.
thop: 20 MACs
================================================================================
**************************************** ptflops ****************************************
Warning: module SimpleModel is treated as a zero-op.
SimpleModel(
  24, 100.000% Params, 24.0 Mac, 100.000% MACs, 
  (layer): Linear(24, 100.000% Params, 24.0 Mac, 100.000% MACs, in_features=5, out_features=4, bias=True)
)
ptflops: 24 MACs
================================================================================
```

**It can be seen that only `torch_flops` can capture the FLOPs of `x+=1`!**

`torchinfo` is not compared here but it does not have this ability. We also find that some of the other libraries cannot calculate the FLOPs of the `bias` item in `nn.Linear` using `python compare.py --linear_no_bias`.


# Supported Operations
The supported operations are listed in the following (the keys of the dicts), which can also be seen in [`flops_ops.py`](torch_flops/flops_ops.py).
Note that in addtion to the modules inherited from `nn.Module` (e.g. `nn.Linear`), the function (e.g. `@`, `+`, `torch.softmax`) and method operations (e.g. `tensor.softmax`) are also supported!

```python
MODULE_FLOPs_MAPPING = {
    'Linear': ModuleFLOPs_Linear,
    'Identity': ModuleFLOPs_zero,
    'Conv1d': ModuleFLOPs_ConvNd,
    'Conv2d': ModuleFLOPs_ConvNd,
    'Conv3d': ModuleFLOPs_ConvNd,
    'AvgPool1d': ModuleFLOPs_AvgPoolNd,
    'AvgPool2d': ModuleFLOPs_AvgPoolNd,
    'AvgPool3d': ModuleFLOPs_AvgPoolNd,
    'AdaptiveAvgPool1d': ModuleFLOPs_AdaptiveAvgPoolNd,
    'AdaptiveAvgPool2d': ModuleFLOPs_AdaptiveAvgPoolNd,
    'AdaptiveAvgPool3d': ModuleFLOPs_AdaptiveAvgPoolNd,
    'MaxPool1d': ModuleFLOPs_MaxPoolNd,
    'MaxPool2d': ModuleFLOPs_MaxPoolNd,
    'MaxPool3d': ModuleFLOPs_MaxPoolNd,
    'AdaptiveMaxPool1d': ModuleFLOPs_AdaptiveMaxPoolNd,
    'AdaptiveMaxPool2d': ModuleFLOPs_AdaptiveMaxPoolNd,
    'AdaptiveMaxPool3d': ModuleFLOPs_AdaptiveMaxPoolNd,
    'LayerNorm': ModuleFLOPs_Norm,
    'BatchNorm1d': ModuleFLOPs_Norm,
    'BatchNorm2d': ModuleFLOPs_Norm,
    'BatchNorm3d': ModuleFLOPs_Norm,
    'InstanceNorm1d': ModuleFLOPs_Norm,
    'InstanceNorm2d': ModuleFLOPs_Norm,
    'InstanceNorm3d': ModuleFLOPs_Norm,
    'GroupNorm': ModuleFLOPs_Norm,
    'Dropout': ModuleFLOPs_zero,
    'GELU': ModuleFLOPs_GELU,
    'ReLU': ModuleFLOPs_elemwise,
    'Flatten': ModuleFLOPs_zero,
}
FUNCTION_FLOPs_MAPPING = {
    'getattr': FunctionFLOPs_zero,
    'getitem': FunctionFLOPs_zero,
    'mul': FunctionFLOPs_elemwise,
    'truediv': FunctionFLOPs_elemwise,
    'sub': FunctionFLOPs_elemwise,
    'matmul': FunctionFLOPs_matmul,
    'add': FunctionFLOPs_elemwise,
    'concat': FunctionFLOPs_zero,
    '_assert': FunctionFLOPs_zero,
    'eq': FunctionFLOPs_elemwise,
    'cat': FunctionFLOPs_zero,
    'linear': FunctionFLOPs_linear,
}
METHOD_FLOPs_MAPPING = {
    'reshape': MethodFLOPs_zero,
    'permute': MethodFLOPs_zero,
    'unbind': MethodFLOPs_zero,
    'transpose': MethodFLOPs_zero,
    'repeat': MethodFLOPs_zero,
    'unsqueeze': MethodFLOPs_zero,
    'exp': MethodFLOPs_elemwise,
    'sum': MethodFLOPs_sum,
    'div': MethodFLOPs_elemwise,
    'softmax': MethodFLOPs_softmax,
    'expand': MethodFLOPs_zero,
    'flatten': MethodFLOPs_zero,
}
```
However, not all the operations in pytorch have been considered since it spends a lot of effort. If you need to add support for a certain operation, please raise an issue. You are also welcome to add more features to this repository.

# Limitations
`torch.fx` can capture all the operations in the forward process, but it requires a high version of pytorch. However, we recommod you to use the newer version of pytorch (>=2.0) to try the new features.

When using `torch.fx`, the model should be able to successfully transformed into a [`graph_model`](torch_flops/flops_engine.py#L317) by [`symbolic_trace()`](https://pytorch.org/docs/stable/fx.html#torch.fx.symbolic_trace). Dynamic control flow is not supported in the `forward` function. Please refer to https://pytorch.org/docs/stable/fx.html#limitations-of-symbolic-tracing for more information.

There are many operations not implemented so far. However, you can raise an issue or contact me (zgxd@mail.nwpu.edu.cn) to add new operations.

# Acknowledgements

`pytorch`: https://github.com/pytorch/pytorch

`timm`: https://github.com/huggingface/pytorch-image-models

`torchscan`: https://frgfm.github.io/torch-scan/index.html

`torchprofile`: https://github.com/zhijian-liu/torchprofile
