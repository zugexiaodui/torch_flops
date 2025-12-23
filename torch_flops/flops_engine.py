import torch
from torch import nn
import torch.fx
from torch.fx import symbolic_trace
from torch.fx.node import Argument, Node, Target, map_aggregate
from torch.fx._compatibility import compatibility
from torch.fx.graph_module import GraphModule
import traceback
from tabulate import tabulate
from typing import Any, Tuple, NamedTuple, Optional, Dict, Sequence, Literal, List, Union
from copy import deepcopy
import time
import csv

from torch_flops.flops_ops import MODULE_FLOPs_MAPPING, METHOD_FLOPs_MAPPING, FUNCTION_FLOPs_MAPPING

'''
REF: https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/shape_prop.py
'''

__all__ = ['TorchFLOPsByFX']


@compatibility(is_backward_compatible=True)
class TensorMetadata(NamedTuple):
    # TensorMetadata is a structure containing pertinent information
    # about a tensor within a PyTorch program.

    # General Tensor metadata
    shape: torch.Size
    dtype: torch.dtype
    requires_grad: bool
    stride: Tuple[int, ...]
    memory_format: Optional[torch.memory_format]
    is_quantized: bool
    qparams: Dict[str, Any]


def _extract_tensor_metadata(result: torch.Tensor) -> TensorMetadata:
    """
    Extract a TensorMetadata NamedTuple describing `result`.
    """
    shape = result.shape
    dtype = result.dtype
    requires_grad = result.requires_grad
    stride = result.stride()

    memory_formats = {
        torch.contiguous_format,
        torch.channels_last,
        torch.channels_last_3d,
    }

    memory_format = None

    for query_format in memory_formats:
        if result.is_contiguous(memory_format=query_format):
            memory_format = query_format
            break

    is_quantized = result.is_quantized
    qparams: Dict[str, Any] = {}
    if is_quantized:
        qscheme = result.qscheme()
        qparams["qscheme"] = qscheme
        if qscheme in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            qparams["scale"] = result.q_scale()  # type: ignore[assignment]
            qparams["zero_point"] = result.q_zero_point()  # type: ignore[assignment]
        elif qscheme in {torch.per_channel_affine, torch.per_channel_affine_float_qparams, torch.per_channel_symmetric}:
            # In this branch, scale and zero_point are expected to be tensors,
            # we store the values as immutable_list in TensorMetadata for
            # easier serialization downstream
            qparams["scale"] = result.q_per_channel_scales().tolist()  # type: ignore[assignment]
            qparams["zero_point"] = result.q_per_channel_zero_points().tolist()  # type: ignore[assignment]
            qparams["axis"] = result.q_per_channel_axis()  # type: ignore[assignment]

    return TensorMetadata(
        shape, dtype, requires_grad, stride, memory_format, is_quantized, qparams)


@compatibility(is_backward_compatible=True)
class ShapeProp(torch.fx.Interpreter):
    """
    Execute an FX graph Node-by-Node and
    record the shape and type of the result
    into the corresponding node.

    Example:
         In this example, we record the shape
         and data type of a module given
         an example input ``torch.randn(50, D_in)``.
         We print the name, shape and dtype of each node.

        class TwoLayerNet(torch.nn.Module):
            def __init__(self, D_in, H, D_out):
                super().__init__()
                self.linear1 = torch.nn.Linear(D_in, H)
                self.linear2 = torch.nn.Linear(H, D_out)
            def forward(self, x):
                h_relu = self.linear1(x).clamp(min=0)
                y_pred = self.linear2(h_relu)
                return y_pred
        N, D_in, H, D_out = 64, 1000, 100, 10
        x = torch.randn(N, D_in)
        y = torch.randn(N, D_out)
        model = TwoLayerNet(D_in, H, D_out)
        gm = torch.fx.symbolic_trace(model)
        sample_input = torch.randn(50, D_in)
        ShapeProp(gm).propagate(sample_input)

        for node in gm.graph.nodes:
            print(node.name, node.meta['tensor_meta'].dtype,
                node.meta['tensor_meta'].shape)

        The output of this code is:

        x torch.float32 torch.Size([50, 1000])
        linear1 torch.float32 torch.Size([50, 100])
        clamp_1 torch.float32 torch.Size([50, 100])
        linear2 torch.float32 torch.Size([50, 10])
        output torch.float32 torch.Size([50, 10])

    Args:
         module (GraphModule): The module to be executed
         fake_mode (FakeTensorMode): A fake mode for copying the gm

    """

    def __init__(self, gm: GraphModule, **kwargs):
        super().__init__(gm)
        mem_func_name: str = kwargs.get('mem_func_name', 'max_memory_allocated')
        assert mem_func_name in ['memory_allocated', 'max_memory_allocated']
        ignore_ops = kwargs.get('ignore_ops', [])

        fake_mode = None
        if fake_mode is not None:
            from torch._dynamo.utils import deepcopy_to_fake_tensor
            # Note:
            # We need fake execution cause the inputs are fake, however, we cannot fakify the module
            # - because we need to write to the tensor_meta of the real module. So we fakify to
            # produce a result (L131 below), to extract tensor meta, and then keep going.
            #
            # If we were to fakify, we would write to the wrong node, and then downstream fusion
            # would be missing the tensor_meta.
            #
            # See torch/_inductor/overrides.py for where this is called upstream of fusion.
            self.fake_module = deepcopy_to_fake_tensor(self.module, fake_mode)
            self.fake_mode = fake_mode
        else:
            self.fake_module = None
            self.fake_mode = None

        self.real_module = self.module
        self.ignore_ops = ignore_ops
        self.mem_func_name = mem_func_name
        self.device = next(gm.parameters()).device

    @compatibility(is_backward_compatible=True)
    def call_module(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Execute a ``call_module`` node and return the result.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Any: The value returned by the module invocation
        """
        # Retrieve executed args and kwargs values from the environment

        assert isinstance(target, str)
        submod = self.fetch_attr(target)

        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        t_start = time.time()

        # Execute the method and return the result
        result = submod(*args, **kwargs)

        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        t_end = time.time()
        exec_time = (t_end - t_start) * 1000

        # 计算出来result之后再计算FLOPs，保证计算过程能正确执行
        mod_name = submod.__class__.__name__
        flops = None
        if mod_name in MODULE_FLOPs_MAPPING:
            if mod_name not in self.ignore_ops:
                flops = MODULE_FLOPs_MAPPING[mod_name](submod, result, *args, **kwargs)
            else:
                flops = 0

        return result, flops, exec_time

    @compatibility(is_backward_compatible=True)
    def call_function(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Execute a ``call_function`` node and return the result.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Any: The value returned by the function invocation
        """
        assert not isinstance(target, str)

        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        t_start = time.time()

        # Execute the function and return the result
        result = target(*args, **kwargs)

        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        t_end = time.time()
        exec_time = (t_end - t_start) * 1000

        # 计算出来result之后再计算FLOPs，保证计算过程能正确执行
        func_name = target.__name__
        flops = None
        if func_name in FUNCTION_FLOPs_MAPPING:
            if func_name not in self.ignore_ops:
                flops = FUNCTION_FLOPs_MAPPING[func_name](result, *args, **kwargs)
            else:
                flops = 0

        return result, flops, exec_time

    @compatibility(is_backward_compatible=True)
    def call_method(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Execute a ``call_method`` node and return the result.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Any: The value returned by the method invocation
        """
        # args[0] is the `self` object for this method call
        self_obj, *args_tail = args

        assert isinstance(target, str)

        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        t_start = time.time()

        # Execute the method and return the result
        result = getattr(self_obj, target)(*args_tail, **kwargs)

        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        t_end = time.time()
        exec_time = (t_end - t_start) * 1000

        # 计算出来result之后再计算FLOPs，保证计算过程能正确执行
        method_name = target
        flops = None
        if method_name in METHOD_FLOPs_MAPPING:
            if method_name not in self.ignore_ops:
                flops = METHOD_FLOPs_MAPPING[method_name](self_obj, result, *args_tail, **kwargs)
            else:
                flops = 0
        return result, flops, exec_time

    def run_node(self, n: Node) -> Any:
        try:
            if self.fake_module is not None:
                # Hacky swap. Alternatively, we could do this with overriding
                # call_module and get_attr.
                self.module = self.fake_module
            try:
                if self.fake_mode is not None:
                    raise ValueError("'fake_mode' must be None.")
                else:
                    with self._set_current_node(n):
                        args, kwargs = self.fetch_args_kwargs_from_env(n)
                        assert isinstance(args, tuple)
                        assert isinstance(kwargs, dict)

                        mem_func = getattr(torch.cuda, self.mem_func_name)
                        if self.mem_func_name == 'max_memory_allocated':
                            torch.cuda.reset_peak_memory_stats(self.device)
                        m_start = mem_func(self.device)

                        if n.op in ('call_module', 'call_function', 'call_method'):
                            result, flops, exec_time = getattr(self, n.op)(n.target, args, kwargs)
                        else:
                            if self.device.type == 'cuda':
                                torch.cuda.synchronize(self.device)
                            t_start = time.time()

                            result = getattr(self, n.op)(n.target, args, kwargs)

                            if self.device.type == 'cuda':
                                torch.cuda.synchronize(self.device)
                            t_end = time.time()
                            exec_time = (t_end - t_start) * 1000

                            flops = 0

                        m_end = mem_func(self.device)

                        assert flops not in n.meta, n.meta.keys()

                        n.meta['flops'] = flops
                        n.meta['time'] = exec_time
                        n.meta['mem_before'] = m_start
                        n.meta['mem_after'] = m_end
                        n.meta['mem_delta'] = m_end - m_start
            finally:
                self.module = self.real_module
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(
                f"ShapeProp error for: node={n.format_node()} with "
                f"meta={n.meta}"
            ) from e

        found_tensor = False

        def extract_tensor_meta(obj):
            if isinstance(obj, torch.Tensor):
                nonlocal found_tensor
                found_tensor = True
                return _extract_tensor_metadata(obj)
            else:
                return obj

        meta = map_aggregate(result, extract_tensor_meta)
        if found_tensor:
            n.meta['tensor_meta'] = meta

        n.meta['type'] = type(result)

        return result

    def propagate(self, *args):
        """
        Run `module` via interpretation and return the result and
        record the shape and type of each node.

        Args:
            *args (Tensor): the sample input.

        Returns:
            Any: The value returned from executing the Module
        """
        if self.fake_mode is not None:
            raise ValueError("'fake_mode' must be None.")
            fake_args = [self.fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t for t in args]
        else:
            fake_args = args
        return super().run(*fake_args)


class TorchFLOPsByFX():
    def __init__(self, model: nn.Module, mem_func_name: Literal['memory_allocated', 'max_memory_allocated'] = 'max_memory_allocated', ignore_ops: Sequence[str] = []):
        '''
        model: the model.
        mem_func_name: which function to measure the GPU memory; choosed from 'memory_allocated' and 'max_memory_allocated'; default: 'max_memory_allocated'.
        ignore_ops: the operations to be ignored for counting FLOPs.
        '''
        model.eval()
        try:
            self.graph_model: GraphModule = symbolic_trace(model)
        except torch.fx.proxy.TraceError as e:
            print("\033[33mNOTE: The model cannot be built as a graph model by 'symbolic_trace()'. Please remove the `assert`, `if` and `for` operations. " +
                  "See 'https://pytorch.org/docs/stable/fx.html#limitations-of-symbolic-tracing' for more instructions.\033[0m")
            raise e
        except TypeError as e:
            print("\033[33mNOTE: The model cannot be built as a graph model by 'symbolic_trace()'. Please replace the `tensor.shape[i]` that servers as the parameter of a function with a pre-defined deterministic value.\033[0m")
            raise e

        assert mem_func_name in ['memory_allocated', 'max_memory_allocated']
        self.mem_func_name = mem_func_name
        if isinstance(ignore_ops, str):
            ignore_ops = [ignore_ops]
        self.ignore_ops = deepcopy(ignore_ops)

        self.result_table = []
        self.result_header = ['node_name', 'node_op', 'op_target', 'which_module', 'flops', 'time(ms)', 'mem_before_op(B)', 'mem_after_op(B)', 'mem_delta(B)']
        self.__missing_values = [''] * 4 + ['ERROR']
        self.__flag_propagated = False

    def propagate(self, *args):
        ShapeProp(self.graph_model, mem_func_name=self.mem_func_name, ignore_ops=self.ignore_ops).propagate(*args)

        result_table = []
        for node in self.graph_model.graph.nodes:
            node: Node

            _target_str = str(node.target)
            if (_pattern := ' at 0x') in _target_str:
                _target_str = f"{_target_str.split(_pattern)[0]}>"
            if (_pattern := ' of type object') in _target_str:
                _target_str = f"{_target_str.split(_pattern)[0]}>"

            _result_row = [node.name, node.op, _target_str]

            node_module_name = ''
            if (_var_name := 'nn_module_stack') in node.meta:
                _modu = next(reversed(node.meta[_var_name].values()))
                if type(_modu) is tuple:
                    node_module_name = _modu[1].__name__
                else:
                    node_module_name = _modu.__name__
                # node_module_name = ".".join([_v.__name__ for _v in node.meta[_var_name].values()])
            _result_row.append(node_module_name)

            for _var_name in ('flops', 'time', 'mem_before', 'mem_after', 'mem_delta'):
                if _var_name in node.meta:
                    _var_val = node.meta[_var_name]
                    if _var_val is None:
                        _result_row.append('not_recognized')
                    elif isinstance(_var_val, (int, float)):
                        if node_module_name in self.ignore_ops:
                            _result_row.append('ignored')
                        else:
                            _result_row.append(_var_val)
                    else:
                        raise TypeError(type(_var_val))
                else:
                    raise KeyError(f"'{_var_name}' must be in node.meta")

            assert len(_result_row) == len(self.result_header)
            result_table.append(_result_row)

        self.result_table = result_table
        self.__flag_propagated = True

    def print_result_table(self, show: bool = True) -> List[List[Union[str, int, float]]]:
        '''
        Print the full result table.
        return: the results in a 2D list (excluding the head of the table).
        '''
        table_str = tabulate(self.result_table, self.result_header, tablefmt='rst',
                             intfmt=[''] * 4 + [','] + [''] + [','] * 2 + ['+,'],
                             floatfmt='.3f',
                             missingval=self.__missing_values)
        if show:
            print(table_str)
        return self.result_table

    def print_total_flops(self, show: bool = True) -> int:
        if not self.__flag_propagated:
            raise RuntimeError(f"Use `propagate()` method first.")

        valid_flops_list = list(filter(lambda _f: isinstance(_f, int), list(zip(*self.result_table))[4]))
        total_flops = sum(valid_flops_list)
        num_empty_flops = len(self.result_table) - len(valid_flops_list)

        if show:
            print(f"total_flops = {total_flops:3,}", f"({num_empty_flops} operations are ignored or not recognized)" if num_empty_flops else "")

        """
        total_flops = None
        try:
            total_flops = sum(filter(list(zip(*self.result_table))[-1]))
        except TypeError as e:
            print("\033[33mNOTE: There may be some operations not recognized. Please check them using `print_result_table()` and then add them to 'flops_ops.py'\033[0m")
            self.print_result_table()
            print(f"\033[33mNot Recognized: {set([_m for _m,_f in zip(*list(zip(*self.result_table))[-2:]) if _f is None])}\033[0m")
            print(f"\033[31m{traceback.format_exc()}\033[0m")
            exit(-1)
        """
        return total_flops

    def print_total_time(self, show: bool = True) -> float:
        if not self.__flag_propagated:
            raise RuntimeError(f"Use `propagate()` method first.")

        valid_time_list = list(zip(*self.result_table))[5]
        total_time = sum(valid_time_list)

        if show:
            print(f"total_time = {total_time:.3f} ms")

        return total_time

    def print_max_memory(self, show=True) -> int:
        if not self.__flag_propagated:
            raise RuntimeError(f"Use `propagate()` method first.")

        valid_mem_list = list(zip(*self.result_table))[7]
        max_mem = max(valid_mem_list)

        if show:
            print(f"max_memory = {max_mem:3,} Bytes")

        return max_mem

    def save_result_to_csv(self, file_path: str, mode: str = 'a'):
        with open(file_path, mode) as f:
            writer = csv.writer(f)
            writer.writerow(self.result_header)
            writer.writerows(self.result_table)
            f.write('\n')
