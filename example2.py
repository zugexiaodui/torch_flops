import os
os.environ['TIMM_FUSED_ATTN'] = "0"
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from timm.layers import use_fused_attn, Mlp, DropPath

from torch_flops import TorchFLOPsByFX


class Attention(nn.Module):
    '''
    REF: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    '''
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    @torch.no_grad()
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, y):
        x = x + y
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


if __name__ == "__main__":
    C = 768
    device = 'cuda:0'

    # Define the model: an attention block (refer to "timm": https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py)
    block = Block(C, num_heads=2, qkv_bias=True)
    block.attn.fused_attn = False
    block.eval()
    model = block

    # Input
    # N: number of tokens
    N = 14**2 + 1
    B = 1
    x = torch.randn([B, N, C]).to(device)
    y = torch.randn([B, N, C]).to(device)
    model.to(device)

    # NOTE: First run the model once for accurate time measurement in the following process.
    with torch.no_grad():
        model(x, y)

    # Output
    # Build the graph of the model. You can specify the operations (listed in `MODULE_FLOPs_MAPPING`, `FUNCTION_FLOPs_MAPPING` and `METHOD_FLOPs_MAPPING` in 'flops_ops.py') to ignore.
    flops_counter = TorchFLOPsByFX(model)
    # Print the grath (not essential)
    print('*' * 120)
    flops_counter.graph_model.graph.print_tabular()
    # Feed the input tensor
    with torch.no_grad():
        flops_counter.propagate(x, y)
    # Print the flops of each node in the graph. Note that if there are unsupported operations, the "flops" of these ops will be marked as 'not recognized'.
    print('*' * 120)
    result_table = flops_counter.print_result_table()
    # Print the total FLOPs
    total_flops = flops_counter.print_total_flops()
    total_time = flops_counter.print_total_time()
    max_memory = flops_counter.print_max_memory()
    flops_counter.save_result_to_csv("./result.csv", 'w')
