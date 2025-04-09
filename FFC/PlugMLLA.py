
import torch
import torch.nn as nn
from timm.models.layers import DropPath
# 论文：Demystify Mamba in Vision: A Linear Attention Perspective (arXiv24年5月)
# 论文地址：https://arxiv.org/pdf/2405.16605

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE(base=10000)

    def forward(self, x, input_resolution):
        b, n, c = x.shape
        num_heads = self.num_heads
        head_dim = c // num_heads

        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k = qk[0], qk[1]

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = self.rope(q.reshape(b, n, c), input_resolution).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, n, c), input_resolution).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = x.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, n, c).permute(0, 2, 1)
        x = x + self.lepe(v).permute(0, 2, 1).reshape(b, n, c)

        return x


class RoPE(nn.Module):
    def __init__(self, base=10000):
        super(RoPE, self).__init__()
        self.base = base

    def forward(self, x, input_resolution):
        feature_dim = x.shape[-1]
        k_max = feature_dim // 2

        theta_ks = 1 / (self.base ** (torch.arange(k_max) / k_max))
        angles = torch.arange(input_resolution).unsqueeze(-1) * theta_ks

        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1).to(x.device)

        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)


class MLLABlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.attn = LinearAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, L, C = x.shape
        x = x + self.cpe1(x.permute(0, 2, 1)).permute(0, 2, 1)
        shortcut = x

        x = self.norm1(x)
        act_res = self.act(self.act_proj(x))
        x = self.in_proj(x).permute(0, 2, 1)
        x = self.act(self.dwc(x)).permute(0, 2, 1)

        x = self.attn(x, L)  # 动态传递长度
        x = self.out_proj(x * act_res)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


if __name__ == '__main__':
    mlla_block = MLLABlock(dim=64, input_resolution=1024)

    batch_size = 1
    N = 1024
    input_tensor = torch.randn(batch_size, N, 64)

    output_tensor = mlla_block(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)