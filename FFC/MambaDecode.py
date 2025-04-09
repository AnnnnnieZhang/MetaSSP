import torch
import torch.nn as nn
import einops
from torch.fft import fft, ifft

class EfficientAdditiveAttentionWithSSM(nn.Module):
    def __init__(self, in_dims, token_dim, num_heads=1, hidden_dim=256):
        super().__init__()

        # 线性投影
        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)

        # 状态空间相关参数
        self.state_proj = nn.Linear(token_dim, hidden_dim)
        self.state_update = nn.Linear(hidden_dim, token_dim)
        self.activation = nn.GELU()

        # 添加频域处理模块
        self.freq_proj = nn.Linear(in_dims, token_dim * num_heads)

        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x):
        # 输入特征的投影
        query = self.to_query(x)
        key = self.to_key(x)

        # 添加频域特征
        freq_features = fft(x, dim=1).real  # 提取频域特征 (仅实部)
        freq_proj = self.freq_proj(freq_features)

        query = torch.nn.functional.normalize(query, dim=-1)  # BxNxD
        key = torch.nn.functional.normalize(key, dim=-1)  # BxNxD

        # 状态空间特性：通过隐式更新捕获动态信息
        
        state = self.activation(self.state_proj(query))
        state = self.state_update(state)

        # 混合状态特性与频域特性
        query_weight = query @ self.w_g  # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor  # BxNx1
        A = torch.nn.functional.normalize(A, dim=1)  # BxNx1

        G = torch.sum(A * query, dim=1)  # BxD
        G = einops.repeat(G, "b d -> b repeat d", repeat=key.shape[1])  # BxNxD

        # 融合频域特性和状态更新结果
        combined_key = key + freq_proj + state

        out = self.Proj(G * combined_key) + query  # BxNxD
        out = self.final(out)  # BxNxD

        return out


if __name__ == '__main__':
    # 假设输入维度为512，token维度为512，头数为1
    attention_layer = EfficientAdditiveAttentionWithSSM(in_dims=512, token_dim=512)

    # 创建一个随机输入张量，形状为[B, N, D]
    B, N, D = 1, 10, 512
    x = torch.randn(B, N, D)

    # 通过注意力层传递输入
    output = attention_layer(x)

    # 打印输入和输出的形状
    print("输入形状:", x.shape)
    print("输出形状:", output.shape)
