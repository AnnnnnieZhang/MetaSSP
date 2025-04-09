import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba

class MambaDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        """
        使用 Mamba 逐点处理的解码器
        Args:
            input_dim (int): 输入特征维度 (C)。
            hidden_dim (int): Mamba 隐藏层维度。
            output_dim (int): 输出特征维度，默认为 1 (SDF)。
        """
        super(MambaDecoder, self).__init__()
        self.mamba = Mamba(
            d_model=input_dim,  # 输入特征维度
            d_state=hidden_dim,  # 隐藏状态维度
            d_conv=4,           # 卷积维度
            expand=2,           # 扩展比例
            dt_rank="auto"      # 时间步矩阵的秩
        )

        # 最终输出层，将 Mamba 输出映射到目标维度
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        前向传播
        Args:
            x (Tensor): 输入张量，形状为 [N, C]。
        Returns:
            Tensor: 输出张量，形状为 [N, 1]。
        """
        x = x.unsqueeze(0)  # 添加 batch 维度 [1, N, C]
        x = self.mamba(x)  # [1, N, C]
        x = x.squeeze(0)   # 去掉 batch 维度 [N, C]
        x = self.fc(x)     # 映射到目标维度 [N, 1]
        return x

# 测试 MambaDecoder
if __name__ == "__main__":
    input_dim = 256  # 输入特征维度
    hidden_dim = 128  # Mamba 隐藏状态维度
    output_dim = 1    # 输出维度 (SDF)
    num_points = 1024

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    decoder = MambaDecoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)

    # 创建一个输入张量
    input_tensor = torch.randn(num_points, input_dim).to(device)  # [N, C]
    output = decoder(input_tensor)  # [N, 1]
    
    print("Output shape:", output.shape)  # 应输出: [1024, 1]
