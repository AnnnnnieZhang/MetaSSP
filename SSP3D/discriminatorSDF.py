import torch
import torch.nn as nn
import torch.nn.functional as F

class SDFDiscriminator(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128):
        """
        输入点的维度为 (x, y, z, sdf)
        """
        super(SDFDiscriminator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 输出为概率值
            nn.Sigmoid()
        )

    def forward(self, points):
        """
        Args:
            points (Tensor): 输入点，形状为 [batch_size, num_points, 4]

        Returns:
            output (Tensor): 判别结果，形状为 [batch_size, num_points, 1]
        """
        batch_size, num_points, _ = points.shape
        points = points.view(batch_size * num_points, -1)  # 展平成 [batch_size * num_points, 4]
        output = self.mlp(points)  # 输出形状为 [batch_size * num_points, 1]
        return output.view(batch_size, num_points, 1)  # 重塑回 [batch_size, num_points, 1]
