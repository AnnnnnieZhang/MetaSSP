import torch
import torch.nn as nn
from ssr.model.embed import get_embedder

class SDFDiscriminatorWithSurfacePoints(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, dropout_prob=0.3, multires=6):
        super(SDFDiscriminatorWithSurfacePoints, self).__init__()
        
        self.embed_fn, self.out_dim = get_embedder(multires, input_dims=input_dim)
        full_input_dim = 39 + 10  # xyz扩展后的维度 + sdf扩展后的维度
        
        self.layers = nn.Sequential(
            nn.Linear(full_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Linear(hidden_dim // 2, 1),
            nn.LeakyReLU(negative_slope=0.1),  # 替换 Sigmoid 前的激活函数
            nn.Sigmoid()
        )

    def extract_surface_points(self, input_data, sdf_threshold=10):
        mask = torch.abs(input_data[:, 3]) < sdf_threshold
        surface_points = input_data[mask, :]
        return surface_points

    def forward(self, input_data, batch_size, sdf_threshold=10):
        total_points = input_data.size(0)
        points_per_sample = total_points // batch_size

        if total_points % batch_size != 0:
            raise ValueError(f"Total points ({total_points}) must be divisible by batch_size ({batch_size}).")

        input_data = input_data.view(batch_size, points_per_sample, -1)
        batch_probabilities = []

        for i in range(batch_size):
            sample_data = input_data[i]
            surface_points = self.extract_surface_points(sample_data, sdf_threshold)

            if surface_points.size(0) == 0:
                batch_probabilities.append(torch.tensor(0.5, device=sample_data.device))
                continue

            xyz = surface_points[:, :3]
            sdf = surface_points[:, 3:]

            xyz_encoded = self.embed_fn(xyz)

            # SDF 编码扩展
            sdf_encoded = torch.cat([torch.sin(sdf * freq) for freq in torch.linspace(1, 10, steps=10)], dim=-1)
            
            # 拼接 xyz 和 sdf 的编码
            combined_input = torch.cat([xyz_encoded, sdf_encoded], dim=-1)

            output = self.layers(combined_input)
            batch_probabilities.append(output.mean())

        return torch.stack(batch_probabilities).mean().unsqueeze(0)
