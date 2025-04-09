import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

class MultiHeadPrototypeAttention(nn.Module):
    def __init__(self, img_feat_dim, proto_dim, num_heads=4, prototype_path="SSP3D/proto_features/proto_kmeans.pkl"):
        super().__init__()
        self.img_feat_dim = img_feat_dim
        self.proto_dim = proto_dim
        self.num_heads = num_heads
        self.head_dim = proto_dim // num_heads
        assert proto_dim % num_heads == 0, "proto_dim 必须能被 num_heads 整除"

        # 类别映射
        self.category_label_mapping = {
            "table": 0, "sofa": 1, "wardrobe": 2, "tool": 3,
            "chair": 4, "bookcase": 5, "bed": 6, "desk": 7, "misc": 8
        }

        # 对 `XYZ` 和 `SDF` 分开处理
        self.xyz_processor = nn.Linear(3, proto_dim // 2)
        self.sdf_processor = nn.Linear(1, proto_dim // 2)

        # 增加交互机制
        self.interaction_layer = nn.Sequential(
            nn.Linear(proto_dim, proto_dim),
            nn.ReLU()
        )

        # 加载几何先验
        with open(prototype_path, "rb") as f:
            proto_data = pickle.load(f)

        proto_tensor = []
        for cat in self.category_label_mapping.keys():
            proto_feature = torch.tensor(proto_data[cat])  # [6272, 4]
            xyz = proto_feature[:, :3]  # 提取 xyz
            sdf = proto_feature[:, 3:]  # 提取 sdf

            # 分别处理 xyz 和 sdf
            xyz_feature = self.xyz_processor(xyz)  # [6272, proto_dim // 2]
            sdf_feature = self.sdf_processor(sdf)  # [6272, proto_dim // 2]

            # 简单交互：xyz 和 sdf 相加 + 交互机制
            interaction_feature = torch.cat([xyz_feature, sdf_feature], dim=-1)  # [6272, proto_dim]
            fused_feature = self.interaction_layer(interaction_feature)  # [6272, proto_dim]

            proto_tensor.append(fused_feature)

        # 创建类别原型张量
        self.prototype_vectors = nn.Parameter(torch.stack(proto_tensor, dim=0), requires_grad=False)  # [9, 6272, proto_dim]

        # 查询、键、值的线性投影
        self.query_proj = nn.Linear(img_feat_dim, proto_dim)
        self.key_proj = nn.Linear(proto_dim, proto_dim)
        self.value_proj = nn.Linear(proto_dim, proto_dim)

    def split_heads(self, x):
        batch_size, seq_len, dim = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def combine_heads(self, x):
        batch_size, num_heads, seq_len, head_dim = x.size()
        return x.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, num_heads * head_dim)

    def forward(self, img_feats):
        """
        img_feats: 输入的特征 [B, img_feat_dim + 9]
        """
        # 分解图像和类别特征
        img_feat_detail = img_feats[:, :self.img_feat_dim]  # 图像特征 [B, img_feat_dim]
        category_feat = img_feats[:, self.img_feat_dim:]   # 类别特征 [B, 9]

        # 查询特征
        query = self.query_proj(img_feat_detail).unsqueeze(1)  # [B, 1, proto_dim]

        # 动态类别权重
        category_weights = F.softmax(category_feat, dim=1)  # [B, 9]

        # 动态融合几何先验
        weighted_protos = torch.einsum("bc,cpd->bpd", category_weights, self.prototype_vectors)  # [B, 6272, proto_dim]

        # 原型特征投影
        keys = self.key_proj(weighted_protos)  # [B, 6272, proto_dim]
        values = self.value_proj(weighted_protos)  # [B, 6272, proto_dim]

        # 多头注意力
        query = self.split_heads(query)  # [B, num_heads, 1, head_dim]
        keys = self.split_heads(keys)  # [B, num_heads, 6272, head_dim]
        values = self.split_heads(values)  # [B, num_heads, 6272, head_dim]

        attn_scores = torch.matmul(query, keys.transpose(-1, -2)) / (self.head_dim ** 0.5)  # [B, num_heads, 1, 6272]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, num_heads, 1, 6272]

        # 注意力加权
        weighted_proto = torch.matmul(attn_weights, values)  # [B, num_heads, 1, head_dim]
        weighted_proto = self.combine_heads(weighted_proto).squeeze(1)  # [B, proto_dim]

        return weighted_proto, attn_weights


class MultiHeadPrototypeAttention_0(nn.Module):
    def __init__(self, img_feat_dim, proto_dim, num_heads=4, mlp_hidden_dim=512, prototype_path="SSP3D/proto_features/proto_kmeans_sdf.pkl"):
        """
        img_feat_dim: 图像特征维度（256）。
        proto_dim: 原型特征维度（256）。
        num_heads: 注意力头的数量。
        mlp_hidden_dim: MLP 的隐藏层维度。
        prototype_path: 预训练的几何原型文件路径。
        """
        super().__init__()
        self.img_feat_dim = img_feat_dim
        self.proto_dim = proto_dim
        self.num_heads = num_heads
        self.head_dim = proto_dim // num_heads

        assert proto_dim % num_heads == 0, "proto_dim 必须能被 num_heads 整除"

        # 多头注意力的查询、键和值投影
        self.query_proj = nn.Linear(img_feat_dim, proto_dim)
        self.key_proj = nn.Linear(1, proto_dim)  # 从 SDF 值投影到潜在空间
        self.value_proj = nn.Linear(1, proto_dim)

        # MLP 用于输出融合
        self.mlp = nn.Sequential(
            nn.Linear(proto_dim + img_feat_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, img_feat_dim)
        )

        # 加载预训练的几何原型
        self.prototype_vectors = self.load_prototypes(prototype_path)

    def load_prototypes(self, path):
        """
        加载预训练的几何原型文件。
        """
        with open(path, "rb") as f:
            prototypes = pickle.load(f)
        # 将每个类别的原型存为 tensor (num_prototypes, point_dim, 1)
        proto_tensor = torch.stack([torch.tensor(proto) for proto in prototypes.values()], dim=0)  # [num_prototypes, point_dim, 1]
        return nn.Parameter(proto_tensor, requires_grad=False)  # 不需要更新

    def split_heads(self, x):
        """
        将最后一个维度分为多头，并调整维度顺序。
        """
        batch_size, seq_len, dim = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # [B, num_heads, seq_len, head_dim]

    def combine_heads(self, x):
        """
        将多头的维度合并回原始维度。
        """
        batch_size, num_heads, seq_len, head_dim = x.size()
        return x.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, num_heads * head_dim)

    def forward(self, img_feats):
        """
        img_feats: 图像特征输入 [B, 256]。
        """
        # 查询计算
        query = self.query_proj(img_feats).unsqueeze(1)  # [B, 1, 256]
        proto_flat = self.prototype_vectors.view(-1, 1)  # [num_prototypes * point_dim, 1]

        # 键和值计算
        keys = self.key_proj(proto_flat).unsqueeze(0)  # [1, num_prototypes * point_dim, 256]
        values = self.value_proj(proto_flat).unsqueeze(0)  # [1, num_prototypes * point_dim, 256]

        # 多头分割
        query = self.split_heads(query)  # [B, num_heads, 1, head_dim]
        keys = self.split_heads(keys)  # [1, num_heads, num_prototypes * point_dim, head_dim]
        values = self.split_heads(values)  # [1, num_heads, num_prototypes * point_dim, head_dim]

        # 注意力分数计算
        attn_scores = torch.matmul(query, keys.transpose(-1, -2)) / (self.head_dim ** 0.5)  # [B, num_heads, 1, num_prototypes * point_dim]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, num_heads, 1, num_prototypes * point_dim]

        # 加权求和
        weighted_proto = torch.matmul(attn_weights, values)  # [B, num_heads, 1, head_dim]
        weighted_proto = self.combine_heads(weighted_proto).squeeze(1)  # [B, proto_dim]

        # 特征融合
        combined_feat = torch.cat([img_feats, weighted_proto], dim=-1)  # [B, 512]
        # output_feat = self.mlp(combined_feat)  # [B, 256]

        return combined_feat, attn_weights

    


class MultiHeadPrototypeAttention3D(nn.Module):
    def __init__(self, img_feat_dim, proto_dim, num_prototypes, point_dim, num_heads=4, mlp_hidden_dim=512):
        """
        img_feat_dim: 图像特征维度（256）。
        proto_dim: 原型特征维度（256）。
        num_prototypes: 原型类别数量（9）。
        point_dim: 每个原型的点数量（6272）。
        num_heads: 注意力头的数量。
        mlp_hidden_dim: MLP 的隐藏层维度。
        """
        super().__init__()
        self.img_feat_dim = img_feat_dim
        self.proto_dim = proto_dim
        self.num_prototypes = num_prototypes
        self.point_dim = point_dim
        self.num_heads = num_heads
        self.head_dim = proto_dim // num_heads

        assert proto_dim % num_heads == 0, "proto_dim 必须能被 num_heads 整除"

        # 多头注意力的查询、键和值投影
        self.query_proj = nn.Linear(img_feat_dim, proto_dim)
        self.key_proj = nn.Linear(proto_dim, proto_dim)
        self.value_proj = nn.Linear(proto_dim, proto_dim)

        # MLP 用于输出融合
        self.mlp = nn.Sequential(
            nn.Linear(proto_dim + img_feat_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, img_feat_dim)
        )

        # 加载预训练的原型特征
        self.prototype_vectors = self.load_prototypes("SSP3D/proto_features_256/proto_kmeans_3D.pkl")

    def load_prototypes(self, path):
        with open(path, "rb") as f:
            prototypes = pickle.load(f)
        # 将每个类别的原型存为 tensor (num_prototypes, point_dim, proto_dim)
        proto_tensor = torch.stack([torch.tensor(proto) for proto in prototypes.values()], dim=0)
        return nn.Parameter(proto_tensor, requires_grad=False)  # 不需要更新

    def split_heads(self, x):
        """
        将最后一个维度分为多头，并调整维度顺序。
        """
        batch_size, seq_len, dim = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # [B, num_heads, seq_len, head_dim]

    def combine_heads(self, x):
        """
        将多头的维度合并回原始维度。
        """
        batch_size, num_heads, seq_len, head_dim = x.size()
        return x.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, num_heads * head_dim)

    def forward(self, img_feats):
        """
        img_feats: 图像特征输入 [B, 256]。
        """
        # 查询、键、值计算
        query = self.query_proj(img_feats).unsqueeze(1)  # [B, 1, 256]
        proto_flat = self.prototype_vectors.view(-1, self.proto_dim)  # [num_prototypes * point_dim, 256]

        keys = self.key_proj(proto_flat).unsqueeze(0)  # [1, num_prototypes * point_dim, 256]
        values = self.value_proj(proto_flat).unsqueeze(0)  # [1, num_prototypes * point_dim, 256]

        # 多头分割
        query = self.split_heads(query)  # [B, num_heads, 1, head_dim]
        keys = self.split_heads(keys)  # [1, num_heads, num_prototypes * point_dim, head_dim]
        values = self.split_heads(values)  # [1, num_heads, num_prototypes * point_dim, head_dim]

        # 注意力分数计算
        attn_scores = torch.matmul(query, keys.transpose(-1, -2)) / (self.head_dim ** 0.5)  # [B, num_heads, 1, num_prototypes * point_dim]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, num_heads, 1, num_prototypes * point_dim]

        # 加权求和
        weighted_proto = torch.matmul(attn_weights, values)  # [B, num_heads, 1, head_dim]
        weighted_proto = self.combine_heads(weighted_proto).squeeze(1)  # [B, proto_dim]

        # 特征融合
        combined_feat = torch.cat([img_feats, weighted_proto], dim=-1)  # [B, 512]
        output_feat = self.mlp(combined_feat)  # [B, 256]

        return output_feat, attn_weights

