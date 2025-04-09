import pickle
import numpy as np

# 加载已保存的特征文件
feature_file_path = "SSP3D/proto_features/feature_vectors.pkl"
with open(feature_file_path, "rb") as f:
    feature_data = pickle.load(f)

# 打印总特征数
total_features = sum(len(features) for features in feature_data.values())
print(f"总特征数: {total_features}")

# 打印每个类别的特征数量和形状
for category, features in feature_data.items():
    print(f"类别: {category}, 特征数量: {len(features)}, 特征形状: {np.array(features).shape}")

# 对每个类别计算整体中心特征
center_features = {}
for category, features in feature_data.items():
    features_array = np.array(features)  # 转为 NumPy 数组
    print(f"类别 {category} 的原始形状: {features_array.shape}")  # (n, point, 256)

    # 逐点计算平均值，保留点和特征维度作为一个整体
    center = np.mean(features_array, axis=0)  # (point, 256)
    print(f"类别 {category} 的中心点形状: {center.shape}")

    center_features[category] = center

# 保存中心特征和类别字典到文件
output_data = {
    "category_label_mapping": {
        "table": 0, "sofa": 1, "wardrobe": 2, "tool": 3,
        "chair": 4, "bookcase": 5, "bed": 6, "desk": 7, "misc": 8
    },
    "center_features": center_features
}

output_file_path = "SSP3D/proto_features/proto_kmeans_sdf.pkl"
with open(output_file_path, "wb") as f:
    pickle.dump(output_data, f)

print(f"类别中心特征和类别映射已保存到: {output_file_path}")
