import torch
import pickle
import numpy as np

# 加载数据
input_file_path = "SSP3D/proto_features/feature_vectors.pkl"
output_file_path = "SSP3D/proto_features/proto_center.pkl"
with open(input_file_path, "rb") as f:
    data = pickle.load(f)

category_mapping = {
    "table": 0, "sofa": 1, "wardrobe": 2, "tool": 3,
    "chair": 4, "bookcase": 5, "bed": 6, "desk": 7, "misc": 8
}
category_prior = {}

for category, data_array in data.items():
    data_array = np.array(data_array)  # [1379, 6272, 4]
    num_objects, num_points, num_features = data_array.shape

    # Step 1: 计算全局平均点云
    global_center = data_array.mean(axis=0)  # [6272, 4]

    # Step 2: 找到距离最近的物体
    min_distance = float('inf')
    closest_idx = -1

    for i, obj in enumerate(data_array):
        # 逐点计算欧几里得距离
        dist = np.linalg.norm(obj - global_center)  # [6272, 4] -> 标量距离
        if dist < min_distance:
            min_distance = dist
            closest_idx = i

    # 保存距离最小的物体
    category_prior[category] = data_array[closest_idx]  # 原始点云 [6272, 4]

    print(f"Category: {category}, closest object shape: {data_array[closest_idx].shape}")

# 保存几何先验
with open(output_file_path, "wb") as f:
    pickle.dump(category_prior, f)

print(f"Geometry priors saved to {output_file_path}")
