import os
import pickle

# 字典映射
# category_label_mapping = {
#     "table": 0, "sofa": 1, "cabinet": 2, "night_stand": 3,
#     "chair": 4, "bookshelf": 5, "bed": 6, "desk": 7, "dresser": 8
# } --> front3d

category_label_mapping = {
    "table": 0, "sofa": 1, "wardrobe": 2, "tool": 3,
    "chair": 4, "bookcase": 5, "bed": 6, "desk": 7, "misc": 8
}   #--> pix3d


class FeatureSaver:
    def __init__(self, save_dir="features", filename="feature_vectors.pkl"):
        os.makedirs(save_dir, exist_ok=True)
        self.save_path = os.path.join(save_dir, filename)
        # 尝试加载现有数据
        if os.path.exists(self.save_path):
            with open(self.save_path, "rb") as f:
                self.data = pickle.load(f)
        else:
            # 初始化数据字典，包含所有类别
            self.data = {category: [] for category in category_label_mapping.keys()}

    def save_feature(self,category_vectors, feature_vectors):

        # 将类别向量映射为单词
        reverse_mapping = {v: k for k, v in category_label_mapping.items()}
        CAT = [reverse_mapping[vec.argmax().item()] for vec in category_vectors]  # 当前 batch 的类别名称列表

        # 将特征和类别保存到字典中
        for feature, cat in zip(feature_vectors, CAT):
            if cat not in self.data:
                raise KeyError(f"Category '{cat}' not found in initialized data keys.")
            # 使用 detach() 分离梯度后再保存
            self.data[cat].append(feature.detach().cpu().numpy())

    def write_to_file(self):
        # 统计总特征数量
        # total_features = sum(len(features) for features in self.data.values())
        
        # 写入文件
        with open(self.save_path, "wb") as f:
            pickle.dump(self.data, f)