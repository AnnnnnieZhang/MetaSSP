import json
import os

def load_json(file_path):
    """读取 JSON 文件，并返回数据对象"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, file_path):
    """保存数据到 JSON 文件，若目录不存在则创建"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def process_pix3d_json(input_path, output_dir):
    """
    读取 Pix3D_10.json 数据后，
    根据不同的划分（train, unlabel, val, test），
    将数据整理为：
        [
            ["img/<类别>/<编号>.jpg", "255", "<类别>"],
            …
        ]
    并分别保存在 output_dir 下的 train.json、unlabel.json、val.json、test.json 中。
    """
    # 加载输入 JSON 数据
    data = load_json(input_path)
    
    # 定义存储四个不同划分数据的字典
    splits = {
        "train": [],
        "unlabel": [],
        "val": [],
        "test": []
    }
    
    # 遍历每个类别的字典
    for entry in data:
        # 使用 taxonomy_id（或 taxonomy_name）作为类别名称
        category = entry.get("taxonomy_id", "").strip()
        # 针对每个划分（键）的列表进行处理
        for split in splits.keys():
            file_ids = entry.get(split, [])
            for file_id in file_ids:
                # 构造图像路径：例如 "img/chair/0368.jpg"
                image_path = f"img/{category}/{file_id}.jpg"
                # 固定值 "255"，以及类别
                record = [image_path, "255", category]
                splits[split].append(record)
    
    # 将各个划分的数据分别保存到对应的 JSON 文件中
    for split, records in splits.items():
        output_path = os.path.join(output_dir, f"{split}.json")
        save_json(records, output_path)
        print(f"保存了 {len(records)} 条记录到 {output_path}")

if __name__ == "__main__":
    # 输入文件路径
    input_json_path = "dataset/Pix3D/SSMP/Pix3D_10.json"
    # 输出目录（同样的文件夹下）
    output_directory = "dataset/Pix3D/SSMP/"
    process_pix3d_json(input_json_path, output_directory)
