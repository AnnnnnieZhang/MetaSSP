import json
import os
import math

def load_json(file_path):
    """加载 JSON 文件，并返回数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json(data, file_path):
    """保存数据到 JSON 文件"""
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def count_categories(data):
    """统计每个类别的样本数，并返回一个字典"""
    counts = {}
    for record in data:
        # 假设类别在每个元素的第三个位置
        category = record[2]
        counts[category] = counts.get(category, 0) + 1
    return counts

def main():
    # 文件路径
    all_json_path = 'dataset/Pix3D/split/train/all.json'
    
    # 1. 加载数据
    data = load_json(all_json_path)
    
    # 2. 统计各类别的样本数量，并打印出来
    category_counts = count_categories(data)
    print("各类别样本总数：")
    for category, count in category_counts.items():
        print(f"  {category}: {count}")
    
    # 3. 用户输入分割的百分比（例如输入1表示1%）
    while True:
        try:
            user_input = input("请输入分割的百分比（正整数，代表每个类别抽取的比例，比如1表示1%）：")
            p = int(user_input)
            if p <= 0 or p >= 100:
                print("请输入一个大于0且小于100的整数。")
                continue
            break
        except ValueError:
            print("输入有误，请输入一个整数。")
    
    # 4. 计算每个类别需要抽取的样本个数（向上取整，至少1个）
    thresholds = {}
    for category, total_count in category_counts.items():
        # 计算比例，至少保证1个样本
        num_select = math.ceil(total_count * (p / 100.0))
        thresholds[category] = num_select
    print("\n每个类别根据分割比例预期抽取的样本数：")
    for category, num in thresholds.items():
        print(f"  {category}: {num}")
    
    # 5. 按照原始顺序分割数据
    supervised_data = []   # 存储抽取的部分
    unsupervised_data = [] # 存储剩余的部分
    current_counts = {}    # 跟踪每个类别已抽取的数量
    
    for record in data:
        category = record[2]
        current_count = current_counts.get(category, 0)
        if current_count < thresholds[category]:
            supervised_data.append(record)
            current_counts[category] = current_count + 1
        else:
            unsupervised_data.append(record)
    
    # 6. 对分割后的数据再做一次检查，统计每个类别在抽取部分的数量
    supervised_counts = count_categories(supervised_data)
    print("\n分割后【抽取的部分】各类别样本数量：")
    for category, count in supervised_counts.items():
        print(f"  {category}: {count}")
    
    # 7. 保存分割后的数据到对应文件
    supervised_file_path = f'dataset/Pix3D/split/train/{p}%.json'
    remaining_percentage = 100 - p
    unsupervised_file_path = f'dataset/Pix3D/split/train/{remaining_percentage}%.json'
    
    save_json(supervised_data, supervised_file_path)
    save_json(unsupervised_data, unsupervised_file_path)
    
    print("\n数据分割完成：")
    print(f"  抽取的部分（{p}%）已保存到：{supervised_file_path}")
    print(f"  剩余部分（{remaining_percentage}%）已保存到：{unsupervised_file_path}")

if __name__ == "__main__":
    main()
