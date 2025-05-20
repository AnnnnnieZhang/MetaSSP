import torch
from .img_enhance import batch_weak_augmentation

def compute_importance_weights(teacher_model, pseudo_loader, device, num_batches=None):
    """
    利用当前教师模型在无标签数据上的输出梯度，计算每个参数的重要性。
    计算逻辑：
      1. 将教师模型置于 eval 模式（保证统计时不受 dropout 或 BN 影响）。
      2. 对每个参数初始化零张量，用于累计梯度的绝对值。
      3. 遍历无标签数据（可限定使用部分 batch），对于每个 batch：
         - 将输入数据送入模型前先做弱增强（与 EMA 更新时一致）。
         - 得到输出后，以输出的 L2 范数平方作为衡量指标，反向传播计算梯度。
         - 将每个参数梯度的绝对值累加到对应的权重中。
      4. 累计结束后，对每个参数除以使用的 batch 数量，得到平均梯度敏感性。
      5. 恢复教师模型的 train 模式，返回 importance_weights 字典。

    Args:
        teacher_model: 教师模型
        pseudo_loader: 无标签数据的 DataLoader
        device: 设备（如 "cuda"）
        num_batches: (可选) 限定计算 batch 数，若为 None 则遍历全部

    Returns:
        importance_weights: dict, 每个参数名称对应一个 tensor
    """
    teacher_model.eval()
    importance_weights = {}
    for name, param in teacher_model.named_parameters():
        importance_weights[name] = torch.zeros_like(param, device=device)

    count = 0
    # 遍历无标签数据（这里假设每个 batch 的结构与训练时相似）
    for batch in pseudo_loader:
        # 假设 batch 的格式为: (pseudo_indices, pseudo_model_input)
        pseudo_indices, pseudo_model_input  = batch

        # 将数据转到 device，并进行弱增强
        pseudo_model_input = {k: v.to(device).to(torch.float32) for k, v in pseudo_model_input.items()}
        if "image" in pseudo_model_input:
            pseudo_model_input["image"] = batch_weak_augmentation(pseudo_model_input["image"])

        # 得到教师模型输出（根据你的模型结构调整调用方式）
        teacher_outputs = teacher_model(pseudo_model_input, pseudo_indices)
        # 例如这里以 'sdf' 输出为衡量指标，计算 L2 范数平方
        loss = teacher_outputs['sdf'].pow(2).sum()

        teacher_model.zero_grad()
        loss.backward()

        # 累计每个参数梯度的绝对值
        for name, param in teacher_model.named_parameters():
            if param.grad is not None:
                importance_weights[name] += param.grad.abs()

        count += 1
        if num_batches is not None and count >= num_batches:
            break

    # 求平均
    for name in importance_weights:
        importance_weights[name] /= count

    # teacher_model.train()
    return importance_weights
