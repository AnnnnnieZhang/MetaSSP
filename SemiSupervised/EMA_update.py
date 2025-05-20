import torch
import math
from collections import OrderedDict
from torch import nn
import torch.nn.functional as F

def ema_update(teacher_model, student_model, step, total_steps, initial_momentum, importance_weights=None, eta=0.0):
    """
    Reflective Teacher 风格的 EMA 更新
    此函数使用余弦退火来动态调整 EMA 的动量，并在更新时根据参数的重要性进行正则化，
    防止对关键信息的破坏，降低灾难性遗忘。

    Args:
        teacher_model: 教师模型
        student_model: 学生模型
        step: 当前训练步数
        total_steps: 总训练步数
        initial_momentum: 初始 EMA 动量（例如 0.99）
        importance_weights: (可选) 一个与教师模型参数对应的字典，记录每个参数的重要性权重，
                            例如通过梯度敏感性计算获得。如果为 None，则不使用反思性正则化。
        eta: 正则化超参数，用来控制重要性正则项的作用大小（应设置为较小的正值）
    """
    # 使用余弦退火策略计算动态 EMA 动量
    momentum = 1 - (1 - initial_momentum) * (math.cos(math.pi * step / total_steps) + 1) / 2

    # 获取教师模型和学生模型的状态字典
    teacher_state = teacher_model.state_dict()
    student_state = student_model.state_dict()
    new_teacher_state = OrderedDict()

    # 对每个参数进行更新
    for key, teacher_param in teacher_state.items():
        if key in student_state:
            student_param = student_state[key]
            
            # 基础的 EMA 更新：teacher_new = momentum * teacher_old + (1 - momentum) * student
            ema_updated = teacher_param * momentum + student_param * (1 - momentum)
            
            # 如果提供了参数重要性，则对更新进行反思性正则化
            if importance_weights is not None and key in importance_weights:
                # 计算更新差值
                diff = student_param - teacher_param
                # 根据重要性缩放更新量：重要性越高，更新越小
                # 此处采用简单缩放方案，你可以根据需要调整缩放函数
                regularized_update = diff * (1 - momentum) / (1.0 + eta * importance_weights[key])
                # 更新后的参数为：旧值加上正则化后的更新量
                ema_updated = teacher_param + regularized_update
            
            new_teacher_state[key] = ema_updated
        else:
            raise ValueError(f"Key {key} not found in student model state dict (after prefix adjustment)")

    # 将更新后的参数加载回教师模型
    teacher_model.load_state_dict(new_teacher_state)

def clean_increment_ema(
    teacher_model,
    student_model,
    train_loader,       # 替换成你的 10% 有标签 DataLoader
    momentum,
    device,
    loss,          # 你的 loss(model_out, gt, epoch, label=True) 接口
    epoch,
    num_batches = 5
):
    """
    在每个 epoch 验证阶段，用有标签数据集做一次“干净增量 EMA”：
      1) 计算 Δθ = θ_student - θ_teacher
      2) 从 sup_loader 取一个 batch，eval teacher，算出对 loss 的梯度 g_sup
      3) 投影 Δθ → Δθ⁺（只保留能减少 sup loss 的方向）
      4) EMA 更新：θ_teacher ← m·θ_teacher + (1-m)·(θ_teacher + Δθ⁺)
    """

    # 1) Δθ = θ_S - θ_T
    t_state = teacher_model.state_dict()
    s_state = student_model.state_dict()
    delta = {
        k: s_state[k].to(device) - t_state[k].to(device)
        for k in t_state if k in s_state
    }

    # 2) 从 sup_loader 取一个 batch 计算“有标签”梯度
    teacher_model.eval()    # 不想影响 BN/Dropout 统计
    count = 0
    for batch in train_loader:
        indices, model_input, ground_truth = batch
        model_input = {k: v.to(device).to(torch.float32) for k,v in model_input.items()}

        # 假设 gt 已经在 dict 里，并且 loss 函数期望同 train 那样接口
        teacher_model.zero_grad()
        teacher_out = teacher_model(model_input, indices)  # 或者你的调用签名
        sup_loss = loss(teacher_out, ground_truth, epoch, label=True)
        sup_loss['total_loss'].backward()                   # 只用 total_loss
        # 收集梯度
        g_sup = {
            name: p.grad.clone().detach()
            for name, p in teacher_model.named_parameters()
            if p.grad is not None
        }
        teacher_model.zero_grad()

        # 3) 投影 Δθ → Δθ⁺
        delta_plus = {}
        for name, d in delta.items():
            grad = g_sup.get(name)
            if grad is not None:
                mask = (d * grad) < 0    # 只有沿 –grad 方向的成分能降低 loss
                delta_plus[name] = torch.where(mask, d, torch.zeros_like(d))
            else:
                delta_plus[name] = torch.zeros_like(d)

        # 4) EMA 更新教师参数（就地 load_state）
        new_state = OrderedDict()
        for name, p in teacher_model.state_dict().items():
            p_dev = p.to(device)
            dp = delta_plus.get(name)
            if dp is not None:
                # θ_new = m·θ_old + (1-m)·(θ_old + Δθ⁺)
                new_state[name] = p_dev * momentum + (p_dev + dp) * (1 - momentum)
            else:
                new_state[name] = p_dev

        teacher_model.load_state_dict(new_state)

        count += 1
        if num_batches is not None and count >= num_batches:
            break

    # teacher_model.train()  # 恢复 train 模式，以便后续正常使用

class MomentumController(nn.Module):
    """
    Meta-Controller for adaptive EMA momentum scheduling.
    Inputs: delta_loss (teacher - student), epoch_norm
    Outputs: gamma multiplier in [0.9, 1.1]
    """
    def __init__(self, hidden=16):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, delta_loss,tea_avg_eval_loss, epoch_norm):
        # delta_loss, epoch_norm: Tensors of shape []
        x = torch.stack([delta_loss,tea_avg_eval_loss, epoch_norm], dim=-1)  # shape [2]
        h = F.relu(self.fc1(x))
        # gamma in [0.995, 1.005]
        gamma = torch.sigmoid(self.fc2(h)) * 0.01 + 0.995
        return gamma.squeeze()