"""Meta-adaptive EMA with parameter-importance regularisation (paper Sec. III-D)."""
import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn


def cosine_momentum(step, total_steps, initial_momentum):
    """Cosine-annealed base momentum m_base (paper Eq. 7).

    Starts at ``initial_momentum`` (m_0) and rises towards 1 as ``step``
    approaches ``total_steps``.
    """
    return 1 - (1 - initial_momentum) * (math.cos(math.pi * step / total_steps) + 1) / 2


def adaptive_momentum(step, total_steps, initial_momentum, gamma=1.0,
                      momentum_min=0.96, momentum_max=0.9996):
    """Meta-adaptive EMA momentum (paper Eq. 9).

        m = clip(gamma * m_base, m_min, m_max)

    where ``m_base`` is the cosine-annealed momentum (Eq. 7) and ``gamma`` is the
    output of the :class:`MomentumController` (Eq. 8).
    """
    m_base = cosine_momentum(step, total_steps, initial_momentum)
    return float(max(min(gamma * m_base, momentum_max), momentum_min))


def ema_update(teacher_model, student_model, momentum, importance_weights=None, eta=0.0):
    """Update the teacher from the student with an importance-regularised EMA.

    Without ``importance_weights`` this is a standard EMA step (paper Eq. 6)

        theta_T <- m * theta_T + (1 - m) * theta_S

    With them, each parameter moves by

        theta_T <- theta_T + (1 - m) / (1 + eta * omega) * (theta_S - theta_T)    (paper Eq. 10)

    so parameters with a large importance ``omega`` receive a smaller share of the
    student update.

    Args:
        teacher_model, student_model: networks with identical state-dict keys.
        momentum: EMA momentum m, e.g. from :func:`adaptive_momentum`.
        importance_weights: optional dict {parameter name: tensor} from
            ``metassp.importance.compute_importance_weights``.
        eta: strength of the importance regularisation (small positive value).
    """
    teacher_state = teacher_model.state_dict()
    student_state = student_model.state_dict()
    new_teacher_state = OrderedDict()

    for key, teacher_param in teacher_state.items():
        if key not in student_state:
            raise ValueError(f"Key {key} not found in the student state dict")
        student_param = student_state[key]

        if importance_weights is not None and key in importance_weights:
            diff = student_param - teacher_param
            scale = (1 - momentum) / (1.0 + eta * importance_weights[key])
            new_teacher_state[key] = teacher_param + diff * scale
        else:
            # Buffers (and all tensors when no importance is given): plain EMA.
            new_teacher_state[key] = teacher_param * momentum + student_param * (1 - momentum)

    teacher_model.load_state_dict(new_teacher_state)


class MomentumController(nn.Module):
    """Meta-controller that rescales the EMA momentum (paper Eq. 8).

    Inputs are the teacher-student SDF loss gap, the teacher evaluation loss and
    the normalised epoch index. The output multiplier is
    ``gamma = sigmoid(MLP(x)) * 0.01 + 0.995``, i.e. gamma lies in [0.995, 1.005].
    """

    def __init__(self, hidden=16):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, delta_loss, teacher_eval_loss, epoch_norm):
        x = torch.stack([delta_loss, teacher_eval_loss, epoch_norm], dim=-1)
        h = F.relu(self.fc1(x))
        gamma = torch.sigmoid(self.fc2(h)) * 0.01 + 0.995
        return gamma.squeeze()
