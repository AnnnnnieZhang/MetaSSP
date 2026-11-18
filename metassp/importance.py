"""Gradient-based parameter importance estimation (paper Sec. III-C, Fig. 2)."""
import torch

from .augmentation import batch_weak_augmentation


def compute_importance_weights(teacher_model, pseudo_loader, device, num_batches=None):
    """Estimate how strongly each teacher parameter influences the predicted SDF.

    For up to ``num_batches`` unlabeled batches, the teacher predicts the SDF of
    the weakly augmented images, the proxy loss ``||sdf||_2^2`` (paper Eq. 1) is
    back-propagated, and the absolute gradients are averaged over the batches.

    Args:
        teacher_model: teacher network.
        pseudo_loader: DataLoader over unlabeled data yielding ``(indices, model_input)``.
        device: torch device.
        num_batches: number of batches to use; ``None`` uses the whole loader.

    Returns:
        dict {parameter name: importance tensor with the parameter's shape}.
    """
    teacher_model.eval()
    importance_weights = {
        name: torch.zeros_like(param, device=device)
        for name, param in teacher_model.named_parameters()
    }

    count = 0
    for pseudo_indices, pseudo_model_input in pseudo_loader:
        pseudo_model_input = {k: v.to(device).to(torch.float32) for k, v in pseudo_model_input.items()}
        if "image" in pseudo_model_input:
            pseudo_model_input["image"] = batch_weak_augmentation(pseudo_model_input["image"])

        teacher_outputs = teacher_model(pseudo_model_input, pseudo_indices)
        proxy_loss = teacher_outputs['sdf'].pow(2).sum()

        teacher_model.zero_grad()
        proxy_loss.backward()

        for name, param in teacher_model.named_parameters():
            if param.grad is not None:
                importance_weights[name] += param.grad.abs()

        count += 1
        if num_batches is not None and count >= num_batches:
            break

    teacher_model.zero_grad()
    for name in importance_weights:
        importance_weights[name] /= max(count, 1)

    return importance_weights
