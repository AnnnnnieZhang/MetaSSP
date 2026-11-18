"""Weak / strong image augmentations for consistency regularisation (paper Sec. III-E)."""
import random

import torch
import torchvision.transforms as transforms
from PIL import ImageFilter

# Rotation range (degrees) of the weak / strong augmentation.
WEAK_ROTATION = 20
STRONG_ROTATION = 20


class RandomGaussianBlur:
    """Apply a Gaussian blur of random radius with probability ``p``."""

    def __init__(self, p=0.7, radius_min=0.5, radius_max=3.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius))
        return img


def add_color_shift(img):
    """Scale the RGB channels of a [3, H, W] tensor by random per-channel factors."""
    shift_values = torch.tensor([
        0.4 + random.uniform(-0.1, 0.1),
        0.7 + random.uniform(-0.1, 0.1),
        1.1 + random.uniform(-0.1, 0.1),
    ], device=img.device)
    return (img * shift_values.view(3, 1, 1)).clamp(0, 1)


def add_white_noise(img):
    """Add sparse white noise of random strength and density to a [C, H, W] tensor."""
    noise = torch.rand_like(img) * random.uniform(0.04, 0.1)
    mask = torch.rand_like(img) < random.uniform(0.06, 0.1)
    return (img + noise * mask).clamp(0, 1)


def weak_augmentation(img):
    """Weak augmentation (WA): mild colour jitter, small rotation, light Gaussian noise.

    Args:
        img (PIL.Image): input image.
    Returns:
        Tensor [C, H, W].
    """
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomRotation(degrees=(-WEAK_ROTATION, WEAK_ROTATION)),
    ])
    img_tensor = transforms.ToTensor()(transform(img))

    # NOTE: the noise (not the sum) is clamped, so it is non-negative. Kept as in
    # the experiments of the paper.
    noise = torch.randn_like(img_tensor) * 0.05
    return img_tensor + noise.clamp(0, 1)


def strong_augmentation(img):
    """Strong augmentation (SA): heavy colour shifts, blur, rotation, 1-4 erased
    patches, per-channel colour shift and white noise.

    Args:
        img (PIL.Image): input image.
    Returns:
        Tensor [C, H, W].
    """
    transform_pil = transforms.Compose([
        transforms.ColorJitter(brightness=0.4 + random.uniform(-0.1, 0.1),
                               contrast=0.4 + random.uniform(-0.1, 0.1),
                               saturation=0.4 + random.uniform(-0.1, 0.1),
                               hue=0.3 + random.uniform(-0.05, 0.05)),
        RandomGaussianBlur(p=0.5,
                           radius_min=0.2 + random.uniform(-0.2, 0.2),
                           radius_max=0.8 + random.uniform(-0.2, 0.2)),
        transforms.RandomRotation(degrees=(-STRONG_ROTATION, STRONG_ROTATION)),
    ])
    img_tensor = transforms.ToTensor()(transform_pil(img))

    for _ in range(random.randint(1, 4)):
        eraser = transforms.RandomErasing(scale=(0.005, 0.025), ratio=(0.5, 1.0), p=1.0, value='random')
        img_tensor = eraser(img_tensor)

    img_tensor = add_color_shift(img_tensor)
    img_tensor = add_white_noise(img_tensor)
    return img_tensor


def _batch_apply(imgs, fn):
    to_pil = transforms.ToPILImage()
    return torch.stack([fn(to_pil(img.cpu())).to(imgs.device) for img in imgs])


def batch_weak_augmentation(imgs):
    """Apply the weak augmentation to a batch of images [B, 3, H, W]."""
    return _batch_apply(imgs, weak_augmentation)


def batch_strong_augmentation(imgs):
    """Apply the strong augmentation to a batch of images [B, 3, H, W]."""
    return _batch_apply(imgs, strong_augmentation)
