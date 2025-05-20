import torchvision.transforms as transforms
import torch
import random
from PIL import Image, ImageFilter


class RandomGaussianBlur:
    """随机高斯模糊"""
    def __init__(self, p=0.7, radius_min=0.5, radius_max=3.0):  # 增强模糊效果
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius))
        return img


def add_color_shift(img, device):
    """
    添加颜色偏移效果。
    Args:
        img (Tensor): 输入图像张量 [C, H, W]。
        device (torch.device): 图像所在设备。
    Returns:
        Tensor: 添加颜色偏移后的图像。
    """
    shift_values = torch.tensor([
        0.4 + random.uniform(-0.1, 0.1),
        0.7 + random.uniform(-0.1, 0.1),
        1.1 + random.uniform(-0.1, 0.1)
    ], device=device)  # 确保 shift_values 在与输入图像相同的设备上
    img = img * shift_values.view(3, 1, 1)
    return img.clamp(0, 1)  # 限制值范围在 [0, 1]


def add_white_noise(img):
    """
    添加随机白噪声点。
    Args:
        img (Tensor): 输入图像张量 [C, H, W]。
    Returns:
        Tensor: 添加白噪声后的图像。
    """
    noise = torch.rand_like(img) * random.uniform(0.04, 0.1)  # 随机化噪声强度
    mask = torch.rand_like(img) < random.uniform(0.06, 0.1)  # 随机化噪声掩码密度
    img = img + (noise * mask)  # 将噪声添加到随机点
    return img.clamp(0, 1)


def weak_augmentation(img):
    """
    弱增强：适合隐式三维重建的轻微调整。
    Args:
        img (PIL.Image): 输入图像。
    Returns:
        Tensor: 增强后的图像。
    """
    # PIL 图像增强
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 增强光照和对比度变化
        transforms.RandomRotation(degrees=(-15, 15)),  # 小角度随机旋转
        # transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.5, 2.0))  # 增强模糊范围
    ])
    img = transform(img)  # 颜色和模糊增强

    # 将 PIL 图像转换为 Tensor
    img_tensor = transforms.ToTensor()(img)

    # 添加轻微噪声
    def add_noise(img_tensor):
        noise = torch.randn_like(img_tensor) * 0.05  # 增强噪声强度
        return img_tensor + noise.clamp(0, 1)  # 保证值在图像范围内

    img_tensor = add_noise(img_tensor)  # 加入随机噪声
    return img_tensor


def strong_augmentation(img):
    """
    强增强：大幅度调整。
    Args:
        img (PIL.Image): 输入图像。
    Returns:
        Tensor: 增强后的图像。
    """
    # PIL 图像增强：颜色调整和模糊
    transform_pil = transforms.Compose([
        transforms.ColorJitter(brightness=0.4 + random.uniform(-0.1, 0.1),  # 随机化亮度
                               contrast=0.4 + random.uniform(-0.1, 0.1),
                               saturation=0.4 + random.uniform(-0.1, 0.1),
                               hue=0.3 + random.uniform(-0.05, 0.05)),  # 随机化色调
        RandomGaussianBlur(p=0.5, radius_min=0.2 + random.uniform(-0.2, 0.2), 
                           radius_max=0.8 + random.uniform(-0.2, 0.2)),  # 随机化模糊半径
        transforms.RandomRotation(degrees=(-20, 20)),  # 小角度随机旋转
    ])
    img = transform_pil(img)  # 对 PIL.Image 增强

    # 转换为 Tensor
    img_tensor = transforms.ToTensor()(img)

    # 确保操作在输入数据所在的设备上
    device = img_tensor.device

    # 动态添加多个遮挡块
    for _ in range(random.randint(1, 4)):  # 随机生成 1 到 4 个遮挡块
        eraser = transforms.RandomErasing(
            scale=(0.005, 0.025),  # 缩小遮挡块尺寸
            ratio=(0.5, 1.0),
            p=1.0,  # 确保每次一定执行
            value='random'
        )
        img_tensor = eraser(img_tensor)

    # 添加颜色偏移和白噪声
    img_tensor = add_color_shift(img_tensor, device)  # 应用颜色偏移
    img_tensor = add_white_noise(img_tensor)  # 添加白噪声

    return img_tensor


def batch_weak_augmentation(imgs):
    """
    对批量图像应用弱增强。
    Args:
        imgs (Tensor): 输入批量图像 [B, 3, H, W]
    Returns:
        Tensor: 增强后的批量图像 [B, 3, H, W]
    """
    device = imgs.device  # 确保设备与输入图像一致
    augmented_imgs = [weak_augmentation(transforms.ToPILImage()(img.cpu())).to(device) for img in imgs]
    return torch.stack(augmented_imgs)  # 拼接回批量格式


def batch_strong_augmentation(imgs):
    """
    批量强增强。
    Args:
        imgs (Tensor): 批量图像 [B, 3, H, W]。
    Returns:
        Tensor: 增强后的批量图像 [B, 3, H, W]。
    """
    device = imgs.device  # 确保设备与输入图像一致
    augmented_imgs = [strong_augmentation(transforms.ToPILImage()(img.cpu())).to(device) for img in imgs]
    return torch.stack(augmented_imgs)  # 拼接回批量格式
