
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import numpy as np
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as T


class MMLSv2ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, image):
        # sample 格式假设为 {'image': tensor[7, 128, 128], 'mask': tensor[1, 128, 128]}
        if random.random() > self.p:
            return image

        
        # 1. 随机生成变换因子
        # alpha 为对比度，beta 为亮度
        alpha = 1.0 + random.uniform(-self.contrast, self.contrast)
        beta = random.uniform(-self.brightness, self.brightness)

        # 2. 分波段处理
        # Band 1-3: Thermal Inertia, Slope, DEM (物理属性，建议轻微抖动或跳过)
        # 这里我们仅对物理波段进行 50% 强度的扰动，保持稳定性
        image[:3, :, :] = image[:3, :, :] * (1.0 + (alpha-1.0)*0.5) + (beta * 0.1)

        # Band 4-7: CTX Grayscale, RGB Viking (光学影像，可以大幅度抖动)
        image[3:, :, :] = image[3:, :, :] * alpha + beta

        # 3. 数值截断，防止溢出（假设数据已归一化到 [0, 1]）
        image = torch.clamp(image, 0, 1)
        
        return image

class MMLSv2RandomResizedCrop:
    def __init__(self, size=(128, 128), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5):
        """
        Args:
            size: 最终输出尺寸 (128, 128)
            scale: 裁剪面积比例
            ratio: 宽高比范围
            p: 触发概率
        """
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.p = p

    def __call__(self, image, mask):
        """
        image: Tensor, shape [C, H, W] (C=7 for MMLSv2)
        mask: Tensor, shape [H, W] OR [1, H, W]
        """
        # --- 1. 状态保存：记住原始 Mask 的维度 ---
        # 很多 Dataset 的 mask 是 [128, 128]，没有通道维
        mask_was_2d = (mask.ndim == 2)
        
        # --- 2. 维度标准化 (为了适配 torchvision API) ---
        # image 必须是 [C, H, W]，通常已经是了，防守性检查一下
        if image.ndim == 2:
            image = image.unsqueeze(0)
            
        # mask 必须暂时升维到 [1, H, W] 才能被 resized_crop 处理
        if mask_was_2d:
            mask = mask.unsqueeze(0)

        # --- 3. 概率判断：如果不触发，直接返回原格式 ---
        if random.random() > self.p:
            # 还原维度：如果原来是 2D，现在必须 squeeze 回去
            if mask_was_2d:
                mask = mask.squeeze(0)
            return image, mask

        # --- 4. 执行增强 (参数同步) ---
        # 获取随机参数 (基于 image 计算)
        i, j, h, w = T.RandomResizedCrop.get_params(
            image, scale=self.scale, ratio=self.ratio
        )

        # 对 Image 执行双线性插值 (Bilinear)
        image = TF.resized_crop(
            image, i, j, h, w, self.size, 
            interpolation=T.InterpolationMode.BILINEAR
        )

        # 对 Mask 执行最近邻插值 (Nearest) - 保护 0/1 标签
        mask = TF.resized_crop(
            mask, i, j, h, w, self.size, 
            interpolation=T.InterpolationMode.NEAREST
        )

        # --- 5. 维度还原 (关键步骤) ---
        # 如果进来时 mask 是 [H, W]，出去时必须变回 [H, W]
        if mask_was_2d:
            mask = mask.squeeze(0)

        return image, mask


class MMLSv2RandomResizedCropv2:
    def __init__(self, 
                 size=(128, 128), 
                 scale=(0.2, 2.0), 
                 ratio=(3./4., 4./3.), 
                 p=0.3,
                 num_attempts=10):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.p = p
        self.num_attempts = num_attempts
        
        self.log_ratio_min = math.log(self.ratio[0])
        self.log_ratio_max = math.log(self.ratio[1])

    def get_params(self, img_h, img_w):
        area = img_h * img_w

        for _ in range(self.num_attempts):
            # 替换 random.uniform 为 torch.empty(1).uniform_()
            rand_scale = torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
            target_area = rand_scale * area
            
            rand_ratio = torch.empty(1).uniform_(self.log_ratio_min, self.log_ratio_max).item()
            aspect_ratio = math.exp(rand_ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= img_w and 0 < h <= img_h:
                # 替换 random.randint，注意 torch.randint 上界是开区间，所以要 +1
                i = torch.randint(0, img_h - h + 1, (1,)).item()
                j = torch.randint(0, img_w - w + 1, (1,)).item()
                return i, j, h, w

        # Fallback 保持不变
        in_ratio = img_w / img_h
        if in_ratio < self.ratio[0]:
            w = img_w
            h = int(round(w / self.ratio[0]))
        elif in_ratio > self.ratio[1]:
            h = img_h
            w = int(round(h * self.ratio[1]))
        else:
            w = img_w
            h = img_h
        i = (img_h - h) // 2
        j = (img_w - w) // 2
        return i, j, h, w

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        # 替换 random.random() 为 torch.rand()
        if torch.rand(1).item() > self.p:
            return image, mask

        _, img_h, img_w = image.shape
        i, j, h, w = self.get_params(img_h, img_w)

        img_crop = image[:, i:i+h, j:j+w]
        mask_crop = mask[i:i+h, j:j+w].unsqueeze(0)

        img_resized = TF.resize(img_crop, self.size, interpolation=T.InterpolationMode.BILINEAR)
        mask_resized = TF.resize(mask_crop, self.size, interpolation=T.InterpolationMode.NEAREST)

        mask_resized = mask_resized.squeeze(0)

        return img_resized, mask_resized


class MMLSv2Mosaic:
    def __init__(self, size=128, p=0.5):
        self.size = size
        self.p = p

    def __call__(self, dataset, index):
        """
        dataset: 你的 PyTorch Dataset 对象
        index: 当前样本索引
        """
        if random.random() > self.p:
            return dataset.get_original_item(index) # 假设原始获取方法

        s = self.size
        # 1. 随机中心点 (在 1/4 到 3/4 范围内，避免某一张图被压得太小)
        xc, yc = [int(random.uniform(s // 4, 3 * s // 4)) for _ in range(2)]

        # 2. 随机抽取 4 个索引（包含当前的）
        indices = [index] + [random.randint(0, len(dataset) - 1) for _ in range(3)]
        
        # 初始化 7 波段输出图像和 Mask
        out_img = torch.zeros((7, s, s))
        out_mask = torch.zeros((1, s, s))

        for i, idx in enumerate(indices):
            sample = dataset.get_original_item(idx) # 获取 128x128 的数据
            img, mask = sample['image'], sample['mask']

            # 3. 计算拼接位置 (左上, 右上, 左下, 右下)
            if i == 0:  # top-left
                out_img[:, 0:yc, 0:xc] = img[:, s-yc:s, s-xc:s]
                out_mask[:, 0:yc, 0:xc] = mask[:, s-yc:s, s-xc:s]
            elif i == 1:  # top-right
                out_img[:, 0:yc, xc:s] = img[:, s-yc:s, 0:s-xc]
                out_mask[:, 0:yc, xc:s] = mask[:, s-yc:s, 0:s-xc]
            elif i == 2:  # bottom-left
                out_img[:, yc:s, 0:xc] = img[:, 0:s-yc, s-xc:s]
                out_mask[:, yc:s, 0:xc] = mask[:, 0:s-yc, s-xc:s]
            elif i == 3:  # bottom-right
                out_img[:, yc:s, xc:s] = img[:, 0:s-yc, 0:s-xc]
                out_mask[:, yc:s, xc:s] = mask[:, 0:s-yc, 0:s-xc]

        return {'image': out_img, 'mask': out_mask}

class MarsAugmentor:
    """
    Simultaneous Data Augmentation for 7-Channel Image and Mask.
    Capabilities:
    1. Random Horizontal/Vertical Flip
    2. Random 90-degree Rotation
    3. Random Resized Crop (Zoom In/Out) - Careful with resolution
    4. Gaussian Noise injection (to specific channels)
    """
    def __init__(self, prob=0.5, bands=7, channel_indices=None):
        self.prob = prob
        self.bands = bands
        # ``image`` may be a modality subset (for example, 4 optical or 3
        # physical channels).  Keep the original source-channel indices so
        # channel-specific noise remains valid after sub-selection.
        self.channel_indices = list(channel_indices) if channel_indices is not None else list(range(bands))
        # self.mmlsv2_color_jitter = MMLSv2ColorJitter()
        self.mmlsv2_random_resized_crop = MMLSv2RandomResizedCropv2()
        
    def __call__(self, image, mask):
        # Image: [C, H, W], Mask: [H, W]
        # Ensure tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
            
        # 1. Flip
        if torch.rand(1) < self.prob:
            image = torch.flip(image, [2]) # Vertical
            mask = torch.flip(mask, [1])
            
        if torch.rand(1) < self.prob:
            image = torch.flip(image, [1]) # Horizontal
            mask = torch.flip(mask, [0])

        # 2. Rotation 90/180/270
        if torch.rand(1) < self.prob:
            k = torch.randint(1, 4, (1,)).item()
            image = torch.rot90(image, k, [1, 2])
            mask = torch.rot90(mask, k, [0, 1])
            
        # 3. Random Gaussian Noise to DEM/Visuals (Channels 2, 3, 5, 6)
        # Only subtle noise to force robustness
        if torch.rand(1) < 0.2:
            noise = torch.randn_like(image) * 0.02
            # Apply mainly to DEM (2) and Visuals (5,6)
            noise_source_channels = {2, 5, 6}
            mask_noise = torch.tensor(
                [1.0 if channel in noise_source_channels else 0.0 for channel in self.channel_indices],
                dtype=image.dtype,
                device=image.device,
            ).view(-1, 1, 1)
            image = image + noise * mask_noise
        
        # 4. MMLSv2 Color Jitter
        # image = self.mmlsv2_color_jitter(image)
        image, mask = self.mmlsv2_random_resized_crop(image, mask)

        return image, mask
