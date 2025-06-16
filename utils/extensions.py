import math
import torch


def resize_and_normalize(img):
    import torchvision.transforms as T
    target_size = (192, 256) # (H, W)
    resize_transform = T.Compose([
        T.ToPILImage(),
        T.Resize(target_size),
        T.ToTensor()
    ])

    img = resize_transform(img)
    # print("Function [transform]: img.shape", img.shape)
    return img.squeeze(0)


def mosaic_frames(x: torch.Tensor) -> torch.Tensor:
    # x: [B, T, C, H, W]
    B, T, C, H, W = x.shape
    # 计算最近的上界整数平方尺寸 k
    k = math.ceil(math.sqrt(T))
    total_patches = k * k
    # 如有需要，补充零帧至 T 达到 k^2
    if T < total_patches:
        pad_frames = total_patches - T
        pad_shape = (B, pad_frames, C, H, W)
        pad_tensor = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        x = torch.cat([x, pad_tensor], dim=1)  # 在时间维度拼接零帧
    # 重塑并permute维度，将 k×k 个帧拼成大图
    x = x.view(B, k, k, C, H, W)                       # [B, k, k, C, H, W]
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()       # [B, C, k, H, k, W]
    x = x.view(B, C, k * H, k * W)                     # [B, C, H*k, W*k]
    return x
