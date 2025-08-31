import os
import sys
import yaml
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.extensions import resize_and_normalize


class RGBSequenceDataset(Dataset):
    def __init__(self, data_root, window_size, stride, enable_transform, label_map, transform=None):
        """
        data_root: 根目录，下面按类别子文件夹存 npy
        window_size: 每个样本固定帧数
        stride: 滑动窗口步长
        enable_transform: 可选的数据增强/预处理方法，对每个帧进行处理
        """
        self.window_size = window_size
        self.stride = stride
        self.enable_transform = enable_transform
        self.transform = transform

        # --- 预扫描 data_root，生成 (npy_path, start_idx, label) 列表 ---
        self.samples = []
        for cls_name, cls_idx in label_map.items():
            # print(f"Processing class: {cls_name} (index: {cls_idx})")
            cls_dir = os.path.join(data_root, cls_name)
            for fname in os.listdir(cls_dir):
                if not fname.endswith('.npy'):
                    continue
                full = os.path.join(cls_dir, fname)
                arr = np.load(full, mmap_mode='r')
                N = arr.shape[0] # 每个 npy 文件的帧数
                # 对每个np文件，用滑窗切出定长片段
                for start in range(0, N - window_size + 1, stride):
                    self.samples.append((full, start, cls_idx))

        print(f"加载了 {len(self.samples)} 个样本，共 {len(label_map)} 个类别")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 1. 拆出这个样本对应的 npy 文件、起始帧、标签
        npy_path, start, label = self.samples[idx]
        arr = np.load(npy_path).astype(np.float32)  # (B, H, W, C)
        # print("arr.shape", arr.shape, "npy_path", npy_path, "start", start, "label", label)

        # 2. 截取 [start : start+window_size] 帧
        clip = arr[start : start + self.window_size]  # (window_size, H, W, C)
        # print("clip.shape", clip.shape)

        # 3. RGB归一化
        clip[..., :3] /= 255.0

        # 4. to Tensor & permute -> (T,C,H,W)
        clip = torch.from_numpy(clip).permute(0,3,1,2)
        # print("clip.shape after permute", clip.shape)

        # 5. 对每一帧做转换
        if self.enable_transform:
            frames = []
            for t in range(clip.size(0)):
                img = self.transform(clip[t])
                frames.append(img.squeeze(0))
            clip = torch.stack(frames, dim=0)
            # print("clip.shape after transform", clip.shape)
            
        return clip, label
    

# python -m datasets.rgb_sequence_dataset
if __name__ == '__main__':
    config_path='configs/har_train_config.yaml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 数据增强
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        # transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 测试数据集
    ds = RGBSequenceDataset(
        data_root          = cfg['dataset']['train_dir'],
        window_size        = cfg['dataset']['window_size'],
        stride             = cfg['dataset']['stride'],
        enable_transform   = cfg['dataset']['enable_transform'],
        label_map          = cfg['dataset']['label_map'],
        transform          = transform_train # None # transform_train
    )
    print("len(ds)", len(ds))
    clip, label = ds[96]
    print("ds[96]: clip.shape", clip.shape, "label", label)
