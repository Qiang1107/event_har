import os
import sys
import yaml
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import pickle
import time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class ECountSeqDataset(Dataset):
    """事件序列数据集,将事件数据转换为适合PointNet++的点云格式"""
    def __init__(self, data_root, window_size, step_size, label_map):
        self.data_root = data_root
        self.window_size = window_size
        self.step_size = step_size
        self.samples = []
        window_durations = []
        total_event_count = 0
        total_sample_count = 0
        for class_name, class_idx in label_map.items():
            class_dir = os.path.join(self.data_root, class_name)
            for file in os.listdir(class_dir):
                if not file.endswith('.npy'):
                    continue
                file_path = os.path.join(class_dir, file)
                events = np.load(file_path, allow_pickle=True) # (N, 4)
                if events.size == 0 or events.shape[0] == 0:
                    print(f"Warning: 文件 {file_path} 包含空的事件数组，跳过处理")
                    continue
                if len(events.shape) == 1:
                    print(f"Warning: {file} 形状为1D !!!")
                    events = np.array([list(event) for event in events])
                    print(f"转换 {file} 为 2D,形状 {events.shape}")
                if events.shape[1] != 4:
                    print(f"Warning: {file} 格式不对({events.shape}),跳过")
                    continue

                # **正确解析数据**
                t, x, y, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
                total_event_count += len(events)

                # **时间归一化**
                t = (t - np.min(t)) / (np.max(t) - np.min(t) + 1e-6)

                # **滑动窗口切片**
                num_slices = 0                
                for start in range(0, len(events) - window_size, step_size):
                    end = start + window_size
                    slice_events = np.stack((t[start:end], x[start:end], y[start:end], p[start:end]), axis=1)
                    num_slices += 1

                    # 计算窗口持续时间
                    window_duration = events[end-1, 0] - events[start, 0]
                    window_durations.append(window_duration)
                    # print(f"Window duration: {window_duration:.2f} µs")

                    # self.samples.append((file_path, slice_events, class_idx))
                    self.samples.append((slice_events, class_idx))

                total_sample_count += num_slices
                # print(f"文件: {file}，原始事件数: {len(events)}，生成样本数: {num_slices}")
            
        print(f"总事件数: {total_event_count}")
        print(f"总生成样本数: {total_sample_count}")
        if len(window_durations) > 0:
            print(f"窗口持续时间范围: {np.min(window_durations):.2f} µs - {np.max(window_durations):.2f} µs")
            avg_duration = np.mean(window_durations)
            print(f"平均窗口持续时间: {avg_duration:.2f} µs")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取单个点云样本"""
        # file_path, events, label = self.samples[idx]
        events, label = self.samples[idx]
        return events, label


if __name__ == '__main__':
    config_path='configs/har_train_config.yaml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 测试数据集
    ds_cfg = cfg['dataset']
    data_ds = ECountSeqDataset(
        data_root          = ds_cfg['val_dir'],
        window_size        = ds_cfg['window_size_event_count'],
        step_size          = ds_cfg['step_size'],
        label_map          = ds_cfg['label_map']
    )
    
    # 保存处理后的数据集到文件
    save_dir = "preprocessing_data"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "val_data_0628_8_ecount_1.pkl")
    with open(save_path, 'wb') as f:
        print(f"正在保存数据集到 {save_path} ...")
        pickle.dump(data_ds, f)

    # 加载并检查保存的数据集
    try:
        with open(save_path, 'rb') as f:
            print(f"正在加载数据集: {save_path} ...")
            loaded_ds = pickle.load(f)
        print(f"成功加载数据集，包含 {len(loaded_ds)} 个样本")
        
        # 随机检查一个样本 shape为 [N, 4]， N为点云点数
        sample_idx = random.randint(0, len(loaded_ds)-1)
        sample_data, sample_label = loaded_ds[sample_idx]
        print(f"\n随机样本 #{sample_idx}:")
        print(f"  - 形状: {sample_data.shape}")
        print(f"  - 标签: {sample_label}")
        print(f"  - 点云范围: t[{sample_data[:, 0].min():.2f}, {sample_data[:, 0].max():.2f}], "
              f"x[{sample_data[:, 1].min():.2f}, {sample_data[:, 1].max():.2f}], "
              f"y[{sample_data[:, 2].min():.2f}, {sample_data[:, 2].max():.2f}]")
        
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"\n数据集文件大小: {file_size_mb:.2f} MB")
    except Exception as e:
        print(f"加载数据集时出错: {e}")
