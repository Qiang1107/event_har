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
from scipy.spatial import KDTree  # 用于高效的近邻搜索
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def filter_noise_by_spatiotemporal_density(events, radius=0.05, min_neighbors=5):
    """
    基于时空密度的事件数据降噪
    参数:
        events: 事件数据，形状为 [N, 4]，对应 t, x, y, p
        radius: 搜索半径，用于寻找近邻点
        min_neighbors: 最小邻居数量阈值，少于此值的点被视为噪声
    返回:
        过滤后的事件数据
    """
    if len(events) == 0:
        return events
    
    # 提取时空坐标 (t,x,y)，将时间视为z坐标
    txy = events[:, 0:3].copy()  # 不包括极性p
    
    # 归一化坐标，使各维度具有相似的尺度
    for i in range(3):
        dim_range = txy[:, i].max() - txy[:, i].min()
        if dim_range > 0:
            txy[:, i] = (txy[:, i] - txy[:, i].min()) / dim_range
    
    # 构建KD树用于高效近邻搜索
    tree = KDTree(txy)
    
    # 对每个点计算半径内的邻居数量
    neighbors_count = np.zeros(len(events), dtype=int)
    for i in range(len(events)):
        print(f"Processing point {i+1}/{len(events)}", end='\r')
        neighbors = tree.query_ball_point(txy[i], radius)
        neighbors_count[i] = len(neighbors) - 1  # 减1排除自身
    
    # 根据邻居数量过滤噪声点
    valid_indices = neighbors_count >= min_neighbors
    filtered_events = events[valid_indices]
    
    print(f"噪声过滤: 从{len(events)}个点移除{len(events) - len(filtered_events)}个噪声点 ({100 * (len(events) - len(filtered_events)) / len(events):.2f}%)")
    
    return filtered_events


def filter_noise_by_spatiotemporal_clustering(events, eps=0.05, min_samples=5):
    """
    使用DBSCAN聚类算法进行降噪(适用于较大数据集)
    参数:
        events: 事件数据，形状为 [N, 4]
        eps: DBSCAN的邻域半径参数
        min_samples: 定义核心点的最小邻居数量
    返回:
        过滤后的事件数据
    """
    from sklearn.cluster import DBSCAN
    
    if len(events) == 0:
        return events
    
    # 提取时空坐标并归一化
    txy = events[:, 0:3].copy()
    for i in range(3):
        dim_range = txy[:, i].max() - txy[:, i].min()
        if dim_range > 0:
            txy[:, i] = (txy[:, i] - txy[:, i].min()) / dim_range
    
    # 应用DBSCAN聚类
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(txy)
    labels = db.labels_
    
    # 过滤掉噪声点（标签为-1的点）
    valid_indices = labels != -1
    filtered_events = events[valid_indices]
    
    print(f"DBSCAN降噪: 从{len(events)}个点移除{len(events) - len(filtered_events)}个噪声点 ({100 * (len(events) - len(filtered_events)) / len(events):.2f}%)")
    
    return filtered_events


class ECountSeqDataset(Dataset):
    """事件序列数据集,将事件数据转换为适合PointNet++的点云格式"""
    def __init__(self, data_root, window_size_event_count, step_size, label_map,
                 denoise=True, denoise_radius=0.05, min_neighbors=5):
        self.data_root = data_root
        self.window_size_event_count = window_size_event_count
        self.step_size = step_size
        self.denoise = denoise  # 是否执行降噪
        self.denoise_radius = denoise_radius  # 降噪搜索半径
        self.min_neighbors = min_neighbors  # 最小邻居数量阈值

        self.samples = []
        window_durations = []
        total_event_count = 0
        filtered_event_count = 0
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
                
                # **统计初始总事件数**
                total_event_count += len(events)

                # **降噪处理**
                if self.denoise:
                    original_count = len(events)
                    # events = filter_noise_by_spatiotemporal_density( # 使用时空密度过滤噪声点
                    #     events, 
                    #     radius=self.denoise_radius, 
                    #     min_neighbors=self.min_neighbors
                    # )
                    events = filter_noise_by_spatiotemporal_clustering( # 使用DBSCAN聚类算法过滤噪声点
                        events, 
                        eps=self.denoise_radius, 
                        min_samples=self.min_neighbors
                    )
                    
                    filtered_event_count += (original_count - len(events))
                
                # 检查降噪后是否还有足够的事件点
                if len(events) < self.window_size_event_count:
                    print(f"Warning: 文件 {file_path} 降噪后事件点数不足，跳过处理")
                    continue

                # **正确解析数据**
                t, x, y, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
                
                # **时间归一化**
                t = (t - np.min(t)) / (np.max(t) - np.min(t) + 1e-6)

                # **滑动窗口切片**
                num_slices = 0                
                for start in range(0, len(events) - self.window_size_event_count, step_size):
                    end = start + self.window_size_event_count
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
        if self.denoise:
            print(f"降噪移除的事件数: {filtered_event_count} ({100 * filtered_event_count / total_event_count:.2f}%)")
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
        data_root                   = ds_cfg['val_dir'],
        window_size_event_count     = ds_cfg['window_size_event_count'],
        step_size                   = ds_cfg['step_size'],
        label_map                   = ds_cfg['label_map'],
        denoise                     = True,        # 启用降噪 True False
        denoise_radius              = 0.05,        # 调整半径参数
        min_neighbors               = 5            # 最小邻居数量阈值
    )
    
    # 保存处理后的数据集到文件
    save_dir = "preprocessing_data"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "val_data_0628_8_ecount_2.pkl")
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
