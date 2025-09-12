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
from scipy.spatial import KDTree  # 用于高效的近邻搜索unified_timestamp

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


def voxel_grid_filter(events, voxel_size_txy):
    """
    使用网格体素法进行快速降噪，对原始(未归一化)数据进行处理
    参数:
        events: 事件数据 [N, 4]，对应 t, x, y, p
        voxel_size_t: 时间维度的体素大小
        voxel_size_x: x维度的体素大小
        voxel_size_y: y维度的体素大小
    返回:
        过滤后的事件数据
    """
    if len(events) == 0:
        return events
        
    # 提取原始坐标，不进行归一化
    t = events[:, 0].copy()  # 时间戳
    x = events[:, 1].copy()  # x坐标
    y = events[:, 2].copy()  # y坐标
    print(f"维度范围 t: {(np.max(t) - np.min(t)):.2f} µs, {(np.max(t) - np.min(t))/1000:.2f} ms")
    
    # 计算体素索引 - 直接使用原始数据
    voxel_idx_t = np.floor((t - np.min(t)) / voxel_size_txy[0]).astype(int)
    voxel_idx_x = np.floor(x / voxel_size_txy[1]).astype(int)
    voxel_idx_y = np.floor(y / voxel_size_txy[2]).astype(int)
    
    # 组合成体素索引
    voxel_indices = np.column_stack((voxel_idx_t, voxel_idx_x, voxel_idx_y))
    print(f"体素voxel_indices shape: {voxel_indices.shape}")
    
    # 创建体素字典
    voxel_dict = {}
    for i in range(len(events)):
        voxel_key = tuple(voxel_indices[i])
        if voxel_key in voxel_dict:
            voxel_dict[voxel_key].append(i)
        else:
            voxel_dict[voxel_key] = [i]
    
    # 计算每个体素中点的数量
    voxel_counts = {k: len(v) for k, v in voxel_dict.items()}
    counts = np.array(list(voxel_counts.values()))
    print(f"体素字典大小: {len(voxel_dict)}")
    print(f"平均每体素点数: {np.mean(counts):.2f}")
    print(f"体素点数分布: {np.min(counts)} - {np.max(counts)}")
    # 只打印前几个体素的点数，避免输出太多
    print(f"counts体素点数前10个: {counts[:10]}")
    
    # 设定阈值，低于此值的体素被视为噪声
    # threshold = max(2, int(np.mean(counts) * 0.1))
    threshold = 2
    
    # 收集有效点的索引
    valid_indices = []
    for voxel_key, indices in voxel_dict.items():
        if len(indices) >= threshold:
            # 保留所有有效体素中的点
            valid_indices.extend(indices)
    
    # 返回过滤后的事件
    filtered_events = events[valid_indices]
    print(f"网格体素降噪: 从{len(events)}个点保留{len(filtered_events)}个点 ({100 * len(filtered_events) / len(events):.2f}%)")
    
    return filtered_events


def histogram_filter(events, bin_size=10, density_threshold=0.1):
    """
    使用直方图统计空间密度进行降噪
    
    参数:
        events: 事件数据 [N, 4]
        bin_size: 直方图分箱大小
        density_threshold: 密度阈值百分比
        
    返回:
        过滤后的事件数据
    """
    if len(events) == 0:
        return events
    
    # 提取坐标
    t, x, y = events[:, 0], events[:, 1], events[:, 2]
    
    # 计算x-y平面的2D直方图
    x_bins = np.linspace(np.min(x), np.max(x), int((np.max(x) - np.min(x)) / bin_size) + 1)
    y_bins = np.linspace(np.min(y), np.max(y), int((np.max(y) - np.min(y)) / bin_size) + 1)
    
    hist, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
    
    # 设定密度阈值
    threshold = np.max(hist) * density_threshold
    
    # 找出密度高于阈值的区域
    dense_regions = hist > threshold
    
    # 对每个点，检查它是否在高密度区域
    valid_indices = []
    for i in range(len(events)):
        x_idx = np.digitize(x[i], x_bins) - 1
        y_idx = np.digitize(y[i], y_bins) - 1
        
        # 防止越界
        if x_idx >= 0 and x_idx < len(x_bins)-1 and y_idx >= 0 and y_idx < len(y_bins)-1:
            if dense_regions[x_idx, y_idx]:
                valid_indices.append(i)
    
    # 返回过滤后的事件
    filtered_events = events[valid_indices]
    print(f"直方图降噪: 从{len(events)}个点保留{len(filtered_events)}个点 ({100 * len(filtered_events) / len(events):.2f}%)")
    
    return filtered_events


def random_sampling_filter(events, sample_rate=0.1, std_dev_multiplier=2.0):
    """
    先随机采样，再用统计方法过滤异常值
    
    参数:
        events: 事件数据 [N, 4]
        sample_rate: 随机采样比例
        std_dev_multiplier: 标准差倍数，用于异常值检测
        
    返回:
        过滤后的事件数据
    """
    if len(events) == 0:
        return events
    
    # 随机采样以加速处理
    sample_size = max(1000, int(len(events) * sample_rate))
    sample_indices = np.random.choice(len(events), sample_size, replace=False)
    sample = events[sample_indices]
    
    # 计算采样点的统计特性
    txy_mean = np.mean(sample[:, 0:3], axis=0)
    txy_std = np.std(sample[:, 0:3], axis=0)
    
    # 设定异常值边界
    lower_bound = txy_mean - std_dev_multiplier * txy_std
    upper_bound = txy_mean + std_dev_multiplier * txy_std
    
    # 过滤所有事件点中的异常值
    valid_indices = np.all((events[:, 0:3] >= lower_bound) & (events[:, 0:3] <= upper_bound), axis=1)
    filtered_events = events[valid_indices]
    
    print(f"随机采样统计降噪: 从{len(events)}个点保留{len(filtered_events)}个点 ({100 * len(filtered_events) / len(events):.2f}%)")
    
    return filtered_events


def temporal_consistency_filter(events, time_bins=100, spatial_bins=16, density_threshold=0.2):
    """
    利用时间一致性进行高效降噪
    
    参数:
        events: 事件数据 [N, 4]
        time_bins: 时间维度的分箱数
        spatial_bins: 空间维度的分箱数
        density_threshold: 密度阈值百分比
        
    返回:
        过滤后的事件数据
    """
    if len(events) == 0:
        return events
    
    # 提取坐标
    t, x, y = events[:, 0], events[:, 1], events[:, 2]
    
    # 将时间轴分成若干段
    t_min, t_max = np.min(t), np.max(t)
    t_edges = np.linspace(t_min, t_max, time_bins + 1)
    
    # 对每个时间段进行处理
    valid_indices = []
    
    for i in range(time_bins):
        t_start, t_end = t_edges[i], t_edges[i+1]
        # 获取当前时间段的点索引
        time_slice_mask = (t >= t_start) & (t < t_end)
        slice_indices = np.where(time_slice_mask)[0]
        
        if len(slice_indices) > 0:
            # 对当前时间段的空间坐标创建2D直方图
            slice_x = x[slice_indices]
            slice_y = y[slice_indices]
            
            hist, _, _ = np.histogram2d(
                slice_x, slice_y, 
                bins=[spatial_bins, spatial_bins],
                range=[[np.min(x), np.max(x)], [np.min(y), np.max(y)]]
            )
            
            # 计算密度阈值
            threshold = np.max(hist) * density_threshold
            if threshold > 0:
                # 对每个点检查它所在的格子是否密度足够高
                x_bins = np.linspace(np.min(x), np.max(x), spatial_bins + 1)
                y_bins = np.linspace(np.min(y), np.max(y), spatial_bins + 1)
                
                for idx in slice_indices:
                    x_idx = np.digitize(x[idx], x_bins) - 1
                    y_idx = np.digitize(y[idx], y_bins) - 1
                    
                    # 防止越界
                    if (0 <= x_idx < spatial_bins and 0 <= y_idx < spatial_bins and 
                        hist[x_idx, y_idx] >= threshold):
                        valid_indices.append(idx)
    
    # 返回过滤后的事件
    filtered_events = events[np.array(valid_indices)]
    print(f"时间一致性降噪: 从{len(events)}个点保留{len(filtered_events)}个点 ({100 * len(filtered_events) / len(events):.2f}%)")
    
    return filtered_events


class ECountSeqDatasetVote(Dataset):
    """事件序列数据集,将事件数据转换为适合PointNet++的点云格式"""
    def __init__(self, data_root, window_size_event_count, step_size, label_map,
                 roi, denoise, denoise_method, denoise_radius, 
                 voxel_size_txy, min_neighbors, denoise_threshold):
        self.data_root = data_root
        self.window_size_event_count = window_size_event_count
        self.step_size = step_size
        self.denoise = denoise  # 是否执行降噪
        self.roi = roi  # 是否使用感兴趣区域
        self.denoise_method = denoise_method  # 降噪方法: 'density', 'dbscan', 'voxel', 'histogram', 'random', 'temporal'
        self.denoise_radius = denoise_radius  # 降噪搜索半径
        self.voxel_size_txy = voxel_size_txy  # 体素大小 [t, x, y] 单位: 微秒 和 像素
        self.min_neighbors = min_neighbors  # 最小邻居数量阈值
        self.denoise_threshold = denoise_threshold  # 密度阈值参数

        self.samples = []
        window_durations = []
        total_event_count = 0
        filtered_event_count = 0
        total_sample_count = 0

        if self.denoise:
            print(f"启用降噪处理，方法: {self.denoise_method}")
        else:
            print("未启用降噪处理")

        if self.roi:
            x_offset, y_offset = 60, 25
            print(f"启用空间裁剪处理, x: ({x_offset}, {346-x_offset}) y: ({y_offset}, {260-y_offset})")
        else:
            print("未启用空间裁剪处理")

        for class_name, class_idx in label_map.items():
            class_dir = os.path.join(self.data_root, class_name)
            for file in os.listdir(class_dir):
                if not file.endswith('.npy'):
                    continue
                file_path = os.path.join(class_dir, file)
                events = np.load(file_path, allow_pickle=True) # (N, 4)
                if events.size == 0 or events.shape[0] == 0:
                    print(f"Warning 1: 文件 {file_path} 包含空的事件数组，跳过处理")
                    continue
                if len(events.shape) == 1:
                    print(f"Warning 2: {file} 形状为1D !!!")
                    events = np.array([list(event) for event in events])
                    print(f"转换 {file} 为 2D,形状 {events.shape}")
                if events.shape[1] != 4:
                    print(f"Warning 3: {file} 格式不对({events.shape}),跳过")
                    continue
                
                # **统计初始总事件数**
                total_event_count += len(events)
                print(f"处理文件: {file_path}，事件数: {len(events)}")

                # **降噪处理**
                if self.denoise:
                    original_count = len(events)
                    if self.denoise_method == 'density':
                        # 时空密度降噪方法（慢）
                        events = filter_noise_by_spatiotemporal_density(
                            events, 
                            radius=self.denoise_radius, 
                            min_neighbors=self.min_neighbors
                        )
                    elif self.denoise_method == 'dbscan':
                        # DBSCAN降噪方法（更慢）
                        events = filter_noise_by_spatiotemporal_clustering(
                            events, 
                            eps=self.denoise_radius, 
                            min_samples=self.min_neighbors
                        )
                    elif self.denoise_method == 'voxel':
                        # 网格体素降噪方法（极快）
                        events = voxel_grid_filter(
                            events, 
                            voxel_size_txy=self.voxel_size_txy
                        )
                    elif self.denoise_method == 'histogram':
                        # 直方图降噪方法（很快）
                        events = histogram_filter(
                            events, 
                            bin_size=10, 
                            density_threshold=self.denoise_threshold
                        )
                    elif self.denoise_method == 'random':
                        # 随机采样+统计降噪（最快）
                        events = random_sampling_filter(
                            events, 
                            sample_rate=0.1, 
                            std_dev_multiplier=2.0
                        )
                    elif self.denoise_method == 'temporal':
                        # 时间一致性降噪（高效且精确）
                        events = temporal_consistency_filter(
                            events,
                            time_bins=50, 
                            spatial_bins=16, 
                            density_threshold=self.denoise_threshold
                    )
                    
                    filtered_event_count += (original_count - len(events))

                # 检查文件是否有足够的事件点
                if len(events) < self.window_size_event_count:
                    print(f"Warning 4: 文件 {file_path} 中事件点数不足，跳过处理")
                    continue

                # **空间裁剪，取xy中心的事件数据**
                if self.roi:
                    # 设定感兴趣区域的中心位置和大小
                    # 346x260是事件图像的分辨率
                    # 保留中心区域的事件点
                    valid_mask = (events[:, 1] >= x_offset) & (events[:, 1] <= (346-x_offset)) & (events[:, 2] >= y_offset) & (events[:, 2] <= (260-y_offset))
                    if not np.any(valid_mask): 
                        print(f"Warning 5: 文件 {file_path} 中xy中心位置没有有效的事件点,跳过处理")
                        continue
                    events = events[valid_mask]
                    print(f"文件 {file_path} 经过空间裁剪后剩余事件数: {len(events)}")

                # 解析事件数据
                t, x, y, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
                
                # **时间归一化**
                t = (t - np.min(t)) / (np.max(t) - np.min(t) + 1e-6)

                # **滑动窗口切片**
                num_slices = 0                
                for start in range(0, len(events) - self.window_size_event_count, step_size):
                    end = start + self.window_size_event_count
                    slice_events = np.stack((t[start:end], x[start:end], y[start:end], p[start:end]), axis=1)
                    num_slices += 1

                    # 计算窗口持续时间和时间戳信息
                    start_timestamp = events[start, 0]  # 原始时间戳（微秒）
                    end_timestamp = events[end-1, 0]    # 原始时间戳（微秒）
                    center_timestamp = (start_timestamp + end_timestamp) / 2
                    window_duration = end_timestamp - start_timestamp
                    window_durations.append(window_duration)
                    
                    self.samples.append((slice_events, class_idx, start_timestamp, end_timestamp, 
                                         center_timestamp, file_path, window_duration, class_name))

                total_sample_count += num_slices
                # print(f"文件: {file}，原始事件数: {len(events)}，生成样本数: {num_slices}")
            
        print(f"总事件数: {total_event_count}")
        if self.denoise:
            print(f"降噪移除的事件数: {filtered_event_count} ({100 * filtered_event_count / total_event_count:.2f}%)")
        print(f"总生成样本数: {total_sample_count}")
        if len(window_durations) > 0:
            print(f"窗口持续时间范围: {np.min(window_durations):.2f} µs - {np.max(window_durations):.2f} µs")
            avg_duration = np.mean(window_durations)
            print(f"平均窗口持续时间: {avg_duration:.2f} µs\n")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取单个点云样本"""
        events, label, start_ts, end_ts, center_ts, file_path, window_duration, class_name = self.samples[idx]
        return events, label, start_ts, end_ts, center_ts, file_path, window_duration, class_name


if __name__ == '__main__':
    config_path='configs/har_train_config.yaml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 测试数据集
    ds_cfg = cfg['dataset']
    data_ds = ECountSeqDatasetVote(
        data_root                   = ds_cfg['test_dir'],
        window_size_event_count     = ds_cfg['window_size_event_count'],
        step_size                   = ds_cfg['step_size'],
        label_map                   = ds_cfg['label_map'],
        roi                         = ds_cfg['roi'],           # 是否使用感兴趣区域
        denoise                     = ds_cfg['denoise'],        # 启用降噪 True False
        denoise_method              = ds_cfg['denoise_method'],    # 降噪方法: 'density', 'dbscan', 'voxel', 'histogram', 'random', 'temporal'
        denoise_radius              = ds_cfg['denoise_radius'],        # 调整半径参数 density: 0.05, dbscan: 0.05
        voxel_size_txy              = ds_cfg['voxel_size_txy'],  # 体素大小 [t, x, y] 单位: 微秒 和 像素
        min_neighbors               = ds_cfg['min_neighbors'],           # 最小邻居数量阈值
        denoise_threshold           = ds_cfg['denoise_threshold']          # 密度阈值参数
    )
    
    # 保存处理后的数据集到文件
    save_dir = "preprocessing_data"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "test_data_0628_8_ecount_3_vote.pkl") 
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
        sample_data, sample_label, sample_start_ts, sample_end_ts, sample_center_ts, sample_file_path, sample_window_duration, sample_class_name = loaded_ds[sample_idx]
        print(f"\n随机样本 #{sample_idx}:")
        print(f"  - 形状: {sample_data.shape}")
        print(f"  - 标签: {sample_label}")
        print(f"  - 点云范围: t[{sample_data[:, 0].min():.2f}, {sample_data[:, 0].max():.2f}], "
              f"x[{sample_data[:, 1].min():.2f}, {sample_data[:, 1].max():.2f}], "
              f"y[{sample_data[:, 2].min():.2f}, {sample_data[:, 2].max():.2f}]")
        print(f"  - 时间戳: start {sample_start_ts}, end {sample_end_ts}, center {sample_center_ts}")
        print(f"  - 窗口持续时间: {sample_window_duration} µs")
        print(f"  - 文件路径: {sample_file_path}")
        print(f"  - 类别名称: {sample_class_name}")
        
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"\n数据集文件大小: {file_size_mb:.2f} MB")
    except Exception as e:
        print(f"加载数据集时出错: {e}")
