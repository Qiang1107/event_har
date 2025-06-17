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


class ESequenceDataset(Dataset):
    """事件序列数据集,将事件数据转换为适合PointNet++的点云格式"""
    def __init__(self, data_root, window_size_us, max_points, 
                 time_dimension, enable_augment, stride_us, label_map):
        """
        参数:
            data_root: 数据根目录，包含各个动作类别的子文件夹
            window_size_us: 时间窗口大小，单位为微秒（用于分割长序列）
            stride_us: 滑动窗口步长,单位为微秒,若为None则默认为window_size_us的一半
            max_points: 每个样本的最大点数，超出则随机采样
            time_dimension: 是否将时间戳作为点云的第三维
            enable_augment: 是否进行数据增强
        """
        # print(f"[DEBUG] Checking data root: {data_root}")
        # print(f"[DEBUG] window_size_us: {window_size_us}, max_points: {max_points}")
        # print(f"[DEBUG] time_dimension: {time_dimension}, enable_augment: {enable_augment}, stride_us: {stride_us}")
        
        self.data_root = data_root
        self.window_size_us = window_size_us
        self.stride_us = stride_us if stride_us is not None else window_size_us // 2
        self.max_points = max_points
        self.time_dimension = time_dimension
        self.enable_augment = enable_augment

        # 处理事件数据的一些参数设置
        self.t_squash_factor = 10000  # 每10毫秒内的时间戳表示成一个时间戳
        self.target_width, self.target_height = 128, 96  # 目标空间坐标大小
        self.min_events_per_window = 4000  # 一个时间窗口中最小事件数阈值
        self.padpoints_noise_scale = 0.02  # 填充点的噪声尺度
        
        # 收集所有文件路径和初始标签
        initial_samples = []
        for cls_name, idx in label_map.items():
            class_dir = os.path.join(data_root, cls_name)
            files = glob.glob(os.path.join(class_dir, "*.npy"))
            for file_path in files:
                initial_samples.append((file_path, idx))
            print(f"[DEBUG] Found {len(files)} files in class {cls_name}")
        
        print(f"找到了来自{len(label_map)}个类别的{len(initial_samples)}个原始文件")
        
        # 应用滑动窗口处理，生成最终样本列表
        self.samples = []
        total_windows = 0
        start_time1 = time.time()
        for file_path, label in initial_samples:
            events = np.load(file_path)
            # print(f"[DEBUG] Loaded events shape: {events.shape}, label: {label}, time range: [{events[:, 0].min()}, {events[:, 0].max()}] us")
            # 检查事件数组是否为空
            if events.size == 0 or events.shape[0] == 0:
                print(f"警告: 文件 {file_path} 包含空的事件数组，跳过处理")
                continue

            # 时间戳转换成微秒
            if events[:, 0].dtype == np.int64:  # Check if timestamps are likely in Unix format
                min_timestamp = events[:, 0].min()
                events[:, 0] = events[:, 0] - min_timestamp
                
            # 时间戳等比例压缩 Group events in 10ms windows and assigning average timestamp
            time_bins = (events[:, 0] // self.t_squash_factor).astype(np.int64)
            unique_bins = np.unique(time_bins)
            for bin_idx in unique_bins:
                bin_mask = time_bins == bin_idx
                if np.sum(bin_mask) > 0:
                    bin_avg_time = np.mean(events[bin_mask, 0]).astype(np.int64)
                    events[bin_mask, 0] = bin_avg_time
            # print(f"[DEBUG] After time squashing, unique timestamps: {len(np.unique(events[:, 0]))}/{len(events)}")
            # 修改事件维度：将4维事件转换为5维，第4和5维分别表示p=0和p=1的计数
            new_events = np.zeros((len(events), 5), dtype=events.dtype)
            new_events[:, :3] = events[:, :3]
            new_events[:, 3] = (events[:, 3] == 1).astype(np.int64)  # p=1的个数
            new_events[:, 4] = (events[:, 3] == 0).astype(np.int64)  # p=0的个数
            events = new_events
            # 空间坐标xy缩放到指定大小
            original_width, original_height = 346, 260  # 原始宽度和高度为346和260
            width_scale = self.target_width / original_width
            height_scale = self.target_height / original_height
            events[:, 1] = np.clip(events[:, 1] * width_scale, 0, self.target_width - 1)
            events[:, 2] = np.clip(events[:, 2] * height_scale, 0, self.target_height - 1)
            # 合并时间戳和空间坐标相同的事件，累加极性计数
            merged_events_dict = {}
            for event in events:
                key = (event[0], event[1], event[2])
                if key in merged_events_dict:
                    merged_events_dict[key][3] += event[3]  # p=1的计数
                    merged_events_dict[key][4] += event[4]  # p=0的计数
                else:
                    merged_events_dict[key] = event.copy()
            events = np.array(list(merged_events_dict.values()))
            # print(f"[DEBUG] After merging: {len(events)} unique events)")
            
            # 【可选】打印前100个事件
            # num_events_to_print = min(1000, len(events))
            # print(f"[DEBUG] First {num_events_to_print} events:")
            # for i in range(num_events_to_print):
            #     print(f"  Event {i}: t={events[i, 0]:.1f}, x={events[i, 1]}, y={events[i, 2]}, p1={events[i, 3]}, p0={events[i, 4]}")
            # # 【可选】打印20个事件：第一个为最小时间戳，最后一个为最大时间戳，中间18个随机选取并按时间排序
            # min_time_idx = np.argmin(events[:, 0])
            # max_time_idx = np.argmax(events[:, 0])
            # remaining_indices = np.array([i for i in range(len(events)) if i != min_time_idx and i != max_time_idx])
            # if len(remaining_indices) > 18:
            #     random_indices = np.random.choice(remaining_indices, 18, replace=False)
            # else:
            #     random_indices = remaining_indices
            # sample_indices = np.concatenate([[min_time_idx], random_indices, [max_time_idx]])
            # sample_events = events[sample_indices]
            # sorted_indices = np.argsort(sample_events[:, 0])
            # sample_events = sample_events[sorted_indices]
            # print(f"[DEBUG] show 20 events [after timestamps processing]: {sample_events}")
            
            # 应用滑动窗口生成多个子序列
            windows = self.sliding_window_events(events)
            # print(f"[DEBUG] Generated {len(windows)} windows from file {file_path}")
            
            # 将每个窗口添加为单独的样本 以便在__getitem__中直接使用
            for window_idx, window_events in enumerate(windows):
                point_cloud = self.events_to_pointcloud(window_events) # 转换为PointNet++兼容的点云格式
                point_cloud_tensor = torch.from_numpy(point_cloud).float()
                self.samples.append((file_path, label, window_idx, point_cloud_tensor))
            
            total_windows += len(windows)

        end_time1 = time.time()
        print(f"[DEBUG] Processed {len(initial_samples)} files in {end_time1 - start_time1:.2f} seconds")
        print(f"使用滑动窗口处理后,共生成{total_windows}个训练样本")
        
    def events_to_pointcloud(self, events):
        """将事件数据转换为点云格式,适合PointNet++处理"""

        # 如果事件数据太多,使用体素网格法进行降采样
        if len(events) > self.max_points:
            if len(events) < (self.max_points * 1.2):
                indices = np.random.choice(len(events), self.max_points, replace=False)
                events = events[indices]
            else:
                print(f"[DEBUG] Reducing events from {len(events)} to max_points={self.max_points} using voxel grid")
                if len(events) < (self.max_points * 1.5):
                    voxel_size_xy = 1.4
                elif len(events) < (self.max_points * 2.0):
                    voxel_size_xy = 1.7
                elif len(events) < (self.max_points * 2.5):
                    voxel_size_xy = 1.9
                elif len(events) < (self.max_points * 3.0):
                    voxel_size_xy = 2.1
                else:
                    voxel_size_xy = 2.2
                voxel_size_t = self.t_squash_factor * voxel_size_xy  # 时间体素大小 (微秒)
                spatial_temporal_hash = {}
                reduced_events = []
                for event in events:
                    voxel_key = (
                        int(event[1] // voxel_size_xy),  # x方向
                        int(event[2] // voxel_size_xy),  # y方向
                        int(event[0] // voxel_size_t)    # t方向
                    )
                    if voxel_key not in spatial_temporal_hash:
                        spatial_temporal_hash[voxel_key] = event
                        reduced_events.append(event)
                # 如果体素网格后仍超过最大点数，继续使用随机采样
                reduced_events = np.array(reduced_events)
                if len(reduced_events) > self.max_points:
                    print(f"[DEBUG] After voxel grid, still {len(reduced_events)} events, applying random sampling")
                    indices = np.random.choice(len(reduced_events), self.max_points, replace=False)
                    events = reduced_events[indices]
                else:
                    events = reduced_events
        
        # 构建点云
        if self.time_dimension:
            # 使用时间作为z坐标，极性计数作为特征：[x, y, t, p1_count, p0_count]
            # print(f"[DEBUG] Using time dimension, point cloud will have 5 features")
            point_cloud = np.zeros((len(events), 5), dtype=np.float32)
            point_cloud[:, 0] = events[:, 1]  # x坐标
            point_cloud[:, 1] = events[:, 2]  # y坐标
            point_cloud[:, 2] = events[:, 0]  # 时间戳作为z坐标
            point_cloud[:, 3] = events[:, 3]  # p=1的计数
            point_cloud[:, 4] = events[:, 4]  # p=0的计数
        else:
            # 仅使用空间坐标和极性计数：[x, y, p1_count, p0_count]
            # print(f"[DEBUG] Not using time dimension, point cloud will have 4 features")
            point_cloud = np.zeros((len(events), 4), dtype=np.float32)
            point_cloud[:, 0] = events[:, 1]  # x坐标
            point_cloud[:, 1] = events[:, 2]  # y坐标
            point_cloud[:, 2] = events[:, 3]  # p=1的计数
            point_cloud[:, 3] = events[:, 4]  # p=0的计数
        
        # 填充到固定大小 (如果点数不足max_points)
        if len(point_cloud) < self.max_points:
            pad_count = self.max_points - len(point_cloud)
            print(f"[DEBUG] Padding with {pad_count} additional points")
            # 创建填充点 (在现有点的基础上添加小随机偏移)
            if len(point_cloud) > 0:
                if len(point_cloud) < pad_count:
                    # 如果原始点太少，可能需要多次重复采样
                    repeat_times = int(np.ceil(pad_count / len(point_cloud)))
                    indices = np.tile(np.arange(len(point_cloud)), repeat_times)[:pad_count]
                else:
                    # 随机选择原始点进行复制
                    indices = np.random.choice(len(point_cloud), pad_count, replace=True)
                
                pad_points = point_cloud[indices].copy()
                # 只对坐标添加噪声 小随机偏移以增加多样性，不修改极性计数
                if self.time_dimension:  # 5维特征: [x, y, t, p1, p0]
                    pad_points[:, :3] += np.random.normal(0, self.padpoints_noise_scale, (pad_count, 3))
                else:  # 4维特征: [x, y, p1, p0]
                    pad_points[:, :2] += np.random.normal(0, self.padpoints_noise_scale, (pad_count, 2))
                
                point_cloud = np.vstack([point_cloud, pad_points])
            else:
                # 如果没有点，创建随机但有意义的点云
                print(f"[DEBUG] No points available, creating synthetic point cloud")
                point_cloud = np.zeros((self.max_points, point_cloud.shape[1]), dtype=np.float32)
                point_cloud[:, 0] = np.random.uniform(0, self.target_width - 1, self.max_points)  # x坐标
                point_cloud[:, 1] = np.random.uniform(0, self.target_height - 1, self.max_points)  # y坐标
                if self.time_dimension:
                    point_cloud[:, 2] = np.sort(np.random.uniform(0, self.window_size_us, self.max_points))
                point_cloud[:, -2:] = np.random.randint(0, 1, (self.max_points, 2))
        
        return point_cloud
    
    def augment_point_cloud(self, point_cloud):
        """对点云进行多种数据增强，返回多个增强后的点云
        Args:
            point_cloud: 输入点云数据
        Returns:
            list: 包含原始点云和多个增强版本的列表
        """
        print(f"[DEBUG] Generating augmented versions of point cloud with shape {point_cloud.shape}")
        augmented_point_clouds = [point_cloud.copy()]  # 保留原始点云
        
        # 1. 旋转增强 - 使用特定角度而不是随机角度
        rotation_angles = [-90, -45, 45, 90]  # 角度，单位为度
        for angle in rotation_angles:
            augmented = point_cloud.copy()
            theta = np.radians(angle)
            print(f"[DEBUG] Aug: Rotating by {angle} degrees ({theta:.2f} radians) around z-axis")
            cos_theta, sin_theta = np.cos(theta), np.sin(theta)
            x, y = augmented[:, 0], augmented[:, 1]
            augmented[:, 0] = x * cos_theta - y * sin_theta
            augmented[:, 1] = x * sin_theta + y * cos_theta
            augmented[:, 0] = np.clip(augmented[:, 0], 0, self.target_width - 1)
            augmented[:, 1] = np.clip(augmented[:, 1], 0, self.target_height - 1)
            augmented_point_clouds.append(augmented)
        
        # 2. 抖动增强
        jitter_scales = [0.005, 0.01, 0.02]
        for scale in jitter_scales:
            augmented = point_cloud.copy()
            print(f"[DEBUG] Aug: Adding xy jitter with scale {scale}")
            augmented[:, 0] += np.random.normal(0, scale, augmented.shape[0])
            augmented[:, 1] += np.random.normal(0, scale, augmented.shape[0])
            augmented[:, 0] = np.clip(augmented[:, 0], 0, self.target_width - 1)
            augmented[:, 1] = np.clip(augmented[:, 1], 0, self.target_height - 1)
            augmented_point_clouds.append(augmented)
        
        # 3. 平移增强 - 在xy方向上使用特定偏移量
        # 平移范围为目标宽高的±10%
        width_shifts = [-0.1, -0.05, 0.05, 0.1]
        height_shifts = [-0.1, -0.05, 0.05, 0.1]
        
        for w_shift_pct in width_shifts:
            for h_shift_pct in height_shifts:
                augmented = point_cloud.copy()
                x_shift = w_shift_pct * self.target_width
                y_shift = h_shift_pct * self.target_height
                print(f"[DEBUG] Aug: Applying translation: x_shift={x_shift:.2f}, y_shift={y_shift:.2f}")
                augmented[:, 0] += x_shift
                augmented[:, 1] += y_shift
                augmented[:, 0] = np.clip(augmented[:, 0], 0, self.target_width - 1)
                augmented[:, 1] = np.clip(augmented[:, 1], 0, self.target_height - 1)
                augmented_point_clouds.append(augmented)
        
        print(f"[DEBUG] Generated {len(augmented_point_clouds)} augmented point clouds (including original)")
        return augmented_point_clouds

    def sliding_window_events(self, events):
        """使用滑动窗口将长事件序列分割成多个短序列"""
        windows = []
        if len(events) == 0:
            # print(f"[DEBUG] No events to process")
            return windows
        
        t_min, t_max = 0, events[:, 0].max()
        t_range = t_max - t_min
        # 如果总时间范围小于窗口大小，直接返回整个序列
        if t_range <= self.window_size_us:
            # print(f"[DEBUG] Time range smaller than window size, returning entire sequence")
            return [events]
        
        # 滑动窗口切分
        start_time = t_min
        window_count = 0
        while start_time + self.window_size_us <= t_max:
            mask = (events[:, 0] >= start_time) & (events[:, 0] < start_time + self.window_size_us)
            window_events = events[mask]
            
            # 只有当窗口内有足够事件时才保留
            total_events = np.sum(window_events[:, 3:5])  # 累加p=1和p=0的计数
            if total_events >= self.min_events_per_window:
                windows.append(window_events)
                window_count += 1
                # print(f"[DEBUG] Window {window_count}: start={start_time}, events={len(window_events)}")
            else:
                print(f"[DEBUG] Skipped window at start={start_time} (only {len(window_events)} events)")
            start_time += self.stride_us
        
        # 确保至少有一个窗口
        if not windows and len(events) > 0:
            # print(f"[DEBUG] No valid windows found, using entire sequence")
            windows.append(events)
        
        return windows

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取单个点云样本"""
        file_path, label, window_idx, point_cloud_tensor = self.samples[idx]
        # print(f"[DEBUG] Getting item {idx}: file={file_path}, label={label}")
        # print(f"[DEBUG] Window events info: point_cloud_tensor shape={point_cloud_tensor.shape}, window_idx={window_idx}")
        # print(f"[DEBUG] Window events info: time=[{point_cloud_tensor[:, 0].min()}, {point_cloud_tensor[:, 0].max()}]")
        return point_cloud_tensor, label

if __name__ == '__main__':
    config_path='configs/har_train_config.yaml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 测试数据集
    ds_cfg = cfg['dataset']
    data_ds = ESequenceDataset(
        data_root          = ds_cfg['train_dir'],
        window_size_us     = ds_cfg['window_size_us'],
        stride_us          = ds_cfg['stride_us'],
        max_points         = ds_cfg['max_points'],
        time_dimension     = ds_cfg['time_dimension'],
        enable_augment     = ds_cfg['enable_augment'],
        label_map          = ds_cfg['label_map']
    )
    
    # 保存处理后的数据集到文件
    save_dir = "preprocessing_data"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "train_dataset10.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(data_ds, f)

    # 加载并检查保存的数据集
    try:
        with open(save_path, 'rb') as f:
            loaded_ds = pickle.load(f)
        print(f"成功加载数据集，包含 {len(loaded_ds)} 个样本")
        
        # 随机检查一个样本 shape为 [N, 5] 或 [N, 4]， N为点云点数
        sample_idx = random.randint(0, len(loaded_ds)-1)
        sample_data, sample_label = loaded_ds[sample_idx]
        print(f"\n随机样本 #{sample_idx}:")
        print(f"  - 形状: {sample_data.shape}")
        print(f"  - 标签: {sample_label}")
        print(f"  - 点云范围: x[{sample_data[:, 0].min():.2f}, {sample_data[:, 0].max():.2f}], "
              f"y[{sample_data[:, 1].min():.2f}, {sample_data[:, 1].max():.2f}]")
        
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"\n数据集文件大小: {file_size_mb:.2f} MB")
    except Exception as e:
        print(f"加载数据集时出错: {e}")

   