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
                 time_dimension, enable_augment, stride_us, label_map,
                 t_squash_factor, target_width, target_height,
                 min_events_per_window):
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
        self.stride_us = stride_us
        self.max_points = max_points
        self.time_dimension = time_dimension
        self.enable_augment = enable_augment
        self.t_squash_factor = t_squash_factor  # 每t_squash_factor微秒内的时间戳表示成一个时间戳
        self.target_width, self.target_height = target_width, target_height  # 目标空间坐标大小 # 原始宽度和高度为346和260
        self.min_events_per_window = min_events_per_window  # 一个时间窗口中最小事件数阈值

        # 处理事件数据的一些参数设置
        self.padpoints_noise_scale = 0.02  # 填充点的噪声尺度
        
        # 收集所有文件路径和初始标签
        self.samples = []
        total_windows = 0
        start_time1 = time.time()
        for cls_name, idx in label_map.items():
            class_dir = os.path.join(data_root, cls_name)
            files = glob.glob(os.path.join(class_dir, "*.npy"))
            for file_path in files:     
                events = np.load(file_path)
                print(f"[DEBUG] Loaded events shape: {events.shape}, label: {idx}, time range: [{events[:, 0].min()}, {events[:, 0].max()}] us")
                # 检查事件数组是否为空
                if events.size == 0 or events.shape[0] == 0:
                    print(f"警告: 文件 {file_path} 包含空的事件数组，跳过处理")
                    continue

                t, x, y, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
                # 时间戳转换成微秒
                if t.dtype != np.int64:  # Check if timestamps are likely in Unix format
                    print(f"[ERROR] 时间戳格式错误，期望为整数类型，实际为 {t.dtype}")
                else:     
                    min_timestamp = t.min()
                    t = t - min_timestamp
                # 时间戳等比例压缩 Group events in 10ms windows and assigning average timestamp
                time_bins = (t // self.t_squash_factor).astype(np.int64)
                unique_bins = np.unique(time_bins)
                for bin_idx in unique_bins:
                    bin_mask = time_bins == bin_idx
                    if np.sum(bin_mask) > 0:
                        bin_avg_time = np.mean(t[bin_mask]).astype(np.int64)
                        t[bin_mask] = bin_avg_time
                # print(f"[DEBUG] After time squashing, unique timestamps: {len(np.unique(events[:, 0]))}/{len(events)}")
                # 空间坐标xy缩放到指定大小
                original_width, original_height = 346, 260  
                width_scale = self.target_width / original_width
                height_scale = self.target_height / original_height
                x = np.clip(x * width_scale, 0, self.target_width - 1)
                y = np.clip(y * height_scale, 0, self.target_height - 1)
                # 通过合并时间戳和空间坐标相同的事件降采样
                event_keys = {}
                merged_events = []
                for i in range(len(t)):
                    key = (t[i], int(x[i]), int(y[i]))  # 创建(t,x,y)元组作为键
                    if key not in event_keys:
                        event_keys[key] = [0, 0]  # [p=0计数, p=1计数]
                    if p[i] == 0:
                        event_keys[key][0] += 1
                    else:
                        event_keys[key][1] += 1
                for key, counts in event_keys.items():
                    t_val, x_val, y_val = key
                    p_val = 1 if counts[1] >= counts[0] else 0
                    merged_events.append([t_val, x_val, y_val, p_val])
                
                # 用合并后的事件替换原始事件数组
                new_events = np.array(merged_events)
                print(f"[DEBUG] After merging, events reduced from {len(t)} to {len(new_events)}")
                # print(f"[DEBUG] After merging,  events time range: [{new_events[:, 0].min()}, {new_events[:, 0].max()}] us")
                
                
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
                windows = self.sliding_window_events(new_events)
                # print(f"[DEBUG] Generated {len(windows)} windows from file {file_path}")
                
                # 将每个窗口添加为单独的样本 以便在__getitem__中直接使用
                for window_idx, window_events in enumerate(windows):
                    point_cloud = self.events_to_pointcloud(window_events) # 转换为PointNet++兼容的点云格式
                    point_cloud_tensor = torch.from_numpy(point_cloud).float()
                    self.samples.append((file_path, idx, window_idx, point_cloud_tensor))
                
                total_windows += len(windows)

        end_time1 = time.time()
        print(f"[DEBUG] Processed time is {end_time1 - start_time1:.2f} seconds")
        print(f"使用滑动窗口处理后,共生成{total_windows}个训练样本来自{len(label_map)}个类别的")
        
    def events_to_pointcloud(self, events):
        """将事件数据转换为点云格式,适合PointNet++处理"""      
        # 构建点云
        if self.time_dimension:
            # 使用时间作为z坐标，极性计数作为特征：[t, x, y, p]
            point_cloud = np.zeros((len(events), 4), dtype=np.float32)
            point_cloud[:, 0] = events[:, 0]  # t
            point_cloud[:, 1] = events[:, 1]  # x
            point_cloud[:, 2] = events[:, 2]  # y
            point_cloud[:, 3] = events[:, 3]  # p
        else:
            # 仅使用空间坐标和极性：[x, y, p]
            point_cloud = np.zeros((len(events), 4), dtype=np.float32)
            point_cloud[:, 0] = events[:, 1]  # x
            point_cloud[:, 1] = events[:, 2]  # y
            point_cloud[:, 2] = events[:, 3]  # p
        
        # 降采样到固定大小 (如果点数超过max_points)
        if len(point_cloud) > self.max_points:
            # print(f"[DEBUG] Downsampling from {len(point_cloud)} points to {self.max_points}")
            if self.time_dimension:
                # 1. 按时间排序
                time_sorted_indices = np.argsort(point_cloud[:, 0])
                sorted_point_cloud = point_cloud[time_sorted_indices]
                # 2. 系统采样：每隔n个点取一个点，确保时间分布均匀
                step = len(sorted_point_cloud) / self.max_points
                indices = np.floor(np.arange(0, len(sorted_point_cloud), step)).astype(int)
                indices = indices[:self.max_points]  # 确保不超过max_points
                point_cloud = sorted_point_cloud[indices]
            else:
                # 随机采样 - 对于没有时间维度的情况
                random_indices = np.random.choice(len(point_cloud), self.max_points, replace=False)
                point_cloud = point_cloud[random_indices]
        
        # 填充到固定大小 (如果点数不足max_points)
        elif len(point_cloud) < self.max_points:
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
                # 只对坐标添加噪声 小随机偏移以增加多样性，不修改极性
                if self.time_dimension:  # 4维特征: [t, x, y, p]
                    pad_points[:, :3] += np.random.normal(0, self.padpoints_noise_scale, (pad_count, 3))
                else:  # 4维特征: [x, y, p, ...]
                    pad_points[:, :2] += np.random.normal(0, self.padpoints_noise_scale, (pad_count, 2))
                
                point_cloud = np.vstack([point_cloud, pad_points])
            else:
                # 如果没有点，创建随机但有意义的点云
                print(f"[DEBUG] No points available, creating synthetic point cloud")
                point_cloud = np.zeros((self.max_points, point_cloud.shape[1]), dtype=np.float32)
                if self.time_dimension:
                    point_cloud[:, 1] = np.random.uniform(0, self.target_width - 1, self.max_points)  # x
                    point_cloud[:, 2] = np.random.uniform(0, self.target_height - 1, self.max_points)  # y
                    point_cloud[:, 0] = np.sort(np.random.uniform(0, 1, self.max_points))  # t
                else:
                    point_cloud[:, 0] = np.random.uniform(0, self.target_width - 1, self.max_points) # x
                    point_cloud[:, 1] = np.random.uniform(0, self.target_height - 1, self.max_points)  # y
                point_cloud[:, -1] = np.random.randint(0, 2, self.max_points)
        
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
            if len(window_events) >= self.min_events_per_window:
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
        
        # 对窗口内的时间戳进行归一化处理
        normalized_windows = []
        windows_t_min = min([window[:, 0].min() for window in windows if len(window) > 0])
        windows_t_max = max([window[:, 0].max() for window in windows if len(window) > 0])
        # print(f"windows_t_max: {windows_t_max}")
        # print(f"windows_t_min: {windows_t_min}")
        for window in windows:
            if len(window) > 0:
                # 深拷贝窗口数据以避免修改原始数据
                normalized_window = window.copy()
                # Convert time values to float
                normalized_window = normalized_window.astype(np.float32)
                normalized_window[:, 0] = ((normalized_window[:, 0] - windows_t_min) / (windows_t_max - windows_t_min))
                # print(f"normalized_window[:, 0]: {normalized_window[:, 0]}")
                # print(f"normalized_window[:, 0] type: {normalized_window[:, 0].dtype}")
                normalized_windows.append(normalized_window)
            else:
                normalized_windows.append(window)
        windows = normalized_windows
        # print(f"[DEBUG] After sliding_window_events,  events time range: "
        #       f"{min([window[:, 0].min() for window in windows if len(window) > 0])} us, "
        #       f"{max([window[:, 0].max() for window in windows if len(window) > 0])} us")

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
        data_root          = ds_cfg['val_dir'],
        window_size_us     = ds_cfg['window_size_us'],
        stride_us          = ds_cfg['stride_us'],
        max_points         = ds_cfg['max_points'],
        time_dimension     = ds_cfg['time_dimension'],
        enable_augment     = ds_cfg['enable_augment'],
        label_map          = ds_cfg['label_map'],
        t_squash_factor    = ds_cfg['t_squash_factor'],
        target_width       = ds_cfg['target_width'],
        target_height      = ds_cfg['target_height'],
        min_events_per_window = ds_cfg['min_events_per_window']
    )
    
    # 保存处理后的数据集到文件
    save_dir = "preprocessing_data"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "val_dataset10_eseq.pkl")
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
        print(f"  - 形状: {sample_data.shape}") # torch.float32
        print(f"  - 标签: {sample_label}")
        print(f"  - 点云范围: 0[{sample_data[:, 0].min()}, {sample_data[:, 0].max()}], "
              f"1[{sample_data[:, 1].min()}, {sample_data[:, 1].max()}], "
              f"2[{sample_data[:, 2].min()}, {sample_data[:, 2].max()}], "
              f"3[{sample_data[:, 3].min()}, {sample_data[:, 3].max()}]")
        
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"\n数据集文件大小: {file_size_mb:.2f} MB")
    except Exception as e:
        print(f"加载数据集时出错: {e}")

   