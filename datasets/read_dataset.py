"""
    e.g., 事件相机时间戳格式16位: 1745258068 633440
    前10位是秒,1745258068 是秒, 等于2025-04-21 19:54:28
    后6位是微秒,633440 是微秒, 等于633440 微秒
    e.g., rgb相机时间戳格式10+6位: 1745258095.815377
"""

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
from event_count_seq_dataset_vote import ECountSeqDatasetVote

if __name__ == '__main__':
    save_dir = "preprocessing_data"
    save_path = os.path.join(save_dir, "test_data_0628_8_ecount_3_vote.pkl") 


    # 加载并检查数据集
    try:
        with open(save_path, 'rb') as f:
            print(f"正在加载数据集: {save_path} ...")
            loaded_ds = pickle.load(f)
        print(f"成功加载数据集，包含 {len(loaded_ds)} 个样本")
        
        # 随机检查一个样本 shape为 [N, 4]， N为点云点数
        sample_idx = random.randint(0, len(loaded_ds)-1)
        sample_data, sample_label, sample_start_ts, sample_end_ts, sample_center_ts, sample_file_path, sample_window_duration, sample_class_name = loaded_ds[sample_idx]
        print(f"\n随机样本 #{sample_idx}:")
        print(f"  - 形状: {sample_data.shape}")  # 形状: (8192, 4)
        print(f"  - 标签: {sample_label}") # 标签: 0
        print(f"  - 点云范围: t[{sample_data[:, 0].min():.2f}, {sample_data[:, 0].max():.2f}], "
              f"x[{sample_data[:, 1].min():.2f}, {sample_data[:, 1].max():.2f}], "
              f"y[{sample_data[:, 2].min():.2f}, {sample_data[:, 2].max():.2f}]")
        # 点云范围: t[0.03, 0.24], x[61.00, 284.00], y[33.00, 235.00]
        print(f"  - 时间戳: start {sample_start_ts}, end {sample_end_ts}, center {sample_center_ts}") 
        # 时间戳: start 1751113355332684, end 1751113355486427, center 1751113355409555.5
        print(f"  - 窗口持续时间: {sample_window_duration} µs") # 窗口持续时间: 153743 µs
        print(f"  - 文件路径: {sample_file_path}")  # 文件路径: data/test/Approach/21_3.npy
        print(f"  - 类别名称: {sample_class_name}") # 类别名称: Approach
        
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"\n数据集文件大小: {file_size_mb:.2f} MB") # 数据集文件大小: 2820.96 MB
    except Exception as e:
        print(f"加载数据集时出错: {e}")
