import pickle
import numpy as np
import os
import torch
import sys
import random

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from torch.utils.data import DataLoader
from datasets.rgbe_sequence_dataset import RGBESequenceDataset
from datasets.event_sequence_dataset import ESequenceDataset


def check_point_counts(pkl_file, max_points=None, verbose=True):
    """
    Check if each data entry in the pkl file has the specified number of points.
    
    Args:
        pkl_file (str): Path to the pickle file
        max_points (int, optional): Expected number of points. If None, will use the maximum found
        verbose (bool): Whether to print detailed information
        
    Returns:
        bool: True if all entries have the same number of points as max_points
    """
    if not os.path.exists(pkl_file):
        print(f"File does not exist: {pkl_file}")
        return False
    
    # 加载并检查保存的数据集
    try:
        with open(pkl_file, 'rb') as f:
            loaded_ds = pickle.load(f)
        print(f"成功加载数据集，包含 {len(loaded_ds)} 个样本")
        print(f"类别到索引映射: {loaded_ds.class_to_idx}")
        
        # 检查所有样本的点数
        counts = []
        consistent = True
        for i in range(len(loaded_ds)):
            data, _ = loaded_ds[i]
            point_count = data.shape[0]
            counts.append(point_count)
            if max_points and point_count != max_points:
                consistent = False
                if verbose:
                    print(f"样本 #{i} 点数: {point_count} (与预期的 {max_points} 不符)")
        
        if consistent:
            print(f"所有样本都包含 {max_points} 个点")
        else:
            unique_counts = np.unique(counts)
            print(f"\n点数分布: {dict(zip(*np.unique(counts, return_counts=True)))}")
            print(f"最小点数: {min(counts)}, 最大点数: {max(counts)}")
        
        # 随机检查一个样本 shape为 [N, 5] 或 [N, 4]， N为点云点数
        sample_idx = random.randint(0, len(loaded_ds)-1)
        sample_data, sample_label = loaded_ds[sample_idx]
        print(f"\n随机样本 #{sample_idx}:")
        print(f"  - 形状: {sample_data.shape}")
        print(f"  - 标签: {sample_label} ({loaded_ds.classes[sample_label]})")
        print(f"  - 点云范围: x[{sample_data[:, 0].min():.2f}, {sample_data[:, 0].max():.2f}], "
              f"y[{sample_data[:, 1].min():.2f}, {sample_data[:, 1].max():.2f}]")
        
        file_size_mb = os.path.getsize(pkl_file) / (1024 * 1024)
        print(f"\n数据集文件大小: {file_size_mb:.2f} MB")
    except Exception as e:
        print(f"加载数据集时出错: {e}")

if __name__ == "__main__":
    # Directly specify parameters here instead of using command line arguments
    pkl_file = 'preprocessing_data/test_dataset.pkl'
    max_points = 8192  # Expected number of points
    verbose = True     # Set to False to suppress detailed output
    
    # Call the function with the specified parameters
    check_point_counts(pkl_file, max_points, verbose=verbose)
