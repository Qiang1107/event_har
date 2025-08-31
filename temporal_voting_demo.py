#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间窗口投票测试脚本
"""

import os
import numpy as np
import yaml

def test_temporal_voting_concept():
    """测试时间窗口投票概念的演示"""
    
    print("=== 时间窗口投票概念演示 ===\n")
    
    # 模拟数据：假设我们有一个文件中的事件数据
    np.random.seed(42)
    
    # 模拟事件时间戳（微秒），总时长约3秒
    total_events = 30000
    timestamps_us = np.sort(np.random.uniform(0, 3000000, total_events))  # 3秒内的事件
    
    # 模拟其他事件属性
    x_coords = np.random.uniform(0, 346, total_events)
    y_coords = np.random.uniform(0, 260, total_events)
    polarities = np.random.choice([0, 1], total_events)
    
    events = np.column_stack((timestamps_us, x_coords, y_coords, polarities))
    
    print(f"模拟事件数据:")
    print(f"  总事件数: {total_events}")
    print(f"  时间范围: {timestamps_us[0]/1e6:.3f}s - {timestamps_us[-1]/1e6:.3f}s")
    print(f"  总时长: {(timestamps_us[-1] - timestamps_us[0])/1e6:.3f}s")
    
    # 创建8192事件的滑动窗口
    window_size = 8192
    step_size = 1024
    
    samples = []
    for start_idx in range(0, len(events) - window_size, step_size):
        end_idx = start_idx + window_size
        window_events = events[start_idx:end_idx]
        
        start_time_sec = window_events[0, 0] / 1e6
        end_time_sec = window_events[-1, 0] / 1e6
        duration_ms = (end_time_sec - start_time_sec) * 1000
        
        samples.append({
            'start_idx': start_idx,
            'end_idx': end_idx,
            'start_time_sec': start_time_sec,
            'end_time_sec': end_time_sec,
            'duration_ms': duration_ms,
            'center_time_sec': (start_time_sec + end_time_sec) / 2
        })
    
    print(f"\n滑动窗口分析:")
    print(f"  窗口大小: {window_size} 事件")
    print(f"  步长: {step_size} 事件")
    print(f"  生成样本数: {len(samples)}")
    
    # 分析每个样本的时间特征
    durations = [s['duration_ms'] for s in samples]
    print(f"  样本时长 - 平均: {np.mean(durations):.1f}ms, 范围: {np.min(durations):.1f}-{np.max(durations):.1f}ms")
    
    # 模拟预测结果
    num_classes = 8
    predictions = []
    
    print(f"\n模拟预测结果:")
    print(f"  类别数: {num_classes}")
    
    for i, sample in enumerate(samples):
        # 模拟预测（随机，但某些时间段更倾向于某类别）
        time_factor = sample['center_time_sec']
        dominant_class = int(time_factor) % num_classes  # 基于时间的"真实"类别
        
        # 97%概率预测正确
        if np.random.random() < 0.97:
            prediction = dominant_class
        else:
            prediction = np.random.choice([c for c in range(num_classes) if c != dominant_class])
        
        confidence = np.random.uniform(0.7, 0.99)
        
        predictions.append({
            'sample_idx': i,
            'center_time_sec': sample['center_time_sec'],
            'prediction': prediction,
            'true_label': dominant_class,
            'confidence': confidence,
            'is_correct': prediction == dominant_class
        })
    
    # 计算单样本准确率
    single_accuracy = np.mean([p['is_correct'] for p in predictions])
    print(f"  单样本准确率: {single_accuracy:.4f}")
    
    # 时间窗口投票
    time_window_sec = 1.0
    print(f"\n时间窗口投票 (窗口大小: {time_window_sec}秒):")
    
    # 按时间窗口分组
    time_windows = {}
    for pred in predictions:
        window_id = int(pred['center_time_sec'] // time_window_sec)
        if window_id not in time_windows:
            time_windows[window_id] = []
        time_windows[window_id].append(pred)
    
    # 在每个窗口内投票
    window_results = []
    for window_id in sorted(time_windows.keys()):
        window_preds = time_windows[window_id]
        
        # 获取真实标签（应该在窗口内一致）
        true_labels = [p['true_label'] for p in window_preds]
        window_true_label = max(set(true_labels), key=true_labels.count)
        
        # 多数投票
        vote_counts = {}
        for pred in window_preds:
            prediction = pred['prediction']
            vote_counts[prediction] = vote_counts.get(prediction, 0) + 1
        
        final_prediction = max(vote_counts.items(), key=lambda x: x[1])[0]
        vote_confidence = vote_counts[final_prediction] / len(window_preds)
        
        window_results.append({
            'window_id': window_id,
            'window_time': f"{window_id * time_window_sec:.1f}-{(window_id + 1) * time_window_sec:.1f}s",
            'num_predictions': len(window_preds),
            'final_prediction': final_prediction,
            'true_label': window_true_label,
            'vote_confidence': vote_confidence,
            'is_correct': final_prediction == window_true_label,
            'individual_predictions': [p['prediction'] for p in window_preds]
        })
    
    # 计算时间窗口准确率
    window_accuracy = np.mean([w['is_correct'] for w in window_results])
    
    print(f"  时间窗口数: {len(window_results)}")
    print(f"  时间窗口准确率: {window_accuracy:.4f}")
    print(f"  准确率提升: {window_accuracy - single_accuracy:+.4f}")
    
    # 显示前几个窗口的详细信息
    print(f"\n前5个时间窗口详情:")
    print(f"{'窗口':<8} {'时间段':<12} {'投票数':<6} {'预测':<4} {'真实':<4} {'正确':<4} {'置信度':<6}")
    print("-" * 50)
    
    for w in window_results[:5]:
        correct_symbol = "✓" if w['is_correct'] else "✗"
        print(f"{w['window_id']:<8} {w['window_time']:<12} {w['num_predictions']:<6} "
              f"{w['final_prediction']:<4} {w['true_label']:<4} {correct_symbol:<4} {w['vote_confidence']:<6.3f}")
    
    print(f"\n=== 关键发现 ===")
    print("1. 8192个事件点的时间跨度通常小于1秒")
    print("2. 1秒时间窗口内包含多个8192事件的预测")
    print("3. 通过多数投票可以进一步提高准确率")
    print("4. 这种方法特别适合实时应用场景")
    
    return {
        'single_accuracy': single_accuracy,
        'window_accuracy': window_accuracy,
        'improvement': window_accuracy - single_accuracy,
        'num_samples': len(predictions),
        'num_windows': len(window_results)
    }

def check_real_data_feasibility():
    """检查真实数据的可行性"""
    
    print("\n=== 真实数据可行性检查 ===")
    
    # 检查是否存在测试数据
    test_data_dir = "data/test"
    if not os.path.exists(test_data_dir):
        print(f"测试数据目录不存在: {test_data_dir}")
        return False
    
    # 检查配置文件
    config_path = "configs/har_test_config.yaml"
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    ds = cfg['dataset']
    label_map = ds['label_map']
    
    print(f"类别映射: {label_map}")
    
    # 检查每个类别的数据文件
    total_files = 0
    for class_name, class_idx in label_map.items():
        class_dir = os.path.join(test_data_dir, class_name)
        if os.path.exists(class_dir):
            npy_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
            print(f"  {class_name}: {len(npy_files)} 个.npy文件")
            total_files += len(npy_files)
        else:
            print(f"  {class_name}: 目录不存在")
    
    print(f"总文件数: {total_files}")
    
    if total_files > 0:
        print("✓ 数据文件存在，可以进行时间窗口投票测试")
        return True
    else:
        print("✗ 没有找到可用的数据文件")
        return False

if __name__ == '__main__':
    # 运行概念演示
    results = test_temporal_voting_concept()
    
    # 检查真实数据
    feasible = check_real_data_feasibility()
    
    if feasible:
        print(f"\n💡 建议的下一步:")
        print(f"1. 运行时间窗口投票测试:")
        print(f"   python3 prediction_temporal_voting.py --time_window 1.0 --voting_method majority")
        print(f"2. 尝试不同的时间窗口大小:")
        print(f"   python3 prediction_temporal_voting.py --time_window 0.5")
        print(f"   python3 prediction_temporal_voting.py --time_window 2.0")
        print(f"3. 测试不同的投票方法:")
        print(f"   python3 prediction_temporal_voting.py --voting_method weighted")
        print(f"   python3 prediction_temporal_voting.py --voting_method confidence_weighted")
    else:
        print(f"\n⚠️  请确保测试数据文件存在后再运行实际测试")
