#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版时间窗口投票预测 - 基于现有预处理数据
"""

import os
import pickle
import time
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, Counter
from torch.utils.data import DataLoader
import yaml

from datasets.event_count_seq_dataset import ECountSeqDataset
from models.backbones.pointnet2_v1 import PointNet2Classifier
from models.backbones.pointnet2msg_v1 import PointNet2MSGClassifier

def simulate_temporal_voting_on_preprocessed_data(config_path, model_path, time_window_sec=1.0):
    """在预处理数据上模拟时间窗口投票"""
    
    print(f"=== 基于预处理数据的时间窗口投票 ===")
    
    # 1. 加载配置
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    model_type = cfg['model_type']
    class_names = list(cfg['dataset']['label_map'].keys())
    
    # 2. 加载预处理的测试数据
    test_data_path = "preprocessing_data/test_data_0628_8_ecount_3.pkl"
    if not os.path.exists(test_data_path):
        print(f"预处理数据不存在: {test_data_path}")
        return
    
    with open(test_data_path, 'rb') as f:
        test_dataset = pickle.load(f)
    
    print(f"加载了 {len(test_dataset)} 个预处理样本")
    
    # 3. 加载模型
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    if model_type == 'pointnet2':
        model = PointNet2Classifier(cfg).to(device)
    elif model_type == 'pointnet2msg':
        model = PointNet2MSGClassifier(cfg).to(device)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"模型加载成功: {model_path}")
    
    # 4. 对所有样本进行预测
    print("开始对所有样本进行预测...")
    all_predictions = []
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            events, true_label = test_dataset[i]
            
            # 转换为tensor
            events_tensor = torch.FloatTensor(events).unsqueeze(0).to(device)
            
            # 预测
            logits = model(events_tensor)
            probs = F.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            confidence = torch.max(probs, dim=1)[0].item()
            
            # 模拟时间戳（基于样本索引）
            # 假设每个样本相隔约100ms，这样可以在1秒内有多个样本
            simulated_timestamp = i * 0.1  # 每100ms一个样本
            
            all_predictions.append({
                'sample_idx': i,
                'timestamp': simulated_timestamp,
                'prediction': prediction,
                'confidence': confidence,
                'true_label': true_label,
                'probabilities': probs.cpu().numpy()[0]
            })
    
    print(f"完成 {len(all_predictions)} 个样本的预测")
    
    # 5. 计算单样本准确率
    individual_accuracy = np.mean([p['prediction'] == p['true_label'] for p in all_predictions])
    print(f"单样本准确率: {individual_accuracy:.4f}")
    
    # 6. 按时间窗口分组进行投票
    print(f"\n开始时间窗口投票 (窗口大小: {time_window_sec}秒)...")
    
    # 按时间窗口分组
    time_windows = defaultdict(list)
    for pred in all_predictions:
        window_id = int(pred['timestamp'] // time_window_sec)
        time_windows[window_id].append(pred)
    
    # 在每个窗口内进行投票
    window_results = []
    for window_id in sorted(time_windows.keys()):
        window_preds = time_windows[window_id]
        
        if len(window_preds) < 2:  # 跳过只有一个预测的窗口
            continue
        
        # 获取窗口内的真实标签（取众数）
        true_labels = [p['true_label'] for p in window_preds]
        window_true_label = Counter(true_labels).most_common(1)[0][0]
        
        # 多数投票
        predictions = [p['prediction'] for p in window_preds]
        vote_counts = Counter(predictions)
        final_prediction = vote_counts.most_common(1)[0][0]
        vote_confidence = vote_counts[final_prediction] / len(predictions)
        
        # 置信度加权投票
        weighted_votes = defaultdict(float)
        total_confidence = 0
        for pred in window_preds:
            weighted_votes[pred['prediction']] += pred['confidence']
            total_confidence += pred['confidence']
        
        confidence_weighted_pred = max(weighted_votes.items(), key=lambda x: x[1])[0]
        
        window_results.append({
            'window_id': window_id,
            'window_start': window_id * time_window_sec,
            'window_end': (window_id + 1) * time_window_sec,
            'num_predictions': len(window_preds),
            'true_label': window_true_label,
            'majority_vote': final_prediction,
            'confidence_weighted_vote': confidence_weighted_pred,
            'vote_confidence': vote_confidence,
            'individual_predictions': predictions,
            'individual_confidences': [p['confidence'] for p in window_preds]
        })
    
    # 7. 计算投票准确率
    majority_accuracy = np.mean([w['majority_vote'] == w['true_label'] for w in window_results])
    confidence_accuracy = np.mean([w['confidence_weighted_vote'] == w['true_label'] for w in window_results])
    
    print(f"\n=== 结果分析 ===")
    print(f"时间窗口数: {len(window_results)}")
    print(f"单样本准确率: {individual_accuracy:.4f}")
    print(f"多数投票准确率: {majority_accuracy:.4f}")
    print(f"置信度加权投票准确率: {confidence_accuracy:.4f}")
    print(f"多数投票提升: {majority_accuracy - individual_accuracy:+.4f}")
    print(f"置信度投票提升: {confidence_accuracy - individual_accuracy:+.4f}")
    
    # 8. 每个窗口的投票统计
    vote_counts = [w['num_predictions'] for w in window_results]
    print(f"\n=== 投票统计 ===")
    print(f"每窗口预测数 - 平均: {np.mean(vote_counts):.1f}, 范围: {np.min(vote_counts)}-{np.max(vote_counts)}")
    
    # 9. 显示前几个窗口的详细信息
    print(f"\n=== 前10个窗口详情 ===")
    print(f"{'窗口':<4} {'时间段':<12} {'投票数':<6} {'真实':<4} {'多数票':<6} {'置信票':<6} {'正确':<6}")
    print("-" * 55)
    
    for w in window_results[:10]:
        majority_correct = "✓" if w['majority_vote'] == w['true_label'] else "✗"
        confidence_correct = "✓" if w['confidence_weighted_vote'] == w['true_label'] else "✗"
        print(f"{w['window_id']:<4} {w['window_start']:.1f}-{w['window_end']:.1f}s   {w['num_predictions']:<6} "
              f"{w['true_label']:<4} {w['majority_vote']:<6} {w['confidence_weighted_vote']:<6} "
              f"{majority_correct}/{confidence_correct:<5}")
    
    # 10. 分析不同投票数量下的准确率
    print(f"\n=== 不同投票数量的准确率 ===")
    vote_accuracy_analysis = {}
    for num_votes in range(2, max(vote_counts) + 1):
        windows_with_n_votes = [w for w in window_results if w['num_predictions'] == num_votes]
        if windows_with_n_votes:
            accuracy = np.mean([w['majority_vote'] == w['true_label'] for w in windows_with_n_votes])
            vote_accuracy_analysis[num_votes] = {
                'count': len(windows_with_n_votes),
                'accuracy': accuracy
            }
            print(f"投票数={num_votes}: 准确率={accuracy:.4f} (窗口数={len(windows_with_n_votes)})")
    
    return {
        'individual_accuracy': individual_accuracy,
        'majority_accuracy': majority_accuracy,
        'confidence_accuracy': confidence_accuracy,
        'majority_improvement': majority_accuracy - individual_accuracy,
        'confidence_improvement': confidence_accuracy - individual_accuracy,
        'num_windows': len(window_results),
        'num_samples': len(all_predictions)
    }

def test_different_time_windows():
    """测试不同时间窗口大小"""
    
    config_path = "configs/har_test_config.yaml"
    model_path = "results/checkpoints/pointnet2_event_0628_8_ecount_11.pth"
    
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    
    # 测试不同的时间窗口大小
    time_windows = [0.5, 1.0, 1.5, 2.0, 3.0]
    results = {}
    
    print("测试不同时间窗口大小的效果:")
    print("=" * 60)
    
    for window_size in time_windows:
        print(f"\n测试时间窗口: {window_size}秒")
        result = simulate_temporal_voting_on_preprocessed_data(config_path, model_path, window_size)
        results[window_size] = result
    
    # 总结对比
    print(f"\n" + "=" * 60)
    print(f"时间窗口大小对比总结:")
    print("=" * 60)
    print(f"{'窗口大小':<8} {'窗口数':<8} {'单样本':<8} {'多数票':<8} {'置信票':<8} {'多数提升':<8} {'置信提升':<8}")
    print("-" * 72)
    
    for window_size in time_windows:
        if window_size in results:
            r = results[window_size]
            print(f"{window_size:<8.1f} {r['num_windows']:<8} {r['individual_accuracy']:<8.4f} "
                  f"{r['majority_accuracy']:<8.4f} {r['confidence_accuracy']:<8.4f} "
                  f"{r['majority_improvement']:<8.4f} {r['confidence_improvement']:<8.4f}")
    
    # 找出最佳配置
    best_majority = max(results.items(), key=lambda x: x[1]['majority_improvement'])
    best_confidence = max(results.items(), key=lambda x: x[1]['confidence_improvement'])
    
    print(f"\n最佳配置:")
    print(f"多数投票最佳: 窗口大小={best_majority[0]}s, 提升={best_majority[1]['majority_improvement']:+.4f}")
    print(f"置信度投票最佳: 窗口大小={best_confidence[0]}s, 提升={best_confidence[1]['confidence_improvement']:+.4f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='简化版时间窗口投票测试')
    parser.add_argument('--time_window', type=float, default=1.0,
                        help='时间窗口大小（秒）')
    parser.add_argument('--test_all', action='store_true',
                        help='测试所有时间窗口大小')
    
    args = parser.parse_args()
    
    if args.test_all:
        test_different_time_windows()
    else:
        config_path = "configs/har_test_config.yaml"
        model_path = "results/checkpoints/pointnet2_event_0628_8_ecount_11.pth"
        
        simulate_temporal_voting_on_preprocessed_data(config_path, model_path, args.time_window)
