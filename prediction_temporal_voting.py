#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间窗口投票预测系统
将多个8192事件点的预测结果在1秒时间窗口内进行投票
"""

import os
import yaml
import time
import torch
import torch.nn.functional as F
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 导入数据集和模型
from datasets.event_count_seq_dataset import ECountSeqDataset
from models.backbones.pointnet2_v1 import PointNet2Classifier
from models.backbones.pointnet2msg_v1 import PointNet2MSGClassifier

class TemporalVotingPredictor:
    """时间窗口投票预测器"""
    
    def __init__(self, time_window_sec=1.0, voting_method='majority', confidence_weight=False):
        """
        Args:
            time_window_sec: 时间窗口大小（秒）
            voting_method: 投票方法 ('majority', 'weighted', 'confidence_weighted')
            confidence_weight: 是否使用置信度加权
        """
        self.time_window_sec = time_window_sec
        self.voting_method = voting_method
        self.confidence_weight = confidence_weight
        
    def group_predictions_by_time(self, predictions_with_time):
        """根据时间戳将预测结果分组到时间窗口"""
        time_windows = {}
        
        for pred_info in predictions_with_time:
            timestamp = pred_info['timestamp']
            window_id = int(timestamp // self.time_window_sec)
            
            if window_id not in time_windows:
                time_windows[window_id] = []
            time_windows[window_id].append(pred_info)
        
        return time_windows
    
    def vote_in_window(self, window_predictions):
        """在单个时间窗口内进行投票"""
        if not window_predictions:
            return None, 0.0
        
        if self.voting_method == 'majority':
            # 简单多数投票
            votes = [pred['prediction'] for pred in window_predictions]
            vote_counts = Counter(votes)
            final_prediction = vote_counts.most_common(1)[0][0]
            confidence = vote_counts[final_prediction] / len(votes)
            
        elif self.voting_method == 'weighted':
            # 基于softmax概率的加权投票
            weighted_probs = defaultdict(float)
            total_weight = 0
            
            for pred in window_predictions:
                probs = pred['probabilities']
                weight = pred.get('weight', 1.0)
                for class_idx, prob in enumerate(probs):
                    weighted_probs[class_idx] += prob * weight
                total_weight += weight
            
            # 归一化
            for class_idx in weighted_probs:
                weighted_probs[class_idx] /= total_weight
            
            final_prediction = max(weighted_probs.items(), key=lambda x: x[1])[0]
            confidence = weighted_probs[final_prediction]
            
        elif self.voting_method == 'confidence_weighted':
            # 基于置信度的加权投票
            weighted_votes = defaultdict(float)
            total_confidence = 0
            
            for pred in window_predictions:
                prediction = pred['prediction']
                confidence = pred['confidence']
                weighted_votes[prediction] += confidence
                total_confidence += confidence
            
            # 归一化
            for pred_class in weighted_votes:
                weighted_votes[pred_class] /= total_confidence
            
            final_prediction = max(weighted_votes.items(), key=lambda x: x[1])[0]
            confidence = weighted_votes[final_prediction]
        
        return final_prediction, confidence
    
    def predict_with_temporal_voting(self, model, test_data_with_time, device, class_names):
        """使用时间窗口投票进行预测"""
        model.eval()
        
        all_predictions = []
        
        print(f"开始时间窗口投票预测，窗口大小: {self.time_window_sec}秒")
        
        with torch.no_grad():
            for sample_info in test_data_with_time:
                # 获取样本数据
                events = sample_info['events']
                true_label = sample_info['true_label']
                start_time = sample_info['start_time']
                end_time = sample_info['end_time']
                file_path = sample_info['file_path']
                
                # 转换为tensor并预测
                events_tensor = torch.FloatTensor(events).unsqueeze(0).to(device)
                logits = model(events_tensor)
                probs = F.softmax(logits, dim=1)
                
                prediction = torch.argmax(logits, dim=1).item()
                confidence = torch.max(probs, dim=1)[0].item()
                
                # 计算样本的中心时间戳
                center_timestamp = (start_time + end_time) / 2
                
                all_predictions.append({
                    'timestamp': center_timestamp,
                    'prediction': prediction,
                    'confidence': confidence,
                    'probabilities': probs.cpu().numpy()[0],
                    'true_label': true_label,
                    'file_path': file_path,
                    'start_time': start_time,
                    'end_time': end_time
                })
        
        # 按时间窗口分组
        time_windows = self.group_predictions_by_time(all_predictions)
        
        # 在每个时间窗口内投票
        window_results = []
        for window_id in sorted(time_windows.keys()):
            window_preds = time_windows[window_id]
            
            # 获取窗口内的真实标签（应该是一致的）
            true_labels_in_window = [pred['true_label'] for pred in window_preds]
            if len(set(true_labels_in_window)) > 1:
                print(f"警告: 时间窗口 {window_id} 内有多个不同的真实标签: {set(true_labels_in_window)}")
            
            window_true_label = Counter(true_labels_in_window).most_common(1)[0][0]
            
            # 投票
            final_prediction, vote_confidence = self.vote_in_window(window_preds)
            
            if final_prediction is not None:
                window_results.append({
                    'window_id': window_id,
                    'window_start_time': window_id * self.time_window_sec,
                    'window_end_time': (window_id + 1) * self.time_window_sec,
                    'final_prediction': final_prediction,
                    'vote_confidence': vote_confidence,
                    'true_label': window_true_label,
                    'num_predictions': len(window_preds),
                    'individual_predictions': [pred['prediction'] for pred in window_preds],
                    'individual_confidences': [pred['confidence'] for pred in window_preds]
                })
        
        return window_results, all_predictions

def load_test_data_with_timestamps(data_path, label_map):
    """加载带时间戳的测试数据"""
    print(f"加载测试数据: {data_path}")
    
    test_data_with_time = []
    
    for class_name, class_idx in label_map.items():
        class_dir = os.path.join(data_path, class_name)
        if not os.path.exists(class_dir):
            continue
            
        for file_name in os.listdir(class_dir):
            if not file_name.endswith('.npy'):
                continue
                
            file_path = os.path.join(class_dir, file_name)
            events = np.load(file_path, allow_pickle=True)
            
            if events.size == 0 or len(events.shape) != 2 or events.shape[1] != 4:
                continue
            
            # 处理时间戳数据并创建滑动窗口
            window_size = 8192
            step_size = 1024
            
            # 时间归一化（转换为秒）
            timestamps = events[:, 0].copy()
            # 打印时间戳信息用于调试
            print(f"文件: {file_name}")
            print(f"原始时间戳 [0-10]: {timestamps[:11]}")
            timestamps_sec = (timestamps - timestamps[0]) / 1e6  # 假设原始时间戳是微秒
            
            # 归一化其他维度用于模型输入
            normalized_events = events.copy()
            t_normalized = (events[:, 0] - np.min(events[:, 0])) / (np.max(events[:, 0]) - np.min(events[:, 0]) + 1e-6)
            normalized_events[:, 0] = t_normalized
            
            # 创建滑动窗口
            for start_idx in range(0, len(events) - window_size, step_size):
                end_idx = start_idx + window_size
                
                window_events = normalized_events[start_idx:end_idx]
                start_time_sec = timestamps_sec[start_idx]
                end_time_sec = timestamps_sec[end_idx - 1]
                
                test_data_with_time.append({
                    'events': window_events,
                    'true_label': class_idx,
                    'start_time': start_time_sec,
                    'end_time': end_time_sec,
                    'file_path': file_path,
                    'class_name': class_name
                })
    
    print(f"总共加载了 {len(test_data_with_time)} 个样本")
    return test_data_with_time

def analyze_temporal_voting_results(window_results, all_predictions, class_names):
    """分析时间窗口投票结果"""
    
    if not window_results:
        print("没有时间窗口结果可分析")
        return
    
    # 计算基础指标
    window_predictions = [r['final_prediction'] for r in window_results]
    window_true_labels = [r['true_label'] for r in window_results]
    individual_predictions = [p['prediction'] for p in all_predictions]
    individual_true_labels = [p['true_label'] for p in all_predictions]
    
    # 准确率对比
    window_accuracy = np.mean(np.array(window_predictions) == np.array(window_true_labels))
    individual_accuracy = np.mean(np.array(individual_predictions) == np.array(individual_true_labels))
    
    print(f"\n=== 时间窗口投票结果分析 ===")
    print(f"时间窗口数量: {len(window_results)}")
    print(f"单样本预测数量: {len(all_predictions)}")
    print(f"单样本准确率: {individual_accuracy:.4f}")
    print(f"时间窗口投票准确率: {window_accuracy:.4f}")
    print(f"准确率提升: {window_accuracy - individual_accuracy:+.4f}")
    
    # 投票统计
    num_predictions_per_window = [r['num_predictions'] for r in window_results]
    print(f"\n=== 投票统计 ===")
    print(f"每窗口预测数量 - 平均: {np.mean(num_predictions_per_window):.1f}, "
          f"最小: {np.min(num_predictions_per_window)}, 最大: {np.max(num_predictions_per_window)}")
    
    # 分析置信度
    vote_confidences = [r['vote_confidence'] for r in window_results]
    individual_confidences = [p['confidence'] for p in all_predictions]
    
    print(f"\n=== 置信度分析 ===")
    print(f"单样本平均置信度: {np.mean(individual_confidences):.4f}")
    print(f"投票平均置信度: {np.mean(vote_confidences):.4f}")
    
    # 每类别准确率
    print(f"\n=== 各类别表现 ===")
    print(f"{'类别':<20} {'窗口数':<8} {'准确率':<10}")
    print("-" * 40)
    
    for class_idx, class_name in enumerate(class_names):
        class_mask = np.array(window_true_labels) == class_idx
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(np.array(window_predictions)[class_mask] == class_idx)
            class_count = np.sum(class_mask)
            print(f"{class_name:<20} {class_count:<8} {class_accuracy:<10.4f}")
    
    return {
        'window_accuracy': window_accuracy,
        'individual_accuracy': individual_accuracy,
        'accuracy_improvement': window_accuracy - individual_accuracy,
        'num_windows': len(window_results),
        'num_individual_predictions': len(all_predictions)
    }

def plot_voting_analysis(window_results, all_predictions, class_names, save_path):
    """绘制投票分析图表"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 准确率对比
    window_predictions = [r['final_prediction'] for r in window_results]
    window_true_labels = [r['true_label'] for r in window_results]
    individual_predictions = [p['prediction'] for p in all_predictions]
    individual_true_labels = [p['true_label'] for p in all_predictions]
    
    window_acc = np.mean(np.array(window_predictions) == np.array(window_true_labels))
    individual_acc = np.mean(np.array(individual_predictions) == np.array(individual_true_labels))
    
    axes[0, 0].bar(['单样本预测', '时间窗口投票'], [individual_acc, window_acc], 
                   color=['skyblue', 'lightcoral'])
    axes[0, 0].set_ylabel('准确率')
    axes[0, 0].set_title('准确率对比')
    axes[0, 0].set_ylim([0.9, 1.0])
    
    # 2. 投票数量分布
    num_votes = [r['num_predictions'] for r in window_results]
    axes[0, 1].hist(num_votes, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('每窗口预测数量')
    axes[0, 1].set_ylabel('窗口数量')
    axes[0, 1].set_title('每窗口投票数量分布')
    
    # 3. 置信度对比
    vote_confidences = [r['vote_confidence'] for r in window_results]
    individual_confidences = [p['confidence'] for p in all_predictions]
    
    axes[1, 0].hist(individual_confidences, bins=30, alpha=0.5, label='单样本置信度', color='skyblue')
    axes[1, 0].hist(vote_confidences, bins=30, alpha=0.5, label='投票置信度', color='lightcoral')
    axes[1, 0].set_xlabel('置信度')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].set_title('置信度分布对比')
    axes[1, 0].legend()
    
    # 4. 混淆矩阵
    cm = confusion_matrix(window_true_labels, window_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[1, 1])
    axes[1, 1].set_xlabel('预测类别')
    axes[1, 1].set_ylabel('真实类别')
    axes[1, 1].set_title('时间窗口投票混淆矩阵')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"分析图表已保存到: {save_path}")

def main(config_path, model_path, log_path=None, time_window_sec=1.0, voting_method='majority'):
    """主函数：时间窗口投票预测"""
    
    # 1. 加载配置
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    ds = cfg['dataset']
    model_type = cfg['model_type']
    class_names = list(ds['label_map'].keys())
    
    print(f"=== 时间窗口投票预测系统 ===")
    print(f"模型类型: {model_type}")
    print(f"时间窗口: {time_window_sec}秒")
    print(f"投票方法: {voting_method}")
    
    # 2. 加载带时间戳的测试数据
    test_data_with_time = load_test_data_with_timestamps(ds['test_dir'], ds['label_map'])
    
    # 3. 构建和加载模型
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    if model_type == 'pointnet2':
        model = PointNet2Classifier(cfg).to(device)
    elif model_type == 'pointnet2msg':
        model = PointNet2MSGClassifier(cfg).to(device)
    else:
        raise ValueError(f"当前只支持PointNet2模型，得到: {model_type}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"模型加载成功: {model_path}")
    
    # 4. 创建时间窗口投票预测器
    predictor = TemporalVotingPredictor(
        time_window_sec=time_window_sec,
        voting_method=voting_method,
        confidence_weight=False
    )
    
    # 5. 进行预测
    print("\n开始时间窗口投票预测...")
    start_time = time.time()
    window_results, all_predictions = predictor.predict_with_temporal_voting(
        model, test_data_with_time, device, class_names
    )
    prediction_time = time.time() - start_time
    
    # 6. 分析结果
    analysis_results = analyze_temporal_voting_results(window_results, all_predictions, class_names)
    
    # 7. 保存结果
    if log_path is None:
        log_path = os.path.join(cfg['test_log_dir'], f'temporal_voting_{voting_method}_{time_window_sec}s.txt')
    
    os.makedirs(cfg['test_log_dir'], exist_ok=True)
    
    with open(log_path, 'w') as f:
        f.write(f"=== 时间窗口投票预测结果 ===\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"模型类型: {model_type}\n")
        f.write(f"时间窗口: {time_window_sec}秒\n")
        f.write(f"投票方法: {voting_method}\n")
        f.write(f"预测时间: {prediction_time:.2f}秒\n\n")
        
        f.write(f"=== 主要指标 ===\n")
        f.write(f"时间窗口数量: {analysis_results['num_windows']}\n")
        f.write(f"单样本预测数量: {analysis_results['num_individual_predictions']}\n")
        f.write(f"单样本准确率: {analysis_results['individual_accuracy']:.4f}\n")
        f.write(f"时间窗口投票准确率: {analysis_results['window_accuracy']:.4f}\n")
        f.write(f"准确率提升: {analysis_results['accuracy_improvement']:+.4f}\n\n")
        
        f.write(f"=== 详细结果 ===\n")
        for i, result in enumerate(window_results[:10]):  # 只保存前10个详细结果
            f.write(f"窗口 {i+1}: 时间[{result['window_start_time']:.1f}s-{result['window_end_time']:.1f}s], "
                   f"真实标签={result['true_label']}, 预测={result['final_prediction']}, "
                   f"置信度={result['vote_confidence']:.3f}, 投票数={result['num_predictions']}\n")
    
    # 8. 绘制分析图表
    plot_path = os.path.join(cfg['test_fig_dir'], f'temporal_voting_analysis_{voting_method}_{time_window_sec}s.png')
    os.makedirs(cfg['test_fig_dir'], exist_ok=True)
    plot_voting_analysis(window_results, all_predictions, class_names, plot_path)
    
    print(f"\n=== 结果总结 ===")
    print(f"时间窗口投票准确率: {analysis_results['window_accuracy']:.4f}")
    print(f"单样本准确率: {analysis_results['individual_accuracy']:.4f}")
    print(f"准确率提升: {analysis_results['accuracy_improvement']:+.4f}")
    print(f"日志文件: {log_path}")
    print(f"分析图表: {plot_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='时间窗口投票预测系统')
    parser.add_argument('--config', type=str, default='configs/har_test_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--model', type=str, 
                        default='results/checkpoints/pointnet2_event_0628_8_ecount_11.pth',
                        help='模型文件路径')
    parser.add_argument('--log', type=str, default='results/test_logs/testlog_pointnet2_event_0628_8_ecount_11_vot1.txt',
                        help='日志文件路径')
    parser.add_argument('--time_window', type=float, default=1.0,
                        help='时间窗口大小（秒）')
    parser.add_argument('--voting_method', type=str, default='majority',
                        choices=['majority', 'weighted', 'confidence_weighted'],
                        help='投票方法')
    
    args = parser.parse_args()
    
    main(args.config, args.model, args.log, args.time_window, args.voting_method)
