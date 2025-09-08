#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的时间窗口投票预测系统
- 使用新的数据预处理逻辑 test_data_0628_8_ecount_3_vote.pkl
- 只使用简单多数投票
- 基于真实时间戳进行时间窗口分组
- 过滤掉混合标签的时间窗口

# 使用不同的时间窗口大小
python simple_temporal_voting.py --data preprocessing_data/test_data_0628_8_ecount_3_vote.pkl --time_window 0.5
python simple_temporal_voting.py --data preprocessing_data/test_data_0628_8_ecount_3_vote.pkl --time_window 1.0
python simple_temporal_voting.py --data preprocessing_data/test_data_0628_8_ecount_3_vote.pkl --time_window 2.0
"""

import os
import yaml
import time
import torch
import pickle
import numpy as np
from collections import defaultdict, Counter
import tqdm
import argparse
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 导入模型
from models.backbones.pointnet2_v1 import PointNet2Classifier
from models.backbones.pointnet2msg_v1 import PointNet2MSGClassifier
from datasets.event_count_seq_dataset_vote import ECountSeqDatasetVote

class SimpleTemporalVoting:
    """简化的时间窗口投票预测器 - 基于真实时间戳"""
    
    def __init__(self, time_window_sec=1.0):
        """
        Args:
            time_window_sec: 时间窗口大小（秒）
        """
        self.time_window_sec = time_window_sec
        self.time_window_microsec = time_window_sec * 1e6  # 转换为微秒
        
    def group_predictions_by_time_window(self, predictions_with_timestamps, log_path):
        """根据真实时间戳将预测结果分组到时间窗口"""
        time_windows = defaultdict(list)
        
        message = f"按 {self.time_window_sec}秒 时间窗口分组预测结果..."
        print(message)
        with open(log_path, 'a') as f:
            f.write(message + "\n")
        
        for pred_info in tqdm.tqdm(predictions_with_timestamps, desc="时间窗口分组"):
            timestamp_microsec = pred_info['center_timestamp']
            # 计算时间窗口ID
            window_id = int(timestamp_microsec // self.time_window_microsec)
            time_windows[window_id].append(pred_info)
        
        return time_windows
    
    def majority_vote_in_window(self, window_predictions, log_path):
        """在单个时间窗口内进行简单多数投票（已预先过滤混合标签窗口）"""
        if not window_predictions:
            return None, None, 0, 0
        
        # 收集所有预测
        votes = [pred['prediction'] for pred in window_predictions]
        true_labels = [pred['true_label'] for pred in window_predictions]
        
        # 简单多数投票
        vote_counts = Counter(votes)
        final_prediction = vote_counts.most_common(1)[0][0]
        vote_count = vote_counts[final_prediction]
        
        # 获取窗口的真实标签（现在应该都一致）
        window_true_label = true_labels[0]  # 所有标签都一致，取第一个即可
        
        return final_prediction, window_true_label, len(votes), vote_count
    
    def predict_with_vote_dataset(self, model, vote_dataset, device, class_names, log_path):
        """使用 vote dataset 进行时间窗口投票预测"""
        model.eval()
        
        all_predictions = []
        
        message = f"开始对 {len(vote_dataset)} 个样本进行预测..."
        print(message)
        with open(log_path, 'a') as f:
            f.write(message + "\n")
        
        # 进行单样本预测
        with torch.no_grad():
            for i in tqdm.tqdm(range(len(vote_dataset)), desc="单样本预测"):
                # 获取样本数据 - 新格式
                events, label, start_ts, end_ts, center_ts, file_path, window_duration, class_name = vote_dataset[i]
                
                # 转换为tensor并预测
                events_tensor = torch.FloatTensor(events).unsqueeze(0).to(device)
                logits = model(events_tensor)
                prediction = torch.argmax(logits, dim=1).item()
                
                # 保存预测结果
                all_predictions.append({
                    'center_timestamp': center_ts,
                    'prediction': prediction,
                    'true_label': label,
                    'file_path': file_path,
                    'class_name': class_name,
                    'window_duration_ms': window_duration / 1000,
                    'start_timestamp': start_ts,
                    'end_timestamp': end_ts
                })
        
        # 按时间窗口分组
        time_windows = self.group_predictions_by_time_window(all_predictions, log_path)
        
        # 在每个时间窗口内投票
        window_results = []
        mixed_label_windows = 0
        
        message = f"在 {len(time_windows)} 个时间窗口内进行多数投票..."
        print(message)
        with open(log_path, 'a') as f:
            f.write(message + "\n")
        
        for window_id in tqdm.tqdm(sorted(time_windows.keys()), desc="时间窗口投票"):
            window_preds = time_windows[window_id]
            
            # 检查是否有混合标签，如果有则跳过该窗口
            unique_labels = set([pred['true_label'] for pred in window_preds])
            if len(unique_labels) > 1:
                mixed_label_windows += 1
                warning_msg = f"警告：时间窗口 {window_id} 内有混合标签 {unique_labels}，跳过该窗口"
                print(warning_msg)
                with open(log_path, 'a') as f:
                    f.write(warning_msg + "\n")
                continue  # 跳过混合标签窗口
            
            # 多数投票
            final_prediction, window_true_label, num_predictions, vote_count = self.majority_vote_in_window(window_preds, log_path)
            
            if final_prediction is not None:
                window_results.append({
                    'window_id': window_id,
                    'final_prediction': final_prediction,
                    'true_label': window_true_label,
                    'num_predictions': num_predictions,
                    'vote_count': vote_count,
                    'vote_ratio': vote_count / num_predictions,
                    'is_correct': final_prediction == window_true_label,
                    'has_mixed_labels': False,  # 现在所有保留的窗口都没有混合标签
                    'window_start_time': min([pred['center_timestamp'] for pred in window_preds]),
                    'window_end_time': max([pred['center_timestamp'] for pred in window_preds])
                })
        
        mixed_message = f"混合标签的时间窗口数: {mixed_label_windows}/{len(time_windows)} ({100*mixed_label_windows/len(time_windows):.1f}%)"
        removed_message = f"移除混合标签窗口后，剩余有效窗口数: {len(window_results)}/{len(time_windows)}"
        print(mixed_message)
        print(removed_message)
        with open(log_path, 'a') as f:
            f.write(mixed_message + "\n")
            f.write(removed_message + "\n")
        
        return window_results, all_predictions, mixed_label_windows, len(time_windows)

def analyze_voting_results(window_results, all_predictions, class_names, log_path, cfg, model_path, time_window_sec, timestamp, mixed_label_windows=0, total_windows=0):
    """分析时间窗口投票结果"""
    
    if not window_results:
        print("没有时间窗口结果可分析")
        with open(log_path, 'a') as f:
            f.write("没有时间窗口结果可分析\n")
        return
    
    # 计算基础指标
    window_predictions = [r['final_prediction'] for r in window_results]
    window_true_labels = [r['true_label'] for r in window_results]
    individual_predictions = [p['prediction'] for p in all_predictions]
    individual_true_labels = [p['true_label'] for p in all_predictions]
    
    # 准确率对比
    window_accuracy = np.mean(np.array(window_predictions) == np.array(window_true_labels))
    individual_accuracy = np.mean(np.array(individual_predictions) == np.array(individual_true_labels))
    
    # 输出到控制台和日志
    analysis_text = f"\n=== 时间窗口投票结果分析 ===\n"
    analysis_text += f"时间窗口数量: {len(window_results)}\n"
    analysis_text += f"单样本预测数量: {len(all_predictions)}\n"
    analysis_text += f"单样本准确率: {individual_accuracy:.4f} ({individual_accuracy*100:.2f}%)\n"
    analysis_text += f"时间窗口投票准确率: {window_accuracy:.4f} ({window_accuracy*100:.2f}%)\n"
    analysis_text += f"准确率提升: {window_accuracy - individual_accuracy:+.4f} ({(window_accuracy - individual_accuracy)*100:+.2f}%)\n"
    
    print(analysis_text)
    with open(log_path, 'a') as f:
        f.write(analysis_text)
    
    # 投票统计
    num_predictions_per_window = [r['num_predictions'] for r in window_results]
    vote_ratios = [r['vote_ratio'] for r in window_results]
    
    voting_stats = f"\n=== 投票统计 ===\n"
    voting_stats += f"每窗口预测数量 - 平均: {np.mean(num_predictions_per_window):.1f}, "
    voting_stats += f"最小: {np.min(num_predictions_per_window)}, 最大: {np.max(num_predictions_per_window)}\n"
    voting_stats += f"投票一致性 - 平均: {np.mean(vote_ratios):.3f}, "
    voting_stats += f"最小: {np.min(vote_ratios):.3f}, 最大: {np.max(vote_ratios):.3f}\n"
    
    print(voting_stats)
    with open(log_path, 'a') as f:
        f.write(voting_stats)
    
    # 检查混合标签情况
    mixed_info = f"混合标签窗口: {mixed_label_windows}/{total_windows} ({100*mixed_label_windows/total_windows:.1f}%) - 已移除\n"
    valid_info = f"有效分析窗口: {len(window_results)}/{total_windows} ({100*len(window_results)/total_windows:.1f}%)\n"
    
    print(mixed_info.strip())
    print(valid_info.strip())
    with open(log_path, 'a') as f:
        f.write(mixed_info)
        f.write(valid_info)
    
    # 每类别准确率
    class_performance = f"\n=== 多数投票后各类别表现 ===\n"
    class_performance += f"{'类别':<20} {'窗口数':<8} {'准确率':<10}\n"
    class_performance += "-" * 40 + "\n"

    print(f"\n=== 多数投票后各类别表现 ===")
    print(f"{'类别':<20} {'窗口数':<8} {'准确率':<10}")
    print("-" * 40)
    
    for class_idx, class_name in enumerate(class_names):
        class_mask = np.array(window_true_labels) == class_idx
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(np.array(window_predictions)[class_mask] == class_idx)
            class_count = np.sum(class_mask)
            class_line = f"{class_name:<20} {class_count:<8} {class_accuracy:<10.4f}\n"
            print(f"{class_name:<20} {class_count:<8} {class_accuracy:<10.4f}")
            class_performance += class_line
    
    with open(log_path, 'a') as f:
        f.write(class_performance)
    
    # 单样本各类别准确率
    individual_class_performance = f"\n=== 单样本各类别表现 ===\n"
    individual_class_performance += f"{'类别':<20} {'样本数':<8} {'准确率':<10}\n"
    individual_class_performance += "-" * 40 + "\n"
    
    print(f"\n=== 单样本各类别表现 ===")
    print(f"{'类别':<20} {'样本数':<8} {'准确率':<10}")
    print("-" * 40)
    
    for class_idx, class_name in enumerate(class_names):
        class_mask = np.array(individual_true_labels) == class_idx
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(np.array(individual_predictions)[class_mask] == class_idx)
            class_count = np.sum(class_mask)
            class_line = f"{class_name:<20} {class_count:<8} {class_accuracy:<10.4f}\n"
            print(f"{class_name:<20} {class_count:<8} {class_accuracy:<10.4f}")
            individual_class_performance += class_line
    
    with open(log_path, 'a') as f:
        f.write(individual_class_performance)
    
    # 计算和保存混淆矩阵
    os.makedirs(cfg['test_fig_dir'], exist_ok=True)
    best_model_filename = os.path.basename(model_path).split('.')[0]
    
    # 1. 窗口级别混淆矩阵
    window_cm = confusion_matrix(window_true_labels, window_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(window_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Window-level Confusion Matrix')

    window_cm_path = os.path.join(cfg['test_fig_dir'], f"{best_model_filename}_window_confusion_matrix_time_window_sec_{time_window_sec}s_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(window_cm_path)
    plt.close()
    print(f"窗口级混淆矩阵保存到: {window_cm_path}")
    
    # 2. 窗口级别归一化混淆矩阵
    window_cm_normalized = window_cm.astype('float') / window_cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(window_cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Window-level Normalized Confusion Matrix')

    window_norm_cm_path = os.path.join(cfg['test_fig_dir'], f"{best_model_filename}_window_norm_confusion_matrix_{time_window_sec}s_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(window_norm_cm_path)
    plt.close()
    print(f"窗口级归一化混淆矩阵保存到: {window_norm_cm_path}")
    
    # 3. 单样本级别混淆矩阵
    individual_cm = confusion_matrix(individual_true_labels, individual_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(individual_cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Individual Sample Confusion Matrix')

    individual_cm_path = os.path.join(cfg['test_fig_dir'], f"{best_model_filename}_individual_confusion_matrix_{time_window_sec}s_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(individual_cm_path)
    plt.close()
    print(f"单样本级混淆矩阵保存到: {individual_cm_path}")
    
    # 4. 单样本级别归一化混淆矩阵
    individual_cm_normalized = individual_cm.astype('float') / individual_cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(individual_cm_normalized, annot=True, fmt='.2f', cmap='Reds', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Individual Sample Normalized Confusion Matrix')

    individual_norm_cm_path = os.path.join(cfg['test_fig_dir'], f"{best_model_filename}_individual_norm_confusion_matrix_{time_window_sec}s_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(individual_norm_cm_path)
    plt.close()
    print(f"单样本级归一化混淆矩阵保存到: {individual_norm_cm_path}")
    
    # 将混淆矩阵输出到日志文件
    with open(log_path, 'a') as f:
        f.write("\n=== 窗口级混淆矩阵 ===\n")
        np.savetxt(f, window_cm, fmt='%d', delimiter=',')
        f.write("\n=== 窗口级归一化混淆矩阵 ===\n")
        np.savetxt(f, window_cm_normalized, fmt='%.2f', delimiter=',')
        f.write("\n=== 单样本级混淆矩阵 ===\n")
        np.savetxt(f, individual_cm, fmt='%d', delimiter=',')
        f.write("\n=== 单样本级归一化混淆矩阵 ===\n")
        np.savetxt(f, individual_cm_normalized, fmt='%.2f', delimiter=',')
        f.write(f"\n混淆矩阵图片保存路径:\n")
        f.write(f"窗口级: {window_cm_path}\n")
        f.write(f"窗口级归一化: {window_norm_cm_path}\n")
        f.write(f"单样本级: {individual_cm_path}\n")
        f.write(f"单样本级归一化: {individual_norm_cm_path}\n")
    
    return {
        'window_accuracy': window_accuracy,
        'individual_accuracy': individual_accuracy,
        'accuracy_improvement': window_accuracy - individual_accuracy,
        'num_windows': len(window_results),
        'num_individual_predictions': len(all_predictions)
    }

def main(config_path, model_path, vote_data_path, log_path=None, time_window_sec=1.0):
    """主函数：基于新预处理数据的时间窗口投票预测"""
    
    # 1. 加载配置
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    ds = cfg['dataset']
    model_type = cfg['model_type']
    class_names = list(ds['label_map'].keys())
    
    # 设置日志路径
    if log_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(cfg['test_log_dir'], f'voting_{time_window_sec}s_{timestamp}.txt')
    
    os.makedirs(cfg['test_log_dir'], exist_ok=True)
    
    # 初始化日志
    with open(log_path, 'w') as f:
        f.write(f"=== 时间窗口投票预测系统 ===\n")
        f.write(f"模型类型: {model_type}\n")
        f.write(f"时间窗口: {time_window_sec}秒\n")
        f.write(f"投票方法: 简单多数投票\n")
        f.write(f"数据来源: {vote_data_path}\n")
        f.write(f"配置文件: {config_path}\n")
        f.write(f"模型文件: {model_path}\n")
        f.write(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    print(f"=== 时间窗口投票预测系统 ===")
    print(f"模型类型: {model_type}")
    print(f"时间窗口: {time_window_sec}秒")
    print(f"投票方法: 简单多数投票")
    print(f"数据来源: {vote_data_path}")
    
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    device_info = f"使用设备: {device}"
    print(device_info)
    with open(log_path, 'a') as f:
        f.write(device_info + "\n")
    
    # 2. 构建和加载模型
    if model_type == 'pointnet2':
        model = PointNet2Classifier(cfg).to(device)
    elif model_type == 'pointnet2msg':
        model = PointNet2MSGClassifier(cfg).to(device)
    else:
        raise ValueError(f"当前只支持PointNet2模型，得到: {model_type}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    load_msg = f"模型加载成功: {model_path}"
    print(load_msg)
    with open(log_path, 'a') as f:
        f.write(load_msg + "\n")
    
    # 3. 加载 vote dataset
    if not os.path.exists(vote_data_path):
        raise FileNotFoundError(f"Vote数据文件不存在: {vote_data_path}")
    
    load_dataset_msg = f"加载 vote dataset: {vote_data_path}"
    print(f"\n{load_dataset_msg}")
    with open(log_path, 'a') as f:
        f.write(f"\n{load_dataset_msg}\n")
    
    with open(vote_data_path, 'rb') as f:
        vote_dataset = pickle.load(f)
    
    dataset_info = f"加载了 {len(vote_dataset)} 个带时间戳的样本"
    print(dataset_info)
    with open(log_path, 'a') as f:
        f.write(dataset_info + "\n")
    
    # 4. 创建投票预测器
    predictor = SimpleTemporalVoting(time_window_sec=time_window_sec)
    
    # 5. 进行预测
    start_time = time.time()
    window_results, all_predictions, mixed_label_windows, total_windows = predictor.predict_with_vote_dataset(
        model, vote_dataset, device, class_names, log_path
    )
    prediction_time = time.time() - start_time
    
    time_info = f"预测完成，耗时: {prediction_time:.2f}秒"
    print(time_info)
    with open(log_path, 'a') as f:
        f.write(time_info + "\n")
    
    # 6. 分析结果
    analysis_results = analyze_voting_results(window_results, all_predictions, class_names, log_path, cfg, model_path, time_window_sec, timestamp, mixed_label_windows, total_windows)
    
    # 7. 更新日志文件
    with open(log_path, 'a') as f:
        f.write(f"\n=== 主要指标 ===\n")
        f.write(f"时间窗口数量: {analysis_results['num_windows']}\n")
        f.write(f"单样本预测数量: {analysis_results['num_individual_predictions']}\n")
        f.write(f"单样本准确率: {analysis_results['individual_accuracy']:.4f}\n")
        f.write(f"时间窗口投票准确率: {analysis_results['window_accuracy']:.4f}\n")
        f.write(f"准确率提升: {analysis_results['accuracy_improvement']:+.4f}\n")
        f.write(f"预测时间: {prediction_time:.2f}秒\n")
        f.write(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    result_summary = f"\n=== 结果总结 ==="
    result_summary += f"\n时间窗口投票准确率: {analysis_results['window_accuracy']:.4f}"
    result_summary += f"\n单样本准确率: {analysis_results['individual_accuracy']:.4f}"
    result_summary += f"\n准确率提升: {analysis_results['accuracy_improvement']:+.4f}"
    result_summary += f"\n日志文件: {log_path}"
    
    print(result_summary)
    with open(log_path, 'a') as f:
        f.write(result_summary + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='时间窗口投票预测系统')
    parser.add_argument('--config', type=str, default='configs/har_test_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--model', type=str, 
                        default='results/checkpoints/pointnet2_event_0628_8_ecount_11.pth',
                        help='模型文件路径')
    parser.add_argument('--data', type=str, 
                        default='preprocessing_data/test_data_0628_8_ecount_3_vote.pkl',
                        help='预处理数据文件路径')
    parser.add_argument('--log', type=str, default=None,
                        help='日志文件路径')
    parser.add_argument('--time_window', type=float, default=1.0,
                        help='时间窗口大小（秒）')
    
    args = parser.parse_args()
    
    main(args.config, args.model, args.data, args.log, args.time_window)
