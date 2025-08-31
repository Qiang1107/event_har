#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import pickle
import argparse
import numpy as np
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import yaml

from datasets.event_count_seq_dataset import ECountSeqDataset  # 点云类型数据集
from datasets.rgbe_sequence_dataset import RGBESequenceDataset      # RGB类型数据集

from models.backbones.vitmodel import VitModel
from models.backbones.cnn import CNN_model
from models.backbones.resnet import ResNet_model
from models.backbones.resnet_pretrained import PretrainedResNet_model

from models.backbones.pointnet2_v1 import PointNet2Classifier
from models.backbones.pointnet2msg_v1 import PointNet2MSGClassifier

class ConfidenceBasedPredictor:
    """置信度预测器，支持多种置信度评估策略"""
    
    def __init__(self, confidence_method='max_prob', confidence_threshold=0.9, 
                 temperature=1.0, ensemble_size=None):
        """
        Args:
            confidence_method: 置信度计算方法
                - 'max_prob': 使用最大softmax概率
                - 'entropy': 使用熵的负值作为置信度
                - 'temperature': 使用温度缩放的softmax
                - 'top_k_gap': 使用前k个概率的差值
            confidence_threshold: 置信度阈值，低于此值的预测将被拒绝
            temperature: 温度缩放参数
            ensemble_size: 集成模型数量（如果使用集成方法）
        """
        self.confidence_method = confidence_method
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
        self.ensemble_size = ensemble_size
        
    def compute_confidence(self, logits):
        """计算预测置信度"""
        # 应用温度缩放
        if self.temperature != 1.0:
            logits = logits / self.temperature
            
        probs = F.softmax(logits, dim=1)
        
        if self.confidence_method == 'max_prob':
            # 使用最大概率作为置信度
            confidence, predictions = torch.max(probs, dim=1)
            
        elif self.confidence_method == 'entropy':
            # 使用熵的负值作为置信度（熵越低，置信度越高）
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            max_entropy = np.log(probs.size(1))  # 最大可能熵
            confidence = 1 - (entropy / max_entropy)  # 归一化到[0,1]
            predictions = torch.argmax(probs, dim=1)
            
        elif self.confidence_method == 'temperature':
            # 使用温度缩放后的最大概率
            confidence, predictions = torch.max(probs, dim=1)
            
        elif self.confidence_method == 'top_k_gap':
            # 使用前两个最高概率的差值作为置信度
            top_k_probs, top_k_indices = torch.topk(probs, k=2, dim=1)
            confidence = top_k_probs[:, 0] - top_k_probs[:, 1]
            predictions = top_k_indices[:, 0]
            
        else:
            raise ValueError(f"Unsupported confidence method: {self.confidence_method}")
            
        return predictions, confidence, probs
    
    def filter_predictions(self, predictions, confidence, labels=None):
        """根据置信度阈值过滤预测"""
        # 高置信度的预测mask
        high_conf_mask = confidence >= self.confidence_threshold
        
        # 过滤后的预测和标签
        filtered_preds = predictions[high_conf_mask]
        filtered_confidence = confidence[high_conf_mask]
        
        if labels is not None:
            filtered_labels = labels[high_conf_mask]
            return filtered_preds, filtered_labels, filtered_confidence, high_conf_mask
        else:
            return filtered_preds, filtered_confidence, high_conf_mask

def evaluate_with_confidence(model, test_loader, confidence_predictor, device, class_names):
    """使用置信度预测器进行评估"""
    model.eval()
    
    # 存储所有预测、标签和置信度
    all_predictions = []
    all_labels = []
    all_confidence = []
    all_probs = []
    
    # 统计信息
    total_samples = 0
    high_conf_samples = 0
    high_conf_correct = 0
    all_correct = 0
    
    class_stats = {name: {'total': 0, 'high_conf_total': 0, 'high_conf_correct': 0, 'all_correct': 0} 
                   for name in class_names}
    
    with torch.no_grad():
        for imgs, labels in tqdm.tqdm(test_loader, desc="Evaluating with confidence"):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            
            # 前向传播
            logits = model(imgs)
            
            # 计算置信度和预测
            predictions, confidence, probs = confidence_predictor.compute_confidence(logits)
            
            # 过滤高置信度预测
            filtered_preds, filtered_labels, filtered_conf, high_conf_mask = \
                confidence_predictor.filter_predictions(predictions, confidence, labels)
            
            # 存储结果
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidence.extend(confidence.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # 更新统计信息
            batch_size = labels.size(0)
            total_samples += batch_size
            high_conf_samples += high_conf_mask.sum().item()
            
            # 计算准确率
            all_batch_correct = (predictions == labels).sum().item()
            all_correct += all_batch_correct
            
            if len(filtered_preds) > 0:
                high_conf_batch_correct = (filtered_preds == filtered_labels).sum().item()
                high_conf_correct += high_conf_batch_correct
            
            # 更新每类统计
            for i in range(batch_size):
                label_idx = labels[i].item()
                pred_idx = predictions[i].item()
                label_name = class_names[label_idx]
                
                class_stats[label_name]['total'] += 1
                if predictions[i] == labels[i]:
                    class_stats[label_name]['all_correct'] += 1
                
                if high_conf_mask[i]:
                    class_stats[label_name]['high_conf_total'] += 1
                    if predictions[i] == labels[i]:
                        class_stats[label_name]['high_conf_correct'] += 1
    
    # 计算整体指标
    overall_accuracy = all_correct / total_samples
    coverage = high_conf_samples / total_samples
    high_conf_accuracy = high_conf_correct / high_conf_samples if high_conf_samples > 0 else 0
    
    results = {
        'all_predictions': np.array(all_predictions),
        'all_labels': np.array(all_labels),
        'all_confidence': np.array(all_confidence),
        'all_probs': np.array(all_probs),
        'overall_accuracy': overall_accuracy,
        'high_conf_accuracy': high_conf_accuracy,
        'coverage': coverage,
        'total_samples': total_samples,
        'high_conf_samples': high_conf_samples,
        'class_stats': class_stats
    }
    
    return results

def analyze_confidence_threshold_curve(results, class_names, thresholds=None):
    """分析不同置信度阈值下的性能"""
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)
    
    all_confidence = results['all_confidence']
    all_predictions = results['all_predictions']
    all_labels = results['all_labels']
    
    threshold_results = []
    
    for threshold in thresholds:
        # 高置信度样本mask
        high_conf_mask = all_confidence >= threshold
        
        if high_conf_mask.sum() == 0:
            continue
            
        # 过滤后的预测和标签
        filtered_preds = all_predictions[high_conf_mask]
        filtered_labels = all_labels[high_conf_mask]
        
        # 计算指标
        coverage = high_conf_mask.sum() / len(all_labels)
        accuracy = (filtered_preds == filtered_labels).sum() / len(filtered_preds)
        
        threshold_results.append({
            'threshold': threshold,
            'coverage': coverage,
            'accuracy': accuracy,
            'samples': high_conf_mask.sum()
        })
    
    return threshold_results

def plot_confidence_analysis(threshold_results, save_path):
    """绘制置信度分析图"""
    thresholds = [r['threshold'] for r in threshold_results]
    coverages = [r['coverage'] for r in threshold_results]
    accuracies = [r['accuracy'] for r in threshold_results]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. 准确率-覆盖率曲线
    ax1.plot(coverages, accuracies, 'b-o', markersize=4)
    ax1.set_xlabel('Coverage')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs Coverage')
    ax1.grid(True, alpha=0.3)
    
    # 2. 置信度阈值 vs 准确率
    ax2.plot(thresholds, accuracies, 'r-o', markersize=4)
    ax2.set_xlabel('Confidence Threshold')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Confidence Threshold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 置信度阈值 vs 覆盖率
    ax3.plot(thresholds, coverages, 'g-o', markersize=4)
    ax3.set_xlabel('Confidence Threshold')
    ax3.set_ylabel('Coverage')
    ax3.set_title('Coverage vs Confidence Threshold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Confidence analysis plot saved to {save_path}")

def main(config_path, premodel_path, log_path=None, 
         confidence_method='max_prob', confidence_threshold=0.9, temperature=1.0):
    """主函数：增强版置信度测试"""
    
    # 1. 加载配置
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    ds = cfg['dataset']
    model_type = cfg['model_type']
    
    print(f"=== 置信度增强测试 ===")
    print(f"模型类型: {model_type}")
    print(f"置信度方法: {confidence_method}")
    print(f"置信度阈值: {confidence_threshold}")
    print(f"温度参数: {temperature}")
    
    # 2. 加载数据集
    if model_type in ['pointnet2', 'pointnet2msg']:
        # 对于PointNet2，从预处理的pickle文件加载
        pnet2_data_dir = "preprocessing_data"
        pnet2_test_pkl = "test_data_0628_8_ecount_3.pkl"
        pnet2_test_path = os.path.join(pnet2_data_dir, pnet2_test_pkl)
        
        if not os.path.exists(pnet2_test_path):
            raise FileNotFoundError(f"PointNet2 test data not found: {pnet2_test_path}")
        
        print(f"Start loading test dataset from {pnet2_test_path}")
        with open(pnet2_test_path, 'rb') as f:
            test_data = pickle.load(f)
        
        print(f"Loaded test dataset with {len(test_data)} samples")
        test_loader = DataLoader(test_data, **cfg['test'])
        print("DataLoader created for PointNet2 datasets.")
        
    else:
        test_ds = RGBESequenceDataset(
            data_root=ds['test_dir'],
            window_size=ds['window_size'],
            stride=ds['stride'],
            enable_transform=ds['enable_transform'],
            label_map=ds['label_map']
        )
        test_loader = DataLoader(test_ds, **cfg['test'])

    # 3. 构建模型
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    if model_type == 'vit':
        model = VitModel(cfg).to(device)
    elif model_type == 'pointnet2':
        model = PointNet2Classifier(cfg).to(device)
    elif model_type == 'pointnet2msg':
        model = PointNet2MSGClassifier(cfg).to(device)
    elif model_type == 'cnn':
        model = CNN_model(cfg).to(device)
    elif model_type == 'resnet':
        model = ResNet_model(cfg).to(device)
    elif model_type == 'resnet_pretrained':
        model = PretrainedResNet_model(cfg).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # 4. 加载训练好的模型
    if not os.path.exists(premodel_path):
        raise FileNotFoundError(f"Model file not found at {premodel_path}")
    
    model.load_state_dict(torch.load(premodel_path, map_location=device))
    print(f"Loaded model from {premodel_path}")

    # 5. 创建置信度预测器
    confidence_predictor = ConfidenceBasedPredictor(
        confidence_method=confidence_method,
        confidence_threshold=confidence_threshold,
        temperature=temperature
    )

    # 6. 创建日志目录
    if log_path is None:
        log_path = os.path.join(cfg['test_log_dir'], f'confidence_testing_{confidence_method}_{confidence_threshold}.txt')
    os.makedirs(cfg['test_log_dir'], exist_ok=True)
    os.makedirs(cfg['test_fig_dir'], exist_ok=True)

    # 7. 进行置信度评估
    class_names = list(ds['label_map'].keys())
    print("\n=== 开始置信度评估 ===")
    
    start_time = time.time()
    results = evaluate_with_confidence(model, test_loader, confidence_predictor, device, class_names)
    eval_time = time.time() - start_time
    
    # 8. 输出结果
    print(f"\n=== 评估结果 ===")
    print(f"总样本数: {results['total_samples']}")
    print(f"高置信度样本数: {results['high_conf_samples']}")
    print(f"覆盖率: {results['coverage']:.4f} ({results['high_conf_samples']}/{results['total_samples']})")
    print(f"整体准确率: {results['overall_accuracy']:.4f}")
    print(f"高置信度准确率: {results['high_conf_accuracy']:.4f}")
    print(f"评估时间: {eval_time:.2f}秒")
    
    # 9. 分析不同置信度阈值
    print("\n=== 置信度阈值分析 ===")
    threshold_results = analyze_confidence_threshold_curve(results, class_names)
    
    # 找出最佳阈值（平衡准确率和覆盖率）
    best_balance = max(threshold_results, key=lambda x: x['accuracy'] * x['coverage'])
    print(f"最佳平衡点: 阈值={best_balance['threshold']:.2f}, "
          f"准确率={best_balance['accuracy']:.4f}, 覆盖率={best_balance['coverage']:.4f}")
    
    # 找出达到99%准确率的最低阈值
    high_acc_results = [r for r in threshold_results if r['accuracy'] >= 0.99]
    if high_acc_results:
        best_high_acc = min(high_acc_results, key=lambda x: x['threshold'])
        print(f"99%准确率最低阈值: 阈值={best_high_acc['threshold']:.2f}, "
              f"准确率={best_high_acc['accuracy']:.4f}, 覆盖率={best_high_acc['coverage']:.4f}")
    
    # 10. 保存结果到日志
    with open(log_path, 'w') as f:
        f.write(f"=== 置信度增强测试结果 ===\n")
        f.write(f"模型路径: {premodel_path}\n")
        f.write(f"模型类型: {model_type}\n")
        f.write(f"置信度方法: {confidence_method}\n")
        f.write(f"置信度阈值: {confidence_threshold}\n")
        f.write(f"温度参数: {temperature}\n")
        f.write(f"评估时间: {eval_time:.2f}秒\n\n")
        
        f.write(f"=== 整体结果 ===\n")
        f.write(f"总样本数: {results['total_samples']}\n")
        f.write(f"高置信度样本数: {results['high_conf_samples']}\n")
        f.write(f"覆盖率: {results['coverage']:.4f}\n")
        f.write(f"整体准确率: {results['overall_accuracy']:.4f}\n")
        f.write(f"高置信度准确率: {results['high_conf_accuracy']:.4f}\n\n")
        
        f.write(f"=== 各类别结果 ===\n")
        for class_name in class_names:
            stats = results['class_stats'][class_name]
            if stats['total'] > 0:
                all_acc = stats['all_correct'] / stats['total']
                if stats['high_conf_total'] > 0:
                    high_conf_acc = stats['high_conf_correct'] / stats['high_conf_total']
                    coverage = stats['high_conf_total'] / stats['total']
                    f.write(f"{class_name}: 整体准确率={all_acc:.4f} ({stats['all_correct']}/{stats['total']}), "
                           f"高置信度准确率={high_conf_acc:.4f} ({stats['high_conf_correct']}/{stats['high_conf_total']}), "
                           f"覆盖率={coverage:.4f}\n")
                else:
                    f.write(f"{class_name}: 整体准确率={all_acc:.4f} ({stats['all_correct']}/{stats['total']}), "
                           f"无高置信度样本\n")
        
        f.write(f"\n=== 置信度阈值分析 ===\n")
        f.write("阈值\t准确率\t覆盖率\t样本数\n")
        for r in threshold_results:
            f.write(f"{r['threshold']:.2f}\t{r['accuracy']:.4f}\t{r['coverage']:.4f}\t{r['samples']}\n")
    
    # 11. 绘制和保存分析图
    model_name = os.path.basename(premodel_path).split('.')[0]
    plot_path = os.path.join(cfg['test_fig_dir'], 
                            f"{model_name}_confidence_analysis_{confidence_method}_{confidence_threshold}.png")
    plot_confidence_analysis(threshold_results, plot_path)
    
    # 12. 绘制高置信度样本的混淆矩阵
    high_conf_mask = results['all_confidence'] >= confidence_threshold
    if high_conf_mask.sum() > 0:
        filtered_preds = results['all_predictions'][high_conf_mask]
        filtered_labels = results['all_labels'][high_conf_mask]
        
        cm = confusion_matrix(filtered_labels, filtered_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'High Confidence Confusion Matrix (threshold={confidence_threshold})')
        
        cm_path = os.path.join(cfg['test_fig_dir'], 
                              f"{model_name}_high_conf_confusion_matrix_{confidence_method}_{confidence_threshold}.png")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"High confidence confusion matrix saved to {cm_path}")
    
    print(f"\n=== 测试完成 ===")
    print(f"日志文件: {log_path}")
    print(f"图表文件: {plot_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='置信度增强的模型测试')
    parser.add_argument('--config', type=str, default='configs/har_test_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--model', type=str, 
                        default='results/checkpoints/pointnet2_event_0628_8_ecount_11.pth',
                        help='预训练模型路径')
    parser.add_argument('--log', type=str, default='results/test_logs/testlog_pointnet2_event_0628_8_ecount_11_cfd1.txt',
                        help='日志文件路径')
    parser.add_argument('--confidence_method', type=str, default='max_prob',
                        choices=['max_prob', 'entropy', 'temperature', 'top_k_gap'],
                        help='置信度计算方法')
    parser.add_argument('--confidence_threshold', type=float, default=0.9,
                        help='置信度阈值')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='温度缩放参数')
    
    args = parser.parse_args()
    
    main(args.config, args.model, args.log, 
         args.confidence_method, args.confidence_threshold, args.temperature)
