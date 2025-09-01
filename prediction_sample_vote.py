"""
测试评估脚本，针对 events, label, start_ts, end_ts, center_ts, file_path, window_duration, class_name 数据进行评估
单样本评估
保存测试日志和混淆矩阵图像
"""


import os
import yaml
import time
import torch
import torch.nn as nn
import tqdm
import pickle
import sys
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sys.path.append(os.path.dirname(__file__))
from torch.utils.data import DataLoader
from datasets.rgbe_sequence_dataset import RGBESequenceDataset
from datasets.rgb_sequence_dataset import RGBSequenceDataset
from datasets.event_sequence_dataset import ESequenceDataset
from datasets.event_count_seq_dataset import ECountSeqDataset
from datasets.event_count_seq_dataset_vote import ECountSeqDatasetVote
from models.backbones.vitmodel import VitModel
from models.backbones.cnn import CNN_model
from models.backbones.resnet import ResNet_model
from models.backbones.resnet_pretrained import PretrainedResNet_model


from models.backbones.pointnet2_v1 import PointNet2Classifier
# from models.backbones.pointnet2_v2 import PointNet2Classifier
# from models.backbones.pointnet2_v3 import PointNet2Classifier
from models.backbones.pointnet2msg_v1 import PointNet2MSGClassifier
# from models.backbones.pointnet2msg_v2 import PointNet2MSGClassifier
# from models.losses.cross_entropy_loss import CrossEntropyLoss
from utils.weight_utils import load_vitpose_pretrained


def main(config_path, premodel_path, log_path):
    # 1. 加载配置
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    model_type = cfg.get('model_type', 'cnn')

    # 2. 构造 Dataset & DataLoader
    ds = cfg['dataset']
    if model_type in ['pointnet2', 'pointnet2msg']:
        pnet2_data_dir = "preprocessing_data"
        pnet2_test_pkl = "test_data_0628_8_ecount_3_vote.pkl"
        pnet2_test_path = os.path.join(pnet2_data_dir, pnet2_test_pkl)
        print(f"Start loading test dataset from {pnet2_test_path}")
        with open(pnet2_test_path, 'rb') as f:
            test_ds = pickle.load(f)
        print(f"Loaded test dataset with {len(test_ds)} samples")
        test_loader = DataLoader(test_ds, **cfg['test'])
        print("DataLoader created for PointNet2 or PointNet2MSG datasets.")

    else:
        test_ds = RGBESequenceDataset(
            data_root          = ds['test_dir'],
            window_size        = ds['window_size'],
            stride             = ds['stride'],
            enable_transform   = ds['enable_transform'],
            label_map          = ds['label_map']
        )
        test_loader = DataLoader(
            test_ds,
            **cfg['test']
        )

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
    if premodel_path is None:
        raise ValueError("Please provide the path to the pre-trained model.")
    if os.path.exists(premodel_path):
        model.load_state_dict(torch.load(premodel_path, map_location=device))
        print(f"Loaded model from {premodel_path}")
    else:
        raise FileNotFoundError(f"Model file not found at {premodel_path}")

    # 创建日志目录和文件
    if log_path is None:
        log_path = os.path.join(cfg['test_log_dir'], 'testing_log_tmp.txt')
    os.makedirs(cfg['test_log_dir'], exist_ok=True)
    os.makedirs(cfg['test_fig_dir'], exist_ok=True)
    with open(log_path, 'a') as f:
        f.write(f" Model loaded from: {premodel_path}\n")
        f.write(f" Model type: {model_type}\n")
        if model_type == 'vit':
            f.write(f" ------ViT Model Configuration------\n")
            f.write(f" Window_size: {ds['window_size']}\n")
            f.write(f" Stride: {ds['stride']}\n")
            f.write(f" ViT Model: {cfg['vit_model']}\n")
        elif model_type == 'cnn':
            f.write(f" Window_size: {ds['window_size']}\n")
            f.write(f" Stride: {ds['stride']}\n")
            f.write(f" ------CNN Model Configuration------\n")
            f.write(f" CNN Model: {cfg['cnn_model']}\n")
        elif model_type == 'resnet':
            f.write(f" Window_size: {ds['window_size']}\n")
            f.write(f" Stride: {ds['stride']}\n")
            f.write(f" ------ResNet Model Configuration------\n")
            f.write(f" ResNet Model: {cfg['resnet_model']}\n")
        elif model_type == 'resnet_pretrained':
            f.write(f" Window_size: {ds['window_size']}\n")
            f.write(f" Stride: {ds['stride']}\n")
            f.write(f" ------Pretrained ResNet Model Configuration------\n")
            f.write(f" Pretrained ResNet Model: {cfg['resnet_pretrained_model']}\n")
        elif model_type in ['pointnet2', 'pointnet2msg']:
            f.write(f" ------Pointnet2 Model Configuration------\n")
            f.write(f" Loaded test data from: {pnet2_test_path}\n")
            f.write(f" Pointnet2 Model: {cfg['pointnet2_model']}\n")
    
    # 5. 测试评估
    for idx in range(1):
        idx=38
        print(f"Running iteration {idx+1}/10...")
        model.eval()
        class_correct = {}
        class_total = {}
        class_names = list(ds['label_map'].keys())
        correct = total = 0
        # Initialize counters for each class
        for class_name in class_names:
            class_correct[class_name] = 0
            class_total[class_name] = 0
        test_start_time = time.time()

        # 收集所有预测和真实标签用于混淆矩阵
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels, _, _, _, _, _, _ in tqdm.tqdm(test_loader):
                imgs = imgs.float().to(device)
                labels = labels.to(device)
                logits = model(imgs)
                preds = logits.argmax(dim=1)

                # 保存预测和标签用于混淆矩阵
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Update per-class statistics
                for i in range(len(labels)):
                    label_idx = labels[i].item()
                    pred_idx = preds[i].item()
                    label_name = class_names[label_idx]
                    class_total[label_name] += 1
                    if pred_idx == label_idx:
                        class_correct[label_name] += 1

                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        test_end_time = time.time()
        test_acc = correct / total
        test_time = test_end_time - test_start_time
        num_test_batches = len(test_loader)
        num_test_samples = len(test_ds)
        print(f"Test statistics: {num_test_samples} samples in {num_test_batches} batches")
        print(f"Test time: {test_time:.2f} seconds")
        print(f"Test Acc: {test_acc:.4f}")
        with open(log_path, 'a') as f:
            f.write(f"\n-----------------------------------------------------------------------\n")
            f.write(f"Iteration {idx+1}:\n")
            f.write(f"Test statistics: {num_test_samples} samples in {num_test_batches} batches\n")
            f.write(f"Test time: {test_time:.2f} seconds\n")
            f.write(f"Test Acc: {test_acc:.4f} ({correct}/{total})\n")

        # Print and log per-class accuracy
        print("Per-class accuracy:")
        with open(log_path, 'a') as f:
            f.write("Per-class accuracy:\n")
            for class_name in class_names:
                if class_total[class_name] > 0:
                    accuracy = class_correct[class_name] / class_total[class_name]
                    print(f"{class_name}: {accuracy:.4f} ({class_correct[class_name]}/{class_total[class_name]})")
                    f.write(f"{class_name}: {accuracy:.4f} ({class_correct[class_name]}/{class_total[class_name]})\n")
                else:
                    print(f"{class_name}: N/A (0/0)")
                    f.write(f"{class_name}: N/A (0/0)\n")
        
        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, 
                    yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # 保存混淆矩阵图像
        os.makedirs(cfg['test_fig_dir'], exist_ok=True)
        best_model_filename = os.path.basename(premodel_path).split('.')[0]
        confusion_matrix_path = os.path.join(cfg['test_fig_dir'], f"{best_model_filename}_confusion_matrix_{idx+1}.png")
        plt.tight_layout()
        plt.savefig(confusion_matrix_path)
        print(f"Confusion matrix saved to {confusion_matrix_path}")
        
        # 计算归一化混淆矩阵（按行归一化，显示每个类别的召回率分布）
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=class_names, 
                    yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Normalized Confusion Matrix')
        
        # 保存归一化混淆矩阵
        norm_confusion_matrix_path = os.path.join(cfg['test_fig_dir'], f"{best_model_filename}_norm_confusion_matrix_{idx+1}.png")
        plt.tight_layout()
        plt.savefig(norm_confusion_matrix_path)
        print(f"Normalized confusion matrix saved to {norm_confusion_matrix_path}")

        # 将混淆矩阵输出到控制台和日志文件
        print("\nConfusion Matrix:")
        print(cm)
        print("\nNormalized Confusion Matrix:")
        print(cm_normalized)
        with open(log_path, 'a') as f:
            f.write("\nConfusion Matrix:\n")
            np.savetxt(f, cm, fmt='%d', delimiter=',')
            f.write("\nNormalized Confusion Matrix:\n")
            np.savetxt(f, cm_normalized, fmt='%.2f', delimiter=',')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/har_test_config.yaml',
                        help='Path to your_action_config.yaml')
    parser.add_argument('--model', type=str, default='results/checkpoints/pointnet2_event_0628_8_ecount_11.pth',
                        help='Path to the pre-trained model')
    parser.add_argument('--log', type=str, default='results/test_logs/testlog_pointnet2_event_0628_8_ecount_11_38.txt',
                        help='Path to the log file')
    args = parser.parse_args()
    
    main(args.config, args.model, args.log)
    