"""
PointNet2 MSG v3 with temporal enhancement for event data processing.
Based on pointnet2_v3.py with Multi-Scale Grouping (MSG) capabilities.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.backbones.pointnet2_v1 import farthest_point_sample, hierarchical_sampling
from models.backbones.pointnet2_v3 import TemporalFeatureEnhancement
from models.backbones.pointnet2msg_v1 import SetAbstractionMSG


class PointNet2MSGV3Classifier(nn.Module):
    """
    PointNet2 MSG V3 分类器 - 结合多尺度分组和时序特征增强
    """

    def __init__(self, cfg: dict):
        super().__init__()
        pointnet2_cfg = cfg["pointnet2_v3_model"]
        self.input_dim = pointnet2_cfg["input_dim"]
        self.use_temporal_enhancement = pointnet2_cfg.get('use_temporal_enhancement', True)
        
        # 时序特征增强配置
        time_feature_dim = pointnet2_cfg.get('time_feature_dim', 8)
        temporal_enhancement_cfg = pointnet2_cfg.get('temporal_enhancement', {})
        encoder_hidden_dim = temporal_enhancement_cfg.get('encoder_hidden_dim', 4)
        fusion_hidden_multiplier = temporal_enhancement_cfg.get('fusion_hidden_multiplier', 2)
        positional_encoding_scale = temporal_enhancement_cfg.get('positional_encoding_scale', 10000)
        
        output_num_class = pointnet2_cfg["num_classes"]

        # 时序特征增强模块 (v3新增)
        if self.use_temporal_enhancement:
            self.temporal_enhancement = TemporalFeatureEnhancement(
                time_feature_dim=time_feature_dim,
                encoder_hidden_dim=encoder_hidden_dim,
                fusion_hidden_multiplier=fusion_hidden_multiplier,
                positional_encoding_scale=positional_encoding_scale
            )
            # 更新后的特征维度
            enhanced_feature_dim = time_feature_dim
        else:
            enhanced_feature_dim = self.input_dim - 3  # 原始特征维度：4-3=1

        # PointNet2 MSG主干网络 - 多尺度分组
        # 第一层：多个小尺度特征提取
        self.sa1 = SetAbstractionMSG(
            npoint=512,
            radii=[0.1, 0.2, 0.4],
            nsamples=[16, 32, 128],
            in_channel=enhanced_feature_dim,
            mlps=[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
            sampling='hierarchical' # random, fps, or hierarchical
        )
        
        # 第二层：中等尺度特征提取，使用FPS采样获得更好的代表性
        self.sa2 = SetAbstractionMSG(
            npoint=128,
            radii=[0.2, 0.4, 0.8],
            nsamples=[32, 64, 128],
            in_channel=320,  # 64+128+128 from sa1
            mlps=[[64, 64, 128], [128, 128, 256], [128, 128, 256]],
            sampling='fps' # random, fps, or hierarchical
        )
        
        # 第三层：全局特征提取
        self.sa3 = SetAbstractionMSG(
            npoint=None,
            radii=[None],
            nsamples=[1],
            in_channel=640,  # 128+256+256 from sa2
            mlps=[[256, 512, 1024]],
            sampling='random'
        )

        # 分类头 (与v3相同)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, output_num_class)

    def forward(self, x):
        # 检查输入维度 原始输入维度是(B, N, C)，需要转换为 (B, C, N)
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got shape {x.shape}")
        if x.shape[1] != self.input_dim:
            x = x.permute(0, 2, 1)
        B, C, N = x.shape
        if C != self.input_dim:
            raise ValueError(f"Expected input_dim {self.input_dim}, got {C}")

        # 输入形状：(B, C, N)，例如 (16, 4, N)
        xyz = x[:, :3, :]  # (B, 3, N) - 前3维是坐标 [x, y, t]
        
        if self.use_temporal_enhancement:
            # v3新增：时序特征增强
            timestamps = x[:, 2, :]  # (B, N) - 时间戳
            polarity = x[:, 3, :]    # (B, N) - 极性
            
            # 生成增强的时序特征
            enhanced_points = self.temporal_enhancement(timestamps, polarity)  # (B, time_feature_dim, N)
        else:
            # 使用原始特征
            enhanced_points = x[:, 3:, :]  # (B, feature_dim, N)

        # PointNet2 MSG主干网络处理 - 多尺度特征提取
        xyz1, points1 = self.sa1(xyz, enhanced_points)
        xyz2, points2 = self.sa2(xyz1, points1)
        _, points3 = self.sa3(xyz2, points2)

        # 全局特征提取
        x = torch.max(points3, dim=2)[0]  # 全局最大池化，降维到 (B, 1024)

        # 处理批次大小为1的情况
        if x.size(0) == 1 and self.training:
            # 复制样本以创建批次大小为2的批次
            x = torch.cat([x, x], dim=0)
            x = self.drop1(F.relu(self.bn1(self.fc1(x))))
            x = self.drop2(F.relu(self.bn2(self.fc2(x))))
            x = self.fc3(x)
            # 只保留原始样本的预测
            return x[:1]
        else:
            # 分类头
            x = self.drop1(F.relu(self.bn1(self.fc1(x))))
            x = self.drop2(F.relu(self.bn2(self.fc2(x))))
            x = self.fc3(x)
            return x


# 为了保持向后兼容性，创建别名
PointNet2MSGClassifier = PointNet2MSGV3Classifier