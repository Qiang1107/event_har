"""
PointNet2 v3 with temporal enhancement for event data processing.
Based on v1 with added temporal feature enhancement module.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.backbones.pointnet2_v1 import farthest_point_sample, hierarchical_sampling


class TemporalFeatureEnhancement(nn.Module):
    """
    时序特征增强模块 - v3版本的核心创新
    对事件数据的时间维度进行处理
    """
    def __init__(self, time_feature_dim=8, encoder_hidden_dim=4, fusion_hidden_multiplier=2, 
                 positional_encoding_scale=10000):
        super().__init__()
        self.time_feature_dim = time_feature_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.positional_encoding_scale = positional_encoding_scale
        
        # 时间编码器 - 将时间戳编码为更丰富的特征
        self.time_encoder = nn.Sequential(
            nn.Linear(1, encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, time_feature_dim)
        )
        
        # 特征融合网络
        fusion_hidden_dim = time_feature_dim * fusion_hidden_multiplier
        self.feature_fusion = nn.Sequential(
            nn.Linear(time_feature_dim + 1, fusion_hidden_dim),  # +1 for original feature
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, time_feature_dim),
            nn.Dropout(0.1)
        )
        
        # 位置编码
        self.register_buffer('div_term', torch.exp(torch.arange(0, time_feature_dim, 2).float() * 
                            -(math.log(positional_encoding_scale) / time_feature_dim)))
    
    def positional_encoding(self, timestamps):
        """
        为时间戳生成位置编码
        输入: timestamps [B, N] 
        输出: pos_encoding [B, N, time_feature_dim]
        """
        B, N = timestamps.shape
        pe = torch.zeros(B, N, self.time_feature_dim, device=timestamps.device)
        
        # 归一化时间戳
        timestamps_norm = timestamps.unsqueeze(-1)  # [B, N, 1]
        
        # 生成sin/cos位置编码
        pe[:, :, 0::2] = torch.sin(timestamps_norm * self.div_term)
        pe[:, :, 1::2] = torch.cos(timestamps_norm * self.div_term)
        
        return pe
    
    def forward(self, timestamps, polarity):
        """
        输入:
            timestamps: [B, N] 时间戳
            polarity: [B, N] 极性
        输出:
            enhanced_features: [B, time_feature_dim, N] 增强的时序特征
        """
        B, N = timestamps.shape
        
        # 1. 时间编码
        time_features = self.time_encoder(timestamps.unsqueeze(-1))  # [B, N, time_feature_dim]
        
        # 2. 位置编码
        pos_encoding = self.positional_encoding(timestamps)  # [B, N, time_feature_dim]
        
        # 3. 特征融合
        combined_input = torch.cat([time_features + pos_encoding, polarity.unsqueeze(-1)], dim=-1)
        enhanced_features = self.feature_fusion(combined_input)  # [B, N, time_feature_dim]
        
        return enhanced_features.transpose(1, 2)  # [B, time_feature_dim, N]


class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, sampling='random'):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.sampling = sampling

        self.conv1 = nn.Conv2d(in_channel + 3, mlp[0], 1)
        self.conv2 = nn.Conv2d(mlp[0], mlp[1], 1)
        self.conv3 = nn.Conv2d(mlp[1], mlp[2], 1)
        self.bn1 = nn.BatchNorm2d(mlp[0])
        self.bn2 = nn.BatchNorm2d(mlp[1])
        self.bn3 = nn.BatchNorm2d(mlp[2])

    def forward(self, xyz, points):
        B, _, N = xyz.shape

        if self.npoint is not None:
            if self.sampling == 'random':
                # 随机采样，速度快 
                idx = torch.randperm(N, device=xyz.device)[:self.npoint]
                idx = idx.unsqueeze(0).expand(B, -1)  # [B, npoint]
            elif self.sampling == 'hierarchical':
                # 分层采样
                idx = hierarchical_sampling(xyz, self.npoint)
            elif self.sampling == 'fps':
                # 标准FPS采样
                idx = farthest_point_sample(xyz, self.npoint)
            else:
                # 默认随机采样
                idx = torch.randperm(N, device=xyz.device)[:self.npoint]
                idx = idx.unsqueeze(0).expand(B, -1)

            new_xyz = torch.gather(xyz, 2, idx.unsqueeze(1).expand(-1, 3, -1))
        else:
            new_xyz = xyz
            self.npoint = N

        dist = torch.cdist(new_xyz.transpose(1, 2), xyz.transpose(1, 2))
        _, idx = dist.topk(self.nsample, dim=-1, largest=False)

        grouped_xyz = torch.gather(xyz.unsqueeze(2).expand(-1, -1, self.npoint, -1),
                                   3, idx.unsqueeze(1).expand(-1, 3, -1, -1))
        grouped_xyz = (grouped_xyz - new_xyz.unsqueeze(-1)).permute(0, 1, 3, 2)

        if points is not None:
            grouped_points = torch.gather(points.unsqueeze(2).expand(-1, -1, self.npoint, -1),
                                          3, idx.unsqueeze(1).expand(-1, points.shape[1], -1, -1))
            new_points = torch.cat([grouped_xyz, grouped_points.permute(0, 1, 3, 2)], dim=1)
        else:
            new_points = grouped_xyz

        new_points = F.relu(self.bn1(self.conv1(new_points)))
        new_points = F.relu(self.bn2(self.conv2(new_points)))
        new_points = self.bn3(self.conv3(new_points))

        new_points = torch.max(new_points, 2)[0]  # (B, mlp[-1], npoint)

        return new_xyz, new_points


class PointNet2V3Classifier(nn.Module):
    """
    PointNet2 V3 分类器 - 在v1基础上增加时序特征增强
    """
    def __init__(self, cfg: dict):
        super().__init__()
        pointnet2_cfg = cfg['pointnet2_v3_model']
        self.input_dim = pointnet2_cfg['input_dim']
        self.use_temporal_enhancement = pointnet2_cfg.get('use_temporal_enhancement', True)
        
        # 时序特征增强配置
        time_feature_dim = pointnet2_cfg.get('time_feature_dim', 8)
        temporal_enhancement_cfg = pointnet2_cfg.get('temporal_enhancement', {})
        encoder_hidden_dim = temporal_enhancement_cfg.get('encoder_hidden_dim', 4)
        fusion_hidden_multiplier = temporal_enhancement_cfg.get('fusion_hidden_multiplier', 2)
        positional_encoding_scale = temporal_enhancement_cfg.get('positional_encoding_scale', 10000)
        
        output_num_class = pointnet2_cfg['num_classes']

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

        # PointNet2主干网络 (基于v1) # sampling方式可选 'random', 'hierarchical', 'fps'
        self.sa1 = SetAbstraction(npoint=512, radius=0.2, nsample=32, 
                                  in_channel=enhanced_feature_dim, mlp=[64, 64, 128], sampling='random')
        self.sa2 = SetAbstraction(npoint=128, radius=0.4, nsample=64, 
                                  in_channel=128, mlp=[128, 128, 256], sampling='fps')
        self.sa3 = SetAbstraction(npoint=None, radius=None, nsample=1, 
                                  in_channel=256, mlp=[256, 512, 1024], sampling='random')

        # 分类头 (与v1相同)
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

        # PointNet2主干网络处理
        xyz1, points1 = self.sa1(xyz, enhanced_points)
        xyz2, points2 = self.sa2(xyz1, points1)
        _, points3 = self.sa3(xyz2, points2)  # points3: (B, 1024, N')
        
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
PointNet2Classifier = PointNet2V3Classifier
