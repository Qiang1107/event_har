"""
PointNet2_v3: Enhanced PointNet2 with Temporal Feature Enhancement for Event-based HAR
Key improvements:
1. Temporal Feature Enhancement Module (TFEM)
2. Adaptive Temporal Pooling
3. Multi-scale Temporal Attention
4. Temporal Consistency Loss
5. Enhanced Set Abstraction with Temporal Awareness
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class TemporalFeatureEnhancement(nn.Module):
    """时序特征增强模块"""
    def __init__(self, input_dim, time_feature_dim=8):
        super().__init__()
        self.time_feature_dim = time_feature_dim
        
        # 时间编码器
        self.time_encoder = nn.Sequential(
            nn.Linear(1, time_feature_dim // 2),
            nn.ReLU(),
            nn.Linear(time_feature_dim // 2, time_feature_dim)
        )
        
        # 时序特征融合
        self.temporal_fusion = nn.Sequential(
            nn.Linear(input_dim + time_feature_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim)
        )
        
        # 位置编码（类似Transformer）
        self.register_buffer('pe_scale', torch.tensor(10000.0))
        
    def positional_encoding(self, t, d_model):
        """生成位置编码"""
        position = t.unsqueeze(-1)  # [B, N, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2, device=t.device).float() * 
                           -(math.log(self.pe_scale.item()) / d_model))
        
        pe = torch.zeros(t.shape[0], t.shape[1], d_model, device=t.device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, features, time_stamps):
        """
        Args:
            features: [B, C, N] 点云特征
            time_stamps: [B, 1, N] 时间戳
        Returns:
            enhanced_features: [B, C, N] 增强后的特征
        """
        B, C, N = features.shape
        t = time_stamps.squeeze(1)  # [B, N]
        
        # 1. 基础时间特征编码
        time_features = self.time_encoder(t.unsqueeze(-1))  # [B, N, time_feature_dim]
        
        # 2. 位置编码
        pos_encoding = self.positional_encoding(t, self.time_feature_dim)  # [B, N, time_feature_dim]
        time_features = time_features + pos_encoding
        
        # 3. 特征融合
        features_flat = features.permute(0, 2, 1)  # [B, N, C]
        combined_features = torch.cat([features_flat, time_features], dim=-1)  # [B, N, C + time_feature_dim]
        
        enhanced_features = self.temporal_fusion(combined_features)  # [B, N, C]
        enhanced_features = enhanced_features.permute(0, 2, 1)  # [B, C, N]
        
        return enhanced_features


class MultiScaleTemporalAttention(nn.Module):
    """多尺度时序注意力模块"""
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        # 多头注意力
        self.q_linear = nn.Linear(feature_dim, feature_dim)
        self.k_linear = nn.Linear(feature_dim, feature_dim)
        self.v_linear = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        # 多尺度时间窗口
        self.temporal_scales = [4, 8, 16]  # 不同时间尺度
        self.scale_weights = nn.Parameter(torch.ones(len(self.temporal_scales)))
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def temporal_windowing(self, x, window_size):
        """对特征进行时间窗口分组"""
        B, N, C = x.shape
        if N < window_size:
            return x.unsqueeze(2)  # [B, N, 1, C]
        
        # 将序列分成若干个时间窗口
        num_windows = N // window_size
        windowed = x[:, :num_windows*window_size, :].view(B, num_windows, window_size, C)
        return windowed.mean(dim=2)  # [B, num_windows, C]
    
    def forward(self, x, time_stamps):
        """
        Args:
            x: [B, C, N] 特征
            time_stamps: [B, 1, N] 时间戳
        Returns:
            attended_features: [B, C, N] 注意力增强特征
        """
        B, C, N = x.shape
        x_input = x.permute(0, 2, 1)  # [B, N, C]
        t = time_stamps.squeeze(1)  # [B, N]
        
        # 多尺度时序建模
        scale_outputs = []
        for i, scale in enumerate(self.temporal_scales):
            # 时间窗口分组
            windowed_x = self.temporal_windowing(x_input, scale)  # [B, num_windows, C]
            
            # 自注意力计算
            q = self.q_linear(windowed_x)
            k = self.k_linear(windowed_x)
            v = self.v_linear(windowed_x)
            
            # 多头注意力
            q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            attended = torch.matmul(attn_weights, v)
            attended = attended.transpose(1, 2).contiguous().view(B, -1, C)
            attended = self.out_proj(attended)
            
            # 上采样回原始序列长度
            if attended.shape[1] < N:
                attended = F.interpolate(attended.permute(0, 2, 1), size=N, mode='linear', align_corners=False)
                attended = attended.permute(0, 2, 1)
            
            scale_outputs.append(attended * self.scale_weights[i])
        
        # 融合多尺度结果
        multi_scale_output = sum(scale_outputs) / len(scale_outputs)
        
        # 残差连接和层归一化
        output = self.layer_norm(x_input + multi_scale_output)
        
        return output.permute(0, 2, 1)  # [B, C, N]


class AdaptiveTemporalPooling(nn.Module):
    """自适应时序池化模块"""
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 学习池化权重
        self.pooling_gate = nn.Sequential(
            nn.Linear(feature_dim + 1, feature_dim),  # +1 for time
            nn.Sigmoid()
        )
        
        # 时序一致性评估
        self.consistency_scorer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features, time_stamps):
        """
        Args:
            features: [B, C, N] 特征
            time_stamps: [B, 1, N] 时间戳
        Returns:
            pooled_features: [B, C] 池化后的全局特征
            temporal_weights: [B, N] 时序权重
        """
        B, C, N = features.shape
        features_t = features.permute(0, 2, 1)  # [B, N, C]
        t = time_stamps.squeeze(1).unsqueeze(-1)  # [B, N, 1]
        
        # 计算时序门控权重
        gate_input = torch.cat([features_t, t], dim=-1)  # [B, N, C+1]
        temporal_gates = self.pooling_gate(gate_input)  # [B, N, C]
        
        # 计算时序一致性分数
        consistency_scores = self.consistency_scorer(features_t).squeeze(-1)  # [B, N]
        
        # 综合权重
        temporal_weights = consistency_scores  # [B, N]
        
        # 加权池化
        weighted_features = features_t * temporal_gates  # [B, N, C]
        temporal_weights_expanded = temporal_weights.unsqueeze(-1)  # [B, N, 1]
        
        pooled_features = torch.sum(weighted_features * temporal_weights_expanded, dim=1)  # [B, C]
        weight_sum = torch.sum(temporal_weights_expanded, dim=1) + 1e-8  # [B, 1]
        pooled_features = pooled_features / weight_sum  # [B, C]
        
        return pooled_features, temporal_weights


class EnhancedSetAbstraction(nn.Module):
    """增强的Set Abstraction模块，融入时序感知"""
    def __init__(self, npoint, radius, nsample, in_channel, mlp, time_feature_dim=8, use_temporal=True):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.use_temporal = use_temporal
        
        # 时序特征增强
        if use_temporal:
            self.temporal_enhancer = TemporalFeatureEnhancement(in_channel, time_feature_dim)
            conv_in_channel = in_channel + 3  # 3 for xyz offset
        else:
            conv_in_channel = in_channel + 3
        
        # 卷积层
        self.conv1 = nn.Conv2d(conv_in_channel, mlp[0], 1)
        self.conv2 = nn.Conv2d(mlp[0], mlp[1], 1)
        self.conv3 = nn.Conv2d(mlp[1], mlp[2], 1)
        self.bn1 = nn.BatchNorm2d(mlp[0])
        self.bn2 = nn.BatchNorm2d(mlp[1])
        self.bn3 = nn.BatchNorm2d(mlp[2])
        
        # 时序注意力
        if use_temporal:
            self.temporal_attention = MultiScaleTemporalAttention(mlp[2])
    
    def temporal_aware_sampling(self, xyz, time_stamps):
        """时序感知的点采样"""
        B, _, N = xyz.shape
        
        if self.npoint is None or self.npoint >= N:
            return torch.arange(N, device=xyz.device).unsqueeze(0).expand(B, -1)
        
        # 计算时间分布的方差，优先采样时间分布均匀的区域
        t = time_stamps.squeeze(1)  # [B, N]
        
        # 简单的时序感知采样：结合随机性和时间分布
        time_weights = torch.softmax(t * 10, dim=-1)  # 给时间较晚的点更高权重
        
        # 加权随机采样
        sampled_indices = []
        for b in range(B):
            weights = time_weights[b] + 0.1  # 添加少量随机性
            indices = torch.multinomial(weights, self.npoint, replacement=False)
            sampled_indices.append(indices)
        
        return torch.stack(sampled_indices, dim=0)  # [B, npoint]
    
    def forward(self, xyz, points, time_stamps):
        """
        Args:
            xyz: [B, 3, N] 坐标
            points: [B, C, N] 特征 (可能为None)
            time_stamps: [B, 1, N] 时间戳
        Returns:
            new_xyz: [B, 3, npoint] 新的采样点坐标
            new_points: [B, mlp[-1], npoint] 新的特征
        """
        B, _, N = xyz.shape
        
        # 时序感知采样
        if self.npoint is not None:
            idx = self.temporal_aware_sampling(xyz, time_stamps)
            new_xyz = torch.gather(xyz, 2, idx.unsqueeze(1).expand(-1, 3, -1))
            new_time = torch.gather(time_stamps, 2, idx.unsqueeze(1))
        else:
            new_xyz = xyz
            new_time = time_stamps
            self.npoint = N
            idx = torch.arange(N, device=xyz.device).unsqueeze(0).expand(B, -1)
        
        # K近邻搜索
        dist = torch.cdist(new_xyz.transpose(1, 2), xyz.transpose(1, 2))
        _, neighbor_idx = dist.topk(self.nsample, dim=-1, largest=False)
        
        # 分组坐标
        grouped_xyz = torch.gather(xyz.unsqueeze(2).expand(-1, -1, self.npoint, -1),
                                   3, neighbor_idx.unsqueeze(1).expand(-1, 3, -1, -1))
        grouped_xyz = (grouped_xyz - new_xyz.unsqueeze(-1)).permute(0, 1, 3, 2)
        
        # 分组特征
        if points is not None:
            # 时序增强特征
            if self.use_temporal:
                points = self.temporal_enhancer(points, time_stamps)
            
            grouped_points = torch.gather(points.unsqueeze(2).expand(-1, -1, self.npoint, -1),
                                          3, neighbor_idx.unsqueeze(1).expand(-1, points.shape[1], -1, -1))
            new_points = torch.cat([grouped_xyz, grouped_points.permute(0, 1, 3, 2)], dim=1)
        else:
            new_points = grouped_xyz
        
        # 卷积处理
        new_points = F.relu(self.bn1(self.conv1(new_points)))
        new_points = F.relu(self.bn2(self.conv2(new_points)))
        new_points = self.bn3(self.conv3(new_points))
        
        # 最大池化
        new_points = torch.max(new_points, 2)[0]  # [B, mlp[-1], npoint]
        
        # 时序注意力增强
        if self.use_temporal:
            new_points = self.temporal_attention(new_points, new_time)
        
        return new_xyz, new_points


class PointNet2TemporalClassifier(nn.Module):
    """增强版PointNet2分类器，专注于时序特征建模"""
    def __init__(self, cfg: dict):
        super().__init__()
        pointnet2_cfg = cfg['pointnet2_v3_model']
        self.input_dim = pointnet2_cfg['input_dim']
        feature_dim = self.input_dim - 3  # 特征维度：4-3=1
        output_num_class = pointnet2_cfg['num_classes']
        
        # 可配置参数
        self.use_temporal_enhancement = pointnet2_cfg.get('use_temporal_enhancement', True)
        self.time_feature_dim = pointnet2_cfg.get('time_feature_dim', 8)
        self.temporal_pooling = pointnet2_cfg.get('temporal_pooling', True)
        self.attention_mechanism = pointnet2_cfg.get('attention_mechanism', True)
        
        # 增强的Set Abstraction层
        self.sa1 = EnhancedSetAbstraction(
            npoint=512, radius=0.2, nsample=32, in_channel=feature_dim, 
            mlp=[64, 64, 128], time_feature_dim=self.time_feature_dim,
            use_temporal=self.use_temporal_enhancement
        )
        self.sa2 = EnhancedSetAbstraction(
            npoint=128, radius=0.4, nsample=64, in_channel=128, 
            mlp=[128, 128, 256], time_feature_dim=self.time_feature_dim,
            use_temporal=self.use_temporal_enhancement
        )
        self.sa3 = EnhancedSetAbstraction(
            npoint=None, radius=None, nsample=1, in_channel=256, 
            mlp=[256, 512, 1024], time_feature_dim=self.time_feature_dim,
            use_temporal=self.use_temporal_enhancement
        )
        
        # 自适应时序池化
        if self.temporal_pooling:
            self.adaptive_pooling = AdaptiveTemporalPooling(1024)
            fc_input_dim = 1024
        else:
            fc_input_dim = 1024
        
        # 全连接分类器
        self.fc1 = nn.Linear(fc_input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, output_num_class)
        
        # 时序一致性损失权重
        self.temporal_consistency_weight = 0.1
        
    def temporal_consistency_loss(self, features, temporal_weights):
        """计算时序一致性损失"""
        if temporal_weights is None:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        # 时序权重应该平滑变化，而不是跳跃变化
        weight_diff = temporal_weights[:, 1:] - temporal_weights[:, :-1]
        smoothness_loss = torch.mean(weight_diff ** 2)
        
        return smoothness_loss
    
    def forward(self, x):
        """
        Args:
            x: [B, C, N] 或 [B, N, C] 输入点云
        Returns:
            logits: [B, num_classes] 分类结果
            temporal_info: dict 包含时序信息的字典
        """
        # 检查输入维度并转换格式
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got shape {x.shape}")
        
        if x.shape[1] != self.input_dim:
            x = x.permute(0, 2, 1)
        
        B, C, N = x.shape
        if C != self.input_dim:
            raise ValueError(f"Expected input_dim {self.input_dim}, got {C}")
        
        # 分离坐标和特征
        xyz = x[:, :3, :]  # [B, 3, N] - x, y, t
        points = x[:, 3:, :] if C > 3 else None  # [B, 1, N] - p (polarity)
        time_stamps = xyz[:, 2:3, :]  # [B, 1, N] - 时间戳
        spatial_xyz = xyz[:, :2, :]  # [B, 2, N] - x, y坐标
        
        # 重构空间坐标，保持时间信息
        xyz_spatial = torch.cat([spatial_xyz, time_stamps], dim=1)  # [B, 3, N]
        
        # 逐层特征提取
        xyz1, points1 = self.sa1(xyz_spatial, points, time_stamps)
        xyz2, points2 = self.sa2(xyz1, points1, time_stamps[:, :, :xyz1.shape[-1]])
        _, points3 = self.sa3(xyz2, points2, time_stamps[:, :, :xyz2.shape[-1]])
        
        # 全局特征聚合
        temporal_weights = None
        if self.temporal_pooling:
            global_features, temporal_weights = self.adaptive_pooling(points3, time_stamps[:, :, :points3.shape[-1]])
        else:
            global_features = torch.max(points3, dim=2)[0]  # [B, 1024]
        
        # 处理批次大小为1的情况（训练时的BatchNorm问题）
        if global_features.size(0) == 1 and self.training:
            global_features = torch.cat([global_features, global_features], dim=0)
            x = self.drop1(F.relu(self.bn1(self.fc1(global_features))))
            x = self.drop2(F.relu(self.bn2(self.fc2(x))))
            logits = self.fc3(x)
            logits = logits[:1]  # 只保留原始样本
        else:
            x = self.drop1(F.relu(self.bn1(self.fc1(global_features))))
            x = self.drop2(F.relu(self.bn2(self.fc2(x))))
            logits = self.fc3(x)
        
        # 返回结果和时序信息
        temporal_info = {
            'temporal_weights': temporal_weights,
            'temporal_consistency_loss': self.temporal_consistency_loss(global_features, temporal_weights) if temporal_weights is not None else None
        }
        
        return logits, temporal_info
    
    def get_temporal_loss(self, temporal_info):
        """获取时序损失"""
        if temporal_info['temporal_consistency_loss'] is not None:
            return self.temporal_consistency_weight * temporal_info['temporal_consistency_loss']
        return torch.tensor(0.0, requires_grad=True)


# 为了兼容性，提供原始接口
class PointNet2Classifier(PointNet2TemporalClassifier):
    """兼容原始接口的包装类"""
    def forward(self, x):
        logits, temporal_info = super().forward(x)
        return logits


class TemporalLoss(nn.Module):
    """结合分类损失和时序一致性损失的复合损失函数"""
    def __init__(self, num_classes, temporal_weight=0.1):
        super().__init__()
        self.classification_loss = nn.CrossEntropyLoss()
        self.temporal_weight = temporal_weight
        
    def forward(self, logits, targets, temporal_info=None):
        # 分类损失
        cls_loss = self.classification_loss(logits, targets)
        
        # 时序一致性损失
        temporal_loss = 0.0
        if temporal_info is not None and temporal_info.get('temporal_consistency_loss', 0.1) is not None: # 时序一致性损失权重 (0.01, 0.1, 0.2)
            temporal_loss = self.temporal_weight * temporal_info['temporal_consistency_loss']
        
        total_loss = cls_loss + temporal_loss
        
        return {
            'total_loss': total_loss,
            'classification_loss': cls_loss,
            'temporal_loss': temporal_loss
        }
    