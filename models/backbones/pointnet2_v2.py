"""
PointNet2 v2 is a OFFICIAL implementation of PointNet++ for point cloud processing.
With extensive functions in pointnet2_utils.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones.pointnet2_utils import PointNetSetAbstraction


class PointNet2Classifier(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        pointnet2_cfg = cfg['pointnet2_model']
        self.input_dim = pointnet2_cfg['input_dim']
        self.normal_channel = pointnet2_cfg['normal_channel']
        in_channel = self.input_dim if self.normal_channel else 3
        output_num_class = pointnet2_cfg['num_classes']

        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        # # 渐进式降采样架构
        # self.sa1 = PointNetSetAbstraction(npoint=4096, radius=0.1, nsample=32, in_channel=in_channel, mlp=[32, 32, 64], group_all=False)
        # self.sa2 = PointNetSetAbstraction(npoint=1024, radius=0.2, nsample=32, in_channel=64 + 3, mlp=[64, 64, 128], group_all=False)                          
        # self.sa3 = PointNetSetAbstraction(npoint=256, radius=0.4, nsample=32, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        # self.sa4 = PointNetSetAbstraction(npoint=64, radius=0.6, nsample=64, in_channel=256 + 3, mlp=[256, 256, 512], group_all=False) 
        # self.sa5 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[512, 1024, 1024], group_all=True)
        
        # 全连接层
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, output_num_class)
        
        
    def forward(self, x):
        # 检查输入维度 原始输入维度是(B, N, C)，需要转换为 (B, C, N)
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got shape {xyz.shape}")
        if x.shape[1] != self.input_dim:
            x = x.permute(0, 2, 1)

        B, C, N = x.shape
        if C != self.input_dim:
            raise ValueError(f"Expected input_dim {self.input_dim}, got {C}")

        # 输入形状：(B, C, N)，例如 (16, 4, N)
        xyz = x[:, :3, :] # 例如 (B, 3, N) - 前3维是坐标
        points = x[:, 3:, :] if self.normal_channel else None  # 例如 (B, 1, N) - 后n维是特征

        xyz1, points1 = self.sa1(xyz, points)
        xyz2, points2 = self.sa2(xyz1, points1)
        xyz3, points3 = self.sa3(xyz2, points2)  # points3: (B, 1024, N')
        x = points3.view(B, 1024)

        # xyz1, points1 = self.sa1(xyz, points)     # 32768 -> 4096
        # xyz2, points2 = self.sa2(xyz1, points1)   # 4096 -> 1024
        # xyz3, points3 = self.sa3(xyz2, points2)   # 1024 -> 256
        # xyz4, points4 = self.sa4(xyz3, points3)   # 256 -> 64
        # _, points5 = self.sa5(xyz4, points4)      # 64 -> global
        # x = points5.view(B, 1024)

        if x.size(0) == 1 and self.training: # 处理批次大小为1的情况
            print("Warning: Batch size is 1. This may cause instability with batch normalization.")
            x = torch.cat([x, x], dim=0) # 复制样本以创建批次大小为2的批次
            x = self.drop1(F.relu(self.bn1(self.fc1(x))))
            x = self.drop2(F.relu(self.bn2(self.fc2(x))))
            x = self.fc3(x)
            return x[:1] # 只保留原始样本的预测
        else: # 全连接层
            x = self.drop1(F.relu(self.bn1(self.fc1(x))))
            x = self.drop2(F.relu(self.bn2(self.fc2(x))))
            x = self.fc3(x)
            return x

