"""
PointNet2 v1 is a SIMPLIFIED implementation of PointNet++ for point cloud processing.
Without using the FPS (Farthest Point Sampling) algorithm, it uses a simplified random sampling method.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def farthest_point_sample(xyz, npoint):
    """
    输入:
        xyz: 点云数据，[B, 3, N]
        npoint: 采样点数量
    输出:
        采样点的索引，[B, npoint]
    """
    device = xyz.device
    B, C, N = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    # 随机初始化第一个点
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        # 获取最远点的坐标
        centroid = xyz[batch_indices, :, farthest].view(B, 3, 1)
        # 计算所有点到当前最远点的距离
        dist = torch.sum((xyz - centroid) ** 2, 1)
        # 更新最小距离
        mask = dist < distance
        distance[mask] = dist[mask]
        # 选择下一个最远点
        farthest = torch.max(distance, -1)[1]
    
    return centroids


def approximate_fps(xyz, npoint, sample_ratio=0.1):
    """
    近似FPS实现 - 在随机子集上执行FPS以加速计算
    
    输入:
        xyz: 点云数据，[B, 3, N]
        npoint: 需要采样的点数量
        sample_ratio: 随机子集大小比例(相对于N)
    输出:
        采样点的索引，[B, npoint]
    """
    device = xyz.device
    B, C, N = xyz.shape
    
    # 转置为[B, N, C]形状以方便计算距离
    xyz_t = xyz.transpose(1, 2).contiguous()  # [B, N, 3]
    
    # 预分配结果数组
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    
    for b in range(B):
        # 计算子集大小
        subset_size = npoint * 2
        
        # 随机选择子集
        subset_idx = torch.randperm(N, device=device)[:subset_size]
        xyz_subset = xyz_t[b, subset_idx]  # [subset_size, 3]
        
        # 在子集上执行标准FPS
        selected = torch.zeros(npoint, dtype=torch.long, device=device)
        distances = torch.ones(subset_size, device=device) * 1e10
        
        # 随机选择第一个点
        farthest = torch.randint(0, subset_size, (1,), device=device)[0]
        
        for i in range(npoint):
            selected[i] = subset_idx[farthest]  # 存储原始索引
            centroid = xyz_subset[farthest].view(1, 3)  # [1, 3]
            
            # 计算子集中所有点到当前点的距离
            dist = torch.sum((xyz_subset - centroid) ** 2, dim=1)  # [subset_size]
            
            # 确保dist和distances形状匹配
            if dist.shape != distances.shape:
                distances = torch.ones_like(dist) * 1e10

            # 更新距离
            mask = dist < distances
            distances[mask] = dist[mask]
            
            # 找出下一个最远点
            farthest = torch.max(distances, dim=0)[1]
        
        centroids[b] = selected
    
    return centroids


class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, sampling='approx_fps'):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.sampling = sampling  # 'random', 'fps', 或 'approx_fps'

        self.conv1 = nn.Conv2d(in_channel + 3, mlp[0], 1)
        self.conv2 = nn.Conv2d(mlp[0], mlp[1], 1)
        self.conv3 = nn.Conv2d(mlp[1], mlp[2], 1)
        self.bn1 = nn.BatchNorm2d(mlp[0])
        self.bn2 = nn.BatchNorm2d(mlp[1])
        self.bn3 = nn.BatchNorm2d(mlp[2])

    def forward(self, xyz, points):
        B, _, N = xyz.shape

        if self.npoint is not None:
            # # 随机采样，速度快 
            # idx = torch.randperm(N)[:self.npoint]  # idx shape: torch.Size([npoint])
            # new_xyz = xyz[:, :, idx].contiguous()
            # # 最远点采样FPS 
            # idx = farthest_point_sample(xyz, self.npoint)  # idx shape: torch.Size([B, npoint])
            # new_xyz = torch.gather(xyz, 2, idx.unsqueeze(1).expand(-1, 3, -1))
            if self.sampling == 'random':
                # 随机采样，速度快 
                idx = torch.randperm(N, device=xyz.device)[:self.npoint]
                idx = idx.unsqueeze(0).expand(B, -1)  # [B, npoint]
            elif self.sampling == 'approx_fps':
                # 近似FPS采样
                idx = approximate_fps(xyz, self.npoint)
            elif self.sampling == 'fps':
                # 标准FPS采样（慢）
                idx = farthest_point_sample(xyz, self.npoint)
            else:
                raise ValueError(f"不支持的采样方法: {self.sampling}")

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


class PointNet2Classifier(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        pointnet2_cfg = cfg['pointnet2_model']
        self.input_dim = pointnet2_cfg['input_dim']
        feature_dim = self.input_dim - 3  # 特征维度：4-3=1
        output_num_class = pointnet2_cfg['num_classes']

        self.sa1 = SetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=feature_dim, mlp=[64, 64, 128], sampling='random')
        self.sa2 = SetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128, mlp=[128, 128, 256], sampling='random')
        self.sa3 = SetAbstraction(npoint=None, radius=None, nsample=1, in_channel=256, mlp=[256, 512, 1024], sampling='random')

        # 渐进式降采样架构
        # self.sa1 = SetAbstraction(npoint=4096, radius=0.1, nsample=32, in_channel=feature_dim, mlp=[32, 32, 64])
        # self.sa2 = SetAbstraction(npoint=1024, radius=0.2, nsample=32, in_channel=64, mlp=[64, 64, 128])                          
        # self.sa3 = SetAbstraction(npoint=256, radius=0.4, nsample=32, in_channel=128, mlp=[128, 128, 256])
        # self.sa4 = SetAbstraction(npoint=64, radius=0.6, nsample=64, in_channel=256, mlp=[256, 256, 512]) 
        # self.sa5 = SetAbstraction(npoint=None, radius=None, nsample=1, in_channel=512, mlp=[512, 1024, 1024])

        # 全连接层
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
            raise ValueError(f"Expected 3D tensor, got shape {xyz.shape}")
        
        if x.shape[1] != self.input_dim:
            x = x.permute(0, 2, 1)
        
        B, C, N = x.shape
        if C != self.input_dim:
            raise ValueError(f"Expected input_dim {self.input_dim}, got {C}")

        # 输入形状：(B, C, N)，例如 (16, 5, N)
        xyz = x[:, :3, :]  # (B, 3, N) - 前3维是坐标
        points = x[:, 3:, :]  # (B, 2, N) - 后n维是特征

        xyz1, points1 = self.sa1(xyz, points)
        xyz2, points2 = self.sa2(xyz1, points1)
        xyz3, points3 = self.sa3(xyz2, points2)  # points3: (B, 1024, N')
        x = torch.max(points3, dim=2)[0] # 全局最大池化，降维到 (B, 1024)

        # xyz1, points1 = self.sa1(xyz, points)     # 32768 -> 4096
        # xyz2, points2 = self.sa2(xyz1, points1)   # 4096 -> 1024
        # xyz3, points3 = self.sa3(xyz2, points2)   # 1024 -> 256
        # xyz4, points4 = self.sa4(xyz3, points3)   # 256 -> 64
        # _, points5 = self.sa5(xyz4, points4)      # 64 -> global
        # x = torch.max(points5, dim=2)[0] # 全局最大池化，降维到 (B, 1024)

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
            # 全连接层
            x = self.drop1(F.relu(self.bn1(self.fc1(x))))
            x = self.drop2(F.relu(self.bn2(self.fc2(x))))
            x = self.fc3(x)
            return x

