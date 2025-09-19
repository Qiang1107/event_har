"""
PointNet2 v1 is a SIMPLIFIED implementation of PointNet++ for point cloud processing.
Without using the FPS (Farthest Point Sampling) algorithm, it uses a simplified random sampling method.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


def voxel_downsample(xyz, npoint, voxel_size=0.05):
    """
    体素下采样实现
    
    输入:
        xyz: 点云数据，[B, 3, N]
        npoint: 近似目标点数
        voxel_size: 体素大小
    输出:
        采样点的索引，[B, ~npoint]
    """
    device = xyz.device
    B, C, N = xyz.shape
    xyz_t = xyz.transpose(1, 2).contiguous()  # [B, N, 3]
    indices = torch.zeros(B, N, dtype=torch.bool, device=device)
    
    for b in range(B):
        # 计算点云的范围
        min_vals, _ = torch.min(xyz_t[b], dim=0)
        max_vals, _ = torch.max(xyz_t[b], dim=0)
        
        # 将点云映射到体素网格
        voxel_indices = ((xyz_t[b] - min_vals) / voxel_size).int()
        
        # 使用字典记录每个体素中的点
        voxel_dict = {}
        for i in range(N):
            voxel_key = tuple(voxel_indices[i].cpu().numpy())
            if voxel_key not in voxel_dict:
                voxel_dict[voxel_key] = i
                indices[b, i] = True
    
    # 如果选取的点太多，随机删除一些
    for b in range(B):
        selected = indices[b].nonzero().squeeze()
        if selected.numel() > npoint:
            perm = torch.randperm(selected.numel(), device=device)[:npoint]
            keep = selected[perm]
            indices[b] = torch.zeros(N, dtype=torch.bool, device=device)
            indices[b, keep] = True
    
    # 将布尔索引转换为整数索引
    result = []
    for b in range(B):
        result.append(indices[b].nonzero().squeeze())
    
    # 填充到相同长度
    max_len = max(res.shape[0] for res in result)
    padded_result = torch.zeros(B, max_len, dtype=torch.long, device=device)
    for b, res in enumerate(result):
        if res.numel() > 0:  # 检查是否为空张量
            padded_result[b, :res.numel()] = res
    
    return padded_result[:, :npoint]  # 截取前npoint个点


def hierarchical_sampling(xyz, npoint):
    """
    分层采样实现 - 先将点云分成多个区域，每个区域随机采样
    
    输入:
        xyz: 点云数据，[B, 3, N]
        npoint: 采样点数量
    输出:
        采样点的索引，[B, npoint]
    """
    device = xyz.device
    B, C, N = xyz.shape
    
    # 确定区域数量 (大约sqrt(npoint)个区域)
    num_regions = int(np.sqrt(npoint))
    points_per_region = npoint // num_regions
    
    xyz_t = xyz.transpose(1, 2)  # [B, N, 3]
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    
    for b in range(B):
        # K-means聚类的简化版本 - 只迭代一次
        # 随机初始化中心点
        idx = torch.randperm(N, device=device)[:num_regions]
        centers = xyz_t[b, idx]  # [num_regions, 3]
        
        # 计算每个点到所有中心的距离
        dists = torch.cdist(xyz_t[b], centers)  # [N, num_regions]
        
        # 为每个点分配最近的中心
        region_assignments = torch.argmin(dists, dim=1)  # [N]
        
        # 对每个区域进行随机采样
        sample_idx = 0
        for r in range(num_regions):
            region_points = (region_assignments == r).nonzero().squeeze()
            if region_points.numel() == 0:
                continue
            
            # 处理单个点的情况，确保是1D张量
            if region_points.dim() == 0:
                region_points = region_points.unsqueeze(0)
                
            # 确定这个区域要采样的点数量
            region_size = region_points.shape[0]
            sample_size = min(points_per_region, region_size)
            
            if region_size > 0:
                if region_size == 1:
                    # 只有一个点的情况
                    region_samples = region_points
                else:
                    # 随机采样
                    perm = torch.randperm(region_size, device=device)[:sample_size]
                    region_samples = region_points[perm]
                
                # 将采样结果添加到结果中
                remaining = npoint - sample_idx
                actual_samples = min(sample_size, remaining)
                if actual_samples > 0:
                    centroids[b, sample_idx:sample_idx+actual_samples] = region_samples[:actual_samples]
                    sample_idx += actual_samples
                
                if sample_idx >= npoint:
                    break
        
        # 如果还没有采样足够的点，用随机点填充
        if sample_idx < npoint:
            remaining = npoint - sample_idx
            perm = torch.randperm(N, device=device)[:remaining]
            centroids[b, sample_idx:] = perm
    
    return centroids


def hybrid_sampling(xyz, npoint, random_ratio=0.5):
    """
    混合采样策略 - 先随机采样，再应用FPS
    
    输入:
        xyz: 点云数据，[B, 3, N]
        npoint: 采样点数量
        random_ratio: 随机采样的比例
    输出:
        采样点的索引，[B, npoint]
    """
    device = xyz.device
    B, C, N = xyz.shape
    
    # 第一步：随机采样得到中间结果
    random_npoint = int(npoint * 2)  # 采样2倍点数
    rand_indices = []
    for b in range(B):
        rand_idx = torch.randperm(N, device=device)[:random_npoint]
        rand_indices.append(rand_idx)
    
    # 堆叠为批量索引
    rand_indices = torch.stack(rand_indices)  # [B, random_npoint]
    
    # 第二步：对随机采样结果应用一个快速的基于距离的采样
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    
    for b in range(B):
        # 获取随机采样的点
        pts_idx = rand_indices[b]
        pts = torch.index_select(xyz[b].t(), 0, pts_idx)  # [random_npoint, 3]
        
        # 在这个小数据集上执行FPS
        selected = torch.zeros(npoint, dtype=torch.long, device=device)
        distances = torch.ones(random_npoint, device=device) * 1e10
        
        # 随机选择第一个点
        farthest = torch.randint(0, random_npoint, (1,), device=device)[0]
        
        # FPS算法
        for i in range(npoint):
            selected[i] = pts_idx[farthest]
            centroid = pts[farthest].view(1, 3)
            
            # 计算距离 - 这里数据集小，FPS快很多
            dist = torch.sum((pts - centroid) ** 2, dim=1)
            mask = dist < distances
            distances[mask] = dist[mask]
            farthest = torch.max(distances, dim=0)[1]
        
        centroids[b] = selected
    
    return centroids


def feature_based_sampling(xyz, points, npoint):
    """
    基于特征的重要性采样
    
    输入:
        xyz: 点云坐标，[B, 3, N]
        points: 点云特征，[B, C, N]
        npoint: 采样点数量
    输出:
        采样点的索引，[B, npoint]
    """
    device = xyz.device
    B, C, N = xyz.shape
    F = points.shape[1]  # 特征维度
    
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    
    for b in range(B):
        # 计算每个点的局部特征变化
        xyz_t = xyz[b].t()  # [N, 3]
        pts_t = points[b].t()  # [N, F]
        
        # 计算简化版的特征重要性分数
        # 这里使用特征值的方差作为重要性指标
        importance = torch.var(pts_t, dim=1)  # [N]
        
        # 按重要性对点排序
        _, sorted_indices = torch.sort(importance, descending=True)
        
        # 选择前npoint*2个点
        candidate_indices = sorted_indices[:min(npoint*2, N)]
        
        # 在这些候选点中应用随机采样或简化版FPS
        if candidate_indices.shape[0] <= npoint:
            centroids[b, :candidate_indices.shape[0]] = candidate_indices
            # 如果候选点不足，随机补充
            if candidate_indices.shape[0] < npoint:
                remaining = npoint - candidate_indices.shape[0]
                extra_indices = torch.randperm(N, device=device)[:remaining]
                centroids[b, candidate_indices.shape[0]:] = extra_indices
        else:
            # 简化版FPS在候选点上
            selected = torch.zeros(npoint, dtype=torch.long, device=device)
            distances = torch.ones(candidate_indices.shape[0], device=device) * 1e10
            farthest = 0  # 从最重要的点开始
            
            for i in range(npoint):
                selected[i] = candidate_indices[farthest]
                centroid = xyz_t[candidate_indices[farthest]].view(1, 3)
                
                # 计算到所有候选点的距离
                candidate_pts = xyz_t[candidate_indices]
                dist = torch.sum((candidate_pts - centroid) ** 2, dim=1)
                
                mask = dist < distances
                distances[mask] = dist[mask]
                farthest = torch.max(distances, dim=0)[1]
            
            centroids[b] = selected
    
    return centroids


class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, sampling='random'):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.sampling = sampling  # 'random', 'fps', 或 'approx_fps', 'voxel', 'hierarchical', 'hybrid', 'feature'

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
            elif self.sampling == 'voxel':
                # 体素采样
                idx = voxel_downsample(xyz, self.npoint)
            elif self.sampling == 'hierarchical':
                # 分层采样
                idx = hierarchical_sampling(xyz, self.npoint)
            elif self.sampling == 'hybrid':
                # 混合采样
                idx = hybrid_sampling(xyz, self.npoint)
            elif self.sampling == 'feature' and points is not None:
                # 特征重要性采样
                idx = feature_based_sampling(xyz, points, self.npoint)
            elif self.sampling == 'approx_fps':
                # 近似FPS采样 慢
                idx = approximate_fps(xyz, self.npoint)
            elif self.sampling == 'fps':
                # 标准FPS采样慢
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
        # self.sa1 = SetAbstraction(npoint=2048, radius=0.1, nsample=16, in_channel=feature_dim, mlp=[32, 32, 64], sampling='random')
        # self.sa2 = SetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=64, mlp=[64, 64, 128], sampling='random')
        # self.sa3 = SetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128, mlp=[128, 128, 256], sampling='random')
        # self.sa4 = SetAbstraction(npoint=None, radius=None, nsample=1, in_channel=256, mlp=[256, 512, 1024], sampling='random')

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
        _, points3 = self.sa3(xyz2, points2)  # points3: (B, 1024, N')
        x = torch.max(points3, dim=2)[0] # 全局最大池化，降维到 (B, 1024)

        # xyz1, points1 = self.sa1(xyz, points)    
        # xyz2, points2 = self.sa2(xyz1, points1)   
        # xyz3, points3 = self.sa3(xyz2, points2)   
        # _, points4 = self.sa4(xyz3, points3)      
        # x = torch.max(points4, dim=2)[0] # 全局最大池化，降维到 (B, 1024)

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

