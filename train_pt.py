import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
import pickle
from datasets.rgbe_sequence_dataset import RGBESequenceDataset
from datasets.event_sequence_dataset import ESequenceDataset


# # 读取数据
# DATA_PATH = "processed_data_1"
# train_data, train_labels = torch.load(os.path.join(DATA_PATH, "train.pt"), weights_only=False)
# val_data, val_labels = torch.load(os.path.join(DATA_PATH, "val.pt"), weights_only=False)

# train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=4, shuffle=True)
# val_loader = DataLoader(TensorDataset(val_data, val_labels), batch_size=4, shuffle=False)

pnet2_data_dir = "preprocessing_data"
pnet2_train_path = os.path.join(pnet2_data_dir, "train_dataset.pkl")
with open(pnet2_train_path, 'rb') as f:
    train_ds = pickle.load(f)
pnet2_val_path = os.path.join(pnet2_data_dir, "val_dataset.pkl")
with open(pnet2_val_path, 'rb') as f:
    val_ds = pickle.load(f)
# pnet2_test_path = os.path.join(pnet2_data_dir, "test_dataset.pkl")
# with open(pnet2_test_path, 'rb') as f:
#     test_ds = pickle.load(f)       
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)




class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super(SetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        self.conv1 = nn.Conv2d(in_channel + 3, mlp[0], 1)
        self.conv2 = nn.Conv2d(mlp[0], mlp[1], 1)
        self.conv3 = nn.Conv2d(mlp[1], mlp[2], 1)
        self.bn1 = nn.BatchNorm2d(mlp[0])
        self.bn2 = nn.BatchNorm2d(mlp[1])
        self.bn3 = nn.BatchNorm2d(mlp[2])

    def forward(self, xyz, points):
        B, _, N = xyz.shape

        if self.npoint is not None:
            idx = torch.randperm(N)[:self.npoint]  # 简化版 FPS
            new_xyz = xyz[:, :, idx].contiguous()
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


class PointNetPlusPlus(nn.Module):
    def __init__(self, num_classes=12, input_dim=5):
        super(PointNetPlusPlus, self).__init__()
        self.input_dim = input_dim
        feature_dim = input_dim - 3  # 特征维度：5-3=2


        self.sa1 = SetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=feature_dim, mlp=[64, 64, 128])
        self.sa2 = SetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128, mlp=[128, 128, 256])
        self.sa3 = SetAbstraction(npoint=None, radius=None, nsample=1, in_channel=256, mlp=[256, 512, 1024])

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        # 检查输入维度
        if x.dim() == 3 and x.size(2) != self.input_dim and x.size(1) == self.input_dim:
            # 已经是 B x C x N 格式
            pass
        else:
            # 确保输入是 B x N x C 格式，然后转为 B x C x N
            x = x.permute(0, 2, 1)
        
        B, C, N = x.shape
        if C != self.input_dim:
            print(f"警告：输入维度 {C} 与预期维度 {self.input_dim} 不匹配")

        # 输入形状：(B, C, N)，例如 (16, 4, N)
        xyz = x[:, :3, :]  # (B, 3, N) - 前3维是坐标
        points = x[:, 3:, :]  # (B, 2, N) - 后2维是特征

        xyz1, points1 = self.sa1(xyz, points)
        xyz2, points2 = self.sa2(xyz1, points1)
        xyz3, points3 = self.sa3(xyz2, points2)  # points3: (B, 1024, N')

        # 全局最大池化，降维到 (B, 1024)
        x = torch.max(points3, dim=2)[0]

        # 处理批次大小为1的情况
        if x.size(0) == 1 and self.training:
            # 复制样本以创建批次大小为2的批次
            x = torch.cat([x, x], dim=0)
            x = self.dropout(F.relu(self.bn1(self.fc1(x))))
            x = self.dropout(F.relu(self.bn2(self.fc2(x))))
            x = self.fc3(x)
            # 只保留原始样本的预测
            return x[:1]
        else:
            # 全连接层
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            x = self.dropout(x)
            x = self.fc3(x)
            return x


# 训练设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PointNetPlusPlus(num_classes=12, input_dim=5).to(device) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.00025)
criterion = nn.CrossEntropyLoss()

# 训练循环
epochs = 100
for epoch in range(epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for i, (data, labels) in enumerate(train_loader):
        # 增加数据维度检查
        if i == 0:
            print(f"数据形状: {data.shape}, 标签形状: {labels.shape}")

        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(data)  # 现在输出应为 (16, 4)
        loss = criterion(output, labels)  # 批量大小应匹配
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(output, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss /= len(train_loader)
    train_accuracy = correct / total * 100
    print(f"第 [{epoch + 1}/{epochs}] 轮, 训练损失: {train_loss:.4f}, 训练准确率: {train_accuracy:.2f}%")

    # 验证
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            loss = criterion(output, labels)
            val_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)
    val_accuracy = correct / total * 100
    print(f"✅ 第 [{epoch + 1}/{epochs}] 轮, 验证损失: {val_loss:.4f}, 验证准确率: {val_accuracy:.2f}%")

# 保存模型
torch.save(model.state_dict(), "results/checkpoints/pointnet_plus_plus.pth")
print("✅ 训练完成，模型已保存！")