import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BasicBlock(nn.Module):
    """残差块 - 基础版本"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """残差块 - 瓶颈版本"""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_model(nn.Module):
    """
    ResNet 序列分类模型
    输入 x: Tensor [B, T, C, H, W]
    输出 logits: Tensor [B, num_classes]
    """
    def __init__(self, cfg: dict):
        super(ResNet_model, self).__init__()
        
        # 从配置中获取参数
        resnet_cfg = cfg['resnet_model']
        self.in_planes = 64
        self.num_classes = resnet_cfg['num_classes']
        self.input_channels = resnet_cfg.get('input_channels', 3)
        
        # 网络架构配置
        block_type = resnet_cfg.get('block_type', 'BasicBlock')
        num_blocks = resnet_cfg.get('num_blocks', [2, 2, 2, 2])  # ResNet18默认
        temporal_strategy = resnet_cfg.get('temporal_strategy', 'frame_avg')
        
        # 选择残差块类型
        if block_type == 'BasicBlock':
            block = BasicBlock
        elif block_type == 'Bottleneck':
            block = Bottleneck
        else:
            raise ValueError(f"Unknown block type: {block_type}")
        
        # 网络结构
        self.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, 
                              padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 时间序列处理
        self.temporal_strategy = temporal_strategy
        if temporal_strategy == 'lstm':
            self.lstm = nn.LSTM(512 * block.expansion, 256, batch_first=True)
            self.fc = nn.Linear(256, self.num_classes)
        elif temporal_strategy == 'transformer':
            self.temporal_attention = nn.MultiheadAttention(512 * block.expansion, 8)
            self.fc = nn.Linear(512 * block.expansion, self.num_classes)
        else:  # frame_avg
            self.fc = nn.Linear(512 * block.expansion, self.num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(resnet_cfg.get('dropout', 0.5))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_single_frame(self, x):
        """处理单帧图像"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        """
        输入: x [B, T, C, H, W]
        输出: logits [B, num_classes]
        """
        B, T, C, H, W = x.shape
        
        # 处理每一帧
        frame_features = []
        for t in range(T):
            frame_feat = self.forward_single_frame(x[:, t])  # [B, 512*expansion]
            frame_features.append(frame_feat)
        
        # 堆叠帧特征
        frame_features = torch.stack(frame_features, dim=1)  # [B, T, 512*expansion]
        
        # 时间序列处理
        if self.temporal_strategy == 'lstm':
            _, (h_n, _) = self.lstm(frame_features)
            features = h_n[-1]  # [B, 256]
        elif self.temporal_strategy == 'transformer':
            # 转置为(T, B, D)格式用于多头注意力
            frame_features_t = frame_features.transpose(0, 1)
            attn_output, _ = self.temporal_attention(frame_features_t, frame_features_t, frame_features_t)
            features = attn_output.mean(dim=0)  # [B, 512*expansion]
        else:  # frame_avg
            features = frame_features.mean(dim=1)  # [B, 512*expansion]
        
        # 分类
        features = self.dropout(features)
        logits = self.fc(features)
        
        return logits


def ResNet18(cfg):
    """ResNet-18模型"""
    cfg['resnet_model']['block_type'] = 'BasicBlock'
    cfg['resnet_model']['num_blocks'] = [2, 2, 2, 2]
    return ResNet_model(cfg)


def ResNet34(cfg):
    """ResNet-34模型"""
    cfg['resnet_model']['block_type'] = 'BasicBlock'
    cfg['resnet_model']['num_blocks'] = [3, 4, 6, 3]
    return ResNet_model(cfg)


def ResNet50(cfg):
    """ResNet-50模型"""
    cfg['resnet_model']['block_type'] = 'Bottleneck'
    cfg['resnet_model']['num_blocks'] = [3, 4, 6, 3]
    return ResNet_model(cfg)


def ResNet101(cfg):
    """ResNet-101模型"""
    cfg['resnet_model']['block_type'] = 'Bottleneck'
    cfg['resnet_model']['num_blocks'] = [3, 4, 23, 3]
    return ResNet_model(cfg)
