# models/model.py
import yaml
import math
import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.necks.sequence_neck import SequenceNeck
from models.heads.classification_head import ClassificationHead
from models.backbones.vit import ViT


def mosaic_frames(x: torch.Tensor) -> torch.Tensor:
    # x: [B, T, C, H, W]
    B, T, C, H, W = x.shape
    # 计算最近的上界整数平方尺寸 k
    k = math.ceil(math.sqrt(T))
    total_patches = k * k
    # 如有需要，补充零帧至 T 达到 k^2
    if T < total_patches:
        pad_frames = total_patches - T
        pad_shape = (B, pad_frames, C, H, W)
        pad_tensor = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        x = torch.cat([x, pad_tensor], dim=1)  # 在时间维度拼接零帧
    # 重塑并permute维度，将 k×k 个帧拼成大图
    x = x.view(B, k, k, C, H, W)                       # [B, k, k, C, H, W]
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()       # [B, C, k, H, k, W]
    x = x.view(B, C, k * H, k * W)                     # [B, C, H*k, W*k]
    return x


class MLP_model(nn.Module):
    """
    MLP 序列分类模型
    输入 x: Tensor [B, T, 4, H, W]
    输出 logits: Tensor [B, num_classes]
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B , T * C * H * W)  # 展平
        logits = self.fc(x)
        return logits


class CNN_model(nn.Module):
    """
    CNN 序列分类模型
    输入 x: Tensor [B, T, C, H, W]
    输出 logits: Tensor [B, num_classes]
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(input_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二层卷积
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三层卷积
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 计算卷积后的特征维度 (假设输入是224x224)
        feature_size = 128 * (224 // 8) * (224 // 8)    # 
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        # 将时间维度与批次维度合并
        x = x.view(B * T, C, H, W)  # [B*T, C, H, W]
        # 通过卷积层
        x = self.conv_layers(x)    # [B*T, conv_layers最后一层输出维度, H/8, W/8]
        # 展平并通过全连接层  
        x = self.fc(x)             # [B*T, output_dim]
        # 重新组织时间维度并进行时间维度上的平均
        x = x.view(B, T, -1)       # [B, T, output_dim]
        x = x.mean(dim=1)          # [B, output_dim]
        return x


class VitModel(nn.Module):
    """
    ViTPose+LSTM 序列分类模型
    输入 x: Tensor [B, T, 4, H, W]
    输出 logits: Tensor [B, output_dim]
    """
    def __init__(self, cfg: dict):
        super().__init__()
        backbone_cfg = cfg['model']['backbone']
        neck_cfg = cfg['model']['neck']
        head_cfg = cfg['model']['head']

        # 1) Backbone：ViTPose
        self.backbone = ViT(**backbone_cfg)
        # 2) Neck：序列特征融合（LSTM 或平均池化）
        self.neck = SequenceNeck(**neck_cfg)
        # 3) Head：分类层
        self.head = ClassificationHead(**head_cfg)

        # 添加一个全局平均池化层用于处理特征图
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):

        # 1) 将 T 帧拼接成大图 [B, C, H*k, W*k]
        big_image = mosaic_frames(x)  # 调用上述拼图函数
        # 2) 一次性送入 ViT 提取特征
        feat_map = self.backbone(big_image)  # 输出 [B, embed_dim, Hp, Wp]
        # 此时 Hp = k, Wp = k（每个patch等于原始帧大小）
        # 3) 提取序列特征或全局特征用于后续任务
        patch_tokens = feat_map.flatten(2).transpose(1, 2)  # [B, Hp*Wp, embed_dim]
        # 如需要保持与原有管线一致的每帧特征序列：
        if T < patch_tokens.shape[1]:
            patch_tokens = patch_tokens[:, :T, :]  # 去除末尾填充的空白patch
        # patch_tokens 形状现为 [B, T, embed_dim]，每帧对应一个特征向量
        # 后续可接入序列Neck（如LSTM或均值池化）或直接用于分类头
        return patch_tokens
    
    def forward(self, x):
        B, T, C, H, W = x.shape

        # 1) Backbone：ViTPose - 逐帧处理
        frame_features = []
        for t in range(T):
            # 提取单帧
            frame = x[:, t]  # [B, C, H, W]
            
            # 通过ViT backbone提取特征
            feat = self.backbone(frame)  # [B, embed_dim, Hp, Wp]
            
            # 全局池化提取全局特征
            feat = self.global_pool(feat)  # [B, embed_dim, 1, 1]
            feat = feat.flatten(1)  # [B, embed_dim]
            
            # 收集每一帧的特征
            frame_features.append(feat)
        
        # 将所有帧特征堆叠成序列
        seq_features = torch.stack(frame_features, dim=1)  # [B, T, embed_dim]
        return seq_features
        
        # # 2) Neck：序列特征融合 (LSTM或平均池化)
        # seq_features = self.neck(seq_features)  # [B, neck_output_dim]
        
        # # 3) Head：最终分类
        # logits = self.head(seq_features)  # [B, output_dim]
        
        # return logits

# python -m models.model
if __name__ == '__main__':
    config_path='/home/qiangubuntu/research/har_rgbe/configs/har_rgbe.yaml'
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    # peusdo data
    x = torch.randn(8, 10, 4, 128, 128).to('cuda')  # [B, T, C, H, W] 
    # model
    model = VitModel(cfg).to('cuda')
    # model = MLP_model(input_dim=4, output_dim=12).to('cuda')
    # model = CNN_model(input_dim=4, output_dim=12).to('cuda') # input_dim和C一致, output_dim是类别
    # forward
    results = model(x)
    print("[results.shape]: ", results.shape) # [B, output_dim]