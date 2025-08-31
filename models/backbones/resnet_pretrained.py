import torch
import torch.nn as nn
import torchvision.models as models


class PretrainedResNet_model(nn.Module):
    """
    使用预训练ResNet的序列分类模型
    输入 x: Tensor [B, T, C, H, W]
    输出 logits: Tensor [B, num_classes]
    """
    def __init__(self, cfg: dict):
        super(PretrainedResNet_model, self).__init__()
        
        resnetpre_cfg = cfg['resnet_pretrained_model']
        self.num_classes = resnetpre_cfg['num_classes']
        model_name = resnetpre_cfg.get('model_name', 'resnet18')
        pretrained = resnetpre_cfg.get('pretrained', True)
        temporal_strategy = resnetpre_cfg.get('temporal_strategy', 'frame_avg')

        # 加载预训练模型
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # 移除最后的分类层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # 时间序列处理
        self.temporal_strategy = temporal_strategy
        if temporal_strategy == 'lstm':
            self.lstm = nn.LSTM(feature_dim, 256, batch_first=True)
            self.classifier = nn.Linear(256, self.num_classes)
        elif temporal_strategy == 'transformer':
            self.temporal_attention = nn.MultiheadAttention(feature_dim, 8)
            self.classifier = nn.Linear(feature_dim, self.num_classes)
        elif temporal_strategy == 'gru':
            self.gru = nn.GRU(feature_dim, 256, batch_first=True)
            self.classifier = nn.Linear(256, self.num_classes)
        else:  # frame_avg
            self.classifier = nn.Linear(feature_dim, self.num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(resnetpre_cfg.get('dropout', 0.5))

    def forward(self, x):
        """
        输入: x [B, T, C, H, W]
        输出: logits [B, num_classes]
        """
        B, T, C, H, W = x.shape
        
        # 将时间维度与批次维度合并
        x = x.view(B * T, C, H, W)  # [B*T, C, H, W]
        
        # 特征提取
        features = self.backbone(x)  # [B*T, feature_dim, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [B*T, feature_dim]
        
        # 重新组织时间维度
        features = features.view(B, T, -1)  # [B, T, feature_dim]
        
        # 时间序列处理
        if self.temporal_strategy == 'lstm':
            _, (h_n, _) = self.lstm(features)
            output = h_n[-1]  # [B, 256]
        elif self.temporal_strategy == 'gru':
            _, h_n = self.gru(features)
            output = h_n[-1]  # [B, 256]
        elif self.temporal_strategy == 'transformer':
            # 转置为(T, B, D)格式
            features_t = features.transpose(0, 1)
            attn_output, _ = self.temporal_attention(features_t, features_t, features_t)
            output = attn_output.mean(dim=0)  # [B, feature_dim]
        else:  # frame_avg
            output = features.mean(dim=1)  # [B, feature_dim]
        
        # 分类
        output = self.dropout(output)
        logits = self.classifier(output)
        
        return logits