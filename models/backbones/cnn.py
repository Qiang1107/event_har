import torch
import torch.nn as nn


class CNN_model(nn.Module):
    """
    CNN 序列分类模型
    输入 x: Tensor [B, T, C, H, W]
    输出 logits: Tensor [B, num_classes]
    """
    def __init__(self, cfg: dict):
        super().__init__()
        # 从配置中获取 CNN 模型参数
        cnn_cfg = cfg['cnn_model']
        input_dim = cnn_cfg['input_dim']
        output_dim = cnn_cfg['output_dim']
        # 如果没有提供 input_width 和 input_height，则使用默认值
        input_width = cnn_cfg.get('input_width', 192)  
        input_height = cnn_cfg.get('input_height', 256)

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
        
        # 自动计算卷积后的特征维度
        with torch.no_grad():
            # 创建一个虚拟输入张量来计算特征大小
            dummy_input = torch.zeros(1, input_dim, input_height, input_width)
            dummy_output = self.conv_layers(dummy_input)
            feature_size = dummy_output.numel()
        
        # print(f"Feature size after conv layers: {feature_size}")
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