# models/heads/classification_head.py
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_prob=0.0):
        """
        input_dim: 输入特征向量长度（即颈部输出维度）
        num_classes: 分类类别数,本任务中为12
        dropout_prob: 可选的dropout概率,可用于防止过拟合
        """
        super().__init__()
        # self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()
        # self.fc = nn.Linear(input_dim, num_classes)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, num_classes)
)
    
    def forward(self, x):
        """
        输入: x 张量,形状 (N, input_dim)
        输出: 分类 logits,形状 (N, num_classes)
        """
        logits = self.classifier(x)
        
        return logits
