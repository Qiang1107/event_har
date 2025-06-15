# models/necks/sequence_neck.py
import torch
import torch.nn as nn

class SequenceNeck(nn.Module):
    def __init__(self, input_dim, hidden_dim, mode='mean'):
        """
        input_dim: 每帧特征向量的维度 (骨干输出维度)
        hidden_dim: LSTM的隐藏层维度 (序列融合后的特征维度,用于head)
        mode: 时序建模方式
            - 'lstm': 用 LSTM 提取序列特征
            - 'mean': 对时间维度做平均池化
            - 'none': 直接返回输入特征(跳过 neck)
        """
        super().__init__()
        assert mode in ['lstm', 'mean', 'none']
        self.mode = mode
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if self.mode == 'lstm':
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        elif self.mode == 'mean':
            self.proj = nn.Linear(input_dim, hidden_dim)
        # 'none' 模式下不定义层，直接 passthrough

    def forward(self, x):
        """
        输入：
        - 'lstm' 或 'mean': x shape [B, T, D], 其中 B=batch_size, T=帧数, D=每帧特征维度
        - 'none': x shape [B, D]

        输出：
        - 全部返回 序列融合后的特征向量,shape: [B, hidden_dim]
        """
        if self.mode == 'lstm':
            _, (h_n, _) = self.lstm(x)  # h_n: [1, B, hidden_dim]
            return h_n[0]               # [B, hidden_dim]

        elif self.mode == 'mean':
            x = x.mean(dim=1)          # [B, D]
            return self.proj(x)        # [B, hidden_dim]

        elif self.mode == 'none':
            return x                   # [B, D], D 就是 hidden_dim
