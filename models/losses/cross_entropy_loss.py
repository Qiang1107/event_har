# models/losses/cross_entropy_loss.py
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # PyTorch 的交叉熵损失 (logits 会内部计算 softmax)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        """
        preds: 模型输出的 logits，形状 (N, num_classes)
        targets: 真实标签，形状 (N,) ，类型为 LongTensor
        返回: 标量损失值 (tensor)
        """
        return self.loss_fn(preds, targets)
