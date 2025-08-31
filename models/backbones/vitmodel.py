import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from models.backbones.vit_utils import ViT
from utils.extensions import mosaic_frames


class VitModel(nn.Module):
    """
    ViTPose+LSTM 序列分类模型
    输入 x: Tensor [B, T, C, H, W]
    输出 logits: Tensor [B, num_classes]
    """

    def __init__(self, cfg: dict):
        super().__init__()

        # 提取配置
        vit_cfg = cfg['vit_model']
        neck_input_dim = vit_cfg['neck_input_dim']
        neck_hidden_dim = vit_cfg['neck_hidden_dim'] 
        head_num_classes = vit_cfg['head_num_classes']
        head_dropout_prob = vit_cfg['head_dropout_prob']

        img_size = vit_cfg['img_size']
        patch_size = vit_cfg['patch_size']
        in_channels = vit_cfg['in_channels']
        embed_dim = vit_cfg['embed_dim']
        depth = vit_cfg['depth']
        num_heads = vit_cfg['num_heads']
        mlp_ratio = vit_cfg['mlp_ratio']
        qkv_bias = vit_cfg['qkv_bias']
        qk_scale = vit_cfg['qk_scale']
        drop_rate = vit_cfg['drop_rate']
        attn_drop_rate = vit_cfg['attn_drop_rate']
        drop_path_rate = vit_cfg['drop_path_rate']
        hybrid_backbone = vit_cfg['hybrid_backbone']
        norm_layer = vit_cfg['norm_layer']
        use_checkpoint = vit_cfg['use_checkpoint']
        frozen_stages = vit_cfg['frozen_stages']
        ratio = vit_cfg['ratio']
        last_norm = vit_cfg['last_norm']
        patch_padding = vit_cfg['patch_padding']
        freeze_attn = vit_cfg['freeze_attn']
        freeze_ffn = vit_cfg['freeze_ffn']

        # 1) Backbone：ViT
        self.backbone = ViT(img_size=img_size, patch_size=patch_size, in_chans=in_channels, num_classes=head_num_classes, embed_dim=embed_dim, 
                            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, 
                            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, hybrid_backbone=hybrid_backbone, norm_layer=norm_layer, 
                            use_checkpoint=use_checkpoint, frozen_stages=frozen_stages, ratio=ratio, last_norm=last_norm, patch_padding=patch_padding, freeze_attn=freeze_attn, freeze_ffn=freeze_ffn)

        # 2) Neck：序列特征融合（LSTM 或平均池化）
        self.neck_mode = vit_cfg['neck_mode']
        if self.neck_mode == 'lstm':
            self.lstm = nn.LSTM(neck_input_dim, neck_hidden_dim, batch_first=True)
        elif self.neck_mode == 'mean':
            self.proj = nn.Linear(neck_input_dim, neck_hidden_dim)

        # 3) Head：分类层
        self.classifier = nn.Sequential(
            nn.Dropout(head_dropout_prob),
            nn.Linear(neck_hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(head_dropout_prob),
            nn.Linear(64, head_num_classes)
        )

        # 添加一个全局平均池化层用于处理特征图
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        B, T, C, H, W = x.shape

        # 将 T 帧拼接成大图 [B, C, H*k, W*k]
        big_image = mosaic_frames(x) 
        # print("大图尺寸 for yaml:", big_image.shape)

        # 1) Backbone：ViTPose - 逐帧处理
        feat_map = self.backbone(big_image)  # [B, embed_dim, Hp, Wp]
        # 此时 Hp = k, Wp = k（每个patch等于原始帧大小）
        patch_tokens = feat_map.flatten(2).transpose(1, 2)  # [B, Hp*Wp, embed_dim]
        # 如需要保持与原有管线一致的每帧特征序列：
        if T < patch_tokens.shape[1]:
            patch_tokens = patch_tokens[:, :T, :]  # 去除末尾填充的空白patch
        # patch_tokens 形状现为 [B, T, embed_dim]，每帧对应一个特征向量
        # 后续可接入序列Neck（如LSTM或均值池化）或直接用于分类头
        
        # 2) Neck：序列特征融合 (LSTM或平均池化)
        if self.neck_mode == 'lstm':
            _, (h_n, _) = self.lstm(patch_tokens)
            features = h_n[0]
        elif self.neck_mode == 'mean':
            features = self.proj(patch_tokens.mean(dim=1)) # [B, neck_output_dim]
        else:  # none
            features = patch_tokens.mean(dim=1)
        
        # 3) Head：最终分类
        logits = self.classifier(features) # [B, output_dim]
        
        return logits

    