import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
# from utils.extensions import mosaic_frames


def mosaic_frames(x):
    """
    将视频帧拼接成大图
    输入: x [B, T, C, H, W]
    输出: big_image [B, C, H*k, W*k] 其中 k = ceil(sqrt(T))
    """
    B, T, C, H, W = x.shape
    
    # 计算拼接网格大小 k x k，使得 k^2 >= T
    k = math.ceil(math.sqrt(T))
    
    # 如果T不能完全填满k x k网格，用零填充
    if T < k * k:
        padding_frames = k * k - T
        padding = torch.zeros(B, padding_frames, C, H, W, dtype=x.dtype, device=x.device)
        x = torch.cat([x, padding], dim=1)  # [B, k*k, C, H, W]
    
    # 重新组织为网格形状
    x = x.view(B, k, k, C, H, W)  # [B, k, k, C, H, W]
    
    # 拼接成大图
    # 先在H维度拼接
    rows = []
    for i in range(k):
        row = torch.cat([x[:, i, j] for j in range(k)], dim=3)  # [B, C, H, W*k]
        rows.append(row)
    
    # 再在W维度拼接
    big_image = torch.cat(rows, dim=2)  # [B, C, H*k, W*k]
    
    return big_image


class PatchEmbedding(nn.Module):
    """将图像分割成patches并嵌入"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # 使用卷积层实现patching和线性嵌入
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # 动态计算patch数量（支持不同尺寸的输入）
        n_patches_h = H // self.patch_size
        n_patches_w = W // self.patch_size
        
        # 打平成序列: [B, embed_dim, n_patches_h * n_patches_w]
        x = self.proj(x)
        x = x.flatten(2)  # [B, embed_dim, n_patches]
        x = x.transpose(1, 2)  # [B, n_patches, embed_dim]
        
        return x, (n_patches_h, n_patches_w)


class Attention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV投影
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape
        
        # 计算QKV: [B, N, 3*D] -> [3, B, n_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个形状: [B, n_heads, N, head_dim]
        
        # 计算注意力权重: [B, n_heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 加权聚合值向量: [B, n_heads, N, head_dim] -> [B, N, D]
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """前馈网络"""
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, 
                 drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, drop=drop)
        
    def forward(self, x):
        # 残差连接
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VideoVisionTransformer(nn.Module):
    """支持视频序列的Vision Transformer - 使用帧拼接策略"""
    def __init__(self, cfg: dict):
        super().__init__()
        vit_cfg = cfg['vit_model']
        self.img_size = vit_cfg['img_size']
        self.patch_size = vit_cfg['patch_size']
        self.in_channels = vit_cfg['in_channels']
        self.num_classes = vit_cfg['num_classes']
        self.embed_dim = vit_cfg['embed_dim']
        self.depth = vit_cfg['depth']
        self.n_heads = vit_cfg['num_heads']
        self.mlp_ratio = vit_cfg['mlp_ratio']
        self.qkv_bias = vit_cfg['qkv_bias']
        self.drop_rate = vit_cfg['drop_rate']
        self.attn_drop_rate = vit_cfg['attn_drop_rate']
        self.emb_drop = vit_cfg['emb_drop']
        
        # 时间序列处理方式
        self.temporal_strategy = vit_cfg.get('temporal_strategy', 'mosaic_frames')
        
        # Patch嵌入（支持动态尺寸）
        self.patch_embed = PatchEmbedding(
            img_size=self.img_size, 
            patch_size=self.patch_size, 
            in_channels=self.in_channels, 
            embed_dim=self.embed_dim
        )
        
        # 可学习的分类token（位置编码将动态生成）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=self.emb_drop)
        
        # Transformer编码器
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=self.embed_dim, n_heads=self.n_heads, mlp_ratio=self.mlp_ratio, 
                qkv_bias=self.qkv_bias, drop=self.drop_rate, attn_drop=self.attn_drop_rate)
            for _ in range(self.depth)
        ])
        self.norm = nn.LayerNorm(self.embed_dim)
        
        # 如果使用mosaic策略，添加帧重组模块
        if self.temporal_strategy == 'mosaic_frames':
            self.frame_aggregator = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.GELU(),
                nn.Dropout(self.drop_rate),
                nn.Linear(self.embed_dim, self.embed_dim)
            )
        
        # 分类头
        self.head = nn.Linear(self.embed_dim, self.num_classes)
        
        # 初始化权重
        self._init_weights()
    
    def _generate_pos_embed(self, n_patches_h, n_patches_w):
        """动态生成位置编码"""
        # 生成2D位置编码
        pos_embed = torch.zeros(1, n_patches_h * n_patches_w + 1, self.embed_dim)
        
        # 简化的2D正弦位置编码
        position = torch.arange(n_patches_h * n_patches_w).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * 
                           -(math.log(10000.0) / self.embed_dim))
        
        pos_embed[0, 1:, 0::2] = torch.sin(position * div_term)
        pos_embed[0, 1:, 1::2] = torch.cos(position * div_term)
        
        return pos_embed.to(self.cls_token.device)
    
    def _init_weights(self):
        # 初始化分类token
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # 初始化线性层
        self.apply(self._init_weights_helper)
    
    def _init_weights_helper(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_single_frame(self, x):
        """处理单帧图像: [B, C, H, W] -> [B, embed_dim]"""
        B = x.shape[0]
        
        # Patch嵌入
        x, (n_patches_h, n_patches_w) = self.patch_embed(x)  # [B, n_patches, embed_dim]
        
        # 添加分类token
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_token, x), dim=1)  # [B, n_patches + 1, embed_dim]
        
        # 生成并添加位置编码
        pos_embed = self._generate_pos_embed(n_patches_h, n_patches_w)
        x = x + pos_embed
        x = self.pos_drop(x)
        
        # 通过Transformer编码器
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # 返回分类token的特征
        return x[:, 0]  # [B, embed_dim]
    
    def forward_mosaic_frames(self, x):
        """
        使用帧拼接策略处理视频序列
        输入: x [B, T, C, H, W]
        输出: logits [B, num_classes]
        """
        B, T, C, H, W = x.shape
        
        # 1. 将T帧拼接成大图
        big_image = mosaic_frames(x)  # [B, C, H*k, W*k]
        
        # 2. 对大图进行patch嵌入
        patches, (n_patches_h, n_patches_w) = self.patch_embed(big_image)  # [B, k*k*patches_per_frame, embed_dim]
        
        # 3. 添加分类token
        cls_token = self.cls_token.expand(B, -1, -1)
        x_combined = torch.cat((cls_token, patches), dim=1)
        
        # 4. 生成并添加位置编码
        pos_embed = self._generate_pos_embed(n_patches_h, n_patches_w)
        x_combined = x_combined + pos_embed
        x_combined = self.pos_drop(x_combined)
        
        # 5. 通过Transformer编码器
        for block in self.blocks:
            x_combined = block(x_combined)
        
        x_combined = self.norm(x_combined)
        
        # 6. 提取特征
        cls_features = x_combined[:, 0]  # [B, embed_dim]
        
        # 7. 可选：对帧特征进行聚合处理
        if hasattr(self, 'frame_aggregator'):
            # 计算每个原始帧对应的patch数量
            k = math.ceil(math.sqrt(T))
            patches_per_original_frame = (H // self.patch_size) * (W // self.patch_size)
            
            # 重新组织patch特征以对应原始帧
            frame_patches = x_combined[:, 1:].view(B, k, k, patches_per_original_frame, self.embed_dim)
            
            # 提取前T个帧的特征（去除填充）
            frame_features = []
            for t in range(min(T, k*k)):
                i, j = t // k, t % k
                frame_feat = frame_patches[:, i, j].mean(dim=1)  # 对每帧的patches求平均
                frame_features.append(frame_feat)
            
            if len(frame_features) < T:
                # 如果帧数不足，用零填充
                for _ in range(T - len(frame_features)):
                    frame_features.append(torch.zeros_like(frame_features[0]))
            
            # 聚合帧特征
            frame_features = torch.stack(frame_features[:T], dim=1)  # [B, T, embed_dim]
            aggregated_features = self.frame_aggregator(frame_features.mean(dim=1))  # [B, embed_dim]
            
            # 结合cls_token特征和聚合的帧特征
            final_features = cls_features + aggregated_features
        else:
            final_features = cls_features
        
        return self.head(final_features)
    
    def forward(self, x):
        """
        输入: x [B, T, C, H, W] - 视频序列 或 [B, C, H, W] - 单帧
        输出: logits [B, num_classes]
        """
        # 检查输入维度
        if x.dim() == 4:
            # 单帧输入: [B, C, H, W]
            features = self.forward_single_frame(x)
            return self.head(features)
        
        elif x.dim() == 5:
            # 视频序列输入: [B, T, C, H, W]
            if self.temporal_strategy == 'mosaic_frames':
                return self.forward_mosaic_frames(x)
            else:
                # 回退到原来的策略
                B, T, C, H, W = x.shape
                frame_features = []
                for t in range(T):
                    frame_feat = self.forward_single_frame(x[:, t])
                    frame_features.append(frame_feat)
                features = torch.stack(frame_features, dim=1).mean(dim=1)
                return self.head(features)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}, expected [B,C,H,W] or [B,T,C,H,W]")


# 向后兼容
VisionTransformer = VideoVisionTransformer