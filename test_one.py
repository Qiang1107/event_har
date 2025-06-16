import os
import yaml
import torch
import numpy as np
import argparse
import random
from models.model import VitModel
from models.backbones.cnn import CNN_model

def predict_single_npy(config_path, model_path, npy_path, window_size=9):
    """
    对单个.npy文件进行预测
    
    参数:
        config_path: 配置文件路径
        model_path: 模型文件路径
        npy_path: 要预测的.npy文件路径
        window_size: 窗口大小，默认为9帧
    """
    # 1. 加载配置
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 2. 加载数据
    try:
        data = np.load(npy_path).astype(np.float32)  # (N, H, W, C)
        print(f"加载文件: {npy_path}")
        print(f"数据形状: {data.shape}")
    except Exception as e:
        print(f"无法加载 {npy_path}: {e}")
        return
    
    # 3. 检查数据帧数是否足够
    if data.shape[0] < window_size:
        print(f"警告: 数据帧数 ({data.shape[0]}) 少于窗口大小 ({window_size})")
        return
    
    # 4. 随机选择一个起始帧
    max_start_idx = data.shape[0] - window_size
    start_idx = random.randint(0, max_start_idx)
    clip = data[start_idx:start_idx+window_size]  # (window_size, H, W, C)
    
    print(f"选择帧范围: {start_idx} 到 {start_idx+window_size-1}")
    
    # 5. 数据归一化处理 (与RGBESequenceDataset中相同的处理)
    # RGBE归一化
    clip[..., :3] /= 255.0
    # e = clip[..., 3]
    # clip[..., 3] = np.where(e == 0, 0.0, 1.0)
    
    # 6. 调整数据格式以匹配模型输入
    # 从(window_size, H, W, C)转换为(1, window_size, C, H, W)
    clip = np.transpose(clip, (0, 3, 1, 2))  # (window_size, C, H, W)
    clip = np.expand_dims(clip, axis=0)  # (1, window_size, C, H, W)
    
    # 转换为PyTorch tensor
    input_tensor = torch.from_numpy(clip)
    print(f"输入张量形状: {input_tensor.shape}")  # (1, window_size, C, H, W)
    # 等比例缩放到 128*128
    _, _, c, h, w = input_tensor.shape
    if h != 128 or w != 128:
        # 计算缩放因子
        resize_transform = torch.nn.functional.interpolate
        input_tensor = resize_transform(
            input_tensor.view(-1, c, h, w),  # 重塑为 (batch*window_size, c, h, w)
            size=(128, 128),
            mode='bilinear',
            align_corners=False
        )
        input_tensor = input_tensor.view(1, window_size, c, 128, 128)  # 恢复原来的形状
        print(f"缩放后张量形状: {input_tensor.shape}")  # (1, window_size, C, 128, 128)
    
    # 7. 加载模型
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    
    # 选择模型
    # model = VitModel(cfg).to(device)
    model = CNN_model(cfg).to(device)
    
    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"模型已加载: {model_path}")
    
    # 8. 进行预测
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        outputs = model(input_tensor)
        print(f"输出形状: {outputs.shape}")  # (1, num_classes)
        _, predicted = torch.max(outputs, 1)
        
        # 获取类别标签
        action_classes = cfg.get('classes', [f"Class_{i}" for i in range(outputs.size(1))])
        print(f"类别标签: {action_classes}")
        print(f"预测类别索引: {predicted.item()}")
        
        # 获取所有类别的概率
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        print("\n预测结果:")
        print(f"预测类别: {action_classes[predicted.item()]}")
        print("\n所有类别概率:")
        
        # 打印所有类别的概率
        for i, prob in enumerate(probabilities):
            class_name = action_classes[i] if i < len(action_classes) else f"Class_{i}"
            print(f"{class_name}: {prob.item():.4f}")
        
        # 找到前三高的概率
        top3_prob, top3_indices = torch.topk(probabilities, min(3, len(probabilities)))
        
        print("\n概率最高的三个类别:")
        for i, idx in enumerate(top3_indices):
            class_name = action_classes[idx.item()] if idx.item() < len(action_classes) else f"Class_{idx.item()}"
            print(f"{i+1}. {class_name}: {top3_prob[i].item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对单个.npy文件进行行为识别预测")
    parser.add_argument("--config", type=str, default="/home/qiangubuntu/research/har_rgbe/configs/har_rgb.yaml",
                        help="配置文件路径")
    parser.add_argument("--model", type=str, default="/home/qiangubuntu/research/har_rgbe/results/checkpoints/cnn_rgb_1.pth",
                        help="预训练模型路径")
    parser.add_argument("--npy", type=str, default="/home/qiangubuntu/research/har_rgbe/utils/train_data/rgb/test/Screw/5_0.npy",
                        help="要预测的.npy文件路径")
    parser.add_argument("--window", type=int, default=10,
                        help="窗口大小 (默认: 9)")
    
    args = parser.parse_args()
    
    predict_single_npy(args.config, args.model, args.npy, args.window)