import os
import yaml
import time
import torch
from torch.utils.data import DataLoader
from datasets.rgbe_sequence_dataset import RGBESequenceDataset
from models.model import VitModel
from models.backbones.cnn import CNN_model
from models.losses.cross_entropy_loss import CrossEntropyLoss
import tqdm

def main(config_path, premodel_path, log_path):
    # 1. 加载配置
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 2. 只构造测试数据集
    ds = cfg['dataset']
    test_ds = RGBESequenceDataset(
        data_root          = ds['test_dir'],
        window_size        = ds['window_size'],
        stride             = ds['stride'],
        enable_transform   = ds['enable_transform']
    )

    test_loader = DataLoader(
        test_ds,
        **cfg['test']
    )

    # 3. 构建模型
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    # model = VitModel(cfg).to(device)
    model = CNN_model(cfg).to(device)

    # 4. 加载预训练模型
    if premodel_path is None:
        premodel_path = os.path.join(cfg['work_dir'], 'cnn_rgbe_1.pth')
    
    if os.path.exists(premodel_path):
        model.load_state_dict(torch.load(premodel_path, map_location=device))
        print(f"Loaded model from {premodel_path}")
    else:
        raise FileNotFoundError(f"Model file not found at {premodel_path}")

    # 创建日志目录和文件
    if log_path is None:
        log_path = os.path.join(cfg['log_dir'], 'training_log_cnn_rgbe_1.txt')

    if not os.path.exists(cfg['log_dir']):
        os.makedirs(cfg['log_dir'], exist_ok=True)
    
    # 5. 测试评估
    test_start_time = time.time()
    print("Testing...")
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in tqdm.tqdm(test_loader):
            imgs   = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            
            # Print predictions and ground truth for each sample
            # for i in range(len(preds)):
            #     print(f"Sample {total+i+1}: Predicted={preds[i].item()}, Ground Truth={labels[i].item()}")
                
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    
    test_acc = correct / total
    test_end_time = time.time()
    
    # 记录测试时间和结果
    test_time = test_end_time - test_start_time
    print(f"Test time: {test_time:.2f} seconds")
    print(f"Test Acc: {test_acc:.4f}")

    with open(log_path, 'a') as f:
        f.write(f"\nModel loaded from: {premodel_path}\n")
        f.write(f"Test time: {test_time:.2f} seconds\n")
        f.write(f"Test Acc: {test_acc:.4f} ({correct}/{total})\n")
        f.write(f"Test statistics: {len(test_ds)} samples in {len(test_loader)} batches\n")
    
    return test_acc

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/home/qiangubuntu/research/har_rgbe/configs/har_rgbe.yaml',
                        help='Path to your_action_config.yaml')
    parser.add_argument('--model', type=str, default='/home/qiangubuntu/research/har_rgbe/results/checkpoints/cnn_rgbe_1.pth',
                        help='Path to the pre-trained model')
    parser.add_argument('--log', type=str, default='/home/qiangubuntu/research/har_rgbe/results/logs/training_log_cnn_rgbe_1.txt',
                        help='Path to the log file')
    args = parser.parse_args()
    main(args.config, args.model, args.log)
    
