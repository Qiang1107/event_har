import os
import yaml
import time
import torch
import torch.nn as nn
import tqdm
import pickle


from torch.utils.data import DataLoader
from datasets.rgbe_sequence_dataset import RGBESequenceDataset
from datasets.event_sequence_dataset import ESequenceDataset
from datasets.event_count_seq_dataset import ECountSeqDataset
from models.model import VitModel
from models.backbones.cnn import CNN_model
from models.backbones.pointnet2 import PointNet2Classifier
from models.backbones.pointnet2msg import PointNet2MSGClassifier
# from models.losses.cross_entropy_loss import CrossEntropyLoss
from utils.weight_utils import load_vitpose_pretrained


def main(config_path, best_model_path, log_path, pretrained_path=None):
    # 1. 加载配置
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    model_type = cfg.get('model_type', 'cnn')

    # 2. 构造 Dataset & DataLoader
    ds = cfg['dataset']
    if model_type in ['pointnet2', 'pointnet2msg']:
        pnet2_data_dir = "preprocessing_data"
        pnet2_train_pkl = "train_dataset10_eseq.pkl"
        pnet2_train_path = os.path.join(pnet2_data_dir, pnet2_train_pkl)
        print(f"Start loading training dataset from {pnet2_train_path}")
        with open(pnet2_train_path, 'rb') as f:
            train_ds = pickle.load(f)
        print(f"Loaded training dataset with {len(train_ds)} samples")
        
        pnet2_val_path = os.path.join(pnet2_data_dir, "val_dataset10_eseq.pkl")
        print(f"Start loading validation dataset from {pnet2_val_path}")
        with open(pnet2_val_path, 'rb') as f:
            val_ds = pickle.load(f)
        print(f"Loaded validation dataset with {len(val_ds)} samples")
        
        pnet2_test_path = os.path.join(pnet2_data_dir, "test_dataset10_eseq.pkl")
        print(f"Start loading test dataset from {pnet2_test_path}")
        with open(pnet2_test_path, 'rb') as f:
            test_ds = pickle.load(f)
        print(f"Loaded test dataset with {len(test_ds)} samples")

        train_loader = DataLoader(train_ds, **cfg['train'])
        val_loader = DataLoader(val_ds, **cfg['val'])
        test_loader = DataLoader(test_ds, **cfg['test'])
        print("DataLoader created for PointNet2 datasets.")
        
    else:
        train_ds = RGBESequenceDataset(
            data_root          = ds['train_dir'],
            window_size        = ds['window_size'],
            stride             = ds['stride'],
            enable_transform   = ds['enable_transform'],
            label_map          = ds['label_map']
        )
        val_ds = RGBESequenceDataset(
            data_root          = ds['val_dir'],
            window_size        = ds['window_size'],
            stride             = ds['stride'],
            enable_transform   = ds['enable_transform'],
            label_map          = ds['label_map']
        )
        test_ds = RGBESequenceDataset(
            data_root          = ds['test_dir'],
            window_size        = ds['window_size'],
            stride             = ds['stride'],
            enable_transform   = ds['enable_transform'],
            label_map          = ds['label_map']
        )
        train_loader = DataLoader(
            train_ds,
            **cfg['train']
        )
        val_loader = DataLoader(
            val_ds,
            **cfg['val']
        )
        test_loader = DataLoader(
            test_ds,
            **cfg['test']
        )


    # 3. 构建模型、损失、优化器  
    # —— 模型 损失函数 —— 
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    if model_type == 'vit':
        model = VitModel(cfg).to(device)
        if pretrained_path:
            load_vitpose_pretrained(model, pretrained_path)
            print(f"Loaded pretrained model from {pretrained_path}")
        loss_fn = nn.CrossEntropyLoss()
    elif model_type == 'pointnet2':
        model = PointNet2Classifier(cfg).to(device)
        # loss_fn = nn.NLLLoss()
        loss_fn = nn.CrossEntropyLoss()
    elif model_type == 'pointnet2msg':
        model = PointNet2MSGClassifier(cfg).to(device)
        loss_fn = nn.CrossEntropyLoss()
    elif model_type == 'cnn':
        model = CNN_model(cfg).to(device)
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # —— 优化器 ——
    optim_cfg = cfg['optimizer']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optim_cfg['lr']),
        weight_decay=float(optim_cfg['weight_decay']),
    )

    # —— 带预热的余弦退火学习率调度器 —— 
    cosine_warmup_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    # —— StepLR 学习率调度器 ——
    step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # —— 余弦退火学习率调度器 —— 
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=1e-6)
    # —— 带预热的线性衰减调度器 —— 
    def warmup_linear_decay(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 1.0 - 0.9 * (epoch - warmup_epochs) / (cfg['epochs'] - warmup_epochs)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_linear_decay)
    # —— 多步衰减学习率调度器 —— 
    milestones = [int(cfg['epochs']*0.5), int(cfg['epochs']*0.75)]
    multistep_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    # —— 余弦退火带热重启 —— 
    warm_restart_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    # —— One Cycle学习率调度器 —— 
    steps_per_epoch = len(train_loader)
    onecycle_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=float(optim_cfg['lr'])*10, 
        steps_per_epoch=steps_per_epoch, epochs=cfg['epochs']
    )

    # 使用其中一个调度器 
    # scheduler = step_scheduler

    # 4. 创建日志目录和文件
    if log_path is None:
        log_path = os.path.join(cfg['log_dir'], 'training_log_tmp.txt')
    os.makedirs(cfg['log_dir'], exist_ok=True)
    os.makedirs(cfg['work_dir'], exist_ok=True)
    with open(log_path, 'a') as f:
        f.write(" -----General Configuration------\n")
        f.write(f" Window_size: {ds['window_size']}\n")
        f.write(f" Stride: {ds['stride']}\n")
        f.write(f" Epochs: {cfg['epochs']}\n")
        f.write(f" Train batch size: {cfg['train']['batch_size']}\n")
        f.write(f" Validation batch size: {cfg['val']['batch_size']}\n")
        f.write(f" Test batch size: {cfg['test']['batch_size']}\n")
        f.write(f" learning rate: {optim_cfg['lr']}\n")
        f.write(f" weight decay: {optim_cfg['weight_decay']}\n")
        f.write(f" Model type: {model_type}\n")
        if model_type == 'vit':
            f.write(f" ------ViT Model Configuration------\n")
            f.write(f" ViT Model: {cfg['vit_model']}\n")
            if pretrained_path is not None:
                f.write(f" Loaded pretrained model: {pretrained_path}\n")
        if model_type == 'cnn':
            f.write(f" ------CNN Model Configuration------\n")
            f.write(f" CNN Model: {cfg['cnn_model']}\n")
        if model_type in ['pointnet2', 'pointnet2msg']:
            f.write(f" ------Pointnet2 Model Configuration------\n")
            f.write(f" Loaded training data from: {pnet2_train_path}\n")
            f.write(f" Pointnet2 Model: {cfg['pointnet2_model']}\n")
            f.write(f" window_size_us: {ds['window_size_us']}\n")
            f.write(f" stride_us: {ds['stride_us']}\n")
            f.write(f" max_points: {ds['max_points']}\n")
            f.write(f" t_squash_factor: {ds['t_squash_factor']}\n")
            f.write(f" target_width: {ds['target_width']}\n")
            f.write(f" target_height: {ds['target_height']}\n")
            f.write(f" min_events_per_window: {ds['min_events_per_window']}\n")
            f.write(f" window_size_event_count: {ds['window_size_event_count']}\n")
            f.write(f" step_size: {ds['step_size']}\n")
    
    # 5. 训练、验证、测试
    best_acc = 0.0
    for epoch in range(cfg['epochs']):
        # —— 训练 —— 
        print(f"[Epoch {epoch+1}/{cfg['epochs']}]:")
        print("Training...")
        train_start_time = time.time()
        model.train()
        total_loss = 0.0
        for imgs, labels in tqdm.tqdm(train_loader):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_end_time = time.time()
        avg_train_loss = total_loss / len(train_loader)

        print(f"Train Loss: {avg_train_loss:.4f}")
        num_train_batches = len(train_loader)
        num_train_samples = len(train_ds)
        print(f"Training statistics: {num_train_samples} samples in {num_train_batches} batches")
        train_time = train_end_time - train_start_time
        print(f"Training time: {train_time:.2f} seconds")
        with open(log_path, 'a') as f:
            f.write(f"\n[Epoch {epoch+1}/{cfg['epochs']}]\n")
            f.write(f"Train Loss: {avg_train_loss:.4f}\n")
            f.write(f"Training statistics: {num_train_samples} samples in {num_train_batches} batches\n")
            f.write(f"Training time: {train_time:.2f} seconds\n")
        
        # —— 验证 —— 
        print("Validating...")
        val_start_time = time.time()
        model.eval()
        val_loss = correct = total = 0.0
        with torch.no_grad():
            for imgs, labels in tqdm.tqdm(val_loader):
                imgs = imgs.float().to(device)
                labels = labels.to(device)
                logits = model(imgs)
                val_loss += loss_fn(logits, labels).item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        val_end_time = time.time()
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        num_val_batches = len(val_loader)
        num_val_samples = len(val_ds)
        print(f"Validation statistics: {num_val_samples} samples in {num_val_batches} batches")
        val_time = val_end_time - val_start_time
        print(f"Validation time: {val_time:.2f} seconds")
        with open(log_path, 'a') as f:
            f.write(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}\n")
            f.write(f"Validation statistics: {num_val_samples} samples in {num_val_batches} batches\n")
            f.write(f"Validation time: {val_time:.2f} seconds\n")

        # scheduler.step()
        # print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        # with open(log_path, 'a') as f:
        #     f.write(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}\n")

        # —— 保存最佳模型 —— 
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print("Saved best model.")
        print(f"-"*30)

    # —— 测试评估 —— 
    model.load_state_dict(torch.load(best_model_path))
    print(f"Loaded best model from {best_model_path}")
    
    print("Testing...")
    model.eval()
    class_correct = {}
    class_total = {}
    class_names = list(ds['label_map'].keys())
    correct = total = 0
    # Initialize counters for each class
    for class_name in class_names:
        class_correct[class_name] = 0
        class_total[class_name] = 0
    test_start_time = time.time()
    with torch.no_grad():
        for imgs, labels in tqdm.tqdm(test_loader):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)

            # Update per-class statistics
            for i in range(len(labels)):
                label_idx = labels[i].item()
                pred_idx = preds[i].item()
                label_name = class_names[label_idx]
                class_total[label_name] += 1
                if pred_idx == label_idx:
                    class_correct[label_name] += 1

            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    test_end_time = time.time()
    test_acc = correct / total
    test_time = test_end_time - test_start_time
    num_test_batches = len(test_loader)
    num_test_samples = len(test_ds)
    print(f"Test statistics: {num_test_samples} samples in {num_test_batches} batches")
    print(f"Test time: {test_time:.2f} seconds")
    print("Best validation accuracy:", best_acc)
    print(f"Test Acc: {test_acc:.4f}")
    with open(log_path, 'a') as f:
        f.write(f"\nTest with best model from {best_model_path}\n")
        f.write(f"Test statistics: {num_test_samples} samples in {num_test_batches} batches\n")
        f.write(f"Test time: {test_time:.2f} seconds\n")
        f.write(f"Best validation accuracy: {best_acc}\n")
        f.write(f"Test Acc: {test_acc:.4f} ({correct}/{total})\n")

    # Print and log per-class accuracy
    print("Per-class accuracy:")
    with open(log_path, 'a') as f:
        f.write("Per-class accuracy:\n")
        for class_name in class_names:
            if class_total[class_name] > 0:
                accuracy = class_correct[class_name] / class_total[class_name]
                print(f"{class_name}: {accuracy:.4f} ({class_correct[class_name]}/{class_total[class_name]})")
                f.write(f"{class_name}: {accuracy:.4f} ({class_correct[class_name]}/{class_total[class_name]})\n")
            else:
                print(f"{class_name}: N/A (0/0)")
                f.write(f"{class_name}: N/A (0/0)\n")
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/har_train_config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default='results/checkpoints/pointnet2_event_7.pth',
                        help='Path to save the best model')
    parser.add_argument('--log', type=str, default='results/logs/training_log_pointnet2_event_7.txt',
                        help='Path to the log file')
    parser.add_argument('--pretrained', type=str, default='pretrained/vitpose-l.pth',
                        help='Path to pre-trained weights')
    args = parser.parse_args()
    main(args.config, args.model, args.log, args.pretrained)
