import re
import matplotlib.pyplot as plt
import os

def parse_log(log_path):
    """
    从训练日志中提取 epoch 序号和对应的训练指标。

    Args:
        log_path (str): 日志文件路径

    Returns:
        dict: 包含 'epochs', 'val_accs', 'val_losses' 等训练指标的字典
    """
    with open(log_path, 'r') as f:
        content = f.read()

    # 匹配形如 "[Epoch 3/30]" 中的 epoch 序号
    epoch_matches = re.findall(r'\[Epoch\s+(\d+)\s*/\s*\d+\]', content)
    # 匹配形如 "Val Loss: 0.1234, Val Acc: 0.8123" 中的 Val Loss 和 Val Acc
    val_loss_matches = re.findall(r'Val Loss:\s*([\d\.]+)', content)
    val_acc_matches = re.findall(r'Val Acc:\s*([\d\.]+)', content)
    train_loss_matches = re.findall(r'Train Loss:\s*([\d\.]+)', content)
    train_time_matches = re.findall(r'Training time:\s*([\d\.]+)', content)
    val_time_matches = re.findall(r'Validation time:\s*([\d\.]+)', content)

    # 转成数字类型
    epochs = list(map(int, epoch_matches))
    val_losses = list(map(float, val_loss_matches))
    val_accs = list(map(float, val_acc_matches))
    train_losses = list(map(float, train_loss_matches))
    train_times = list(map(float, train_time_matches))
    val_times = list(map(float, val_time_matches))

    # 构建结果字典
    results = {
        'epochs': epochs,
        'val_accs': val_accs,
        'val_losses': val_losses,
        'train_losses': train_losses,
        'train_times': train_times,
        'val_times': val_times
    }

    # 检查数据一致性
    lengths = {k: len(v) for k, v in results.items()}
    print(f"解析到的数据长度: {lengths}")
    if len(set(lengths.values())) > 1:
        print(f"警告: 解析到的数据长度不一致！{lengths}")

    return results


if __name__ == "__main__":
    os.makedirs('results/figs', exist_ok=True)    
    # ====== 配置区 ======
    Log_path   = 'results/logs/training_log_vit_rgbe_1.txt'  # 日志文件
    save_path  = 'results/figs/vit_rgbe_1_val_acc.png' # None  # 如果指定，保存为该文件，否则直接 plt.show()
    # save_path  = 'results/figs/vit_rgbe_1_train_loss.png'
    # save_path  = 'results/figs/vit_rgbe_1_train_val_time.png'

    # Log_path   = 'results/logs/training_log_vit_rgbd_1.txt'  # 日志文件
    # save_path  = 'results/figs/vit_rgbd_1_val_acc.png' # None  # 如果指定，保存为该文件，否则直接 plt.show()
    # save_path  = 'results/figs/vit_rgbd_1_train_loss.png'
    # save_path  = 'results/figs/vit_rgbd_1_train_val_time.png'

    # Log_path   = 'results/logs/training_log_vit_rgb_1.txt'  # 日志文件
    # save_path  = 'results/figs/vit_rgb_3_val_acc.png' # None  # 如果指定，保存为该文件，否则直接 plt.show()
    # save_path  = 'results/figs/vit_rgb_3_train_loss.png'
    # save_path  = 'results/figs/vit_rgb_3_train_val_time.png'

    # Log_path   = 'results/logs/training_log_vit_event_1.txt'  # 日志文件
    # save_path  = 'results/figs/vit_event_1_val_acc.png' # None  # 如果指定，保存为该文件，否则直接 plt.show()
    # save_path  = 'results/figs/vit_event_1_train_loss.png'
    # save_path  = 'results/figs/vit_event_1_train_val_time.png'
    # ====================

    results = parse_log(Log_path)

    epochs = results['epochs']
    val_accs = results['val_accs']
    train_losses = results['train_losses']
    train_times = results['train_times']
    val_times = results['val_times']

    # 交互选择要绘制的图表
    print("请选择绘制哪种图:")
    print("1 - Validation Accuracy")
    print("2 - Training Loss")
    print("3 - Training & Validation Time")
    
    choice = input("请输入选项编号(1-3): ")
    
    if choice == '1':
        # 绘制验证准确率图
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, val_accs, marker='o', linestyle='-')
        # 找出最大值并标注
        max_acc_idx = val_accs.index(max(val_accs))
        max_acc = val_accs[max_acc_idx]
        max_epoch = epochs[max_acc_idx]
        # 在最大值点处添加特殊标记
        plt.plot(max_epoch, max_acc, 'ro', markersize=8)
        # 添加文本标注
        plt.annotate(f'Max: {max_acc:.4f}', 
                    xy=(max_epoch, max_acc),
                    xytext=(max_epoch + 1, max_acc),  # 文本位置稍微偏右
                    ha='left')  # 左对齐，使文本从指定位置向右延伸
        # 找出最后一个值并标注
        last_acc_idx = -1  # 最后一个值的索引
        last_acc = val_accs[last_acc_idx]
        last_epoch = epochs[last_acc_idx]
        # 在最后一个值点处添加特殊标记
        plt.plot(last_epoch, last_acc, 'rs', markersize=8)  
        # 添加文本标注
        plt.annotate(f'Last: {last_acc:.4f}', 
                     xy=(last_epoch, last_acc),
                     xytext=(last_epoch, last_acc + 0.005),  # 文本位置稍微偏上
                     ha='center')  # 中心对齐，使文本从指定位置向上延伸
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.title('Epoch vs Validation Accuracy')
        plt.grid(True)
        # 设置坐标轴从0开始
        # plt.xlim(0, max(epochs))
        # plt.ylim(0, 1.0)  # 假设准确率最高为1
        plt.tight_layout()
        
    elif choice == '2':
        # 绘制训练损失图
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, marker='o', linestyle='-')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Epoch vs Training Loss')
        plt.grid(True)
        plt.tight_layout()
        
    elif choice == '3':
        # 绘制训练和验证的时间图
        plt.figure(figsize=(10, 6))
        # 主图显示训练和验证时间
        plt.subplot(2, 1, 1)
        plt.plot(epochs, train_times, marker='o', linestyle='-', label='Training Time')
        plt.plot(epochs, val_times, marker='s', linestyle='-', label='Validation Time')
        # 计算平均时间
        avg_train_time = sum(train_times) / len(train_times)
        avg_val_time = sum(val_times) / len(val_times)
        # 添加平均时间水平线
        plt.axhline(y=avg_train_time, color='r', linestyle='--', alpha=0.7)
        plt.axhline(y=avg_val_time, color='g', linestyle='--', alpha=0.7)
        # 添加平均时间标注
        plt.text(max(epochs), avg_train_time - 1, f'Avg: {avg_train_time:.2f}s', 
                 va='top', ha='right', color='r')
        plt.text(max(epochs), avg_val_time + 1, f'Avg: {avg_val_time:.2f}s', 
                 va='bottom', ha='right', color='g')
        plt.xlabel('Epoch')
        plt.ylabel('Time (s)')
        plt.title('Training & Validation Time per Epoch')
        plt.legend()
        plt.grid(True)
        # 子图显示时间比例
        plt.subplot(2, 1, 2)
        total_times = [t + v for t, v in zip(train_times, val_times)]
        plt.plot(epochs, total_times, marker='D', linestyle='-', label='Total Time')
        plt.fill_between(epochs, 0, train_times, alpha=0.3, label='Train')
        plt.fill_between(epochs, train_times, total_times, alpha=0.3, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Time (s)')
        plt.title('Time Distribution per Epoch')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    else:
        print("无效选项,请选择1-3")
        exit()
 
    if save_path:
        # plt.show()
        plt.savefig(save_path, dpi=150)
        print(f"图已保存到 {save_path}")
    else:
        plt.show()
