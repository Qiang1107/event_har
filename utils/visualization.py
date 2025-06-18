import re
import matplotlib.pyplot as plt
import os
import numpy as np


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
    val_time_matches = re.findall(r'Validation time:\s*([\d\.]+)', content)
    val_acc_matches = re.findall(r'Val Acc:\s*([\d\.]+)', content)
    train_loss_matches = re.findall(r'Train Loss:\s*([\d\.]+)', content)
    train_time_matches = re.findall(r'Training time:\s*([\d\.]+)', content)

    # 匹配形如 "Align_screwdriver: 0.8630 (126/146)" 中的准确率值
    action_acc_matches = re.findall(r'(\w+):\s*([\d\.]+)\s*\((\d+)/(\d+)\)', content)
    action_accs = {}
    if action_acc_matches:
        for action, acc, correct, total in action_acc_matches:
            action_accs[action] = float(acc)
        # print(f"action_accs: {action_accs}")
    
    # 转成数字类型
    epochs = list(map(int, epoch_matches))
    val_losses = list(map(float, val_loss_matches))
    val_times = list(map(float, val_time_matches))
    val_accs = list(map(float, val_acc_matches))
    train_losses = list(map(float, train_loss_matches))
    train_times = list(map(float, train_time_matches))
    
    # 构建结果字典
    results = {
        'epochs': epochs,
        'val_losses': val_losses,
        'val_times': val_times,
        'val_accs': val_accs,
        'train_losses': train_losses,
        'train_times': train_times,
        **action_accs  # Add all entries from action_accs dictionary
    }
    # print(results)

    # 检查数据一致性
    main_keys = ['epochs', 'val_losses', 'val_times', 'val_accs', 'train_losses', 'train_times']
    lengths = {k: len(results[k]) for k in main_keys if k in results}
    print(f"解析到的数据长度: {lengths}")
    if len(set(lengths.values())) > 1:
        print(f"警告: 解析到的数据长度不一致！{lengths}")

    return results


if __name__ == "__main__":
    os.makedirs('results/figs', exist_ok=True)    
    # ====== 配置区 ======
    Log_path   = 'results/logs/training_log_pointnet2_event_7.txt'  # 日志文件

    # None  # 如果指定，保存为该文件，否则直接 plt.show()
    val_acc_save_path  = 'results/figs/pointnet2_event_7_val_acc.png' 
    train_loss_save_path  = 'results/figs/pointnet2_event_7_train_loss.png'
    trainval_loss_save_path  = 'results/figs/pointnet2_event_7_trainval_loss.png'
    trainval_time_save_path  = 'results/figs/pointnet2_event_7_trainval_time.png'
    test_acc_save_path  = 'results/figs/pointnet2_event_7_test_acc.png' 
    manually_save_path = 'results/figs/all_models_test_acc.png' 
    # ====================

    results = parse_log(Log_path)

    epochs = results['epochs']
    val_losses = results['val_losses']
    val_times = results['val_times']
    val_accs = results['val_accs']
    train_losses = results['train_losses']
    train_times = results['train_times']
    
    # 交互选择要绘制的图表
    print("请选择绘制哪种图:")
    print("1 - Validation Accuracy")
    print("2 - Training Loss")
    print("3 - Training & Validation Loss")
    print("4 - Training & Validation Time")
    print("5 - Test Accuracy")
    print("6 - All Models' Test Accuracy")
    
    choice = input("请输入选项编号(1-n): ")
    
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
        if val_acc_save_path:
            # plt.show()
            plt.savefig(val_acc_save_path, dpi=150)
            print(f"图已保存到 {val_acc_save_path}")
        else:
            plt.show()
        
    elif choice == '2':
        # 绘制训练损失图
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, marker='o', linestyle='-')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Epoch vs Training Loss')
        plt.grid(True)
        plt.tight_layout()
        if train_loss_save_path:
            # plt.show()
            plt.savefig(train_loss_save_path, dpi=150)
            print(f"图已保存到 {train_loss_save_path}")
        else:
            plt.show()
    
    elif choice == '3':
        # 绘制训练和验证损失图
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, marker='o', linestyle='-', label='Training Loss')
        plt.plot(epochs, val_losses, marker='s', linestyle='-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if trainval_loss_save_path:
            # plt.show()
            plt.savefig(trainval_loss_save_path, dpi=150)
            print(f"图已保存到 {trainval_loss_save_path}")
        else:
            plt.show()

    elif choice == '4':
        # 绘制训练和验证的时间图
        plt.figure(figsize=(10, 6))
        # 主图显示训练和验证时间
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
        plt.tight_layout()
        if trainval_time_save_path:
            # plt.show()
            plt.savefig(trainval_time_save_path, dpi=150)
            print(f"图已保存到 {trainval_time_save_path}")
        else:
            plt.show()
    
    elif choice == '5':
        # 绘制测试准确率柱形图
        test_accs = {}
        for key, value in results.items():
            if key not in ['epochs', 'val_losses', 'val_times', 'val_accs', 
              'train_losses', 'train_times']:
                if key == 'Acc':  # Change 'Acc' key to 'Overall'
                    test_accs['Overall'] = value
                else:
                    test_accs[key] = value
        if not test_accs:
            print("Warning: No action accuracy data found in the log file.")
            exit()
        # 颜色方案选项
        color_schemes = {
            '1': 'skyblue',  # 原始颜色
            '2': "#19547e",  # 自定义色彩，可选
            '3': ['#e74c3c', '#8e44ad', '#3498db', '#2ecc71', '#f1c40f', 
                 '#e67e22', '#1abc9c', '#2c3e50', '#95a5a6', '#d35400'] # 参考颜色
        }
        actions = list(test_accs.keys())
        accuracies = list(test_accs.values())
        plt.figure(figsize=(10, 6))
        bars = plt.bar(actions, accuracies, color='#19547e')
        plt.xlabel('Actions')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy for Each Action')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        # 添加每个柱子的数值标签
        for i, acc in enumerate(accuracies):
            if actions[i] == 'Overall':  # 为Overall添加特殊标签
                plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center', va='bottom',
                         fontweight='bold', color='black', fontsize=11)
            else:
                plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center', va='bottom')
        plt.tight_layout()
        if test_acc_save_path:
            # plt.show()
            plt.savefig(test_acc_save_path, dpi=150)
            print(f"图已保存到 {test_acc_save_path}")
        else:
            plt.show()

    elif choice == '6':
        # 手动绘制柱形图，手动设置xy数据。
        color_schemes = {
            '1': 'skyblue',  # 原始颜色
            '2': "#19547e",  # 自定义色彩，可选
            '3': ['#e74c3c', '#8e44ad', '#3498db', '#2ecc71', '#f1c40f', 
                 '#e67e22', '#1abc9c', '#2c3e50', '#95a5a6', '#d35400'] # 参考颜色
        }
        x_val = ['Align_screwdriver', 'Align_wrench', 'Align_hammer',
                   'Align_saw', 'Align_pliers', 'Align_screwdriver_2', 
                   'Align_wrench_2', 'Align_hammer_2', 'Align_saw_2', 
                   'Align_pliers_2', 'Overall']
        y_val = [0.8630, 0.8700, 0.8500, 0.8400, 0.8300, 
                      0.8600, 0.8800, 0.8700, 0.8600, 0.8500, 0.8654]
        plt.figure(figsize=(10, 6))
        bars = plt.bar(x_val, y_val, color='#19547e')
        plt.xlabel('Model Sets')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy for Different Model Sets')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        # 添加每个柱子的数值标签
        for i, acc in enumerate(y_val):
            plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center', va='bottom')
        plt.tight_layout()
        if manually_save_path:
            # plt.show()
            plt.savefig(manually_save_path, dpi=150)
            print(f"图已保存到 {manually_save_path}")
        else:
            plt.show()

    else:
        print("无效选项,请选择1-5")
        exit()
 
    
