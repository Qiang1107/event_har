"""
针对同一个log文件包含多个测试阶段，提取所有测试准确率，并绘制柱形图
"""


import re
import matplotlib.pyplot as plt
import os
import numpy as np


def parse_log(log_path):
    with open(log_path, 'r') as f:
        content = f.read()

    # 查找所有形如 "Test Acc: 0.8394 (1307/1557)" 中的 0.8394，然后计算平均值
    Test_acc_matches = re.findall(r'Test Acc:\s*([\d\.]+)', content)
    Approach_acc_matches = re.findall(r'Approach:\s*([\d\.]+)', content)
    Pick_and_Place_Bolt_acc_matches = re.findall(r'Pick_and_Place_Bolt:\s*([\d\.]+)', content)
    Pick_and_Place_Cover_acc_matches = re.findall(r'Pick_and_Place_Cover:\s*([\d\.]+)', content)
    Pick_and_Place_Part1_Small_acc_matches = re.findall(r'Pick_and_Place_Part1_Small:\s*([\d\.]+)', content)
    Pick_and_Place_Part2_Big_acc_matches = re.findall(r'Pick_and_Place_Part2_Big:\s*([\d\.]+)', content)
    Pick_and_Place_Screwdriver_acc_matches = re.findall(r'Pick_and_Place_Screwdriver:\s*([\d\.]+)', content)
    Screw_acc_matches = re.findall(r'Screw:\s*([\d\.]+)', content)
    Transition_acc_matches = re.findall(r'Transition:\s*([\d\.]+)', content)

    # Convert matched accuracy strings to float values for each action
    test_acc_values = list(map(float, Test_acc_matches))
    approach_values = list(map(float, Approach_acc_matches))
    pick_and_place_bolt_values = list(map(float, Pick_and_Place_Bolt_acc_matches))
    pick_and_place_cover_values = list(map(float, Pick_and_Place_Cover_acc_matches))
    pick_and_place_part1_small_values = list(map(float, Pick_and_Place_Part1_Small_acc_matches))
    pick_and_place_part2_big_values = list(map(float, Pick_and_Place_Part2_Big_acc_matches))
    pick_and_place_screwdriver_values = list(map(float, Pick_and_Place_Screwdriver_acc_matches))
    screw_values = list(map(float, Screw_acc_matches))
    transition_values = list(map(float, Transition_acc_matches))

    # Calculate average accuracy for each action
    test_avg_acc = sum(test_acc_values) / len(test_acc_values) if test_acc_values else 0
    approach_avg = sum(approach_values) / len(approach_values) if approach_values else 0
    pick_and_place_bolt_avg = sum(pick_and_place_bolt_values) / len(pick_and_place_bolt_values) if pick_and_place_bolt_values else 0
    pick_and_place_cover_avg = sum(pick_and_place_cover_values) / len(pick_and_place_cover_values) if pick_and_place_cover_values else 0
    pick_and_place_part1_small_avg = sum(pick_and_place_part1_small_values) / len(pick_and_place_part1_small_values) if pick_and_place_part1_small_values else 0
    pick_and_place_part2_big_avg = sum(pick_and_place_part2_big_values) / len(pick_and_place_part2_big_values) if pick_and_place_part2_big_values else 0
    pick_and_place_screwdriver_avg = sum(pick_and_place_screwdriver_values) / len(pick_and_place_screwdriver_values) if pick_and_place_screwdriver_values else 0
    screw_avg = sum(screw_values) / len(screw_values) if screw_values else 0
    transition_avg = sum(transition_values) / len(transition_values) if transition_values else 0
    
    Test_time_matches = re.findall(r'Test time:\s*([\d\.]+)\s*seconds', content)
    test_time_values = list(map(float, Test_time_matches))
    test_avg_time = sum(test_time_values) / len(test_time_values) if test_time_values else 0

    # 查找所有形如 "Test statistics: 1557 samples in 195 batches" 中的 1557，然后计算平均值
    Test_samples_matches = re.findall(r'Test statistics:\s*(\d+)\s*samples', content)
    test_samples_values = list(map(int, Test_samples_matches))
    test_avg_samples = sum(test_samples_values) / len(test_samples_values) if test_samples_values else 0

    # 构建结果字典
    results = {
        'Test Acc': test_avg_acc
        , 'Approach': approach_avg
        , 'Pick_and_Place_Bolt': pick_and_place_bolt_avg
        , 'Pick_and_Place_Cover': pick_and_place_cover_avg
        , 'Pick_and_Place_Part1_Small': pick_and_place_part1_small_avg
        , 'Pick_and_Place_Part2_Big': pick_and_place_part2_big_avg
        , 'Pick_and_Place_Screwdriver': pick_and_place_screwdriver_avg
        , 'Screw': screw_avg
        , 'Transition': transition_avg
        , 'Test Time': test_avg_time
        , 'Test Samples': test_avg_samples
    }
    print(results)

    return results


if __name__ == "__main__":
    os.makedirs('results/figs', exist_ok=True)    
    # ====== 配置区 ======
    log_name = "testlog_pointnet2_event_0628_8_ecount_11"
    log_path = f"results/test_logs/{log_name}.txt"  # 日志文件

    # None  # 如果指定，保存为该文件，否则直接 plt.show()
    test_acc_save_path  = f"results/test_figs/{log_name}_average_test_acc.png"
    # ====================

    results = parse_log(log_path)
    test_avg_time = results['Test Time']
    test_avg_samples = results['Test Samples']
    
    # 交互选择要绘制的图表
    print("请选择绘制哪种图:")
    print("1 - Test Accuracy")

    
    choice = input("请输入选项编号(1-n): ")
    
    if choice == '1':
        print(f"Test Time for one simple: {(test_avg_time / test_avg_samples * 1000):.2f} ms")
        # 绘制测试准确率柱形图
        test_accs = {}
        for key, value in results.items():
            if 'Test Time' in key or 'Test Samples' in key:
                continue
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
            if actions[i] == 'Test Acc':  # 为Test Acc添加特殊标签
                plt.text(i, acc - 0.02, f'{acc*100:.2f}%', ha='center', va='top',
                         fontweight='bold', color='black', fontsize=12)
            else:
                plt.text(i, acc - 0.02, f'{acc*100:.2f}%', ha='center', va='top')
        plt.tight_layout()
        if test_acc_save_path:
            # plt.show()
            plt.savefig(test_acc_save_path, dpi=150)
            print(f"图已保存到 {test_acc_save_path}")
        else:
            plt.show()

    else:
        print("无效选项,请选择1-n")
        exit()
 
    
