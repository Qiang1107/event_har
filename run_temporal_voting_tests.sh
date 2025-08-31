#!/bin/bash

# 时间窗口投票测试脚本
# 测试不同的时间窗口大小和投票方法

echo "=== 时间窗口投票测试开始 ==="

# 基础配置
CONFIG="configs/har_test_config.yaml"
MODEL="results/checkpoints/pointnet2_event_0628_8_ecount_11.pth"

# 检查文件是否存在
if [ ! -f "$CONFIG" ]; then
    echo "配置文件不存在: $CONFIG"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "模型文件不存在: $MODEL"
    exit 1
fi

echo "配置文件: $CONFIG"
echo "模型文件: $MODEL"
echo ""

# 测试1: 不同时间窗口大小（使用多数投票）
echo "1. 测试不同时间窗口大小..."

echo "1.1 时间窗口: 0.5秒"
python3 prediction_temporal_voting.py \
    --config $CONFIG \
    --model $MODEL \
    --time_window 0.5 \
    --voting_method majority \
    --log results/logs/test_logs/temporal_voting_0.5s_majority.txt

echo "1.2 时间窗口: 1.0秒"
python3 prediction_temporal_voting.py \
    --config $CONFIG \
    --model $MODEL \
    --time_window 1.0 \
    --voting_method majority \
    --log results/logs/test_logs/temporal_voting_1.0s_majority.txt

echo "1.3 时间窗口: 2.0秒"
python3 prediction_temporal_voting.py \
    --config $CONFIG \
    --model $MODEL \
    --time_window 2.0 \
    --voting_method majority \
    --log results/logs/test_logs/temporal_voting_2.0s_majority.txt

# 测试2: 不同投票方法（使用1秒窗口）
echo ""
echo "2. 测试不同投票方法..."

echo "2.1 加权投票方法"
python3 prediction_temporal_voting.py \
    --config $CONFIG \
    --model $MODEL \
    --time_window 1.0 \
    --voting_method weighted \
    --log results/logs/test_logs/temporal_voting_1.0s_weighted.txt

echo "2.2 置信度加权投票"
python3 prediction_temporal_voting.py \
    --config $CONFIG \
    --model $MODEL \
    --time_window 1.0 \
    --voting_method confidence_weighted \
    --log results/logs/test_logs/temporal_voting_1.0s_confidence_weighted.txt

# 测试3: 细粒度时间窗口
echo ""
echo "3. 测试细粒度时间窗口..."

echo "3.1 时间窗口: 0.3秒"
python3 prediction_temporal_voting.py \
    --config $CONFIG \
    --model $MODEL \
    --time_window 0.3 \
    --voting_method majority \
    --log results/logs/test_logs/temporal_voting_0.3s_majority.txt

echo "3.2 时间窗口: 1.5秒"
python3 prediction_temporal_voting.py \
    --config $CONFIG \
    --model $MODEL \
    --time_window 1.5 \
    --voting_method majority \
    --log results/logs/test_logs/temporal_voting_1.5s_majority.txt

echo ""
echo "=== 所有测试完成! ==="
echo ""
echo "结果文件位置:"
echo "- 日志文件: results/logs/test_logs/temporal_voting_*.txt"
echo "- 分析图表: results/figs/test_figs/temporal_voting_analysis_*.png"
echo ""
echo "查看结果摘要:"
echo "grep '准确率提升' results/logs/test_logs/temporal_voting_*.txt"
