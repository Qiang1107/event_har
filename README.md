# Event & Human action recognition
## Training Script

`train.py` supports selecting different backbones and loading a pretrained
model before training. Use the `--model_type` flag to choose between `cnn`,
`vit` and `pointnet2`. PointNet2 uses the configuration block `pointnet2_model`
in the yaml file. The `--pretrained` argument can be used for ViT models:


```
python train.py --config path/to/config.yaml \
                --model path/to/save_best.pth \
                --log path/to/log.txt \
                --pretrained path/to/pretrained_weights.pth
```

# pointnet++网络模型 log 说明
## test_data_0628_8_ecount_1.pkl 32768 4096
trainlog_pointnet2_event_0628_8_ecount_6.txt 94.63%

## test_data_0628_8_ecount_2.pkl 16384 2048
trainlog_pointnet2_event_0628_8_ecount_7.txt 95.83%

## test_data_0628_8_ecount_3.pkl 8192 1024  
trainlog_pointnet2_event_0628_8_ecount_8.txt 96.47%
trainlog_pointnet2_event_0628_8_ecount_9.txt 重复8的实验 96.30%
trainlog_pointnet2_event_0628_8_ecount_10.txt epoch100 p2v1    96.85%
trainlog_pointnet2_event_0628_8_ecount_11.txt epoch30 p2msgv1  97.41%

## test_data_0628_8_ecount_3_vote.pkl 8192 1024
对于每个样本增加了时间戳，用于简单多数投票预测

## test_data_0628_8_ecount_4.pkl 8192 1024
增加了时间戳。取消了roi。修改了xyt顺序，原先为txy
trainlog_pointnet2_event_0628_8_14.txt epoch30 p2v1    fpsfps采样 93.24%
trainlog_pointnet2_event_0628_8_15.txt epoch30 p2v1    ramram采样 94.80%
trainlog_pointnet2_event_0628_8_16.txt epoch30 p2v1    ramfps采样 95.24%
trainlog_pointnet2_event_0628_8_13.txt epoch30 p2v3    ramram采样 95.98%
trainlog_pointnet2_event_0628_8_17.txt epoch30 p2v3    ramfps采样 96.21%
trainlog_pointnet2_event_0628_8_12.txt epoch30 p2msgv1 ramram采样 96.80% 同19
trainlog_pointnet2_event_0628_8_18.txt epoch30 p2msgv1 ramfps采样 96.05%
trainlog_pointnet2_event_0628_8_19.txt epoch30 p2msgv1 ramram采样 96.35% 同12
trainlog_pointnet2_event_0628_8_20.txt epoch30 p2msgv3 ramfps采样 97.09%
trainlog_pointnet2_event_0628_8_21.txt epoch30 p2msgv3 ramram采样 97.94%
trainlog_pointnet2_event_0628_8_22.txt epoch30 p2msgv3 hiefps采样 
trainlog_pointnet2_event_0628_8_23.txt epoch30 p2msgv3 ramhie采样 



# resnet网络模型
## resnet 预训练模型下载位置
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /home/qiang_qin/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
## resnet log 说明



# Data Processing
## 数据说明
保存的文件夹: /home/qiangubuntu/research/data_collection/src/data/
data_buffer.zarr            原始rgb+d数据
event_buffer.zarr           原始event数据，包含事件帧和时间戳
event_csv                   原始event数据对应的csv文件，包含timestamp,x,y,polarity
event_png                   原始event数据的png图片
video                       rgb数据的视频

## 生成的数据说明
rgb_png                     rgb数据的png图片
time_aligned_event_png      时间对齐的事件数据png图片
pixel_aligned_event_png     像素对齐的事件数据png图片
single_channel_event_png    单通道事件数据

## 帧格式的事件数据和RGB数据 的处理流程 (rgb数据小于event数据)
1. 使用save_data_to_png.py获取rgb数据的png图片，得到 rgb_png
同理可以得到事件数据的png图片，得到 event_png
2. 基于 rgb_png，使用align_eventimg.py对事件数据进行时间对齐，得到 time_aligned_event_png
2. (可选) 如果是csv文件，可通过align_eventcsv.py对事件数据进行时间对齐，得到 time_aligned_event_csv
3. 基于 time_aligned_event_png，继续使用align_eventimg.py对事件数据进行像素对齐，得到 pixel_aligned_event_png
（手动对齐像素）
4. 基于 pixel_aligned_event_png
对事件数据进行单通道变换，得到 single_channel_event
多通道变换，得到 multiple_channels_RGBE
5. 整理各个动作的RGB的起始帧和结束帧，得到action_list_0428.xlsx 和 action_list_abnorm_0428.xlsx
6. 对数据进行动作分类: /home/qiangubuntu/research/data_collection/src/dataprocessing/zarr_categorize.py
保存在zarr_categorize文件夹
7. 对分好类的数据进行[训练 验证 测试]的划分: /home/qiangubuntu/research/event_har/utils/clip_dataset.py

## 帧格式的事件数据和RGB数据 的处理流程 (rgb数据大于event数据)
1. 使用save_data_to_png.py获取rgb数据的png图片，得到 rgb_png
同理可以得到事件数据的png图片，得到 event_png
2. rgb数据大于event数据，但已经通过rgb整理了各个动作的起始帧和结束帧，因此和上面步骤保持一致。
基于 rgb_png，使用align_eventimg.py对事件数据进行时间对齐，得到 time_aligned_event_png
3. 基于 time_aligned_event_png，继续使用align_eventimg.py对事件数据进行像素对齐，得到 pixel_aligned_event_png
（手动对齐像素）
4. 基于 pixel_aligned_event_png, 使用 to_rgbe_channels.py
对事件数据进行单通道变换，得到 single_channel_event
多通道变换，得到 multiple_channels_RGBE
5. 整理各个动作的起始帧和结束帧，得到action_list_0428.xlsx 和 action_list_abnorm_0428.xlsx
6. 对数据进行动作分类: /home/qiangubuntu/research/data_collection/src/dataprocessing/zarr_categorize.py
保存在zarr_categorize文件夹
7. 对分好类的数据进行[训练 验证 测试]的划分: /home/qiangubuntu/research/event_har/utils/clip_dataset.py

## csv格式的仅事件数据 的处理流程 (rgb数据小于event数据)
1. 使用save_data_to_png.py获取rgb数据的png图片，得到 rgb_png
(可选，使用save_data_to_png.py处理event_buffer.zarr 中的事件帧数据，得到event_png，仅用做可视化)
2. 整理各个动作的RGB的起始帧和结束帧，得到action_list_0628_8.xlsx
3. 对csv数据进行整合: eventcsv_concatenate.py
保存在: /home/qiangubuntu/research/data_collection/src/data/[idx]/event_csv_concatenated
4. 对数据进行动作分类: /home/qiangubuntu/research/data_collection/src/dataprocessing/eventcsv_categorize.py
保存在eventcsv_categorize文件夹
**
eventcsv_categorize_0428_12.py 用来处理4月28日的数据action_list_0428.xlsx，12种动作
eventcsv_categorize_0428_10.py 用来处理4月28日的数据action_list_0428.xlsx，10种动作,没有idle
eventcsv_categorize_0628_8.py 用来处理6月28日的数据action_list_0628.xlsx，8种动作,没有idle
**
5. 对分好类的数据进行[训练 验证 测试]的划分: /home/qiangubuntu/research/event_har/utils/clip_dataset.py

