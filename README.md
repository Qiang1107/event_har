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

## 帧格式的事件数据的处理流程
1. 使用save_data_to_png.py获取rgb数据的png图片，得到 rgb_png
2. 基于 rgb_png，继续使用align_eventimg.py对事件数据进行时间对齐，得到 time_aligned_event_png
（如果是csv文件，可选，可通过align_eventcsv.py对事件数据进行时间对齐，得到 time_aligned_event_csv）
3. 基于 time_aligned_event_png，继续使用align_eventimg.py对事件数据进行像素对齐，得到 pixel_aligned_event_png
（手动对齐像素）
4. 基于 pixel_aligned_event_png
对事件数据进行单通道变换，得到 single_channel_event
多通道变换，得到 multiple_channels_RGBE
5. 整理各个动作的起始帧和结束帧，得到action_list_0428.xlsx 和 action_list_abnorm_0428.xlsx
6. 对数据进行动作分类: /home/qiangubuntu/research/data_collection/src/dataprocessing/zarr_categorize.py
保存在zarr_categorize文件夹
7. 对分好类的数据进行[训练 验证 测试]的划分: /home/qiangubuntu/research/event_har/utils/clip_dataset.py

## csv格式的事件数据的处理流程
1. 使用save_data_to_png.py获取rgb数据的png图片，得到 rgb_png
(可选，使用save_data_to_png.py处理event_buffer.zarr 中的事件帧数据，得到event_png，仅用做可视化)
2. 整理各个动作的起始帧和结束帧，得到action_list_0628_8.xlsx
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



