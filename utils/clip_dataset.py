import os
import shutil
import random


"""
origin_data里面是12个子文件夹,每个子文件夹里面有若干个npy文件。
将每个子文件夹里面的npy文件按照613 or 622的比例随机复制到train,val,test三个文件夹中同名的子文件夹中。
# 比如子文件夹Align_screwdriver里面有32个npy文件,四舍五入,随机划分为?个,32*a个,32*b个,
并相应的复制到train,val,test三个文件夹中同名的子文件夹中。
""" 

def split_dataset(origin_data_path, output_path, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    # 确保比例之和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

    # 创建输出文件夹
    train_path = os.path.join(output_path, 'train')
    val_path = os.path.join(output_path, 'val')
    test_path = os.path.join(output_path, 'test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # 遍历origin_data中的每个子文件夹
    for subfolder in os.listdir(origin_data_path):
        subfolder_path = os.path.join(origin_data_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue  # 跳过非文件夹项

        # 创建对应的子文件夹
        train_subfolder = os.path.join(train_path, subfolder)
        val_subfolder = os.path.join(val_path, subfolder)
        test_subfolder = os.path.join(test_path, subfolder)
        os.makedirs(train_subfolder, exist_ok=True)
        os.makedirs(val_subfolder, exist_ok=True)
        os.makedirs(test_subfolder, exist_ok=True)

        # 获取子文件夹中的所有文件
        all_files = [f for f in os.listdir(subfolder_path) if f.endswith('.npy')]
        random.shuffle(all_files)  # 随机打乱文件顺序

        # 计算每个子集的大小
        total_files = len(all_files)
        val_count = int(total_files * val_ratio)
        test_count = int(total_files * test_ratio)

        # 划分数据集
        val_files = all_files[:val_count]
        test_files = all_files[val_count:val_count + test_count]
        train_files = all_files[val_count + test_count:]

        # 将文件复制到对应的文件夹
        for file in train_files:
            shutil.copy(os.path.join(subfolder_path, file), os.path.join(train_subfolder, file))
        for file in val_files:
            shutil.copy(os.path.join(subfolder_path, file), os.path.join(val_subfolder, file))
        for file in test_files:
            shutil.copy(os.path.join(subfolder_path, file), os.path.join(test_subfolder, file))

        print(f"Subfolder '{subfolder}' split completed: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test.")

if __name__ == "__main__":
    # origin_data_path = "/home/qiangubuntu/research/har_rgbe/utils/origin_data/origin_data_rgbe"
    # origin_data_path = "/home/qiangubuntu/research/har_rgbe/utils/origin_data/origin_data_rgb"
    # origin_data_path = "/home/qiangubuntu/research/har_rgbe/utils/origin_data/origin_data_rgbd"
    # origin_data_path = "/home/qiangubuntu/research/har_rgbe/utils/origin_data/origin_data_event"
    origin_data_path = "utils/origin_data/eventcsv_categorize"
    if not os.path.exists(origin_data_path):
        raise FileNotFoundError(f"Origin data path '{origin_data_path}' does not exist.")
    # output_path = "/home/qiangubuntu/research/har_rgbe/utils/train_data/rgbe"
    # output_path = "/home/qiangubuntu/research/har_rgbe/utils/train_data/rgb"
    # output_path = "/home/qiangubuntu/research/har_rgbe/utils/train_data/rgbd"
    # output_path = "/home/qiangubuntu/research/har_rgbe/utils/train_data/event"
    output_path = "utils/train_data/event_csv"
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Output path '{output_path}' does not exist.")
    split_dataset(origin_data_path, output_path)