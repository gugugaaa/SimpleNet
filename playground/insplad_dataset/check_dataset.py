import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文支持（假设系统有SimHei字体；否则下载并指定路径）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用于正常显示负号

base_dir = 'F:/ManMadeRam/fake_insplad/adjust_split/insplad'
class_name = 'polymer-insulator-lower-shackle'

def collect_resolutions(base_dir, class_name):
    class_path = os.path.join(base_dir, class_name)
    
    # train/good
    train_good_path = os.path.join(class_path, 'train', 'good')
    train_good_res = get_resolutions(train_good_path)
    
    # test/good
    test_good_path = os.path.join(class_path, 'test', 'good')
    test_good_res = get_resolutions(test_good_path)
    
    # test/非good (合并所有非good子文件夹)
    test_defect_res = []
    test_dir = os.path.join(class_path, 'test')
    for subdir in os.listdir(test_dir):
        if subdir.lower() != 'good':  # 只要不是good
            defect_path = os.path.join(test_dir, subdir)
            test_defect_res.extend(get_resolutions(defect_path))
    
    return {
        'train_good': np.array(train_good_res),  # shape: (N, 2) [width, height]
        'test_good': np.array(test_good_res),
        'test_defect': np.array(test_defect_res)
    }

def get_resolutions(folder_path):
    resolutions = []
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return resolutions
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # 支持常见图像格式
            try:
                img_path = os.path.join(folder_path, file)
                with Image.open(img_path) as img:
                    resolutions.append(img.size)  # (width, height)
            except Exception as e:
                print(f"读取图像失败: {img_path}, 错误: {e}")
    return resolutions

import seaborn as sns  # seaborn用于KDE平滑（如果环境有；否则用scipy.stats.gaussian_kde）

def plot_histograms(data):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))  # 1行3列
    subsets = ['train_good', 'test_good', 'test_defect']
    titles = ['train/good尺寸', 'test/good尺寸', 'test/非good尺寸']

    for i, subset in enumerate(subsets):
        if len(data[subset]) == 0:
            continue
        # 计算每张图片的尺寸为宽高均值
        sizes = data[subset].mean(axis=1)
        sns.histplot(sizes, kde=True, ax=axs[i], color='blue')
        axs[i].set_title(titles[i])
        axs[i].set_xlabel('尺寸(宽高均值)')
        axs[i].axvline(288, color='red', linestyle='--', label='截断线288')
        axs[i].legend()
    
    plt.tight_layout()
    plt.show()

data = collect_resolutions(base_dir, class_name); plot_histograms(data)