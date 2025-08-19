"""
rgb图片路径rgb_path=results/predict/img/rgb，和depth数据集路径depth_path=E:/竞赛资料/暑假/电塔/数据集/depth_dataset
遍历rgb图片rgb_image（不确定格式）
找到{rgb_image}_depth.npy
    注意，可能在depth_path/good里面，所以要找depth_path下的所有文件，而不是直接子文件
保存到results/predict/img/depth
"""

import os
import shutil

rgb_path = 'results/predict/img/rgb'
depth_path = 'E:/竞赛资料/暑假/电塔/数据集/depth_dataset'
save_path = 'results/predict/img/depth'

if not os.path.exists(save_path):
    os.makedirs(save_path)

# 获取depth_path下所有文件（递归）
depth_files = {}
for root, dirs, files in os.walk(depth_path):
    for file in files:
        if file.endswith('_depth.npy'):
            depth_files[file] = os.path.join(root, file)

# 遍历rgb图片
for rgb_file in os.listdir(rgb_path):
    rgb_name, ext = os.path.splitext(rgb_file)
    depth_filename = f"{rgb_name}_depth.npy"
    if depth_filename in depth_files:
        src = depth_files[depth_filename]
        dst = os.path.join(save_path, depth_filename)
        shutil.copyfile(src, dst)
        print(f"Copied: {src} -> {dst}")
    else:
        print(f"Depth file not found for: {rgb_file}")