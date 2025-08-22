# 作为kaggle笔记本的一个补丁单元格，让insplad数据集适配SimpleNet代码
"""
insplad/
├── class 1/
│   ├── train/
│   │   └── good/
│   └── test/
│       ├── good/
│       ├── defect class 1/
│       └── ...
├── class 2/
│   └── ...

它的问题在于，相比于标准的MvTec AD，没有ground_truth，而且某些类别不存在test/defect*
所以我们先检查，再处理defect*，最后瞎编ground_truth里面的Mask（Simple Net要求mask数量和defect的图片数量一致）

虽然会让测评时的像素级指标没有意义，但至少可以训练模型了，因为训练仅使用高斯噪音的类GAN逻辑。
"""
import os
import shutil
from glob import glob

# 用户自定义要处理的类别索引（例如 [1, 2, 3]）
selected_classes = [10]  # 👈 修改这里来选择你要训练/测试的类别

# 所有类别名称（顺序必须与你的 markdown 列表一致）
all_class_names = [
    "damper-preformed",
    "damper-stockbridge",
    "glass-insulator",
    "glass-insulator-big-shackle",
    "glass-insulator-small-shackle",
    "glass-insulator-tower-shackle",
    "lightning-rod-shackle",
    "lightning-rod-suspension",
    "plate",
    "polymer-insulator",
    "polymer-insulator-lower-shackle",
    "polymer-insulator-tower-shackle",
    "polymer-insulator-upper-shackle",
    "spacer",
    "vari-grip",
    "yoke",
    "yoke-suspension"
]
# 为了方便数数,总共0-16,其中plate在第8个

src_dir = '/kaggle/input/insplad'
dst_dir = '/kaggle/working/insplad'

# 清理目标目录
if os.path.exists(dst_dir):
    shutil.rmtree(dst_dir)
os.makedirs(dst_dir)

# 获取用户选择的类别名
selected_class_names = [all_class_names[i] for i in selected_classes]

print("Selected classes:")
for idx, name in zip(selected_classes, selected_class_names):
    print(f"  {idx}: {name}")

# 复制选定类别，并处理 test 和 ground_truth 目录
for class_name in selected_class_names:
    src_class_path = os.path.join(src_dir, class_name)
    dst_class_path = os.path.join(dst_dir, class_name)

    if not os.path.exists(src_class_path):
        print(f"[Warning] Source class folder not found: {src_class_path}")
        continue

    # 复制整个类别文件夹
    shutil.copytree(src_class_path, dst_class_path)

    test_dir = os.path.join(dst_class_path, 'test')
    gt_dir = os.path.join(dst_class_path, 'ground_truth')

    if not os.path.exists(test_dir):
        print(f"[Warning] No test directory in {class_name}")
        continue

    # 统计 test 下非 good 的 defect 文件夹及图像数
    defect_count = {}
    for item in os.listdir(test_dir):
        item_path = os.path.join(test_dir, item)
        if os.path.isdir(item_path) and item != 'good':
            img_count = len(glob(os.path.join(item_path, "*")))
            defect_count[item] = img_count

    total_defect_images = sum(defect_count.values())
    print(f"\nClass '{class_name}': Found {len(defect_count)} defect types with {total_defect_images} images in test/!good")

    # 创建 ground_truth 目录
    os.makedirs(gt_dir, exist_ok=True)

    # 为每个 defect 类型创建对应的 mask 文件夹并生成空白掩码
    for defect_type, img_count in defect_count.items():
        # 创建 ground_truth 下的 defect 文件夹
        mask_folder = os.path.join(gt_dir, defect_type)
        os.makedirs(mask_folder, exist_ok=True)
        
        # 获取该 defect 文件夹中的所有图像
        defect_img_paths = glob(os.path.join(test_dir, defect_type, "*"))
        
        # 为每个图像生成对应的黑色掩码
        for img_path in defect_img_paths:
            img_name = os.path.basename(img_path)
            mask_name = os.path.splitext(img_name)[0] + "_mask.png"
            mask_path = os.path.join(mask_folder, mask_name)
            
            # 创建黑色掩码图像（这里假设使用PIL库）
            from PIL import Image
            # 创建一个示例图像来获取尺寸（如果已有图像信息可直接使用）
            with Image.open(img_path) as img:
                width, height = img.size
                black_mask = Image.new('L', (width, height), 0)  # 'L' 表示灰度模式，0表示黑色
                black_mask.putpixel((0, 0), 1)
                black_mask.save(mask_path)

    print(f"Generated {total_defect_images} black masks for class '{class_name}'")

print("\n Dataset prepared successfully.")