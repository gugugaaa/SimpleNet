"""
完整的数据处理和测试流程
1. 模拟 util_cell.py 的处理流程
2. 测试数据加载器
"""
import sys
import os
import shutil
from glob import glob
from PIL import Image
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from datasets.mvtec import MVTecDataset, DatasetSplit

def simulate_util_cell_processing(src_dir, selected_classes):
    """
    模拟 util_cell.py 的完整处理流程
    """
    # 所有类别名称（与util_cell.py保持一致）
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
    
    # 创建模拟的工作目录
    dst_dir = os.path.join(os.path.dirname(__file__), 'fake_kaggle_working', 'insplad')
    
    # 清理目标目录
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)
    
    # 获取用户选择的类别名
    selected_class_names = [all_class_names[i] for i in selected_classes]
    
    print("=== 模拟 util_cell.py 处理流程 ===")
    print("选择的类别:")
    for idx, name in zip(selected_classes, selected_class_names):
        print(f"  {idx}: {name}")
    
    # 复制选定类别，并处理 test 和 ground_truth 目录
    for class_name in selected_class_names:
        src_class_path = os.path.join(src_dir, class_name)
        dst_class_path = os.path.join(dst_dir, class_name)
        
        if not os.path.exists(src_class_path):
            print(f"[警告] 源类别文件夹不存在: {src_class_path}")
            continue
        
        print(f"\n处理类别: {class_name}")
        
        # 复制整个类别文件夹
        try:
            shutil.copytree(src_class_path, dst_class_path)
            print(f"  ✅ 复制完成: {src_class_path} -> {dst_class_path}")
        except Exception as e:
            print(f"  ❌ 复制失败: {e}")
            continue
        
        test_dir = os.path.join(dst_class_path, 'test')
        gt_dir = os.path.join(dst_class_path, 'ground_truth')
        
        if not os.path.exists(test_dir):
            print(f"  [警告] 测试目录不存在: {test_dir}")
            continue
        
        # 列出test目录下的所有子目录
        test_subdirs = [item for item in os.listdir(test_dir) 
                       if os.path.isdir(os.path.join(test_dir, item))]
        print(f"  测试子目录: {test_subdirs}")
        
        # 统计 test 下非 good 的 defect 文件夹及图像数
        defect_count = {}
        good_count = 0
        
        for item in test_subdirs:
            item_path = os.path.join(test_dir, item)
            if os.path.isdir(item_path):
                # 查找图像文件
                img_files = [f for f in os.listdir(item_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
                img_count = len(img_files)
                
                if item == 'good':
                    good_count = img_count
                    print(f"    good: {good_count} 张图片")
                else:
                    defect_count[item] = img_count
                    print(f"    {item}: {img_count} 张图片")
        
        total_defect_images = sum(defect_count.values())
        print(f"  统计: good={good_count}, defect_types={len(defect_count)}, total_defects={total_defect_images}")
        
        # 创建 ground_truth 目录
        os.makedirs(gt_dir, exist_ok=True)
        
        # 为每个 defect 类型创建对应的 mask 文件夹并生成空白掩码
        masks_created = 0
        for defect_type, img_count in defect_count.items():
            print(f"    处理缺陷类型: {defect_type} ({img_count} 张图片)")
            
            # 创建 ground_truth 下的 defect 文件夹
            mask_folder = os.path.join(gt_dir, defect_type)
            os.makedirs(mask_folder, exist_ok=True)
            
            # 获取该 defect 文件夹中的所有图像
            defect_img_dir = os.path.join(test_dir, defect_type)
            defect_img_files = [f for f in os.listdir(defect_img_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            # 为每个图像生成对应的黑色掩码 - 确保文件名匹配
            defect_img_files_sorted = sorted(defect_img_files)
            print(f"      排序后的图片文件: {defect_img_files_sorted[:3]}...")  # 显示前3个
            
            for img_file in defect_img_files_sorted:
                img_path = os.path.join(defect_img_dir, img_file)
                # 使用相同的文件名（不加_mask后缀）来确保匹配
                mask_name = img_file  # 保持相同的文件名
                mask_path = os.path.join(mask_folder, mask_name)
                
                try:
                    # 创建黑色掩码图像
                    with Image.open(img_path) as img:
                        width, height = img.size
                        black_mask = Image.new('L', (width, height), 0)  # 'L' 表示灰度模式，0表示黑色
                        black_mask.save(mask_path)
                        masks_created += 1
                except Exception as e:
                    print(f"      [警告] 创建掩码失败 {img_file}: {e}")
            
            # 验证掩码文件
            mask_files = [f for f in os.listdir(mask_folder) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            mask_files_sorted = sorted(mask_files)
            print(f"      生成的掩码文件: {len(mask_files_sorted)} 个")
            print(f"      排序后的掩码文件: {mask_files_sorted[:3]}...")  # 显示前3个
            
            # 检查文件名匹配
            if len(defect_img_files_sorted) != len(mask_files_sorted):
                print(f"      ⚠️  警告: 图片数量({len(defect_img_files_sorted)}) != 掩码数量({len(mask_files_sorted)})")
            else:
                print(f"      ✅ 图片和掩码数量匹配: {len(defect_img_files_sorted)}")
        
        print(f"  ✅ 为类别 '{class_name}' 生成了 {masks_created} 个黑色掩码")
    
    print(f"\n=== 处理完成 ===")
    print(f"处理后的数据保存到: {dst_dir}")
    return dst_dir

def test_dataloader_after_processing(processed_data_path, classname):
    """
    测试处理后的数据加载器
    """
    print(f"\n=== 测试数据加载器 ===")
    print(f"数据路径: {processed_data_path}")
    print(f"类别名称: {classname}")
    
    # 检查路径结构
    class_path = os.path.join(processed_data_path, classname)
    test_path = os.path.join(class_path, 'test')
    gt_path = os.path.join(class_path, 'ground_truth')
    
    print(f"\n路径检查:")
    print(f"  类别路径: {class_path} - {'存在' if os.path.exists(class_path) else '不存在'}")
    print(f"  测试路径: {test_path} - {'存在' if os.path.exists(test_path) else '不存在'}")
    print(f"  真值路径: {gt_path} - {'存在' if os.path.exists(gt_path) else '不存在'}")
    
    if os.path.exists(test_path):
        test_subdirs = [d for d in os.listdir(test_path) 
                       if os.path.isdir(os.path.join(test_path, d))]
        print(f"  测试子目录: {test_subdirs}")
        
        for subdir in test_subdirs:
            subdir_path = os.path.join(test_path, subdir)
            files = [f for f in os.listdir(subdir_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            print(f"    {subdir}: {len(files)} 张图片")
    
    if os.path.exists(gt_path):
        gt_subdirs = [d for d in os.listdir(gt_path) 
                     if os.path.isdir(os.path.join(gt_path, d))]
        print(f"  真值子目录: {gt_subdirs}")
        
        # 详细检查mask文件
        for subdir in gt_subdirs:
            subdir_path = os.path.join(gt_path, subdir)
            mask_files = [f for f in os.listdir(subdir_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            print(f"    {subdir}: {len(mask_files)} 个掩码文件")
            
            # 检查对应的test文件
            test_subdir_path = os.path.join(test_path, subdir)
            if os.path.exists(test_subdir_path):
                test_files = [f for f in os.listdir(test_subdir_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
                test_files_sorted = sorted(test_files)
                mask_files_sorted = sorted(mask_files)
                
                print(f"      对应测试文件: {len(test_files)} 个")
                print(f"      测试文件示例: {test_files_sorted[:3]}")
                print(f"      掩码文件示例: {mask_files_sorted[:3]}")
                
                if len(test_files) != len(mask_files):
                    print(f"      ⚠️  文件数量不匹配! test={len(test_files)}, mask={len(mask_files)}")
                else:
                    print(f"      ✅ 文件数量匹配: {len(test_files)}")
    else:
        print(f"  真值路径不存在!")
    
    try:
        # 创建测试数据集
        test_dataset = MVTecDataset(
            source=processed_data_path,
            classname=classname,
            resize=256,
            imagesize=224,
            split=DatasetSplit.TEST
        )
        
        print(f"\n✅ 数据集创建成功!")
        print(f"总测试样本数: {len(test_dataset)}")
        print(f"图像大小: {test_dataset.imagesize}")
        print(f"待迭代数据长度: {len(test_dataset.data_to_iterate)}")
        
        # 查看所有样本的分布
        anomaly_distribution = {}
        for classname_item, anomaly, image_path, mask_path in test_dataset.data_to_iterate:
            if anomaly not in anomaly_distribution:
                anomaly_distribution[anomaly] = 0
            anomaly_distribution[anomaly] += 1
        
        print(f"\n异常类型分布:")
        for anomaly_type, count in anomaly_distribution.items():
            is_anomaly_val = int(anomaly_type != 'good')
            print(f"  {anomaly_type}: {count} 样本 (is_anomaly={is_anomaly_val})")
        
        # 预计算期望的标签分布
        expected_labels = []
        for classname_item, anomaly, image_path, mask_path in test_dataset.data_to_iterate:
            expected_labels.append(int(anomaly != 'good'))
        
        expected_counter = Counter(expected_labels)
        print(f"\n期望的标签分布:")
        print(f"  正常 (0): {expected_counter[0]} 个")
        print(f"  异常 (1): {expected_counter[1]} 个")
        
        if len(expected_counter) == 1:
            print(f"  ⚠️  警告: 期望的标签只有一种类型: {list(expected_counter.keys())}")
        
        # 查看前几个样本
        print(f"\n前10个样本详情:")
        for i, (classname_item, anomaly, image_path, mask_path) in enumerate(test_dataset.data_to_iterate[:10]):
            is_anomaly = int(anomaly != 'good')
            print(f"  {i}: class={classname_item}, anomaly={anomaly}, is_anomaly={is_anomaly}")
            print(f"     image: {os.path.basename(image_path)}")
            print(f"     mask:  {os.path.basename(mask_path) if mask_path else 'None'}")
        
        # 创建数据加载器
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0  # 设为0避免多进程问题
        )
        
        print(f"\n✅ 数据加载器创建成功!")
        print(f"批次大小: 4")
        print(f"批次数量: {len(test_dataloader)}")
        
        # 检查前几个批次
        labels_collected = []
        anomaly_types_collected = []
        image_paths_collected = []
        
        print(f"\n遍历数据加载器...")
        
        # 检查更多批次，包括中间和末尾的批次
        batches_to_check = [0, 1, 2, 80, 81, 82, 158, 159, 160]  # 前面、中间、末尾
        
        for batch_idx, data in enumerate(test_dataloader):
            if batch_idx in batches_to_check:
                batch_labels = data["is_anomaly"].numpy().tolist()
                batch_anomalies = data["anomaly"]
                batch_paths = data["image_path"]
                
                labels_collected.extend(batch_labels)
                anomaly_types_collected.extend(batch_anomalies)
                image_paths_collected.extend(batch_paths)
                
                print(f"\n批次 {batch_idx}:")
                print(f"  批次大小: {len(batch_labels)}")
                print(f"  标签 (is_anomaly): {batch_labels}")
                print(f"  异常类型: {batch_anomalies}")
                print(f"  图像路径示例: {os.path.basename(batch_paths[0]) if batch_paths else 'None'}")
            
            # 收集所有数据用于最终统计
            if batch_idx < 5 or batch_idx >= len(test_dataloader) - 5:  # 前5个和后5个批次
                batch_labels = data["is_anomaly"].numpy().tolist()
                batch_anomalies = data["anomaly"]
                if batch_idx >= 5:  # 避免重复添加前5个
                    labels_collected.extend(batch_labels)
                    anomaly_types_collected.extend(batch_anomalies)
        
        # 统计所有批次的结果
        print(f"\n=== 数据加载器结果汇总 ===")
        print(f"处理的样本总数: {len(labels_collected)}")
        print(f"收集的标签: {labels_collected}")
        print(f"唯一标签值: {set(labels_collected)}")
        
        label_counter = Counter(labels_collected)
        print(f"标签计数: 正常={label_counter[0]}, 异常={label_counter[1]}")
        
        anomaly_type_counter = Counter(anomaly_types_collected)
        print(f"异常类型分布: {dict(anomaly_type_counter)}")
        
        # 关键检查：是否会导致ROC AUC问题
        if len(set(labels_collected)) == 1:
            print(f"\n❌ 严重警告: 只发现一种标签类型!")
            print(f"这将导致 ROC AUC 计算失败: 'Only one class present in y_true'")
            print(f"所有标签都是: {set(labels_collected)}")
            
            if 0 in set(labels_collected):
                print("  - 只有正常样本，缺少异常样本")
            else:
                print("  - 只有异常样本，缺少正常样本")
                
            print("\n可能的原因:")
            print("  1. 原始数据中某个文件夹为空")
            print("  2. 图像文件格式不被识别")
            print("  3. 路径结构不正确")
            
        else:
            print(f"\n✅ 优秀: 发现了两种标签类型!")
            print(f"可以正常计算 ROC AUC")
        
        # 显示一些示例路径
        print(f"\n=== 路径示例 ===")
        for i, (label, anomaly_type, path) in enumerate(zip(labels_collected[:3], anomaly_types_collected[:3], image_paths_collected[:3])):
            print(f"样本 {i}: label={label}, anomaly='{anomaly_type}'")
            print(f"          path='{path}'")
        
        return True
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    主函数 - 完整的数据处理和测试流程
    """
    print("🚀 完整的数据处理和测试流程")
    print("="*60)
    
    # 配置参数 - 请根据你的实际情况修改
    src_dir = input("请输入原始 insplad 数据路径: ").strip()
    if not src_dir:
        src_dir = "E:/path/to/your/original/insplad"  # 默认路径
        print(f"使用默认路径: {src_dir}")
    
    selected_classes = [10]  # polymer-insulator-lower-shackle
    
    print(f"原始数据路径: {src_dir}")
    print(f"选择的类别索引: {selected_classes}")
    
    if not os.path.exists(src_dir):
        print(f"❌ 错误: 源数据路径不存在: {src_dir}")
        print("请确保路径正确，或者修改脚本中的 src_dir 变量")
        return
    
    try:
        # 步骤1: 模拟 util_cell.py 处理流程
        print(f"\n📋 步骤1: 模拟 util_cell.py 处理...")
        processed_data_path = simulate_util_cell_processing(src_dir, selected_classes)
        
        # 步骤2: 测试数据加载器
        print(f"\n📊 步骤2: 测试数据加载器...")
        classname = "polymer-insulator-lower-shackle"  # 对应 selected_classes[0] = 10
        success = test_dataloader_after_processing(processed_data_path, classname)
        
        if success:
            print(f"\n🎉 测试完成!")
            print(f"处理后的数据位于: {processed_data_path}")
        else:
            print(f"\n💥 测试失败!")
            
    except KeyboardInterrupt:
        print(f"\n⏹️  用户中断了操作")
    except Exception as e:
        print(f"\n💥 未知错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
