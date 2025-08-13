#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from datasets.mvtec import MVTecDataset, DatasetSplit
from torch.utils.data import DataLoader
import tqdm

def debug_predict_issue():
    """调试predict方法中的问题"""
    
    print("🔍 调试predict方法中labels_gt的生成过程")
    print("=" * 60)
    
    # 数据路径
    data_path = "playground/fake_kaggle_working/insplad"
    classname = "polymer-insulator-lower-shackle"
    
    # 创建测试数据集
    test_dataset = MVTecDataset(
        source=data_path,
        classname=classname,
        resize=224,
        imagesize=224,
        split=DatasetSplit.TEST,
        train_val_split=1.0,
        augment=False,
    )
    
    # 创建数据加载器
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,  # 重要：不洗牌，保持顺序
        num_workers=0,
        pin_memory=True,
    )
    
    print(f"测试数据集总样本数: {len(test_dataset)}")
    print(f"数据加载器批次数: {len(test_dataloader)}")
    
    # 模拟predict_dataloader方法的标签收集过程
    labels_gt = []
    
    print("\n🚀 开始模拟predict_dataloader的标签收集...")
    
    # 完整遍历所有批次
    batch_count = 0
    with tqdm.tqdm(test_dataloader, desc="Processing batches") as data_iterator:
        for data in data_iterator:
            batch_count += 1
            
            # 收集标签（模拟predict_dataloader中的逻辑）
            if isinstance(data, dict):
                batch_labels = data["is_anomaly"].numpy().tolist()
                labels_gt.extend(batch_labels)
                
                # 打印前几个和最后几个批次的信息
                if batch_count <= 5 or batch_count >= len(test_dataloader) - 5:
                    print(f"批次 {batch_count-1}: 标签 = {batch_labels}")
            
            # 检查是否有中断的可能性
            if batch_count % 50 == 0:
                print(f"已处理 {batch_count} 个批次, 当前收集的标签数: {len(labels_gt)}")
                unique_labels = set(labels_gt)
                print(f"当前标签种类: {unique_labels}")
                if len(unique_labels) > 1:
                    print("✅ 已经收集到两种标签类型!")
                else:
                    print("⚠️  目前只有一种标签类型")
    
    print(f"\n📊 最终结果:")
    print(f"总批次数: {batch_count}")
    print(f"收集的标签总数: {len(labels_gt)}")
    print(f"标签种类: {set(labels_gt)}")
    
    # 统计标签分布
    normal_count = labels_gt.count(0)
    anomaly_count = labels_gt.count(1)
    print(f"正常样本: {normal_count}")
    print(f"异常样本: {anomaly_count}")
    
    if len(set(labels_gt)) == 1:
        print("❌ 问题重现：只有一种标签类型！")
        print("这会导致ROC AUC计算失败")
        
        # 分析原因
        if normal_count > 0 and anomaly_count == 0:
            print("原因：只收集到正常样本，没有异常样本")
        elif anomaly_count > 0 and normal_count == 0:
            print("原因：只收集到异常样本，没有正常样本")
    else:
        print("✅ 成功收集到两种标签类型，应该可以计算ROC AUC")
    
    return labels_gt

def test_memory_and_interruptions():
    """测试是否有内存或其他中断问题"""
    
    print("\n🔧 测试可能的中断原因...")
    print("=" * 60)
    
    try:
        # 检查GPU内存
        if torch.cuda.is_available():
            print(f"GPU可用内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"当前GPU内存使用: {torch.cuda.memory_allocated() / 1e9:.4f} GB")
        
        # 模拟一个简单的推理过程
        data_path = "playground/fake_kaggle_working/insplad"
        classname = "polymer-insulator-lower-shackle"
        
        test_dataset = MVTecDataset(
            source=data_path,
            classname=classname,
            resize=224,
            imagesize=224,
            split=DatasetSplit.TEST,
            train_val_split=1.0,
            augment=False,
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        
        print("开始模拟推理过程...")
        batch_count = 0
        
        for data in test_dataloader:
            batch_count += 1
            
            # 模拟推理过程（不实际运行模型）
            image = data["image"]
            labels = data["is_anomaly"]
            
            if batch_count % 20 == 0:
                print(f"处理批次 {batch_count}, 图像形状: {image.shape}, 标签: {labels.tolist()}")
            
            # 检查内存使用
            if torch.cuda.is_available() and batch_count % 50 == 0:
                current_memory = torch.cuda.memory_allocated() / 1e9
                print(f"当前GPU内存使用: {current_memory:.4f} GB")
        
        print(f"✅ 成功处理所有 {batch_count} 个批次，没有中断")
        
    except Exception as e:
        print(f"❌ 发现问题: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 调试标签收集问题
    labels_gt = debug_predict_issue()
    
    # 测试可能的中断原因
    test_memory_and_interruptions()
    
    print("\n🎯 结论:")
    print("=" * 60)
    print("如果这个脚本能正常收集到两种标签，那么问题可能在于:")
    print("1. SimpleNet的predict方法中有隐式的数据限制")
    print("2. 训练过程中的某个检查点或状态导致的问题")
    print("3. 模型推理过程中的内存不足或其他异常")
    print("4. 在实际训练时，某些配置参数导致了不同的行为")
