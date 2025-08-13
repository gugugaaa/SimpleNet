#!/usr/bin/env python3
"""
调试脚本：检查测试数据的标签分布
"""

import sys
import torch
sys.path.append('datasets')

from datasets.mvtec import MVTecDataset, DatasetSplit

def debug_test_data_labels():
    """检查测试数据的标签分布"""
    
    # 创建测试数据集
    test_dataset = MVTecDataset(
        source="e:/github_projects/SimpleNet/playground/fake_kaggle_working/insplad",
        classname="polymer-insulator-lower-shackle",
        resize=256,
        imagesize=224,
        split=DatasetSplit.TEST,
    )
    
    print(f"测试数据集总样本数: {len(test_dataset)}")
    
    # 创建数据加载器
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=4,  # 与主程序保持一致
        shuffle=False,  # 不洗牌，保持原有顺序
        num_workers=0,  # 简化调试
    )
    
    print(f"批次数量: {len(test_dataloader)}")
    
    # 收集前几个批次的标签
    batch_labels = []
    for i, data in enumerate(test_dataloader):
        labels = data["is_anomaly"].numpy().tolist()
        batch_labels.extend(labels)
        
        print(f"批次 {i}: 标签 = {labels}, 异常类型 = {data['anomaly']}")
        
        # 只检查前20个批次
        if i >= 19:
            break
    
    print(f"\n前20个批次收集的标签: {batch_labels}")
    print(f"标签种类: {set(batch_labels)}")
    print(f"正常样本数: {batch_labels.count(0)}")
    print(f"异常样本数: {batch_labels.count(1)}")
    
    # 检查整个数据集的标签分布
    all_labels = []
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        all_labels.append(sample["is_anomaly"])
    
    print(f"\n整个测试集标签分布:")
    print(f"总样本数: {len(all_labels)}")
    print(f"正常样本数: {all_labels.count(0)}")
    print(f"异常样本数: {all_labels.count(1)}")
    
    # 找到第一个异常样本的位置
    first_anomaly_idx = all_labels.index(1) if 1 in all_labels else -1
    print(f"第一个异常样本在索引: {first_anomaly_idx}")
    
    if first_anomaly_idx > 0:
        first_anomaly_batch = first_anomaly_idx // 4
        print(f"第一个异常样本在第 {first_anomaly_batch} 个批次")

if __name__ == "__main__":
    debug_test_data_labels()
