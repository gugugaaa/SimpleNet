# SimpleNet


![](imgs/cover.png)

**SimpleNet: A Simple Network for Image Anomaly Detection and Localization**

*Zhikang Liu, Yiming Zhou, Yuansheng Xu, Zilei Wang**

[Paper link](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf)

##  Introduction

This repo contains source code for **SimpleNet** implemented with pytorch.

SimpleNet is a simple defect detection and localization network that built with a feature encoder, feature generator and defect discriminator. It is designed conceptionally simple without complex network deisng, training schemes or external data source.

## Get Started 

### Environment 

**Python3.8**

**Packages**:
- torch==1.12.1
- torchvision==0.13.1
- numpy==1.22.4
- opencv-python==4.5.1

(Above environment setups are not the minimum requiremetns, other versions might work too.)


### Data

Edit `run.sh` to edit dataset class and dataset path.

#### MvTecAD

Download the dataset from [here](https://www.mvtec.com/company/research/datasets/mvtec-ad/).

The dataset folders/files follow its original structure.

### Run

#### Demo train

Please specicy dataset path (line1) and log folder (line10) in `run.sh` before running.

`run.sh` gives the configuration to train models on MVTecAD dataset.
```
bash run.sh
```

## Citation
```
@inproceedings{liu2023simplenet,
  title={SimpleNet: A Simple Network for Image Anomaly Detection and Localization},
  author={Liu, Zhikang and Zhou, Yiming and Xu, Yuansheng and Wang, Zilei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20402--20411},
  year={2023}
}
```

## Acknowledgement

Thanks for great inspiration from [PatchCore](https://github.com/amazon-science/patchcore-inspection)

## License

All code within the repo is under [MIT license](https://mit-license.org/)

# 以下是我补充的内容

## 训练
### 预期输入

目录结构
`mvtec/<class>/train`

命令
`net/dataset`两个子命令

### 预期输出
```c
results/
└─ project/group/run_name/
   ├─ models/
   │  └─ 0/
   │     ├─ ckpt.pth          # 判别器+投影层权重
   │     └─ params.pkl        # patch 大小、通道数等超参
   ├─ tensorboard/            # 训练曲线
   └─ results.csv             # AUROC 汇总
```

## 推理

```python
score, mask = net.predict(img)
net.load_from_path('./results/.../models/0') 
```

## 全部参数

```mermaid
flowchart TD
    A[命令行参数] --> B[主要配置参数]
    A --> C[模型相关参数]
    A --> D[数据集相关参数]
    A --> E[训练相关参数]
    A --> F[优化器相关参数]
    A --> G[数据增强参数]

    subgraph 主要配置参数
        B1[results_path<br/>结果保存路径]
        B2[gpu<br/>GPU设备ID]
        B3[seed<br/>随机种子]
        B4[log_group<br/>日志组名]
        B5[log_project<br/>项目日志名]
        B6[run_name<br/>运行名称]
        B7[test<br/>是否测试模式]
        B8[save_segmentation_images<br/>是否保存分割图像]
    end

    subgraph 模型相关参数
        C1[backbone_names<br/>主干网络名称]
        C2[layers_to_extract_from<br/>提取层名称]
        C3[pretrain_embed_dimension<br/>预训练嵌入维度]
        C4[target_embed_dimension<br/>目标嵌入维度]
        C5[patchsize<br/>图像块大小]
        C6[embedding_size<br/>嵌入大小]
        C7[pre_proj<br/>预投影层]
        C8[proj_layer_type<br/>投影层类型]
    end

    subgraph 数据集相关参数
        D1[name<br/>数据集名称]
        D2[data_path<br/>数据路径]
        D3[subdatasets<br/>子数据集列表]
        D4[batch_size<br/>批次大小]
        D5[resize<br/>调整图像大小]
        D6[imagesize<br/>图像尺寸]
        D7[num_workers<br/>数据加载工作线程数]
    end

    subgraph 训练相关参数
        E1[meta_epochs<br/>元训练轮数]
        E2[aed_meta_epochs<br/>AED元训练轮数]
        E3[gan_epochs<br/>GAN训练轮数]
        E4[noise_std<br/>噪声标准差]
        E5[auto_noise<br/>自动噪声]
        E6[mix_noise<br/>混合噪声]
        E7[train_backbone<br/>是否训练主干网络]
        E8[cos_lr<br/>是否使用余弦学习率]
    end

    subgraph 优化器相关参数
        F1[dsc_layers<br/>判别器层数]
        F2[dsc_hidden<br/>判别器隐藏层大小]
        F3[dsc_margin<br/>判别器边界]
        F4[dsc_lr<br/>判别器学习率]
    end

    subgraph 数据增强参数
        G1[rotate_degrees<br/>旋转角度]
        G2[translate<br/>平移比例]
        G3[scale<br/>缩放比例]
        G4[brightness<br/>亮度调整]
        G5[contrast<br/>对比度调整]
        G6[saturation<br/>饱和度调整]
        G7[gray<br/>灰度概率]
        G8[hflip<br/>水平翻转概率]
        G9[vflip<br/>垂直翻转概率]
        G10[augment<br/>是否启用数据增强]
    end
```