2025/8/22 晚

尝试骨干网络：EfficientNet/ConvNeXt
    TorchCAM ✔ 没啥用
    Eigen-CAM ✔ efficient net v2m效果好
Low（浅，纹理）、Middle（中，平衡）、High（深，语义）

笔记：
eigen cam高，说明高方差（cam的原理用的是PCA）
所以这里选择背景低、主体高的blocks
高方差说明特征分布更丰富，但愿能被判别器注意到/(ㄒoㄒ)/~~