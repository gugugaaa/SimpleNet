<a href="https://ibb.co/cKdyTy98"><img src="https://i.ibb.co/LDf0p0yg/inshort.png" alt="inshort" border="0"></a>

使用 datasets.mvtec.InsPLADDataset 或类似自定义类加载图像，其中图像路径为 .jpg，同尺寸相对深度图为同路径的 _depth.npy（uint8 格式，0-255，低值表示近/前景，高值表示远/背景）使用深度图生成掩膜，进行加权/分区域噪音添加。

深度掩膜生成：在训练时，从深度图生成前景掩膜（e.g., depth > threshold 为前景，mask=1；否则 mask=0 或小值）。掩膜下采样到 patch 尺寸（特征空间分辨率）。PS 我制作的npy前景的预测值更高
加权噪音：前景区域噪音强度高（fg_noise_std，默认 0.05），背景低（bg_noise_std，默认 0.01）。噪音公式：noise = torch.normal(0, fg_noise_std, shape) * mask + torch.normal(0, bg_noise_std, shape) * (1 - mask)。
位置对应：在 _embed 中返回空间特征（[B, H_p, W_p, C]），生成空间 mask，然后加噪音，再 flatten 送判别器。
启用条件：添加 --use_depth 标志（默认 False），仅当启用时加载/使用深度。