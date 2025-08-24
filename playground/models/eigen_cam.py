"""
指定网络和目标层（字符串）
求img_dir的所有图像的EigenCAM

这种cam不需要反向传播/类别/得分。原理是PCA
如果hf上下载权重403，那么改用全局代理，新加坡节点
"""

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import timm
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def resolve_layer(model, layer_str: str):
    """
    把类似 'layer3.1.conv2'、'blocks.4' 的路径解析成真正的 nn.Module
    """
    mod = model
    for part in layer_str.split('.'):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    return mod

def eigen_cam_on_dir(img_dir, model_name='tf_efficientnetv2_m', target_layer_names=['blocks.3', 'blocks.5'], device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = timm.create_model(model_name, pretrained=True)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 用字符串路径解析目标层
    target_layers = [resolve_layer(model, name) for name in target_layer_names]

    # 保存图片到指定目录
    save_dir = os.path.join(os.path.dirname(__file__), 'eigen_cam')
    os.makedirs(save_dir, exist_ok=True)
    # 清空保存目录
    for f in os.listdir(save_dir):
        fp = os.path.join(save_dir, f)
        if os.path.isfile(fp):
            os.remove(fp)

    # 用于拼接大图
    all_rows = []

    for img_file in os.listdir(img_dir):
        if not (img_file.endswith('.jpg') or img_file.endswith('.png')):
            continue

        img_path = os.path.join(img_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)

        # 归一化到0-1，show_cam_on_image要求float32
        rgb_img = np.array(img.resize((224, 224))).astype(np.float32) / 255.0

        # 可视化：原图 + 每个目标层的EigenCAM
        n_layers = len(target_layers)
        plt.figure(figsize=(5 * (n_layers + 1), 5))
        plt.subplot(1, n_layers + 1, 1)
        plt.imshow(rgb_img)
        plt.title('Original Image')
        plt.axis('off')

        # 用于拼接大图的本行
        row_imgs = [rgb_img]

        for idx, (layer, layer_name) in enumerate(zip(target_layers, target_layer_names)):
            cam = EigenCAM(
                model,
                [layer]
            )
            grayscale_cam = cam(input_tensor=input_tensor, eigen_smooth=True)[0]  # HxW
            vis = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            plt.subplot(1, n_layers + 1, idx + 2)
            plt.imshow(vis)
            plt.title(f'{layer_name}')
            plt.axis('off')
            # 添加到本行
            row_imgs.append(vis.astype(np.float32) / 255.0 if vis.max() > 1.1 else vis)

        plt.suptitle(f'Image: {img_file} Eigen CAM')
        plt.figtext(0.5, 0.1, f'Model: {model_name}', ha='center', fontsize=14)  # 模型名显示在正下方

        # 保存图片到指定目录，每次刷新
        img_name = os.path.splitext(img_file)[0]
        save_path = os.path.join(save_dir, f"{img_name}_cam.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        # 拼接本行
        all_rows.append(np.concatenate(row_imgs, axis=1))

    # 保存大图
    if all_rows:
        big_img = np.concatenate(all_rows, axis=0)  # 多行拼接
        plt.figure(figsize=(5 * (n_layers + 1), 5 * len(all_rows)))
        plt.imshow(big_img)
        plt.axis('off')
        plt.title(f'{model_name}_eigen_cam')
        big_img_path = os.path.join(save_dir, f"{model_name}_eigen_cam.png")
        plt.savefig(big_img_path, bbox_inches='tight')
        plt.close()

# 常用：[timm模型名, 推荐的目标层]：'resnet18'[layer2, layer3]   'wide_resnet50_2'[layer2, layer3]
# 'tf_efficientnetv2_m'[blocks.3, blocks.5]  'tf_efficientnetv2_s'[blocks.2, blocks.4]

# 使用示例
eigen_cam_on_dir('playground/visualize/results/predict/img/rgb', 
                 model_name='wide_resnet50_2', 
                 target_layer_names=['layer2', 'layer3'])