# predict_img.py
"""
一个mvp，推理一个文件夹img_dir里面的所有图片，保存异常热力图

读取配置sh：
    找到net的配置参数，因为推理使用的网络和训练网络是相同的参数和维度，所以可以作为net.load的传参。
    读取datasets的配置参数，以获取尺寸信息
其他配置（batch最大size、device）
backbones.load()加载骨干网络
在device上load SimpleNet实例（注意sh里面没写的默认参数）
load_state_dict加载给定路径的模型
读取给定文件夹的图片，预处理，制作成一个/多个batch
> 注：mvtec.py使用的transform：
transform = T.Compose([
    T.Resize(resize),   
    T.CenterCrop(imagesize),    
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
遍历调用net._predict(期望的输入是 tensor，且 shape 为 [B, C, H, W])
    输出是list(image_scores), list(masks), list(features)
    遍历返回的masks，渲染成灰度热力图{原图名}_{img_score}.png
    对于img score大于0.5的，保存到{img_dir}/run/good/，否则是bad/
"""

import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import re
import matplotlib.pyplot as plt

from simplenet import SimpleNet
import backbones

# ----------- 配置 -----------
img_dir = "results/predict/img"
shellfile_path = "my_scripts/01v_resnet18_class_10.sh"
ckpt_path = "results/train/01/c10_ckpt.pth"
output_dir = "results/predict/run/01/"

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4

# ----------- 解析shell脚本参数 -----------
def parse_shell(shellfile):
    params = {}
    with open(shellfile, "r", encoding="utf-8") as f:
        content = f.read().replace("\\\n", " ")  # 合并多行命令
        # backbone
        m = re.search(r"-b\s+(\S+)", content)
        if m:
            params["backbone"] = m.group(1)
        # imagesize
        m = re.search(r"--imagesize\s+(\d+)", content)
        if m:
            params["imagesize"] = int(m.group(1))
        # resize
        m = re.search(r"--resize\s+(\d+)", content)
        if m:
            params["resize"] = int(m.group(1))
        # pretrain_embed_dimension
        m = re.search(r"--pretrain_embed_dimension\s+(\d+)", content)
        if m:
            params["pretrain_embed_dimension"] = int(m.group(1))
        # target_embed_dimension
        m = re.search(r"--target_embed_dimension\s+(\d+)", content)
        if m:
            params["target_embed_dimension"] = int(m.group(1))
        # embedding_size
        m = re.search(r"--embedding_size\s+(\d+)", content)
        if m:
            params["embedding_size"] = int(m.group(1))
        # patchsize
        m = re.search(r"--patchsize\s+(\d+)", content)
        if m:
            params["patchsize"] = int(m.group(1))
        # pre_proj
        m = re.search(r"--pre_proj\s+(\d+)", content)
        if m:
            params["pre_proj"] = int(m.group(1))
        # proj_layer_type
        m = re.search(r"--proj_layer_type\s+(\d+)", content)
        if m:
            params["proj_layer_type"] = int(m.group(1))
        # layers_to_extract_from
        params["layers_to_extract_from"] = re.findall(r"-le\s+(\S+)", content)
    return params

params = parse_shell(shellfile_path)
resize = params.get("resize", 256)
imagesize = params.get("imagesize", 224)

# ----------- transform -----------
transform = T.Compose([
    T.Resize(resize),
    T.CenterCrop(imagesize),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ----------- 加载模型 -----------
backbone = backbones.load(params["backbone"])
net = SimpleNet(device)
net.load(
    backbone=backbone,
    layers_to_extract_from=params["layers_to_extract_from"],
    device=device,
    input_shape=(3, imagesize, imagesize),
    pretrain_embed_dimension=params.get("pretrain_embed_dimension", 1024),
    target_embed_dimension=params.get("target_embed_dimension", 1024),
    patchsize=params.get("patchsize", 3),
    embedding_size=params.get("embedding_size", 1024),
    pre_proj=params.get("pre_proj", 0),
    proj_layer_type=params.get("proj_layer_type", 0),
)
net.eval()
net.to(device)
net.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)

# ----------- 推理并保存 -----------
os.makedirs(os.path.join(output_dir, "run"), exist_ok=True)

img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
for i in range(0, len(img_paths), batch_size):
    batch_paths = img_paths[i:i+batch_size]
    imgs = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
    imgs_tensor = torch.stack(imgs).to(device)
    with torch.no_grad():
        image_scores, masks, features = net._predict(imgs_tensor)

    for idx, img_path in enumerate(batch_paths):
        img_score = float(image_scores[idx])
        base = os.path.splitext(os.path.basename(img_path))[0]
        # 读取原图
        orig_img = Image.open(img_path).convert("RGB")
        # 处理mask
        mask_img = None
        if masks is not None:
            mask = masks[idx]
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            mask_img = (mask * 255).astype(np.uint8)
        # 绘制并保存plt
        plt.figure(figsize=(8, 4))
        plt.suptitle(f"{base}  score={img_score:.3f}", fontsize=14)
        # 左：原图
        plt.subplot(1, 2, 1)
        plt.imshow(orig_img)
        plt.title("Original")
        plt.axis("off")
        # 右：热力图
        plt.subplot(1, 2, 2)
        if mask_img is not None:
            plt.imshow(mask_img, cmap="jet")
        else:
            plt.imshow(np.zeros((imagesize, imagesize)), cmap="gray")
        plt.title("Anomaly Map")
        plt.axis("off")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(os.path.join(output_dir, "run", f"{base}_{img_score:.3f}_plt.png"))
        plt.close()