"""detection methods."""
import logging
import os
import pickle
from collections import OrderedDict

import math
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter

import common
import metrics
from utils import plot_segmentation_images

LOGGER = logging.getLogger(__name__)

def init_weight(m):

    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)

# 判别器网络
class Discriminator(torch.nn.Module):
    # 传入：输入维度、全连接层数、隐藏层维度（None时自动计算）
    def __init__(self, in_planes, n_layers=1, hidden=None):
        super(Discriminator, self).__init__()

        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        # 循环构建中间层（self.body）
        for i in range(n_layers-1):
            # 控制IO维度
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            # 添加固定的模块
            self.body.add_module('block%d'%(i+1),
                                 torch.nn.Sequential(
                                     torch.nn.Linear(_in, _hidden),
                                     torch.nn.BatchNorm1d(_hidden),
                                     torch.nn.LeakyReLU(0.2)
                                 ))
        self.tail = torch.nn.Linear(_hidden, 1, bias=False)
        self.apply(init_weight)

    # 输出判别器的得分
    def forward(self,x):
        x = self.body(x)
        x = self.tail(x)
        return x

# 投影网络：高维👉低维
class Projection(torch.nn.Module):
    # 传入：输入维度、输出维度、FC层数、激活函数类型
    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(Projection, self).__init__()
        
        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        # 循环构建FC层
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes 
            self.layers.add_module(f"{i}fc", 
                                   torch.nn.Linear(_in, _out))
            # 添加激活函数
            if i < n_layers - 1:
                # if layer_type > 0:
                #     self.layers.add_module(f"{i}bn", 
                #                            torch.nn.BatchNorm1d(_out))
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu",
                                           torch.nn.LeakyReLU(.2))
        self.apply(init_weight)
    
    # 返回投影后的特征
    def forward(self, x):
        
        # 未启用残差连接？
        # x = .1 * self.layers(x) + x
        x = self.layers(x)
        return x

# 日志
class TBWrapper:
    
    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)
    
    def step(self):
        self.g_iter += 1

# 核心类
class SimpleNet(torch.nn.Module):
    # 设备
    def __init__(self, device):
        """anomaly detection class."""
        super(SimpleNet, self).__init__()
        self.device = device

    # 初始化，构建整个网络
    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension, # 1536
        target_embed_dimension, # 1536
        patchsize=3, # 3
        patchstride=1, 
        embedding_size=None, # 256
        meta_epochs=1, # 40
        aed_meta_epochs=1,
        gan_epochs=1, # 4
        noise_std=0.05,
        mix_noise=1,
        noise_type="GAU",
        dsc_layers=2, # 2
        dsc_hidden=None, # 1024
        dsc_margin=.8, # .5
        dsc_lr=0.0002,
        train_backbone=False,
        auto_noise=0,
        cos_lr=False,
        lr=1e-3,
        pre_proj=0, # 1
        proj_layer_type=0,
        **kwargs,
    ):
        pid = os.getpid()
        def show_mem():
            return(psutil.Process(pid).memory_info())

        # 骨干网络
        self.backbone = backbone.to(device)
        # 列表，指定哪些层负责特征提取
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        # 分块
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        # 动态的注册子模块/分支网络
        # 
        self.forward_modules = torch.nn.ModuleDict({})

        # 特征提取，可以指定到哪一层
        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone
        )
        # 取出特征维度
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        # 特征预处理
        preprocessing = common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        # 处理后的维度（concat）
        self.target_embed_dimension = target_embed_dimension

        # 融合特征
        preadapt_aggregator = common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator
        # 得到最终的特征

        # 定位异常热力图
        self.anomaly_segmentor = common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.embedding_size = embedding_size if embedding_size is not None else self.target_embed_dimension
        
        # --- 训练参数 ---
        self.meta_epochs = meta_epochs
        self.lr = lr
        self.cos_lr = cos_lr
        self.train_backbone = train_backbone
        # adamw优化器
        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(self.forward_modules["feature_aggregator"].backbone.parameters(), lr)
        # AED
        self.aed_meta_epochs = aed_meta_epochs

        # 特征投影
        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = Projection(self.target_embed_dimension, self.target_embed_dimension, pre_proj, proj_layer_type)
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.AdamW(self.pre_projection.parameters(), lr*.1)

        # 判别器
        self.auto_noise = [auto_noise, None]
        self.dsc_lr = dsc_lr
        self.gan_epochs = gan_epochs
        self.mix_noise = mix_noise
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
        self.discriminator.to(self.device)
        self.dsc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.dsc_lr, weight_decay=1e-5)
        self.dsc_schl = torch.optim.lr_scheduler.CosineAnnealingLR(self.dsc_opt, (meta_epochs - aed_meta_epochs) * gan_epochs, self.dsc_lr*.4)
        self.dsc_margin= dsc_margin 

        self.model_dir = ""
        self.dataset_name = ""
        self.tau = 1
        self.logger = None

    def set_model_dir(self, model_dir, dataset_name):

        self.model_dir = model_dir 
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = TBWrapper(self.tb_dir) #SummaryWriter(log_dir=tb_dir)
    
    # 一批图像 👉 特征向量
    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            # 遍历这批图片
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                    input_image = image.to(torch.float).to(self.device)
                with torch.no_grad():
                    # 对每张图片转换成特征向量
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    # 特征提取
    def _embed(self, images, detach=True, provide_patch_shapes=False, evaluation=False):
        """Returns feature embeddings for images."""

        # B是批次大小batch
        B = len(images)
        # 训练模式下，更新特征提取器（骨干网络）
        if not evaluation and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else:
            _ = self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)

        # 从指定的层提取特征
        features = [features[layer] for layer in self.layers_to_extract_from]

        for i, feat in enumerate(features):
            # 处理ViT输出的三维特征
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)

        # 分块
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            # 统一不同分辨率的特征的尺寸
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        
        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features) # pooling each feature to same channel and stack together
        features = self.forward_modules["preadapt_aggregator"](features) # further pooling        


        return features, patch_shapes

    
    def test(self, training_data, test_data, save_segmentation_images):
        
        # 加载模型
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt.pth")
        if os.path.exists(ckpt_path):
            state_dicts = torch.load(ckpt_path, map_location=self.device)
            if "discriminator" in state_dicts:
                self.discriminator.load_state_dict(state_dicts["discriminator"])
            if "pre_projection" in state_dicts and hasattr(self, "pre_projection"):
                self.pre_projection.load_state_dict(state_dicts["pre_projection"])
            if "backbone" in state_dicts:
                # 恢复骨干网络（若曾参与训练）
                self.forward_modules["feature_aggregator"].backbone.load_state_dict(state_dicts["backbone"])            

        # 初始化特征提取器
        aggregator = {"scores": [], "segmentations": [], "features": []}
        
        # 推理打分
        scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
        aggregator["scores"].append(scores)
        aggregator["segmentations"].append(segmentations)
        aggregator["features"].append(features)

        # 归一化
        scores = np.array(aggregator["scores"])
        min_scores = scores.min(axis=-1).reshape(-1, 1)
        max_scores = scores.max(axis=-1).reshape(-1, 1)
        scores = (scores - min_scores) / (max_scores - min_scores)
        scores = np.mean(scores, axis=0)

        # 求分割图
        segmentations = np.array(aggregator["segmentations"])
        min_scores = (
            segmentations.reshape(len(segmentations), -1)
            .min(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        max_scores = (
            segmentations.reshape(len(segmentations), -1)
            .max(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        segmentations = (segmentations - min_scores) / (max_scores - min_scores)
        # 用平均值作为最终分数
        segmentations = np.mean(segmentations, axis=0)

        # 生成异常标签
        anomaly_labels = [
            x[1] != "good" for x in test_data.dataset.data_to_iterate
        ]

        if save_segmentation_images:
            self.save_segmentation_images(test_data, segmentations, scores)
            
        auroc = metrics.compute_imagewise_retrieval_metrics(
            scores, anomaly_labels
        )["auroc"]
        # 只返回图像级AUROC
        return auroc
    
    def _evaluate(self, test_data, scores, segmentations, features, labels_gt, masks_gt):
        scores = np.squeeze(np.array(scores))
        img_min_scores = scores.min(axis=-1)
        img_max_scores = scores.max(axis=-1)
        scores = (scores - img_min_scores) / (img_max_scores - img_min_scores)
        # 图像级AUROC
        auroc = metrics.compute_imagewise_retrieval_metrics(
            scores, labels_gt 
        )["auroc"]
        return auroc
        
    
    def fit(self, training_data, test_data):
        state_dict = {}
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt.pth")
        # 如果存在模型，那么不再训练，只做评估
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=self.device)
            if 'discriminator' in state_dict:
                self.discriminator.load_state_dict(state_dict['discriminator'])
                if "pre_projection" in state_dict and hasattr(self, "pre_projection"):
                    self.pre_projection.load_state_dict(state_dict["pre_projection"])
                if "backbone" in state_dict:
                    self.forward_modules["feature_aggregator"].backbone.load_state_dict(state_dict["backbone"])            
            else:
                self.load_state_dict(state_dict, strict=False)

            # self.predict(training_data, "train_")
            scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
            auroc = self._evaluate(test_data, scores, segmentations, features, labels_gt, masks_gt)
            return auroc

        def update_state_dict(d):
            state_dict["discriminator"] = OrderedDict({
                k:v.detach().cpu() 
                for k, v in self.discriminator.state_dict().items()})
            if self.pre_proj > 0:
                state_dict["pre_projection"] = OrderedDict({
                    k:v.detach().cpu() 
                    for k, v in self.pre_projection.state_dict().items()})
            if self.train_backbone:
                # 保存骨干网络的权重（若参与训练）
                state_dict["backbone"] = OrderedDict({
                    k:v.detach().cpu()
                    for k, v in self.forward_modules["feature_aggregator"].backbone.state_dict().items()})

        # 如果预先没有模型，那么正常训练和保存。
        best_auroc = None
        for i_mepoch in range(self.meta_epochs):
            self._train_discriminator(training_data)
            scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
            auroc = self._evaluate(test_data, scores, segmentations, features, labels_gt, masks_gt)
            self.logger.logger.add_scalar("i-auroc", auroc, i_mepoch)
            if best_auroc is None or auroc > best_auroc:
                best_auroc = auroc
                update_state_dict(state_dict)
            print(f"----- {i_mepoch} I-AUROC:{round(auroc, 4)}(MAX:{round(best_auroc, 4)}) -----")
        torch.save(state_dict, ckpt_path)
        return best_auroc
            
    # 训练判别器 细节
    def _train_discriminator(self, input_data):
        """Computes and sets the support features for SPADE."""
        # 打开eval模式
        _ = self.forward_modules.eval()
        
        # 如果有投影网络，也训练
        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()
        # self.feature_enc.eval()
        # self.feature_dec.eval()
        i_iter = 0
        LOGGER.info(f"Training discriminator...")
        # 循环gan_epochs次
        with tqdm.tqdm(total=self.gan_epochs) as pbar:
            for i_epoch in range(self.gan_epochs):
                all_loss = []
                all_p_true = []
                all_p_fake = []
                all_p_interp = []
                embeddings_list = []
                # 遍历输入数据
                for data_item in input_data:
                    # 清零优化器梯度
                    self.dsc_opt.zero_grad()
                    if self.pre_proj > 0:
                        self.proj_opt.zero_grad()
                    # self.dec_opt.zero_grad()

                    i_iter += 1
                    img = data_item["image"]
                    img = img.to(torch.float).to(self.device)
                    # 用embed方法从骨干网络提取特征，可选经过预投影
                    # 得到真实特征 true_feats [N, C]
                    if self.pre_proj > 0:
                        true_feats = self.pre_projection(self._embed(img, evaluation=False)[0])
                    else:
                        true_feats = self._embed(img, evaluation=False)[0]
                    
                    # 生成噪声
                    # 每条特征向量抽一个噪声特征（强度和分布）
                    noise_idxs = torch.randint(0, self.mix_noise, torch.Size([true_feats.shape[0]]))
                    # 每条特征向量选中的那个噪声为1，其余为0
                    noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=self.mix_noise).to(self.device) # (N, K)
                    # 构造完整的噪声 [N, K, C]
                    noise = torch.stack([
                        torch.normal(0, self.noise_std * 1.1**(k), true_feats.shape)
                        for k in range(self.mix_noise)], dim=1).to(self.device) # (N, K, C)
                    # 选择对应的噪声
                    noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)

                    # 添加到真实特征上
                    fake_feats = true_feats + noise

                    # 把真实特征和伪造特征拼接（N+N=2N），都是送入判别器
                    scores = self.discriminator(torch.cat([true_feats, fake_feats]))

                    # 判别器给出得分的数组：前N个评分是对于输入的真实特征的评分
                    # 评分越高，越认为是真实特征
                    true_scores = scores[:len(true_feats)]
                    fake_scores = scores[len(fake_feats):]
                    
                    # 计算损失
                    th = self.dsc_margin
                    # 监控判断正确/错误的比例
                    p_true = (true_scores.detach() >= th).sum() / len(true_scores)
                    p_fake = (fake_scores.detach() < -th).sum() / len(fake_scores)
                    # 真实特征损失
                    true_loss = torch.clip(-true_scores + th, min=0)
                    # 伪造特征损失
                    fake_loss = torch.clip(fake_scores + th, min=0)

                    self.logger.logger.add_scalar(f"p_true", p_true, self.logger.g_iter)
                    self.logger.logger.add_scalar(f"p_fake", p_fake, self.logger.g_iter)

                    # 判别器损失=真实特征损失的平均+伪造特征损失的平均
                    loss = true_loss.mean() + fake_loss.mean()
                    self.logger.logger.add_scalar("loss", loss, self.logger.g_iter)
                    self.logger.step()

                    # 反向传播
                    # opt.step()的意思是让优化器更新一次他所管理的模型
                    loss.backward()
                    if self.pre_proj > 0:
                        self.proj_opt.step()
                    if self.train_backbone:
                        self.backbone_opt.step()
                    self.dsc_opt.step()

                    loss = loss.detach().cpu() 
                    all_loss.append(loss.item())
                    all_p_true.append(p_true.cpu().item())
                    all_p_fake.append(p_fake.cpu().item())
                
                if len(embeddings_list) > 0:
                    self.auto_noise[1] = torch.cat(embeddings_list).std(0).mean(-1)
                
                # 学习率调度
                if self.cos_lr:
                    self.dsc_schl.step()
                
                all_loss = sum(all_loss) / len(input_data)
                all_p_true = sum(all_p_true) / len(input_data)
                all_p_fake = sum(all_p_fake) / len(input_data)
                cur_lr = self.dsc_opt.state_dict()['param_groups'][0]['lr']

                # 更新进度条
                pbar_str = f"epoch:{i_epoch} loss:{round(all_loss, 5)} "
                pbar_str += f"lr:{round(cur_lr, 6)}"
                pbar_str += f" p_true:{round(all_p_true, 3)} p_fake:{round(all_p_fake, 3)}"
                if len(all_p_interp) > 0:
                    pbar_str += f" p_interp:{round(sum(all_p_interp) / len(input_data), 3)}"
                pbar.set_description_str(pbar_str)
                pbar.update(1)


    def predict(self, data, prefix=""):
        # 入口，区分是加载器还是单张图片，调用不同的预测方法
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data, prefix)
        return self._predict(data)

    # 处理数据加载器
    def _predict_dataloader(self, dataloader, prefix):
        """This function provides anomaly scores/maps for full dataloaders."""
        # 关闭梯度计算
        _ = self.forward_modules.eval()

        img_paths = []
        scores = []
        masks = []
        features = []
        labels_gt = []
        masks_gt = []
        from sklearn.manifold import TSNE

        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for data in data_iterator:
                # 收集ground truth
                if isinstance(data, dict):
                    labels_gt.extend(data["is_anomaly"].numpy().tolist())
                    if data.get("mask", None) is not None:
                        masks_gt.extend(data["mask"].numpy().tolist())
                    image = data["image"]
                    img_paths.extend(data['image_path'])
                # 推理
                _scores, _masks, _feats = self._predict(image)
                # 结果得分
                for score, mask, feat, is_anomaly in zip(_scores, _masks, _feats, data["is_anomaly"].numpy().tolist()):
                    scores.append(score)
                    masks.append(mask)

        # 这里的mask对应的是预测结果，而不是ground truth mask，其实写成segmentations更好理解
        # 后面都是当作segmentations读出来的
        # 想要绘制的画，直接plt就可以保存，不需要用utils
        return scores, masks, features, labels_gt, masks_gt

    # 处理一个batch的图片
    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()
        with torch.no_grad():
            # 得到特征向量
            features, patch_shapes = self._embed(images,
                                                 provide_patch_shapes=True, 
                                                 evaluation=True)
            # 特征投影
            if self.pre_proj > 0:
                features = self.pre_projection(features)

            # features = features.cpu().numpy()
            # features = np.ascontiguousarray(features.cpu().numpy())
            
            # 判别器输出的是各个patch的得分
            patch_scores = image_scores = -self.discriminator(features)
            patch_scores = patch_scores.cpu().numpy()
            image_scores = image_scores.cpu().numpy()

            # 图像级分数
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            # 各个Patch的分数（用于制作异常热力图）
            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            features = features.reshape(batchsize, scales[0], scales[1], -1)
            masks, features = self.anomaly_segmentor.convert_to_segmentation(patch_scores, features)

        return list(image_scores), list(masks), list(features)

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "params.pkl")

    def save_to_path(self, save_path: str, prepend: str = ""):
        LOGGER.info("Saving data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(params, save_file, pickle.HIGHEST_PROTOCOL)

    def save_segmentation_images(self, data, segmentations, scores):
        image_paths = [
            x[2] for x in data.dataset.data_to_iterate
        ]
        mask_paths = [
            x[3] for x in data.dataset.data_to_iterate
        ]

        def image_transform(image):
            in_std = np.array(
                data.dataset.transform_std
            ).reshape(-1, 1, 1)
            in_mean = np.array(
                data.dataset.transform_mean
            ).reshape(-1, 1, 1)
            image = data.dataset.transform_img(image)
            return np.clip(
                (image.numpy() * in_std + in_mean) * 255, 0, 255
            ).astype(np.uint8)

        def mask_transform(mask):
            return data.dataset.transform_mask(mask).numpy()

        plot_segmentation_images(
            './output',
            image_paths,
            segmentations,
            scores,
            mask_paths,
            image_transform=image_transform,
            mask_transform=mask_transform,
        )

# 图像分割成小块，小块拼装回图像
class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=None):
        # 每个块的大小
        self.patchsize = patchsize
        self.stride = stride
        # 评分时选择k个最大值
        self.top_k = top_k

    # 把输入的特征图分割成小块
    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        # pad来确保覆盖
        padding = int((self.patchsize - 1) / 2)

        # 使用Unfold将特征图展开为小块
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)

        # 计算小块数量
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))

        # 调整张量形状（顺序）
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    # 把评分的小块拼接回去
    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])


    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        # 取最大值来降维
        while x.ndim > 2:
            x = torch.max(x, dim=-1).values
        if x.ndim == 2:
            # 多个top_k求平均
            if self.top_k > 1:
                x = torch.topk(x, self.top_k, dim=1).values.mean(1)
            else:
                x = torch.max(x, dim=1).values
        if was_numpy:
            return x.numpy()
        return x
