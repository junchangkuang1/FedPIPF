# %%

import os
import copy
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
import torch
import torch.nn as nn
import math
from torchvision import transforms
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
# import torch
# torch.cuda.empty_cache()

from unet import UNet
from dice_loss import dice_coeff

import matplotlib.pyplot as plt
from IPython.display import clear_output
import re
############################
# Helper func
############################
from helper import *
#################################
TRAIN_RATIO = 0.8
RS = 30448  # random state
N_CHANNELS, N_CLASSES = 1, 1
bilinear = True
BATCH_SIZE, EPOCHS = 4,250
# BATCH_SIZE, EPOCHS = 16, 100
img_size = 1024
CROP_SIZE = (1024, 1024)
#########################################
data_path = r'/root/autodl-tmp/project/kjc/data'
# data_path = r'G:\thirdWork_datasets\second'
# data_path = r'/root/autodl-tmp'
# CLIENTS = ['XY','SX','GD','JM']
CLIENTS = ['BIDMC','I2CVB','RUNMC','UCL','HK','BMC']
TOTAL_CLIENTS = len(CLIENTS)
# 假设路径为 './checkpoints/medsam_vit.pth'
# print(1111)
device = torch.device('cuda:0')
LR, WD, TH = 1e-5, 1e-5, 0.9
import torch
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score
# 指定模型类型为 vit_b
model_type = "vit_b"

# 你的 MedSAM 权重文件路径（请修改为你实际保存的路径）
checkpoint_path = "/root/autodl-tmp/project/kjc/medsam_vit_b.pth"
def get_feature_embedding(self, x):
    feat = self.encoder(x)  # 假设 encoder 是输出 [B, C, H, W]
    pooled = torch.mean(feat, dim=[2, 3])  # [B, C]
    return pooled  # ✅ 不转 numpy

# get_client_embedding 函数
def get_client_embedding(model, dataloader, device):
    model.eval()
    embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['img'].unsqueeze(0).to(device)
            output = model.get_feature_embedding(images)  # 返回 numpy
            embeddings.append(output)

    return np.mean(embeddings, axis=0).reshape(-1)  # 或 .squeeze()


# 加载模型
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam.to(device)
def load_sam_model(checkpoint_path):
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)  # 或者具体模型名称如 "vit_h"

    return sam
# 初始化预测器（用于后续图像推理）
predictor = SamPredictor(sam)
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry

import torch
import torch.nn as nn
import torch.nn.functional as F

class MedSAMWrapper(nn.Module):
    def __init__(self, sam_checkpoint_path, device, img_size=(256, 256)):
        super().__init__()
        self.device = device
        self.sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint_path).to(device)
        self.image_encoder = self.sam.image_encoder
        self.prompt_encoder = self.sam.prompt_encoder

        self.image_size = img_size

        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 2, 128, 3, padding=1),  # 原特征+prompt特征
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, image, point_coords=None, point_labels=None, box_coords=None):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        B, C, H, W = image.shape

        if C == 1:
            image = image.repeat(1, 3, 1, 1)

        image = F.interpolate(image, size=self.image_size, mode='bilinear', align_corners=False)
        image_embedding = self.image_encoder(image)  # [B, 256, h, w]

        # ========== 处理 Prompt ==========

        if box_coords is not None:
            # 框提示优先级高于点提示
            # box_coords: [B, 4], 格式为 (x1, y1, x2, y2), 归一化
            sparse_embeddings, _ = self.prompt_encoder(
                points=None,
                boxes=box_coords,
                masks=None
            )  # [B, N, 256]

            # 直接使用框坐标均值做 2 通道提示（x_mean, y_mean）
            prompt_feat = box_coords[:, :2] + (box_coords[:, 2:] - box_coords[:, :2]) / 2  # center
            prompt_feat = prompt_feat.view(B, 2, 1, 1).expand(-1, -1,
                                                              image_embedding.shape[2],
                                                              image_embedding.shape[3])
        elif point_coords is not None and point_labels is not None:
            if point_coords.dim() == 2:
                point_coords = point_coords.unsqueeze(1)
            elif point_coords.dim() != 3:
                raise ValueError(f"Unexpected point_coords shape: {point_coords.shape}")

            sparse_embeddings, _ = self.prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None
            )
            prompt_feat = point_coords.mean(dim=1).view(B, 2, 1, 1).expand(-1, -1,
                                                                           image_embedding.shape[2],
                                                                           image_embedding.shape[3])
        else:
            # 无提示输入
            prompt_feat = torch.zeros((B, 2, image_embedding.shape[2], image_embedding.shape[3]),
                                      device=image.device)

        # ========== 拼接并解码 ==========
        features = torch.cat([image_embedding, prompt_feat], dim=1)  # [B, 256+2, h, w]
        mask_pred = self.decoder(features)  # [B, 1, h, w]
        mask_pred = F.interpolate(mask_pred, size=(H, W), mode='bilinear', align_corners=False)

        return mask_pred.squeeze(1)  # [B, H, W]


lung_dataset = dict()
for client in CLIENTS:
    lung_dataset[client + '_train'] = BasicDataset(data_path, split=client, train=True,
                                                   transforms=transforms.Compose(
                                                       [RandomGenerator(output_size=(1024,1024), train=True)]))
    if client != 'GX':
        lung_dataset[client + '_test'] = BasicDataset(data_path, split=client, train=False,
                                                      transforms=transforms.Compose(
                                                          [RandomGenerator(output_size=(1024,1024), train=False)]))



# %% md

## Initialize the weights

# %%

TOTAL_DATA = []
for client in CLIENTS:
    if client != 'Interobs' and client != 'Lung1':
        print(len(lung_dataset[client + '_train']))
        TOTAL_DATA.append(len(lung_dataset[client + '_train']))

DATA_AMOUNT = sum(TOTAL_DATA)
WEIGHTS = [t / DATA_AMOUNT for t in TOTAL_DATA]

ORI_WEIGHTS = copy.deepcopy(WEIGHTS)

score = [0, 0, 0, 0,0,0]
dice = [0, 0, 0, 0,0,0]

# %% md

# storage file

# %%

training_clients, testing_clients = dict(), dict()
for client in CLIENTS:
    if client + '_train' in lung_dataset:
        training_clients[client] = lung_dataset[client + '_train']
    else:
        print(f"Warning: No training data for client {client}")
    if client + '_test' in lung_dataset:
        testing_clients[client] = lung_dataset[client + '_test']
    else:
        print(f"Warning: No testing data for client {client}")

acc_train, acc_valid, loss_train, loss_test = dict(), dict(), \
                                              dict(), dict()
loss_test = dict()
alpha_acc = []

nets, optimizers = dict(), dict()

acc_train1 = []
loss_train1 = []
# %%

sam_checkpoint_path = "/root/autodl-tmp/project/kjc/medsam_vit_b.pth"

nets = {client: {'medsam': None, 'unet': None} for client in CLIENTS}

# 2. 初始化 global 模型，确保格式一样
nets['global'] = {'medsam': None, 'unet': None}
nets['global']['unet'] =UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=True).to(device)
ema_net = nets['global']['unet']  # 举例，把global的unet作为ema模型

# 将 MedSAM 作为冻结的推理模型使用（不微调）
# 修改位置主要有两部分：
# 1. 不再为 MedSAM 初始化 optimizer
# 2. 不再训练 MedSAM 的部分代码中屏蔽训练流程

# ============================
# Step 1: 移除 MedSAM 的 optimizer
# ============================
optimizers = {client: {'unet': None} for client in CLIENTS}  # 不再初始化 medsam 优化器

# ============================
# Step 2: MedSAM Wrapper 仍需保留用于推理
# ============================
nets = {client: {'medsam': None, 'unet': None} for client in CLIENTS}
nets['global'] = {'medsam': None, 'unet': None}
nets['global']['unet'] = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=True).to(device)
ema_net = nets['global']['unet']

for client in CLIENTS:
    nets[client]['medsam'] = MedSAMWrapper(sam_checkpoint_path=sam_checkpoint_path, device=device, img_size=(1024, 1024)).to(device)
    nets[client]['unet'] = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=True).to(device)
    optimizers[client]['unet'] = optim.Adam(nets[client]['unet'].parameters(), lr=LR, weight_decay=WD)

# ============================
# Step 3: 在训练循环中移除 MedSAM 的训练部分
# ============================
# 替换如下部分：
# acc_, loss_ = train_model_sam_lora(...) => 删除
# total_loss = loss_unet + loss_ => 改为：total_loss = loss_unet
# epoch_loss.append(loss_) => 删除
# epoch_train_acc.append(acc_) => 删除

# ============================
# 示例：修改后的训练循环片段
# ============================
def generate_box_prompt_from_mask(mask: np.ndarray):
    """
    根据掩码生成一个 bounding box，用于 SAM 的 box prompt。

    参数:
        mask: np.ndarray, 二值掩码，shape = [H, W]

    返回:
        box: np.ndarray, shape = [1, 4]，格式为 [[x_min, y_min, x_max, y_max]]
    """
    foreground_indices = np.argwhere(mask == 1)

    if len(foreground_indices) == 0:
        raise ValueError("掩码中不包含任何前景像素，无法生成框标注。")

    y_min, x_min = foreground_indices.min(axis=0)
    y_max, x_max = foreground_indices.max(axis=0)

    # 注意顺序是 [x_min, y_min, x_max, y_max]
    box = np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32)
    return box



def test_medsam_zero_shot(testloader, medsam_model, device):
    """
    仅使用 box prompt 的 MedSAM zero-shot 推理。
    """
    medsam_model.eval()
    sigmoid = torch.nn.Sigmoid()

    total_dice, total_jc, total_asd, total_hd95 = 0.0, 0.0, 0.0, 0.0
    sample_count = 0

    with torch.no_grad():
        for batch in testloader:
            images = batch['img'].to(device=device, dtype=torch.float32)
            masks_true = batch['mask'].to(device=device, dtype=torch.float32)

            if masks_true.dim() == 3:
                masks_true = masks_true.unsqueeze(1)
            B, _, H, W = masks_true.shape

            box_coords = []
            for b in range(B):
                mask_np = masks_true[b, 0].cpu().numpy().astype(np.uint8)
                foreground = np.argwhere(mask_np == 1)
                if foreground.shape[0] == 0:
                    box = np.array([[0.0, 0.0, 1.0, 1.0]])  # fallback
                else:
                    y_min, x_min = foreground.min(axis=0)
                    y_max, x_max = foreground.max(axis=0)
                    box = np.array([[x_min / W, y_min / H, x_max / W, y_max / H]])
                box_coords.append(box)

            box_coords = torch.tensor(np.concatenate(box_coords, axis=0), dtype=torch.float32, device=device)

            pred_masks = medsam_model(images, box_coords=box_coords)  # 仅传 box_coords
            pred_probs = sigmoid(pred_masks)
            pred_bin = (pred_probs > 0.5).float()

            for i in range(B):
                if pred_bin[i].sum() == 0 or masks_true[i].sum() == 0:
                    max_val = pred_probs[i].max()
                    pred_bin[i] = (pred_probs[i] == max_val).float()

                t = masks_true[i, 0].cpu().numpy()
                p = pred_bin[i].cpu().numpy()
                dice = metric.binary.dc(t, p)
                jc = metric.binary.jc(t, p)
                asd = metric.binary.asd(t, p)
                hd95 = metric.binary.hd95(t, p)

                total_dice += dice
                total_jc += jc
                total_asd += asd
                total_hd95 += hd95
                sample_count += 1

    return total_dice / sample_count, total_jc / sample_count, total_asd / sample_count, total_hd95 / sample_count




def test_all_clients_medsam():
    results = {}
    for client in CLIENTS:
        print(f"Testing client: {client}")
        test_loader = DataLoader(training_clients[client], batch_size=BATCH_SIZE, shuffle=False)
        medsam_net = nets[client]['medsam']  # 已经加载权重的 MedSAMWrapper 实例

        dice, jc, asd, hd95 = test_medsam_zero_shot(test_loader, medsam_net, device)

        print(
            f"[MedSAM Zero-shot] Client: {client} | Dice: {dice:.4f} | JC: {jc:.4f} | ASSD: {asd:.4f} | HD95: {hd95:.4f}\n")
        results[client] = {
            "dice": dice,
            "jc": jc,
            "assd": asd,
            "hd95": hd95
        }
    return results
# 直接调用
results = test_all_clients_medsam()
