from http import client
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from scipy.ndimage import binary_erosion

import copy
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomCrop
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import adjust_gamma as intensity_shift
import torch.nn as nn
# from scipy.ndimage import distance_transform_edt as distance
# from skimage import segmentation as skimage_seg
import numpy as np
# from dice_loss import dice_coeff
import random
import logging
from torch.nn import BCEWithLogitsLoss, BCELoss, CrossEntropyLoss
# from monai.losses.focal_loss import FocalLoss
# from monai.losses.tversky import TverskyLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
from os import listdir
from os.path import splitext
from pathlib import Path
import torch.optim as optim
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
# from scipy import ndimage
from scipy.ndimage import zoom
import argparse

sigmoid = nn.Sigmoid()
CE_Loss = BCELoss()
from dice_loss import dice_coeff

# from medpy import metric
###############################################
#### CONSTANTS
###############################################

colors = ['r', 'g', 'b', 'c', 'k', 'y', 'm', 'c']


def dice_coefficient(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return 2. * intersection / (y_true.sum() + y_pred.sum() + 1e-8)


def jaccard_index(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return intersection / (union + 1e-8)


from scipy.ndimage import distance_transform_edt
import numpy as np


def average_surface_distance(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    dt_true = distance_transform_edt(~y_true)
    dt_pred = distance_transform_edt(~y_pred)

    surface_true = y_true ^ binary_erosion(y_true)
    surface_pred = y_pred ^ binary_erosion(y_pred)

    sds_true = dt_pred[surface_true]
    sds_pred = dt_true[surface_pred]

    return (sds_true.sum() + sds_pred.sum()) / (len(sds_true) + len(sds_pred) + 1e-8)


from scipy.spatial.distance import directed_hausdorff


def hd95(y_true, y_pred):
    true_points = np.argwhere(y_true)
    pred_points = np.argwhere(y_pred)

    if len(true_points) == 0 or len(pred_points) == 0:
        return np.nan  # or a large number

    d1 = directed_hausdorff(true_points, pred_points)[0]
    d2 = directed_hausdorff(pred_points, true_points)[0]
    return np.percentile([d1, d2], 95)


def aggr_fed(CLIENTS, WEIGHTS_CL, nets, fed_name='global'):
    for param_tensor in nets[fed_name].state_dict():
        tmp = None

        for client, w in zip(CLIENTS, WEIGHTS_CL):
            if client != 'Interobs' and client != 'Lung422':
                if tmp == None:
                    tmp = copy.deepcopy(w * nets[client].state_dict()[param_tensor])
                else:
                    tmp += w * nets[client].state_dict()[param_tensor]
        nets[fed_name].state_dict()[param_tensor].data.copy_(tmp)
        del tmp


import numpy as np
import random


# def generate_fg_bg_prompts_from_mask(mask: np.ndarray, num_fg_points=10, num_bg_points=10, seed=None):
#
#     if seed is not None:
#         random.seed(seed)
#         np.random.seed(seed)
#
#     fg_indices = np.argwhere(mask == 1)
#     bg_indices = np.argwhere(mask == 0)
#
#     if len(fg_indices) == 0:
#         raise ValueError("掩码中不包含任何前景像素，无法生成前景点。")
#     if len(bg_indices) == 0:
#         raise ValueError("掩码中不包含任何背景像素，无法生成背景点。")
#
#     if num_fg_points > len(fg_indices):
#         raise ValueError("请求的前景点数量超过了实际前景像素数。")
#     if num_bg_points > len(bg_indices):
#         raise ValueError("请求的背景点数量超过了实际背景像素数。")
#
#     fg_selected = fg_indices[np.random.choice(len(fg_indices), num_fg_points, replace=False)]
#     bg_selected = bg_indices[np.random.choice(len(bg_indices), num_bg_points, replace=False)]
#
#     point_coords = np.concatenate([fg_selected, bg_selected], axis=0)
#     point_labels = np.concatenate([
#         np.ones(num_fg_points, dtype=np.int64),
#         np.zeros(num_bg_points, dtype=np.int64)
#     ], axis=0)
#
#     return point_coords, point_labels

# --- 客户端代码 ---
def client_compute_and_upload_cluster_centers(true_masks):
    client_fg_centers, client_bg_centers = [], []

    for mask in true_masks:
        # 如果是 numpy 数组，转换为 PyTorch 张量
        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask, dtype=torch.int8)  # 转换为 PyTorch 张量

        # 确保 mask 是 PyTorch 张量
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy().astype(np.uint8)  # 转到 CPU 并转成 numpy
        else:
            raise ValueError("Expected mask to be a PyTorch tensor or NumPy array.")

        # 前景点
        fg_points = np.argwhere(mask_np == 1)
        # 背景点
        bg_points = np.argwhere(mask_np == 0)

        if len(fg_points) > 0:
            fg_center = fg_points.mean(axis=0)  # (y, x)
            client_fg_centers.append(fg_center)

        if len(bg_points) > 0:
            bg_center = bg_points.mean(axis=0)  # (y, x)
            client_bg_centers.append(bg_center)

    return np.array(client_fg_centers), np.array(client_bg_centers)


# --- 服务器端代码 ---
def server_aggregate_cluster_centers(all_clients_centers):
    """
    all_clients_centers: list of (fg_centers, bg_centers) from each client
    """

    fg_list, bg_list = [], []

    for fg, bg in all_clients_centers:
        if fg is not None and len(fg) > 0:
            fg_list.append(fg)
        if bg is not None and len(bg) > 0:
            bg_list.append(bg)

    # 合并成二维数组
    if len(fg_list) > 0:
        merged_fg = np.vstack(fg_list)  # shape (N, 2)
    else:
        merged_fg = np.empty((0, 2))

    if len(bg_list) > 0:
        merged_bg = np.vstack(bg_list)  # shape (M, 2)
    else:
        merged_bg = np.empty((0, 2))

    return merged_fg, merged_bg


# --- 客户端利用全局中心辅助采样 ---
def client_sample_points_with_global_centers(mask, local_fg_centers, local_bg_centers,
                                             global_fg_centers, global_bg_centers,
                                             num_fg_points=10, num_bg_points=10):
    # 参考之前提到的距离引导采样函数
    def select_by_distance(local_centers, global_centers, num_points):
        if len(local_centers) == 0:
            return np.empty((0, 2))
        if len(global_centers) == 0:
            return local_centers[:num_points]
        dist_matrix = np.linalg.norm(local_centers[:, None, :] - global_centers[None, :, :], axis=2)
        min_dist = np.min(dist_matrix, axis=1)
        idx_sorted = np.argsort(-min_dist)
        return local_centers[idx_sorted[:num_points]]

    fg_selected = select_by_distance(local_fg_centers, global_fg_centers, num_fg_points)
    bg_selected = select_by_distance(local_bg_centers, global_bg_centers, num_bg_points)

    point_coords = np.concatenate([fg_selected, bg_selected], axis=0)
    point_labels = np.concatenate([
        np.ones(len(fg_selected), dtype=np.int64),
        np.zeros(len(bg_selected), dtype=np.int64)
    ])

    return point_coords, point_labels


import numpy as np
import torch


def generate_prompts_cluster_federated(
        mask,
        global_fg_centers: np.ndarray,
        global_bg_centers: np.ndarray,
        num_fg_points=10,
        num_bg_points=10,
        eps=5,
        min_samples=10,
):
    # ---- 1. 确保 mask 是 numpy ----
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # ---- 2. 提取前景 / 背景点 ----
    fg_points = np.argwhere(mask == 1)
    bg_points = np.argwhere(mask == 0)

    # ---- 3. 本地聚类，提取中心 ----
    local_fg_centers, _ = cluster_points(fg_points, eps=eps, min_samples=min_samples)
    local_bg_centers, _ = cluster_points(bg_points, eps=eps, min_samples=min_samples)

    # ---- 4. 根据与全局中心的距离选择点 ----
    def select_by_distance(local_centers, global_centers, num_points):
        if len(local_centers) == 0:
            return np.empty((0, 2))
        if len(global_centers) == 0:
            return local_centers[:num_points]
        dist_matrix = np.linalg.norm(
            local_centers[:, None, :] - global_centers[None, :, :], axis=2
        )
        min_dist = np.min(dist_matrix, axis=1)
        idx_sorted = np.argsort(-min_dist)  # 远离全局中心的优先
        return local_centers[idx_sorted[:num_points]]

    fg_selected = select_by_distance(local_fg_centers, global_fg_centers, num_fg_points)
    bg_selected = select_by_distance(local_bg_centers, global_bg_centers, num_bg_points)

    # ---- 5. 拼接坐标与标签 ----
    point_coords = np.concatenate([fg_selected, bg_selected], axis=0)
    point_labels = np.concatenate([
        np.ones(len(fg_selected), dtype=np.int64),
        np.zeros(len(bg_selected), dtype=np.int64)
    ])

    return point_coords, point_labels


from sklearn.cluster import DBSCAN


def cluster_points(points, eps=5, min_samples=10):
    if len(points) == 0:
        return np.empty((0, 2)), np.array([])
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    unique_labels = set(labels)
    centers = []
    for label in unique_labels:
        if label == -1:
            continue  # 噪声点忽略
        cluster_pts = points[labels == label]
        center = cluster_pts.mean(axis=0)
        centers.append(center)
    return np.array(centers), labels


# def generate_foreground_prompts_from_mask(mask: np.ndarray, num_fg_points=10, seed=None):
#     """
#     从掩码中生成前景点坐标（不包含背景）
#
#     参数:
#         mask: np.ndarray, 二值掩码（0=背景, 1=前景），shape=[H, W]
#         num_fg_points: int, 生成的前景点个数
#         seed: int, 可选，用于控制随机性
#
#     返回:
#         point_coords: np.ndarray, shape = [num_fg_points, 2]，格式为 [[y1, x1], [y2, x2], ...]
#         point_labels: np.ndarray, shape = [num_fg_points]，值全部为 1
#     """
#     if seed is not None:
#         random.seed(seed)
#         np.random.seed(seed)
#
#     foreground_indices = np.argwhere(mask == 1)
#
#     if len(foreground_indices) == 0:
#         raise ValueError("掩码中不包含任何前景像素，无法生成前景点。")
#
#     if num_fg_points > len(foreground_indices):
#         raise ValueError("请求的前景点数量超过了实际前景像素数。")
#
#     selected_points = foreground_indices[np.random.choice(len(foreground_indices), num_fg_points, replace=False)]
#     point_labels = np.ones(num_fg_points, dtype=np.int64)
#
#     return selected_points, point_labels


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseCleaner(nn.Module):
    """噪声伪标签清洗器，对单模型输出去噪修正"""

    def __init__(self, channel=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(8, channel, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 简单去噪修正：先卷积提取，再sigmoid保证[0,1]范围
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)
        return out


class CrossSelfUpdateFusion(nn.Module):
    """交叉自校正融合模块"""

    def __init__(self, channel=1):
        super().__init__()
        # 用于融合的轻量卷积，结合两个去噪后的预测
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channel * 2, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, P_u_clean, P_m_clean):
        # α和β是融合系数，初始化为0.5，可以改成可学习参数
        alpha = 0.5
        beta = 0.4

        # 交叉更新
        P_u_updated = alpha * P_u_clean + (1 - alpha) * P_m_clean
        P_m_updated = beta * P_m_clean + (1 - beta) * P_u_clean

        # 拼接更新后的两个分支
        fusion_input = torch.cat([P_u_updated, P_m_updated], dim=1)  # [B, 2, H, W]

        # 融合输出
        fused = self.fusion_conv(fusion_input)  # [B,1,H,W]

        return fused


class LabelDiscriminator(nn.Module):
    """简化版标签判别器，判断伪标签可靠度"""

    def __init__(self, channel=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel * 3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, image, P_u, P_m):
        # 输入图像和两个预测，判断伪标签可靠度
        x = torch.cat([image, P_u, P_m], dim=1)  # [B,3,H,W]
        mask = self.conv(x)  # [B,1,H,W], 0~1表示可靠度
        return mask


class AttentionFusionModule(nn.Module):
    def __init__(self, channel=1):
        super().__init__()
        self.noise_cleaner_u = NoiseCleaner(channel)
        self.noise_cleaner_m = NoiseCleaner(channel)
        self.cross_fusion = CrossSelfUpdateFusion(channel)
        self.label_discriminator = LabelDiscriminator(channel)

    def forward(self, image, P_u, P_m):
        """
        image: [B,1,H,W] 输入图像（灰度）
        P_u, P_m: [B,1,H,W] 两个模型预测
        """

        # 1. 对两个预测做噪声清洗
        P_u_clean = self.noise_cleaner_u(P_u)
        P_m_clean = self.noise_cleaner_m(P_m)

        # 2. 交叉自校正融合
        fused = self.cross_fusion(P_u_clean, P_m_clean)

        # 3. 伪标签可靠度判别掩码
        reliability_mask = self.label_discriminator(image, P_u, P_m)  # 可信度mask

        # 4. 用掩码筛选高可信区域作为伪标签训练目标
        final_pseudo_label = fused * reliability_mask

        return final_pseudo_label


class BasicDataset(Dataset):
    def __init__(self, base_dir: str, split, train=False, transforms=None):
        print(split)
        self.transform = transforms  # using transform in torch!
        self.split = split
        self.image_list = []
        self._base_dir = base_dir
        self.train = train
        if train:
            with open(self._base_dir + '/{}_train.txt'.format(split), 'r') as f:
                self.image_list = f.readlines()

        else:
            with open(self._base_dir + '/{}_test.txt'.format(split), 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '') for item in self.image_list]

        print("{} has total {} samples".format(split, len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        image_name = self.image_list[idx]
        img_path = os.path.join(self._base_dir, self.split, 'image', image_name)
        mask_path = os.path.join(self._base_dir, self.split, 'mask', image_name)
        image = np.array(Image.open(img_path).convert('L'))
        mask = np.array(Image.open(mask_path).convert('L'))

        img = (image - np.min(image)) / (np.max(image) - np.min(image))
        mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
        mask[mask > 0] = 1
        mask[mask < 0] = 0
        sample = {'img': img, 'mask': mask, 'filename': img_path.split('\\')[-1]}
        if self.transform:
            sample = self.transform(sample)
        return sample


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size, train=False):
        self.output_size = output_size
        self.train = train

    def __call__(self, sample):

        img, mask, filename = sample['img'], sample['mask'], sample['filename']
        if self.train:
            if random.random() > 0.5:
                img, mask = random_rot_flip(img, mask)
            elif random.random() > 0.5:
                img, mask = random_rotate(img, mask)
        x, y = img.shape
        # print('original shape: ',image.shape,label.shape)
        if x != self.output_size[0] or y != self.output_size[1]:
            img = zoom(img, (self.output_size[0] / x, self.output_size[1] / y), order=0)  # why not 3?
            mask = zoom(mask, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # print(image.shape,label.shape)
        mask[mask >= 1] = 1

        img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        mask = torch.from_numpy(mask.astype(np.float32))
        sample = {'img': img, 'mask': mask, 'filename': filename}
        return sample


############################################
#### copy federated model to client
#### input: CLIENTS <list of client>
####      : nets <collection of dictionaries>
############################################
def copy_fed(CLIENTS, nets, model_type='unet', fed_name='global'):
    for client in CLIENTS:
        if 'Interobs' not in client and 'Lung422' not in client:
            nets[client][model_type].load_state_dict(
                copy.deepcopy(nets[fed_name][model_type].state_dict())
            )


import numpy as np
import cv2


def generate_edge_based_prompts(mask, num_fg_points=15, num_bg_points=15):
    """
    基于 mask 边缘的点提示生成方式（用于消融实验）
    mask: [H, W] 0/1 的二值 mask
    """

    H, W = mask.shape

    # ---- Step 1: 提取边缘 ----
    mask_uint8 = (mask * 255).astype(np.uint8)
    edges = cv2.Canny(mask_uint8, 50, 150)  # 二值边缘图，0/255

    edge_points = np.column_stack(np.where(edges > 0))  # [N,2] (y,x)

    if len(edge_points) == 0:
        # 没有边缘时，默认选中心点
        fg_points = np.array([[H // 2, W // 2]])
    else:
        # ---- Step 2: 在边缘上均匀采样 ----
        idx = np.linspace(0, len(edge_points) - 1, num_fg_points, dtype=int)
        fg_points = edge_points[idx]  # shape [num_fg_points, 2]

    # ---- Step 3: 采样背景点（不在前景 mask 内） ----
    bg_coords = np.column_stack(np.where(mask == 0))
    if len(bg_coords) == 0:
        bg_points = np.array([[0, 0]])  # fallback
    else:
        idx_bg = np.random.choice(len(bg_coords), num_bg_points, replace=len(bg_coords) < num_bg_points)
        bg_points = bg_coords[idx_bg]

    # ---- Step 4: 合并前景与背景点 ----
    point_coords = np.concatenate([fg_points, bg_points], axis=0)  # [N,2]
    point_labels = np.concatenate([
        np.ones(len(fg_points), dtype=np.int64),  # 前景点 label=1
        np.zeros(len(bg_points), dtype=np.int64)  # 背景点 label=0
    ])

    return point_coords, point_labels


#############################################
### A helper function to randomly find bbox #
#############################################
###########################
## Test the network acc ###
###########################
# def test_sam(testloader, net, device, acc=None, loss=None):
#     net.eval()
#     t_loss, t_acc = 0, 0
#     JC, ASSD, HD95 = 0, 0, 0
#     sigmoid = nn.Sigmoid()
#     CE_Loss = nn.BCELoss()
#
#     # def dice_coeff(pred, target, eps=1e-7):
#     #     intersection = (pred * target).sum(dim=[1, 2])
#     #     union = pred.sum(dim=[1, 2]) + target.sum(dim=[1, 2])
#     #     dice = (2. * intersection + eps) / (union + eps)
#     #     return dice.mean()
#
#     with torch.no_grad():
#         for batch in testloader:
#             image, mask_true = batch['img'], batch['mask']
#             image = image.to(device=device, dtype=torch.float32)
#             mask_true = mask_true.to(device=device, dtype=torch.float32)
#
#             if mask_true.dim() == 2:
#                 mask_true = mask_true.unsqueeze(0)
#             if mask_true.dim() == 3:
#                 mask_true = mask_true.unsqueeze(1)
#
#             B, _, H, W = mask_true.shape
#
#             # ========== 生成点提示（每张图一个前景点） ==========
#             point_coords_list, point_labels_list = [], []
#             for b in range(B):
#                 mask_np = mask_true[b, 0].cpu().numpy().astype(np.uint8)
#                 coords, labels = generate_fg_bg_prompts_from_mask(mask_np, num_fg_points=30, num_bg_points=30,
#                                                                   seed=42)
#                 if len(coords) == 0:
#                     coords = np.array([[H // 2, W // 2]])
#                     labels = np.array([1])
#                 point_coords_list.append(coords[0])
#                 point_labels_list.append(labels[0])
#
#             point_coords_array = np.array(point_coords_list)
#             point_labels_array = np.array(point_labels_list)
#
#             point_coords_tensor = torch.tensor(point_coords_array, dtype=torch.float32, device=device).unsqueeze(1)
#             point_labels_tensor = torch.tensor(point_labels_array, dtype=torch.int, device=device).unsqueeze(1)
#
#             # ========== 前向传播 MedSAM ==========
#             mask_pred = net(image, point_coords=point_coords_tensor, point_labels=point_labels_tensor)
#
#             if mask_pred.dim() == 3:
#                 mask_pred = mask_pred.unsqueeze(1)
#
#             mask_pred_norm = sigmoid(mask_pred)
#             mask_pred_1 = (mask_pred_norm > 0.5).float()
#
#             if torch.sum(mask_pred_1) == 0 or torch.sum(mask_true) == 0:
#                 percentile = torch.max(mask_pred_norm)
#                 mask_pred_1 = (mask_pred_norm == percentile).float()
#
#             # ========== 计算指标 ==========
#             t = mask_true.squeeze().cpu().numpy()
#             p = mask_pred_1.squeeze().cpu().numpy()
#
#             t_acc_network = metric.binary.dc(t, p)
#             jc = metric.binary.jc(t, p)
#             asd = metric.binary.asd(t, p)
#             hd95 = metric.binary.hd95(t, p)
#
#             t_acc += t_acc_network
#             JC += jc
#             ASSD += asd
#             HD95 += hd95
#
#     if acc is not None:
#         acc.append(t_acc / len(testloader))
#     if loss is not None:
#         loss.append(t_loss / len(testloader))
#
#     return t_acc / len(testloader), JC / len(testloader), ASSD / len(testloader), HD95 / len(testloader)


class KDLoss(nn.Module):
    """
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    """

    def __init__(self, T):
        super(KDLoss, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = (
                F.kl_div(F.log_softmax(out_s / self.T, dim=0),
                         F.softmax(out_t / self.T, dim=0), reduction="batchmean")
                * self.T
                * self.T
        )
        return loss


# CE_LOSS = nn.BCELoss()
def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def test_unet(testloader, net, device, acc=None, loss=None):
    net.eval()
    t_loss, t_acc = 0, 0
    JC, ASSD, HD95 = 0, 0, 0
    # CE_Loss = BCEWithLogitsLoss()
    # Dice_Loss = DiceLoss(1)
    with torch.no_grad():
        for batch in testloader:
            image, mask_true = batch['img'], batch['mask']
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            sigmoid = nn.Sigmoid()
            # mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
            # print(mask_true.size())
            mask_true = mask_true.float()
            ###########################################
            # predict the mask
            if image.ndim == 3:
                image = image.unsqueeze(0)  # 添加 batch 维

            mask_pred = net(image)
            mask_pred_norm = sigmoid(mask_pred.squeeze(1))
            # loss_ce = CE_Loss(mask_pred,mask_true.float())

            # loss_dice = dice_coeff(mask_pred,mask_true)
            # loss_total = 0.25*loss_ce + 0.75*loss_dice
            # t_loss += loss_total.item()

            # dice_loss += val_loss_dice
            # ce_loss += val_loss_ce
            # t_acc_network = dice_coeff(mask_true.type(torch.float), mask_pred).item()
            # t_acc += t_acc_network
            #######################################################
            mask_pred_1 = (mask_pred_norm > 0.5).float()
            if torch.sum(mask_pred_1) == 0 or torch.sum(mask_true) == 0:
                # print(torch.sum(mask_pred_1) ,torch.sum(mask_true), batch['filename'])
                # percentile = torch.quantile(mask_pred_norm, 0.0005)
                percentile = torch.max(mask_pred_norm)
                mask_pred_1 = (mask_pred_norm == percentile).float()
            t = mask_true.squeeze().cpu().numpy()
            p = mask_pred_1.squeeze().cpu().numpy()
            target_size = (224, 224)

            if t.shape != target_size:
                t = cv2.resize(t.astype(np.float32), target_size, interpolation=cv2.INTER_NEAREST)

            if p.shape != target_size:
                p = cv2.resize(p.astype(np.float32), target_size, interpolation=cv2.INTER_NEAREST)

            t_acc_network = dice_coefficient(t, p)
            jc = jaccard_index(t, p)
            asd = average_surface_distance(t, p)
            hd951 = hd95(t, p)

            t_acc += t_acc_network
            JC += jc
            ASSD += asd
            HD95 += hd951
    if acc is not None:
        acc.append(t_acc / len(testloader))
        # print('val_loss_ce: ',val_loss_ce / len(testloader),'val_loss_dice: ',val_loss_dice / len(testloader),'acc: ',t_acc / len(testloader) )

    if loss is not None:
        loss.append(t_loss / len(testloader))
    # del t_acc, t_loss

    return t_acc / len(testloader), JC / len(testloader), ASSD / len(testloader), HD95 / len(testloader)


# CE_LOSS = nn.BCELoss()
def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        if ema_param.data.shape == param.data.shape:
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
        else:
            print(f"[EMA SKIP] Shape mismatch: ema_param {ema_param.data.shape}, param {param.data.shape}")


def train_model_unet(epoch, trainloader, optimizer_stu, device, net_stu, ema_model=None, net_medsam=None,
                     fusion_module=None,
                     acc=None, supervision_type='labeled',
                     loss=None, learning_rate=None, iter_num=0):
    net_stu.train()
    if ema_model is not None:
        ema_model.train()
    if net_medsam is not None:
        net_medsam.eval()  # MedSAM通常用eval模式推理
    if fusion_module is not None:
        fusion_module.train()

    t_loss, t_acc = 0, 0
    max_iterations = 30000

    for i, batch in enumerate(trainloader):
        images = batch['img']
        true_masks = batch['mask']
        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.int8)
        if images.dim() == 3:
            images = images.unsqueeze(0)  # batch维度

        mask_pred = net_stu(images).squeeze(1)
        masks_pred = torch.sigmoid(mask_pred).float()  # [B,H,W]

        if supervision_type == 'labeled':
            if true_masks.dim() == 2:
                true_masks = true_masks.unsqueeze(0)  # [B,H,W]
            # 监督损失
            loss_ce = CE_Loss(masks_pred, true_masks.float())
            # loss_dice = (1 - dice_coeff(masks_pred, true_masks.float()))[0]
            loss_dice = (1 - dice_coeff(masks_pred, true_masks.float()))

            with torch.no_grad():
                masks_pred_binary = (masks_pred > 0.5).float()  # [B,H,W]
            batch_point_coords = []
            batch_point_labels = []
            for b in range(masks_pred.shape[0]):  # 遍历当前 batch 中的每张图像
                mask_pred = masks_pred[b].detach().cpu().numpy().astype(np.uint8)  # 使用当前batch中的图像
                true_mask_b = true_masks[b].cpu().numpy()  # 获取对应的真实mask

                try:
                    # 获取每张图片的前景和背景中心
                    client_fg_centers, client_bg_centers = client_compute_and_upload_cluster_centers(true_mask_b)
                    all_clients_centers = [(client_fg_centers, client_bg_centers)]
                    # 合并聚类中心，得到全局的前景和背景中心
                    global_fg_centers, global_bg_centers = server_aggregate_cluster_centers(all_clients_centers)

                    # 生成点标注

                    # point_coords, point_labels = generate_edge_based_prompts(true_mask_b)

                    point_coords, point_labels = generate_prompts_cluster_federated(
                        true_mask_b,  # 每张图的 mask
                        global_fg_centers,
                        global_bg_centers,
                        num_fg_points=20,
                        num_bg_points=20
                    )
                except ValueError:
                    print("有标注客户端采用默认情况")
                    # 默认情况下的处理
                    H, W = mask_pred.shape
                    point_coords = np.array([[H // 2, W // 2]])  # 默认中心点
                    point_labels = np.array([1])  # 默认标签为1

                batch_point_coords.append(point_coords)  # 每张图一个 [N, 2]
                batch_point_labels.append(point_labels)  # 每张图一个 [N]

            # 转换成 tensor
            point_coords_tensor = torch.tensor(batch_point_coords, dtype=torch.float32, device=device)  # [B, N, 2]
            point_labels_tensor = torch.tensor(batch_point_labels, dtype=torch.int64, device=device)  # [B, N]

            # MedSAM 输出（eval 模式下，且不参与梯度）
            with torch.no_grad():
                medsam_pred = torch.sigmoid(
                    net_medsam(images, point_coords=point_coords_tensor, point_labels=point_labels_tensor).squeeze(
                        1))  # [B,H,W]

            # 一致性损失（MSE）
            consistency_loss = F.mse_loss(masks_pred, medsam_pred)
            consistency_loss2 = F.mse_loss(medsam_pred, true_masks)
            loss_ce2 = CE_Loss(medsam_pred, true_masks.float())

            # 加权总损失
            loss_total = 1 * loss_ce + 1 * loss_ce2 + 1 * loss_dice + 1 * consistency_loss + 1 * consistency_loss2  # 0.1 是一致性损失权重
        else:
            with torch.no_grad():
                masks_pred_binary = (masks_pred > 0.5).float()  # [B,H,W]
            masks_pred_c = masks_pred.unsqueeze(1)
            batch_point_coords = []
            batch_point_labels = []
            for b in range(masks_pred_binary.shape[0]):  # 遍历当前 batch 中的每张图像
                mask_pred = masks_pred_binary[b].detach().cpu().numpy().astype(np.uint8)  # 使用当前batch中的图像
                true_mask_b = masks_pred_binary[b].cpu().numpy()  # 获取对应的真实mask
                try:
                    # 获取每张图片的前景和背景中心
                    client_fg_centers, client_bg_centers = client_compute_and_upload_cluster_centers(true_mask_b)
                    all_clients_centers = [(client_fg_centers, client_bg_centers)]
                    # 合并聚类中心，得到全局的前景和背景中心
                    global_fg_centers, global_bg_centers = server_aggregate_cluster_centers(all_clients_centers)

                    # 生成点标注

                    point_coords, point_labels = generate_prompts_cluster_federated(
                        true_mask_b,  # 每张图的 mask
                        global_fg_centers,
                        global_bg_centers,
                        num_fg_points=20,
                        num_bg_points=20
                    )
                except ValueError:
                    print("无标注客户端采用默认点")
                    # 默认情况下的处理
                    H, W = mask_pred.shape
                    point_coords = np.array([[H // 2, W // 2]])  # 默认中心点
                    point_labels = np.array([1])  # 默认标签为1

                batch_point_coords.append(point_coords)  # 每张图一个 [N, 2]
                batch_point_labels.append(point_labels)  # 每张图一个 [N]

            # 转换成 tensor
            point_coords_tensor = torch.tensor(batch_point_coords, dtype=torch.float32, device=device)  # [B, N, 2]
            point_labels_tensor = torch.tensor(batch_point_labels, dtype=torch.int64, device=device)  # [B, N]

            # 用点标注指导 MedSAM
            with torch.no_grad():
                medsam_pred = torch.sigmoid(
                    net_medsam(images, point_coords=point_coords_tensor, point_labels=point_labels_tensor).squeeze(
                        1))  # [B,H,W]

            # 融合

            medsam_pred_c = medsam_pred.unsqueeze(1)

            fused_pred = fusion_module(masks_pred_c, medsam_pred_c)
            fused_pred = fused_pred.squeeze(1)# 博弈融入那个和

            confidence_mask = fused_pred  # fused_pred 越接近0或1，越可靠

            # Step 3: 生成二值伪标签
            pseudo_label = (fused_pred.detach() > 0.5).float()

            # Step 4: 基础监督 loss
            loss_ce = CE_Loss(masks_pred, pseudo_label)
            # loss_dice = (1 - dice_coeff(masks_pred, pseudo_label))[0]
            loss_dice = (1 - dice_coeff(masks_pred, pseudo_label))

            # Step 5: 差异区域不稳定性
            diff_region = (masks_pred.detach() - pseudo_label.detach()).abs()  # 差异区域
            uncertain_mask = torch.sigmoid(10 * (diff_region - 0.3))  # 差异大 -> mask接近1

            # Step 6: 差异区域正则
            loss_uncertainty = (masks_pred * uncertain_mask * (1 - masks_pred)).mean()  # 熵最小化

            # Step 7: 一致性约束 (可选)
            loss_consistency = F.mse_loss(masks_pred, fused_pred.detach())

            # Step 8: 总 loss (自适应加权)
            loss_total = (
                    1 * loss_ce +
                    1 * loss_dice +
                    0.5 * loss_consistency +
                    0.5 * loss_uncertainty
            )

            # EMA 更新
            if ema_model is not None:
                update_ema_variables(net_stu, ema_model, 0.99, iter_num)

        # 学习率更新
        lr_ = learning_rate * (1.0 - iter_num / max_iterations) ** 0.9
        for param_group in optimizer_stu.param_groups:
            param_group['lr'] = lr_

        iter_num += 1

        optimizer_stu.zero_grad()
        loss_total.backward()
        optimizer_stu.step()

        t_loss += loss_total.item()

        masks_pred_bin = (masks_pred.detach() > 0.5).float()
        if true_masks.dim() == 2:
            true_masks = true_masks.unsqueeze(0)
        if true_masks.shape != masks_pred_bin.shape:
            true_masks = true_masks.expand_as(masks_pred_bin)
        true_masks = true_masks.float()
        t_acc_network = dice_coeff(masks_pred_bin, true_masks).item()
        t_acc += t_acc_network

    if acc is not None:
        try:
            acc.append(t_acc / len(trainloader))
        except:
            acc.append(0.0)
    if loss is not None:
        try:
            loss.append(t_loss / len(trainloader))
        except:
            loss.append(0.0)
    return t_acc / len(trainloader), t_loss / len(trainloader)


# def train_SAMmodel(trainloader,supervision_t, optimizer_stu, device, net_stu,  \
#                 acc=None,  learning_rate=None, iter_num=0):
#     net_stu.train()
#     t_loss, t_acc = 0, 0
#     max_iterations = 30000
#     dice = []
#
#     for i, batch in enumerate(trainloader):
#         images = batch['img']
#         true_masks = batch['mask']
#         SAM_output = batch['SAM_output']
#
#         images = images.to(device=device, dtype=torch.float32)
#
#         true_masks = true_masks.to(device=device)
#         SAM_output = SAM_output.to(device=device)
#
#         mask_pred = net_stu(images).squeeze(1)
#         masks_pred = sigmoid(mask_pred)
#         masks_pred = masks_pred.float()
#         temperature = 4
#         alpha = 0.7
#         if supervision_t == 'labeled':
#             loss_ce = F.cross_entropy(masks_pred, true_masks)
#             soft_teacher = F.softmax(SAM_output / temperature, dim=1)
#             soft_student = F.log_softmax(masks_pred / temperature, dim=1)
#             loss_kl = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)
#             total_loss = alpha * loss_kl + (1 - alpha) * loss_ce
#         else:
#             A = (masks_pred > 0.5).type(torch.int)  # 保证A是0/1
#             B = (SAM_output > 0.5).type(torch.int)
#             intersection = (A & B).float()
#             # true_masks = true_masks.to(device=device)
#             loss_ce = F.cross_entropy(masks_pred, intersection)
#             soft_teacher = F.softmax(SAM_output / temperature, dim=1)
#             soft_student = F.log_softmax(masks_pred / temperature, dim=1)
#             loss_kl = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)
#             total_loss = alpha * loss_kl + (1 - alpha) * loss_ce
#
#         dice.append(loss_kl)
#         lr_ = learning_rate * (1.0 - iter_num / max_iterations) ** 0.9
#
#         for param_group in optimizer_stu.param_groups:
#             param_group['lr'] = lr_
#
#         iter_num += 1
#
#         optimizer_stu.zero_grad()
#
#         total_loss.backward()
#         optimizer_stu.step()
#     return sum(dice)/len(dice)


def train_model_sam_lora(epoch, trainloader, optimizer_stu, device, net_stu,
                         acc=None, loss=None, learning_rate=1e-4, iter_num=0, max_iterations=30000):
    net_stu.train()

    CE_Loss = nn.BCELoss()
    t_loss, t_acc = 0.0, 0.0

    for i, batch in enumerate(trainloader):
        if torch.isnan(batch['mask']).any():
            print(f"Batch {i} contains NaN in mask, skipping...")
            continue  # 跳过本 batc

        images = batch['img'].to(device=device, dtype=torch.float32)
        true_masks = batch['mask'].to(device=device, dtype=torch.float32)

        # 调整 true_masks 维度，保证 [B, 1, H, W]
        if true_masks.dim() == 2:
            true_masks = true_masks.unsqueeze(0).unsqueeze(0)
        elif true_masks.dim() == 3:
            true_masks = true_masks.unsqueeze(1)

        B, _, H, W = true_masks.shape

        # 生成点提示
        point_coords_list, point_labels_list = [], []
        # 生成点提示
        batch_point_coords = []
        batch_point_labels = []

        for b in range(B):  # 遍历 batch 中的每张图
            mask_np = true_masks[b, 0].cpu().numpy().astype(np.uint8)

            try:
                # 1. 客户端计算每张图像的前景/背景聚类中心
                # print("----------------生成全局聚类点-------------")

                client_fg_centers, client_bg_centers = client_compute_and_upload_cluster_centers(mask_np)

                # 2. 聚合（这里只示范单客户端，实际联邦时应收集多客户端的 centers）
                all_clients_centers = [(client_fg_centers, client_bg_centers)]
                global_fg_centers, global_bg_centers = server_aggregate_cluster_centers(all_clients_centers)

                # 3. 用全局中心生成点标注
                coords, labels = generate_prompts_cluster_federated(
                    mask_np,
                    global_fg_centers,
                    global_bg_centers,
                    num_fg_points=30,
                    num_bg_points=30,
                    eps=5,
                    min_samples=10,
                )
            except ValueError:
                print("----------------生成全局聚类点失败，使用默认情况-------------")

                # 默认情况
                H, W = mask_np.shape
                coords = np.array([[H // 2, W // 2]])
                labels = np.array([1])

            batch_point_coords.append(coords)  # [N, 2]
            batch_point_labels.append(labels)  # [N]

        # 转 tensor
        point_coords_tensor = torch.tensor(batch_point_coords, dtype=torch.float32, device=device)  # [B, N, 2]
        point_labels_tensor = torch.tensor(batch_point_labels, dtype=torch.int64, device=device)  # [B, N]

        # forward
        images.requires_grad_(True)
        mask_pred = net_stu(images, point_coords=point_coords_tensor, point_labels=point_labels_tensor)

        # 保证 mask_pred 形状 [B, 1, H, W]
        if mask_pred.dim() == 3:
            mask_pred = mask_pred.unsqueeze(1)

        masks_pred = torch.sigmoid(mask_pred)

        # 计算损失
        loss_ce = CE_Loss(masks_pred, true_masks)
        loss_dice = 1 - dice_coeff(masks_pred.squeeze(1), true_masks.squeeze(1))
        loss_total = 1 * loss_ce + 0.75 * loss_dice

        # 调整学习率
        lr_ = learning_rate * (1.0 - iter_num / max_iterations) ** 0.9
        for param_group in optimizer_stu.param_groups:
            param_group['lr'] = lr_

        iter_num += 1

        optimizer_stu.zero_grad()

        try:
            loss_total.backward()
        except:
            print("Skip one batch due to CUDA assert.")
            continue
        optimizer_stu.step()

        t_loss += loss_total.item()

        with torch.no_grad():
            masks_pred_binary = (masks_pred > 0.5).float()
            t_acc += dice_coeff(masks_pred_binary.squeeze(1), true_masks.squeeze(1)).item()
    avg_acc = t_acc / len(trainloader) if len(trainloader) > 0 else 0
    avg_loss = t_loss / len(trainloader) if len(trainloader) > 0 else 0

    if acc is not None:
        acc.append(avg_acc)
    if loss is not None:
        loss.append(avg_loss)

    return avg_acc, avg_loss


######################################
def plot_graphs(num, CLIENTS, index, y_axis, title):
    idx_clr = 0
    plt.figure(num)
    for client in CLIENTS:
        plt.plot(index, y_axis[client], colors[idx_clr], label=client + title)
        idx_clr += 1
    plt.legend()
    plt.show()


########################################
def train_fedmix(trainloader, net_stu, optimizer_stu, \
                 device, acc=None, loss=None, supervision_type='labeled', \
                 FedMix_network=1):
    net_stu.train()
    t_loss, t_acc = 0, 0
    # labeled_len = len(trainloader)
    # labeled_iter = iter(trainloader)
    for i, batch in enumerate(trainloader):
        imgs, masks, y_pl = batch['img'], batch['mask'], batch['y_pl']
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer_stu.zero_grad()
        ###################################################
        l_ = 0
        ## get the prediction from the model of interest ##
        masks_stu = torch.sigmoid(net_stu(imgs))
        ### if supervision type is labeled, just train as normal with dice ###
        if supervision_type == 'labeled':
            l_stu = (1 - dice_coeff(masks_stu, masks.type(torch.float)))[0]
            l_ = l_stu
        else:
            if FedMix_network == 1:
                masks_teach = y_pl.to(device)
            else:
                masks_teach = masks.to(device)

            l_stu = (1 - dice_coeff(masks_stu, masks_teach.type(torch.float)))[0]
            l_ = l_stu
        #############################
        print("dice is :", l_.item())
        l_.backward()
        optimizer_stu.step()

        # for evaluation
        t_loss += l_.item()
        masks_stu = (masks_stu.detach() > 0.5).float()
        t_acc_network = dice_coeff(masks_stu, masks.type(torch.float)).item()
        t_acc += t_acc_network

    if acc is not None:
        try:
            acc.append(t_acc / len(trainloader))
        except:
            acc.append(0.0)
    if loss is not None:
        try:
            loss.append(t_loss / len(trainloader))
        except:
            loss.append(0.0)


########################################
#### save model
#### input: PTH <saving path>
####      : epoch <identifier>
####      : nets [collection to save]
####      : acc_train : list of clients
#########################################


def save_model_4(PTH, dice, epoch, nets):
    torch.save(nets['global'], os.path.join(PTH, 'FedLoRA_{}_net_{}.pth'.format(epoch, dice)))
    # torch.save(nets2, os.path.join(PTH , 'eam_ne_{}t.pth'.format(dice)))


def save_model_ll(PTH, epoch, nets, CLIENTS):
    for client in CLIENTS:
        p_global = PTH + 'llglobal/' + client
        os.makedirs(p_global, exist_ok=True)
        torch.save(nets[client], p_global + '/tvtmodel_' + str(epoch) + '.pth')


def save_model_centralize(PTH, epoch, nets):
    p_global = PTH + 'cenglobal2/'
    torch.save(nets, p_global + 'tvtmodel_' + str(epoch) + '.pth')


def sort_rows(matrix, num_rows):
    matrix_T = torch.transpose(matrix, 0, 1)
    sorted_T = torch.topk(matrix_T, num_rows)[0]
    return torch.transpose(sorted_T, 1, 0)

